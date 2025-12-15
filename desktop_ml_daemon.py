# src/desktop_ml_daemon.py
import time
import tempfile
import subprocess
from pathlib import Path
from typing import List, Dict, Any

import requests
import torch
from torchvision import transforms
from PIL import Image

from model_def import MultiLabelTagModel

# =======================
# 설정 부분
# =======================

# 🔥 백엔드 주소 (지금 쓰는 10.10.10.2:11002)
BACKEND_BASE_URL = "http://10.10.10.2:11002"

PENDING_API_URL = f"{BACKEND_BASE_URL}/api/videos/features/pending-desktop"
AUTO_TAG_API_URL = f"{BACKEND_BASE_URL}/api/videos/features/auto-tags"
STREAM_URL_TEMPLATE = f"{BACKEND_BASE_URL}/api/videos/{{video_no}}/stream"

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "multilabel_tags_v1.pt"

IMAGE_SIZE = 224
BATCH_SIZE = 64
PRESENT_THRESHOLD = 0.4

# ffmpeg가 PATH에 등록되어 있으면 그냥 "ffmpeg"
# 아니면 예: r"C:\ffmpeg\bin\ffmpeg.exe" 로 바꿔줘
FFMPEG_PATH = "ffmpeg"

# 주기 (초) - 할 일 없을 때 대기 시간
POLL_INTERVAL_SEC = 30


# =======================
# 기본 전처리 & 모델 로드
# =======================

def get_transform():
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_model(model_path: str, device: torch.device):
    ckpt = torch.load(model_path, map_location=device)

    label_names: List[str] = ckpt["label_names"]
    num_labels = len(label_names)

    model = MultiLabelTagModel(num_labels=num_labels)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    return model, label_names


def predict_for_frames_dir(
    frames_dir: Path,
    model: MultiLabelTagModel,
    label_names: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    """
    frames_dir 아래의 모든 JPG/PNG 프레임을 읽어서
    - 모델에 배치로 넣고
    - 프레임별 확률 평균 → 영상 단위 확률
    - top3 / present_tags / all_scores 반환
    """
    tf = get_transform()

    img_paths = sorted(
        [p for p in frames_dir.iterdir()
         if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    )

    if not img_paths:
        raise ValueError(f"프레임이 없습니다: {frames_dir}")

    all_probs = []

    for start in range(0, len(img_paths), BATCH_SIZE):
        batch_paths = img_paths[start:start + BATCH_SIZE]

        batch_imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            x = tf(img)
            batch_imgs.append(x)

        x = torch.stack(batch_imgs, dim=0).to(device)

        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits)

        all_probs.append(probs.cpu())

    probs_all = torch.cat(all_probs, dim=0)   # (F, L)
    probs_video = probs_all.mean(dim=0).numpy()  # (L,)

    sorted_idx = probs_video.argsort()[::-1]

    top3 = []
    for i in range(3):
        idx = sorted_idx[i]
        name = label_names[idx]
        score = float(probs_video[idx])
        top3.append({"name": name, "score": score})

    present_tags = []
    for i, name in enumerate(label_names):
        score = float(probs_video[i])
        if score >= PRESENT_THRESHOLD:
            present_tags.append({"name": name, "score": score})

    all_scores = {
        label_names[i]: float(probs_video[i])
        for i in range(len(label_names))
    }

    return {
        "top3": top3,
        "present_tags": present_tags,
        "all_scores": all_scores,
        "frame_count": int(probs_all.shape[0]),
    }


# =======================
# 백엔드와 통신 관련 함수
# =======================

def fetch_pending(limit: int = 3) -> List[Dict[str, Any]]:
    """
    아직 DESKTOP_ML 태그가 없는 승인된 영상 목록 가져오기
    """
    resp = requests.get(PENDING_API_URL, params={"limit": limit}, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        return []
    return data


def download_video(video_no: int, download_dir: Path) -> Path:
    """
    백엔드 스트림 엔드포인트에서 영상 다운로드
    """
    url = STREAM_URL_TEMPLATE.format(video_no=video_no)
    print(f"[WORKER] 영상 다운로드: {url}")

    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    # 확장자는 ffmpeg 입장에선 크게 중요하지 않아서 .mp4 로 통일
    video_path = download_dir / f"{video_no}.mp4"
    with open(video_path, "wb") as f:
        for chunk in resp.iter_content(8192):
            if chunk:
                f.write(chunk)

    return video_path


def extract_frames_with_ffmpeg(video_path: Path, frames_dir: Path) -> None:
    """
    ffmpeg 를 이용해 1초당 1프레임 캡처
    """
    frames_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(frames_dir / "frame-%03d.jpg")

    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        "fps=1",
        out_pattern,
    ]

    print(f"[WORKER] ffmpeg 실행: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if proc.returncode != 0:
        err_msg = proc.stderr.decode(errors="ignore")
        raise RuntimeError(f"ffmpeg 실패 (code={proc.returncode}): {err_msg}")


def build_payload(video_no: int, result: Dict[str, Any]) -> Dict[str, Any]:
    """
    VideoAutoTagRequest 형식에 맞는 payload 생성
    """
    top3 = result.get("top3", [])
    payload = {
        "videoNo": video_no,
        "mainTag": top3[0] if top3 else None,
        "subTags": top3[1:] if len(top3) > 1 else [],
        "presentTags": result.get("present_tags", []),
        "allScores": result.get("all_scores", {}),
        "frameCount": result.get("frame_count", 0),
    }
    return payload


def post_auto_tags(payload: Dict[str, Any]) -> None:
    """
    백엔드 /api/videos/features/auto-tags 로 태그 전송
    """
    resp = requests.post(AUTO_TAG_API_URL, json=payload, timeout=30)
    print(f"[WORKER] 태그 전송 응답 코드: {resp.status_code}")
    resp.raise_for_status()


# =======================
# 개별 영상 처리
# =======================

def process_one_video(
    video_no: int,
    model: MultiLabelTagModel,
    label_names: List[str],
    device: torch.device,
):
    print(f"[WORKER] ===== 영상 {video_no} 처리 시작 =====")

    # 임시 디렉터리 하나 만들어서 그 안에 영상 + 프레임 저장
    with tempfile.TemporaryDirectory(prefix=f"video_{video_no}_") as tmpdir:
        tmp_dir = Path(tmpdir)

        # 1) 영상 다운로드
        video_path = download_video(video_no, tmp_dir)

        # 2) ffmpeg 로 프레임 추출
        frames_dir = tmp_dir / "frames"
        extract_frames_with_ffmpeg(video_path, frames_dir)

        # 3) 로컬 모델로 멀티라벨 태깅
        result = predict_for_frames_dir(
            frames_dir=frames_dir,
            model=model,
            label_names=label_names,
            device=device,
        )

        # 4) payload 만들고 백엔드에 전송
        payload = build_payload(video_no, result)
        print("[WORKER] 태그 payload:", payload)

        post_auto_tags(payload)

    print(f"[WORKER] ===== 영상 {video_no} 처리 완료 =====")


# =======================
# 메인 루프
# =======================

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("사용 장치:", device)

    model, label_names = load_model(str(MODEL_PATH), device)

    print("[WORKER] 데스크탑 ML 데몬 시작")
    print("[WORKER] 백엔드:", BACKEND_BASE_URL)
    print("[WORKER] 모델 경로:", MODEL_PATH)

    while True:
        try:
            pending_list = fetch_pending(limit=3)
        except Exception as e:
            print("[WORKER] pending 목록 조회 실패:", e)
            time.sleep(POLL_INTERVAL_SEC)
            continue

        if not pending_list:
            # 처리할 영상이 없으면 잠깐 쉼
            print(f"[WORKER] 처리할 영상 없음. {POLL_INTERVAL_SEC}초 대기...")
            time.sleep(POLL_INTERVAL_SEC)
            continue

        print(f"[WORKER] 처리할 영상 목록: {[p['videoNo'] for p in pending_list]}")

        for item in pending_list:
            video_no = int(item.get("videoNo"))
            try:
                process_one_video(video_no, model, label_names, device)
            except Exception as e:
                print(f"[WORKER] 영상 {video_no} 처리 중 오류:", e)

        # 한 번 다 돌리고 잠깐 쉰 뒤 다음 루프로
        time.sleep(3)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WORKER] 종료 요청 감지. 프로그램을 종료합니다.")
