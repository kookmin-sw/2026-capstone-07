"""
YOLOv8 + COCO 베이스라인 평가 스크립트
출력: image size / mAP@0.5:0.95 / Params / GFLOPs / FPS  ← 5개 핵심 지표

사용법:
    python baseline_eval.py
    python baseline_eval.py --model yolov8s.pt
    python baseline_eval.py --model runs/detect/train/weights/best.pt
"""

import argparse
import time
import torch
from ultralytics import YOLO


def measure_fps(model, imgsz, device, n_warmup=20, n_iter=200, batch=1):
    """PyTorch FP32 기준 GPU 추론 FPS 측정 (TensorRT 아님)"""
    nn_model = model.model.to(device).eval()
    dummy = torch.randn(batch, 3, imgsz, imgsz, device=device)

    with torch.no_grad():
        for _ in range(n_warmup):
            _ = nn_model(dummy)

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = nn_model(dummy)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    ms_per_img = elapsed / n_iter * 1000 / batch
    fps = batch * n_iter / elapsed
    return ms_per_img, fps


def main(args):
    # 환경
    print(f"[env] torch={torch.__version__} cuda={torch.cuda.is_available()}")
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"

    # 1. 모델 로드
    model = YOLO(args.model)

    # 2. Params / FLOPs (model.info 가 4-tuple 반환: layers, params, grads, flops)
    n_layers, n_params, _, n_flops = model.info(detailed=False, imgsz=args.imgsz)

    # 3. COCO val2017 평가
    metrics = model.val(
        data="coco.yaml",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        save_json=True,
        plots=True,
        verbose=False,
    )
    map_5095 = float(metrics.box.map)

    # 4. FPS 측정 (PyTorch FP32, batch=1)
    fps_device = args.device if torch.cuda.is_available() else "cpu"
    ms, fps = measure_fps(model, imgsz=args.imgsz, device=fps_device)

    # 5. 핵심 5개 지표 출력
    print("\n" + "=" * 60)
    print(f" Result: {args.model}   (device: {device_name})")
    print("=" * 60)
    print(f"  Image size      : {args.imgsz}")
    print(f"  mAP@0.5:0.95    : {map_5095:.4f}   ({map_5095*100:.2f})")
    print(f"  Parameters      : {n_params/1e6:.2f} M")
    print(f"  GFLOPs          : {n_flops:.2f}")
    print(f"  FPS (PyTorch)   : {fps:.1f}   ({ms:.2f} ms/img)")
    print("=" * 60)
    print(f"  결과 폴더       : {metrics.save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str,   default="yolov8m.pt")
    parser.add_argument("--imgsz",  type=int,   default=640)
    parser.add_argument("--batch",  type=int,   default=16)
    parser.add_argument("--device", default=0,
                        help="0 = GPU 0번, 'cpu' = CPU")
    args = parser.parse_args()
    main(args)
