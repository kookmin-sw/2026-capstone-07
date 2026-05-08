"""
웹캠 실시간 사람 디텍션 데모.

사용법:
    python webcam_demo.py                             # yolov8m.pt + 웹캠 0번 + person만
    python webcam_demo.py --model runs/.../best.pt    # 본인 프루닝 모델
    python webcam_demo.py --source 1                  # 두 번째 웹캠
    python webcam_demo.py --source video.mp4          # 동영상 파일
    python webcam_demo.py --classes -1                # 80개 클래스 전부
    python webcam_demo.py --conf 0.5                  # confidence 임계값

조작:
    q 또는 ESC → 종료
    s         → 현재 프레임 스크린샷 저장 (snapshot_NNN.jpg)
"""

import argparse
import time
import cv2
import torch
from ultralytics import YOLO


def main(args):
    # 1. 모델 로드
    print(f"[load] {args.model}")
    model = YOLO(args.model)

    # 2. 웹캠 / 비디오 열기
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"카메라/비디오를 열 수 없어요: {args.source}")

    # 해상도 지정 (가능한 경우)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    print(f"[cam] {int(cap.get(3))}x{int(cap.get(4))} @ "
          f"{cap.get(cv2.CAP_PROP_FPS):.0f} FPS")

    # 3. 클래스 필터: -1 이면 전체, 아니면 person(0)만 (기본)
    classes = None if args.classes == -1 else [args.classes]
    label   = "all" if classes is None else f"class={classes}"

    # 4. 표시할 모델 정보 (좌상단 오버레이)
    n_layers, n_params, _, n_flops = model.info(detailed=False, imgsz=args.imgsz)
    info_str = (f"{args.model} | "
                f"{n_params/1e6:.1f}M params | "
                f"{n_flops:.1f} GFLOPs | "
                f"{label}")

    # 5. FPS 이동평균
    ema_fps, alpha = 0.0, 0.9
    snapshot_id = 0

    print("[run] q 또는 ESC = 종료, s = 스크린샷")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # 추론
        t0 = time.time()
        results = model.predict(
            source=frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            classes=classes,
            device=args.device,
            verbose=False,
        )
        dt = time.time() - t0
        cur_fps = 1.0 / max(dt, 1e-6)
        ema_fps = cur_fps if ema_fps == 0 else alpha*ema_fps + (1-alpha)*cur_fps

        # 박스 그리기 (ultralytics가 자동으로 그려줌)
        annotated = results[0].plot()

        # 좌상단 오버레이: 모델 정보 + FPS
        h, w = annotated.shape[:2]
        cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 0), -1)
        cv2.putText(annotated, info_str, (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(annotated, f"FPS: {ema_fps:5.1f}  ({dt*1000:.1f} ms/frame)",
                    (10, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)

        # 화면 출력
        cv2.imshow("YOLOv8 Webcam Demo", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):     # ESC, q
            break
        if key == ord("s"):
            fname = f"snapshot_{snapshot_id:03d}.jpg"
            cv2.imwrite(fname, annotated)
            print(f"[save] {fname}")
            snapshot_id += 1

    cap.release()
    cv2.destroyAllWindows()
    print(f"[done] 평균 FPS: {ema_fps:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default="yolov8m.pt")
    parser.add_argument("--source",  default="0",
                        help="0/1/... = 웹캠 인덱스, 또는 video.mp4 경로")
    parser.add_argument("--imgsz",   type=int,   default=640)
    parser.add_argument("--conf",    type=float, default=0.4,
                        help="confidence 임계값 (낮으면 박스 많이, 높으면 엄격)")
    parser.add_argument("--iou",     type=float, default=0.5)
    parser.add_argument("--classes", type=int,   default=-1,
                        help="-1=전체 80개 클래스 (기본), 0=person만, 2=car 등")
    parser.add_argument("--width",   type=int,   default=1280)
    parser.add_argument("--height",  type=int,   default=720)
    parser.add_argument("--device",  default=0,
                        help="0=GPU0, 'cpu'=CPU")
    args = parser.parse_args()
    main(args)
