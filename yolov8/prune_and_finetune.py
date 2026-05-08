"""
YOLOv8m 백본 구조적 프루닝 + COCO fine-tune 통합 파이프라인.

사용법:
    pip install ultralytics torch-pruning

  python prune_and_finetune.py --ratio 30 --epochs 20                                                                        
  python prune_and_finetune.py --ratio 50 --epochs 30                                                                        
  python prune_and_finetune.py --ratio 70 --epochs 30                                                                      
  python prune_and_finetune.py --ratio 90 --epochs 50   

흐름:
    1) yolov8m.pt 로드
    2) C2f → C2f_v2 변환 (DepGraph 호환)
    3) 백본 stage별 추천 ratio dict 구성 (--ratio 프리셋 30/50/70/90)
    4) DepGraph 기반 구조적 프루닝 (채널 실제 제거)
    5) 프루닝 직후 mAP 측정 (회복 전)
    6) COCO fine-tune (남은 파이프라인 그대로 이어서 학습)
    7) 최종 mAP 측정 + 비교 출력

QA importance 도입 시:
    아래 `IMPORTANCE_FN` 한 줄만 본인의 QA 클래스로 교체하면 됨.
"""

import argparse
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn

from ultralytics import YOLO, __version__
from ultralytics.nn.modules import Detect, C2f, Conv, Bottleneck
from ultralytics.engine.trainer import BaseTrainer
from ultralytics.utils import (LOGGER, RANK,
                                DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS)
from ultralytics.utils.checks import check_yaml


# ---------- ultralytics 버전 호환 패치 ----------

# attempt_load_one_weight: ckpt 파일 → (model, ckpt) 반환
try:
    from ultralytics.nn.tasks import attempt_load_one_weight
except ImportError:
    def attempt_load_one_weight(weight, device=None, inplace=True, fuse=False):
        """ultralytics 신버전 호환: 단일 .pt 로드 후 (model, ckpt) 반환."""
        ckpt = torch.load(weight, map_location='cpu', weights_only=False)
        # ema 가 있으면 우선, 없으면 model
        model = (ckpt.get('ema') or ckpt['model'])
        if device is not None:
            model = model.to(device)
        model = model.float().eval()
        # 자주 쓰이는 속성 보강
        if not hasattr(model, 'stride'):
            model.stride = torch.tensor([32.0])
        if 'train_args' in ckpt:
            try:
                model.args = ckpt['train_args']
            except Exception:
                pass
        return model, ckpt

# yaml_load: 옛 버전 → 새 YAML.load → PyYAML
try:
    from ultralytics.utils import yaml_load
except ImportError:
    try:
        from ultralytics.utils import YAML
        def yaml_load(file):
            return YAML.load(file)
    except ImportError:
        import yaml
        def yaml_load(file):
            with open(file, encoding='utf-8') as f:
                return yaml.safe_load(f)

# de_parallel: 단순 unwrap 이라 직접 정의 (버전 무관 안전)
try:
    from ultralytics.utils.torch_utils import de_parallel
except ImportError:
    def de_parallel(model):
        """DP / DDP wrapper 가 있으면 벗기고 raw 모델 반환."""
        return model.module if isinstance(
            model,
            (torch.nn.parallel.DataParallel,
             torch.nn.parallel.DistributedDataParallel),
        ) else model

import torch_pruning as tp


# ===========================================================
# 1. C2f → C2f_v2 변환 (DepGraph가 chunk()를 못 따라가는 문제 해결)
# ===========================================================
class C2f_v2(nn.Module):
    """C2f 동등 동작이지만 chunk 대신 cv0/cv1 두 conv로 명시적 분리."""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(
            Bottleneck(self.c, self.c, shortcut, g,
                       k=((3, 3), (3, 3)), e=1.0)
            for _ in range(n)
        )

    def forward(self, x):
        y = [self.cv0(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


def _infer_shortcut(bottleneck):
    c1 = bottleneck.cv1.conv.in_channels
    c2 = bottleneck.cv2.conv.out_channels
    return c1 == c2 and hasattr(bottleneck, 'add') and bottleneck.add


def _transfer_weights(c2f, c2f_v2):
    c2f_v2.cv2 = c2f.cv2
    c2f_v2.m = c2f.m
    sd, sd_v2 = c2f.state_dict(), c2f_v2.state_dict()

    old = sd['cv1.conv.weight']
    half = old.shape[0] // 2
    sd_v2['cv0.conv.weight'] = old[:half]
    sd_v2['cv1.conv.weight'] = old[half:]
    for k in ['weight', 'bias', 'running_mean', 'running_var']:
        old_bn = sd[f'cv1.bn.{k}']
        sd_v2[f'cv0.bn.{k}'] = old_bn[:half]
        sd_v2[f'cv1.bn.{k}'] = old_bn[half:]
    for k in sd:
        if not k.startswith('cv1.'):
            sd_v2[k] = sd[k]
    c2f_v2.load_state_dict(sd_v2)


def replace_c2f_with_c2f_v2(module):
    for name, child in module.named_children():
        if isinstance(child, C2f):
            new_block = C2f_v2(
                child.cv1.conv.in_channels,
                child.cv2.conv.out_channels,
                n=len(child.m),
                shortcut=_infer_shortcut(child.m[0]),
                g=child.m[0].cv2.conv.groups,
                e=child.c / child.cv2.conv.out_channels,
            )
            _transfer_weights(child, new_block)
            # ultralytics layer-meta 속성 복사 (f, i, type, np 등)
            # 이게 빠지면 model.predict() 시 'no attribute f' 에러
            for attr in ('f', 'i', 'type', 'np'):
                if hasattr(child, attr):
                    setattr(new_block, attr, getattr(child, attr))
            setattr(module, name, new_block)
        else:
            replace_c2f_with_c2f_v2(child)


# ===========================================================
# 2. 백본 stage별 prune ratio 프리셋 (추천 매트릭스)
#    stage 0 = stem (conservative), stage 6/7/8 = deep (aggressive)
# ===========================================================
RATIO_PRESETS = {
    30: {0: 0.00, 1: 0.15, 2: 0.25, 3: 0.25, 4: 0.30,
         5: 0.30, 6: 0.40, 7: 0.30, 8: 0.40, 9: 0.30},
    50: {0: 0.00, 1: 0.25, 2: 0.40, 3: 0.40, 4: 0.45,
         5: 0.45, 6: 0.55, 7: 0.50, 8: 0.55, 9: 0.45},
    70: {0: 0.10, 1: 0.40, 2: 0.55, 3: 0.55, 4: 0.65,
         5: 0.65, 6: 0.75, 7: 0.70, 8: 0.75, 9: 0.65},
    90: {0: 0.30, 1: 0.60, 2: 0.75, 3: 0.75, 4: 0.85,
         5: 0.85, 6: 0.92, 7: 0.90, 8: 0.92, 9: 0.85},
}


def build_ratio_dict(nn_model, preset_key: int):
    """stage idx → ratio 매핑을 layer 객체 → ratio 로 변환."""
    preset = RATIO_PRESETS[preset_key]
    layer_dict = {}
    for idx in range(min(10, len(nn_model.model))):  # backbone stages 0~9
        stage = nn_model.model[idx]
        ratio = preset.get(idx, 0.0)
        if ratio > 0:
            layer_dict[stage] = ratio
    return layer_dict


# ===========================================================
# 3. fine-tune monkey patch — 프루닝된 in-memory 모델을
#    ultralytics 기본 train()이 yaml에서 새로 만들지 않도록.
# ===========================================================
def save_model_v2(self: BaseTrainer):
    """Half-precision 저장 비활성화 (정확도 손실 방지)."""
    ckpt = {
        'epoch': self.epoch,
        'best_fitness': self.best_fitness,
        'model': deepcopy(de_parallel(self.model)),
        'ema': deepcopy(self.ema.ema),
        'updates': self.ema.updates,
        'optimizer': self.optimizer.state_dict(),
        'train_args': vars(self.args),
        'date': datetime.now().isoformat(),
        'version': __version__,
    }
    torch.save(ckpt, self.last)
    if self.best_fitness == self.fitness:
        torch.save(ckpt, self.best)
    if (self.epoch > 0) and (self.save_period > 0) and \
       (self.epoch % self.save_period == 0):
        torch.save(ckpt, self.wdir / f'epoch{self.epoch}.pt')
    del ckpt


def strip_optimizer_v2(f: Union[str, Path] = 'best.pt', s: str = '') -> None:
    x = torch.load(f, map_location='cpu', weights_only=False)
    args = {**DEFAULT_CFG_DICT, **x['train_args']}
    if x.get('ema'):
        x['model'] = x['ema']
    for k in 'optimizer', 'ema', 'updates':
        x[k] = None
    for p in x['model'].parameters():
        p.requires_grad = False
    x['train_args'] = {k: v for k, v in args.items() if k in DEFAULT_CFG_KEYS}
    torch.save(x, s or f)
    LOGGER.info(f"Optimizer stripped from {f}, "
                f"{f' saved as {s},' if s else ''} "
                f"{os.path.getsize(s or f) / 1e6:.1f}MB")


def final_eval_v2(self: BaseTrainer):
    for f in self.last, self.best:
        if f.exists():
            strip_optimizer_v2(f)
            if f is self.best:
                LOGGER.info(f'\nValidating {f}...')
                self.metrics = self.validator(model=f)
                self.metrics.pop('fitness', None)
                self.run_callbacks('on_fit_epoch_end')


def train_v2(self: YOLO, **kwargs):
    """Train but skip yaml-based model rebuild (preserves pruned structure)."""
    self._check_is_pytorch_model()
    overrides = self.overrides.copy()
    overrides.update(kwargs)
    if kwargs.get('cfg'):
        overrides = yaml_load(check_yaml(kwargs['cfg']))
    overrides['mode'] = 'train'
    if not overrides.get('data'):
        raise AttributeError("Dataset required, e.g. data='coco.yaml'")
    if overrides.get('resume'):
        overrides['resume'] = self.ckpt_path

    self.task = overrides.get('task') or self.task
    trainer_class = self._smart_load("trainer")
    self.trainer = trainer_class(overrides=overrides, _callbacks=self.callbacks)

    # ⭐ 핵심: yaml 에서 모델 새로 만들지 말고 in-memory pruned 모델 사용
    self.trainer.model = self.model
    self.trainer.save_model = save_model_v2.__get__(self.trainer)
    self.trainer.final_eval = final_eval_v2.__get__(self.trainer)

    self.trainer.hub_session = getattr(self, 'session', None)
    self.trainer.train()

    if RANK in (-1, 0):
        self.model, _ = attempt_load_one_weight(str(self.trainer.best))
        self.overrides = self.model.args
        self.metrics = getattr(self.trainer.validator, 'metrics', None)


# ===========================================================
# 4. 메인 파이프라인
# ===========================================================

def get_importance(method: str, prune_ratio: float):
    """--method 인자에 따라 importance 객체 반환."""
    if method == "l2":
        return tp.importance.GroupMagnitudeImportance()   # L2 norm baseline
    elif method == "qubo":
        try:
            from qa_importance import QAImportance
        except ImportError as e:
            raise ImportError(
                "QUBO 모드에는 qa_importance.py 와 QAImportance 클래스가 필요합니다. "
                "(pip install dwave-neal dimod 도 필요)"
            ) from e
        return QAImportance(prune_ratio=prune_ratio)
    else:
        raise ValueError(f"unknown method: {method}")


def _normalize_torch_device(d):
    """argparse 값('6', 6, '0,1', 'cpu') → torch.tensor.to() 가 받는 형식."""
    if isinstance(d, int):
        return f'cuda:{d}'
    s = str(d).strip()
    if s.lower() == 'cpu':
        return 'cpu'
    if s.isdigit():               # '6' → 'cuda:6'
        return f'cuda:{s}'
    if ',' in s:                  # '0,1' → 첫 GPU 만 (tensor 생성용)
        return f'cuda:{s.split(",")[0].strip()}'
    return s                      # 이미 'cuda:6' 같은 형식


def main(args):
    print("=" * 60)
    print(f"  YOLOv8m PRUNE + FINETUNE")
    print(f"  method = {args.method}    ratio preset = {args.ratio}%")
    print("=" * 60)

    device = _normalize_torch_device(args.device)

    # ----- [1/6] Baseline mAP 측정 (별도 인스턴스, 끝나면 버림) -----
    # ultralytics model.val() 이 모델을 fuse + inference-mode 로 바꿔서
    # 이후 requires_grad=True 설정이 막힘 → val 전용 인스턴스 분리.
    print("\n[1/6] Baseline validation (fresh YOLOv8m, throwaway)")
    val_only = YOLO(args.model)
    base_metrics = val_only.val(data=args.data, imgsz=args.imgsz,
                                 batch=args.batch, device=args.device,
                                 verbose=False)
    base_map = float(base_metrics.box.map)
    print(f"  Baseline mAP = {base_map:.4f}")
    del val_only
    torch.cuda.empty_cache()

    # ----- [2/6] 프루닝용 모델 fresh 로드 + C2f 교체 -----
    print("\n[2/6] Load model for pruning + replace C2f → C2f_v2")
    model = YOLO(args.model)
    model.train_v2 = train_v2.__get__(model)

    nn_model = model.model
    nn_model.train()
    for p in nn_model.parameters():
        p.requires_grad = True

    replace_c2f_with_c2f_v2(nn_model)
    nn_model = nn_model.to(device)
    model.model = nn_model

    example_inputs = torch.randn(1, 3, args.imgsz, args.imgsz).to(device)
    base_macs, base_params = tp.utils.count_ops_and_params(
        nn_model, example_inputs)
    print(f"  Baseline: {base_params/1e6:.2f} M params, "
          f"{base_macs/1e9:.2f} GMACs, mAP={base_map:.4f}")

    # ----- [3/6] Pruner 구성 -----
    print(f"\n[3/6] Build pruner")
    ignored = [m for m in nn_model.modules() if isinstance(m, Detect)]
    ratio_dict = build_ratio_dict(nn_model, args.ratio)
    print("  Layer-wise ratios (backbone):")
    for idx, r in RATIO_PRESETS[args.ratio].items():
        print(f"    stage {idx}: {r*100:>5.1f}%")

    pruner = tp.pruner.GroupNormPruner(
        nn_model, example_inputs,
        importance=get_importance(args.method, args.ratio / 100.0),
        pruning_ratio=args.ratio / 100.0,
        pruning_ratio_dict=ratio_dict,
        ignored_layers=ignored,
    )

    # ----- [4/6] 실제 채널 제거 -----
    print(f"\n[4/6] Apply structural pruning (channels removed)")
    pruner.step()

    after_macs, after_params = tp.utils.count_ops_and_params(
        nn_model, example_inputs)
    print(f"  Pruned:   {after_params/1e6:.2f} M params "
          f"({(1-after_params/base_params)*100:.1f}% ↓), "
          f"{after_macs/1e9:.2f} GMACs "
          f"({(1-after_macs/base_macs)*100:.1f}% ↓)")

    # 잘려나간 직후 모델 한 번 저장 (검증/이후 분석용)
    pruned_path = f"yolov8m_pruned_{args.method}_r{args.ratio}.pt"
    torch.save({'model': deepcopy(nn_model)}, pruned_path)
    print(f"  Saved pruned structure: {pruned_path}")

    # ----- [5/6] Pre-finetune mAP (회복 전) -----
    # ⚠️ 원본 model 에 직접 .val() 하면 fuse + inference-mode 로 오염되어
    # 이후 fine-tune 시 'requires_grad on inference tensor' 에러 발생.
    # → deepcopy 사본에 val 해서 원본 model 은 보존.
    print(f"\n[5/6] mAP right after pruning (before fine-tune)")
    val_copy = deepcopy(model)
    pruned_metrics = val_copy.val(data=args.data, imgsz=args.imgsz,
                                    batch=args.batch, device=args.device,
                                    verbose=False)
    pruned_map = float(pruned_metrics.box.map)
    del val_copy
    torch.cuda.empty_cache()
    print(f"  mAP (no fine-tune yet): {pruned_map:.4f} "
          f"  (drop: {(base_map - pruned_map)*100:+.2f} pts)")

    # ----- [6/6] Fine-tune -----
    if args.skip_finetune:
        print("\n[6/6] Skip fine-tune (--skip-finetune)")
        print("\nDone.")
        return

    print(f"\n[6/6] Fine-tune on {args.data} for {args.epochs} epochs")
    model.train_v2(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        name=f'pruned_{args.method}_r{args.ratio}_ft',
    )

    final_metrics = model.val(data=args.data, imgsz=args.imgsz,
                               batch=args.batch, device=args.device,
                               verbose=False)
    final_map = float(final_metrics.box.map)

    # ----- 최종 요약 -----
    print("\n" + "=" * 60)
    print("  FINAL RESULT")
    print("=" * 60)
    print(f"  Method             : {args.method}")
    print(f"  Ratio preset       : {args.ratio}%")
    print(f"  Params             : {base_params/1e6:.2f} M  →  {after_params/1e6:.2f} M  "
          f"({(1-after_params/base_params)*100:.1f}% ↓)")
    print(f"  GMACs              : {base_macs/1e9:.2f}  →  {after_macs/1e9:.2f}  "
          f"({(1-after_macs/base_macs)*100:.1f}% ↓)")
    print(f"  mAP (baseline)     : {base_map:.4f}")
    print(f"  mAP (pruned only)  : {pruned_map:.4f}")
    print(f"  mAP (fine-tuned)   : {final_map:.4f}")
    print(f"  mAP drop vs base   : {(base_map - final_map)*100:+.2f} pts")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ratio",  type=int, required=True,
                        choices=[30, 50, 70, 90],
                        help="prune ratio preset (백본 stage별 layer-wise 적용)")
    parser.add_argument("--method", type=str, default="l2",
                        choices=["l2", "qubo"],
                        help="채널 importance: l2 = L2 norm baseline, "
                             "qubo = QA-pruning (qa_importance.py 필요)")
    parser.add_argument("--model",  type=str, default="yolov8m.pt")
    parser.add_argument("--data",   type=str, default="coco.yaml")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--device", default=0,
                        help="0=GPU0, 'cpu', 또는 GPU id")
    parser.add_argument("--skip-finetune", action="store_true",
                        help="프루닝만 하고 학습 스킵 (구조 검증용)")
    args = parser.parse_args()
    main(args)
