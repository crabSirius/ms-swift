"""在 damage_tree_0910 与 damage_tree_0901 两个数据集上，
对比基线 (base) 与 LoRA 版本的性能，并把结果汇总到 README.md。

实现方式：
    - 推理阶段：通过覆写 sys.argv 调用 ``run_pred.main`` ；
    - 评测阶段：通过覆写 sys.argv 调用 ``eval_pr.main`` ；
      同时直接复用 ``eval_pr.load_jsonl`` / ``eval_pr.compute_pr`` 拿到结构化指标用于汇总 README。

用法示例:
    python deepinnet-tools/damage_tree/compare_baseline_vs_lora.py \
        --adapter /root/liantaoding_dev/codes/ms-swift/output/v1/checkpoint-250

可以用 ``--only`` 选择只跑某一组实验，也可以用 ``--skip_infer``
在结果已存在时只重新做评测并更新 README 。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import eval_pr  # noqa: E402
import run_pred  # noqa: E402


DEFAULT_ADAPTER = '/root/liantaoding_dev/codes/ms-swift/output/v1/checkpoint-250'
DEFAULT_MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
DEFAULT_SUBSET = 'tree_damage_per_label_with_negatives'
DEFAULT_DATA_ROOT = '/root/liantaoding_datas/images/damage_tree'
DEFAULT_OUTPUT_ROOT = os.path.join(
    '/root/liantaoding_dev/codes/ms-swift', 'deepinnet-tools/damage_tree/eval_runs')


@dataclass
class ExpConfig:
    name: str                     # e.g. "0910_base"
    dataset_tag: str              # "0910" / "0901"
    variant: str                  # "base" / "lora"
    local_path: str
    subset: str
    output_dir: str
    adapters: List[str] = field(default_factory=list)


def build_exp_configs(args) -> List[ExpConfig]:
    configs: List[ExpConfig] = []
    for dataset_tag in args.datasets:
        local_path = os.path.join(
            args.data_root, f'damage_tree_{dataset_tag}', 'ms_swift')
        for variant in args.variants:
            adapters = [args.adapter] if variant == 'lora' else []
            name = f'{dataset_tag}_{variant}'
            output_dir = os.path.join(args.output_root, name)
            configs.append(
                ExpConfig(
                    name=name,
                    dataset_tag=dataset_tag,
                    variant=variant,
                    local_path=local_path,
                    subset=args.subset,
                    output_dir=output_dir,
                    adapters=adapters,
                ))
    if args.only:
        wanted = set(args.only)
        configs = [c for c in configs if c.name in wanted]
    return configs


def _result_jsonl(cfg: ExpConfig) -> str:
    return os.path.join(cfg.output_dir, f'{cfg.subset}_result.jsonl')


def _gt_jsonl(cfg: ExpConfig) -> str:
    return os.path.join(
        cfg.output_dir,
        '_prepared_dataset',
        f'infer_{cfg.subset}',
        f'{cfg.subset}.jsonl')


def _pred_out(cfg: ExpConfig) -> str:
    return os.path.join(cfg.output_dir, 'pred.jsonl')


def run_inference(cfg: ExpConfig, args) -> None:
    os.makedirs(cfg.output_dir, exist_ok=True)
    argv = [
        'run_pred.py',
        '--model', args.model,
        '--local_path', cfg.local_path,
        '--subset', cfg.subset,
        '--output_dir', cfg.output_dir,
        '--pred_out', _pred_out(cfg),
        '--infer_backend', args.infer_backend,
    ]
    if cfg.adapters:
        argv.append('--adapters')
        argv.extend(cfg.adapters)
    if args.merge_lora and cfg.adapters:
        argv.extend(['--merge_lora', 'True'])
    if args.eval_limit is not None:
        argv.extend(['--eval_limit', str(args.eval_limit)])

    saved_argv = sys.argv
    try:
        sys.argv = argv
        run_pred.main()
    finally:
        sys.argv = saved_argv


def run_evaluation(cfg: ExpConfig, args) -> dict:
    gt = _gt_jsonl(cfg)
    pred = _result_jsonl(cfg)
    if not os.path.exists(gt) or not os.path.exists(pred):
        raise FileNotFoundError(
            f'[{cfg.name}] 评测所需文件缺失: gt={gt}  pred={pred} '
            '请先执行推理 (不要加 --skip_infer)。')

    draw_dir = os.path.join(cfg.output_dir, 'draw') if args.draw else ''
    argv = [
        'eval_pr.py',
        '--gt_jsonl', gt,
        '--pred_jsonl', pred,
        '--iou_threshold', str(args.iou_threshold),
    ]
    if args.ignore_label:
        argv.append('--ignore_label')
    if draw_dir:
        argv.extend(['--draw_dir', draw_dir])
    else:
        argv.extend(['--draw_dir', ''])

    saved_argv = sys.argv
    try:
        sys.argv = argv
        eval_pr.main()
    finally:
        sys.argv = saved_argv

    gt_records = eval_pr.load_jsonl(gt)
    pred_records = eval_pr.load_jsonl(pred)
    metrics = eval_pr.compute_pr(
        gt_records=gt_records,
        pred_records=pred_records,
        iou_thr=args.iou_threshold,
        use_label=not args.ignore_label,
        max_samples=None,
        draw_dir=None)
    return metrics


def _fmt(v: float) -> str:
    return f'{v:.4f}' if isinstance(v, float) else str(v)


def render_readme(all_results: List[dict], args, readme_path: str) -> None:
    by_dataset: dict = {}
    for r in all_results:
        by_dataset.setdefault(r['dataset_tag'], {})[r['variant']] = r

    lines: List[str] = []
    lines.append('# damage_tree 基线 vs LoRA 性能对比\n')
    lines.append(f'- 基础模型: `{args.model}`')
    lines.append(f'- LoRA adapter: `{args.adapter}`')
    lines.append(f'- 子集: `{args.subset}`')
    lines.append(f'- IoU 阈值: `{args.iou_threshold}`')
    lines.append(f'- 匹配是否使用 label: `{not args.ignore_label}`')
    lines.append(f'- 生成时间: `{time.strftime("%Y-%m-%d %H:%M:%S")}`\n')

    lines.append('## 主要指标\n')
    lines.append('| 数据集 | 版本 | Samples | TP | FP | FN | Precision | Recall | F1 |')
    lines.append('|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for tag in args.datasets:
        for variant in args.variants:
            r = by_dataset.get(tag, {}).get(variant)
            if r is None:
                continue
            m = r['metrics']
            lines.append(
                f'| damage_tree_{tag} | {variant} | {m["samples"]} | {m["tp"]} | '
                f'{m["fp"]} | {m["fn"]} | {_fmt(m["precision"])} | '
                f'{_fmt(m["recall"])} | {_fmt(m["f1"])} |')

    lines.append('\n## LoRA 相对基线提升\n')
    lines.append('| 数据集 | ΔPrecision | ΔRecall | ΔF1 |')
    lines.append('|---|---:|---:|---:|')
    for tag in args.datasets:
        base = by_dataset.get(tag, {}).get('base')
        lora = by_dataset.get(tag, {}).get('lora')
        if not base or not lora:
            continue
        mb = base['metrics']
        ml = lora['metrics']
        lines.append(
            f'| damage_tree_{tag} | {ml["precision"] - mb["precision"]:+.4f} | '
            f'{ml["recall"] - mb["recall"]:+.4f} | '
            f'{ml["f1"] - mb["f1"]:+.4f} |')

    lines.append('\n## 输出目录\n')
    for r in all_results:
        lines.append(f'- **{r["name"]}** -> `{r["output_dir"]}`')
    lines.append('')

    lines.append('## 原始指标 (JSON)\n')
    lines.append('```json')
    lines.append(json.dumps(
        {r['name']: r['metrics'] for r in all_results},
        ensure_ascii=False, indent=2))
    lines.append('```')

    os.makedirs(os.path.dirname(readme_path), exist_ok=True)
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[README] 已写入: {readme_path}')


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--adapter', default=DEFAULT_ADAPTER,
                   help='LoRA adapter checkpoint 路径')
    p.add_argument('--data_root', default=DEFAULT_DATA_ROOT)
    p.add_argument('--subset', default=DEFAULT_SUBSET)
    p.add_argument('--datasets', nargs='+', default=['0910', '0901'],
                   help='要对比的数据集标签（对应 damage_tree_<tag>）')
    p.add_argument('--variants', nargs='+', default=['base', 'lora'],
                   choices=['base', 'lora'])
    p.add_argument('--output_root', default=DEFAULT_OUTPUT_ROOT,
                   help='四个实验的输出根目录')
    p.add_argument('--readme', default=os.path.join(
        HERE, 'RESULT.md'), help='生成的汇总 README 路径')
    p.add_argument('--infer_backend', default='transformers',
                   choices=['vllm', 'transformers', 'sglang', 'lmdeploy'])
    p.add_argument('--eval_limit', type=int, default=None)
    p.add_argument('--iou_threshold', type=float, default=0.3)
    p.add_argument('--ignore_label', action='store_true')
    p.add_argument('--merge_lora', action='store_true',
                   help='LoRA 推理前是否先 merge 权重')
    p.add_argument('--draw', action='store_true',
                   help='评测时输出 GT/Pred 可视化图')
    p.add_argument('--skip_infer', action='store_true',
                   help='跳过推理，只重新评测并更新 README')
    p.add_argument('--only', nargs='*', default=None,
                   help='只跑指定实验名，例如 0910_base 0901_lora')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    configs = build_exp_configs(args)
    if not configs:
        raise SystemExit('没有匹配的实验配置，请检查 --only/--datasets/--variants')

    print('[PLAN] 将执行以下实验:')
    for c in configs:
        tag = 'LoRA' if c.adapters else 'BASE'
        print(f'  - {c.name:<16}  [{tag}]  data={c.local_path}  out={c.output_dir}')
    print()

    all_results = []
    for cfg in configs:
        print(f'\n==================== {cfg.name} ====================')
        if not args.skip_infer:
            run_inference(cfg, args)
        else:
            print(f'[SKIP_INFER] 复用 {_result_jsonl(cfg)}')
        metrics = run_evaluation(cfg, args)
        print(f'[METRICS] {cfg.name}: {json.dumps(metrics, ensure_ascii=False)}')
        all_results.append({
            'name': cfg.name,
            'dataset_tag': cfg.dataset_tag,
            'variant': cfg.variant,
            'output_dir': cfg.output_dir,
            'metrics': metrics,
        })

    render_readme(all_results, args, args.readme)


if __name__ == '__main__':
    main()
