使用damage_tree_0901 微调训练

# damage_tree 基线 vs LoRA 性能对比

- 基础模型: `Qwen/Qwen3-VL-8B-Instruct`
- LoRA adapter: `/root/liantaoding_dev/codes/ms-swift/output/v1/checkpoint-250`
- 子集: `tree_damage_per_label_with_negatives`
- IoU 阈值: `0.3`
- 匹配是否使用 label: `True`
- 生成时间: `2026-04-22 16:08:13`

## 主要指标


| 数据集              | 版本   | Samples | TP  | FP  | FN  | Precision | Recall | F1     |
| ---------------- | ---- | ------- | --- | --- | --- | --------- | ------ | ------ |
| damage_tree_0910（测试） | base | 52      | 38  | 46  | 33  | 0.4524    | 0.5352 | 0.4903 |
| damage_tree_0910（测试） | lora | 52      | 37  | 15  | 34  | 0.7115    | 0.5211 | 0.6016 |
| damage_tree_0901（训练） | base | 262     | 55  | 224 | 51  | 0.1971    | 0.5189 | 0.2857 |
| damage_tree_0901（训练） | lora | 262     | 90  | 16  | 16  | 0.8491    | 0.8491 | 0.8491 |


## LoRA 相对基线提升


| 数据集              | ΔPrecision | ΔRecall | ΔF1     |
| ---------------- | ---------- | ------- | ------- |
| damage_tree_0910（测试）| +0.2592    | -0.0141 | +0.1113 |
| damage_tree_0901（训练）| +0.6519    | +0.3302 | +0.5633 |


## 输出目录

- **0910_base** -> `/root/liantaoding_dev/codes/ms-swift/deepinnet-tools/damage_tree/eval_runs/0910_base`
- **0910_lora** -> `/root/liantaoding_dev/codes/ms-swift/deepinnet-tools/damage_tree/eval_runs/0910_lora`
- **0901_base** -> `/root/liantaoding_dev/codes/ms-swift/deepinnet-tools/damage_tree/eval_runs/0901_base`
- **0901_lora** -> `/root/liantaoding_dev/codes/ms-swift/deepinnet-tools/damage_tree/eval_runs/0901_lora`

## 原始指标 (JSON)

```json
{
  "0910_base": {
    "iou_threshold": 0.3,
    "use_label": true,
    "samples": 52,
    "tp": 38,
    "fp": 46,
    "fn": 33,
    "precision": 0.4523809523809524,
    "recall": 0.5352112676056338,
    "f1": 0.4903225806451613
  },
  "0910_lora": {
    "iou_threshold": 0.3,
    "use_label": true,
    "samples": 52,
    "tp": 37,
    "fp": 15,
    "fn": 34,
    "precision": 0.7115384615384616,
    "recall": 0.5211267605633803,
    "f1": 0.6016260162601627
  },
  "0901_base": {
    "iou_threshold": 0.3,
    "use_label": true,
    "samples": 262,
    "tp": 55,
    "fp": 224,
    "fn": 51,
    "precision": 0.1971326164874552,
    "recall": 0.5188679245283019,
    "f1": 0.2857142857142857
  },
  "0901_lora": {
    "iou_threshold": 0.3,
    "use_label": true,
    "samples": 262,
    "tp": 90,
    "fp": 16,
    "fn": 16,
    "precision": 0.8490566037735849,
    "recall": 0.8490566037735849,
    "f1": 0.8490566037735849
  }
}
```

