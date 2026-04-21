import argparse

from convert_dataset import convert_to_eval_jsonl, convert_to_train_jsonl
from eval_pr import compute_pr, load_jsonl
from run_pred import run_eval_and_export


def parse_args():
    parser = argparse.ArgumentParser(description='Wrapper for split evaluation scripts.')
    parser.add_argument('--source_json', default='/root/.cache/modelscope/hub/datasets/Echo0174/Trash_floater_qwen3/test.json')
    parser.add_argument('--train_out', default='test.jsonl')
    parser.add_argument('--eval_out', default='test_eval.jsonl')
    parser.add_argument('--pred_jsonl', default='pred.jsonl')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--max_samples', type=int, default=10)
    parser.add_argument('--ignore_label', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    convert_to_train_jsonl(args.source_json, args.train_out, args.max_samples)
    convert_to_eval_jsonl(args.source_json, args.eval_out, args.max_samples)
    run_eval_and_export(
        argparse.Namespace(
            model='Qwen/Qwen3-VL-8B-Instruct',
            local_path='/root/liantaoding_dev/codes/ms-swift',
            subset='test_eval',
            eval_output_dir='eval_output',
            eval_limit=args.max_samples,
            pred_out=args.pred_jsonl))

    gt_records = load_jsonl(args.train_out)
    pred_records = load_jsonl(args.pred_jsonl)
    result = compute_pr(
        gt_records=gt_records,
        pred_records=pred_records,
        iou_thr=args.iou_threshold,
        use_label=not args.ignore_label,
        max_samples=args.max_samples)
    print(result)


if __name__ == '__main__':
    main()