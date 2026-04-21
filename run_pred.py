import argparse
import glob
import json
import os
import re
from multiprocessing import freeze_support

from swift.arguments import EvalArguments
from swift.pipelines.eval import SwiftEval


def find_latest_review_file(eval_output_dir, model_name, subset_name):
    model_suffix = model_name.split('/')[-1]
    pattern = os.path.join(
        eval_output_dir,
        'native',
        '*',
        'reviews',
        model_suffix,
        f'general_vqa_{subset_name}.jsonl')
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError(f'no review file found with pattern: {pattern}')
    return max(candidates, key=os.path.getmtime)


def extract_image_from_input(input_text):
    if not isinstance(input_text, str):
        return ''
    m = re.search(r'gradio_api/file=([^)]+)\)', input_text)
    if m:
        return m.group(1).strip()
    return ''


def export_pred_jsonl(review_path, pred_out):
    with open(review_path, 'r', encoding='utf-8') as fin, open(pred_out, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            score_info = row.get('sample_score', {}).get('score', {})
            prediction = score_info.get('prediction', '')
            target = row.get('target', '')
            image_path = extract_image_from_input(row.get('input', ''))
            pred_row = {
                'images': [image_path] if image_path else [],
                'prediction': prediction,
                'target': target
            }
            fout.write(json.dumps(pred_row, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--local_path', default='/root/liantaoding_dev/codes/ms-swift')
    parser.add_argument('--subset', default='test_eval')
    parser.add_argument('--eval_output_dir', default='eval_output')
    parser.add_argument('--eval_limit', type=int, default=10)
    parser.add_argument('--pred_out', default='pred.jsonl')
    return parser.parse_args()


def run_eval_and_export(args):
    eval_args = EvalArguments(
        model=args.model,
        eval_dataset=['general_vqa'],
        eval_dataset_args={
            'general_vqa': {
                'local_path': args.local_path,
                'subset_list': [args.subset],
                'metric_list': ['BLEU', 'Rouge']
            }
        },
        eval_output_dir=args.eval_output_dir,
        eval_num_proc=1,
        eval_limit=args.eval_limit)
    SwiftEval(eval_args).main()
    review_path = find_latest_review_file(args.eval_output_dir, args.model, args.subset)
    export_pred_jsonl(review_path, args.pred_out)
    print(f'generated: {args.pred_out}')
    print(f'from review: {review_path}')


def main():
    args = parse_args()
    run_eval_and_export(args)


if __name__ == '__main__':
    freeze_support()
    main()
