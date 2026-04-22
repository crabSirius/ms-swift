import argparse
import json
import os
from multiprocessing import freeze_support

from swift.arguments import InferArguments
from swift.pipelines.infer import infer_main
from swift.utils import read_from_jsonl


def _first_ref(objects):
    refs = objects.get('ref', [])
    if isinstance(refs, list):
        for ref in refs:
            if isinstance(ref, str) and ref.strip():
                return ref.strip()
        return ''
    if isinstance(refs, str):
        return refs.strip()
    return ''


def _build_assistant_target(objects):
    bboxes = objects.get('bbox', [])
    if not isinstance(bboxes, list):
        return '[]'
    ref_name = _first_ref(objects)
    if not bboxes:
        return '[]'
    payload = []
    for bbox in bboxes:
        if not isinstance(bbox, list):
            continue
        payload.append({'bbox_2d': bbox, 'label': ref_name})
    return json.dumps(payload, ensure_ascii=False)


def prepare_infer_dataset(local_path, subset, output_dir):
    """Prepare the dataset JSONL for swift infer.

    Handles two data formats:
    1. Objects-based: messages with <ref-object>/<bbox> placeholders + objects field
    2. Answer-based: messages + answer field

    Adds an assistant message as ground truth so InferRequest.remove_response
    can extract it as labels.
    """
    src_path = os.path.join(local_path, f'{subset}.jsonl')
    if not os.path.exists(src_path):
        raise FileNotFoundError(f'subset file not found: {src_path}')

    prepared_root = os.path.join(output_dir, '_prepared_dataset')
    os.makedirs(prepared_root, exist_ok=True)
    safe_subset = subset.replace('/', '_').replace('\\', '_')
    prepared_dir = os.path.join(prepared_root, f'infer_{safe_subset}')
    os.makedirs(prepared_dir, exist_ok=True)
    dst_path = os.path.join(prepared_dir, f'{subset}.jsonl')

    with open(src_path, 'r', encoding='utf-8') as fin, \
         open(dst_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            raw = line.strip()
            if not raw:
                continue
            row = json.loads(raw)
            objects = row.get('objects', {})
            ref_name = _first_ref(objects)
            assistant_target = _build_assistant_target(objects)

            messages = row.get('messages', [])
            if isinstance(messages, list):
                for msg in messages:
                    if not isinstance(msg, dict):
                        continue
                    content = msg.get('content')
                    if not isinstance(content, str):
                        continue
                    if '<ref-object>' in content and ref_name:
                        content = content.replace('<ref-object>', ref_name)
                    if msg.get('role') == 'assistant' and (
                            '<bbox>' in content or '<ref-object>' in content):
                        content = assistant_target
                    msg['content'] = content
                row['messages'] = messages

            has_assistant = any(
                isinstance(m, dict) and m.get('role') == 'assistant'
                for m in messages)
            if not has_assistant:
                answer = row.get('answer', '')
                if not answer and assistant_target != '[]':
                    answer = assistant_target
                if answer:
                    messages.append({'role': 'assistant', 'content': answer})
                    row['messages'] = messages

            out_row = {'messages': row['messages']}
            for key in ('images', 'videos', 'audios'):
                if key in row:
                    out_row[key] = row[key]
            fout.write(json.dumps(out_row, ensure_ascii=False) + '\n')

    return dst_path


def export_pred_jsonl(result_path, pred_out):
    data_list = read_from_jsonl(result_path)
    with open(pred_out, 'w', encoding='utf-8') as fout:
        for data in data_list:
            prediction = data.get('response', '')
            target = data.get('labels', '') or ''
            images = data.get('images', [])
            pred_row = {
                'images': images if images else [],
                'prediction': prediction,
                'target': target,
            }
            fout.write(json.dumps(pred_row, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='Qwen/Qwen3-VL-8B-Instruct')
    parser.add_argument('--adapters', nargs='*', default=['/root/liantaoding_dev/codes/ms-swift/output/v1/checkpoint-250'],
                        help='LoRA adapter paths, e.g. output/v1/checkpoint-250')
    parser.add_argument('--merge_lora', default=True,
                        help='Merge LoRA weights into the base model before inference')
    parser.add_argument('--local_path',
                        default='/root/liantaoding_datas/images/damage_tree/damage_tree_0910/ms_swift')
    parser.add_argument('--subset', default='tree_damage_per_label_with_negatives')
    parser.add_argument('--output_dir', default='eval_output_0910')
    parser.add_argument('--eval_limit', type=int, default=None)
    parser.add_argument('--pred_out', default='pred.jsonl')
    parser.add_argument('--infer_backend', default='transformers',
                        choices=['vllm', 'transformers', 'sglang', 'lmdeploy'])
    return parser.parse_args()


def run_infer_and_export(args):
    prepared_path = prepare_infer_dataset(
        local_path=args.local_path,
        subset=args.subset,
        output_dir=args.output_dir)

    result_path = os.path.join(args.output_dir, f'{args.subset}_result.jsonl')

    infer_kwargs = dict(
        model=args.model,
        model_type='qwen3_vl',
        template='qwen3_vl',
        infer_backend=args.infer_backend,
        val_dataset=[prepared_path],
        val_dataset_sample=args.eval_limit,
        result_path=result_path,
        temperature=0.3,
        max_new_tokens=2048,
        top_k=20,
        top_p=0.7,
        repetition_penalty=1.05,
        stream=False)

    if args.adapters:
        infer_kwargs['adapters'] = args.adapters
    if args.merge_lora:
        infer_kwargs['merge_lora'] = True

    infer_args = InferArguments(**infer_kwargs)

    infer_main(infer_args)

    export_pred_jsonl(result_path, args.pred_out)
    print(f'generated: {args.pred_out}')
    print(f'from result: {result_path}')


def main():
    args = parse_args()
    run_infer_and_export(args)


if __name__ == '__main__':
    freeze_support()
    main()
