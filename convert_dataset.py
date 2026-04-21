import argparse
import json


def build_assistant_template(obj_num):
    if obj_num <= 0:
        obj_num = 1
    lines = ['[']
    for i in range(obj_num):
        suffix = ',' if i < obj_num - 1 else ''
        lines.append(f'\t{{"bbox_2d": <bbox>, "label": "<ref-object>"}}{suffix}')
    lines.append(']')
    return '\n'.join(lines)


def convert_to_train_jsonl(input_path, output_path, max_samples=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if max_samples and max_samples > 0:
        data = data[:max_samples]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in data:
            objects = item.get('objects', {})
            refs = objects.get('ref', [])
            bboxes = objects.get('bbox', [])
            images = item.get('images', [])
            image_path = images[0] if images else ''
            obj_num = min(len(refs), len(bboxes))

            record = {
                'messages': [{
                    'role': 'user',
                    'content': '<image>找到图像中的<ref-object>'
                }, {
                    'role': 'assistant',
                    'content': build_assistant_template(obj_num)
                }],
                'images': [image_path],
                'objects': {
                    'ref': refs[:obj_num],
                    'bbox': bboxes[:obj_num]
                }
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')


def convert_to_eval_jsonl(input_path, output_path, max_samples=None):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if max_samples and max_samples > 0:
        data = data[:max_samples]

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for item in data:
            objects = item.get('objects', {})
            refs = objects.get('ref', [])
            bboxes = objects.get('bbox', [])
            images = item.get('images', [])
            image_path = images[0] if images else ''
            obj_num = min(len(refs), len(bboxes))
            label = refs[0] if refs else '目标'

            answer_items = []
            for ref, bbox in zip(refs[:obj_num], bboxes[:obj_num]):
                answer_items.append({'bbox_2d': bbox, 'label': ref})

            record = {
                'messages': [{
                    'role': 'user',
                    'content': [{
                        'type': 'text',
                        'text': f'找到图像中的{label}'
                    }, {
                        'type': 'image_url',
                        'image_url': {
                            'url': image_path
                        }
                    }]
                }],
                'answer': json.dumps(answer_items, ensure_ascii=False)
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + '\n')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--source_json',
        default='/root/.cache/modelscope/hub/datasets/Echo0174/Trash_floater_qwen3/test.json')
    parser.add_argument('--train_out', default='test.jsonl')
    parser.add_argument('--eval_out', default='test_eval.jsonl')
    parser.add_argument('--max_samples', type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    convert_to_train_jsonl(args.source_json, args.train_out, args.max_samples)
    convert_to_eval_jsonl(args.source_json, args.eval_out, args.max_samples)
    print(f'generated: {args.train_out}, {args.eval_out}')


if __name__ == '__main__':
    main()
