import argparse
import json
import os
import re


def load_jsonl(path):
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _extract_image_path(record):
    """从 record 中提取图像路径，兼容 str 列表和 dict 列表两种格式。"""
    images = record.get('images', [])
    if isinstance(images, list) and images:
        first = images[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            path = first.get('path') or first.get('url')
            if isinstance(path, str):
                return path
    messages = record.get('messages', [])
    if isinstance(messages, list):
        for msg in messages:
            content = msg.get('content')
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get('type') == 'image_url':
                        image_obj = part.get('image_url', {})
                        if isinstance(image_obj, dict) and isinstance(image_obj.get('url'), str):
                            return image_obj['url']
    return None


def _extract_prompt(record):
    """从 record 的 user message 中提取 prompt 文本（去掉 <image> 标记）。"""
    messages = record.get('messages', [])
    if isinstance(messages, list):
        for msg in messages:
            if msg.get('role') == 'user':
                content = msg.get('content', '')
                if isinstance(content, str):
                    return content.replace('<image>', '').strip()
                if isinstance(content, list):
                    texts = []
                    for part in content:
                        if isinstance(part, str):
                            texts.append(part)
                        elif isinstance(part, dict) and part.get('type') == 'text':
                            texts.append(part.get('text', ''))
                    return ''.join(texts).replace('<image>', '').strip()
    return ''


def get_record_key(record, default_key):
    """使用 image_path + prompt 组合作为匹配 key。"""
    image_path = _extract_image_path(record)
    prompt = _extract_prompt(record)
    if image_path:
        return f'{image_path}||{prompt}' if prompt else image_path
    return default_key


def parse_bbox_items(value):
    if value is None:
        return []
    data = value
    if isinstance(data, str):
        data = data.strip()
        if not data:
            return []
        if '<bbox>' in data or '<ref-object>' in data:
            return []
        try:
            data = json.loads(data)
        except json.JSONDecodeError:
            m = re.search(r'(\[.*\])', data, flags=re.DOTALL)
            if not m:
                return []
            try:
                data = json.loads(m.group(1))
            except json.JSONDecodeError:
                return []
    if not isinstance(data, list):
        return []
    items = []
    for item in data:
        if not isinstance(item, dict):
            continue
        bbox = item.get('bbox_2d') or item.get('bbox')
        label = item.get('label', '')
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                bbox = [float(x) for x in bbox]
            except (TypeError, ValueError):
                continue
            items.append({'bbox': bbox, 'label': str(label)})
    return items


def extract_gt_items(record):
    # 1) objects.ref + objects.bbox format
    objects = record.get('objects', {})
    refs = objects.get('ref', []) if isinstance(objects, dict) else []
    bboxes = objects.get('bbox', []) if isinstance(objects, dict) else []
    n = min(len(refs), len(bboxes))
    items = []
    for ref, bbox in zip(refs[:n], bboxes[:n]):
        if isinstance(bbox, list) and len(bbox) == 4:
            try:
                bbox = [float(x) for x in bbox]
            except (TypeError, ValueError):
                continue
            items.append({'bbox': bbox, 'label': str(ref)})
    if items:
        return items

    # 2) answer / target field
    for key in ('answer', 'target'):
        items = parse_bbox_items(record.get(key))
        if items:
            return items

    # 3) assistant message content
    messages = record.get('messages', [])
    if isinstance(messages, list):
        for msg in messages:
            if msg.get('role') == 'assistant':
                items = parse_bbox_items(msg.get('content'))
                if items:
                    return items
    return items


def extract_pred_items(record):
    for key in ('prediction', 'pred', 'predict', 'response', 'answer', 'target'):
        items = parse_bbox_items(record.get(key))
        if items:
            return items
    messages = record.get('messages', [])
    if isinstance(messages, list):
        for msg in messages:
            if msg.get('role') == 'assistant':
                items = parse_bbox_items(msg.get('content'))
                if items:
                    return items
    return []


def bbox_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def draw_boxes_on_image(image_path, gt_items, pred_items, save_path):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    except (IOError, OSError):
        font = ImageFont.load_default()

    for item in gt_items:
        x1, y1, x2, y2 = item['bbox']
        x1 = x1 / 1000 * img.width
        y1 = y1 / 1000 * img.height
        x2 = x2 / 1000 * img.width
        y2 = y2 / 1000 * img.height
        draw.rectangle([x1, y1, x2, y2], outline='green', width=2)
        label = f"GT:{item['label']}" if item['label'] else 'GT'
        draw.text((x1, max(y1 - 18, 0)), label, fill='green', font=font)

    for item in pred_items:
        x1, y1, x2, y2 = item['bbox']
        x1 = x1 / 1000 * img.width
        y1 = y1 / 1000 * img.height
        x2 = x2 / 1000 * img.width
        y2 = y2 / 1000 * img.height
        draw.rectangle([x1, y1, x2, y2], outline='red', width=2)
        label = f"Pred:{item['label']}" if item['label'] else 'Pred'
        draw.text((x1, max(y1 - 18, 0)), label, fill='red', font=font)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def _build_key_map(records, source_name):
    """构建 key -> record 的映射，并校验 key 唯一性。"""
    key_map = {}
    for i, rec in enumerate(records):
        key = get_record_key(rec, f'idx_{i}')
        if key in key_map:
            raise ValueError(
                f'{source_name} 中存在重复 key: "{key}"（第 {key_map[key][0]+1} 行与第 {i+1} 行冲突）')
        key_map[key] = (i, rec)
    return {k: v[1] for k, v in key_map.items()}


def compute_pr(gt_records, pred_records, iou_thr=0.5, use_label=True, max_samples=None,
               draw_dir=None):
    gt_map = _build_key_map(gt_records, 'gt_jsonl')
    pred_map = _build_key_map(pred_records, 'pred_jsonl')
    keys = sorted(set(gt_map.keys()) | set(pred_map.keys()))
    if max_samples is not None and max_samples > 0:
        keys = keys[:max_samples]

    tp = 0
    fp = 0
    fn = 0
    for idx, key in enumerate(keys):
        gt_items = extract_gt_items(gt_map.get(key, {}))
        pred_items = extract_pred_items(pred_map.get(key, {}))

        if draw_dir:
            image_path = key.split('||')[0] if '||' in key else key
            if os.path.isfile(image_path):
                save_name = f'{idx:04d}_{os.path.basename(image_path)}'
                draw_boxes_on_image(image_path, gt_items, pred_items,
                                    os.path.join(draw_dir, save_name))

        pairs = []
        for p_idx, pred in enumerate(pred_items):
            for g_idx, gt in enumerate(gt_items):
                if use_label and pred['label'] and gt['label'] and pred['label'] != gt['label']:
                    continue
                iou = bbox_iou(pred['bbox'], gt['bbox'])
                if iou >= iou_thr:
                    pairs.append((iou, p_idx, g_idx))
        pairs.sort(reverse=True, key=lambda x: x[0])

        matched_pred = set()
        matched_gt = set()
        for _, p_idx, g_idx in pairs:
            if p_idx in matched_pred or g_idx in matched_gt:
                continue
            matched_pred.add(p_idx)
            matched_gt.add(g_idx)

        tp += len(matched_pred)
        fp += len(pred_items) - len(matched_pred)
        fn += len(gt_items) - len(matched_gt)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {
        'iou_threshold': iou_thr,
        'use_label': use_label,
        'samples': len(keys),
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_jsonl', default='eval_output_0910_base/_prepared_dataset/infer_tree_damage_per_label_with_negatives/tree_damage_per_label_with_negatives.jsonl')
    parser.add_argument('--pred_jsonl', default='eval_output_0910_base/tree_damage_per_label_with_negatives_result.jsonl')
    parser.add_argument('--iou_threshold', type=float, default=0.3)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--ignore_label', action='store_true')
    parser.add_argument('--draw_dir', default='eval_output_0910_base',
                        help='Save images with GT(green) and Pred(red) boxes to this directory')
    return parser.parse_args()


def main():
    args = parse_args()
    gt_records = load_jsonl(args.gt_jsonl)
    pred_records = load_jsonl(args.pred_jsonl)
    result = compute_pr(
        gt_records=gt_records,
        pred_records=pred_records,
        iou_thr=args.iou_threshold,
        use_label=not args.ignore_label,
        max_samples=args.max_samples,
        draw_dir=args.draw_dir)
    print('pr_result:', json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
