import argparse
import json
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


def get_record_key(record, default_key):
    images = record.get('images', [])
    if isinstance(images, list) and images and isinstance(images[0], str):
        return images[0]
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


def compute_pr(gt_records, pred_records, iou_thr=0.5, use_label=True, max_samples=None):
    gt_map = {get_record_key(rec, f'idx_{i}'): rec for i, rec in enumerate(gt_records)}
    pred_map = {get_record_key(rec, f'idx_{i}'): rec for i, rec in enumerate(pred_records)}
    keys = sorted(set(gt_map.keys()) | set(pred_map.keys()))
    if max_samples is not None and max_samples > 0:
        keys = keys[:max_samples]

    tp = 0
    fp = 0
    fn = 0
    for key in keys:
        gt_items = extract_gt_items(gt_map.get(key, {}))
        pred_items = extract_pred_items(pred_map.get(key, {}))

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
    parser.add_argument('--gt_jsonl', default='test.jsonl')
    parser.add_argument('--pred_jsonl', default='pred.jsonl')
    parser.add_argument('--iou_threshold', type=float, default=0.5)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--ignore_label', action='store_true')
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
        max_samples=args.max_samples)
    print('pr_result:', json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
