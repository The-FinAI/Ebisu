import json
import argparse
from collections import Counter
from typing import List, Any, Tuple, Set, Dict, Optional
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score


def load_annotations(path: str) -> List[dict]:
    """Load full Label Studio JSON export."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_choice_label(ann: dict, target_from_name: str = "commitment_label") -> Any:
    """
    从一个 annotation 里抽取单一的分类标签。
    默认使用 from_name='commitment_label' 的 choices 结果。
    """
    label = None
    for res in ann.get("result", []):
        if res.get("type") != "choices":
            continue
        if res.get("from_name") != target_from_name:
            continue
        choices = res.get("value", {}).get("choices", [])
        if not choices:
            continue
        # 同一个 annotator 对同一个样本多个 label 的情况，这里简单覆盖为最后一个
        label = choices[0]
    return label


def extract_spans(
    ann: dict,
    target_from_name: str = "finance_spans",
    use_label: bool = True,
) -> Set[Tuple[int, int, str]]:
    """
    从一个 annotation 中抽取 span 标注，返回 set[(start, end, label)].

    - 如果 use_label=True，则保留每个 label（例如 FinanceTerm/Company 等）。
    - 如果 use_label=False，则所有 span 的 label 统一为 "SPAN"，只区分边界。
    """
    spans = set()
    for res in ann.get("result", []):
        if res.get("type") != "labels":
            continue
        if res.get("from_name") != target_from_name:
            continue
        value = res.get("value", {})
        start = value.get("start")
        end = value.get("end")
        labels = value.get("labels", [])
        if start is None or end is None:
            continue
        start = int(start)
        end = int(end)
        if start >= end:
            continue
        if not labels:
            spans.add((start, end, "SPAN"))
        else:
            for lab in labels:
                spans.add((start, end, lab if use_label else "SPAN"))
    return spans


def krippendorff_alpha_nominal(ratings: List[List[Any]]) -> float:
    """
    Krippendorff's alpha for nominal data.
    ratings: [[rater1_label, rater2_label, ...] for each item]
             可以包含 None 表示缺失标注。
    """
    cats = set()
    for row in ratings:
        for v in row:
            if v is not None:
                cats.add(v)
    cats = sorted(cats)
    m = len(cats)
    if m == 0:
        return float("nan")

    cat2idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((m, m), dtype=float)  # coincidence matrix

    # 构建 coincidence matrix
    for row in ratings:
        vals = [v for v in row if v is not None]
        n = len(vals)
        if n < 2:
            continue
        counts = Counter(vals)
        for c in cats:
            n_c = counts.get(c, 0)
            for d in cats:
                n_d = counts.get(d, 0)
                i, j = cat2idx[c], cat2idx[d]
                if c == d:
                    O[i, j] += n_c * (n_c - 1) / (n - 1)
                else:
                    O[i, j] += n_c * n_d / (n - 1)

    N = O.sum()
    if N == 0:
        return float("nan")

    # Nominal distance：同类 0，异类 1
    Do = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                Do += O[i, j]
    Do /= N

    row_sums = O.sum(axis=1)
    De = 0.0
    for i in range(m):
        for j in range(m):
            if i != j:
                De += row_sums[i] * row_sums[j]
    De /= (N * N)

    if De == 0:
        return 1.0  # 完全一致的特殊情况

    return 1.0 - Do / De


# ===================== Pairwise metrics helpers =====================

def compute_choice_agreement_pair(
    data: List[dict],
    ann_a: int,
    ann_b: int,
    from_name: str,
) -> Dict[str, Any]:
    """Return metrics dict for (A,B) in choice mode."""
    y_a, y_b, ratings = [], [], []
    skipped_items = 0

    for task in data:
        anns = task.get("annotations", [])
        label_a = None
        label_b = None

        for ann in anns:
            uid = ann.get("completed_by")
            lab = extract_choice_label(ann, from_name)
            if uid == ann_a:
                label_a = lab
            elif uid == ann_b:
                label_b = lab

        if label_a is None or label_b is None:
            skipped_items += 1
            continue

        y_a.append(label_a)
        y_b.append(label_b)
        ratings.append([label_a, label_b])

    n_items = len(y_a)
    if n_items == 0:
        return {
            "n_items": 0,
            "skipped_items": skipped_items,
            "macro_f1": float("nan"),
            "kappa": float("nan"),
            "alpha": float("nan"),
            "label_set": [],
        }

    label_set = sorted(set(y_a) | set(y_b))
    macro_f1 = f1_score(y_a, y_b, average="macro", labels=label_set)
    kappa = cohen_kappa_score(y_a, y_b)
    alpha = krippendorff_alpha_nominal(ratings)

    return {
        "n_items": n_items,
        "skipped_items": skipped_items,
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "alpha": float(alpha),
        "label_set": label_set,
    }


def compute_span_classification_agreement_pair(
    data: List[dict],
    ann_a: int,
    ann_b: int,
    from_name: str,
    use_label: bool = True,
) -> Dict[str, Any]:
    """Return metrics dict for (A,B) in span_class mode."""
    y_a, y_b, ratings = [], [], []
    tasks_with_both = 0
    tasks_skipped = 0

    for task in data:
        anns = task.get("annotations", [])
        if not anns:
            continue

        ann_rec_a = None
        ann_rec_b = None
        for ann in anns:
            uid = ann.get("completed_by")
            if uid == ann_a and ann_rec_a is None:
                ann_rec_a = ann
            elif uid == ann_b and ann_rec_b is None:
                ann_rec_b = ann

        if ann_rec_a is None or ann_rec_b is None:
            tasks_skipped += 1
            continue

        spans_a = extract_spans(ann_rec_a, from_name, use_label)
        spans_b = extract_spans(ann_rec_b, from_name, use_label)
        tasks_with_both += 1

        universe = spans_a | spans_b
        for span in universe:
            la = 1 if span in spans_a else 0
            lb = 1 if span in spans_b else 0
            y_a.append(la)
            y_b.append(lb)
            ratings.append([la, lb])

    if not y_a:
        return {
            "tasks_with_both": tasks_with_both,
            "tasks_skipped": tasks_skipped,
            "n_spans": 0,
            "macro_f1": float("nan"),
            "kappa": float("nan"),
            "alpha": float("nan"),
        }

    label_set = sorted(set(y_a) | set(y_b))
    macro_f1 = f1_score(y_a, y_b, average="macro", labels=label_set)
    kappa = cohen_kappa_score(y_a, y_b)
    alpha = krippendorff_alpha_nominal(ratings)

    return {
        "tasks_with_both": tasks_with_both,
        "tasks_skipped": tasks_skipped,
        "n_spans": len(y_a),
        "macro_f1": float(macro_f1),
        "kappa": float(kappa),
        "alpha": float(alpha),
    }


def get_all_annotators(data: List[dict]) -> List[int]:
    return sorted(
        {
            ann["completed_by"]
            for task in data
            for ann in task.get("annotations", [])
            if ann.get("result")
        }
    )


def print_choice_report(ann_a: int, ann_b: int, m: Dict[str, Any]) -> None:
    print(f"\n[Choice mode] A={ann_a} vs B={ann_b}")
    print(f"Items with both: {m['n_items']}, skipped: {m['skipped_items']}")
    if m["n_items"] == 0:
        print("No items with both annotators.")
        return
    print("Label set:", m["label_set"])
    print(f"Macro-F1: {m['macro_f1']:.4f}")
    print(f"Cohen's kappa: {m['kappa']:.4f}")
    print(f"Krippendorff's alpha: {m['alpha']:.4f}")


def print_span_report(ann_a: int, ann_b: int, m: Dict[str, Any]) -> None:
    print(f"\n[Span classification mode] A={ann_a} vs B={ann_b}")
    print(f"Tasks with both: {m['tasks_with_both']}, skipped tasks: {m['tasks_skipped']}")
    print(f"Total span samples (union): {m['n_spans']}")
    if m["n_spans"] == 0:
        print("No span universe — cannot compute span-level agreement.")
        return
    print(f"Macro-F1: {m['macro_f1']:.4f}")
    print(f"Cohen's kappa: {m['kappa']:.4f}")
    print(f"Krippendorff's alpha: {m['alpha']:.4f}")


# ============================ main ============================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute agreement metrics for Label Studio annotations.\n"
            "Modes:\n"
            "  - choice      : single-label classification (Macro-F1, kappa, alpha)\n"
            "  - span_class  : span-level agreement as 0/1 classification (Macro-F1, kappa, alpha)\n"
        )
    )
    parser.add_argument("--file", required=True, help="Path to Label Studio JSON export.")
    parser.add_argument(
        "--mode",
        choices=["choice", "span_class"],
        default="choice",
        help="Agreement type.",
    )
    parser.add_argument(
        "--from_name",
        default=None,
        help=(
            "Name of the field in Label Studio.\n"
            "  - choice mode    : choices field (default: commitment_label)\n"
            "  - span_class     : span/labels field (default: finance_spans)"
        ),
    )
    parser.add_argument(
        "--ann_a",
        type=int,
        default=None,
        help="Annotator ID A. If --one_vs_others is set, this must be provided (or will default to the first annotator).",
    )
    parser.add_argument(
        "--ann_b",
        type=int,
        default=None,
        help="Annotator ID B (pairwise). If omitted and --one_vs_others is not set, script expects exactly 2 annotators in the file.",
    )
    parser.add_argument(
        "--one_vs_others",
        action="store_true",
        help="Compute A vs every other annotator (B,C,...) instead of a single pair (A,B).",
    )
    parser.add_argument(
        "--ignore_span_label",
        action="store_true",
        help="In span_class mode, ignore span label type and only require (start, end) to match.",
    )
    args = parser.parse_args()

    data = load_annotations(args.file)

    # 默认 from_name
    if args.from_name is None:
        from_name = "commitment_label" if args.mode == "choice" else "finance_spans"
    else:
        from_name = args.from_name

    annotators = get_all_annotators(data)
    if not annotators:
        raise ValueError("No annotators found in this file (no annotation results).")

    # 选择 A
    ann_a = args.ann_a if args.ann_a is not None else annotators[0]
    if ann_a not in annotators:
        raise ValueError(f"Annotator A={ann_a} not found. Known annotators: {annotators}")

    print(f"Mode: {args.mode}")
    print(f"from_name: {from_name}")
    print(f"Known annotators: {annotators}")
    if args.mode == "span_class":
        print(f"ignore_span_label: {args.ignore_span_label}")

    # one-vs-others：A vs all others
    if args.one_vs_others:
        others = [x for x in annotators if x != ann_a]
        if not others:
            raise ValueError(f"Only one annotator ({ann_a}) found; cannot do one-vs-others.")

        print(f"\n[One-vs-others] A={ann_a} vs {others}")

        for ann_b in others:
            if args.mode == "choice":
                m = compute_choice_agreement_pair(data, ann_a, ann_b, from_name)
                print_choice_report(ann_a, ann_b, m)
            else:
                m = compute_span_classification_agreement_pair(
                    data, ann_a, ann_b, from_name, use_label=not args.ignore_span_label
                )
                print_span_report(ann_a, ann_b, m)
        return

    # 原来的 pairwise 逻辑：如果没提供 ann_b，则要求文件里恰好 2 个 annotator
    if args.ann_b is None:
        if len(annotators) != 2:
            raise ValueError(
                f"Expected exactly 2 annotators when --ann_b not set (and --one_vs_others not set), "
                f"but found {len(annotators)} annotators: {annotators}\n"
                f"Tip: use --one_vs_others with --ann_a {ann_a}."
            )
        ann_b = annotators[1] if annotators[0] == ann_a else annotators[0]
    else:
        ann_b = args.ann_b
        if ann_b not in annotators:
            raise ValueError(f"Annotator B={ann_b} not found. Known annotators: {annotators}")
        if ann_b == ann_a:
            raise ValueError("ann_a and ann_b must be different.")

    print(f"\n[Pairwise] Using annotators: A={ann_a}, B={ann_b}")

    if args.mode == "choice":
        m = compute_choice_agreement_pair(data, ann_a, ann_b, from_name)
        print_choice_report(ann_a, ann_b, m)
    else:
        m = compute_span_classification_agreement_pair(
            data, ann_a, ann_b, from_name, use_label=not args.ignore_span_label
        )
        print_span_report(ann_a, ann_b, m)


if __name__ == "__main__":
    main()
