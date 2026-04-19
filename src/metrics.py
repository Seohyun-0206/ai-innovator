import re
from collections import defaultdict
from typing import Optional
import numpy as np


ANSWER_PATTERN = re.compile(r"\b([A-D])\b")


def parse_answer(text: str) -> Optional[str]:
    text = text.strip()
    # "Answer: X" 또는 "The answer is X" 패턴 우선
    explicit = re.search(r"(?:answer\s*(?:is|:)\s*)([A-D])", text, re.IGNORECASE)
    if explicit:
        return explicit.group(1).upper()
    # 첫 번째 A/B/C/D 토큰
    match = ANSWER_PATTERN.search(text)
    if match:
        return match.group(1).upper()
    return None


def compute_metrics(entries: list[dict]) -> dict:
    latencies = [e["latency"] for e in entries if e.get("latency") is not None]
    input_tokens = [e["input_tokens"] for e in entries]
    output_tokens = [e["output_tokens"] for e in entries]
    costs = [e["cost"] for e in entries]

    # Accuracy
    answered = [e for e in entries if not e["api_error"]]
    parsed = [e for e in answered if e["parsed_answer"] is not None]
    correct = [e for e in parsed if e["correct"]]

    total = len(entries)
    api_fail = sum(1 for e in entries if e["api_error"])
    parse_fail = len(answered) - len(parsed)

    # 카테고리별 정확도
    cat_totals: dict[str, int] = defaultdict(int)
    cat_correct: dict[str, int] = defaultdict(int)
    for e in parsed:
        cat_totals[e["category"]] += 1
        if e["correct"]:
            cat_correct[e["category"]] += 1
    category_accuracy = {
        cat: cat_correct[cat] / cat_totals[cat] if cat_totals[cat] else 0.0
        for cat in cat_totals
    }

    # 과목별 정확도
    subj_totals: dict[str, int] = defaultdict(int)
    subj_correct: dict[str, int] = defaultdict(int)
    for e in parsed:
        subj_totals[e["subject"]] += 1
        if e["correct"]:
            subj_correct[e["subject"]] += 1
    subject_accuracy = {
        subj: subj_correct[subj] / subj_totals[subj] if subj_totals[subj] else 0.0
        for subj in subj_totals
    }
    subject_std = float(np.std(list(subject_accuracy.values()))) if subject_accuracy else 0.0

    lat_arr = np.array(latencies) if latencies else np.array([0.0])
    total_cost = sum(costs)

    return {
        "total_questions": total,
        "api_failure_count": api_fail,
        "parse_failure_count": parse_fail,
        "api_failure_rate": api_fail / total if total else 0.0,
        "parse_failure_rate": parse_fail / total if total else 0.0,
        "accuracy": len(correct) / len(parsed) if parsed else 0.0,
        "category_accuracy": category_accuracy,
        "subject_accuracy": subject_accuracy,
        "subject_accuracy_std": subject_std,
        "latency_mean": float(lat_arr.mean()),
        "latency_p50": float(np.percentile(lat_arr, 50)),
        "latency_p95": float(np.percentile(lat_arr, 95)),
        "total_input_tokens": sum(input_tokens),
        "total_output_tokens": sum(output_tokens),
        "total_cost_usd": total_cost,
        "cost_per_question_usd": total_cost / total if total else 0.0,
        "cost_per_1k_questions_usd": (total_cost / total * 1000) if total else 0.0,
    }


def compute_score(metrics: dict) -> dict:
    accuracy = metrics["accuracy"]
    stability = 1.0 - metrics["api_failure_rate"] - metrics["parse_failure_rate"]
    stability = max(0.0, stability)

    performance = 0.7 * accuracy + 0.3 * stability

    lat_score = max(0.0, 1.0 - metrics["latency_mean"] / 30.0)
    cost_ref = 0.01  # $0.01 per question as reference
    cost_score = max(0.0, 1.0 - metrics["cost_per_question_usd"] / cost_ref)
    token_ref = 2000
    token_ratio = (metrics["total_input_tokens"] + metrics["total_output_tokens"]) / (
        max(metrics["total_questions"], 1) * token_ref
    )
    token_score = max(0.0, 1.0 - token_ratio)
    efficiency = 0.4 * lat_score + 0.4 * cost_score + 0.2 * token_score

    capability = stability

    total = 0.55 * performance + 0.25 * efficiency + 0.20 * capability

    return {
        "performance": round(performance * 100, 2),
        "efficiency": round(efficiency * 100, 2),
        "capability": round(capability * 100, 2),
        "total": round(total * 100, 2),
    }
