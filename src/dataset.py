import csv
import random
from dataclasses import dataclass
from pathlib import Path

# CSV 파일 위치: data/test/<subject>_test.csv, data/dev/<subject>_dev.csv
DATA_DIR = Path(__file__).parent.parent / "data"

CATEGORIES: dict[str, list[str]] = {
    "STEM": [
        "abstract_algebra", "astronomy", "college_biology", "college_chemistry",
        "college_computer_science", "college_mathematics", "college_physics",
        "computer_security", "conceptual_physics", "electrical_engineering",
        "elementary_mathematics", "high_school_biology", "high_school_chemistry",
        "high_school_computer_science", "high_school_mathematics", "high_school_physics",
        "high_school_statistics", "machine_learning",
    ],
    "Humanities": [
        "formal_logic", "high_school_european_history", "high_school_us_history",
        "high_school_world_history", "international_law", "jurisprudence",
        "logical_fallacies", "moral_disputes", "moral_scenarios", "philosophy",
        "prehistory", "professional_law", "world_religions",
    ],
    "Social Sciences": [
        "econometrics", "high_school_geography", "high_school_government_and_politics",
        "high_school_macroeconomics", "high_school_microeconomics", "high_school_psychology",
        "human_sexuality", "professional_psychology", "public_relations",
        "security_studies", "sociology", "us_foreign_policy",
    ],
    "Other": [
        "anatomy", "business_ethics", "clinical_knowledge", "college_medicine",
        "global_facts", "human_aging", "management", "marketing", "medical_genetics",
        "miscellaneous", "nutrition", "professional_accounting", "professional_medicine",
        "virology",
    ],
}

SUBJECT_TO_CATEGORY: dict[str, str] = {
    subject: cat
    for cat, subjects in CATEGORIES.items()
    for subject in subjects
}

CHOICES = ["A", "B", "C", "D"]


@dataclass
class Question:
    subject: str
    category: str
    question: str
    choices: list[str]   # [A, B, C, D]
    answer: str          # "A" | "B" | "C" | "D"
    few_shot_examples: list["Question"]


def _read_csv(path: Path) -> list[dict]:
    # MMLU CSV 형식: 헤더 없음, 컬럼 순서 = question, A, B, C, D, answer(0~3)
    rows = []
    with path.open(encoding="utf-8") as f:
        for row in csv.reader(f):
            if len(row) < 6:
                continue
            rows.append({
                "question": row[0],
                "choices": [row[1], row[2], row[3], row[4]],
                "answer": int(row[5]),
            })
    return rows


def _load_subject(subject: str) -> tuple[list[dict], list[dict]]:
    test_path = DATA_DIR / "test" / f"{subject}_test.csv"
    dev_path  = DATA_DIR / "dev"  / f"{subject}_dev.csv"

    if not test_path.exists():
        raise FileNotFoundError(f"파일 없음: {test_path}")
    if not dev_path.exists():
        raise FileNotFoundError(f"파일 없음: {dev_path}")

    return _read_csv(test_path), _read_csv(dev_path)


def _row_to_question(row: dict, subject: str) -> Question:
    return Question(
        subject=subject,
        category=SUBJECT_TO_CATEGORY.get(subject, "Other"),
        question=row["question"],
        choices=row["choices"],
        answer=CHOICES[row["answer"]],
        few_shot_examples=[],
    )


def load_balanced_questions(
    total: int = 100,
    num_shots: int = 5,
    seed: int = 42,
) -> list[Question]:
    rng = random.Random(seed)
    per_category = total // len(CATEGORIES)
    questions: list[Question] = []

    for category, subjects in CATEGORIES.items():
        # 1차: 과목별 균등 배분
        per_subject = max(1, per_category // len(subjects))
        category_questions: list[Question] = []
        subject_pools: dict[str, tuple[list, list]] = {}

        for subject in subjects:
            try:
                test_rows, dev_rows = _load_subject(subject)
            except FileNotFoundError:
                continue
            subject_pools[subject] = (test_rows, dev_rows)
            dev_qs = [_row_to_question(r, subject) for r in dev_rows]
            sampled = rng.sample(test_rows, min(per_subject, len(test_rows)))
            for row in sampled:
                q = _row_to_question(row, subject)
                q.few_shot_examples = dev_qs[:num_shots]
                category_questions.append(q)

        # 2차: 부족분을 남은 풀에서 추가 샘플링해 per_category 채우기
        already_used: dict[str, set] = {
            q.subject: set() for q in category_questions
        }
        for q in category_questions:
            already_used[q.subject].add(q.question)

        remaining_subjects = list(subject_pools.keys())
        rng.shuffle(remaining_subjects)
        idx = 0
        while len(category_questions) < per_category and remaining_subjects:
            subject = remaining_subjects[idx % len(remaining_subjects)]
            idx += 1
            test_rows, dev_rows = subject_pools[subject]
            used = already_used.setdefault(subject, set())
            pool = [r for r in test_rows if r["question"] not in used]
            if not pool:
                remaining_subjects = [s for s in remaining_subjects if s != subject]
                continue
            row = rng.choice(pool)
            used.add(row["question"])
            dev_qs = [_row_to_question(r, subject) for r in dev_rows]
            q = _row_to_question(row, subject)
            q.few_shot_examples = dev_qs[:num_shots]
            category_questions.append(q)

        rng.shuffle(category_questions)
        questions.extend(category_questions[:per_category])

    rng.shuffle(questions)
    return questions
