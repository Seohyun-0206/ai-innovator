import csv
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
_HF_REPO = "cais/mmlu"


def _is_data_ready() -> bool:
    test_dir = DATA_DIR / "test"
    return test_dir.exists() and len(list(test_dir.glob("*_test.csv"))) > 0


def download_mmlu_if_needed(verbose: bool = True) -> None:
    if _is_data_ready():
        return

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    from src.dataset import CATEGORIES

    all_subjects = [s for subjects in CATEGORIES.values() for s in subjects]

    test_dir = DATA_DIR / "test"
    dev_dir = DATA_DIR / "dev"
    test_dir.mkdir(parents=True, exist_ok=True)
    dev_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"MMLU 데이터셋 다운로드 중... (총 {len(all_subjects)}개 과목)")

    for subject in all_subjects:
        test_path = test_dir / f"{subject}_test.csv"
        dev_path = dev_dir / f"{subject}_dev.csv"
        if test_path.exists() and dev_path.exists():
            continue
        if verbose:
            print(f"  다운로드: {subject}")
        _download_subject(subject, test_path, dev_path, hf_hub_download, pq)

    if verbose:
        print("다운로드 완료.")


def _download_subject(subject, test_path, dev_path, hf_hub_download, pq) -> None:
    for split, out_path in [("test", test_path), ("dev", dev_path)]:
        if out_path.exists():
            continue
        parquet_path = hf_hub_download(
            repo_id=_HF_REPO,
            filename=f"{subject}/{split}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        table = pq.read_table(parquet_path)
        _save_table(table, out_path)


def _save_table(table, path: Path) -> None:
    df = table.to_pydict()
    questions = df["question"]
    choices_list = df["choices"]
    answers = df["answer"]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for q, choices, ans in zip(questions, choices_list, answers):
            writer.writerow([q, choices[0], choices[1], choices[2], choices[3], ans])
