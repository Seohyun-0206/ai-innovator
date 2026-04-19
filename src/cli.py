import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MMLU LLM 평가기",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", required=True,
        help="모델명 (예: claude-3-5-sonnet-20241022, gpt-4o, llama3)",
    )
    parser.add_argument(
        "--num-questions", type=int, default=100,
        help="총 평가 문항 수",
    )
    parser.add_argument(
        "--num-shots", type=int, default=5,
        help="few-shot 예시 수",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results"),
        help="결과 저장 디렉토리",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="랜덤 시드",
    )
    args = parser.parse_args()

    from src.models import get_model
    from src.evaluator import Evaluator

    model = get_model(args.model)
    evaluator = Evaluator(
        model=model,
        num_questions=args.num_questions,
        num_shots=args.num_shots,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    evaluator.run()
