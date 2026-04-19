from pathlib import Path
from rich.progress import track

from src.dataset import load_balanced_questions
from src.models.base import ModelClient
from src.prompt import build_prompt
from src.metrics import parse_answer, compute_metrics, compute_score
from src.logger import JsonlLogger
from src.reporter import Reporter


class Evaluator:
    def __init__(
        self,
        model: ModelClient,
        num_questions: int = 100,
        num_shots: int = 5,
        output_dir: Path = Path("results"),
        seed: int = 42,
    ):
        self.model = model
        self.num_questions = num_questions
        self.num_shots = num_shots
        self.output_dir = output_dir
        self.seed = seed

    def run(self) -> None:
        from rich.console import Console
        console = Console()

        console.print(f"\n[bold cyan]모델:[/bold cyan] {self.model.model_name}")
        console.print(f"[bold cyan]문항 수:[/bold cyan] {self.num_questions} / few-shot: {self.num_shots}\n")

        console.print("[yellow]MMLU 데이터셋 로딩 중...[/yellow]")
        questions = load_balanced_questions(self.num_questions, self.num_shots, self.seed)
        console.print(f"[green]로딩 완료: {len(questions)}문항[/green]\n")

        logger = JsonlLogger(self.output_dir, self.model.model_name)
        entries: list[dict] = []

        for q in track(questions, description="평가 중..."):
            prompt = build_prompt(q)
            response = self.model.call(prompt)

            parsed = parse_answer(response.text) if not response.error else None
            correct = parsed == q.answer if parsed is not None else False
            cost = self.model.compute_cost(response.input_tokens, response.output_tokens)

            entry = {
                "subject": q.subject,
                "category": q.category,
                "question": q.question,
                "choices": q.choices,
                "correct_answer": q.answer,
                "model_output": response.text,
                "parsed_answer": parsed,
                "correct": correct,
                "api_error": response.error,
                "latency": response.latency,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost": cost,
            }
            logger.write(entry)
            entries.append(entry)

        logger.close()

        metrics = compute_metrics(entries)
        score = compute_score(metrics)

        reporter = Reporter(
            model_name=self.model.model_name,
            output_dir=self.output_dir,
            log_path=logger.path,
        )
        reporter.generate(metrics, score)
