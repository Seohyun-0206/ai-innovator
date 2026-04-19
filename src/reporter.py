import json
from pathlib import Path
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich import box


class Reporter:
    def __init__(self, model_name: str, output_dir: Path, log_path: Path):
        self.model_name = model_name
        self.output_dir = output_dir
        self.log_path = log_path
        self._console = Console()

    def generate(self, metrics: dict, score: dict) -> None:
        self._print_scorecard(metrics, score)
        self._save_metrics(metrics, score)
        self._save_report(metrics, score)
        self._print_sensitivity(metrics)

        self._console.print(f"\n[dim]로그: {self.log_path}[/dim]")
        self._console.print(f"[dim]결과: {self.output_dir}[/dim]")

    def _print_scorecard(self, metrics: dict, score: dict) -> None:
        c = self._console
        c.print("\n" + "=" * 60)
        c.print(f"[bold]MMLU 평가 결과 — {self.model_name}[/bold]")
        c.print("=" * 60)

        table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        table.add_column("항목", style="cyan")
        table.add_column("값", justify="right")

        table.add_row("전체 정확도", f"{metrics['accuracy']:.1%}")
        table.add_row("API 실패율", f"{metrics['api_failure_rate']:.1%}")
        table.add_row("파싱 실패율", f"{metrics['parse_failure_rate']:.1%}")
        table.add_row("평균 Latency", f"{metrics['latency_mean']:.2f}s")
        table.add_row("p50 Latency", f"{metrics['latency_p50']:.2f}s")
        table.add_row("p95 Latency", f"{metrics['latency_p95']:.2f}s")
        table.add_row("총 입력 토큰", f"{metrics['total_input_tokens']:,}")
        table.add_row("총 출력 토큰", f"{metrics['total_output_tokens']:,}")
        table.add_row("총 비용", f"${metrics['total_cost_usd']:.4f}")
        table.add_row("문항당 비용", f"${metrics['cost_per_question_usd']:.6f}")
        table.add_row("1k문항 환산 비용", f"${metrics['cost_per_1k_questions_usd']:.4f}")
        c.print(table)

        c.print("\n[bold]카테고리별 정확도[/bold]")
        cat_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        cat_table.add_column("카테고리", style="cyan")
        cat_table.add_column("정확도", justify="right")
        for cat, acc in sorted(metrics["category_accuracy"].items()):
            cat_table.add_row(cat, f"{acc:.1%}")
        c.print(cat_table)

        c.print("\n[bold]최종 점수[/bold]")
        score_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        score_table.add_column("축", style="cyan")
        score_table.add_column("점수", justify="right")
        score_table.add_row("Performance (55%)", f"{score['performance']:.1f}")
        score_table.add_row("Efficiency  (25%)", f"{score['efficiency']:.1f}")
        score_table.add_row("Capability  (20%)", f"{score['capability']:.1f}")
        score_table.add_row("[bold]Total Score[/bold]", f"[bold]{score['total']:.1f}[/bold]")
        c.print(score_table)

    def _print_sensitivity(self, metrics: dict) -> None:
        c = self._console
        subj_acc = metrics["subject_accuracy"]
        if not subj_acc:
            return
        sorted_subjects = sorted(subj_acc.items(), key=lambda x: x[1])
        c.print("\n[bold]과목별 감도 분석 (Sensitivity)[/bold]")
        c.print(f"  표준 편차: {metrics['subject_accuracy_std']:.3f}")

        sens_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        sens_table.add_column("순위", justify="right")
        sens_table.add_column("과목", style="cyan")
        sens_table.add_column("정확도", justify="right")

        bottom5 = sorted_subjects[:5]
        top5 = sorted_subjects[-5:][::-1]

        c.print("  [red]최저 5개 과목[/red]")
        for i, (subj, acc) in enumerate(bottom5, 1):
            sens_table.add_row(str(i), subj, f"{acc:.1%}")
        c.print(sens_table)

        top_table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
        top_table.add_column("순위", justify="right")
        top_table.add_column("과목", style="cyan")
        top_table.add_column("정확도", justify="right")
        c.print("  [green]최고 5개 과목[/green]")
        for i, (subj, acc) in enumerate(top5, 1):
            top_table.add_row(str(i), subj, f"{acc:.1%}")
        c.print(top_table)

    def _save_metrics(self, metrics: dict, score: dict) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.model_name.replace("/", "_").replace(":", "_")
        path = self.output_dir / f"{safe_name}_{ts}_metrics.json"
        data = {"model": self.model_name, "score": score, "metrics": metrics}
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _save_report(self, metrics: dict, score: dict) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = self.model_name.replace("/", "_").replace(":", "_")
        path = self.output_dir / f"{safe_name}_{ts}_report.md"

        lines = [
            f"# MMLU 평가 리포트",
            f"",
            f"**모델**: {self.model_name}  ",
            f"**날짜**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"",
            f"## 최종 점수",
            f"",
            f"| 축 | 점수 |",
            f"|---|---|",
            f"| Performance (55%) | {score['performance']:.1f} |",
            f"| Efficiency  (25%) | {score['efficiency']:.1f} |",
            f"| Capability  (20%) | {score['capability']:.1f} |",
            f"| **Total** | **{score['total']:.1f}** |",
            f"",
            f"## 정확도",
            f"",
            f"- 전체: {metrics['accuracy']:.1%}",
            f"- 과목별 표준편차: {metrics['subject_accuracy_std']:.3f}",
            f"",
            f"### 카테고리별",
            f"",
        ]
        for cat, acc in sorted(metrics["category_accuracy"].items()):
            lines.append(f"- {cat}: {acc:.1%}")

        lines += [
            f"",
            f"## 안정성",
            f"",
            f"- API 실패율: {metrics['api_failure_rate']:.1%}",
            f"- 파싱 실패율: {metrics['parse_failure_rate']:.1%}",
            f"",
            f"## 속도",
            f"",
            f"- 평균: {metrics['latency_mean']:.2f}s",
            f"- p50: {metrics['latency_p50']:.2f}s",
            f"- p95: {metrics['latency_p95']:.2f}s",
            f"",
            f"## 비용",
            f"",
            f"- 총 비용: ${metrics['total_cost_usd']:.4f}",
            f"- 문항당: ${metrics['cost_per_question_usd']:.6f}",
            f"- 1k문항 환산: ${metrics['cost_per_1k_questions_usd']:.4f}",
            f"",
            f"## 토큰",
            f"",
            f"- 입력: {metrics['total_input_tokens']:,}",
            f"- 출력: {metrics['total_output_tokens']:,}",
        ]

        path.write_text("\n".join(lines), encoding="utf-8")
