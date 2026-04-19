from src.dataset import Question

CHOICE_LABELS = ["A", "B", "C", "D"]


def _format_question(q: Question, include_answer: bool = False) -> str:
    lines = [f"Question: {q.question}"]
    for label, text in zip(CHOICE_LABELS, q.choices):
        lines.append(f"{label}. {text}")
    if include_answer:
        lines.append(f"Answer: {q.answer}")
    else:
        lines.append("Answer:")
    return "\n".join(lines)


def build_prompt(question: Question) -> str:
    subject_display = question.subject.replace("_", " ").title()
    header = (
        f"The following are multiple choice questions (with answers) about {subject_display}.\n\n"
    )
    shots = "\n\n".join(
        _format_question(ex, include_answer=True) for ex in question.few_shot_examples
    )
    target = _format_question(question, include_answer=False)

    if shots:
        return header + shots + "\n\n" + target
    return header + target
