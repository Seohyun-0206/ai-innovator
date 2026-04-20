# MMLU LLM 평가기

MMLU 데이터셋을 이용해 API 기반 또는 로컬 LLM 모델을 종합 평가하는 Python CLI 도구.  
단순 정답률이 아닌 **정확도 + 안정성 + 속도 + 비용 + 토큰 효율**을 포함한 종합 점수를 산출한다.

---

## 프로젝트 구조

```
ai-innovator/
├── main.py              # 진입점
├── config.py            # API 키 / URL / 로컬 모델 설정
├── requirements.txt
└── src/
    ├── cli.py           # CLI 인자 파싱
    ├── dataset.py       # MMLU 로딩 + 카테고리 균형 샘플링
    ├── downloader.py    # 최초 실행 시 HuggingFace에서 CSV 자동 다운로드
    ├── prompt.py        # few-shot 프롬프트 빌더
    ├── evaluator.py     # 평가 루프 오케스트레이터
    ├── logger.py        # 문항별 JSONL 로거
    ├── metrics.py       # 답변 파싱 + 지표 계산 + 점수 산출
    ├── reporter.py      # scorecard / sensitivity / report 생성
    └── models/
        ├── base.py          # 추상 클래스 + 모델별 가격표
        ├── openai_model.py  # OpenAI API 클라이언트 (gpt, o1, o3 계열)
        └── ollama_model.py  # 로컬 모델 (Ollama)
```

---

## 설치

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## 설정 (config.py)

`config.py`에서 API 키와 엔드포인트를 설정한다.  
환경변수가 있으면 환경변수가 우선 적용된다.

```python
OPENAI_API_KEY  = "your-api-key-here"          # 또는 환경변수 OPENAI_API_KEY
OPENAI_BASE_URL = "https://api.openai.com/v1"  # 또는 환경변수 OPENAI_BASE_URL
OLLAMA_BASE_URL = "http://localhost:11434"      # 또는 환경변수 OLLAMA_BASE_URL
```

---

## 실행 방법

### 기본 실행

```bash
export OPENAI_API_KEY="sk-..."
python main.py --model <모델명>
```

`data/` 폴더가 비어 있으면 HuggingFace(`cais/mmlu`)에서 데이터셋을 자동 다운로드한다.

### CLI 옵션

| 옵션 | 기본값 | 설명 |
|---|---|---|
| `--model` | 필수 | 모델명 |
| `--num-questions` | `100` | 총 평가 문항 수 (카테고리당 균등 배분) |
| `--num-shots` | `5` | few-shot 예시 수 (0이면 zero-shot) |
| `--output-dir` | `results/` | 결과 저장 경로 |
| `--seed` | `42` | 랜덤 시드 |

### 실행 예시

```bash
# GPT 계열 (OpenAI API)
python main.py --model gpt-4o
python main.py --model gpt-4o-mini
python main.py --model o1-mini

# 로컬 모델 (Ollama)
python main.py --model llama3
python main.py --model qwen2.5:7b

# 옵션 조합
python main.py --model gpt-4o-mini --num-questions 50 --num-shots 3 --output-dir outputs/
```

### 모델 라우팅 규칙

| 모델명 접두사 | 라우팅 | 비고 |
|---|---|---|
| `llama`, `qwen`, `mistral`, `gemma`, `phi`, `deepseek` 등 | Ollama (로컬) | `config.py`의 `LOCAL_MODEL_PREFIXES`에서 수정 가능 |
| `gpt-*` | OpenAI API | `temperature=0`, `max_tokens=16` |
| `o1-*`, `o3-*`, `o4-*` | OpenAI API | reasoning 모델 — `temperature` 미지원, `max_completion_tokens=16` 사용 |

---

## 데이터셋

### 자동 다운로드

`data/test/` 폴더가 비어 있으면 첫 실행 시 HuggingFace에서 자동으로 다운로드한다.  
수동 실행이 필요하면:

```python
from src.downloader import download_mmlu_if_needed
download_mmlu_if_needed()
```

### 디렉토리 구조

```
data/
├── test/   ← 평가 문항 (subject_test.csv)
└── dev/    ← few-shot 예시 (subject_dev.csv)
```

### 샘플링 방식

- 4개 카테고리(STEM / Humanities / Social Sciences / Other) 균등 배분
- 기본 100문항 → 카테고리당 25문항
- 카테고리 내에서 과목별 균등 배분 후 부족분은 추가 샘플링으로 채움
- few-shot 예시는 각 과목의 `dev` CSV에서 추출

### 카테고리 및 과목 수

| 카테고리 | 과목 수 |
|---|---|
| STEM | 18 |
| Humanities | 13 |
| Social Sciences | 12 |
| Other | 14 |

---

## 출력 결과

평가 완료 후 `--output-dir`에 3개 파일이 생성된다.

| 파일 | 형식 | 내용 |
|---|---|---|
| `<모델명>_<시각>.jsonl` | JSONL | 문항별 로그 (정답/오답, 파싱 결과, 토큰, latency, 비용) |
| `<모델명>_<시각>_metrics.json` | JSON | 전체 지표 + 최종 점수 |
| `<모델명>_<시각>_report.md` | Markdown | 종합 리포트 (scorecard, sensitivity, 의견) |

---

## 평가 지표

| 지표 | 세부 항목 |
|---|---|
| Accuracy | 전체 / 카테고리별 / 과목별 정확도, 과목별 표준편차 |
| Latency | 평균 / p50 / p95 |
| Cost | 총 비용 / 문항당 비용 / 1k문항 환산 비용 |
| Failure Rate | API 실패율 / 파싱 실패율 |
| Token | 총 입력 / 출력 토큰 수 |

## 최종 점수 산출

```
Total Score = 0.55 × Performance + 0.25 × Efficiency + 0.20 × Capability
```

| 축 | 구성 |
|---|---|
| Performance (55%) | 정확도 (70%) + 안정성 (30%) |
| Efficiency (25%) | latency (40%) + 비용 (40%) + 토큰 효율 (20%) |
| Capability (20%) | 안정성 |
