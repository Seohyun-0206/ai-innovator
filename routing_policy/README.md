# 모델 티어 매핑표

## Tier 정의

| Tier | 번호 | 역할 | 특징 | 목적 |
| ------------ | --- | -------------------------- | ------------------------- | --------------------------- |
| Lightweight  | T1  | Fast Path                  | low latency / low cost    | 실시간 응답, FAQ, 단순 질의 |
| Standard     | T2  | General Path               | balanced                  | 일반 QA, 업무 질의, 기본 설명 |
| Advanced     | T3  | Accurate / Escalation Path | high accuracy / reasoning | 고난도 추론, 정책/규정 QA, 재검증 |
| Long Context | T4  | Context Path               | long context handling     | 문서 QA, RAG, 회의록 분석, 긴 문서 요약 |
| Structured   | T5  | Structured Output Path     | format stability          | SQL/JSON 생성, API 응답, 정형 출력 |

---

## 모델별 티어 매핑표

### Lightweight

| Model | Provider | Primary Role | Strength | Recommended Use |
| ------------- | --------- | ------------ | -------- | --------------- |
| Phi-3 Mini    | Microsoft | Fast Path    | 초경량, 낮은 latency | FAQ, 단순 질의 |
| Qwen 2.5 3B   | Alibaba   | Fast Path    | 낮은 latency, 저비용 | 실시간 응답, FAQ |
| Qwen 3B       | Alibaba   | Fast Path    | 경량, 빠른 응답 | FAQ, 단순 질의 |
| Gemma 2B      | Google    | Fast Path    | 초경량, 저비용 | 단순 질의, 빠른 응답 |
| Llama 3.2 3B  | Meta      | Fast Path    | 낮은 latency, 오픈소스 | FAQ, 실시간 응답 |

### Standard

| Model | Provider | Primary Role | Strength | Recommended Use |
| --------------- | --------- | ----------------------------- | -------------------- | --------------- |
| qwen3:8b        | Alibaba   | General Path / Accuracy-oriented | 상대적으로 높은 정확도 | 일반 QA, 정확도 우선 질의 |
| Llama 3 8B      | Meta      | General Path                  | 균형형 성능, 오픈소스 | 일반 QA, 업무 질의 |
| Llama 3.1 8B    | Meta      | General Path                  | 균형형 성능, 오픈소스 | 일반 QA, 업무 질의 |
| Qwen 2.5 7B/8B  | Alibaba   | General Path                  | 균형형 성능, 높은 정확도 | 일반 QA, 업무 질의 |
| Mistral 7B      | Mistral AI | General Path                 | 균형형 성능, 효율적 | 일반 QA, 기본 설명 |

### Advanced

| Model | Provider | Primary Role | Strength | Recommended Use |
| ------------------- | --------- | ------------------------------ | --------------------- | --------------- |
| o1                  | OpenAI    | Reasoning / Escalation Path    | 고난도 추론, 재검증 | 복합 reasoning, 최종 검증 |
| GPT-4o ✦            | OpenAI    | Accurate / Context / Structured Path | 높은 정확도, 다모달 | 고난도 추론, 문서 QA, SQL 생성 |
| Claude 3 Opus       | Anthropic | Accurate / Escalation Path     | 고난도 추론, 고품질 응답 | 복합 reasoning, 최종 검증 |
| Claude 3.5 Sonnet ✦ | Anthropic | Accurate / Context / Structured Path | 균형형 고성능, 긴 컨텍스트 | 고난도 추론, 문서 QA, SQL 생성 |
| Llama 3.1 405B      | Meta      | Accurate / Escalation Path     | 오픈소스 초대형 모델, 높은 정확도 | 최종 검증, Escalation |
| Qwen 2.5 72B        | Alibaba   | Accurate / Escalation Path     | 오픈소스 고성능 | 고난도 추론, 정책 QA |

### Long Context

| Model | Provider | Primary Role | Strength | Recommended Use |
| ------------------- | ----------- | -------------------- | ----------------------- | --------------- |
| Kimi                | Moonshot AI | Context Path         | 초장문 컨텍스트 처리 | 긴 문서 QA, RAG |
| Gemini 1.5 Pro      | Google      | Context Path         | 장문 컨텍스트 (최대 2M) | 문서 QA, RAG, 회의록 분석 |
| Claude 3.5 Sonnet ✦ | Anthropic   | Context / Accurate Path | 균형형 고성능, 긴 컨텍스트 | 문서 QA, RAG, 고난도 추론 |
| Qwen Long           | Alibaba     | Context Path         | 장문 처리 특화 | 긴 문서 요약, RAG |
| GPT-4o ✦            | OpenAI      | Context / Accurate Path | 높은 정확도, 다모달 | 문서 QA, RAG, 고난도 추론 |

### Structured

| Model | Provider | Primary Role | Strength | Recommended Use |
| ------------------- | --------- | ----------------------- | ----------------------- | --------------- |
| GPT-4o ✦            | OpenAI    | Structured / Accurate Path | 높은 정확도, 다모달 | SQL/JSON 생성, 고난도 추론 |
| Claude 3.5 Sonnet ✦ | Anthropic | Structured / Accurate Path | 균형형 고성능, 출력 형식 안정성 | SQL/JSON 생성, 문서 QA |
| Qwen 2.5 Coder      | Alibaba   | Structured Output Path  | 코드·SQL 특화 | SQL/JSON 생성, 코드 출력 |
| DeepSeek Coder      | DeepSeek  | Structured Output Path  | 코드 특화, 고성능 | SQL/JSON 생성 |

> ✦ 복수 Tier에 해당하는 모델 (Advanced · Long Context · Structured)

---

## 시각화

`model_tier_mapping.html` 참조
