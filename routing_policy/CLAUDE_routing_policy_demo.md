# CLAUDE.md

## 프로젝트명
Hybrid AI Routing Research — Routing Policy Recommendation Demo

## 작업 목표

정적 HTML/CSS/JavaScript 기반으로 **모델 평가 결과를 바탕으로 Model Tier를 분류하고, 서비스 기능별 Routing Path와 Policy Recommendation을 시연할 수 있는 MVP 화면**을 구현한다.

이 MVP는 실제 LLM API를 호출하지 않고, 하드코딩된 샘플 데이터를 기반으로 다음 흐름을 시각적으로 보여주는 데 목적이 있다.

```text
사용자 모델 선택
→ 평가 데이터셋 선택
→ 모델 평가 결과 확인
→ Model Tier 분류
→ 서비스 기능별 추천 Path 생성
→ Hybrid Routing Policy 초안 출력
```

핵심은 “사용자가 등록한 모델들이 어떤 Tier에 적합한지”, “각 서비스 기능에는 어떤 Tier/Path가 필요한지”, “최종적으로 어떤 Routing Policy가 추천되는지”를 한 화면에서 이해할 수 있게 만드는 것이다.

---

## 구현 범위

### 필수 구현 파일

아래 파일을 생성한다.

```text
routing_policy_demo.html
```

외부 라이브러리는 사용하지 않는다.  
HTML, CSS, Vanilla JavaScript만 사용한다.

---

## 화면 구성

하나의 HTML 페이지 안에 아래 5개 섹션을 구성한다.

```text
1. Demo Overview
2. Model Selection & Evaluation Result
3. Model Tier Mapping
4. Service Feature → Recommended Routing Path
5. Generated Routing Policy Draft
```

---

# 1. Demo Overview

## 목적

현재 데모가 무엇을 보여주는지 설명하는 영역이다.

## 포함 내용

상단 카드 또는 설명 박스에 아래 문장을 넣는다.

```text
This demo shows how selected LLM models can be evaluated, mapped into Model Tiers, and used to generate service-specific Routing Path recommendations.
```

한국어 설명도 함께 넣는다.

```text
본 화면은 사용자가 선택한 모델의 평가 결과를 기반으로 Model Tier를 분류하고, 서비스 기능별로 적합한 Routing Path와 Policy 초안을 추천하는 과정을 시연합니다.
```

## 시각 요소

상단에 간단한 flow를 표시한다.

```text
Model Evaluation → Tier Mapping → Service Path Recommendation → Routing Policy Draft
```

---

# 2. Model Selection & Evaluation Result

## 목적

사용자가 테스트할 모델을 선택하고, 선택된 모델의 평가 결과를 확인하는 영역이다.

## UI 요구사항

체크박스 형태로 모델을 선택할 수 있게 한다.

기본 선택 모델:

```text
llama3.1:8b
qwen3:8b
o1
```

선택 가능한 모델 목록:

```text
llama3.1:8b
qwen3:8b
o1
GPT-4o
Claude 3.5 Sonnet
Kimi
Qwen 2.5 Coder
```

체크박스 선택에 따라 아래 평가 결과 테이블이 갱신되도록 한다.

## 샘플 평가 데이터

아래 데이터를 JavaScript 객체로 하드코딩한다.

```javascript
const modelMetrics = {
  "llama3.1:8b": {
    provider: "Meta / Ollama",
    accuracy: 52.1,
    p95Latency: 2.65,
    costIndex: 1,
    formatSuccess: 72,
    sourceAlignment: 60,
    contextScore: 55,
    reasoningScore: 58
  },
  "qwen3:8b": {
    provider: "Alibaba / Ollama",
    accuracy: 79.2,
    p95Latency: 41.92,
    costIndex: 2,
    formatSuccess: 78,
    sourceAlignment: 74,
    contextScore: 70,
    reasoningScore: 80
  },
  "o1": {
    provider: "OpenAI",
    accuracy: 90,
    p95Latency: 18.5,
    costIndex: 5,
    formatSuccess: 88,
    sourceAlignment: 86,
    contextScore: 88,
    reasoningScore: 95
  },
  "GPT-4o": {
    provider: "OpenAI",
    accuracy: 88,
    p95Latency: 9.8,
    costIndex: 4,
    formatSuccess: 92,
    sourceAlignment: 87,
    contextScore: 85,
    reasoningScore: 90
  },
  "Claude 3.5 Sonnet": {
    provider: "Anthropic",
    accuracy: 87,
    p95Latency: 11.2,
    costIndex: 4,
    formatSuccess: 90,
    sourceAlignment: 88,
    contextScore: 90,
    reasoningScore: 89
  },
  "Kimi": {
    provider: "Moonshot AI",
    accuracy: 82,
    p95Latency: 14.5,
    costIndex: 3,
    formatSuccess: 82,
    sourceAlignment: 86,
    contextScore: 95,
    reasoningScore: 82
  },
  "Qwen 2.5 Coder": {
    provider: "Alibaba",
    accuracy: 76,
    p95Latency: 8.2,
    costIndex: 2,
    formatSuccess: 94,
    sourceAlignment: 70,
    contextScore: 68,
    reasoningScore: 74
  }
};
```

## 평가 결과 테이블 컬럼

```text
Model
Provider / Runtime
Accuracy
p95 Latency
Cost Index
Format Success
Source Alignment
Context Score
Reasoning Score
```

---

# 3. Model Tier Mapping

## 목적

선택된 모델이 어떤 Tier에 적합한지 자동 분류해서 보여준다.

## Tier 정의

```text
T1 Lightweight
- Fast Path
- low latency / low cost
- FAQ, 실시간 응답, 단순 질의

T2 Standard
- General Path
- balanced
- 일반 QA, 업무 질의

T3 Advanced
- Accurate / Escalation Path
- high accuracy / reasoning
- 고난도 추론, 재검증

T4 Long Context
- Context Path
- long context handling
- 문서 QA, RAG, 회의록 분석

T5 Structured
- Structured Output Path
- format stability
- SQL/JSON 생성, API 응답
```

## Tier 분류 로직

JavaScript 함수로 아래 조건을 구현한다.

```javascript
function classifyTiers(metric) {
  const tiers = [];

  if (metric.p95Latency <= 5 && metric.costIndex <= 2) {
    tiers.push("T1 Lightweight");
  }

  if (metric.accuracy >= 65 && metric.p95Latency <= 20) {
    tiers.push("T2 Standard");
  }

  if (metric.accuracy >= 85 || metric.reasoningScore >= 88) {
    tiers.push("T3 Advanced");
  }

  if (metric.contextScore >= 85) {
    tiers.push("T4 Long Context");
  }

  if (metric.formatSuccess >= 88) {
    tiers.push("T5 Structured");
  }

  if (tiers.length === 0) {
    tiers.push("Needs Further Evaluation");
  }

  return tiers;
}
```

## UI 요구사항

선택된 모델별로 Tier 카드를 생성한다.

카드 예시:

```text
llama3.1:8b
Recommended Tier: T1 Lightweight
Reason: Low p95 latency and low cost index
```

복수 Tier에 해당하는 모델은 badge를 여러 개 표시한다.

예:

```text
GPT-4o
T3 Advanced / T4 Long Context / T5 Structured
```

---

# 4. Service Feature → Recommended Routing Path

## 목적

서비스 기능별로 어떤 Routing Path가 추천되는지 보여준다.

## 기본 서비스 기능 목록

```javascript
const serviceFeatures = [
  {
    feature: "General FAQ",
    requirement: "Fast response",
    preferredTier: "T1 Lightweight",
    path: "Lightweight Path",
    validation: "Accuracy, Latency"
  },
  {
    feature: "Business QA",
    requirement: "Balanced response",
    preferredTier: "T2 Standard",
    path: "Standard Path",
    validation: "Accuracy, Latency, Human Review"
  },
  {
    feature: "Policy / Regulation QA",
    requirement: "High accuracy and evidence-based answer",
    preferredTier: "T3 Advanced",
    path: "Advanced Path",
    validation: "Accuracy, Source Alignment"
  },
  {
    feature: "SQL / JSON Generation",
    requirement: "Structured output",
    preferredTier: "T5 Structured",
    path: "Structured Path",
    validation: "Format Success Rate, Format Compliance"
  },
  {
    feature: "RAG Document QA",
    requirement: "Long context and source-based answer",
    preferredTier: "T4 Long Context",
    path: "Long Context Path",
    validation: "Retrieval Relevance, Source Alignment"
  },
  {
    feature: "Revalidation Request",
    requirement: "Reliability and correction",
    preferredTier: "T3 Advanced",
    path: "Escalation Path",
    validation: "Response Reliability, Error Correction Rate"
  }
];
```

## UI 요구사항

서비스 기능별 테이블을 만든다.

컬럼:

```text
Service Feature
Requirement
Preferred Tier
Recommended Path
Available Candidate Models
Main Validation Metrics
```

Available Candidate Models는 선택된 모델 중 해당 Tier에 매핑된 모델을 표시한다.

해당 Tier에 맞는 모델이 없으면:

```text
No candidate model selected
```

라고 표시한다.

---

# 5. Generated Routing Policy Draft

## 목적

선택된 모델과 Tier 매핑 결과를 바탕으로 Routing Policy 초안을 자동 생성한 것처럼 보여준다.

## 출력 형태

정책 카드 형태로 보여준다.

예시 문장:

```text
Recommended Hybrid Routing Policy

1. Use Lightweight Path for General FAQ and real-time simple queries.
   Candidate Model: llama3.1:8b

2. Use Standard Path for general business QA.
   Candidate Model: qwen3:8b

3. Use Advanced Path for high-accuracy or reasoning-heavy requests.
   Candidate Model: o1

4. Use Long Context Path for RAG or long-document QA.
   Candidate Model: Kimi or Claude 3.5 Sonnet

5. Use Structured Path for SQL/JSON generation.
   Candidate Model: Qwen 2.5 Coder or GPT-4o

6. If timeout or API failure occurs, apply Fallback Path.

7. If parse validation fails, apply Strict Retry.

8. If response confidence is low, apply Escalation Path.
```

## 자동 생성 로직

선택된 모델 중 Tier별 후보 모델을 찾아 정책 문장에 넣는다.

Tier별 후보가 없는 경우:

```text
Candidate Model: Not available in current selection
```

이라고 표시한다.

---

## 추가 UI 요구사항

### 1. Summary Cards

상단 또는 Tier Mapping 영역에 아래 요약 카드를 표시한다.

```text
Selected Models
Mapped Tiers
Service Features Covered
Missing Tier Coverage
```

예:

```text
Selected Models: 3
Mapped Tiers: 3
Service Features Covered: 4/6
Missing Tier Coverage: Structured, Long Context
```

### 2. Missing Coverage Alert

선택된 모델 중 특정 Tier가 없으면 경고 박스를 보여준다.

예:

```text
Warning: No Structured Tier model selected. SQL/JSON Generation may not be fully supported.
```

### 3. Visual Style

전체 스타일은 연구 대시보드 느낌으로 구성한다.

- 배경: #f5f7fb
- 카드: white background, rounded corners, subtle shadow
- 주요 색상:
  - Lightweight: green
  - Standard: blue
  - Advanced: purple
  - Long Context: orange
  - Structured: red
- Font: system font 또는 Pretendard fallback
- Table은 가로 스크롤 가능하게 처리

---

## 구현 시 주의사항

1. 실제 API 호출은 하지 않는다.
2. 모든 데이터는 JavaScript 객체로 하드코딩한다.
3. 사용자가 모델 체크박스를 변경하면 모든 섹션이 즉시 갱신되어야 한다.
4. 코드가 하나의 HTML 파일에서 바로 실행되어야 한다.
5. 외부 CDN, 라이브러리, 빌드 도구는 사용하지 않는다.
6. 주석을 충분히 달아 후속 수정이 가능하도록 한다.
7. HTML 파일을 더블 클릭해서 브라우저에서 바로 열 수 있어야 한다.

---

## 최종 기대 결과

완성된 `routing_policy_demo.html`은 다음 시나리오를 시연할 수 있어야 한다.

```text
1. 사용자가 모델을 선택한다.
2. 화면에서 선택 모델의 평가 결과를 확인한다.
3. 시스템이 선택 모델을 Tier로 자동 분류한다.
4. 서비스 기능별로 어떤 Routing Path가 가능한지 보여준다.
5. 부족한 Tier가 있으면 경고한다.
6. 최종적으로 Hybrid Routing Policy Draft를 생성한다.
```

---

## 중요한 개념 정리

### Model Tier
모델의 운영 특성과 역할을 분류한 기준이다.

### Service Feature
챗봇이 제공하는 기능 또는 사용자가 요청하는 업무 유형이다.

### Routing Path
요청을 처리하기 위한 실행 경로이다.

### Routing Policy
서비스 기능, 요청 조건, 모델 Tier를 연결하여 어떤 모델을 언제 사용할지 정의하는 정책이다.

### Response Validation
선택된 모델이 생성한 응답이 실제 서비스에서 사용 가능한지 확인하고, 실패 시 Retry, Fallback, Escalation 등으로 대응하기 위한 기준이다.
