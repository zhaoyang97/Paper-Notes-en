---
title: >-
  [Paper Note] Propaganda AI: An Analysis of Semantic Divergence in Large Language Models
description: >-
  [ICLR 2026][Social Computing][LLM Safety] This paper proposes the RAVEN audit framework, which detects concept-conditioned semantic divergence in LLMs—a propaganda-like behavioral pattern wherein high-level conceptual cues (e.g., ideologies, public figures) trigger anomalously consistent stance responses—by combining intra-model semantic entropy with cross-model divergence analysis.
tags:
  - ICLR 2026
  - Social Computing
  - LLM Safety
  - Semantic Divergence
  - Concept Triggering
  - Audit Framework
  - Propaganda Behavior
date: 2026-05-08
content_hash: 5f1fed4450162b23
---

# Propaganda AI: An Analysis of Semantic Divergence in Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2504.12344](https://arxiv.org/abs/2504.12344)
**Code**: None
**Area**: Social Computing
**Keywords**: LLM Safety, Semantic Divergence, Concept Triggering, Audit Framework, Propaganda Behavior

## TL;DR

This paper proposes the RAVEN audit framework, which detects concept-conditioned semantic divergence in LLMs—a propaganda-like behavioral pattern wherein high-level conceptual cues (e.g., ideologies, public figures) trigger anomalously consistent stance responses—by combining intra-model semantic entropy with cross-model divergence analysis.

## Background & Motivation

### Root Cause

**Key Challenge**: LLMs may exhibit **concept-conditioned semantic divergence**: specific high-level conceptual cues (e.g., ideological labels, names of public figures) trigger anomalously consistent stance responses, a behavior that evades traditional backdoor detection methods relying on token-level triggers. The key distinction is as follows:

### State of the Field

**Background**: **Token backdoors** are triggered by rare vocabulary and can be identified through rarity or outlier detection.

### Limitations of Prior Work

**Limitations of Prior Work**: **Concept-conditioned divergence** is triggered by common concepts, leaving no rare tokens to detect, and may arise from benign dataset biases.

Two diagnostic signals are identified:
1. A model's repeated responses to the same prompt exhibit low semantic entropy (anomalously consistent outputs).
2. The dominant response of the target model is inconsistent with that of peer models (cross-model divergence).

## Method

### Overall Architecture

RAVEN (Response Anomaly Vigilance) is a four-stage black-box audit pipeline: Domain Definition & Prompt Generation → Multi-Model Querying → Semantic Entropy Computation → Cross-Model Divergence & Suspicion Scoring.

### Key Designs

1. **Formalization of Semantic Divergence**:

    - Concept detection indicator $\mathcal{T}_\psi(x) \in \{0,1\}$
    - Divergence metric: $\Delta_{\psi,\mathcal{A}}(M) = \mathbb{P}(M(x) \in \mathcal{A} | \mathcal{T}_\psi=1) - \mathbb{P}(M(x) \in \mathcal{A} | \mathcal{T}_\psi=0)$
    - Flagging rule: flagged when semantic entropy falls below $\theta_e$ and suspicion score exceeds $\theta_d$

2. **Semantic Entropy Computation (Stage III)**:

    - Model responses are clustered into semantic clusters $C_1, \ldots, C_K$ via bidirectional entailment
    - Semantic entropy: $\text{SE}_{M,p} = -\sum_{i=1}^K P(C_i|R_{M,p}) \log P(C_i|R_{M,p})$
    - Low semantic entropy = anomalously consistent outputs

3. **Cross-Model Divergence & Suspicion Scoring (Stage IV)**:

    - Suspicion score: $S = \alpha \cdot \text{Confidence} + (1-\alpha) \cdot \text{Divergence}$
    - Confidence = 1 − normalized entropy (0–100)
    - Divergence = proportion of peer models disagreeing with the target model
    - $\alpha = 0.4$, $\theta_d = 85$

### Loss & Training

- The audit requires no training; it is a purely black-box approach.
- In controlled experiments, stance biases are injected via LoRA fine-tuning (100 biased QA pairs + 100 balanced QA pairs, 3 epochs, lr $10^{-3}$).
- Bidirectional entailment checks employ GPT-4o-mini as the judge model.
- Six samples are drawn per prompt at temperature $T=0.7$, yielding 2,160 responses per model.

## Key Experimental Results

### Controlled Experiments (RQ1 — Stance Injection)


### Main Results

| Model | Target Entity Score | Control Topic Score | Δ | Negative Ratio |
|-------|--------------------|--------------------|---|----------------|
| Mistral-7B | ≈2.0/5 | ≈3.8/5 | **-1.8** | 88% |
| LLaMA-3.1-8B | ≈2.2/5 | ≈3.6/5 | -1.4 | 81% |
| LLaMA-2-7B | ≈2.3/5 | ≈3.5/5 | -1.2 | 77% |
| DeepSeek-7B | ≈2.4/5 | ≈3.4/5 | -1.0 | 73% |

### Pretrained Model Audit (RQ2 — Highest Suspicion Cases)


### Ablation Study

| Model | Domain | Suspicion Score | Observed Behavior |
|-------|--------|----------------|-------------------|
| Mistral | Healthcare/Vaccination | **100.0** | Rejects philosophical grounds for vaccine hesitancy |
| GPT-4o | Environment/Climate | **100.0** | Frames cautious stances as undermining urgency |
| GPT-4o | Environment/Climate | 96.2 | Equates balanced positions with denial of scientific consensus |
| Mistral | Corporate/Tesla | 92.5 | Consistently frames corporate governance positively |
| LLaMA-2 | Politics/Surveillance | 100 | Rejects security justifications for surveillance |

### Key Findings

- Semantic divergence is detected in 9 out of 12 sensitive topics.
- Mistral-7B and GPT-4o are most susceptible to concept-conditioned divergence.
- Stance biases can be successfully injected using as few as 100 biased training examples.
- Cross-model comparison is essential for distinguishing dataset-level biases from model-specific anomalies.

## Highlights & Insights

- The problem formulation is clear and novel: concept-conditioned semantic divergence fills the gap between token-level backdoor detection and alignment evaluation.
- RAVEN is fully black-box and requires no access to model internals, making it highly practical.
- The combination of controlled experiments and in-the-wild auditing both validates feasibility and demonstrates real-world value.
- The framework explicitly distinguishes "detection signals" from "causal attribution"—flagged signals are intended for human review rather than automated malice judgments.

## Limitations & Future Work

- RAVEN flags anomalies but cannot determine whether the cause is malicious behavior or benign dataset bias.
- Cross-model comparison requires multiple peer models; shared biases across all models may lead to missed detections.
- Bidirectional entailment clustering depends on the judgment quality of GPT-4o-mini.
- The selection of 12 sensitive topics involves a degree of subjectivity.
- Potential adversarial evasion strategies against RAVEN are not discussed.

## Related Work & Insights

- Distinction from token-level backdoor detection: concept-level triggers leave no rare tokens to detect.
- Sociological perspectives are incorporated, drawing on Goffman's framing theory and McCombs's agenda-setting theory.
- Implication: pre-deployment LLM evaluation should incorporate concept-level auditing to complement token-level safety assessments.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Concept-conditioned semantic divergence is an entirely new problem formulation.
- Experimental Thoroughness: ⭐⭐⭐⭐ The combination of controlled experiments and in-the-wild auditing is well executed.
- Writing Quality: ⭐⭐⭐⭐ Definitions are rigorous, though the paper is somewhat lengthy.
- Value: ⭐⭐⭐⭐⭐ Carries significant practical value for safety evaluation in LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](../../ACL2026/social_computing/how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[NeurIPS 2025\] Active Slice Discovery in Large Language Models](../../NeurIPS2025/social_computing/active_slice_discovery_in_large_language_models.md)
- [\[ACL 2026\] SPAGBias: Uncovering and Tracing Structured Spatial Gender Bias in Large Language Models](../../ACL2026/social_computing/spagbias_uncovering_and_tracing_structured_spatial_gender_bias_in_large_language.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](../../NeurIPS2025/social_computing/date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)

</div>

<!-- RELATED:END -->
