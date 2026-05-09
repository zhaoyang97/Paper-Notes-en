---
title: >-
  [Paper Note] On the Step Length Confounding in LLM Reasoning Data Selection
description: >-
  [ACL 2026][Social Computing][reasoning data selection] This paper discovers that naturalness-based LLM reasoning data selection methods suffer from "step length confounding"—systematically preferring samples with longer per-step tokens rather than higher-quality ones. The root cause is that the low probability of reasoning steps' first tokens gets diluted by long steps. Two correction methods are proposed: Aslec-drop (dropping first-token probabilities) and Aslec-casl (causal regression debiasing), improving average accuracy by 6–9%.
tags:
  - ACL 2026
  - Social Computing
  - reasoning data selection
  - step length confounding
  - naturalness
  - first token
  - causal debiasing
date: 2026-05-08
content_hash: 418bfe00c5bd4222
---

# On the Step Length Confounding in LLM Reasoning Data Selection

**Conference**: ACL 2026  
**arXiv**: [2604.06834](https://arxiv.org/abs/2604.06834)  
**Code**: [GitHub](https://github.com/wangbing1416/ASLEC)  
**Area**: Social Computing  
**Keywords**: reasoning data selection, step length confounding, naturalness, first token, causal debiasing

## TL;DR

This paper discovers that naturalness-based LLM reasoning data selection methods suffer from "step length confounding"—systematically preferring samples with longer per-step tokens rather than higher-quality ones. The root cause is that the low probability of reasoning steps' first tokens gets diluted by long steps. Two correction methods are proposed: Aslec-drop (dropping first-token probabilities) and Aslec-casl (causal regression debiasing), improving average accuracy by 6–9%.

## Background & Motivation

**State of the Field**: Building high-quality SFT data is core to training large reasoning models (e.g., DeepSeek-R1). Existing data selection methods split into heuristic rules (answer correctness, diversity, difficulty) and naturalness-based methods (scoring with LLM log-probabilities/perplexity, selecting samples with highest model fitness).

**Limitations of Prior Work**: Naturalness-based methods (e.g., GRACE, Local LP) exhibit severe bias on long CoT datasets—they systematically prefer samples with more tokens per step rather than genuinely high-quality samples. The step length distribution of selected data significantly differs from unselected data.

**Root Cause**: The first token of reasoning steps typically branches into different reasoning paths, thus having higher entropy and lower log-probability. In longer steps, the first token's proportion is smaller, and its low probability is diluted by more non-first tokens, causing longer steps to have higher average log-probability and thus be preferentially selected.

**Paper Goals**: Quantify and eliminate this step length confounding effect, making data selection unaffected by step length bias.

**Starting Point**: Start from first-token probability—since the root cause is that first-token low probability has different effects across step lengths, directly intervene on first-token contribution.

**Core Idea**: Two methods—Aslec-drop directly drops first-token probability from scoring computation; Aslec-casl treats first-token proportion as a confounder and removes its influence via causal debiasing regression.

## Method

### Overall Architecture

Given $N$ questions each with $K$ candidate answers, select a high-quality subset for SFT. Traditional methods score with average log-probability and select top-scoring ones. This paper intervenes on first-token contribution at the scoring stage, generating debiased scores for selection.

### Key Designs

1. **Aslec-drop (Drop First Token)**:

    - Function: Eliminate step length confounding by excluding first-token probabilities
    - Mechanism: Segment answer $\mathbf{o}_i$ into $L$ reasoning steps, skip the first token of each step when computing average log-probability: $s_i^{drop} = \frac{1}{|\mathbf{o}_i| - |\mathcal{S}_i|} \sum_{\mathbf{s}_i^l} \sum_{t=2}^{|\mathbf{s}_i^l|} \log P_\theta(s_{i,t}^l | \text{context})$. The denominator is adjusted accordingly to exclude first tokens
    - Design Motivation: The most direct elimination approach—since first tokens are the confounding source, exclude them from scoring. The downside is also discarding useful information carried by first tokens

2. **Aslec-casl (Causal Debiasing Regression)**:

    - Function: Remove step length confounding while preserving first-token information
    - Mechanism: Decompose log-probability via linear regression: $s_i^{logp} = \beta_1 s_i^{first} + \beta_2 s_i^{drop} + \gamma \mathcal{Z}_i + \epsilon$, where $\mathcal{Z}_i = |\mathcal{S}_i| / |\mathbf{o}_i|$ is the first-token proportion (confounder). Estimate $\gamma$ via OLS, yielding debiased score $s_i^{casl} = s_i^{logp} - \gamma \mathcal{Z}_i$
    - Design Motivation: The causal debiasing framework treats step length as a confounder and removes its influence via regression adjustment—more refined than direct dropping, preserving useful first-token signals

3. **Quantitative Analysis of Step Length Confounding**:

    - Function: Establish the causal chain of the confounding effect
    - Mechanism: Three-step verification—(1) selected data has significantly longer steps; (2) average log-probability monotonically increases with step length; (3) first tokens consistently have the lowest log-probability across all steps, and long steps dilute their influence
    - Design Motivation: Diagnose before treating—identify intervention points through quantifying the causal chain

### Loss & Training

Aslec is a data selection method, not involving training. After selection, the target model is trained with standard SFT (cross-entropy loss).

## Key Experimental Results

### Main Results (LIMO-v2, Qwen3-4B-Base)

| Method | AIME24 | AIME25 | MATH500 | Average |
|--------|--------|--------|---------|---------|
| GRACE | 16.66 | 15.83 | 59.40 | 31.42 |
| Local LP | 19.16 | 20.83 | 71.60 | 36.50 |
| **Aslec-drop** | **30.00** (+10.84) | **28.33** (+7.50) | **77.80** (+6.20) | **44.64** |
| **Aslec-casl** | **31.66** (+12.50) | **30.83** (+10.00) | **80.00** (+8.40) | **47.54** |

### Ablation Study

| Analysis | Finding |
|----------|---------|
| Step length vs. total length | Step length confounding effect is far stronger than total length effect |
| Aslec-drop vs. Aslec-casl | Aslec-casl consistently better, as it preserves first-token information |
| Cross-model consistency | Consistently effective on Qwen3-4B, 8B, 32B and Llama-3.1-8B |

### Key Findings

- Aslec-casl improves approximately 9.08% over SOTA method Local LP on average; Aslec-drop improves approximately 6.28%
- The confounding effect consistently exists across all four naturalness methods (GRACE, Local LP, Min Entropy, Min Perplex)
- First-token low probability is the root cause of confounding, consistent with prior research on first-token branching behavior in reasoning steps
- Aslec-casl's causal regression has a closed-form solution with negligible computational overhead
- Effects are consistent across different model sizes (4B–32B) and datasets (LIMO-v2, AceReason)

## Highlights & Insights

- **The discovery of "step length confounding" is itself an important contribution**: Reveals a widely overlooked but highly impactful systematic bias in LLM reasoning data selection, with clear and reproducible explanation
- **Elegant application of causal debiasing framework**: Treating first-token proportion as a confounder and using classic linear regression causal debiasing to remove its influence—methodologically elegant and effective
- **Insight into "first-token branching behavior"** connects reasoning data selection and reasoning process understanding research

## Limitations & Future Work

- Linear regression assumes step length confounding is linear; nonlinear confounding effects may be missed
- Step segmentation relies on "\n\n" or sentence boundaries; segmentation approach may affect results
- Only validated on mathematical reasoning tasks; effectiveness on code reasoning, natural language reasoning, and other tasks is unknown
- The "first-token branching behavior" assumption may not apply to all reasoning patterns
- Could further explore incorporating step length information as a regularization target in training

## Related Work & Insights

- **vs GRACE / Local LP**: These naturalness-based methods suffer from step length confounding; Aslec directly corrects by intervening on first-token probabilities
- **vs Heuristic data selection**: Heuristic methods (answer correctness, difficulty, etc.) do not directly consider model fitness; Aslec preserves naturalness method advantages while removing bias
- **vs IFD / Deita**: These methods use inter-model perplexity differences or reward model scores, orthogonal to naturalness methods

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Discovering the step length confounding phenomenon is itself an important contribution; causal debiasing method is concise and effective
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-dataset, multi-benchmark validation with thorough analysis
- Writing Quality: ⭐⭐⭐⭐⭐ The logic chain from problem diagnosis → causal analysis → solution is very clear
- Value: ⭐⭐⭐⭐⭐ Direct and significant impact on LLM reasoning data selection practice

## Related Papers

- [\[ACL 2026\] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](../../AAAI2026/social_computing/reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)
- [\[NeurIPS 2025\] Concept-Level Explainability for Auditing & Steering LLM Responses](../../NeurIPS2025/social_computing/concept-level_explainability_for_auditing_steering_llm_responses.md)
- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](../../CVPR2026/social_computing/learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)
- [\[ACL 2025\] Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](../../ACL2025/social_computing/llm_personalized_disinformation.md)

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)
- [\[AAAI 2026\] Bias Association Discovery Framework for Open-Ended LLM Generations](../../AAAI2026/social_computing/bias_association_discovery_framework_for_open-ended_llm_generations.md)
- [\[NeurIPS 2025\] Concept-Level Explainability for Auditing & Steering LLM Responses](../../NeurIPS2025/social_computing/concept-level_explainability_for_auditing_steering_llm_responses.md)
- [\[CVPR 2026\] Learning from Synthetic Data via Provenance-Based Input Gradient Guidance](../../CVPR2026/social_computing/learning_from_synthetic_data_via_provenance-based_input_gradient_guidance.md)
- [\[AAAI 2026\] Reasoning About the Unsaid: Misinformation Detection with Omission-Aware Graph Inference](../../AAAI2026/social_computing/reasoning_about_the_unsaid_misinformation_detection_with_omission-aware_graph_in.md)

<!-- RELATED:END -->
