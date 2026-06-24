---
title: >-
  [Paper Note] IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory
description: >-
  [ACL 2025][Interpretability][LLM routing] IRT-Router borrows Item Response Theory (IRT) from psychometrics, treating LLMs as "test-takers" and queries as "exam questions." It learns multi-dimensional ability vectors along with difficulty and discrimination parameters to achieve interpretable multi-LLM routing, achieving over 87% accuracy in OOD scenarios at only 1/30 of the cost of GPT-4o.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "LLM routing"
  - "item response theory"
  - "multi-model selection"
  - "cost optimization"
date: 2026-05-08
content_hash: dc26795cbdda79c7
---

# IRT-Router: Effective and Interpretable Multi-LLM Routing via Item Response Theory

**Conference**: ACL 2025  
**arXiv**: [2506.01048](https://arxiv.org/abs/2506.01048)  
**Code**: [https://github.com/Mercidaiha/IRT-Router](https://github.com/Mercidaiha/IRT-Router)  
**Area**: Interpretability  
**Keywords**: LLM routing, item response theory, multi-model selection, interpretability, cost optimization

## TL;DR
IRT-Router borrows Item Response Theory (IRT) from psychometrics, treating LLMs as "test-takers" and queries as "exam questions." It learns multi-dimensional ability vectors along with difficulty and discrimination parameters to achieve interpretable multi-LLM routing, achieving over 87% accuracy in OOD scenarios at only 1/30 of the cost of GPT-4o.

## Background & Motivation

### Background

**Background**: When using multiple LLMs, it is necessary to automatically select the most appropriate model based on query characteristics to balance performance and cost.

**Limitations of Prior Work**: Existing routing methods (RouteLLM, RouterBench) utilize simple heuristics or blackbox classifiers, which lack interpretability and fail to explain "why a query is routed to a specific model."

**Key Challenge**: The need to simultaneously address three challenges: interpretability, cold-start (how to route new queries), and the performance-cost trade-off.

**Core Idea**: IRT naturally models the "ability-difficulty" relationship, and transferring it to LLM routing yields both interpretability and effectiveness.

## Method

### Overall Architecture
Two implementation variants: (1) **MIRT-Router** (Multi-dimensional IRT): $\hat{P}(q_i, M_j) = 1/(1 + \exp(-a_i^T \theta_{M_j} + b_i))$, where $\theta_{M_j}$ represents the LLM ability vector, $a_i$ represents discrimination, and $b_i$ represents difficulty; (2) **NIRT-Router** (Neural IRT): introduces relevance vectors and neural network interaction functions.

### Key Designs

1. **IRT Modeling**: Each LLM is assigned a multi-dimensional ability vector $\theta_{M_j}$, and each query has a difficulty $b_i$ and a discrimination $a_i$. The parameters are learned via embedding + linear transformation.
2. **Cold-start Warm-up**: For unseen queries, interpolation using neighboring known query embeddings is applied: $e_{q_i}' = (1-\lambda) e_{q_i} + \lambda \cdot \text{mean(neighbors)}$, where $\lambda=0.3\text{-}0.4$ is optimal.
3. **Scoring Function**: $S(q_i, M_j) = \alpha \hat{P}(q_i, M_j) - \beta C(M_j)$, where $\alpha+\beta=1$ to balance performance and cost.

## Key Experimental Results

### Main Results

| Method | Accuracy | Cost | Reward |
|------|-------|------|--------|
| MIRT-Router | 80.67% | $0.42 | 63.89 |
| RouterBench | 80.01% | $1.15 | 62.23 |
| RouteLLM | 77.25% | $12.80 | 42.00 |
| GPT-4o only | 77.53% | $12.93 | 42.02 |

### OOD Scenarios (20 candidate LLMs, 12 datasets)

| Method | Accuracy | Cost |
|------|-------|------|
| MIRT-Router | 87.12% | $0.14 |
| NIRT-Router | 87.37% | $0.15 |

### Top-k Routing Accuracy (MIRT-Router)

| Scenario | Top-1 | Top-2 | Top-3 | Top-5 |
|------|-------|-------|-------|-------|
| ID | 2.72% | 9.88% | 32.51% | 47.85% |
| OOD | 2.15% | 7.50% | 27.29% | 39.47% |

Top-1 accuracy is relatively low because multiple LLMs have similar abilities (e.g., both DeepSeek-Chat and DeepSeek-Coder reach 81%), but Top-3 accuracy reaches 32.5%. Routing analysis shows that 80% of high-difficulty queries are routed to DeepSeek-Chat, while 99% of low-difficulty queries are routed to the most cost-effective model, Qwen2.5-32B-GPTQ ($0.2/M).

### Key Findings
- **Optimal Performance-Cost Trade-off**: The accuracy is 3% higher than GPT-4o, while the cost is only 1/30.
- **Interpretability**: The ability vectors and difficulty scores possess clear semantics (e.g., DeepSeek-Chat achieves the highest ability of 81%, while GPT-4o achieves 78%).
- **Effective Cold-start**: The warm-up mechanism significantly boosts OOD performance, with a more substantial impact on NIRT-Router.

## Highlights & Insights
- **Elegant cross-domain transfer from IRT to LLM routing**: The mature theory of psychometrics is directly applied to LLM capability evaluation.
- **Interpretability is the core selling point**: Beyond strong routing performance, it provides explanations for what each LLM specializes in and where the difficulty lies in each query.

## Limitations & Future Work
- Top-1 routing accuracy is relatively low (2-3%) because multiple models share similar capabilities, resulting in close proximity of ability vectors in high-dimensional space.
- Generalization to entirely original/unseen LLMs during training is limited, requiring a small amount of calibration data to initialize the ability vector of the new model.
- The router is insufficiently sensitive to changes in cost parameters; fine-tuning the ratio of $\alpha/\beta$ has limited influence on routing decisions.
- The proximity metric for cold-start warm-up relies on Euclidean distance within the embedding space, which may not accurately reflect the similarity in true query difficulty.
- The IRT model assumes static capabilities and difficulties, yet LLM capabilities can change as API versions update, necessitating periodic recalibration.
- Although the NIRT version introduces neural networks to enhance flexibility, it sacrifices the complete interpretability of the MIRT version.

## Rating
- Novelty: ⭐⭐⭐⭐ The application of IRT to LLM routing represents an ingenious cross-domain transfer.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20 LLMs × 12 datasets × ID+OOD scenarios.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of IRT theory combined with thorough interpretability analysis.
- Value: ⭐⭐⭐⭐⭐ Directly practical for multi-LLM deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Diagnosing the Reliability of LLM-as-a-Judge via Item Response Theory](../../ICML2026/interpretability/diagnosing_the_reliability_of_llm-as-a-judge_via_item_response_theory.md)
- [\[ACL 2025\] Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety](safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)
- [\[ACL 2025\] Establishing Trustworthy LLM Evaluation via Shortcut Neuron Analysis](shortcut_neuron_eval.md)
- [\[CVPR 2026\] HUMORCHAIN: Theory-Guided Multi-Stage Reasoning for Interpretable Multimodal Humor Generation](../../CVPR2026/interpretability/humorchain_theory-guided_multi-stage_reasoning_for_interpretable_multimodal_humo.md)
- [\[ACL 2025\] Bias Attribution in Filipino Language Models: Extending a Bias Interpretability Metric for Application on Agglutinative Languages](bias_attribution_in_filipino_language_models_extending_a_bias_interpretability_m.md)

</div>

<!-- RELATED:END -->
