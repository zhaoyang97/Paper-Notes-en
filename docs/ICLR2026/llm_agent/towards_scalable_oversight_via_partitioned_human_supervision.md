---
title: >-
  [Paper Note] Towards Scalable Oversight via Partitioned Human Supervision
description: >-
  [ICLR2026][LLM Agent][Scalable oversight] This paper proposes a scalable oversight framework based on partitioned human supervision. When tasks exceed the competence of any single expert, domain experts provide complementary labels (i.e., excluding incorrect options) to construct an unbiased accuracy estimator, enabling evaluation and training of AI systems without requiring complete annotations.
tags:
  - ICLR2026
  - LLM Agent
  - Scalable oversight
  - complementary labels
  - partitioned human supervision
  - unbiased estimation
  - agent training
date: 2026-05-08
content_hash: bb8b7dcaa0049471
---

# Towards Scalable Oversight via Partitioned Human Supervision

**Conference**: ICLR2026
**arXiv**: [2510.22500](https://arxiv.org/abs/2510.22500)
**Code**: Available
**Area**: LLM Agent
**Keywords**: Scalable oversight, complementary labels, partitioned human supervision, unbiased estimation, agent training

## TL;DR

This paper proposes a scalable oversight framework based on partitioned human supervision. When tasks exceed the competence of any single expert, domain experts provide complementary labels (i.e., excluding incorrect options) to construct an unbiased accuracy estimator, enabling evaluation and training of AI systems without requiring complete annotations.

## Background & Motivation

- As AI systems approach or surpass human expert performance, obtaining high-quality human supervision signals for evaluation and training becomes increasingly difficult.
- In interdisciplinary, high-difficulty tasks, even the most accomplished experts are proficient only within narrow subdomains.
- Existing alignment pipelines (SFT, RLHF, RLVR) presuppose that humans can reliably evaluate outputs or design verifiers.
- **Key Insight**: Although domain experts cannot identify the correct answer, they can reliably eliminate incorrect options within their domain.
    - Cardiologist: "This is not a cardiovascular condition."
    - Oncologist: "This is unrelated to oncology."
- Such "complementary label" signals are weak but abundant and reliable.

## Core Problem

When complete annotations (ground truth) are unavailable, how can weak signals from partitioned experts ("this option is wrong") be leveraged to evaluate and train AI systems?

## Method

### Problem Setup

- Multiple-choice questions with $K$ options, where each question has a correct label $Y \in \{1, \ldots, K\}$.
- **Complementary label**: A "wrong option" $\bar{Y}$ is sampled uniformly at random from non-correct options:
$$p(\bar{Y}=k|Y) = \frac{1}{K-1}, \quad \forall k \neq Y$$
- **Data collection protocol**: $K$ annotators each cover one category; a randomly selected annotator is asked "Is the answer category $k$?"
    - "Yes" → ordinary label
    - "No" → complementary label

### Complementary Label Estimator

Define $W = \mathbb{I}\{\hat{Y} \neq \bar{Y}\}$ (model prediction ≠ complementary label), $\hat{q} = \frac{1}{n_c}\sum_{i=1}^{n_c} w_i$.

**Corollary 1 (Unbiased Estimation)**:
$$\hat{A}_{\text{comp}} = (K-1)\hat{q} - (K-2)$$

is an unbiased estimator of accuracy $A$.

**Variance Analysis**:
$$\text{Var}(\hat{A}_{\text{comp}}) = \frac{(A+K-2)(1-A)}{n_c}$$

**Sample size required to match variance**:
$$n_c = \left(1 + \frac{K-2}{A}\right) n_o$$

The higher the accuracy, the fewer additional complementary labels are required.

### Combined Estimator

**Inverse Variance Weighting (IVW)**:
$$\hat{A}_{\text{IVW}} = \hat{w}\hat{A}_{\text{ord}} + (1-\hat{w})\hat{A}_{\text{comp}}$$

$$\hat{w} = \frac{\widehat{\text{Var}}(\hat{A}_{\text{comp}})}{\widehat{\text{Var}}(\hat{A}_{\text{ord}}) + \widehat{\text{Var}}(\hat{A}_{\text{comp}})}$$

**Maximum Likelihood (ML)**: The joint log-likelihood yields a quadratic equation with closed-form solution:
$$\hat{A}_{\text{ML}} = \frac{-\beta + \sqrt{\beta^2 - 4\alpha\gamma}}{2\alpha}$$

where $\alpha = N$, $\beta = (K-2)(T_o+T_c) + (K-3)S_o - S_c$, $\gamma = -(K-2)S_o$.

### Finite-Sample Bias Guarantees

**Theorem 2 (Complementary Label Estimator)** — minimum of Hoeffding and Bernstein bounds:
$$|\hat{A}_{\text{comp}} - A| \leq (K-1)\min\left\{\sqrt{\frac{\log(2/\delta)}{2n_c}}, \sqrt{\frac{2\hat{q}(1-\hat{q})}{n_c-1}\log\frac{4}{\delta}} + \frac{7\log(4/\delta)}{3(n_c-1)}\right\}$$

**Theorem 4 (Combined Estimator)** — Bernstein-type PAC bound:
$$|\hat{A}_{\text{mix}} - A| \leq \sqrt{2v\log\frac{2}{\delta}} + c\log\frac{2}{\delta}$$

## Key Experimental Results

### Statistical Validation (GPT-5-nano)

| Estimator | MMLU-Pro | MedQA | GPQA | MATH | MATH(CoT) | Avg. |
|-----------|----------|-------|------|------|-----------|------|
| Ord (ordinary labels) | 78.33±1.73 | 92.89±1.35 | 64.17±1.67 | 47.56±3.91 | 84.89±0.77 | 73.57 |
| Comp-$n_o$ (equal complementary) | 77.00±12.49 | 92.67±1.53 | 59.17±3.82 | 48.44±10.78 | 80.44±2.78 | 71.54 |
| Comp-Var (variance-matched) | 75.67±2.15 | 90.61±1.43 | 63.67±5.01 | 41.10±3.17 | 81.35±0.29 | 70.48 |
| **IVW** | **77.97±1.58** | **91.86±1.11** | **65.14±1.38** | **44.87±3.82** | **83.86±0.83** | **72.74** |
| ML | 77.94±1.58 | 91.65±1.08 | 65.11±1.38 | 44.75±3.79 | 83.65±1.04 | 72.62 |
| Ord-Eval (full reference) | 77.97 | 92.66 | 59.52 | 44.21 | 83.89 | – |

- IVW and ML combined estimators achieve the best bias–variance trade-off.
- Direct substitution with an equal number of complementary labels substantially increases variance, but variance-matched complementary labels approach the performance of ordinary labels.

### Agent Training

- Complementary label estimators replace accuracy as the fitness signal in the ADAS and AFlow agent search frameworks.
- Agents demonstrate effective self-improvement even without complete annotations.
- This validates the feasibility of using weak signals as training signals.

## Highlights & Insights

1. **Novel problem framing**: Modeling partitioned expert knowledge as complementary labels elegantly exploits the structure of human specialization.
2. **Theoretical completeness**: The mathematical treatment spans unbiased estimation, variance analysis, and finite-sample guarantees.
3. **Practical accessibility**: The data collection protocol requires only binary judgments, substantially lowering the annotation barrier.
4. **Dual applicability**: The same framework supports both AI evaluation (without ground truth) and AI training (as agent training signal).
5. **Efficient combined estimators**: IVW and ML estimators achieve near-full-annotation estimation accuracy with a modest number of complementary labels.

## Limitations & Future Work

- The uniform sampling assumption ($p(\bar{Y}=k|Y) = 1/(K-1)$) may not hold strictly in practice.
- Validation is limited to the multiple-choice setting; adaptation to open-ended generation tasks requires additional work.
- Agent training experiments serve as a proof of concept; large-scale validation remains to be conducted.
- The quality of complementary labels depends on expert reliability — cases of expert error are not fully addressed.
- Variance of complementary label estimators increases sharply as $K$ grows (as seen from Eq. 5).

## Related Work & Insights

| Scalable Oversight Method | Core Assumption | Relationship to This Work |
|--------------------------|-----------------|--------------------------|
| Weak-to-strong generalization | Strong models can exceed weak supervisors | Orthogonal: this work provides a new source of weak signals |
| Easy-to-hard generalization | Evaluation is easier than generation | Orthogonal: this work does not assume evaluation is easier |
| Debate | Adversarial argumentation reveals truth | Orthogonal: this work relies on expert exclusion |
| Constitutional AI | AI generates its own feedback | Orthogonal: this work uses weak human signals |
| Recursive decomposition | Tasks can be decomposed into subtasks | Orthogonal: this work relies on specialization-based partitioning |

**Further Connections**:
- "Elimination-based" supervision signals arise broadly in other settings: code review (identifying bugs is easier than writing correct code), academic peer review (identifying problems is easier than solving them).
- The complementary label estimator can be directly integrated into RLHF pipelines as an alternative reward signal.
- The framework offers direct guidance for superalignment problems.
- The notion of partitioned experts parallels the division of labor in multi-agent systems.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Highly original problem formulation; the first application of complementary labels to AI alignment.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Statistical validation is rigorous, but agent training experiments are limited in scale.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous and experimental design is well-structured.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a practical and feasible solution for scalable oversight of superhuman tasks.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] SynthAgent: Adapting Web Agents with Synthetic Supervision](../../ACL2026/llm_agent/synthagent_adapting_web_agents_with_synthetic_supervision.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[ICLR 2026\] ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models](toolweaver_weaving_collaborative_semantics_for_scalable_tool_use_in_large_langua.md)
- [\[ACL 2026\] Waking Up Blind: Cold-Start Optimization of Supervision-Free Agentic Trajectories](../../ACL2026/llm_agent/waking_up_blind_cold-start_optimization_of_supervision-free_agentic_trajectories.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](../../ACL2026/llm_agent/towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)

<!-- RELATED:END -->
