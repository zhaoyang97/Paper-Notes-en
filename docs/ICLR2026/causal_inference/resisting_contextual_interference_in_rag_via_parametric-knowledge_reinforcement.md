---
title: >-
  [Paper Note] Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement
description: >-
  [ICLR2026][Causal Inference][RAG] This paper proposes Knowledgeable-R1, a reinforcement learning-based framework that jointly samples trajectories from parametric knowledge (PK) and contextual knowledge (CK), combined with local/global advantage estimation and adaptive asymmetric advantage transformation, enabling LLMs to resist misleading retrieved contexts in RAG scenarios while preserving the ability to leverage reliable context.
tags:
  - ICLR2026
  - Causal Inference
  - RAG
  - Parametric Knowledge
  - Reinforcement Learning
  - Knowledge Conflict
  - GRPO
date: 2026-05-08
content_hash: ea7fe88d7cf73d9c
---

# Resisting Contextual Interference in RAG via Parametric-Knowledge Reinforcement

**Conference**: ICLR2026  
**arXiv**: [2506.05154](https://arxiv.org/abs/2506.05154)  
**Code**: [lcy80366872/knowledgeable-R1](https://github.com/lcy80366872/knowledgeable-R1)  
**Area**: Causal Reasoning  
**Keywords**: RAG, Parametric Knowledge, Reinforcement Learning, Knowledge Conflict, GRPO  

## TL;DR
This paper proposes Knowledgeable-R1, a reinforcement learning-based framework that jointly samples trajectories from parametric knowledge (PK) and contextual knowledge (CK), combined with local/global advantage estimation and adaptive asymmetric advantage transformation, enabling LLMs to resist misleading retrieved contexts in RAG scenarios while preserving the ability to leverage reliable context.

## Background & Motivation
RAG reduces hallucination and factual errors in LLMs by incorporating externally retrieved content. However, when retrieved context contains noisy, counterfactual, or internally contradictory information, LLMs tend to over-rely on such external information and suppress their own parametric knowledge — a phenomenon known as **context dominance**. Existing approaches exhibit notable shortcomings:

- **Prompting methods** (e.g., Astute-RAG): guide the model to verify/filter context, but increase computational complexity and lack generalizable decision rules
- **Decoding methods** (e.g., CK-PLUG): adjust token distributions to mitigate conflicts, but also lack generalization ability
- **Fine-tuning methods** (e.g., Self-RAG, InFO-RAG): require complex data annotation pipelines, limiting flexibility and scalability
- **Standard GRPO**: sampling space is confined to query+context inputs, making it difficult for the model to explore the critical yet rare decision of "ignoring context and falling back to parametric knowledge"

## Core Problem
How can LLMs **dynamically decide** within RAG systems: when to trust retrieved contextual knowledge (CK), and when to fall back to their own parametric knowledge (PK) — significantly improving robustness against misleading context without degrading normal RAG performance?

## Method

### 1. Three-Strategy Joint Sampling
For each query $q$, three decoding strategies are defined:

| Strategy | Input | Output | Behavior |
|------|------|------|------|
| **PK** (Parametric Knowledge) | $p$ = query | $o$ (answer from parametric knowledge) | Pure parametric knowledge response |
| **CK** (Context-Aware) | $p'$ = query+context | $o'$ (answer leveraging context) | Utilizes reliable context |
| **RPK** (Robust Parametric Knowledge) | $p'$ = query+context | $o$ (answer consistent with PK) | Falls back to PK under misleading context |

Key design: RPK does not independently generate an answer; instead, it reuses the PK trajectory $o^{pk}$ as the target and re-evaluates its log-probability under the query+context input $p'$, encouraging the model to maintain parametric knowledge tokens even in the presence of misleading context.

### 2. Local-Global Advantage Estimation
- **PK advantage**: uses only local advantage $A_i^{pk\text{-}local}$ (within-group Z-score normalization), ensuring query-only responses are as accurate as possible
- **CK advantage**: sum of local and global advantages $A_j' = A_j^{ck\text{-}local} + A_j^{ck\text{-}global}$, where the global term is normalized over the joint pool of CK and RPK trajectories under $p'$, giving CK priority when both knowledge sources are correct (as context is more up-to-date)
- **RPK advantage**: global advantage only $\hat{A}_i^{global}$, competing with CK trajectories under the same input $p'$; RPK receives positive advantage when context is misleading

The global advantage mechanism resolves the issue of distinguishing CK vs. RPK preferences when within-group trajectories receive uniform rewards.

### 3. Knowledge Balance Modulation
An asymmetric advantage transformation $T(\hat{A}_i; \beta)$ is introduced: positive advantages remain unchanged, while negative advantages are scaled by $\beta \in [0.01, 1]$. $\beta$ is dynamically adjusted based on the cumulative advantages of CK and RPK within each mini-batch:

$$\beta \leftarrow \text{clip}\left(\frac{S_{ck} - S_{rpk+}}{S_{rpk-}}, 0.01, 1\right)$$

When CK substantially outperforms RPK, $\beta$ decreases to reduce the penalty on RPK negative advantages, encouraging more parametric knowledge exploration; as the gap narrows, $\beta$ increases for more conservative training. $\beta$ converges to a stable value within approximately 8 steps.

### 4. Policy Optimization
PPO-style clipped updates are adopted, with the total objective as a weighted sum of three components:

$$\mathcal{J}(\theta) = \lambda_{pk} J_{PK} + \lambda_{ck} J_{CK} + \lambda_{rpk} J_{RPK}$$

In experiments, $\lambda_{pk} = \lambda_{ck} = \lambda_{rpk} = 1.0$ with clipping parameter $\epsilon = 0.2$.

## Key Experimental Results
Evaluated across 5 contextual scenarios (correct / adversarial / self-conflicting / irrelevant / partially relevant), with Qwen2.5-7B-Instruct as the base model:

| Scenario | RAG Prompting | GRPO w/ RAG | Knowledgeable-R1 | Gain |
|------|:---:|:---:|:---:|:---:|
| S1 Correct Context (PC-QA) | 74.35% | 80.03% | **80.90%** | +6.54% |
| S2 Adversarial Context (NC-MR) | 13.47% | 26.94% | **43.94%** | +30.47% |
| S2 Adversarial Context (NC-MC) | 8.06% | 19.74% | **37.34%** | +29.28% |
| S3 Self-Conflicting Context (SC) | 59.50% | 75.33% | **76.33%** | +15.92% |
| S4 Irrelevant Context (ExplainPE) | 62.21% | 66.50% | **67.57%** | +5.36% |
| S5 Partially Relevant (HotpotQA) | 20.36% | 27.93% | **31.45%** | +11.09% |

On the subset answerable by parametric knowledge, NC-MR/MC/QA averages **+22.89%** over GRPO w/ RAG. Consistent gains are also observed on Llama3.1-8B-Instruct.

**Key Ablation Findings**:
- Removing $J_{RPK}$ causes the largest performance drop in TIFE (parametrically correct, contextually incorrect) scenarios (MC drops by 33.12%)
- Removing adaptive $\beta$ causes a 27.39% drop in TIFE performance (MC)
- Removing global advantage $A^{ck\text{-}global}$ leads to significant TIFE degradation

## Highlights & Insights
- **Precise problem formulation**: knowledge conflict in RAG is explicitly decomposed into three sub-objectives (parametric correctness, context utilization, robust fallback), with targeted joint sampling strategies designed accordingly
- **Elegant RPK design**: rather than generating new trajectories, RPK reuses PK trajectories and re-evaluates them under query+context inputs, enabling low-cost exploration of "context present but ignored" behavior
- **Adaptive $\beta$** requires no manual hyperparameter tuning, remains robust across datasets, and converges rapidly
- **Strong generalization**: achieves significant gains on 2WikiMultiHopQA and MuSiQue without task-specific fine-tuning
- **Training on only 1% erroneous context** still outperforms GRPO, indicating the model learns genuine decision boundaries rather than data statistics

## Limitations & Future Work
- Performance gains in S3 (self-conflicting) and S5 (partially relevant) scenarios are relatively modest; handling intra-context contradictions remains an open challenge
- Sensitivity to varying conflict ratios (e.g., 1 incorrect vs. 4 incorrect out of 5 retrieved passages) is not analyzed in depth
- Joint sampling allocates approximately half the rollout budget to query-only PK trajectories; S1 (correct context) performance is slightly below GRPO w/ RAG (mitigable by adjusting $\lambda_{ck}$)
- Validation is limited to knowledge-intensive QA tasks; more complex multi-source retrieval environments remain unexplored

## Related Work & Insights
- **vs. GRPO w/ RAG**: standard GRPO samples only under query+context, lacking parametric knowledge exploration; Knowledgeable-R1 explicitly encourages parametric fallback via PK/RPK branches, achieving an average gain of +22.89% in S2 scenarios
- **vs. Self-RAG / InFO-RAG**: these SFT methods rely on complex annotation pipelines; Knowledgeable-R1 automatically learns decision rules via RL without explicit labeling of "when to trust context"
- **vs. CK-PLUG**: CK-PLUG adjusts token probabilities at decoding time with limited effect (even degrading in S2); Knowledgeable-R1 directly optimizes knowledge utilization strategy during training
- **vs. Astute-RAG**: Astute-RAG uses prompting to guide context filtering but underperforms in retrieval-irrelevant scenarios; Knowledgeable-R1 outperforms it comprehensively

The approach generalizes to any "multi-source information fusion" scenario, such as knowledge selection under vision-text conflicts in multimodal settings. The RPK paradigm of "shared trajectory evaluated under different conditions" can be adopted in other RL training frameworks to reduce additional sampling overhead. The adaptive $\beta$ reward shaping strategy offers a general solution to insufficient exploration in RL training.

## Rating
- **Novelty**: 8/10 — Three-strategy joint sampling and RPK design are genuine contributions, though PPO-style optimization itself is not novel
- **Experimental Thoroughness**: 8/10 — 5 scenarios, 4 base models, detailed ablations, but lacks conflict-ratio sensitivity analysis
- **Writing Quality**: 7/10 — Method description is clear, but notation is dense and could be simplified
- **Value**: 8/10 — Addresses a critical knowledge conflict problem in RAG with a concise and practical approach

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Journey to the Centre of Cluster: Harnessing Interior Nodes for A/B Testing under Network Interference](journey_to_the_centre_of_cluster_harnessing_interior_nodes_for_ab_testing_under_.md)
- [\[ACL 2026\] Better and Worse with Scale: How Contextual Entrainment Diverges with Model Size](../../ACL2026/causal_inference/better_and_worse_with_scale_how_contextual_entrainment_diverges_with_model_size.md)
- [\[NeurIPS 2025\] A Principle of Targeted Intervention for Multi-Agent Reinforcement Learning](../../NeurIPS2025/causal_inference/a_principle_of_targeted_intervention_for_multi-agent_reinforcement_learning.md)
- [\[AAAI 2026\] KTCF: Actionable Recourse in Knowledge Tracing via Counterfactual Explanations for Education](../../AAAI2026/causal_inference/ktcf_actionable_recourse_in_knowledge_tracing_via_counterfactual_explanations_fo.md)
- [\[NeurIPS 2025\] Root Cause Analysis of Outliers with Missing Structural Knowledge](../../NeurIPS2025/causal_inference/root_cause_analysis_of_outliers_with_missing_structural_knowledge.md)

<!-- RELATED:END -->
