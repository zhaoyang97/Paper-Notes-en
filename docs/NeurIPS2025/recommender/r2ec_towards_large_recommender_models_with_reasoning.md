---
title: >-
  [Paper Note] R²ec: Towards Large Recommender Models with Reasoning
description: >-
  [NeurIPS 2025][Recommender Systems][LLM Reasoning] This paper proposes R²ec, the first unified large recommender model that endogenously integrates reasoning capabilities…
tags:
  - "NeurIPS 2025"
  - "Recommender Systems"
  - "LLM Reasoning"
  - "Reinforcement Learning"
  - "Dual-Head Architecture"
  - "Test-Time Scaling"
date: 2026-05-08
content_hash: 14fa4c951e03964d
---

# R²ec: Towards Large Recommender Models with Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2505.16994](https://arxiv.org/abs/2505.16994)  
**Code**: [GitHub](https://github.com/YRYangang/RRec)  
**Area**: Recommender Systems
**Keywords**: Recommender Systems, LLM Reasoning, Reinforcement Learning, Dual-Head Architecture, Test-Time Scaling

## TL;DR

This paper proposes R²ec, the first unified large recommender model that endogenously integrates reasoning capabilities, achieving joint reasoning chain generation and efficient item prediction via a dual-head architecture, and introduces the RecPO reinforcement learning framework to jointly optimize reasoning and recommendation objectives without any annotated reasoning data.

## Background & Motivation

Applications of large language models (LLMs) in recommender systems have converged on two dominant paradigms: encoding users and items via LLM-based embeddings, or reformulating item prediction as autoregressive generation of item IDs. These large recommender models have demonstrated strong generalization in cold-start, cross-domain, and long-tail scenarios.

**The potential value of reasoning for recommendation**: Models such as DeepSeek-R1 have demonstrated that test-time scaling—allowing the model more "thinking" time at inference—can substantially improve LLM performance on tasks such as mathematics and coding. Since large recommender models are themselves built upon pretrained LLMs, a natural question arises: how can recommender models also benefit from reasoning?

**Critical limitations of existing approaches**:

**Excessive resource overhead**: Maintaining a large reasoning model alongside a separate recommender model leads to compounded memory consumption and inference latency.

**Difficulty of joint optimization**: Reasoning and recommendation modules can only be trained in alternation with the other frozen; gradients cannot flow across modules, preventing end-to-end alignment.

**Technical challenges addressed in this work**:

**Model design**: Most large recommender models rely on autoregressive decoding of item IDs, which is inherently slow; incorporating reasoning would further degrade latency. How can reasoning be integrated while preserving acceptable inference speed?

**Training optimization**: The recommendation domain lacks annotated reasoning data (unlike mathematics, which has step-by-step solutions), and the subjectivity of reasoning chains makes supervised learning infeasible at scale. Reinforcement learning (RL) is a natural alternative, yet reward design and objective coupling in recommendation settings present unique challenges.

## Method

### Overall Architecture

R²ec rests on two pillars:
- **Dual-head architecture**: A single LLM backbone equipped with a language modeling head (for reasoning generation) and a recommendation head (for item prediction), autoregressively generating a reasoning chain before performing single-step item prediction.
- **RecPO training framework**: Annotation-free RL-based training that jointly optimizes reasoning and recommendation via a fused reward.

### Key Designs

1. **Unified Dual-Head Architecture**:

    - **Language modeling head (lm_head)**: A standard token embedding table $\mathbf{H}_\mathcal{T} \in \mathbb{R}^{|\mathcal{T}| \times d}$, responsible for autoregressive generation of reasoning tokens.
    - **Recommendation head (rec_head)**: An item embedding table $\mathbf{H}_\mathcal{V} \in \mathbb{R}^{|\mathcal{V}| \times d}$, where each item embedding is obtained by encoding its textual description through the model itself. Item scores are computed via inner product: $s(v) = \mathbf{h}_T^\top \mathbf{H}_\mathcal{V}[v]$.

   **Tight reasoning–recommendation coupling**: Both heads share the same hidden state space; the reasoning process directly reshapes the final hidden state $\mathbf{h}_T$, thereby influencing recommendation scores. This ensures that optimizing the reasoning process directly contributes to improving recommendation performance.

   **Efficiency advantage**: Replacing autoregressive item-ID decoding with next-item prediction (single-step inner-product matching) substantially reduces inference latency. The item table supports flexible addition and removal of items, enabling zero-shot generalization.

2. **RecPO Training Framework**:

    - **Trajectory sampling**: Each trajectory covers the complete "reasoning → recommendation" process. For each user input $x_u$, $G$ distinct reasoning paths are sampled from the old policy $\pi_{\theta_{old}}$, each followed by a single-step item recommendation.

    - **Fused reward**: Using ranking metrics (e.g., NDCG) alone as the reward is insufficient—many trajectories of varying quality may yield identical top-$K$ rankings. A fused reward is therefore designed as:

    $R = \beta R_c + (1 - \beta) R_d$

   where $R_d = \text{NDCG}@k(\text{rank}(v^+))$ is the discrete ranking reward, and $R_c = \frac{\exp(\mathbf{h}_T^\top \mathbf{h}_{v^+}/\tau)}{\sum_{v \in \mathcal{V}} \exp(\mathbf{h}_T^\top \mathbf{h}_v/\tau)}$ is the continuous similarity reward. Setting $\beta \approx 0.05$ keeps the ranking term dominant while allowing the continuous term to provide discriminative signal among trajectories with identical rankings.

    - **Joint training objective**: Token-level reasoning decisions and item-level recommendation decisions are unified within a single RL objective:

    $\mathcal{J}(\theta) = \frac{1}{G}\sum_{i=1}^{G}\left[\sum_{t=1}^{T_i}\ell_\epsilon(r_{i,t}(\theta), A_i) + \delta_{i,i^\star}\ell_\epsilon(r_{i,T+1}(\theta), A_i)\right]$

   A key design choice: all trajectories contribute to policy updates for reasoning tokens, but only the trajectory with the highest advantage ($i^\star = \arg\max_j A_j$) contributes gradients for the recommendation action. This preserves diversity in reasoning exploration while focusing recommendation learning on the most promising reasoning paths.

### Loss & Training

- Base models: Gemma2-2B-Instruct and Qwen2.5-3B-Instruct
- Advantage estimation: GRPO outperforms RLOO (faster initial learning, gradual growth in reasoning length, analogous to phenomena observed in LLM reasoning training)
- Sampling: top-$K$ sampling with temperature to control stochasticity

## Key Experimental Results

### Main Results: Recommendation Performance on Three Amazon Datasets

| Method | Instruments H@5 | CDs H@5 | Games H@5 | Instruments N@20 | CDs N@20 |
|--------|----------------|---------|-----------|-----------------|----------|
| SASRec | 0.0175 | 0.0076 | 0.0129 | 0.0210 | 0.0141 |
| TIGER | 0.0171 | 0.0067 | 0.0123 | 0.0134 | 0.0069 |
| LangPTune | 0.0127 | 0.0074 | 0.0049 | 0.0145 | 0.0094 |
| D³ (Gemma) | 0.0072 | 0.0216 | 0.0117 | 0.0114 | 0.0194 |
| **R²ec (Qwen)** | **0.0237** | **0.0513** | **0.0288** | **0.0259** | **0.0457** |
| **R²ec (Gemma)** | **0.0264** | **0.0573** | **0.0326** | **0.0257** | **0.0527** |

R²ec improvements over the best baseline: H@5 on CDs +63.7%, N@10 on CDs +72.3%; H@10 on Instruments +67.0%.

### Ablation Study

| Configuration | Instruments H@5 | CDs H@5 | Games H@5 | Note |
|---------------|----------------|---------|-----------|------|
| w/ ClsHead (classification head) | 0.0044 | 0.0030 | 0.0012 | Decoupled reasoning–recommendation, severely degraded |
| w/o Reasoning | 0.0176 | 0.0469 | 0.0277 | No reasoning, contrastive learning only |
| w/o $R_d$ (continuous reward only) | 0.0198 | 0.0521 | 0.0302 | Insufficient discrimination |
| w/o $R_c$ (ranking reward only) | 0.0244 | 0.0543 | 0.0316 | Slightly below fused |
| **R²ec** | **0.0264** | **0.0588** | **0.0326** | Fused reward optimal |

### Key Findings

1. **Reasoning substantially improves recommendation**: Introducing reasoning yields an average improvement of approximately 15%, validating the effectiveness of test-time scaling in recommendation settings.
2. **Tight reasoning–recommendation coupling is critical**: The classification head variant (w/ ClsHead) suffers a dramatic performance collapse, indicating that reasoning and recommendation must share the same hidden state space to mutually benefit each other.
3. **The fused reward design is effective**: The discrete ranking reward $R_d$ is the primary signal; the continuous similarity reward $R_c$ provides supplementary fine-grained information. Using $R_c$ alone introduces noise.
4. **GRPO outperforms RLOO**: GRPO's unit-variance normalization amplifies reward gradients in the recommendation setting, accelerating early-stage learning, and reasoning length grows progressively during training—a phenomenon analogous to that observed in DeepSeek-R1.
5. **Smaller models can excel**: Gemma2-2B outperforms Qwen-3B on most tasks, suggesting that model selection matters more than parameter count.

## Highlights & Insights

- The dual-head architecture is particularly elegant: the shared backbone allows reasoning gradients to flow naturally into recommendation parameters, avoiding the gradient disconnection of two-stage approaches; using inner-product matching over an item embedding table is far more efficient than autoregressive item-ID generation.
- The work successfully transfers insights from LLM reasoning training (e.g., GRPO) to the recommendation domain, bridging two rapidly advancing research directions.
- The fused reward design reflects a deep understanding of challenges specific to recommendation—the discreteness of ranking metrics necessitates continuous signals as a complement.

## Limitations & Future Work

- Analysis of reasoning chain interpretability is primarily qualitative; systematic quantitative evaluation is lacking.
- Validation is currently limited to three Amazon datasets; generalizability to larger-scale and more diverse settings remains to be confirmed.
- The item embedding table must be constructed in advance, which may increase maintenance costs for rapidly changing item catalogs.
- Automatic control of reasoning length and further efficiency optimization are important directions for future work.

## Related Work & Insights

- The fundamental distinction from reasoning-augmented recommendation methods such as LangPTune is that R²ec is a genuinely end-to-end unified model, completing reasoning and recommendation within a single forward pass.
- The RecPO framework provides a reusable paradigm for RL training in recommender systems, particularly the fused reward design and the selective gradient backpropagation strategy.
- The work offers broader inspiration for "LLM + vertical domain" applications: how to design unified architectures that allow reasoning capabilities to naturally serve domain-specific tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First unified reasoning–recommendation model; both the dual-head architecture and RecPO are original contributions
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three datasets, multiple baselines, and nine analyses make for exceptionally comprehensive evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated arguments
- Value: ⭐⭐⭐⭐⭐ Opens a new paradigm for reasoning-enhanced recommendation with substantial empirical gains

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Think before Recommendation: Autonomous Reasoning-enhanced Recommender](think_before_recommendation_autonomous_reasoning-enhanced_recommender.md)
- [\[NeurIPS 2025\] Inference-Time Reward Hacking in Large Language Models](inference-time_reward_hacking_in_large_language_models.md)
- [\[NeurIPS 2025\] Radial Neighborhood Smoothing Recommender System](radial_neighborhood_smoothing_recommender_system.md)
- [\[ICLR 2026\] From Evaluation to Defense: Advancing Safety in Video Large Language Models](../../ICLR2026/recommender/from_evaluation_to_defense_advancing_safety_in_video_large_language_models.md)
- [\[AAAI 2026\] Inference-Aware Prompt Optimization for Aligning Black-Box Large Language Models](../../AAAI2026/recommender/inference-aware_prompt_optimization_for_aligning_black-box_large_language_models.md)

</div>

<!-- RELATED:END -->
