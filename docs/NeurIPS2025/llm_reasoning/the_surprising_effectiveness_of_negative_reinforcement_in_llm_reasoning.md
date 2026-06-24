---
title: >-
  [Paper Note] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning
description: >-
  [NeurIPS 2025][Reasoning][RLVR decomposition] This paper decomposes reinforcement learning with verifiable rewards (RLVR) into positive sample reinforcement (PSR, which increases the probability of correct responses) and negative sample reinforcement (NSR, which penalizes incorrect responses). It finds that NSR alone consistently improves reasoning performance across the full Pass@k spectrum and typically matches or surpasses PPO/GRPO. Based on this finding…
tags:
  - "NeurIPS 2025"
  - "Reasoning"
  - "RLVR decomposition"
  - "negative sample reinforcement"
  - "positive sample reinforcement"
  - "Pass@k"
  - "Weighted-REINFORCE"
date: 2026-05-08
content_hash: e2f4d9920d460177
---

# The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2506.01347](https://arxiv.org/abs/2506.01347)  
**Code**: [GitHub](https://github.com/TianHongZXY/RLVR-Decomposed)  
**Area**: Signal Communication
**Keywords**: RLVR decomposition, negative sample reinforcement, positive sample reinforcement, Pass@k, Weighted-REINFORCE

## TL;DR

This paper decomposes reinforcement learning with verifiable rewards (RLVR) into positive sample reinforcement (PSR, which increases the probability of correct responses) and negative sample reinforcement (NSR, which penalizes incorrect responses). It finds that NSR alone consistently improves reasoning performance across the full Pass@k spectrum and typically matches or surpasses PPO/GRPO. Based on this finding, the paper proposes Weighted-REINFORCE (reducing the PSR weight to 0.1), achieving state-of-the-art results across MATH, AIME 2025, and AMC23.

## Background & Motivation

**Background**: Reinforcement learning with verifiable rewards (RLVR) has become a core technique for training LLM reasoning capabilities. DeepSeek-R1 and Kimi K1.5 have demonstrated emergent long-chain reasoning and self-reflection via RLVR. RLVR employs binary rewards (correct: +1 / incorrect: −1), which is conceptually simple and requires no complex reward model.

**Limitations of Prior Work**: (1) The underlying learning mechanism of RLVR remains unclear—specifically, how the model learns from correct versus incorrect samples independently. (2) Existing evaluations focus on Pass@1 (greedy accuracy) while neglecting Pass@k at large $k$, obscuring deeper behavioral changes. Recent work (Yue et al., 2025) reports that models after RL training underperform base models on Pass@k at large $k$, suggesting diversity loss.

**Key Challenge**: RLVR simultaneously performs two operations—reinforcing correct responses and penalizing incorrect ones—whose effects are entangled. Intuitively, positive reinforcement should be the dominant signal, yet it may over-narrow the output distribution; the contribution of negative reinforcement may be systematically underestimated.

**Goal**: To decouple RLVR into PSR and NSR, independently study their effects on the Pass@k reasoning scaling curve, and design an improved training objective.

**Key Insight**: The RLVR objective is decomposed exactly as $\mathcal{L}_{RLVR} = \mathcal{L}_{PSR} + \mathcal{L}_{NSR}$. By training each component separately and evaluating across the full Pass@k spectrum ($k = 1$ to $256$), the independent effect of each signal can be observed and then explained via token-level gradient analysis.

**Core Idea**: Penalizing incorrect responses alone can effectively improve reasoning—by suppressing errors and redistributing probability mass according to the model's prior, NSR refines existing knowledge without requiring positive demonstrations.

## Method

### Overall Architecture

The RLVR objective $\mathcal{L} = -\mathbb{E}[r(\bm{x}, \bm{y})]$ (with $r \in \{-1, +1\}$) is decomposed by reward sign into two sub-objectives: PSR (maximizing the likelihood of positive samples) and NSR (minimizing the likelihood of negative samples). Each sub-objective is trained independently on Qwen2.5-Math-7B, Qwen3-4B, and Llama-3.1-8B-Instruct, and evaluated on MATH, AIME, and AMC23 via Pass@k.

### Key Designs

1. **RLVR Objective Decomposition and Independent Evaluation**:

    - Function: Isolating the contributions of positive and negative reinforcement signals.
    - Mechanism: $\mathcal{L}_{PSR} = -\mathbb{E}[\sum_{r=1} \pi_\theta(\bm{y}|\bm{x})]$; $\mathcal{L}_{NSR} = \mathbb{E}[\sum_{r=-1} \pi_\theta(\bm{y}|\bm{x})]$. During training, only samples with the corresponding reward sign are used to update the policy. PSR-only and NSR-only each use fewer samples than PPO/GRPO (each utilizing only half the batch).
    - Design Motivation: Decoupling enables direct attribution of each signal's contribution in RLVR, answering the question of which—positive or negative reinforcement—is more important. The full Pass@k spectrum provides a more comprehensive capability evaluation than Pass@1 alone.

2. **Token-Level Gradient Analysis**:

    - Function: Explaining the distinct effects of PSR and NSR on the output distribution from a gradient perspective.
    - Mechanism: For PSR, the gradient direction for the sampled token $y_t$ is $\propto \pi_v(1-\pi_v)$ (increasing), while all other tokens follow $\propto -\pi_{y_t}\pi_v$ (decreasing)—resulting in distribution sharpening and diversity reduction. For NSR, the gradient direction for incorrect tokens is $\propto -\pi_v(1-\pi_v)$ (decreasing), while other tokens follow $\propto \pi_{y_t} \cdot \pi_v$ (increasing). Crucially, **NSR's probability redistribution is proportional to the current model probability**, meaning the model automatically identifies alternatives guided by its own prior.
    - Design Motivation: This analysis reveals NSR's "self-calibration" mechanism—it does not teach new behaviors, but instead removes errors and allows the model's own prior to emerge. In contrast, PSR forces probability mass to concentrate on observed correct paths, suppressing other potentially correct answers.

3. **Weighted-REINFORCE**:

    - Function: Designing an improved RL objective informed by the PSR/NSR analysis.
    - Mechanism: $\mathcal{L}_{W\text{-}REINFORCE} = \lambda \cdot \mathcal{L}_{PSR} + \mathcal{L}_{NSR}$, with $\lambda = 0.1$ substantially reducing the weight of positive reinforcement.
    - Design Motivation: PSR improves Pass@1 but degrades Pass@k at large $k$, while NSR preserves the full spectrum at a marginal cost in Pass@1. Setting $\lambda = 0.1$ achieves the optimal balance between accuracy and diversity.

### Loss & Training

The training set consists of 7,500 MATH problems; the prompt batch size is 1,024 with 8 rollouts per prompt; the learning rate is 1e-6. The maximum sequence length is 4,096 for Qwen2.5-Math-7B and Llama, and 32,768 for Qwen3-4B. Evaluation samples 256 or 64 responses per problem using the unbiased Pass@k estimator of Chen et al. (2021).

## Key Experimental Results

### Main Results

| Method | MATH P@1 | MATH P@256 | AIME P@1 | AIME P@256 | AMC P@1 | AMC P@256 |
|--------|---------|-----------|---------|-----------|--------|----------|
| Base | 63.2 | 96.9 | 6.1 | 46.7 | 41.0 | 100.0 |
| PPO | 76.6 | 96.3 | 8.5 | 43.3 | 62.0 | 97.5 |
| GRPO | 76.3 | 95.5 | 10.3 | 50.0 | 61.7 | 97.5 |
| PSR | 74.1 | 91.2 | 11.6 | 43.3 | 62.6 | 92.5 |
| NSR | 75.7 | **96.9** | 10.0 | **53.3** | 60.9 | **100.0** |
| W-REINF | **76.6** | 96.7 | **10.6** | **56.7** | **62.0** | 97.5 |

### Ablation Study

| Training Dynamics | PSR Effect | NSR Effect |
|-------------------|-----------|-----------|
| Test-set entropy | Sharp decrease → diversity loss | Remains close to base → diversity preserved |
| Training-set correctness ratio | Rapid increase → overfitting | Slow increase → positive but not excessive |
| All-correct proportion | Highest → overconfidence | Lowest → uncertainty retained |
| Pass@1 trend | Rapid improvement then saturation | Steady, continuous improvement |

### Key Findings

- **Core finding for NSR**: Training solely on negative samples achieves MATH Pass@1 = 75.7 (approaching PPO's 76.6), with Pass@256 = 96.9 fully matching the base model—demonstrating that accuracy gains and diversity preservation are simultaneously achievable without any positive samples.
- **The cost of PSR is clear**: MATH Pass@256 drops from 96.9 to 91.2, and Pass@k falls below the base model for $k > 8$. PSR induces distribution collapse.
- **Qwen3-4B case**: PSR completely fails to activate latent reasoning capabilities in non-thinking mode; NSR improves Pass@1 from ~80% to 94%, approaching the thinking-mode result of 94.5%.
- **Llama case**: All RL methods degrade performance, but NSR degrades the least—indicating that the quality of the base model's prior determines the benefit obtainable from RL.
- **Weighted-REINFORCE**: AIME Pass@256 = 56.7 substantially surpasses GRPO (50%) and PPO (43.3%).

## Highlights & Insights

- **The counterintuitive conclusion that "negative reinforcement matters more than positive reinforcement"**: The conventional view holds that models require positive demonstrations to improve, yet this paper demonstrates that eliminating errors alone suffices for effective learning. The core mechanism is that NSR's probability redistribution follows the model's prior—effectively clearing noise and allowing the model to find correct answers on its own. This finding has broader implications for the RLHF/RLAIF community.
- **The full Pass@k evaluation paradigm**: Evaluating only Pass@1 is misleading—PPO/GRPO appear to improve accuracy while actually sacrificing reasoning coverage. The full Pass@k spectrum is necessary to faithfully characterize a model's capability boundary.
- **The elegance of Weighted-REINFORCE**: A single parameter $\lambda = 0.1$ consistently outperforms the more complex PPO and GRPO algorithms, challenging the assumption that more sophisticated RL methods are inherently superior.
- The $(1-\pi_v)$ factor in the NSR gradient provides a natural stopping mechanism: as the probability of erroneous tokens approaches zero, the gradient vanishes, preventing excessive penalization.

## Limitations & Future Work

- **Strong model dependency**: NSR yields substantial improvements on Qwen models but causes widespread degradation on Llama, suggesting that the method's effectiveness is highly dependent on the quality of the base model's prior.
- Validation is limited to mathematical reasoning; whether similar findings hold for code generation, scientific reasoning, and other tasks requires further investigation.
- The choice of $\lambda = 0.1$ may not be universally optimal; different values may be required for different models and tasks.
- The dynamics of NSR across different training phases are not analyzed: whether PSR/NSR weights should be adjusted early versus late in training remains an open question.
- Connections to preference learning methods such as DPO and KTO are not discussed.

## Related Work & Insights

- **vs. Yue et al. (2025)**: That work observes a decline in Pass@k after RL training. The present paper explains the source of this degradation—PSR causes diversity collapse, while NSR provides a path to avoid it.
- **vs. Dang et al. (2025)**: That work restores post-SFT diversity through weight interpolation. The present paper instead reduces PSR weight at the level of the training objective itself.
- **vs. DeepSeek-R1**: R1 treats RLVR as a monolithic objective. The decomposition perspective proposed here implies that W-REINFORCE could further improve R1's reasoning scaling performance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The decomposition perspective is concise and insightful; the findings are counterintuitive and significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Full-spectrum evaluation across three models and three benchmarks, complemented by gradient analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Narrative is fluent and figures are refined.
- Value: ⭐⭐⭐⭐⭐ — The work has paradigm-shifting implications for understanding RLVR mechanisms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] ResRL: Boosting LLM Reasoning via Negative Sample Projection Residual Reinforcement Learning](../../ICML2026/llm_reasoning/resrl_boosting_llm_reasoning_via_negative_sample_projection_residual_reinforceme.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution](swe-rl_advancing_llm_reasoning_via_reinforcement_learning_on_open_software_evolu.md)
- [\[NeurIPS 2025\] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [\[ACL 2025\] Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness](../../ACL2025/llm_reasoning/towards_better_chain-of-thought_a_reflection_on_effectiveness_and_faithfulness.md)

</div>

<!-- RELATED:END -->
