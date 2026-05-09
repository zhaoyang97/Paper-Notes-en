---
title: >-
  [Paper Note] The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning
description: >-
  [NeurIPS 2025][LLM Reasoning][RLVR Decomposition] This paper decomposes Reinforcement Learning from Verifiable Rewards (RLVR) into Positive Sample Reinforcement (PSR, which increases the probability of correct responses) and Negative Sample Reinforcement (NSR, which penalizes incorrect responses). The authors find that NSR alone consistently improves reasoning performance across the entire Pass@k spectrum and typically matches or surpasses PPO/GRPO. Based on this finding, the paper proposes Weighted-REINFORCE (reducing the PSR weight to 0.1), which achieves state-of-the-art results across MATH, AIME 2025, and AMC23.
tags:
  - NeurIPS 2025
  - LLM Reasoning
  - RLVR Decomposition
  - Negative Sample Reinforcement
  - Positive Sample Reinforcement
  - Pass@k
  - Weighted-REINFORCE
date: 2026-05-08
content_hash: 17695be55e257960
---

# The Surprising Effectiveness of Negative Reinforcement in LLM Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2506.01347](https://arxiv.org/abs/2506.01347)
**Code**: [GitHub](https://github.com/TianHongZXY/RLVR-Decomposed)
**Area**: LLM Reasoning
**Keywords**: RLVR Decomposition, Negative Sample Reinforcement, Positive Sample Reinforcement, Pass@k, Weighted-REINFORCE

## TL;DR

This paper decomposes Reinforcement Learning from Verifiable Rewards (RLVR) into Positive Sample Reinforcement (PSR, which increases the probability of correct responses) and Negative Sample Reinforcement (NSR, which penalizes incorrect responses). The authors find that NSR alone consistently improves reasoning performance across the entire Pass@k spectrum and typically matches or surpasses PPO/GRPO. Based on this finding, the paper proposes Weighted-REINFORCE (reducing the PSR weight to 0.1), which achieves state-of-the-art results across MATH, AIME 2025, and AMC23.

## Background & Motivation

**Background**: RLVR has become a core technique for training LLM reasoning capabilities. DeepSeek-R1 and Kimi K1.5 have demonstrated emergent long-chain reasoning and self-reflection via RLVR. RLVR employs binary rewards (correct: +1 / incorrect: −1), making it conceptually simple and independent of complex reward models.

**Limitations of Prior Work**: (1) The underlying learning mechanism of RLVR remains unclear—specifically, how the model learns separately from correct and incorrect samples. (2) Existing evaluations focus primarily on Pass@1 (greedy accuracy) and neglect Pass@k at large $k$, obscuring deeper changes in model behavior. Recent work (Yue et al., 2025) found that RL-trained models underperform base models on Pass@k at large $k$, suggesting a loss of diversity.

**Key Challenge**: RLVR simultaneously performs two operations—reinforcing correct outputs and penalizing incorrect ones—whose effects are entangled. Intuitively, positive reinforcement should be the dominant signal, yet it may overly narrow the output distribution; conversely, the contribution of negative reinforcement may be systematically underestimated.

**Goal**: To decouple RLVR into PSR and NSR, study their respective effects on Pass@k reasoning scaling, and design improved training objectives.

**Key Insight**: The RLVR objective is decomposed exactly as $\mathcal{L}_{RLVR} = \mathcal{L}_{PSR} + \mathcal{L}_{NSR}$. By training each component independently and evaluating with the full Pass@k spectrum ($k = 1$ to $256$), the independent effects of each signal can be observed and explained via token-level gradient analysis.

**Core Idea**: Penalizing incorrect responses alone is sufficient to improve reasoning—NSR refines existing knowledge by suppressing errors and redistributing probability mass according to the model's prior.

## Method

### Overall Architecture

The RLVR objective $\mathcal{L} = -\mathbb{E}[r(\bm{x}, \bm{y})]$ (where $r \in \{-1, +1\}$) is decomposed by reward sign into two sub-objectives: PSR (maximizing the likelihood of positive samples) and NSR (minimizing the likelihood of negative samples). Each is trained independently on Qwen2.5-Math-7B, Qwen3-4B, and Llama-3.1-8B-Instruct, and evaluated via Pass@k on MATH, AIME, and AMC23.

### Key Designs

1. **RLVR Objective Decomposition and Independent Evaluation**

   - **Function**: Isolates the contributions of positive and negative reinforcement signals.
   - **Mechanism**: $\mathcal{L}_{PSR} = -\mathbb{E}[\sum_{r=1} \pi_\theta(\bm{y}|\bm{x})]$; $\mathcal{L}_{NSR} = \mathbb{E}[\sum_{r=-1} \pi_\theta(\bm{y}|\bm{x})]$. During training, the policy is updated using only samples corresponding to the respective reward sign. PSR-only and NSR-only each use fewer samples than PPO/GRPO (only half the batch data each).
   - **Design Motivation**: Decoupling enables direct attribution of each signal's contribution to RLVR, addressing the question of which component—positive or negative reinforcement—matters more. Full-spectrum Pass@k provides a more comprehensive capability assessment than Pass@1 alone.

2. **Token-Level Gradient Analysis**

   - **Function**: Explains the distinct effects of PSR and NSR on the output distribution at the gradient level.
   - **Mechanism**: For PSR, the gradient on the sampled token $y_t$ is $\propto \pi_v(1 - \pi_v)$ (increasing), while for other tokens it is $\propto -\pi_{y_t}\pi_v$ (decreasing)—sharpening the distribution and reducing diversity. For NSR, the gradient on incorrect tokens is $\propto -\pi_v(1 - \pi_v)$ (decreasing), while for other tokens it is $\propto \pi_{y_t} \cdot \pi_v$ (increasing). Crucially, **NSR's probability redistribution is proportional to the current model probability**, meaning the model automatically identifies alternatives guided by its own prior.
   - **Design Motivation**: This reveals NSR's "self-calibration" mechanism—rather than teaching new behaviors, it removes errors and allows the model's own prior to surface. PSR, in contrast, forces concentration onto observed correct paths, suppressing other potentially valid solutions.

3. **Weighted-REINFORCE**

   - **Function**: Translates insights from the PSR/NSR analysis into an improved RL objective.
   - **Mechanism**: $\mathcal{L}_{W\text{-}REINFORCE} = \lambda \cdot \mathcal{L}_{PSR} + \mathcal{L}_{NSR}$, with $\lambda = 0.1$ substantially down-weighting positive reinforcement.
   - **Design Motivation**: PSR improves Pass@1 but degrades Pass@k at large $k$, while NSR maintains the full spectrum at a slight cost to Pass@1. Setting $\lambda = 0.1$ achieves the best balance between accuracy and diversity.

### Loss & Training

The training set consists of 7,500 MATH problems, with a prompt batch size of 1,024 and 8 rollouts per prompt; learning rate is 1e-6. The maximum sequence length is 4,096 for Qwen2.5-Math-7B and Llama, and 32,768 for Qwen3-4B. At evaluation, 256 or 64 responses are sampled per problem, and the unbiased Pass@k estimator from Chen et al. (2021) is applied.

## Key Experimental Results

### Main Results

| Method | MATH P@1 | MATH P@256 | AIME P@1 | AIME P@256 | AMC P@1 | AMC P@256 |
|--------|----------|------------|----------|------------|---------|-----------|
| Base | 63.2 | 96.9 | 6.1 | 46.7 | 41.0 | 100.0 |
| PPO | 76.6 | 96.3 | 8.5 | 43.3 | 62.0 | 97.5 |
| GRPO | 76.3 | 95.5 | 10.3 | 50.0 | 61.7 | 97.5 |
| PSR | 74.1 | 91.2 | 11.6 | 43.3 | 62.6 | 92.5 |
| NSR | 75.7 | **96.9** | 10.0 | **53.3** | 60.9 | **100.0** |
| W-REINF | **76.6** | 96.7 | **10.6** | **56.7** | **62.0** | 97.5 |

### Ablation Study

| Training Dynamics | PSR Effect | NSR Effect |
|-------------------|------------|------------|
| Test-set entropy | Drops sharply → diversity loss | Remains close to base → diversity preserved |
| Training-set accuracy | Rises rapidly → overfitting | Rises slowly → positive but not excessive |
| All-correct proportion | Highest → overconfidence | Lowest → uncertainty maintained |
| Pass@1 trend | Rapid improvement then saturation | Steady, sustained improvement |

### Key Findings

- **Core finding for NSR**: Training with negative samples alone achieves MATH Pass@1 = 75.7 (close to PPO's 76.6), with Pass@256 = 96.9 fully matching the base model—demonstrating that accuracy can be improved while diversity is preserved, without any positive examples.
- **Cost of PSR is clear**: MATH Pass@256 drops from 96.9 to 91.2, and Pass@k falls below the base model for $k > 8$. PSR causes distributional collapse.
- **Qwen3-4B case**: PSR entirely fails to activate the latent reasoning capability of the non-thinking mode; NSR improves Pass@1 from ~80% to 94%, approaching the thinking mode's 94.5%.
- **Llama case**: All RL methods degrade, but NSR degrades the least—confirming that the quality of the base model prior determines RL gains.
- **Weighted-REINFORCE**: AIME Pass@256 = 56.7 substantially outperforms GRPO (50.0%) and PPO (43.3%).

## Highlights & Insights

- **Counter-intuitive conclusion that "negative reinforcement matters more than positive reinforcement"**: The conventional assumption is that models require positive demonstrations to improve. This paper demonstrates that eliminating errors alone enables effective learning. The core mechanism is that NSR's probability redistribution follows the model's prior—effectively clearing interference so the model can find correct answers on its own. This has broader implications for the RLHF/RLAIF paradigm.
- **Full-spectrum Pass@k evaluation paradigm**: Evaluating Pass@1 alone is misleading—PPO/GRPO appear to improve accuracy but actually sacrifice reasoning coverage. Only the full Pass@k spectrum faithfully characterizes the model's capability frontier.
- **Elegance of Weighted-REINFORCE**: A single parameter $\lambda = 0.1$ consistently outperforms the more complex PPO and GRPO, challenging the assumption that more sophisticated RL algorithms are necessarily better.
- The $(1 - \pi_v)$ factor in NSR gradients provides a natural stopping mechanism: when the probability of an incorrect token is already low, the gradient approaches zero, preventing excessive penalization.

## Limitations & Future Work

- **Strong model dependence**: NSR yields significant gains on Qwen models but causes across-the-board degradation on Llama, indicating that the method's effectiveness is highly correlated with the quality of the base model's prior.
- Validation is limited to mathematical reasoning; whether similar effects hold for code generation, scientific reasoning, and other tasks requires further investigation.
- The choice of $\lambda = 0.1$ may not be universally optimal; different models and tasks may require different values.
- The dynamics of NSR across training stages are not analyzed—whether PSR/NSR weights should be adjusted between early and late training phases remains an open question.
- Connections to preference learning methods such as DPO and KTO are not discussed.

## Related Work & Insights

- **vs. Yue et al. (2025)**: That work observes a decline in Pass@k after RL training. The present paper identifies the source—PSR induces distributional collapse, while NSR provides a path to avoid such degradation.
- **vs. Dang et al. (2025)**: That work recovers post-SFT diversity through weight interpolation. This paper addresses the issue at the source by reducing the PSR weight in the training objective.
- **vs. DeepSeek-R1**: R1 treats RLVR as a monolithic objective. The decomposition perspective introduced here suggests that W-REINFORCE could further improve R1's reasoning scaling performance.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The decomposition perspective is concise and insightful; the findings are counter-intuitive and significant.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Full-spectrum evaluation across three models and three benchmarks, complemented by gradient analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ — Narrative is fluent and figures are well-crafted.
- Value: ⭐⭐⭐⭐⭐ — Paradigm-level impact on the understanding of RLVR mechanisms.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning](expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)
- [\[NeurIPS 2025\] Reasoning Models Hallucinate More: Factuality-Aware Reinforcement Learning for Large Reasoning Models](reasoning_models_hallucinate_more_factuality-aware_reinforcement_learning_for_la.md)
- [\[NeurIPS 2025\] GPO: Learning from Critical Steps to Improve LLM Reasoning](gpo_learning_from_critical_steps_to_improve_llm_reasoning.md)
- [\[NeurIPS 2025\] SQL-R1: Training Natural Language to SQL Reasoning Model By Reinforcement Learning](sql-r1_training_natural_language_to_sql_reasoning_model_by_reinforcement_learnin.md)
- [\[ICLR 2026\] Temperature as a Meta-Policy: Adaptive Temperature in LLM Reinforcement Learning](../../ICLR2026/llm_reasoning/temperature_as_a_meta-policy_adaptive_temperature_in_llm_reinforcement_learning.md)

<!-- RELATED:END -->
