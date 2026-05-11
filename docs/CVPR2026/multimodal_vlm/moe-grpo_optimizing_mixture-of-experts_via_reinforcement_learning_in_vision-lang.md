---
title: >-
  [Paper Note] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models
description: >-
  [CVPR 2026][Multimodal VLM][Mixture-of-Experts] This paper models expert selection in MoE as a sequential decision problem and optimizes the routing strategy via GRPO-based reinforcement learning. By introducing modality-aware router guidance, the proposed method consistently outperforms deterministic top-K routing and its variants on image and video understanding tasks in VLMs.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Mixture-of-Experts
  - Reinforcement Learning
  - Routing Strategy Optimization
  - Vision-Language Models
  - GRPO
date: 2026-05-08
content_hash: c82023bbf9891701
---

# MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models

**Conference**: CVPR 2026
**arXiv**: [2603.24984](https://arxiv.org/abs/2603.24984)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Mixture-of-Experts, Reinforcement Learning, Routing Strategy Optimization, Vision-Language Models, GRPO

## TL;DR

This paper models expert selection in MoE as a sequential decision problem and optimizes the routing strategy via GRPO-based reinforcement learning. By introducing modality-aware router guidance, the proposed method consistently outperforms deterministic top-K routing and its variants on image and video understanding tasks in VLMs.

## Background & Motivation

**Background**: Mixture-of-Experts (MoE) reduces the computational cost of Transformers by sparsely activating a subset of parameters while maintaining high model capacity. MoE has recently been extended to vision-language models (VLMs) for efficient multimodal understanding. The standard practice is to greedily select experts layer-wise using deterministic top-K routing.

**Limitations of Prior Work**: Deterministic top-K routing restricts exploration of diverse expert combinations and is prone to overfitting to a small subset of experts. Methods such as V-MoE introduce stochasticity by adding Gaussian noise to gating scores, but this heuristic perturbation only partially alleviates the problem without explicitly optimizing the expert selection "policy."

**Key Challenge**: Existing approaches are either deterministic (lacking exploration) or noise-based stochastic (lacking directional guidance), and neither truly learns an optimal expert routing policy. Expert routing is inherently a sequential decision problem, yet it has consistently been treated as a simple softmax + top-K operation.

**Goal**: (1) How can MoE routers learn better expert combinations? (2) How can routing policy exploration be conducted efficiently and stably in multimodal settings?

**Key Insight**: Expert selection is explicitly modeled as a sequential decision problem. Reinforcement learning (GRPO) is employed to explore diverse expert combinations through multiple rollouts and to optimize the routing policy based on reward feedback. The observation that tokens from different modalities exhibit distinct expert preferences is leveraged as a prior to constrain the exploration space.

**Core Idea**: Deterministic top-K routing is replaced by GRPO-based reinforcement learning. Token-GRPO and Gate-GRPO are jointly applied to optimize token generation and layer-wise expert selection policies, respectively, while modality-aware guidance is introduced to accelerate convergence.

## Method

### Overall Architecture

Given an input image (or video) and a question $\boldsymbol{x}$, the rollout module $g_\text{old}$ samples $G$ groups of expert routing strategies $\{\boldsymbol{E}^i\}_{i=1}^G$ from the gating network. Each routing strategy $\boldsymbol{E}^i$ corresponds to a sequence of expert selections across all layers. Under each routing, the model generates an output token sequence $\boldsymbol{y}^i$ and computes an accuracy reward $R^i$. Group-relative rewards are used to compute advantage estimates $\hat{A}^i$, which guide policy updates toward higher-reward expert combinations.

### Key Designs

1. **Token-GRPO (Token-level Generation Optimization)**:

    - **Function**: Optimizes expert selection strategies at the output token level.
    - **Mechanism**: Building on standard GRPO, expert routing conditions are incorporated. For each rollout-sampled expert strategy $\boldsymbol{E}^i$, the model generates the corresponding token sequence $\boldsymbol{y}^i$. A PPO-style clipped ratio objective optimizes token-level generation probabilities based on group-relative rewards. The ratio is defined as $r_t^i = \pi_\theta(y_t^i | \boldsymbol{x}, \boldsymbol{y}_{<t}^i; \boldsymbol{E}_{<t}^i) / \pi_\text{old}(y_t^i | \boldsymbol{x}, \boldsymbol{y}_{<t}^i; \boldsymbol{E}_{<t}^i)$.
    - **Design Motivation**: Token-level optimization is directly tied to task rewards and serves as the primary driver of performance gains. Ablation studies show that removing Token-GRPO alone causes average accuracy to drop sharply from 55.7% to 50.9%.

2. **Gate-GRPO (Layer-wise Routing Strategy Optimization)**:

    - **Function**: Directly optimizes the expert selection strategy of the gating network at each layer.
    - **Mechanism**: At each layer and token position, the routing ratio is computed as $\hat{r}_{t,l}^i = g_\theta^l(E_{t,l}^i) / g_\text{old}^l(E_{t,l}^i)$, averaged over all layers and token positions, and optimized using the same clipped objective. Unlike Token-GRPO, Gate-GRPO provides dense layer-wise supervision signals that directly act on the gating function.
    - **Design Motivation**: Token-GRPO only indirectly influences routing through output-level feedback, whereas Gate-GRPO provides fine-grained, dense supervision at each layer. The two are complementary: removing Gate-GRPO reduces average accuracy by 1.8%.

3. **Modality-Aware Router Guidance**:

    - **Function**: Constrains the routing exploration space to avoid wasting exploration on irrelevant experts.
    - **Mechanism**: The selection counts of each expert for visual tokens and text tokens, $N_v(e_i)$ and $N_t(e_i)$, are first accumulated to compute normalized modality-aware scores $\hat{s}_v(e_i)$ and $\hat{s}_t(e_i)$. When processing visual tokens, experts are ranked by score and the bottom $P\%$ are masked by setting their gating scores to $-\infty$; multinomial sampling is then performed over the remaining experts. $P = 25\%$ is used in experiments.
    - **Design Motivation**: The search space for RL exploration is large (choosing $K$ out of $N$ experts per layer, across all layers and token positions). Undirected exploration is inefficient. Modality-aware guidance leverages prior knowledge of expert modality preferences to reduce redundant exploration and accelerate convergence. Ablations show gains of 1.5% and 0.9% over noise-based and multinomial sampling without guidance, respectively.

### Loss & Training

The final objective is $\mathcal{L}_\text{MoE-GRPO} = \mathcal{L}_\text{Token-GRPO} + \mathcal{L}_\text{Gate-GRPO}$. Since the gating network is trained from scratch without a pre-trained routing policy, KL divergence regularization is omitted (unlike standard GRPO). The reward function uses accuracy reward (correct = 1, incorrect = 0). Training is based on InternVL3.5-1B converted to a MoE architecture ($N = 8$ experts, $K = 2$ activated), with 2.9B total parameters and 1.3B activated. A dataset of 100K multi-choice visual instruction tuning samples is used for 25K training steps, completing in approximately one day on 4 GPUs.

## Key Experimental Results

### Main Results

| Model | Architecture | Activated/Total Params | MMBench | MMStar | MLVU | LongVideoBench | Avg. |
|-------|-------------|----------------------|---------|--------|------|----------------|------|
| InternVL3.5 + Det-FT | MoE | 1.3B/2.9B | 75.8 | 45.6 | 48.6 | 45.3 | 54.0 |
| InternVL3.5 + Stoch-FT-Noise | MoE | 1.3B/2.9B | 76.3 | 46.1 | 51.1 | 45.3 | 54.3 |
| InternVL3.5 + MoE-GRPO | MoE | 1.3B/2.9B | **77.5** | 45.7 | **53.1** | **46.5** | **56.0** |
| InternVL2.5 (Dense) | Dense | 1B/1B | 70.7 | 50.1 | 57.3 | 47.9 | - |

MoE-GRPO achieves the best results on 7 out of 9 benchmarks, surpassing the three baselines by 2.0%, 2.3%, and 1.7% in average accuracy, respectively.

### Ablation Study

| Configuration | Avg. | Note |
|--------------|------|------|
| Token-GRPO + Gate-GRPO (Full) | 55.7 | Full model |
| Token-GRPO only | 53.9 | −1.8% without Gate-GRPO |
| Gate-GRPO only | 50.9 | −4.8% without Token-GRPO; token-level optimization is the primary driver |
| Modality-aware guidance | 55.7 | Best |
| Modality-agnostic (noise) | 54.2 | −1.5% |
| Modality-agnostic (multinomial) | 54.8 | −0.9% |

### Key Findings

- Token-GRPO is the primary driver of performance; Gate-GRPO provides complementary fine-grained layer-wise supervision.
- Modality-aware guidance outperforms undirected approaches by 0.9–1.5% on average, with faster convergence and lower reward variance.
- MoE-GRPO substantially improves expert diversity: routing distribution entropy increases from 1.05 (Det-FT) to 1.82.
- In cross-dataset generalization experiments (CLIP-MoE), MoE-GRPO outperforms Det-FT by 3.1% on average, while Det-FT degrades due to overfitting.
- In cross-domain generalization, MoE-GRPO consistently improves across all OOD datasets, outperforming CLIP-MoE by 4.1% on average.

## Highlights & Insights

- **A New Paradigm for RL-based Routing Optimization**: This is the first work to model MoE expert selection as a sequential decision problem and optimize it with RL. The formulation is novel and effective, opening a new direction for routing optimization in MoE architectures. More sophisticated RL algorithms can be explored in future work.
- **Complementary Dual-level GRPO Design**: Token-GRPO and Gate-GRPO provide supervision at the output level and layer level, respectively, combining coarse-grained task alignment with fine-grained routing optimization. This hierarchical optimization paradigm is transferable to other layered decision-making problems.
- **Modality-Aware Constraint on Exploration Space**: By leveraging modality–expert statistical priors to constrain the exploration space, training efficiency and stability are significantly improved without additional complexity. This idea of using prior knowledge to guide RL exploration is broadly applicable to other RL + large model settings.

## Limitations & Future Work

- Validation is currently limited to relatively small-scale models (1.3B activated parameters); effectiveness on larger-scale MoE-VLMs (e.g., DeepSeek-V3 scale) remains unknown.
- Using $G = 8$ rollouts increases training computation (requiring generation of 8 output sets); maintaining efficiency at larger training scales is a challenge.
- The reward function relies solely on accuracy (multiple-choice tasks); applicability to open-ended generation tasks has yet to be verified.
- Modality-aware guidance is based on static expert preference statistics, which may limit dynamic adaptability.

## Related Work & Insights

- **vs. V-MoE**: V-MoE introduces exploration via Gaussian noise on gating scores, but this is an undirected heuristic perturbation that does not optimize a "policy." MoE-GRPO explicitly optimizes the routing policy and achieves superior results.
- **vs. Expert Choice / Optimal Transport Routing**: These methods optimize routing from a load-balancing perspective, whereas MoE-GRPO optimizes from a task reward perspective. The two are complementary: combining them yields a further gain of 0.9%.
- **vs. Standard GRPO (DeepSeek-R1)**: Standard GRPO explores only at the token level. MoE-GRPO extends the action space to layer-wise expert selection, enabling finer-grained control.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of RL to MoE routing strategy optimization with a clear formulation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Nine benchmarks covering image and video understanding, cross-dataset and cross-domain generalization, multiple ablations, and routing analysis.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, rich figures and tables, complete methodological derivation.
- **Value**: ⭐⭐⭐⭐ Provides a new paradigm for MoE routing optimization, though rollout overhead may limit practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Quant Experts: Token-aware Adaptive Error Reconstruction with Mixture of Experts for Large Vision-Language Models Quantization](quant_experts_token_aware_vlm_quantization.md)
- [\[CVPR 2026\] TRivia: Self-supervised Fine-tuning of Vision-Language Models for Table Recognition](trivia_self-supervised_fine-tuning_of_vision-language_models_for_table_recogniti.md)
- [\[CVPR 2026\] MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models](mupo_all_roads_lead_to_rome_incentivizing_divergent_thinking_in_vlms.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)
- [\[CVPR 2026\] EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)

</div>

<!-- RELATED:END -->
