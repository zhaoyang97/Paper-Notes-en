---
title: >-
  [Paper Note] Spotlight on Token Perception for Multimodal Reinforcement Learning
description: >-
  [ICLR 2026][Reinforcement Learning][RLVR] This paper proposes VPPO (Visually-Perceptive Policy Optimization), which quantifies the visual dependency of each token and refines learning signals at both the trajectory level…
tags:
  - "ICLR 2026"
  - "Reinforcement Learning"
  - "RLVR"
  - "multimodal reasoning"
  - "token perception"
  - "visual dependency"
  - "policy optimization"
date: 2026-05-08
content_hash: 487114b491f94c2f
---

# Spotlight on Token Perception for Multimodal Reinforcement Learning

**Conference**: ICLR 2026
**arXiv**: [2510.09285](https://arxiv.org/abs/2510.09285)  
**Code**: [https://github.com/huaixuheqing/VPPO-RL](https://github.com/huaixuheqing/VPPO-RL)  
**Area**: Multimodal Reinforcement Learning / Vision-Language Models
**Keywords**: RLVR, multimodal reasoning, token perception, visual dependency, policy optimization

## TL;DR

This paper proposes VPPO (Visually-Perceptive Policy Optimization), which quantifies the visual dependency of each token and refines learning signals at both the trajectory level and the token level, significantly enhancing the multimodal reasoning capabilities of large vision-language models.

## Background & Motivation

- **Limitations of RLVR in multimodal settings**: Existing RLVR methods (e.g., GRPO, DAPO) are primarily designed for text-based reasoning and neglect the critical role of visual perception in multimodal scenarios. They broadcast a uniform learning signal to all tokens, failing to distinguish which tokens genuinely depend on visual information.
- **Coupling of perception and reasoning**: Effective multimodal reasoning requires accurate visual perception as the foundation for logical inference. For instance, in geometry problems, the model must first identify from the image that OA and OB are radii of a circle before concluding that the triangle is isosceles.
- **Core findings**:
    - **Insight 1**: The visual dependency of tokens within a trajectory follows a sparse distribution — only a small fraction of tokens exhibit high visual dependency.
    - **Insight 2**: Different reasoning trajectories show significant heterogeneity in overall visual dependency — not all correct paths constitute genuinely vision-driven reasoning.

## Method

### Overall Architecture

VPPO introduces two modules on top of standard GRPO: **Trajectory-level Advantage Shaping (TAS)** and **Token-level Gradient Filtering (TGF)**, which hierarchically regulate learning signals via visual dependency scores.

### 1. Quantifying Token Visual Dependency

The visual dependency of a token at time step $t$ is defined as the KL divergence between the output distributions conditioned on the original image and on a masked image:

$$\mathcal{S}(s_t, I) := D_{\text{KL}}\left(\pi_\theta(\cdot|s_t, I) \| \pi_\theta(\cdot|s_t, I')\right)$$

where $I$ is the original image and $I'$ is a non-informative masked version. A high $\mathcal{S}$ value indicates that the prediction of this token is highly dependent on visual input.

### 2. Token-level Gradient Filtering (TGF, Micro-level)

For each trajectory $\tau_i$, the top-$k\%$ tokens with the highest visual dependency scores are selected to construct a binary gradient mask:

$$m_{i,t} = \mathbb{I}(t \in \mathcal{K}_i)$$

Policy gradients are computed only for these critical tokens, filtering out noise from generic tokens and combating signal dilution.

### 3. Trajectory-level Advantage Shaping (TAS, Macro-level)

The mean visual dependency $\bar{\mathcal{S}}(\tau_i)$ of each trajectory is computed, and a shaping factor is generated via normalization:

$$\alpha(\tau_i) = \beta_{\min} + (\beta_{\max} - \beta_{\min}) \frac{\bar{\mathcal{S}}(\tau_i) - \min_{\tau_j} \bar{\mathcal{S}}(\tau_j)}{\max_{\tau_j} \bar{\mathcal{S}}(\tau_j) - \min_{\tau_j} \bar{\mathcal{S}}(\tau_j)}$$

The shaped advantage is $\hat{A}'(\tau_i) = \alpha(\tau_i) \cdot \hat{A}_{\text{GRPO}}(\tau_i)$, amplifying updates for trajectories with high visual engagement while suppressing those with low visual dependency.

### 4. VPPO Objective

$$\mathcal{L}^{\text{VPPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\frac{1}{|o_i|}\sum_{t=1}^{|o_i|} m_{i,t} \cdot \min\left(r_{i,t}(\theta)\hat{A}'_i, \text{clip}(r_{i,t}(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}'_i\right)\right]$$

### Theoretical Analysis

A variance reduction theorem is established: $\text{Var}(\mathbf{g}_{\text{VPPO}}) \approx k \cdot \mathbb{E}[\alpha(\tau)^2] \cdot \text{Var}(\mathbf{g}_{\text{GRPO}})$, where $k \in (0,1)$ is the sparsity ratio and $\alpha(\tau)$ is scaled to a narrow band around 1, resulting in a significant reduction in gradient variance.

## Key Experimental Results

### Main Results: 8 Multimodal Reasoning Benchmarks (avg@8 acc %)

| Model | MathVerse | DynaMath | MMK12 | Geo3k | MathVision | We-Math | LogicVista | MMMU-Pro | Avg. |
|------|-----------|----------|-------|-------|------------|---------|------------|----------|------|
| Qwen2.5-VL-7B | 39.0 | 55.7 | 42.5 | 37.1 | 18.4 | 46.4 | 42.4 | 25.1 | 38.3 |
| + GRPO | 66.5 | 65.8 | 72.3 | 40.2 | 30.7 | 68.1 | 45.6 | 35.2 | 53.1 |
| + DAPO | 68.3 | 66.6 | 82.1 | 41.5 | 30.5 | 68.0 | 46.8 | 35.9 | 55.0 |
| **+ VPPO** | **71.6** | **68.1** | **82.8** | **46.5** | **33.3** | **71.5** | **47.9** | **37.9** | **57.5** |

### Scaling to 32B

| Model | Avg. |
|------|------|
| Qwen2.5-VL-32B + GRPO | 62.6 |
| Qwen2.5-VL-32B + DAPO | 63.5 |
| **Qwen2.5-VL-32B + VPPO** | **64.6** |

### Key Findings

- On the 7B model, VPPO achieves an average improvement of **19.2%** over the baseline, outperforming all open-source RL methods.
- On the 32B model, an average gain of **7.6%** is observed.
- Training is more stable and convergence is faster.

## Ablation Study

| Setting | Avg. Acc |
|------|----------|
| VPPO (full) | 57.5 |
| TAS only (trajectory-level advantage shaping) | 55.8 |
| TGF only (token-level gradient filtering) | 56.2 |
| w/o TAS + w/o TGF (DAPO baseline) | 55.0 |

- TAS and TGF are each independently effective; their combination yields the best performance.
- A gradient filtering ratio of $k=0.4$ is found to be optimal.

## Highlights & Insights

1. **First token-perception perspective on multimodal RLVR**: The work reveals two key insights — the sparse distribution of visual dependency and the heterogeneity across trajectories.
2. **Dual-level signal regulation**: The hierarchical design combining trajectory-level and token-level control is both elegant and effective.
3. **Plug-and-play compatibility**: The method integrates seamlessly into mainstream algorithms such as GRPO and DAPO.
4. **Theoretical grounding**: A formal proof of variance reduction is provided.

## Limitations & Future Work

- Computing visual dependency requires additional forward passes over masked images, incurring extra computational cost.
- Validation is limited to the Qwen2.5-VL model family; generalizability to other architectures remains to be examined.
- The choice of masking strategy (i.e., how to construct $I'$) may affect the quality of dependency estimation.

## Related Work & Insights

- **RL for multimodal reasoning**: GRPO, DAPO, NoisyRollout, VL-Rethinker, etc., all of which overlook visual perception.
- **Reward design**: Perception-aware reward approaches such as PAPO-D, which however do not introduce algorithmic-level improvements.
- **Critical token identification**: Branching-point detection and low-confidence exploration in RLHF, which are not tailored to visual dependency in multimodal settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — A fresh perspective on multimodal RL through token-level visual dependency analysis.
- **Technical Depth**: ⭐⭐⭐⭐ — Solid theoretical analysis with a formal proof of variance reduction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Eight benchmarks, two model scales, and comprehensive ablations.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play design with substantial empirical gains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] APPLE: Toward General Active Perception via Reinforcement Learning](apple_toward_general_active_perception_via_reinforcement_learning.md)
- [\[ICLR 2026\] From Narrow to Panoramic Vision: Attention-Guided Cold-Start Reshapes Multimodal Reasoning](from_narrow_to_panoramic_vision_attention-guided_cold-start_reshapes_multimodal_.md)
- [\[NeurIPS 2025\] Real-World Reinforcement Learning of Active Perception Behaviors](../../NeurIPS2025/reinforcement_learning/real-world_reinforcement_learning_of_active_perception_behaviors.md)
- [\[ICLR 2026\] MARS-Sep: Multimodal-Aligned Reinforced Sound Separation](mars-sep_multimodal-aligned_reinforced_sound_separation.md)
- [\[ICLR 2026\] Metis-SPECS: Decoupling Multimodal Learning via Self-distilled Preference-based Cold Start](metis-specs_decoupling_multimodal_learning_via_self-distilled_preference-based_c.md)

</div>

<!-- RELATED:END -->
