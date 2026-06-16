---
title: >-
  [Paper Note] RebRL: Reinforcing Discrete Visual Diffusion Models with Rebalanced Timestep Credits
description: >-
  [CVPR 2026][Image Generation][Reinforcement Learning] Addressing the neglected issue of "severe structural imbalance in timestep credit assignment" when applying GRPO to Discrete Diffusion Models (DDM), this paper derives the mathematical roots of this imbalance from policy gradients. It proposes RebRL, a plug-and-play method that flattens cumulative gradients using two l
tags:
  - CVPR 2026
  - Image Generation
  - Reinforcement Learning
  - GRPO
date: 2026-05-08
content_hash: 3baba5203339bde0
---
# RebRL: Reinforcing Discrete Visual Diffusion Models with Rebalanced Timestep Credits

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Zhang_RebRL_Reinforcing_Discrete_Visual_Diffusion_Models_with_Rebalanced_Timestep_Credits_CVPR_2026_paper.html)  
**Code**: None (Not mentioned as open source)  
**Area**: Diffusion Models / Image Generation / Reinforcement Learning  
**Keywords**: Discrete Diffusion Models, Reinforcement Learning, GRPO, Credit Assignment, Text-to-Image Generation  

## TL;DR
Addressing the neglected issue of "severe structural imbalance in timestep credit assignment" when applying GRPO to Discrete Diffusion Models (DDM), this paper derives the mathematical roots of this imbalance from policy gradients. It proposes RebRL, a plug-and-play method that flattens cumulative gradients using two levels of rebalancing factors—timestep-level and token-level. It achieves SOTA on GenEval, improves human preference scores by up to 3.40, and reduces training steps by approximately 40%.

## Background & Motivation

**Background**: Discrete Diffusion Models (DDM) are becoming a new focal point in visual generation due to their ability to predict multiple discrete tokens in parallel during a single forward pass and their natural compatibility with language models. Applying RL methods like GRPO for post-training (aligning with human preferences and improving compositional generation) is gaining traction, with representative works including diffu-GRPO, UniGRPO, and MaskGRPO.

**Limitations of Prior Work**: DDM lacks the "per-token log-likelihood decomposition" found in auto-regressive models; therefore, the importance ratio can only be computed at the full sequence level, making per-token estimation unavailable. To enable the model to learn from different inference timesteps, existing methods simulate multi-step reasoning by applying different mask ratios to fully denoised token sequences. This study finds that this "simulated demasking" process leads to **severe credit assignment imbalance**: tokens decoded later (with lower mask ratios) appear more frequently in the policy gradient summation, resulting in larger cumulative gradient scales.

**Key Challenge**: The authors use $\Delta$HPS (variation in human preference scores along the inference trajectory) to measure "exploration potential." They found that early timesteps have high $\Delta$HPS and determine global structure, meaning they should receive larger gradients. However, because the GRPO objective is only evaluated on "currently masked tokens," these high-value early tokens receive the **smallest** gradients. The gradient scale trend is exactly opposite to the importance trend—this is the essence of the imbalance.

**Goal**: Flatten the cumulative gradient scales across timesteps and tokens to improve the exploration-exploitation trade-off and accelerate convergence without introducing additional storage/computation overhead or hyperparameters.

**Key Insight**: First, perform a principled derivation of the DDM policy gradient to express the cause of imbalance as a closed-form cumulative scale $w(\Delta t_j)$. Then, design rebalancing factors based on this derivation—ensuring the correction is "analytically grounded" rather than a heuristic.

**Core Idea**: Replace the uniform strategy of treating all timesteps equally with a two-layer rebalancing: "timestep-level factor $\lambda(t_j)$ + token-level inverse frequency $1/F$."

## Method

### Overall Architecture
RebRL does not alter the overall GRPO workflow but inserts two rebalancing factors into the gradients of the GRPO objective. The pipeline remains a standard GRPO post-training loop: for each prompt $c$, sample $G$ rollouts, calculate group-relative advantages $A_i$, and use $\mu$ inner iterations with different mask ratios $t_j$ to simulate generation timesteps. The modifications in RebRL are concentrated entirely on "how to weight each timestep and token when calculating the loss." Thus, it is plug-and-play, requires almost zero extra parameters or compute, and introduces no new hyperparameters.

DDM treats generation as a continuous-time absorbing Markov process: clean data is $x_0$, and noise is gradually added as $t \to 1$. When adapting GRPO to DDM, the importance ratio is calculated per-step for the entire sequence. The policy gradient approximation (Eq. 9) reveals the key issue: due to the indicator function $\delta(o^{t_j}_k, m)$, a token contributes to the gradient as long as it remains masked. Consequently, **tokens decoded later appear in more terms of the summation**. For a linear schedule $t_j = j/\mu$, when a token is revealed at mask ratio $t_j$, its contribution is weighted by a cumulative scale:

$$w(\Delta t_j) = \sum_{j'=j}^{\mu} \frac{1}{t_{j'}} = \mu \sum_{j'=j}^{\mu} \frac{1}{j'}$$

As $w(\Delta t_j)$ monotonically increases for tokens revealed later, this provides an analytical source for the imbalance. RebRL flattens this across two layers.

### Key Designs

**1. Analytical Diagnosis of Policy Gradient Imbalance**

This serves as the foundation. The authors approximate the DDM policy gradient into a computable form (Eq. 9), using the DDM's ELBO loss term $\ell_{\pi_\theta}$ as a proxy for the intractable log-likelihood $\log\pi_\theta(o|o^{t_j},c)$. Since tokens only enter the sum while masked, late-decoded tokens appear in more $j$ terms. The cumulative scale $w(\Delta t_j)$ increases monotonically as the decoding step gets later, and the step increment increases as the reverse index $j$ decreases, further exacerbating the imbalance. This diagnosis turns the lack of theoretical basis in existing mask strategies into a target for precise correction.

**2. Timestep-level Rebalancing $\lambda(t_j)$**

To address the first-order issue of excessive cumulative gradients in late timesteps, the authors multiply the policy gradient by a timestep factor $\lambda(t_j)$ that increases monotonically with $t_j$ (Eq. 11), giving higher weights to tokens at large mask ratios (early generation). Two options are provided: a first-order factor $\lambda(t_j)=t_j$ and a second-order factor $\lambda(t_j)=t_j^2$. The resulting rebalanced scales are:

$$w^{(1)}(\Delta t_j) = 1-\frac{j}{\mu}, \qquad w^{(2)}(\Delta t_j) = \frac{\sum_{j'=j}^{\mu} j'}{\sum_{j'=1}^{\mu} j'}$$

This significantly suppresses the overall scale of late timesteps. However, its limitation is that $\lambda(t_j)$ treats all masked tokens in a sequence identically; late-decoded tokens still accumulate more gradients—it mitigates but does not cure the issue.

**3. Token-level Rebalancing $1/F$**

To solve the root cause, the "one-size-fits-all" treatment within a timestep must be broken. The authors note that tokens in a batch participate in gradient updates a different number of times. They construct a **mask frequency map** $F$, counting the total times each token is masked during the rollout phase: $F(k)=\sum_{j=1}^{\mu}\delta(o^{t_j}_k, m)$. They then multiply each token's loss by the pre-calculated weight $1/F(k)$. Tokens masked more frequently (later decoding, larger cumulative gradient) receive smaller weights, equalizing the contributions. This factor is derived entirely from the model's own masking statistics without new hyperparameters, advancing the fix from "timestep-level mitigation" to "token-level cure."

### Loss & Training
The backbone is the multimodal DDM **MMaDA-8B-Base**, initialized from public pre-trained checkpoints. Learning rate $3\times10^{-6}$, 6x A100 for training (2 additional cards for the reward server), global batch size 72, $G=9$ rollouts per prompt, 12 inference steps, CFG scale 3.5. Training spans 1800 global steps with $\mu=8$ inner GRPO iterations. The objective follows the standard GRPO "reward - KL penalty" trade-off (Eq. 4), with RebRL simply multiplying $\lambda(t_j)$ and $1/F$ into the objective $J$.

## Key Experimental Results

### Main Results

GenEval compositional generation benchmark (using MMaDA as backbone):

| Method | Counting | Position | Attr. Binding | Overall↑ |
| :--- | :--- | :--- | :--- | :--- |
| MMaDA* (Base) | 0.60 | 0.61 | 0.67 | 0.74 |
| w/ diffu-GRPO | 0.72 | 0.78 | 0.74 | 0.82 |
| w/ UniGRPO | 0.77 | 0.75 | 0.75 | 0.82 |
| w/ MaskGRPO | 0.69 | 0.77 | 0.72 | 0.84 |
| **w/ RebRL (Ours)** | **0.81** | **0.83** | **0.76** | **0.86** |

RebRL increases the base score from 0.74 to 0.86, with gains concentrated in attributes determining global structure—Counting 0.81 (vs. 0.77 for UniGRPO) and Position 0.83 (vs. 0.78 for diffu-GRPO). This confirms that flattening gradients allows the policy to learn more effectively from early samples with high mask ratios.

Human Preference Alignment (HPS series):

| Method | HPSv3↑ | ImageReward↑ | DeQA↑ |
| :--- | :--- | :--- | :--- |
| MMaDA | 9.61 | 0.99 | 3.98 |
| w/ diffu-GRPO | 11.99 | 1.20 | 4.15 |
| w/ UniGRPO | 12.09 | 1.19 | 4.16 |
| w/ MaskGRPO | 12.35 | 1.21 | 4.19 |
| **w/ RebRL (Ours)** | **13.01** | **1.28** | **4.22** |

RebRL's HPSv3 score of 13.01 significantly outperforms all DDM-RL methods, with a gain of ~3.40 over the base model.

### Ablation Study

Comparison of four rebalancing strategies:

| Configuration | Reward Convergence | Stability (KL) | Description |
| :--- | :--- | :--- | :--- |
| w/o Re-balance (UniGRPO) | Slowest, lowest final value | KL high and rising fast | Baseline with imbalanced gradients |
| 1-Order ($\lambda=t_j$) | Significantly faster | KL low and stable | Timestep-level mitigation |
| 2-Order ($\lambda=t_j^2$) | Significantly faster | KL low and stable | Timestep-level mitigation |
| Token-level ($1/F$) | **Fastest, highest final value** | KL low and stable | Token-level cure |

### Key Findings
- **Token-level rebalancing contributes most**: It resolves the imbalance at the individual token level, resulting in the fastest convergence and highest final reward.
- **Significant difference in training stability**: The KL loss for the baseline is an order of magnitude higher than for the rebalancing strategies, indicating that imbalanced gradients cause the policy to deviate violently from the reference, leading to reward hacking.
- **diffu-GRPO's full-mask strategy is a different trade-off**: By masking the entire rollout, it avoids cumulative imbalance and converges very fast early on ($<400$ steps). However, its "single-step prediction" deviates from multi-step iterative inference, leading to a training-inference bias and lower performance saturation.
- **Efficiency**: RebRL reduces the number of training steps required to reach the same performance by ~40% compared to UniGRPO.

## Highlights & Insights
- **Converting heuristics into provable targets**: By quantifying the imbalance with the closed-form $w(\Delta t_j)$ and designing factors to cancel it, the "diagnosis-then-prescription" approach is more convincing than simple reweighting and explains why no hyperparameter tuning is needed.
- **Elegance of $1/F$ inverse frequency weighting**: Weights are derived entirely from rollout statistics. This adaptive, zero-parameter trick could be transferred to other mask-based generative RL models.
- **Empirical proof of "Early Step Value" via $\Delta$HPS**: Using the variation in human preference scores to measure exploration potential provides a diagnostic tool for future credit assignment research.

## Limitations & Future Work
- The current rebalancing factors are somewhat **empirical** ($t_j$, $t_j^2$, $1/F$). Exploring how to upgrade these to a learnable paradigm remains a future task.
- Experiments were conducted primarily on the MMaDA-8B backbone. Generalization across different backbones and tasks (like video discrete diffusion) has not been fully tested.
- Future direction: Parameterizing $\lambda(t_j)$ and $1/F$ as learnable outputs or adaptively adjusting weights using $\Delta$HPS signals.

## Related Work & Insights
- **vs. UniGRPO / MaskGRPO**: These use varying mask ratios but uniform weighting, causing imbalance. RebRL acts as a plug-and-play patch for these frameworks.
- **vs. diffu-GRPO**: While it avoids imbalance via full masking, it suffers from training-inference mismatch. RebRL maintains fidelity to the multi-step process while solving the gradient issue.
- **vs. Credit Assignment in Flow-Matching**: Similar observations have been made in continuous models regarding the inadequacy of uniform credit assignment. RebRL specializes this insight for discrete mask-based DDMs with a token-level solution.

## Rating
- Novelty: ⭐⭐⭐⭐ First to analytically diagnose and solve DDM-RL credit imbalance with zero hyperparameters.
- Experimental Thoroughness: ⭐⭐⭐ Strong results on GenEval/HPS, but limited to a single backbone.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from diagnosis to methodology, well-supported by formulas and visualizations.
- Value: ⭐⭐⭐⭐ Plug-and-play, ~40% speedup, high practical value for RL post-training of mask-based models.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[CVPR 2025\] Generative Multimodal Pretraining with Discrete Diffusion Timestep Tokens](../../CVPR2025/image_generation/generative_multimodal_pretraining_with_discrete_diffusion_timestep_tokens.md)
- [\[CVPR 2026\] Visual Diffusion Models are Geometric Solvers](visual_diffusion_models_are_geometric_solvers.md)
- [\[CVPR 2026\] Seeing What Matters: Visual Preference Policy Optimization for Visual Generation](seeing_what_matters_visual_preference_policy_optimization_for_visual_generation.md)
- [\[CVPR 2026\] Fine-Grained GRPO for Precise Preference Alignment in Flow Models](fine-grained_grpo_for_precise_preference_alignment_in_flow_models.md)

</div>

<!-- RELATED:END -->
