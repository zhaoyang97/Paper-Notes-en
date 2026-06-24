---
title: >-
  [Paper Note] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences
description: >-
  [ICML 2025][Image Generation][Preference Optimization] SmPO-Diffusion is proposed, which replaces binary preference labels with smoothed preference modeling and replaces forward noising estimation with ReNoise Inversion. It achieves SOTA performance in T2I diffusion model preference alignment while significantly reducing training costs (6.5 times faster than DPO and 26 times faster than KTO).
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Preference Optimization"
  - "Diffusion Model Alignment"
  - "Smoothed Preference Distribution"
  - "ReNoise Inversion"
  - "DPO"
date: 2026-05-08
content_hash: cb62073a9f5f154c
---

# Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences

**Conference**: ICML 2025  
**arXiv**: [2506.02698](https://arxiv.org/abs/2506.02698)  
**Code**: [Project Page](https://jaydenlyh.github.io/SmPO-project-page/)  
**Area**: LLM Alignment/RLHF  
**Keywords**: Preference Optimization, Diffusion Model Alignment, Smoothed Preference Distribution, ReNoise Inversion, DPO

## TL;DR

SmPO-Diffusion is proposed, which replaces binary preference labels with smoothed preference modeling and replaces forward noising estimation with ReNoise Inversion. It achieves SOTA performance in T2I diffusion model preference alignment while significantly reducing training costs (6.5 times faster than DPO and 26 times faster than KTO).

## Background & Motivation

Current preference alignment methods for T2I diffusion models (such as Diffusion-DPO) suffer from two core problems:

**Coarse preference labels**: Existing datasets employ binary preference annotations (winner/loser) for image pairs, ignoring the fact that "aesthetics vary by individual"—forcing a win/loss annotation on two images of similar quality leads to over-optimization.

**Inaccurate optimization objective estimation**: When estimating the trajectory preference distribution of diffusion models, Diffusion-DPO replaces the actual sampling process $p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)$ with the forward noising process $q(\mathbf{x}_{1:T}|\mathbf{x}_0)$. Since noise is randomly sampled rather than being image-dependent, this leads to objective misalignment and low training efficiency.

The core insight of this work is that in the diffusion model framework, existing methods suffer from significant inaccuracies at both levels of **preference modeling** and **optimization estimation**, which need to be resolved simultaneously.

## Method

### Overall Architecture

SmPO-Diffusion consists of two complementary improvement modules:

- **Step 1 — Smoothed Preference Modeling**: A reward model (PickScore) is used to automatically calculate smoothed preference labels for all image pairs, replacing manually annotated binary labels.
- **Step 2 — ReNoise Inversion Optimization**: ReNoise Inversion is used to estimate the sampling trajectories of the diffusion model, replacing the estimation based on forward noising in the original DPO to achieve more accurate optimization objectives.

The final loss function only introduces an adaptive weight factor $(2\alpha - \gamma)$ on top of standard Diffusion-DPO, formulated as a simple modification to the DPO loss.

### Key Designs

1. **Smoothed Preference Distribution**

   **Mechanism**: A weighted average mixture distribution is used to replace the binary distribution. When the preferences for two images are similar, the loss function naturally approaches zero, avoiding over-optimization caused by forcing a gap.

   **Specific Approach**: Assume the probability density of winner and loser is in a weighted mixture form $\tilde{p}(\mathbf{x}_0^w|\mathbf{c}) \propto p(\mathbf{x}_0^w|\mathbf{c})^\alpha \cdot p(\mathbf{x}_0^l|\mathbf{c})^{\gamma-\alpha}$, where $\alpha$ is the weight factor and $\gamma$ is the sensitivity factor. After substituting this into the DPO target, the entire smoothed preference modeling is equivalent to multiplying the original DPO loss by an adaptive coefficient $(2\alpha - \gamma)$. When $\alpha = 1, \gamma = 1$, it degenerates to standard DPO.

   **Design Motivation**: The weighted average plays a smoothing role, effectively adjusting the scale of likelihood—weaker constraints for image pairs with closer preferences, and stronger constraints for image pairs with larger preference gaps.

2. **Reward Model-Driven Smoothed Label Generation**

   **Mechanism**: Without manual annotation, the PickScore reward model is utilized to automatically generate smoothed preference labels.

   **Specific Approach**: Define the weight-to-sensitivity ratio $\alpha/\gamma$ as the probability of being the winner:
    $\frac{\alpha}{\gamma} = \frac{\exp(r'(\mathbf{x}_0^w, \mathbf{c}))}{\exp(r'(\mathbf{x}_0^w, \mathbf{c})) + \exp(r'(\mathbf{x}_0^l, \mathbf{c}))}$
   where $r'$ is the global normalization (min-max normalization) of the reward scores, and then the probability value is obtained through Softmax. The sensitivity $\gamma$ is fixed as a constant to control the fluctuation range.

   **Design Motivation**: AI reward models are highly aligned with human preferences (a consensus in the RLHF field) and can serve as a reliable substitute for expert ratings or human votes without additional manual annotation costs.

3. **ReNoise Inversion Trajectory Estimation**

   **Mechanism**: ReNoise Inversion is used instead of forward noising to estimate the sampling trajectory $p_\theta(\mathbf{x}_{1:T}|\mathbf{x}_0)$ of the diffusion model, eliminating the objective misalignment issue.

   **Specific Approach**: The specific approach consists of two steps:
    - **Step A — Few-step DDIM Inversion** ($\le 10$ steps): Starting from $\mathbf{x}_0$, an approximate estimation of $\hat{\mathbf{x}}_t$ is iteratively obtained through the DDIM Inversion formula.
    - **Step B — One-step ReNoise Correction**: Based on $\hat{\mathbf{x}}_t$, a one-step correction is made using the noise prediction of the current model at $\hat{\mathbf{x}}_t$, yielding a more accurate $\tilde{\mathbf{x}}_t$.

   The corrected score function becomes: $\tilde{s}_\theta^t = \|\tau_t - \epsilon_\theta^t(\tilde{\mathbf{x}}_t, \mathbf{c})\|^2 - \|\tau_t - \epsilon_{\text{ref}}^t(\tilde{\mathbf{x}}_t, \mathbf{c})\|^2$, where $\tau_t = (\tilde{\mathbf{x}}_t - \sqrt{\bar\alpha_t}\mathbf{x}_0)/\sqrt{1-\bar\alpha_t}$.

   **Design Motivation**: The random Gaussian noise used in forward noising is independent of the image, whereas the latent variables obtained via Inversion are highly correlated with the image. This allows for a more accurate estimation of the optimization objective, thereby significantly improving training efficiency.

### Loss & Training

Final loss function:
$$\mathcal{L}(\theta) = -\mathbb{E}_{t,\mathcal{D}} \log \sigma\left(-(2\alpha - \gamma)\beta \left(\tilde{s}_\theta^t(\mathbf{x}_0^w, \mathbf{c}) - \tilde{s}_\theta^t(\mathbf{x}_0^l, \mathbf{c})\right)\right)$$

Training configurations:
- Dataset: Pick-a-Pic v2 (851K data pairs, 59K unique prompts)
- Optimizer: AdamW for SD1.5, Adafactor for SDXL
- 8×A800 GPUs, batch size=1/GPU, 128 gradient accumulation steps $\rightarrow$ effective batch size=1024
- SD1.5: $\beta=2000$; SDXL: $\beta=5000$
- DDIM Inversion steps=9, CFG=1, $\gamma=10$

## Key Experimental Results

### Main Results

| Model | PickScore↑ | HPSv2.1↑ | ImReward↑ | Aesthetic↑ | GPU Hours↓ |
|------|-----------|----------|-----------|-----------|-----------|
| Base-SDXL | 22.75 | 28.45 | 0.881 | 6.114 | - |
| DPO-SDXL | 23.13 | 30.06 | 1.184 | 6.112 | ~976 |
| MaPO-SDXL | 22.81 | 29.11 | 1.224 | 6.309 | ~834 |
| **SmPO-SDXL** | **23.62** | **32.53** | **1.331** | 6.264 | **~151** |
| Base-SD1.5 | 20.83 | 23.61 | -0.078 | 5.390 | - |
| DPO-SD1.5 | 21.29 | 25.11 | 0.195 | 5.530 | ~205 |
| KTO-SD1.5 | 21.54 | 28.28 | 0.706 | 5.692 | ~1056 |
| **SmPO-SD1.5** | **22.08** | **29.31** | **0.885** | **5.831** | **~41** |

Median score on HPDv2 test set. SmPO-SDXL achieves an 86.7% win-rate against DPO-SDXL on HPSv2.1, with only 15.5% of the training time.

### Ablation Study

| Configuration | PickScore↑ | HPSv2.1↑ | ImReward↑ | Description |
|------|-----------|----------|-----------|------|
| DPO (baseline) | 21.29 | 25.11 | 0.195 | Standard Diffusion-DPO |
| +DDIM Inversion | 21.72 | 28.71 | 0.761 | Replacing forward noising with Inversion |
| +ReNoise | 21.87 | 29.01 | 0.778 | Adding one-step ReNoise correction |
| +Smoothed Pref | **22.08** | **29.31** | **0.885** | Adding smoothed preference modeling (Full Method) |

| Hyperparameter Ablation | Optimal Value | Key Points |
|-----------|--------|------|
| Inversion Steps | 9 steps (balancing quality/efficiency) | 19 steps is optimal but doubles GPU hours |
| CFG during Inversion | 1 | DDIM Inversion is sensitive to prompts |
| Sensitivity $\gamma$ | 10 | Too small $\rightarrow$ insensitive to reward; too large $\rightarrow$ over-optimization |
| Regularization $\beta$ | 2000 | Too small $\rightarrow$ degenerates to pure reward model; too large $\rightarrow$ overly strong KL restriction |

### Key Findings

1. The progressive addition of the three modules (DDIM Inversion $\rightarrow$ ReNoise $\rightarrow$ Smoothed Preference) yields consistent and significant improvements, validating the correctness of the two core assumptions.
2. **The training efficiency is exceptionally high**: SmPO-SD1.5 requires only 41.3 GPU-hours, which is 1/26 of KTO and 1/5 of DPO.
3. Using PickScore as the reward model yields the best results because it can be regarded as pseudolabels for the Pick-a-Pic dataset, effectively acting as data cleaning.
4. Models trained with SmPO can be directly applied to ControlNet conditional generation (canny/depth map) without additional training.

## Highlights & Insights

1. **Extremely elegant improvements**: The entire method is equivalent to multiplying the DPO loss by an adaptive coefficient and replacing the forward noising process with Inversion. Code modifications are minimal, but the effect is outstanding.
2. **Win-win in both training efficiency and performance**: Rather than trading massive resources for performance, both aspects are simultaneously improved through more accurate modeling and estimation.
3. **An elegant way of eliminating label noise**: Replacing hard labels with soft labels generated by a reward model requires zero manual annotation while reducing over-optimization caused by label noise.
4. **A novel application for Inversion**: Innovatively applying DDIM Inversion/ReNoise technologies (originally from the image editing domain) to trajectory estimation in preference optimization.

## Limitations & Future Work

1. **Dataset bias**: The method relies on the Pick-a-Pic v2 dataset, which may contain social biases such as gender stereotypes, causing the model to generate overly feminized images for neutral prompts.
2. **Offline learning**: The current method is offline. It can be integrated into online learning schemes for continuous performance improvement in the future.
3. **Dependency on reward models**: The quality of smoothed labels completely depends on the quality of PickScore. Swapping in other reward models leads to performance fluctuations (Table 7).
4. **Inversion computational overhead**: Although overall much faster than DPO, increasing the number of Inversion steps still linearly increases the training time.

## Related Work & Insights

- **Diffusion-DPO** (Wallace et al., 2023): First to apply DPO to diffusion models by optimizing the upper bound of the joint distribution of trajectories, serving as the direct baseline of this work.
- **MaPO**: Jointly maximizes the likelihood gap between preferred and non-preferred sets without requiring a reference model.
- **DDIM-InPO** (Lu et al., 2025): Uses DDIM Inversion to align specific latent variables, which is related to this work's idea but only uses Inversion without ReNoise.
- **ReNoise Inversion** (Garibi et al., 2024): Originally used for precise Inversion in image editing, this work innovatively introduces it into preference optimization.
- Insights for LLM Alignment: The concept of soft labels combined with more accurate distribution estimation can be generalized to LLM DPO.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The two individual improvements are not entirely new (soft labels, Inversion), but their combination applied to diffusion model preference optimization is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Dual models (SD1.5/SDXL), five evaluation metrics, comprehensive ablation studies, and conditional generation tests.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, complete derivations, and standardized mathematical notation.
- **Value**: ⭐⭐⭐⭐⭐ — 5-26$\times$ training efficiency improvement + SOTA performance, highly valuable for practical applications.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Fine-Grained Attribution: Instance-Aware Preference Optimization for Aligning Diffusion Models](../../CVPR2026/image_generation/towards_fine-grained_attribution_instance-aware_preference_optimization_for_alig.md)
- [\[CVPR 2025\] InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment](../../CVPR2025/image_generation/inpo_inversion_preference_optimization_diffusion_alignment.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2025\] Calibrated Multi-Preference Optimization for Aligning Diffusion Models](../../CVPR2025/image_generation/calibrated_multi-preference_optimization_for_aligning_diffusion_models.md)
- [\[CVPR 2025\] Curriculum Direct Preference Optimization for Diffusion and Consistency Models](../../CVPR2025/image_generation/curriculum_direct_preference_optimization_for_diffusion_and_consistency_models.md)

</div>

<!-- RELATED:END -->
