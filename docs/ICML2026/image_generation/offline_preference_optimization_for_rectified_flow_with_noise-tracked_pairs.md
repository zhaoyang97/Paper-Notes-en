---
title: >-
  [Paper Note] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs
description: >-
  [ICML 2026][Image Generation][Rectified Flow] This paper proposes PNAPO for Rectified Flow (RF) text-to-image models—an offline preference optimization framework that saves the "prior noise used during generation" alongs…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Rectified Flow"
  - "Diffusion-DPO"
  - "Preference Optimization"
  - "Prior Noise"
  - "Dynamic Regularization"
date: 2026-05-08
content_hash: f2150e4b28f610ad
---

# Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs

**Conference**: ICML 2026  
**arXiv**: [2605.09433](https://arxiv.org/abs/2605.09433)  
**Code**: None (Repository link not disclosed in the paper)  
**Area**: Alignment RLHF / Diffusion Models / Text-to-Image  
**Keywords**: Rectified Flow, Diffusion-DPO, Preference Optimization, Prior Noise, Dynamic Regularization

## TL;DR
This paper proposes PNAPO for Rectified Flow (RF) text-to-image models—an offline preference optimization framework that saves the "prior noise used during generation" alongside "winner/loser images" as sextuplets. By leveraging the RF linear trajectory hypothesis for trajectory estimation and dynamic regularization coefficient scheduling, it achieves significant performance gains on SD3-M/FLUX while reducing training compute to 1/12 compared to Diffusion-DPO.

## Background & Motivation

**Background**: The mainstream approach for post-training alignment in text-to-image (T2I) generation involves collecting (prompt, winner, loser) preference triplets and utilizing RL (DDPO, DPOK) or RL-free DPO-style objectives (Diffusion-DPO, D3PO, etc.) to bias the generator toward winners. RL-free methods are generally preferred for their stability and simplicity.

**Limitations of Prior Work**: Existing preference datasets (Pick-a-Pic, HPDv2, ImageReward, etc.) only save the final images, discarding the "prior noise used to generate the image." However, generation in diffusion/flow models is inherently a trajectory process starting from a specific noise. Methods like Diffusion-DPO can only approximate the reverse trajectory using independently sampled forward noise, which mismatches the true reverse dynamics, leading to unstable training and inefficient credit assignment.

**Key Challenge**: In standard diffusion models, reverse trajectories are stochastic and curved, making it intractable to sample an exact reverse path for a given pair of endpoints. However, RF is different—the training objective of RF is to "straighten" the data-noise coupling into near-linear trajectories. Thus, "discarding the prior noise" in RF is a more severe loss than in ordinary diffusion models.

**Goal**: (1) Enable preference data to retain prior noise; (2) Design a DPO-style objective consistent with RF geometry; (3) Solve two legacy issues in DPO: weak updates in late-stage training due to fixed $\beta$ and treating all samples identically.

**Key Insight**: The authors observe a key property of RF: $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_T$ is a linear interpolation between endpoints. If both $\boldsymbol{x}_0$ and $\boldsymbol{x}_T$ are stored in the dataset, intermediate states can be directly estimated via interpolation without additional noise injection. This simplifies intractable reverse sampling into a linear interpolation, drastically reducing variance.

**Core Idea**: Extend preference triplets into sextuplets $(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l, \boldsymbol{x}_T^w, \boldsymbol{x}_T^l)$ with a continuous reward difference $\delta r$. Use RF linear interpolation for intermediate state estimation, combined with a dynamic $\beta$ scheduled by both reward differences and training steps.

## Method

### Overall Architecture
PNAPO is an offline, off-policy RL-free alignment pipeline consisting of three steps: (1) **Data Construction**: Use a base RF model to sample two prior noises per prompt → generate image pairs → score with HPSv2.1 reward model → obtain sextuplets + continuous reward difference $\delta r$; (2) **Trajectory Estimation**: Leverage RF linearity to interpolate intermediate states directly from the stored $(\boldsymbol{x}_0^*, \boldsymbol{x}_T^*)$ endpoint pairs using $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_T$, skipping any resampling; (3) **Optimization**: Perform LoRA-style updates using the RF-consistent PNAPO objective + dynamic $\beta(\delta r, n)$ scheduling, with a frozen reference model $v_{\text{ref}}$.

### Key Designs

1.  **Prior Noise-Tracked Preference Sextuplets**:
    - **Function**: Expands traditional triplets $(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l)$ into sextuplets including $\boldsymbol{x}_T^w, \boldsymbol{x}_T^l$ and reward difference $\delta r$, allowing DPO loss trajectory estimation to be conditioned on endpoints.
    - **Mechanism**: Select 20k high-quality prompts from DiffusionDB (NSFW filtering → Jaccard/CLIP deduplication → 100 KNN cluster resampling). For each prompt, sample two noises using the base RF model → two images. HPSv2.1 scoring provides $\delta r = r_\theta(\boldsymbol{x}_0^w) - r_\theta(\boldsymbol{x}_0^l)$. Note that images are generated by the model itself (off-policy but same model family), ensuring noise and policy consistency.
    - **Design Motivation**: Traditional datasets discard noise, forcing DPO to resample from independent $\boldsymbol{x}_T^* \sim \mathcal{N}(0, I)$ to estimate the reverse process, introducing a source of variance mismatched with actual training. By keeping the noise, $p_\theta(\boldsymbol{x}_T^* | \boldsymbol{x}_0^*)$ is explicitly preserved, effectively narrowing the decision space from "all possible trajectories" to "the specific trajectory that produced this image."

2.  **RF-Consistent Trajectory Estimation and Objective**:
    - **Function**: Approximates the intractable $p_\theta(\boldsymbol{x}_{1:T-1} | \boldsymbol{x}_0)$ with $p_\theta(\boldsymbol{x}_T | \boldsymbol{x}_0) q(\boldsymbol{x}_{1:T-1} | \boldsymbol{x}_0, \boldsymbol{x}_T)$, proving this approximation is a tighter surrogate for RF.
    - **Mechanism**: After Jensen's inequality and KL decomposition, the loss simplifies to $\mathcal{L}_{\text{PNAPO}}(\theta) = -\mathbb{E}_{(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l, \boldsymbol{x}_T^w, \boldsymbol{x}_T^l), t} \log \sigma(-\beta(\boldsymbol{s}_\theta^t(\boldsymbol{x}_0^w, \boldsymbol{x}_T^w, \boldsymbol{c}) - \boldsymbol{s}_\theta^t(\boldsymbol{x}_0^l, \boldsymbol{x}_T^l, \boldsymbol{c})))$, where $\boldsymbol{s}_\theta^t(\boldsymbol{x}_0^*, \boldsymbol{x}_T^*, \boldsymbol{c}) = \|(\boldsymbol{x}_T^* - \boldsymbol{x}_0^*) - v_\theta(\boldsymbol{x}_t^*, t, \boldsymbol{c})\|^2_2 - \|(\boldsymbol{x}_T^* - \boldsymbol{x}_0^*) - v_{\text{ref}}(\boldsymbol{x}_t^*, t, \boldsymbol{c})\|^2_2$, with $\boldsymbol{x}_t^* = (1-t)\boldsymbol{x}_0^* + t\boldsymbol{x}_T^*$. The objective essentially makes $v_\theta$ more accurate than ref on winner trajectories and worse on loser trajectories.
    - **Design Motivation**: The authors formally prove $D_{KL}(p_\theta(\boldsymbol{x}_T|\boldsymbol{x}_0) q(\boldsymbol{x}_{1:T-1}|\boldsymbol{x}_0, \boldsymbol{x}_T) \| p_\theta(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)) \leq D_{KL}(q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) \| p_\theta(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0))$, showing PNAPO's trajectory approximation is strictly superior to Diffusion-DPO's forward noise approximation. Analogous to sparse reward problems in RL, reducing the decision space directly lowers gradient variance and accelerates training.

3.  **Dynamic $\beta$ Scheduling Based on Reward Difference and Training Progress**:
    - **Function**: Allows the regularization strength $\beta$ to automatically respond to "sample difficulty" (winner/loser reward gap) and "training stage," mitigating the issue where a fixed $\beta$ pulls the model back to the reference too strongly in later stages.
    - **Mechanism**: $\beta(\delta r, n) = \beta \cdot f(\delta r) \cdot g(n)$, where $f(\delta r) = 2\sigma(\delta r) - 1$ is a sample controller monotonically increasing to 1; $g(n)$ is an annealing factor—maintains 1 for the first $n_1$ steps, decays to $1/2$ via cosine between $n_1$ and $n_2$, and stays at $1/2$ thereafter. When the margin is negative, increasing $\delta r$ raises $\beta$ to accelerate alignment; after the margin turns positive, the effect reverses for gentler updates.
    - **Design Motivation**: Through gradient decomposition of $\nabla_\theta \mathcal{L}_{\text{PNAPO}}$, the authors find that fixed $\beta$ has two issues: uniform weighting of all image pairs (ignoring difficulty) and strong late-stage regularization. Dynamic $\beta$ gives pairs with large reward differences ("obviously better") higher weight and allows more deviation from ref in later training stages.

### Loss & Training
The core loss is the PNAPO objective function. Optimizer: AdamW, learning rate $1\mathrm{e}{-6}$; $\beta=2000$ for FLUX, $\beta=5000$ for SD3-M. 20k prompts × 2 images per prompt, Euler discrete scheduler, 50 steps, guidance scale=1. Hardware: 8× NVIDIA H800 GPUs.

## Key Experimental Results

### Main Results
Baselines include the original model, SFT, Diffusion-DPO, IPO, and CaPO, all reproduced with identical hyperparameters and configurations. Evaluation on HPDv2 (3200 prompts) and OPDv1 (7459 prompts) includes PickScore, HPSv2.1, ImageReward, LAION Aesthetic, and CLIP; GenEval is used for object generation alignment.

| Test Set / Model | Metric | Base Model | DPO | PNAPO | Gain |
|--------------|------|--------|-----|-------|------|
| OPDv1 SD3-M | HPSv2.1 | 31.96 | 32.39 | 33.09 | +1.13 (vs base) |
| OPDv1 FLUX | HPSv2.1 | 30.74 | 30.79 | 32.10 | +1.36 (vs base) |
| OPDv1 FLUX | ImageReward | 1.202 | 1.209 | 1.238 | +0.036 |
| OPDv1 FLUX | Aesthetic | 6.550 | 6.548 | 6.692 | +0.142 |
| GenEval SD3-M | Overall | 0.68 | — | 0.73 | +7.4% rel. |
| GenEval FLUX | Overall | 0.65 | 0.66 | 0.69 | +6.2% rel. |
| HPSv2.1 Win Rate FLUX | PNAPO vs DPO | — | — | 84.6% | — |

### Ablation Study
Training compute comparison (NVIDIA H800 GPU-Hours):

| Model | Diffusion-DPO | PNAPO | Saving |
|------|--------------|-------|------|
| SD3-M | ~249.6 | ~20.8 | 12× |
| FLUX | ~422.4 | ~35.2 | 12× |

User study (10 participants, 20 image pairs, PNAPO-FLUX vs baselines):

| Evaluation Dimension | PNAPO Preference Rate |
|---------|------------|
| Overall Preference | 56% |
| Visual Appeal | 72% |
| Text-Image Alignment | 52% |

### Key Findings
- **Win-Win in Quality and Compute**: While outperforming Diffusion-DPO on all metrics, it reduces GPU time to 1/12, validating the efficiency gains from "tighter trajectory estimation."
- **Background Blur Alleviated**: FLUX's signature background blur artifacts are significantly reduced under PNAPO. Qualitative results also show improved text rendering and composition.
- **Cross-Architecture Generalization**: Consistent improvements across two different RF architectures (SD3-M / FLUX) indicate the method relies on RF geometric properties rather than specific models.
- **CLIP Text-Image Alignment**: Improved from 35.97 to 36.89 on FLUX, proving dynamic $\beta$ does not sacrifice text alignment for aesthetics.

## Highlights & Insights
- **Paradigm Shift from "Discarding Noise" to "Tracking Noise"**: Previous preference datasets saved only images. This paper identifies that noise is part of the trajectory identity for RF—a long-overlooked "free lunch." Simply storing a noise tensor during data construction yields a 10x+ compute saving.
- **Geometric Consistency with Theoretical Guarantees**: The authors use KL chain inequalities to strictly prove that PNAPO's trajectory approximation is tighter than Diffusion-DPO's, grounding "better" results in theory rather than just empirical observation.
- **Decoupled Dynamic $\beta$ Factor Design**: $f(\delta r)$ handles sample difficulty while $g(n)$ handles training progress. This clean decoupling allows independent integration into other DPO variants (D3PO, IPO, Diffusion-KTO could all benefit).
- **Engineering Friendliness of Offline RL-free Pipelines**: Compared to online RL methods like GRPO, PNAPO requires only a single data collection phase followed by stable offline training, making it more practical for production environments with compute/scheduling constraints.

## Limitations & Future Work
- **Dependency on Reward Models**: Using HPSv2.1 as a pseudo-human annotator risks amplifying reward model biases and blind spots; the paper does not discuss reward hacking risks.
- **Restricted to RF Models**: The core mechanism (linear interpolation) strictly depends on RF trajectory linearity and cannot be directly migrated to pure DDPM/DDIM; the authors explicitly limit the scope to RF.
- **Small Data Scale**: 20k prompts is relatively small for T2I preference datasets. Whether dynamic $\beta$ scheduling remains stable when scaled to 100k+ samples requires verification.
- **Lack of Online RL Comparison**: Positioned as an RL-free supplement, but lacks a head-to-head fair compute comparison with GRPO-style methods, leaving the true gap between offline and online gains unquantified.
- **Manual Hyperparameters for $n_1, n_2$**: Thresholds for cosine annealing are set empirically. Different models/datasets may require retuning without an adaptive scheme.

## Related Work & Insights
- **vs Diffusion-DPO (Wallace 2024)**: Similar logic but uses forward diffusion to approximate reverse trajectories; PNAPO proves tighter trajectory approximation on RF, is 12x faster, and explicitly exploits RF geometry.
- **vs D3PO (Yang 2024)**: D3PO uses iterative reverse processes to estimate step-wise preference, which is computationally expensive; PNAPO uses interpolation to skip the reverse process for higher efficiency.
- **vs SPO / InPO / SmPO**: These methods align preferences throughout the denoising process, requiring DDIM Inversion; PNAPO is end-to-end using stored noise, making it simpler to implement.
- **vs Diffusion-NPO / Self-NPO**: Trains "negative sample models" from a CFG perspective for guidance; PNAPO is a positive update, offering a complementary approach that could be combined.
- **vs GRPO series (Online RL)**: High alignment but requires massive online sampling and fine-tuning; PNAPO takes the "sample once, train offline" route, making it more practical under compute/engineering constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The modification of the data structure to "store noise" is simple yet strikes the variance source of Diffusion-DPO through RF geometric analysis—a elegant idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual models, two datasets, multiple metrics + user study + GPU-Hours comparison. The only missing pieces are comparisons with online RL and larger-scale data validation.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, from motivation to objective to dynamic $\beta$. Formula notation is slightly dense.
- Value: ⭐⭐⭐⭐⭐ A plug-and-play method that saves an order of magnitude in compute for RF-based T2I post-training, offering high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Straighten Viscous Rectified Flow via Noise Optimization](../../ICCV2025/image_generation/straighten_viscous_rectified_flow_via_noise_optimization.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[ICML 2026\] Support-Proximity Augmented Diffusion Estimation for Offline Black-Box Optimization](support-proximity_augmented_diffusion_estimation_for_offline_black-box_optimizat.md)
- [\[ICLR 2026\] Flow Matching with Injected Noise for Offline-to-Online Reinforcement Learning](../../ICLR2026/image_generation/flow_matching_with_injected_noise_for_offline-to-online_reinforcement_learning.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](../../NeurIPS2025/image_generation/guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)

</div>

<!-- RELATED:END -->
