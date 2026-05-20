---
title: >-
  [Paper Note] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs
description: >-
  [ICML 2026][Image Generation][Rectified Flow] This paper targets rectified flow (RF) text-to-image models and proposes PNAPO—an offline preference optimization framework that saves both the "prior noise used for generati…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Rectified Flow"
  - "Diffusion-DPO"
  - "Preference Optimization"
  - "Prior Noise"
  - "Dynamic Regularization"
date: 2026-05-08
content_hash: 3c37e0b832b8232c
---

# Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs

**Conference**: ICML 2026  
**arXiv**: [2605.09433](https://arxiv.org/abs/2605.09433)  
**Code**: None (repository link not released)  
**Area**: Alignment RLHF / Diffusion Models / Text-to-Image  
**Keywords**: Rectified Flow, Diffusion-DPO, Preference Optimization, Prior Noise, Dynamic Regularization

## TL;DR
This paper targets rectified flow (RF) text-to-image models and proposes PNAPO—an offline preference optimization framework that saves both the "prior noise used for generation" and "winner/loser images" as sextuplets. Leveraging the RF straight-line trajectory assumption for trajectory estimation and dynamic regularization scheduling, PNAPO outperforms Diffusion-DPO on SD3-M/FLUX while reducing training compute to 1/12.

## Background & Motivation

**Background**: The mainstream approach for post-training alignment in text-to-image (T2I) is to collect (prompt, winner, loser) triplet preference data, then use RL (DDPO, DPOK) or RL-free DPO-style objectives (Diffusion-DPO, D3PO, etc.) to bias the generator toward the winner. RL-free methods are more popular due to their stability and simplicity.

**Limitations of Prior Work**: Existing preference datasets (Pick-a-Pic, HPDv2, ImageReward, etc.) only store the final images, discarding the "prior noise used to generate the image"—yet diffusion/flow models fundamentally generate images via a trajectory starting from specific noise. Methods like Diffusion-DPO can only approximate the reverse trajectory using independently sampled forward noise, which mismatches the true reverse dynamics, leading to unstable training and inefficient credit assignment.

**Key Challenge**: In standard diffusion models, the reverse trajectory is random and curved, making it intractable to sample the exact reverse path given endpoints. RF is different—the RF training objective is to "straighten" the data-noise coupling into nearly linear trajectories, with the prior noise directly determining the trajectory. Thus, "discarding prior noise" is a more severe loss for RF than for ordinary diffusion models.

**Goal**: (1) Preserve prior noise in preference data; (2) Design a DPO-style objective consistent with RF geometry; (3) Address two longstanding DPO issues: fixed $\beta$ causing weak updates in late training and uniform treatment of all samples.

**Key Insight**: The authors observe a key RF property: $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_T$ is a straight-line interpolation between endpoints. If both $\boldsymbol{x}_0$ and $\boldsymbol{x}_T$ are stored in the dataset, intermediate states can be directly estimated by interpolation, eliminating the need for extra noise addition. This reduces the intractable reverse sampling to a linear interpolation, drastically lowering variance.

**Core Idea**: Extend preference triplets to sextuplets $(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l, \boldsymbol{x}_T^w, \boldsymbol{x}_T^l)$ with a continuous reward difference $\delta r$, estimate intermediate states via RF linear interpolation, and introduce a dynamic $\beta$ scheduled by reward difference and training steps.

## Method

### Overall Architecture
PNAPO is an offline, off-policy, RL-free alignment pipeline with three steps: (1) **Data Construction**—use an RF base model to sample two prior noises per prompt → generate image pairs → score with HPSv2.1 reward model → obtain sextuplets + continuous reward difference $\delta r$; (2) **Trajectory Estimation**—leverage the RF linear property to interpolate intermediate states from stored $(\boldsymbol{x}_0^*, \boldsymbol{x}_T^*)$ endpoints via $\boldsymbol{x}_t = (1-t)\boldsymbol{x}_0 + t\boldsymbol{x}_T$, skipping any resampling; (3) **Optimization**—apply the RF-consistent PNAPO objective with dynamic $\beta(\delta r, n)$ scheduling for LoRA-style updates, freezing the reference model $v_{\text{ref}}$.

### Key Designs

1. **Preference Sextuplets with Prior Noise Tracking**:

    - **Function**: Extend traditional triplets $(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l)$ to sextuplets, attaching $\boldsymbol{x}_T^w, \boldsymbol{x}_T^l$ and reward difference $\delta r$, enabling DPO loss trajectory estimation from endpoint conditions.
    - **Mechanism**: Select 20k high-quality prompts from DiffusionDB (NSFW filtering → Jaccard/CLIP deduplication → 100 KNN cluster resampling). For each prompt, sample two noises with the RF base model → generate two images, score with HPSv2.1 to get $\delta r = r_\theta(\boldsymbol{x}_0^w) - r_\theta(\boldsymbol{x}_0^l)$. Images are sampled by the model itself (off-policy but same model family), ensuring noise and model policy consistency.
    - **Design Motivation**: Traditional datasets discard noise, forcing DPO to resample $\boldsymbol{x}_T^* \sim \mathcal{N}(0, I)$ independently to estimate the reverse process, introducing variance sources mismatched with actual training. By retaining noise, $p_\theta(\boldsymbol{x}_T^* | \boldsymbol{x}_0^*)$ is explicitly preserved, effectively shrinking the decision space from "all possible trajectories" to "the actual trajectory that produced this image".

2. **RF-Consistent Trajectory Estimation and Objective Function**:

    - **Function**: Approximate the intractable $p_\theta(\boldsymbol{x}_{1:T-1} | \boldsymbol{x}_0)$ with $p_\theta(\boldsymbol{x}_T | \boldsymbol{x}_0) q(\boldsymbol{x}_{1:T-1} | \boldsymbol{x}_0, \boldsymbol{x}_T)$, proving this surrogate is tighter for RF.
    - **Mechanism**: After Jensen's inequality and KL decomposition, the loss simplifies to $\mathcal{L}_{\text{PNAPO}}(\theta) = -\mathbb{E}_{(\boldsymbol{c}, \boldsymbol{x}_0^w, \boldsymbol{x}_0^l, \boldsymbol{x}_T^w, \boldsymbol{x}_T^l), t} \log \sigma(-\beta(\boldsymbol{s}_\theta^t(\boldsymbol{x}_0^w, \boldsymbol{x}_T^w, \boldsymbol{c}) - \boldsymbol{s}_\theta^t(\boldsymbol{x}_0^l, \boldsymbol{x}_T^l, \boldsymbol{c})))$, where $\boldsymbol{s}_\theta^t(\boldsymbol{x}_0^*, \boldsymbol{x}_T^*, \boldsymbol{c}) = \|(\boldsymbol{x}_T^* - \boldsymbol{x}_0^*) - v_\theta(\boldsymbol{x}_t^*, t, \boldsymbol{c})\|^2_2 - \|(\boldsymbol{x}_T^* - \boldsymbol{x}_0^*) - v_{\text{ref}}(\boldsymbol{x}_t^*, t, \boldsymbol{c})\|^2_2$, and $\boldsymbol{x}_t^* = (1-t)\boldsymbol{x}_0^* + t\boldsymbol{x}_T^*$. The objective is to make $v_\theta$ more accurate than the reference on winner trajectories and less accurate on loser trajectories.
    - **Design Motivation**: The authors formally prove $D_{KL}(p_\theta(\boldsymbol{x}_T|\boldsymbol{x}_0) q(\boldsymbol{x}_{1:T-1}|\boldsymbol{x}_0, \boldsymbol{x}_T) \| p_\theta(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0)) \leq D_{KL}(q(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0) \| p_\theta(\boldsymbol{x}_{1:T}|\boldsymbol{x}_0))$, showing PNAPO's trajectory approximation is strictly tighter than Diffusion-DPO's forward-noise approximation. Analogous to sparse reward problems in RL, shrinking the decision space directly reduces gradient variance and accelerates training.

3. **Dynamic $\beta$ Scheduling Based on Reward Difference and Training Progress**:

    - **Function**: Allow regularization strength $\beta$ to automatically respond to "sample difficulty" (winner/loser reward gap) and "training phase", alleviating the issue of fixed $\beta$ pulling the model back to the reference in late training.
    - **Mechanism**: $\beta(\delta r, n) = \beta \cdot f(\delta r) \cdot g(n)$, where $f(\delta r) = 2\sigma(\delta r) - 1$ is a monotonically increasing sample controller; $g(n)$ is an annealing factor—kept at 1 for the first $n_1$ steps, then cosine-decayed to $1/2$ between $n_1$ and $n_2$, and held at $1/2$ thereafter. When the margin is negative, increasing $\delta r$ raises $\beta$ to accelerate alignment; after the margin turns positive, the effect reverses for softer updates.
    - **Design Motivation**: Gradient decomposition of $\nabla_\theta \mathcal{L}_{\text{PNAPO}}$ reveals two issues with fixed $\beta$: indiscriminate weighting of all image pairs (ignoring difficulty) and strong regularization pulling the model back to the reference in late training. Dynamic $\beta$ gives higher weight to pairs with large reward gaps ("obviously better"), and allows more deviation from the reference in late training.

### Loss & Training

The core loss is the PNAPO objective. Optimizer: AdamW, learning rate $1\mathrm{e}{-6}$; $\beta=2000$ for FLUX, $\beta=5000$ for SD3-M. 20k prompts × 2 images per prompt, Euler discrete scheduler, 50 steps, guidance scale=1. 8× NVIDIA H800 GPUs.

## Key Experimental Results

### Main Results
Baselines include the original model, SFT, Diffusion-DPO, IPO, and CaPO, all reproduced with identical hyperparameters and model configurations for fairness. Evaluations on HPDv2 (3200 prompts) and OPDv1 (7459 prompts) use PickScore, HPSv2.1, ImageReward, LAION Aesthetic, and CLIP; GenEval is used for object generation alignment.

| Test Set / Model | Metric | Original | DPO | PNAPO | Gain |
|------------------|--------|----------|-----|-------|------|
| OPDv1 SD3-M | HPSv2.1 | 31.96 | 32.39 | 33.09 | +1.13 (vs base) |
| OPDv1 FLUX | HPSv2.1 | 30.74 | 30.79 | 32.10 | +1.36 (vs base) |
| OPDv1 FLUX | ImageReward | 1.202 | 1.209 | 1.238 | +0.036 |
| OPDv1 FLUX | Aesthetic | 6.550 | 6.548 | 6.692 | +0.142 |
| GenEval SD3-M | Overall | 0.68 | — | 0.73 | +7.4% relative |
| GenEval FLUX | Overall | 0.65 | 0.66 | 0.69 | +6.2% relative |
| HPSv2.1 Win Rate FLUX | PNAPO vs DPO | — | — | 84.6% | — |

### Ablation Study
Training compute comparison (NVIDIA H800 GPU-Hours):

| Model | Diffusion-DPO | PNAPO | Savings |
|-------|---------------|-------|---------|
| SD3-M | ~249.6 | ~20.8 | 12× |
| FLUX | ~422.4 | ~35.2 | 12× |

User study (10 participants, 20 image pairs, PNAPO-FLUX vs baselines):

| Evaluation Dimension | PNAPO Preference Rate |
|---------------------|----------------------|
| Overall Preference | 56% |
| Visual Appeal | 72% |
| Text-Image Alignment | 52% |

### Key Findings
- **Dual Gains in Quality and Compute**: Outperforms Diffusion-DPO on all metrics while reducing GPU time to 1/12, validating that "tighter trajectory estimation" directly improves training efficiency.
- **Background Blur Mitigated**: The characteristic background blur artifact of FLUX is significantly reduced under PNAPO; qualitative results show improved text rendering and composition.
- **Cross-Architecture Generalization**: Consistent improvements on two different RF backbones (SD3-M / FLUX) indicate the method relies on RF geometry rather than specific models.
- **CLIP Text-Image Alignment**: FLUX improves from 35.97 to 36.89, demonstrating that dynamic $\beta$ does not sacrifice text alignment for aesthetics.

## Highlights & Insights
- **Paradigm Shift from "Discarding Noise" to "Tracking Noise"**: Previous preference datasets only stored images; this work points out that for RF, noise is part of trajectory identity—a long-overlooked "free lunch". Simply storing the noise tensor during data construction yields a 12× compute saving, with extremely high cost-effectiveness.
- **Geometrically Consistent Approximation with Theoretical Guarantees**: The authors rigorously prove via KL chain inequalities that PNAPO's trajectory approximation is tighter than Diffusion-DPO's, elevating "better" from empirical observation to theoretical result—a rare solidity.
- **Two Independent Factors in Dynamic $\beta$ Design**: $f(\delta r)$ controls sample difficulty, $g(n)$ controls training progress; the decoupling is clean and can be independently combined with other DPO variants (D3PO, IPO, Diffusion-KTO, etc.).
- **Engineering Friendliness of Offline RL-Free**: Compared to GRPO-style online RL methods, PNAPO only requires a single data collection and then stable offline training, making it more suitable for production environments with compute/scheduling constraints.

## Limitations & Future Work
- **Dependence on Reward Model**: Using HPSv2.1 as a pseudo-human labeler amplifies the reward model's biases and blind spots; the paper does not discuss reward hacking risks.
- **Covers Only RF Models**: The core mechanism (linear interpolation) strictly relies on RF's straight-line trajectories and cannot be directly transferred to pure DDPM/DDIM; the authors explicitly limit the scope to RF.
- **Small Data Scale**: 20k prompts is relatively small for T2I preference datasets; stability of dynamic $\beta$ scheduling at 100k+ scale remains to be validated.
- **No Comparison with Online RL**: The paper positions itself as an RL-free supplement but lacks fair compute comparisons with GRPO-series methods, so the true offline/online benefit gap is unquantified.
- **Manual Tuning of $n_1, n_2$**: The two thresholds for cosine annealing are set empirically; different models/datasets require retuning, and there is no adaptive scheme.

## Related Work & Insights
- **vs Diffusion-DPO (Wallace 2024)**: Core idea is similar but uses forward noise to approximate reverse trajectories; PNAPO proves tighter trajectory approximation and 12× faster training on RF, explicitly leveraging RF geometry.
- **vs D3PO (Yang 2024)**: D3PO estimates preference at each reverse step iteratively, which is computationally expensive; PNAPO skips the reverse process via interpolation, achieving higher efficiency.
- **vs SPO / InPO / SmPO**: These methods align preferences throughout denoising, requiring DDIM Inversion; PNAPO directly uses stored noise end-to-end, making engineering simpler.
- **vs Diffusion-NPO / Self-NPO**: These train "negative sample models" for guidance from a CFG perspective; PNAPO is a positive update, complementary in approach and can be combined.
- **vs GRPO Series (Online RL)**: High alignment but requires extensive online sampling and fine-tuning; PNAPO adopts a "single offline training" route, more practical under compute/engineering constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "store noise" data structure change seems simple, but combined with RF geometric analysis, it directly targets the variance source of Diffusion-DPO—a beautiful idea.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual-model, dual-dataset, multi-metric + user study + GPU-Hours comparison; only missing comparison with online RL methods and large-scale data validation.
- Writing Quality: ⭐⭐⭐⭐ Clear derivations, seamless flow from motivation to objective to dynamic $\beta$; notation is somewhat dense.
- Value: ⭐⭐⭐⭐⭐ For RF-based T2I post-training, this is a plug-and-play method that saves an order of magnitude in compute—high engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/llm_alignment/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[AAAI 2026\] On the Exponential Convergence for Offline RLHF with Pairwise Comparisons](../../AAAI2026/llm_alignment/on_the_exponential_convergence_for_offline_rlhf_with_pairwise_comparisons.md)
- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](../../CVPR2025/llm_alignment/debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[ICML 2026\] TUR-DPO: Topology- and Uncertainty-Aware Direct Preference Optimization](tur-dpo_topology-_and_uncertainty-aware_direct_preference_optimization.md)
- [\[NeurIPS 2025\] Rethinking Direct Preference Optimization in Diffusion Models](../../NeurIPS2025/llm_alignment/rethinking_direct_preference_optimization_in_diffusion_models.md)

</div>

<!-- RELATED:END -->
