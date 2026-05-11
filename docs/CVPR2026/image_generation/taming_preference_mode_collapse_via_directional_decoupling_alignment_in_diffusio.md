---
title: >-
  [Paper Note] Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning
description: >-
  [CVPR 2026][Image Generation][preference mode collapse] This paper proposes D2-Align, a framework that learns a directional correction vector in the reward model's embedding space to debias reward signals…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "preference mode collapse"
  - "RLHF alignment"
  - "diffusion models"
  - "reward correction"
  - "generation diversity"
date: 2026-05-08
content_hash: 6208a32e276f35ca
---

# Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning

**Conference**: CVPR 2026
**arXiv**: [2512.24146](https://arxiv.org/abs/2512.24146)
**Code**: N/A
**Area**: Image Generation
**Keywords**: preference mode collapse, RLHF alignment, diffusion models, reward correction, generation diversity

## TL;DR

This paper proposes D2-Align, a framework that learns a directional correction vector in the reward model's embedding space to debias reward signals, addressing preference mode collapse (PMC) in RLHF-aligned diffusion models — a phenomenon where over-optimization of rewards leads to severe degradation in generation diversity. DivGenBench is also introduced as a benchmark for quantitative diversity evaluation.

## Background & Motivation

RLHF has been widely adopted to align T2I diffusion models with human preferences, but existing methods produce significant side effects while pursuing high reward scores:

- **Preference Mode Collapse (PMC)**: Models converge to narrow modes favored by the reward model — exhibiting homogeneous styles, excessive over-rendering/overexposure, and identity-collapsed faces — resulting in severe degradation of generation diversity. This represents reward hacking manifested along the diversity dimension.
- **Lack of Diversity Evaluation**: Existing work primarily focuses on reward hacking in terms of quality, with insufficient attention to the collapse of generation diversity. Unlike fidelity, diversity lacks standardized evaluation metrics.
- **Existing Countermeasures Only Adjust Magnitude, Not Direction**:
  - KL regularization in Flow-GRPO requires extensive manual tuning and increases training overhead.
  - Multi-reward ensembling in DanceGRPO is sensitive to weight selection and exhibits training instability.
  - These methods fundamentally only regulate the **magnitude** of rewards without correcting the intrinsic bias **direction** of the reward model.
- **Root Cause Hypothesis**: Reward models such as HPS-v2.1 harbor intrinsic preference biases, and the optimization process naturally drives models to overfit these biases, resulting in distributional collapse.

## Method

### Overall Architecture

D2-Align is a two-stage decoupled framework. Stage 1 learns a directional correction vector to debias the reward signal while keeping the generator frozen. Stage 2 uses the learned correction direction to guide generator optimization, preventing the model from collapsing into specific modes. The core idea is to **correct the direction of reward signals rather than their magnitude**.

### Key Designs

1. **Directional Correction Vector $\bm{b}_v$ (Stage 1)**:

   - **Core Idea**: A learnable vector $\bm{b}_v \in \mathbb{R}^d$ is introduced into the text embedding space of the reward model (HPS-v2.1, CLIP-based), constructing a corrected reward signal via a mechanism analogous to classifier-free guidance.
   - **Design Motivation**: The authors observe that appending descriptors such as "realistic" to prompts corresponding to over-rendered images suppresses reward scores closer to human judgment. This suggests that prompt perturbations can counteract reward model biases. However, the discrete vocabulary space is limited and requires manual selection, motivating a shift toward learning an optimal correction direction in the continuous embedding space.
   - **Implementation**: Starting from the original text embedding $\bm{e}_{\text{text}} = \Phi_{\text{text}}(c)$, positive and negative perturbed embeddings are constructed as:
     $$\bm{e}_+ = \text{normalize}(\bm{e}_{\text{text}} + \bm{b}_v), \quad \bm{e}_- = \text{normalize}(\bm{e}_{\text{text}} - \bm{b}_v)$$
     A corrected text embedding is obtained via extrapolation (rather than interpolation) using guidance scale $\omega > 1$:
     $$\tilde{\bm{e}}_{\text{text}} = \bm{e}_- + \omega \cdot (\bm{e}_+ - \bm{e}_-)$$
     The corrected reward signal is: $R_{\text{guided}}(\bm{x}_0, c; \bm{b}_v) = \text{score}(\bm{e}_{\text{img}}, \tilde{\bm{e}}_{\text{text}})$
   - **Training**: The generator $G_\theta$ is frozen, and the objective $\mathcal{L}_{\text{stage1}}(\bm{b}_v) = \mathbb{E}[-R_{\text{guided}}]$ is minimized, converging in approximately 2,000 steps.

2. **Guided Generator Optimization (Stage 2)**:

   - **Core Idea**: The correction vector $\bm{b}_v^*$ learned in Stage 1 is frozen, and the generator parameters $\theta$ are optimized using the corrected reward signal.
   - **Design Motivation**: The corrected reward signal suppresses the reward model's preference inflation effect (e.g., inflated scores for over-rendered images), forcing the generator to explore broader generation modes rather than obtaining high scores by catering to a single narrow preference.
   - **Optimization Objective**:
     $$\mathcal{L}_{\text{stage2}}(\theta) = \mathbb{E}_{c \sim \mathcal{D}, \bm{x}_0 \sim G_\theta(c)}[-R_{\text{guided}}(\bm{x}_0, c; \bm{b}_v^*)]$$
   - The two-stage decoupling avoids instability and directional drift caused by jointly optimizing $\bm{b}_v$ and $\theta$.

3. **Ground-Truth Noise Prior Technique**:

   - **Core Idea**: Addresses the contradiction between requiring clean images for reward evaluation and conducting optimization over noisy latents.
   - **Design Motivation**: Standard one-step denoising predictions $\hat{\bm{x}}_0$ are inaccurate at high-noise timesteps (large $t$), leading to unstable reward signals. Using ground-truth noise enables reliable reconstruction.
   - **Implementation**: A noisy latent $\bm{x}_t = \alpha_t \bm{x}_0 + \sigma_t \bm{\epsilon}_{\text{gt}}$ is constructed from a clean image $\bm{x}_0$ and known noise $\bm{\epsilon}_{\text{gt}}$, and the clean image is recovered via one-step denoising: $\hat{\bm{x}}_0 = (\bm{x}_t - \sigma_t \bm{\epsilon}_\theta(\bm{x}_t, t)) / \alpha_t$, enabling uniform sampling of timestep $t$ over $[0,1]$.

4. **DivGenBench Diversity Benchmark**:

   - **Core Idea**: Fills the gap in quantitative diversity evaluation for T2I alignment.
   - **Prompt Design**: Keyword-driven prompts actively probe the generative boundaries of models, rather than relying on output variance from ambiguous prompts.
   - **Four Dimensions**: ID (high-level semantics, e.g., facial identity), Style (mid-level aesthetics, e.g., artistic style), Layout (structural relations, e.g., spatial arrangement), and Tonal (low-level physical properties, e.g., color tone and brightness).
   - **Scale**: 3,200 prompts in total, 800 per dimension.
   - **Four Customized Metrics**: IDS (Identity Divergence Score), ASC (Aesthetic Style Coverage), SDI (Spatial Dispersion Index), and PVS (Photographic Variance Score), computed using domain-specific extractors such as ArcFace.

### Loss & Training

- **Stage 1 Loss**: $\mathcal{L}_{\text{stage1}}(\bm{b}_v) = \mathbb{E}_{c, \bm{x}_0 \sim G_{\theta_{\text{frozen}}}}[-R_{\text{guided}}]$; only the low-dimensional vector $\bm{b}_v$ is optimized, making training efficient.
- **Stage 2 Loss**: $\mathcal{L}_{\text{stage2}}(\theta) = \mathbb{E}[-R_{\text{guided}}(\bm{x}_0, c; \bm{b}_v^*)]$; the generator is optimized using the frozen correction vector.
- **Base Model**: FLUX.1.Dev
- **Reward Model**: HPS-v2.1 (primary configuration); a dual-reward configuration of HPS-v2.1 + CLIP is also evaluated.
- **Training Efficiency**: D2-Align reaches higher scores in fewer training steps; DanceGRPO and Flow-GRPO require 250+ steps to reach comparable performance.
- **Guidance Scale**: $\omega = 1.5$ is the optimal hyperparameter.

## Key Experimental Results

### Main Results (Quality Evaluation, Table 1)

Based on FLUX.1.Dev with single HPS-v2.1 reward:

| Method | Aesthetic ↑ | ImageReward ↑ | PickScore ↑ | Q-Align ↑ | HPS-v2.1 ↑ | CLIP ↑ | GenEval ↑ |
|--------|------------|---------------|-------------|-----------|------------|--------|-----------|
| FLUX Baseline | 6.417 | 1.670 | 0.240 | 4.922 | 0.310 | 0.315 | 0.663 |
| DanceGRPO | 6.068 | 1.664 | 0.241 | 4.930 | **0.361** | 0.293 | 0.522 |
| Flow-GRPO | 5.888 | 1.703 | 0.239 | 4.969 | **0.367** | 0.283 | 0.517 |
| SRPO | 6.614 | 1.533 | 0.241 | 4.866 | 0.296 | 0.302 | 0.623 |
| **D2-Align** | 6.450 | **1.771** | **0.246** | **4.969** | 0.343 | **0.323** | **0.636** |

DanceGRPO and Flow-GRPO achieve inflated HPS-v2.1 scores but show comprehensive degradation in Aesthetic, CLIP, and GenEval — a typical manifestation of reward hacking. D2-Align leads on all metrics except HPS-v2.1.

### Ablation Study (DivGenBench Diversity, Table 2)

| Method | IDS ↓ | ASC ↑ | SDI ↑ | PVS ↑ |
|--------|-------|-------|-------|-------|
| FLUX Baseline | 0.280 | 0.179 | 0.563 | 0.408 |
| DanceGRPO | 0.348 | 0.130 | 0.488 | 0.259 |
| Flow-GRPO | 0.391 | 0.044 | 0.389 | 0.168 |
| SRPO | 0.259 | 0.234 | 0.580 | 0.352 |
| **D2-Align** | **0.251** | **0.253** | **0.636** | **0.412** |

Flow-GRPO exhibits the most severe diversity collapse (ASC 0.044 vs. FLUX baseline 0.179), corroborating the PMC phenomenon. D2-Align achieves the best performance on all diversity metrics, even surpassing the unaligned FLUX baseline.

### Key Findings

- **PMC is quantitatively confirmed**: Despite inflated HPS-v2.1 scores, DanceGRPO and Flow-GRPO suffer comprehensive diversity collapse, with significant degradation across IDS, ASC, SDI, and PVS.
- **$\bm{b}_v$ converges in ~2,000 steps**: The correction effect stabilizes and the vector can be frozen before entering Stage 2.
- **$\omega = 1.5$ is optimal**: Excessively large guidance scales lead to over-correction and quality degradation, while insufficient scales provide inadequate correction.
- **Continuous vectors outperform discrete vocabulary**: The learned $\bm{b}_v$ outperforms manually selected vocabulary perturbations (e.g., "realistic") and uncorrected baselines across all dimensions on radar charts.
- **$\bm{b}_v^*$ is transferable**: Applying the learned correction vector to other methods such as DanceGRPO and Flow-GRPO also mitigates PMC.
- **Quality and diversity are no longer a trade-off**: D2-Align simultaneously improves both quality and diversity, breaking the conventional zero-sum relationship in alignment.

## Highlights & Insights

- **First systematic definition and quantification of PMC**: Re-examines reward hacking from a diversity perspective, identifying a neglected but important research direction.
- **Directional correction vs. magnitude adjustment**: Unlike KL regularization and multi-reward ensembling, D2-Align directly corrects the bias direction of the reward model, representing a more fundamental solution.
- **Efficient two-stage design**: Stage 1 learns only a single low-dimensional vector; Stage 2 converges faster than baselines, incurring low training cost.
- **DivGenBench fills an evaluation gap**: Keyword-driven prompt design combined with dimension-specific metrics provides the community with a standardized diversity evaluation tool.
- **Generality of embedding-space correction**: The approach of learning correction directions in continuous embedding spaces is potentially applicable to other reward hacking settings such as LLM alignment and video generation.

## Limitations & Future Work

- Validation is limited to FLUX.1.Dev; generalization to other architectures such as SD3 and SDXL has not been tested.
- $\bm{b}_v$ is a single globally shared direction vector, which may not cover all bias modes of the reward model (e.g., bias directions may differ across prompt types).
- The four dimensions of DivGenBench may not exhaustively cover all aspects of diversity (e.g., texture and lighting variation).
- Only HPS-v2.1 is validated as the reward model; bias patterns from other reward models may require relearning $\bm{b}_v$.
- Applicability to other modalities such as video generation or 3D generation has not been explored.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to define PMC and propose a directional correction framework; both the problem formulation and methodological design are highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Dual-dimensional evaluation across quality and diversity, plus ablations and user studies; however, validation is limited to FLUX.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly developed, with a coherent logical chain from empirical observation to method design.
- **Value**: ⭐⭐⭐⭐⭐ — PMC is a genuine pain point in practical deployment; $\bm{b}_v$ transferability extends applicability to other methods, and DivGenBench has the potential to become a standard benchmark.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Refining Few-Step Text-to-Multiview Diffusion via Reinforcement Learning](refining_few-step_text-to-multiview_diffusion_via_reinforcement_learning.md)
- [\[ICLR 2026\] Diffusion Blend: Inference-Time Multi-Preference Alignment for Diffusion Models](../../ICLR2026/image_generation/diffusion_blend_inference-time_multi-preference_alignment_for_diffusion_models.md)
- [\[CVPR 2026\] DiP: Taming Diffusion Models in Pixel Space](dip_taming_diffusion_models_in_pixel_space.md)
- [\[CVPR 2026\] Taming Score-Based Denoisers in ADMM: A Convergent Plug-and-Play Framework](taming_score-based_denoisers_in_admm_a_convergent_plug-and-play_framework.md)
- [\[ICLR 2026\] Hierarchical Entity-centric Reinforcement Learning with Factored Subgoal Diffusion](../../ICLR2026/image_generation/hierarchical_entity-centric_reinforcement_learning_with_factored_subgoal_diffusi.md)

</div>

<!-- RELATED:END -->
