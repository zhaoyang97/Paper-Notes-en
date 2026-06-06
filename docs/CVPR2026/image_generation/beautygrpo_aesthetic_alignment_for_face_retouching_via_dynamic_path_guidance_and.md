---
title: >-
  [Paper Note] BeautyGRPO: Aesthetic Alignment for Face Retouching via Dynamic Path Guidance and Fine-Grained Preference Modeling
description: >-
  [CVPR2026][Image Generation][face retouching] This paper proposes BeautyGRPO, a reinforcement learning-based face retouching framework that constructs a fine-grained preference dataset FRPref-10K to train a dedicated rew…
tags:
  - "CVPR2026"
  - "Image Generation"
  - "face retouching"
  - "reinforcement learning"
  - "aesthetic alignment"
  - "flow matching"
  - "preference modeling"
  - "GRPO"
  - "dynamic path guidance"
date: 2026-05-08
content_hash: 731fcef2f878b9ec
---

# BeautyGRPO: Aesthetic Alignment for Face Retouching via Dynamic Path Guidance and Fine-Grained Preference Modeling

**Conference**: CVPR2026
**arXiv**: [2603.01163](https://arxiv.org/abs/2603.01163)  
**Code**: TBD (Project Page available)  
**Area**: Image Generation / Face Restoration
**Keywords**: face retouching, reinforcement learning, aesthetic alignment, flow matching, preference modeling, GRPO, dynamic path guidance

## TL;DR

This paper proposes BeautyGRPO, a reinforcement learning-based face retouching framework that constructs a fine-grained preference dataset FRPref-10K to train a dedicated reward model, and introduces a Dynamic Path Guidance (DPG) mechanism to balance stochastic exploration and high fidelity, achieving natural retouching results aligned with human aesthetic preferences.

## Background & Motivation

**Growing demand for face retouching**: The proliferation of social media and photography devices drives increasing demand for high-quality face retouching, requiring the removal of blemishes and acne while preserving identity-related features such as moles and pores.

**Fundamental limitations of supervised learning**: Existing methods (CNN-based and Transformer-based) rely on pixel-level supervised training, which can only imitate annotated images and fails to capture complex subjective aesthetic preferences, producing visually rigid and unnatural results.

**Performance ceiling of supervised learning**: Pixel-level reconstruction objectives cause models to overfit specific retouching styles, preventing discovery of aesthetic solutions that surpass the quality of training data.

**Inadequacy of general-purpose image editing models**: Large models such as NanoBanana and SeedDream can perform retouching operations but frequently produce unnatural effects including identity alterations, overly smooth plastic-like skin, and artifacts.

**Fidelity conflict in online RL**: T2I-RL frameworks such as FlowGRPO achieve exploration by injecting stochastic noise (SDE), but the accumulated random drift severely compromises the high fidelity required for face retouching, introducing conspicuous noise artifacts.

**Insufficient granularity of reward models**: Existing T2I reward models (ImageReward, HPSv2) primarily focus on global aesthetics or text–image consistency, and lack the capability to evaluate fine-grained perceptual dimensions such as skin smoothness, blemish removal, and texture preservation.

## Method

### Overall Architecture

BeautyGRPO consists of three major components: (1) the fine-grained preference dataset FRPref-10K; (2) a dedicated multi-dimensional reward model; and (3) the Dynamic Path Guidance (DPG) algorithm. The framework is built upon FluxKontext-LoRA as the backbone, with LoRA fine-tuning performed prior to RL training.

### Key Design 1: FRPref-10K Preference Dataset

- **Data sources**: Source portraits are sampled from FFHQR and a private high-resolution collection to ensure diverse demographics and shooting conditions.
- **Candidate generation**: Multiple models — NanoBanana, FluxKontext, RetouchFormer — with varied random seeds generate retouching candidates for each input image.
- **Preference pair construction**: 10,000 high-resolution preference pairs are constructed via inter-model comparisons (output vs. output) and output-versus-label comparisons.
- **Hybrid annotation pipeline**: (1) Multiple VLMs (GPT-4o, Qwen2.5-VL-72B, Gemini 2.5 Pro) evaluate images along five dimensions (skin smoothness, blemish removal, texture quality, sharpness, and identity preservation) and provide structured reasoning; (2) human reviewers audit under the same criteria; (3) disagreements are resolved by senior experts.

### Key Design 2: Three-Stage Reward Model Training

Initialized from Qwen2.5-VL-7B and trained under the UnifiedReward-Thinking paradigm:

- **Stage 1 (Structured reasoning initialization)**: SFT on a 2K subset trains the model to generate five-dimensional structured reasoning within `<think>` blocks and produce final preference decisions within `<answer>` blocks.
- **Stage 2 (Self-training with consistency filtering)**: The Stage 1 model generates multiple reasoning trajectories for the remaining 8K samples; consistent trajectories are filtered by both preference correctness and reasoning coherence, followed by another round of SFT.
- **Stage 3 (GRPO robustness enhancement)**: GRPO is applied to inconsistent samples to explore diverse reasoning paths, supervised jointly by an outcome reward (preference accuracy) and a process reward (reasoning coherence evaluated by DeBERTa-V3).

### Key Design 3: Dynamic Path Guidance (DPG)

**Problem**: FlowGRPO converts the ODE to an SDE (injecting noise $\sigma_t d\omega$) to enable exploration, but cumulative stochastic drift causes trajectories to deviate from the high-fidelity manifold.

**Core Idea of DPG**: At each sampling timestep, the guidance trajectory is dynamically re-planned to pull the current state back toward the high-fidelity manifold near an anchor point while preserving controlled exploration.

- **Stable anchor**: High-preference samples $x_0^{\text{anchor}}$ from FRPref-10K are selected as anchors — introduced only during sampling (not as a supervision target) — to constrain trajectories near the high-fidelity manifold.
- **Dynamic trajectory re-planning**: A straight-line ODE trajectory is re-planned between the current state $x_t$ and the anchor, computing the target state at the next timestep as $x_{t-\Delta t}^* = (\Delta t/t) x_0^{\text{anchor}} + (1 - \Delta t/t) x_t$.
- **Guided correction vector**: $z_t^{\text{anchor}} = (x_{t-\Delta t}^* - \mu_t) / \sigma_{\text{step}}$, indicating the direction that pulls the trajectory back toward the anchor path.
- **Mixed noise**: $z_t^{\text{mix}} = \lambda(t) z_t^{\text{anchor}} + (1-\lambda(t)) z_t^{\text{std}}$, where $\lambda(t) = t / \max(1, T-1)$ is a time-dependent coefficient — stronger anchor guidance at early timesteps corrects structural deviations, while greater reliance on stochastic noise at later timesteps encourages fine-grained exploration.
- **Efficiency optimization**: The full sampling trajectory is divided into $K=3$ segments; DPG updates are applied at one randomly selected timestep per segment, with ODE updates applied at the remaining steps.

### Loss & Training

The GRPO objective (clipped surrogate objective) is adopted with normalized advantages $\hat{A}^i$. Under DPG, transition probabilities follow a Gaussian distribution $\mathcal{N}(\mu_{\text{new}}, \sigma_{\text{new}}^2 I)$, where $\mu_{\text{new}} = (1-\lambda)\mu_t + \lambda x_{t-\Delta t}^*$ and $\sigma_{\text{new}} = (1-\lambda)\sigma_{\text{step}}$, used to compute the stepwise likelihood ratio $r_t(\theta)$ for policy updates.

## Key Experimental Results

### Experimental Setup

- **Backbone**: FluxKontext + LoRA fine-tuning
- **Data**: FFHQR test set (1,000 images) + in-the-wild internet portraits (1,000 images)
- **Evaluation metrics**: No-reference perceptual/aesthetic metrics (NIQE↓, NIMA↑, MUSIQ↑, MANIQA↑, NRQM↑, TOPIQ↑) + ArcFace identity preservation + FID
- **Hardware**: 8×H20 GPUs

### Main Results

| Method | Category | NIQE↓ | NIMA↑ | MUSIQ↑ | MANIQA↑ | TOPIQ↑ | ArcFace↑ |
|--------|----------|-------|-------|--------|---------|--------|----------|
| RetouchFormer | Specialized | 11.153 | 4.723 | 4.465 | 1.036 | 0.605 | 0.986 |
| NanoBanana | General editing | 11.301 | 4.919 | 4.681 | 1.009 | 0.621 | 0.889 |
| FluxKontext+LoRA | Baseline | 12.913 | 4.694 | 4.459 | 1.035 | 0.601 | 0.973 |
| FlowGRPO | RL baseline | 15.024 | 4.573 | 4.271 | 0.935 | 0.571 | 0.882 |
| **BeautyGRPO (Ours)** | **RL** | **10.831** | **5.123** | **4.906** | **1.079** | **0.676** | 0.952 |

### User Study

| Method | VRetouchEr | RetouchFormer | NanoBanana | Kontext LoRA | **Ours** |
|--------|-----------|---------------|------------|-------------|---------|
| Win Rate | 6.50% | 8.50% | 9.75% | 12.00% | **63.25%** |

100 participants voted on 20 test sample groups; BeautyGRPO achieves a win rate of 63.25%, substantially outperforming all baselines.

### Ablation Study

**Reward model comparison**: When replacing the editing reward model with alternatives, the proposed reward model outperforms EditReward, EditScore, and UnifiedReward-Edit across all metrics: NIMA (5.123), MUSIQ (4.906), MANIQA (1.079), and TOPIQ (0.676).

**Backbone generalizability**: Applying BeautyGRPO to Qwen-Image-Edit improves NIMA from 4.571 (original) / 4.824 (+LoRA) to 5.351 (+BeautyGRPO), and TOPIQ from 0.563 to 0.664.

**Effect of DPG step count $K$**: $K=3$ achieves performance comparable to $K=5$ with faster sampling and is adopted as the default.

### Key Findings

- Directly applying FlowGRPO to face retouching leads to severe degradation (NIQE 15.024 vs. BeautyGRPO 10.831), validating the necessity of DPG.
- Although BeautyGRPO's FID is slightly higher than the best supervised baseline (4.054 vs. 2.229), this reflects the model exploring aesthetic solutions beyond the training label distribution.
- ArcFace scores remain high (0.952 / 0.944), demonstrating robust identity preservation alongside aesthetic enhancement.

## Highlights & Insights

- This is the first work to introduce RL alignment based on human aesthetic preferences into face retouching, breaking through the performance ceiling of supervised learning.
- The DPG mechanism is elegantly designed: anchor guidance combined with a time-dependent mixing coefficient achieves a graceful balance between fidelity and exploration.
- The three-stage reward model training pipeline (SFT → self-training → GRPO), combined with a five-dimensional evaluation framework, constructs fine-grained feedback signals tailored to retouching.
- A 63.25% user study win rate — far exceeding the runner-up at 12% — provides compelling qualitative evidence of effectiveness.

## Limitations & Future Work

- Although large, FRPref-10K relies on a specific set of VLMs for annotation; annotation biases may propagate into the reward model.
- The impact of the anchor selection strategy on output quality is insufficiently analyzed, and the matching scheme between anchors and input images is not clearly specified.
- Evaluation is primarily conducted on FFHQR and internet portraits; robustness under challenging conditions such as extreme lighting, large head poses, and heavy occlusion is not explored.
- ArcFace scores are slightly lower than some supervised methods (0.952 vs. 0.986); the trade-off between identity preservation and aesthetic enhancement warrants further optimization.
- Computational costs are not thoroughly discussed (training on 8×H20 GPUs; DPG introduces additional overhead at inference).

## Related Work & Insights

- **Specialized retouching models**: ABPN, RestoreFormer(++), VRetouchEr, RetouchFormer — rely on supervised learning and are constrained by pixel-level objectives.
- **General-purpose editing models**: NanoBanana, ICEdit, SeedDream4.0, FluxKontext — capable but insufficiently precise and natural for retouching.
- **RL alignment for diffusion models**: DDPO, DPOK (policy gradient), DiffusionDPO (offline), FlowGRPO (online SDE exploration) — BeautyGRPO builds on FlowGRPO and introduces DPG to address fidelity degradation.
- **Reward modeling**: ImageReward, HPSv2, EditScore, EditReward, UnifiedReward — lack fine-grained dimensions specific to retouching.

## Rating

- Novelty: ⭐⭐⭐⭐ — The DPG mechanism and retouching-oriented preference alignment constitute a novel combination, though the core GRPO framework is adapted from FlowGRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage via quantitative results, qualitative analysis, user study, multiple ablations, and cross-backbone validation.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, well-motivated problem formulation, and complete mathematical derivations.
- Value: ⭐⭐⭐⭐ — The first work to systematically introduce RL preference alignment into face retouching, opening a new research direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Implicit Preference Alignment for Human Image Animation](../../ICML2026/image_generation/implicit_preference_alignment_for_human_image_animation.md)
- [\[ICCV 2025\] MoFRR: Mixture of Diffusion Models for Face Retouching Restoration](../../ICCV2025/image_generation/mofrr_mixture_of_diffusion_models_for_face_retouching_restoration.md)
- [\[ICCV 2025\] CharaConsist: Fine-Grained Consistent Character Generation](../../ICCV2025/image_generation/characonsist_fine-grained_consistent_character_generation.md)
- [\[CVPR 2026\] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping](memory-efficient_fine-tuning_diffusion_transformers_via_dynamic_patch_sampling_a.md)
- [\[CVPR 2026\] Taming Preference Mode Collapse via Directional Decoupling Alignment in Diffusion Reinforcement Learning](taming_preference_mode_collapse_via_directional_decoupling_alignment_in_diffusio.md)

</div>

<!-- RELATED:END -->
