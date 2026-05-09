---
title: >-
  [Paper Note] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference
description: >-
  [AAAI 2026][LLM Alignment][Preference Alignment] This paper proposes MaPO (Margin-aware Preference Optimization), a reference-free preference alignment method that aligns T2I diffusion models by directly optimizing the likelihood margin between preferred and dispreferred outputs under the Bradley-Terry model. MaPO outperforms DPO and task-specific methods across 5 domains, including style adaptation, safety generation, and general preference alignment.
tags:
  - AAAI 2026
  - LLM Alignment
  - Preference Alignment
  - Diffusion Models
  - Text-to-Image Generation
  - Reference-free
  - DPO
date: 2026-05-08
content_hash: c7b6e96ecff9b06f
---

# Margin-aware Preference Optimization for Aligning Diffusion Models without Reference

**Conference**: AAAI 2026
**arXiv**: [2406.06424](https://arxiv.org/abs/2406.06424)
**Code**: [https://github.com/mapo-t2i/mapo](https://github.com/mapo-t2i/mapo)
**Area**: Diffusion Models / Alignment RLHF
**Keywords**: Preference Alignment, Diffusion Models, Text-to-Image Generation, Reference-free, DPO

## TL;DR
This paper proposes MaPO (Margin-aware Preference Optimization), a reference-free preference alignment method that aligns T2I diffusion models by directly optimizing the likelihood margin between preferred and dispreferred outputs under the Bradley-Terry model. MaPO outperforms DPO and task-specific methods across 5 domains, including style adaptation, safety generation, and general preference alignment.

## Background & Motivation

**State of the Field**: RLHF/DPO-based preference alignment methods have been widely adopted to align T2I diffusion models (e.g., SDXL) with human preferences. These methods typically rely on a frozen reference model for KL divergence regularization to ensure training stability.

**Limitations of Prior Work**: The authors identify a critical **"reference mismatch"** problem in T2I diffusion models — when the distribution of preference data deviates significantly from that of the reference model (e.g., learning a new artistic style or personalizing to a specific subject), the reference model actively **impedes effective adaptation**. The unstructured nature of the visual modality makes this problem more severe than in LLM settings.

**Root Cause**: The larger the reference mismatch, the more severely methods like DPO degrade in performance. Yet practical applications frequently require adapting models to preferences that substantially differ from the pre-training distribution (e.g., from photorealistic to anime style), precisely the scenario where reference mismatch is most acute.

**Paper Goals**: Design a reference-free preference alignment method that completely eliminates the negative impact of reference mismatch on T2I diffusion model alignment.

**Starting Point**: Directly maximize the likelihood margin between preferred and dispreferred outputs under the Bradley-Terry preference model, while simultaneously maximizing the likelihood of preferred outputs, without anchoring to any reference model.

**Core Idea**: Unify T2I alignment as reference-free pairwise preference optimization that jointly learns general stylistic features and specific preferences.

## Method

### Overall Architecture
MaPO operates under the Bradley-Terry model and directly optimizes the likelihood margin between preferred and dispreferred images. Unlike DPO, MaPO requires no frozen reference model. The objective consists of two components: (1) maximizing the likelihood margin between preferred and dispreferred outputs; and (2) simultaneously maximizing the likelihood of preferred outputs to prevent both from decreasing.

### Key Designs

1. **Reference-free Preference Optimization**:

    - **Function**: Directly optimizes the margin under the Bradley-Terry model without relying on a reference model.
    - **Mechanism**: The standard DPO loss takes the form $\mathcal{L}_\text{DPO} = -\log \sigma(\beta [\log \frac{\pi_\theta(y_w|x)}{\pi_\text{ref}(y_w|x)} - \log \frac{\pi_\theta(y_l|x)}{\pi_\text{ref}(y_l|x)}])$, which requires a reference model $\pi_\text{ref}$. MaPO eliminates the reference model and directly optimizes $\log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)$ (the likelihood margin), with an additional $\log \pi_\theta(y_w|x)$ term to prevent simultaneous degradation of both likelihoods.
    - **Design Motivation**: In reference mismatch scenarios, the reference model acts as an "anchoring obstacle" — it constrains the model's freedom to move toward the preferred distribution. Removing it releases this constraint and allows the model to freely adapt toward the preferred distribution.

2. **Dataset Construction for Reference Mismatch Scenarios**:

    - **Function**: Constructs two datasets, Pick-Style and Pick-Safety, to simulate reference-chosen mismatch and reference-rejected mismatch, respectively.
    - **Mechanism**: Pick-Style prepends prompts such as "Disney style animated image" or "Pixel art style image" to generate preferred images and "Realistic 8k image" to generate dispreferred images, simulating stylistic preference shift (where the reference model is far from the preferred style). Pick-Safety uses a "Sexual, nudity" prefix to generate dispreferred images, simulating safety preference (where the reference model is far from the dispreferred content).
    - **Design Motivation**: Existing preference datasets do not allow direct control over the degree of reference mismatch; purpose-built datasets are necessary to validate MaPO's advantages across varying mismatch levels.

3. **Unified Multi-task T2I Alignment**:

    - **Function**: Unifies 5 distinct T2I tasks (safety generation, style adaptation, cultural representation, personalization, and general preference alignment) under a single pairwise preference optimization framework.
    - **Design Motivation**: Traditional methods require task-specific alignment strategies (e.g., DreamBooth for personalization). MaPO's reference-free framework is sufficiently flexible to accommodate all scenarios without modification.

### Loss & Training
- MaPO loss: margin-aware loss + preferred likelihood maximization term
- Built on SDXL; reduces training time by 14.5% compared to DPO by eliminating the reference model's forward pass
- Requires no additional GPU memory for storing a reference model

## Key Experimental Results

### Main Results

| Task | Dataset | MaPO | DPO | Notes |
|------|---------|------|-----|-------|
| Style Adaptation | Pick-Style (Cartoon) | **Significantly better** | Limited by ref mismatch | Advantage increases with mismatch severity |
| Safety Generation | Pick-Safety | **Significantly better** | Limited by ref mismatch | Clear advantage in safety scenarios |
| General Preference | Pick-a-Pic v2 | **Better** | Second best | Gains persist even under mild mismatch |
| Imgsys Ranking | Public leaderboard | **7th place** | 20th place | Outperforms 21/25 SOTA T2I models |
| Personalization | — | **Surpasses DreamBooth** | — | Replaces task-specific methods |

Key finding: MaPO's advantage over DPO **grows sharply** as the degree of reference mismatch increases.

### Ablation Study

| Analysis Dimension | Key Findings |
|-------------------|-------------|
| Reference mismatch severity | Greater mismatch leads to more severe DPO degradation and larger MaPO advantage |
| Training efficiency | 14.5% reduction in training time compared to DPO (no reference model forward pass required) |
| Memory efficiency | No need to store a reference model; more memory-friendly |

## Highlights & Insights
- **Precise identification of the reference mismatch problem**: The paper systematically analyzes the issue and its impact in T2I settings — this diagnosis itself constitutes a significant contribution. Many prior works directly apply DPO to diffusion models while overlooking this critical distinction.
- **Minimalist yet effective method**: The core modification of MaPO is simply removing the reference model and introducing a margin loss. Despite its simplicity, the approach achieves strong results, with a 7th-place ranking on Imgsys (vs. 20th for DPO) providing compelling evidence.
- **Unified multi-task framework**: A single method covers safety, style, personalization, and other tasks, eliminating the need to engineer separate alignment strategies for each.

## Limitations & Future Work
- The absence of a reference model may lack regularization in certain scenarios, introducing a risk of mode collapse
- Validation is limited to SDXL; applicability to other diffusion architectures (e.g., DiT-based models) remains unexplored
- Pick-Style and Pick-Safety are synthetic datasets; real user preferences may be considerably more complex
- Detailed sensitivity analysis of the margin hyperparameter is lacking

## Related Work & Insights
- **vs. Diffusion-DPO**: Diffusion-DPO directly transfers DPO to diffusion models while retaining the reference model, resulting in limited performance under reference mismatch. MaPO eliminates the reference model, yielding substantial advantages when mismatch is severe.
- **vs. DreamBooth**: DreamBooth is a dedicated personalization method, yet MaPO — as a general-purpose alignment approach — surpasses it in personalization scenarios.
- **vs. SimPO (LLM setting)**: SimPO also explores reference-free preference optimization but in the LLM domain. MaPO transfers this idea to T2I diffusion models and addresses the reference mismatch problem unique to the visual modality.

## Rating
- Novelty: ⭐⭐⭐⭐ Precisely identifies the reference mismatch problem and proposes a concise solution
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across 5 task domains and the Imgsys public leaderboard
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear with well-motivated objectives
- Value: ⭐⭐⭐⭐ Offers direct practical value for T2I diffusion model alignment

## Additional Notes
- The methodology and experimental design of this work provide a useful reference for related research
- Future work should validate the method's generalizability and scalability across more scenarios and larger scales
- Potential research value exists in combining this work with recent developments (e.g., intersections with RL/MCTS and multimodal methods)
- Deployment feasibility and computational efficiency should be evaluated against practical application requirements
- The choice of datasets and evaluation metrics may affect the generalizability of conclusions; cross-validation on additional benchmarks is recommended

## Additional Notes
- The methodology and experimental design of this work provide a useful reference for related research
- Future work should validate the method's generalizability and scalability across more scenarios and larger scales
- Potential research value exists in combining this work with recent developments (e.g., intersections with RL/MCTS and multimodal methods)

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[AAAI 2026\] AMaPO: Adaptive Margin-attached Preference Optimization for Language Model Alignment](amapo_adaptive_margin-attached_preference_optimization_for_l.md)
- [\[ICLR 2026\] Mitigating Mismatch within Reference-based Preference Optimization](../../ICLR2026/llm_alignment/mitigating_mismatch_within_reference-based_preference_optimization.md)
- [\[AAAI 2026\] Align to Structure: Aligning Large Language Models with Structural Information](align_to_structure_aligning_large_language_models_with_struc.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/llm_alignment/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)

<!-- RELATED:END -->
