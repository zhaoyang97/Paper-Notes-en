---
title: >-
  [Paper Note] Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion
description: >-
  [ICCV 2025][Image Generation][T2I diffusion model] This paper proposes **Dual Recursive Feedback (DRF)**, a training-free dual recursive feedback system that recursively refines intermediate latents via **appearance feedback** and **generation feedback**, addressing the insufficient structure/appearance disentanglement of controllable T2I diffusion models in class-invariant scenarios, thereby achieving fine-grained pose transfer and appearance fusion.
tags:
  - ICCV 2025
  - Image Generation
  - T2I diffusion model
  - controllable generation
  - pose transfer
  - appearance fidelity
  - Score Distillation
  - training-free
date: 2026-05-08
content_hash: 7ada6e1e32d1c2e1
---

# Dual Recursive Feedback on Generation and Appearance Latents for Pose-Robust Text-to-Image Diffusion

**Conference**: ICCV 2025
**arXiv**: 2508.09575
**Code**: [GitHub](https://github.com/jwonkm/DRF)
**Area**: Image Generation
**Keywords**: T2I diffusion model, controllable generation, pose transfer, appearance fidelity, Score Distillation, training-free
**Authors**: Jiwon Kim, Pureum Kim, SeonHwa Kim et al. (Korea University, Sookmyung Women's University)

## TL;DR

This paper proposes **Dual Recursive Feedback (DRF)**, a training-free dual recursive feedback system that recursively refines intermediate latents via **appearance feedback** and **generation feedback**, addressing the insufficient structure/appearance disentanglement of controllable T2I diffusion models in class-invariant scenarios, thereby achieving fine-grained pose transfer and appearance fusion.

## Background & Motivation

The core objective of controllable T2I diffusion models is to generate images that simultaneously satisfy **structural control** (pose, edge, depth) and **appearance transfer** according to user intent. Existing methods each have limitations:

**ControlNet**: Requires fine-tuning separate models for different spatial factors, increasing training overhead.

**FreeControl**: Requires no additional training, but relies on gradient optimization to minimize latent space loss, resulting in high computational cost.

**Ctrl-X**: More efficient, maintaining structure via feed-forward feature replacement and aligning appearance statistics through self-attention layers, but suffers significant performance degradation in **class-invariant** settings.

Key challenge: When the structure image and appearance image belong to different categories (e.g., transferring human motion to a tiger), simply separating appearance and structural information in attention maps is far from sufficient—leading to *appearance leakage* and degraded structural fidelity.

## Method

### Overall Architecture

DRF builds upon Ctrl-X. The structure image $\mathbf{I}^s$ and appearance image $\mathbf{I}^a$ are first injected into a pretrained T2I model to obtain an initial generation latent:

$$\mathbf{z}_{t-1}^g = \text{Ctrl-X}(\mathbf{z}_t^g | t, y, f_t^s, A_t^s, h_t^a)$$

During the diffusion generation process, DRF then recursively refines the latents through dual feedback.

### Key Designs

#### Appearance Feedback

Addresses *appearance leakage*—ensuring that appearance information is not lost during generation.

Key design: Inspired by the fixed-point concept in IDS, but avoids the limited editability caused by over-constraint. To address the over-guidance problem arising from the large discrepancy between the posterior mean $\mathbf{z}_{0|t}^a$ and $\mathbf{z}_0^a$ at large timesteps, a **corrected stochastic latent** is proposed:

$$\tilde{\mathbf{z}}_t^a = \sqrt{\frac{\alpha_t}{\alpha_{t-1}}} \mathbf{z}_0^a + \sqrt{1 - \frac{\alpha_t}{\alpha_{t-1}}} \epsilon$$

Appearance feedback loss:
$$\mathcal{L}_{\text{app}} = d(\mathbf{z}_{0|t}^a, \mathbf{z}_0^a)$$

#### Generation Feedback

Addresses the appearance overfitting that occurs when only appearance feedback is applied—ensuring the generated result aligns with user intent.

The generation result $\mathbf{z}_{\text{prev}}^g$ from the previous iteration serves as another fixed point, aligning the current update direction with it:

$$\mathcal{L}_{\text{gen}} = d(\mathbf{z}_{0|t}^g, \mathbf{z}_{\text{prev}}^g)$$

### Loss & Training

An exponential weighting scheme is adopted to progressively adjust feedback weights:

$$w_{\text{iter}}^{(i)} = \sqrt{\frac{\exp(k \cdot \frac{i}{N-1}) - 1}{\exp(k) - 1}}$$

Design rationale: In early iterations, appearance feedback dominates to ensure identity information is correctly reflected; as iterations progress, the generation feedback weight gradually increases to better match user intent.

Final DRF loss:
$$\mathcal{L}_{\text{DRF}}^{(i)} = d(\mathbf{z}_{0|t}^a, \mathbf{z}_0^a) + \rho \cdot w_{\text{iter}}^{(i)} \cdot d(\mathbf{z}_{0|t}^g, \mathbf{z}_{\text{prev}}^g)$$

The DRF loss is minimized by updating the injected noise $\epsilon$:
$$\epsilon \leftarrow \epsilon - \lambda \nabla_\epsilon \mathcal{L}_{\text{DRF}}^{(i)}$$

DRF is applied over the intermediate 20 steps (skipping the first 5), with $N$ recursive iterations performed at each step.

## Key Experimental Results

### Main Results: Quantitative Comparison

| Method | Mesh Self-Sim↓ | Mesh CLIP↑ | Mesh DINO-I↑ | Pose Self-Sim↓ | Pose CLIP↑ | Pose DINO-I↑ | Successive Rate |
|------|---------------|------------|-------------|---------------|------------|-------------|----------------|
| T2I-Adapter+IP-Adapter | 0.2374 | 0.3062 | 0.6627 | 0.2949 | 0.2865 | 0.5304 | 0.9718 |
| ControlNet+IP-Adapter | 0.2024 | 0.3320 | 0.6068 | 0.3035 | 0.2904 | 0.6857 | 0.8873 |
| FreeControl | 0.1503 | 0.3270 | 0.7288 | 0.2839 | 0.2880 | 0.6162 | 0.9152 |
| Ctrl-X | 0.1542 | 0.3464 | 0.7139 | 0.2332 | 0.3429 | 0.7378 | 0.9577 |
| **DRF (Ours)** | **0.1532** | **0.3492** | **0.7342** | **0.2294** | **0.3503** | **0.7391** | **0.9859** |

DRF achieves the highest Successive Rate of **0.9859**, indicating that the generated images most faithfully integrate both structure and appearance.

### Ablation Study

#### User Study

| Method | Text Preference↑ | Structure Preference↑ | Appearance Preference↑ |
|------|-----------------|----------------------|----------------------|
| T2I-Adapter+IP-Adapter | 7.53% | 8.09% | 11.43% |
| ControlNet+IP-Adapter | 16.02% | 12.90% | 14.14% |
| FreeControl | 12.93% | 14.83% | 17.63% |
| Ctrl-X | 17.37% | 9.44% | 13.75% |
| **DRF (Ours)** | **35.52%** | **43.73%** | **33.72%** |

DRF leads comprehensively across all three dimensions, with a structure preference rate as high as **43.73%**.

#### Scheduler-Agnostic Validation

| Scheduler | Steps | CLIP↑ | Self-Sim↓ | DINO-I↑ | Time(s) |
|--------|-------|-------|-----------|---------|---------|
| DPM-Solver++ + DRF | 10 | 0.3256 | 0.1605 | 0.826 | 15.56 |
| DDIM + DRF | 30 | 0.3492 | 0.1204 | 0.891 | 35.74 |
| DPM-Solver++ + DRF | 10 vs DDIM 40 | comparable | comparable | comparable | **3× faster** |

Combining DRF with a fast solver (DPM-Solver++) reduces latency by approximately 3× while maintaining perceptual quality.

#### Cross-Backbone Validation

| Backbone | CLIP↑ | Self-Sim↓ | DINO-I↑ | GPU Memory(GiB) |
|------|-------|-----------|---------|-----------------|
| SD 1.5 + DRF | 0.3108 | 0.1820 | 0.6176 | 5.87 |
| SD 2.0 + DRF | 0.2934 | 0.2166 | 0.453 | 6.28 |
| **SDXL + DRF** | **0.3331** | **0.1586** | **0.6957** | 18.98 |

DRF is portable to SD 1.5 and SD 2.0, substantially reducing GPU memory consumption while preserving fidelity.

## Highlights & Insights

1. **Complementary dual-feedback design**: Appearance feedback ensures identity preservation, while generation feedback prevents appearance overfitting—both are indispensable (appearance feedback alone leads to over-emphasis on the appearance image at the expense of structure).
2. **Corrected stochastic latent technique**: Scaling by $\alpha_t/\alpha_{t-1}$ avoids over-guidance at large timesteps, representing an effective improvement over IDS.
3. **Exponential-weighted iteration strategy**: The appearance-first, gradually-increasing-generation weighting schedule is consistent with the intuition underlying diffusion generation.
4. **Cross-category pose transfer**: Successfully achieves extreme cross-category fusion such as human motion → tiger/penguin.
5. **Plug-and-play property**: Can be integrated into existing models such as ControlNet+IP-Adapter to further improve performance.
6. **Scheduler-agnostic**: Compatible with multiple schedulers including DDIM, DPM-Solver++, and UniPC.

## Limitations & Future Work

1. **Computational overhead**: Dual recursive feedback inevitably increases inference time ($N=3$ iterations require approximately 57 seconds vs. 15 seconds for Ctrl-X).
2. **Fine-grained detail preservation**: Limited capability in preserving fine-grained details such as ordinary human faces not covered in training data.
3. **Dependence on base model capacity**: The upper bound of DRF enhancement is constrained by the expressive capability of the underlying T2I model.
4. **Resolution limited to 512×512**: Not validated at higher resolutions.

## Related Work & Insights

- **Relationship to Ctrl-X**: DRF uses Ctrl-X as its backbone and addresses its failure cases in class-invariant settings through recursive feedback.
- **Connection to IDS/SDS**: Appearance feedback draws on the fixed-point regularization of IDS, but avoids over-constraint through the corrected stochastic latent.
- **Implications for controllable generation**: Training-free methods using score guidance can achieve fine-grained structure–appearance disentanglement, providing a new direction for efficient controllable generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual recursive feedback design is innovative, with well-supported arguments for the complementary mechanism of appearance/generation feedback.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Quantitative metrics + user study + ablation + cross-scheduler/cross-backbone validation constitute a comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear pipeline diagrams, complete derivations, and intuitive qualitative results.
- **Value**: ⭐⭐⭐⭐ — Training-free and plug-and-play properties confer high practical applicability.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Rethink Sparse Signals for Pose-guided Text-to-Image Generation](rethink_sparse_signals_for_pose-guided_text-to-image_generation.md)
- [\[ICCV 2025\] DPoser-X: Diffusion Model as Robust 3D Whole-Body Human Pose Prior](dposer-x_diffusion_model_as_robust_3d_whole-body_human_pose_prior.md)
- [\[ICCV 2025\] Dense2MoE: Restructuring Diffusion Transformer to MoE for Efficient Text-to-Image Generation](dense2moe_restructuring_diffusion_transformer_to_moe_for_efficient_text-to-image.md)
- [\[ICCV 2025\] DiffuMatch: Category-Agnostic Spectral Diffusion Priors for Robust Non-rigid Shape Matching](diffumatch_category-agnostic_spectral_diffusion_priors_for_robust_non-rigid_shap.md)
- [\[ICCV 2025\] Fix-CLIP: Dual-Branch Hierarchical Contrastive Learning via Synthetic Captions for Better Understanding of Long Text](fix-clip_dual-branch_hierarchical_contrastive_learning_via_synthetic_captions_fo.md)

<!-- RELATED:END -->
