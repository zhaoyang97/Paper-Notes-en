---
title: >-
  [Paper Note] CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing
description: >-
  [CVPR2026][Image Generation][Image Editing] Proposes CARE-Edit, a condition-aware expert routing framework that implements dynamic computation allocation on a DiT backbone via heterogeneous experts (Text/Mask/Reference/Base) coupled with a lightweight latent-attention router. This effectively addresses issues like color bleeding and identity drift caused by conflicting multi-conditional signals (text, mask, reference image) in unified image editors.
tags:
  - CVPR2026
  - Image Generation
  - Image Editing
  - Mixture-of-Experts
  - Condition-Aware Routing
  - Diffusion Transformer
  - Multimodal Fusion
date: 2026-05-08
content_hash: 6cfc044deb168fd7
---

# CARE-Edit: Condition-Aware Routing of Experts for Contextual Image Editing

**Conference**: CVPR2026  
**arXiv**: [2603.08589](https://arxiv.org/abs/2603.08589)  
**Code**: To be released  
**Area**: Image Generation  
**Keywords**: Image Editing, Mixture-of-Experts, Condition-Aware Routing, Diffusion Transformer, Multimodal Fusion

## TL;DR

Proposes CARE-Edit, a condition-aware expert routing framework that implements dynamic computation allocation on a DiT backbone via heterogeneous experts (Text/Mask/Reference/Base) coupled with a lightweight latent-attention router. This effectively addresses issues like color bleeding and identity drift caused by conflicting multi-conditional signals (text, mask, reference image) in unified image editors.

## Background & Motivation

1. **Task Interference in Unified Editors**: Existing unified diffusion editors (e.g., OmniGen2, ACE++) use a fixed shared backbone to process all editing tasks, failing to adapt to heterogeneous requirements (local vs. global, semantic vs. photometric), leading to mutual interference between tasks.

2. **Fundamental Flaws of Static Fusion**: Methods like ControlNet and OmniControl fuse multimodal conditions (text, mask, reference image) through simple concatenation or additive adapters, failing to dynamically adjust signal priorities according to the denoising process. This results in text semantics potentially overriding mask constraints or reference identity/style being incorrectly applied.

3. **Time-Varying Importance of Conditional Signals**: In the diffusion denoising trajectory, the importance of different conditions changes with timesteps—early steps focus on semantic layout, while later steps focus on boundary refinement and style consistency. Static methods cannot adapt to this dynamic balance.

4. **Specific Manifestations of Multi-Condition Conflicts**: Color bleeding at mask boundaries, identity/style drift of the reference image, global adjustments intruding into areas that should be preserved, and unpredictable behavior under multi-condition inputs.

5. **Uncontrollable User Mask Quality**: Rough masks provided by users often misalign with target object boundaries; direct use leads to editing artifacts. Dynamic mask refinement during denoising is required.

6. **Under-utilization of MoE in Image Editing**: Existing diffusion MoEs (e.g., EC-DiT) use homogeneous experts and lack heterogeneous expert designs tailored for different modalities/conditions, failing to fundamentally resolve multi-condition conflicts.

## Method

### Overall Architecture
CARE-Edit embeds condition-aware expert routing into a frozen DiT backbone (based on FLUX.1 Dev), training only lightweight adapters, routers, and fusion layers. The core contains three modules: Routing Select, Mask Repaint, and Latent Mixture.

### Four Heterogeneous Experts
1. **Text Expert**: Performs semantic reasoning and object synthesis via cross-attention with text tokens.
2. **Mask Expert**: Achieves spatial precision and boundary refinement via convolutional operations combined with refined masks.
3. **Reference Expert**: Learns identity/style-consistent transformations from reference features via FiLM modulation.
4. **Base Expert**: Maintains global consistency and background fidelity via cross-attention with base image features.

Each expert's output is projected via LayerNorm + Linear to maintain feature scale consistency.

### Routing Select (Top-K Routing)
- Computes a token-specific key (encoding local information) and a global conditioning query (encoding editing task goals) for each token.
- Calculates logit scores for each expert via an MLP, then selects top-K (K=3) experts after softmax normalization.
- The routing temperature $\tau$ is gradually annealed during training, and EMA smoothing is applied to routing logits to reduce variance.
- A fixed proportion $\lambda_{shared}$ of tokens is always routed to a shared expert to prevent routing collapse.
- Uses convex residual fusion to aggregate expert outputs.

### Mask Repaint (Mask Refinement)
- At each diffusion step $t$, the current latent, reference encoding, and the previous step's predicted mask are used to estimate a residual mask field $\Delta m$ via convolution.
- Superimposed onto the previous mask after sigmoid activation: $\hat{M}(t) = \text{clip}(\hat{M}(t-1) + \Delta m, 0, 1)$.
- Implements progressive boundary tightening via boundary consistency losses (gradient alignment + smoothing regularization) during training.
- The refined mask is fed back into the routing process of the next DiT block.

### Latent Mixture (Expert Output Fusion)
- Token-wise fusion: Performs a convex combination of expert outputs based on routing probability weights $w_e$.
- Timestep-adaptive mixing: Mixes the fusion result with the base expert output through a learned timestep-dependent gate $\gamma$.
- TV regularization encourages spatial smoothness of the mixing weight map.

### Progressive Training Curriculum
- First 40K steps: Trained on basic single-task data to establish general representations.
- Last 60K steps: Switched to complex multi-task data to evolve the routing layer from general to specialized.
- Total training of 100K steps completed on 8×NVIDIA L20, with a learning rate of 1e-4 and batch size of 16.

## Key Experimental Results

### Table 1: Instruction Editing Performance Comparison (EMU-Edit & MagicBrush Test Sets)

| Method | Type | EMU-Edit CLIPim↑ | CLIPout↑ | L1↓ | DINO↑ | MagicBrush CLIPout↑ | DINO↑ |
|------|------|-----------------|----------|-----|-------|-------------------|-------|
| InstructPix2Pix | Special | 0.834 | 0.219 | 0.121 | 0.762 | 0.245 | 0.767 |
| EMU-Edit | Special | 0.859 | 0.231 | 0.094 | 0.819 | 0.261 | 0.879 |
| OmniGen2 | Unified | 0.865 | 0.306 | 0.088 | 0.832 | 0.306 | 0.889 |
| AnyEdit | Unified | 0.866 | 0.284 | 0.095 | 0.812 | 0.273 | 0.877 |
| **CARE-Edit** | Unified | **0.868** | **0.313** | **0.082** | **0.835** | **0.324** | 0.885 |

### Table 2: Ablation Study (DreamBench++ Multi-Object Setting)

| Variant | DINO-I↑ | CLIP-I↑ | CLIP-T↑ |
|------|---------|---------|---------|
| w/o Experts | 0.485 | 0.652 | 0.296 |
| w/o Latent Mixture | 0.509 | 0.678 | 0.301 |
| w/o Mask Repaint | 0.523 | 0.693 | 0.304 |
| K=2 | 0.541 | 0.707 | 0.312 |
| K=4 | 0.562 | 0.716 | 0.325 |
| **Full Model (K=3)** | **0.568** | **0.720** | **0.327** |

Removing expert routing leads to the largest performance drop, verifying the core value of condition-aware dynamic allocation. K=3 is optimal.

## Highlights & Insights

1. **Heterogeneous Expert Design Precises Editing Demands**: Four types of experts handle semantics, space, style, and global consistency respectively. Unlike the general design of traditional homogeneous MoEs, each expert has clear modal specialization.
2. **Task-Aware Dynamic Routing**: Experimental analysis shows that different tasks (erase/replace/style transfer/text editing) activate different expert combinations, verifying the effectiveness of condition-aware routing—Mask Expert dominates structural editing, while Reference Expert dominates style transfer.
3. **Mask Repaint Enables Progressive Mask Refinement**: Gradually corrects rough masks using latent information from the diffusion process itself without requiring additional segmentation models.
4. **High Training Data Efficiency**: Achieves performance competitive with OmniGen2 with only 120K training samples (the latter uses significantly more data).
5. **Comprehensive Lead on DreamBench++**: Superior to strong baselines like OmniGen2 and UNO in both single-object and multi-object settings.

## Limitations & Future Work

1. **Hyperparameter Sensitivity**: MoE-inherent hyperparameters such as the top-K value, routing temperature annealing strategy, and $\lambda_{shared}$ require careful tuning.
2. **Fixed Expert Set**: Currently, only four types of experts cover common modalities; facing new editing types (e.g., 3D-aware editing, physics-consistent editing) may require dynamic expert loading or expansion.
3. **Computational Overhead**: Although sparse routing and a frozen backbone are used, the additional computational cost of the four expert branches + router + Mask Repaint is not explicitly quantified in the paper.
4. **Dependency on FLUX.1 Pre-trained Model**: The framework's generality is limited by the choice of DiT backbone; applicability to other backbones (e.g., SD3, SDXL) has not been verified.
5. **DINO Metric on MagicBrush Slightly Lower than OmniGen2**: Not a comprehensive lead on all benchmarks.

## Related Work & Insights

- **vs. OmniGen2/ACE++**: Unified editor baselines use a fixed shared backbone for all tasks, lacking condition-aware dynamic computation allocation. CARE-Edit surpasses them on most metrics via heterogeneous expert routing.
- **vs. ControlNet/OmniControl**: Fuses conditional signals through static concatenation or additive adapters, unable to dynamically prioritize or suppress conflicting modalities. CARE-Edit's top-K routing enables token-level condition selection.
- **vs. EC-DiT**: Also a diffusion MoE, but EC-DiT uses homogeneous experts + expert-choice routing, suitable for general generation. CARE-Edit introduces heterogeneous experts with modal division of labor specifically to resolve multi-condition editing conflicts.
- **vs. DreamBooth/BLIP-Diffusion**: Subject-driven methods rely on embedding learning or adapters, prone to overfitting or uncontrollable editing scope. CARE-Edit treats reference guidance as a conditional capability handled by specialized experts.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introducing heterogeneous MoE to image editing to solve multi-condition conflicts is a novel entry point; the design of each module is sound.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Covers both instruction editing and subject-driven scenarios with complete ablations, though lacks computational overhead quantification.
- Writing Quality: ⭐⭐⭐⭐ — Problem definition is clear; expert activation analysis and training dynamic visualizations enhance interpretability.
- Value: ⭐⭐⭐⭐ — Provides an effective solution for condition conflicts in unified image editors with high practical value.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] TAG-MoE: Task-Aware Gating for Unified Generative Mixture-of-Experts](tag-moe_task-aware_gating_for_unified_generative_mixture-of-experts.md)
- [\[CVPR 2026\] Group Editing: Edit Multiple Images in One Go](group_editing_edit_multiple_images_in_one_go.md)
- [\[ICLR 2026\] Routing Matters in MoE: Scaling Diffusion Transformers with Explicit Routing Guidance](../../ICLR2026/image_generation/routing_matters_in_moe_scaling_diffusion_transformers_with_explicit_routing_guid.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)
- [\[CVPR 2026\] RAZOR: Ratio-Aware Layer Editing for Targeted Unlearning in Vision Transformers and Diffusion Models](razor_ratio-aware_layer_editing_for_targeted_unlearning_in_vision_transformers_a.md)

<!-- RELATED:END -->
