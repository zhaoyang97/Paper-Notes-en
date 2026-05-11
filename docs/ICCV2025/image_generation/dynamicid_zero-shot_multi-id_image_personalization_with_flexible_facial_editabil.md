---
title: >-
  [Paper Note] DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability
description: >-
  [Image Generation] DynamicID achieves zero-shot single/multi-identity personalized image generation through two core components — Semantic Activation Attention (SAA) and Identity-Motion Reconfigurer (IMR) — while maintai…
tags:
  - "Image Generation"
date: 2026-05-08
content_hash: 68e6916d85de38eb
---

# DynamicID: Zero-Shot Multi-ID Image Personalization with Flexible Facial Editability

## Basic Information

- **Conference**: ICCV 2025
- **arXiv**: 2503.06505
- **Code**: Not released
- **Area**: Image Generation
- **Keywords**: personalized image generation, multi-ID generation, facial editing, attention mechanism, diffusion models

## TL;DR

DynamicID achieves zero-shot single/multi-identity personalized image generation through two core components — Semantic Activation Attention (SAA) and Identity-Motion Reconfigurer (IMR) — while maintaining high fidelity and flexible facial editability.

## Background & Motivation

Personalized human image generation aims to preserve consistent identity from reference images while incorporating user-specified text prompts. Existing tuning-free methods suffer from two critical limitations:

**Limited multi-ID generation**: Most methods are designed for single-identity scenarios and face severe identity blending in multi-identity settings — facial features from different reference subjects become entangled during generation.

**Insufficient facial editability**: Existing methods do not explicitly disentangle identity features (facial structure, skin texture) from motion features (expression, pose), preventing flexible editing of facial attributes.

Existing mitigation strategies (e.g., FastComposer's local cross-attention, InstantFamily's masked cross-attention) are built on single-ID frameworks, leading to degradation of core functionality.

## Method

### Overall Architecture

DynamicID comprises three core components: a face encoder (extracting facial features), IMR (disentangling and reconfiguring facial motion and identity in feature space), and SAA (injecting processed features into the T2I model). A task-decoupled training paradigm is adopted:

- **Anchoring stage**: Joint training of SAA and the face encoder (single-ID datasets only).
- **Reconfiguration stage**: The above components are frozen; IMR is trained using the VariFace-10k dataset.

### Semantic Activation Attention (SAA)

The softmax normalization in standard cross-attention forces each query to distribute a fixed total attention mass across all keys, even when no semantic relevance exists between a query and the keys. This leads to: (1) disruption of the original model behavior; and (2) identity blending in multi-ID scenarios.

SAA introduces a query-level activation gating mechanism:

$$z_{\text{new}} = z + \text{Expand}(w) \odot \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right) V$$

$$w = \text{Norm}(QK^\top J)$$

where $J \in \mathbb{R}^{k \times 1}$ is an all-ones column vector and $\text{Norm}(\cdot)$ applies min-max normalization to $[0,1]$. The activation weight $w$ reflects the semantic relevance of each query to the reference facial information:

- Facial-region queries → strong activation (close to 1)
- Background-region queries → suppressed (close to 0)
- Body-region queries → moderate activation

This yields three zero-shot generalization capabilities:
- **Context decoupling**: Non-facial regions are not disturbed by facial information.
- **Layout control**: Spatial layout can be controlled by modulating activation weights via masks.
- **Multi-ID personalization**: When injecting facial information for one subject, queries in regions corresponding to other subjects are suppressed.

### Identity-Motion Reconfigurer (IMR)

IMR consists of DisentangleNet $\phi_1$ and EntangleNet $\phi_2$, which disentangle and reconfigure identity and motion in feature space:

$$\xi_{\text{pred}} = \phi_2(\phi_1(\xi_{\text{src}}, \psi_{\text{src}}), \psi_{\text{tgt}})$$

where motion features $\psi$ are derived from facial prompts (text encoding) and facial keypoints (keypoint encoder), fused via an MLP.

### Loss & Training

**Anchoring stage** uses the standard diffusion noise prediction loss:

$$\mathcal{L}_{\text{noise}} = E_{z,t,\xi,\tau,\epsilon} \|\epsilon - \epsilon_\theta(z_t, t, \xi, \tau)\|_2^2$$

**Reconfiguration stage** uses a dual-objective loss:

$$\mathcal{L}_{\text{edit}} = \|\xi_{\text{pred}} - \xi_{\text{tgt}}\|_2^2 + \lambda \|\epsilon'_\theta(\xi_{\text{pred}}) - \epsilon'_\theta(\xi_{\text{tgt}})\|_2^2$$

The first term is direct feature matching; the second is latent diffusion consistency (ensuring predicted features share the same semantics in the generative model's latent space), with $\lambda=1$.

### VariFace-10k Dataset

Constructed for IMR training: 10k distinct identities, each with 35 facial images spanning varied expressions, poses, and lighting conditions.

## Key Experimental Results

### Main Results: Single-ID Personalization Quantitative Comparison

| Method | Arch. | CLIP-T ↑ | FaceSim ↑ | Expr ↑ | Pose ↑ |
|---|---|---|---|---|---|
| PuLID-FLUX | FLUX | 0.237 | 0.667 | 0.181 | 0.273 |
| PhotoMaker-v2 | SDXL | 0.238 | 0.592 | 0.243 | 0.869 |
| InstantID | SDXL | 0.233 | 0.723 | 0.151 | 0.264 |
| IPA-FaceID-Plus | SD1.5 | 0.236 | 0.712 | 0.156 | 0.266 |
| MasterWeaver | SD1.5 | 0.237 | 0.651 | 0.189 | 0.278 |
| **DynamicID (Ours)** | SD1.5 | **0.239** | 0.671 | **0.456** | **0.878** |

DynamicID achieves substantial margins on Expr (+0.213) and Pose (+0.009 vs. PhotoMaker), demonstrating far superior facial editability. FaceSim is slightly lower than InstantID/IPA-FaceID, but the latter's high FaceSim scores reflect direct facial copying (as confirmed by their low Expr/Pose values).

### Ablation Study: Contributions of SAA and IMR

| Method | CLIP-T ↑ | FaceSim ↑ | Expr ↑ | Pose ↑ |
|---|---|---|---|---|
| Ours w/o SAA | 0.224 | 0.682 | 0.422 | 0.862 |
| Ours w/o IMR | 0.228 | 0.712 | 0.161 | 0.253 |
| **Ours (Full)** | **0.239** | 0.671 | **0.456** | **0.878** |

- Removing SAA → significant drop in CLIP-T (0.224 vs. 0.239), indicating that SAA preserves the model's original text-editing capability.
- Removing IMR → sharp degradation in Expr/Pose (0.161/0.253 vs. 0.456/0.878), while FaceSim rises (direct copying of the reference face).

### Multi-ID Personalization Comparison

| Method | CLIP-T ↑ | FaceSim ↑ | Expr ↑ | Pose ↑ |
|---|---|---|---|---|
| FastComposer | 0.233 | 0.594 | 0.144 | 0.256 |
| UniPortrait | 0.235 | 0.718 | 0.149 | 0.268 |
| StoryMaker | 0.219 | 0.678 | 0.147 | 0.296 |
| **DynamicID** | **0.237** | 0.664 | **0.431** | **0.867** |

DynamicID maintains a significant advantage in multi-ID scenarios, with Expr and Pose substantially outperforming all baselines.

## Highlights & Insights

1. **Core insight behind SAA**: The softmax normalization in standard cross-attention is the root cause of disrupted model behavior and identity blending; query-level activation gating is an elegant solution.
2. **Zero-shot multi-ID generalization**: Multi-identity generation is achieved without multi-ID training data, relying solely on manipulation of SAA activation weights.
3. **Task-decoupled training**: Splitting joint training into two stages reduces data requirements (Anchoring requires only single-ID data; IMR requires only multiple images of the same individual).
4. **Feature-space operations**: IMR performs identity–motion disentanglement in latent space rather than pixel space, yielding computational efficiency and strong generalization.

## Limitations & Future Work

- Built on SD1.5, generation quality is bounded by the base model's capacity; adaptation to stronger models such as SDXL/FLUX has not been explored.
- The diversity of the VariFace-10k dataset may be insufficient to cover all facial variations.
- Layout control for three or more identities requires manual bounding box specification.
- Inference uses 50-step DDIM, leaving room for efficiency improvement.

## Related Work & Insights

- **InstantID** achieves high fidelity via ControlNet but lacks expression editing capability.
- **PhotoMaker** fuses facial features in the text embedding space, preserving partial editability at the cost of fidelity.
- **IP-Adapter**'s decoupled cross-attention pioneered attention-based feature injection, albeit at a coarse granularity.
- The activation gating idea in SAA is generalizable to other conditional injection settings (e.g., style transfer, pose control).

## Rating

⭐⭐⭐⭐ — The method is elegantly designed; in particular, SAA's query-level activation gating gracefully addresses both multi-ID generation and model behavior preservation. Facial editability metrics substantially surpass prior state of the art. The SD1.5 backbone, however, imposes an upper bound on generation quality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MAVFlow: Preserving Paralinguistic Elements with Conditional Flow Matching for Zero-Shot AV2AV Multilingual Translation](mavflow_preserving_paralinguistic_elements_with_conditional_flow_matching_for_ze.md)
- [\[ICCV 2025\] MamTiff-CAD: Multi-Scale Latent Diffusion with Mamba+ for Complex Parametric Sequence](mamtiff-cad_multi-scale_latent_diffusion_with_mamba_for_complex_parametric_seque.md)
- [\[ICCV 2025\] FlowTok: Flowing Seamlessly Across Text and Image Tokens](flowtok_flowing_seamlessly_across_text_and_image_tokens.md)
- [\[ICCV 2025\] Timestep-Aware Diffusion Model for Extreme Image Rescaling](timestep-aware_diffusion_model_for_extreme_image_rescaling.md)
- [\[ICCV 2025\] EEdit: Rethinking the Spatial and Temporal Redundancy for Efficient Image Editing](eedit_rethinking_the_spatial_and_temporal_redundancy_for_efficient_image_editing.md)

</div>

<!-- RELATED:END -->
