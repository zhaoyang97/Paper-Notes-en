---
title: >-
  [Paper Note] Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token
description: >-
  [CVPR 2026][Multimodal VLM][MLLM segmentation] This paper proposes SELF1E, the first MLLM segmentation method that requires neither a dedicated mask decoder nor more than a single [SEG] token. By introducing Residual Features Refilling (RFR) and Residual Features Amplifier (RFA) to recover resolution lost during pixel-shuffle compression, SELF1E achieves performance competitive with decoder-based methods across multiple segmentation tasks.
tags:
  - CVPR 2026
  - Multimodal VLM
  - MLLM segmentation
  - decoder-free segmentation
  - single-token segmentation
  - Pixel-Unshuffle
  - feature refinement
date: 2026-05-08
content_hash: 7fbdaf4a810c3c95
---

# Rethinking MLLM Itself as a Segmenter with a Single Segmentation Token

**Conference**: CVPR 2026
**arXiv**: [2603.19026](https://arxiv.org/abs/2603.19026)
**Code**: [https://github.com/ANDYZAQ/SELF1E](https://github.com/ANDYZAQ/SELF1E)
**Area**: Multimodal VLM
**Keywords**: MLLM segmentation, decoder-free segmentation, single-token segmentation, Pixel-Unshuffle, feature refinement

## TL;DR
This paper proposes SELF1E, the first MLLM segmentation method that requires neither a dedicated mask decoder nor more than a single [SEG] token. By introducing Residual Features Refilling (RFR) and Residual Features Amplifier (RFA) to recover resolution lost during pixel-shuffle compression, SELF1E achieves performance competitive with decoder-based methods across multiple segmentation tasks.

## Background & Motivation
**Background**: Existing MLLM segmentation methods (LISA, GSVA, OMG-LLaVA, etc.) primarily generate segmentation masks by attaching dedicated mask decoders (SAM / Mask2Former) to MLLMs.

**Limitations of Prior Work**:
   - Dedicated decoders introduce additional parameters and complex architectures, compromising methodological simplicity and creating dependency on external foundation models.
   - UFO attempts a decoder-free approach but requires 16 [SEG] tokens to compensate for resolution loss, increasing computational cost.
   - Root cause: pixel-shuffle downsampling in modern MLLMs significantly reduces visual feature resolution (e.g., 4× compression), discarding fine-grained spatial information essential for segmentation.

**Key Challenge**: Pixel-shuffle compression is necessary for efficient MLLM processing, yet the resulting spatial information loss is the fundamental bottleneck for decoder-free segmentation.

**Goal**: To demonstrate that a single [SEG] token is sufficient for high-quality segmentation — the bottleneck lies in feature resolution, not token count.

**Key Insight**: Pre-compression image encoder features retain full resolution and can be preserved as "pre-compressed features"; LLM-processed features carry finer semantic discriminability. The two are complementary.

**Core Idea**: Retain uncompressed encoder output features + collect residual features from LLM layers and fuse via upsampling + apply Pixel-Unshuffle to further amplify resolution.

## Method

### Overall Architecture
Image → Vision Encoder → Branch 1: pixel-shuffle + MLP compression → LLM → [SEG] token + compressed image features; Branch 2: self-replication to retain uncompressed features → RFR residual fusion → RFA further amplification → dot product to generate high-resolution mask.

### Key Designs

1. **Residual Features Refilling (RFR)**:

    - Retains uncompressed encoder output features $F_{V_1}^{HQ} \in \mathbb{R}^{N_0 \times d}$ (obtained by self-replicating each pixel $\alpha$ times and passing through the same MLP).
    - Collects the residual between pre- and post-LLM features: $F_R = F_{IMG} - F_{V_1}$.
    - Upsamples and fuses the residual: $F_{IMG}' = F_{V_1}^{HQ} + \mathcal{I}(F_R)$.
    - Effect: injects the fine-grained semantic discriminability learned by the LLM into the high-resolution features.

2. **Residual Features Amplifier (RFA)**:

    - Applies MLP + Pixel-Unshuffle separately to $F_{V_1}$ (pre-LLM) and $F_{IMG}$ (post-LLM).
    - Amplified residual: $F_{RFA} = f_{PUS}'(F_{IMG}) - f_{PUS}(F_{V_1})$.
    - Final fusion: $F_{IMG}' = f_{PUS}(F_{V_1}^{HQ}) + \mathcal{I}(F_{RFA})$, achieving resolution $\alpha N_0 \times d$.
    - Design Motivation: each embedding in the compressed features implicitly encodes information from $\alpha$ pixels; Pixel-Unshuffle can recover this latent information.
    - The [SEG] token is also passed through Pixel-Unshuffle and averaged: $F_{SEG}' = \text{mean}(f_{PUS}'(F_{SEG}))$.

3. **Segmentation-Specific Attention Mask**:

    - Designs a dual-perception pathway: image-to-image (bidirectional attention among image tokens) + image-to-segmentation (bidirectional interaction between image tokens and [SEG] token).
    - Provides richer inter-pixel and pixel-semantic interaction than standard causal attention.
    - Ensures the [SEG] token can fully attend to information at all image positions.

### Loss & Training
Training is based on the InternVL series. The two Pixel-Unshuffle MLPs in RFA require training.

## Key Experimental Results

### Main Results (Referring Expression Segmentation)

| Method | No Dedicated Decoder | Single Token | RefCOCO val | RefCOCO+ val | RefCOCOg val |
|------|:----------:|:------:|:-----------:|:------------:|:------------:|
| LISA-7B | ✗ | ✓ | 74.9 | 65.1 | 67.9 |
| u-LLaVA | ✗ | ✓ | 83.0 | 77.1 | 77.1 |
| UFO (16-token) | ✓ | ✗ | - | - | - |
| **SELF1E** | **✓** | **✓** | **~80+** | **~73+** | **~75+** |

### Ablation Study

| Configuration | Key Effect |
|------|---------|
| Direct prediction from compressed resolution | IoU significantly lower (~10%+ drop) |
| + RFR (residual refilling only) | IoU substantially improved, validating high-resolution + semantic residual |
| + RFA (residual amplification) | Further gain of 2–3%; Pixel-Unshuffle recovers latent information |
| + Segmentation attention mask | Additional gain of 1–2%; bidirectional interaction is beneficial |

### Key Findings
- First demonstration that decoder-free, single-token MLLM segmentation is feasible, with performance approaching SAM/Mask2Former-based methods.
- RFR contributes the most: recovering high-resolution features is the key, not increasing the number of [SEG] tokens.
- VQA capability is preserved: segmentation training does not degrade the model's general VQA performance.
- Pixel-shuffle compression is the root source of the resolution bottleneck, not the number of [SEG] tokens.

## Highlights & Insights
- **Challenges the prevailing paradigm that segmentation requires a decoder**: demonstrates that MLLMs inherently possess segmentation capability, requiring only recovery of the spatially compressed information.
- **Design philosophy of RFR/RFA**: rather than introducing new modules, the approach cleverly exploits information already present in the MLLM (encoder features, LLM residuals, the inverse of pixel-shuffle) to recover lost information via a "subtract-then-add" strategy.
- **Insight into MLLM architecture design**: while pixel-shuffle compression is favorable for VQA, it poses a fundamental obstacle for pixel-level tasks; future MLLM designs should consider how to preserve spatial information during compression.

## Limitations & Future Work
- Current performance remains slightly below the strongest decoder-based methods (e.g., u-LLaVA), leaving room for improvement.
- The Pixel-Unshuffle MLPs in RFA introduce additional trainable parameters.
- The segmentation attention mask requires modification of the LLM's attention computation, making it not fully plug-and-play.
- Open-vocabulary segmentation remains more challenging due to ambiguity in category vocabularies.

## Related Work & Insights
- **vs. LISA / GLaMM**: These methods feed the [SEG] token into a SAM decoder to generate masks, relying on external model capability. SELF1E is entirely self-contained.
- **vs. UFO**: UFO also removes the decoder but requires 16 [SEG] tokens, essentially compensating for insufficient resolution with token count. SELF1E directly addresses the resolution problem, enabling single-token operation.

## Supplementary Analysis
- The pixel-shuffle ratio $\alpha$ in the InternVL series is typically 4, reducing resolution to 1/4 after compression.
- The self-replication operation copies each pixel feature $\alpha$ times and passes through the same MLP, simulating the pre-shuffled features of neighboring pixels.
- RFR and RFA can be used independently or in combination; the combined configuration yields the best performance.
- The dual-perception pathway of the segmentation attention mask allows bidirectional interaction between image tokens and [SEG] tokens, whereas standard causal attention permits only unidirectional flow.
- The method introduces no external segmentation foundation models (SAM/Mask2Former), achieving truly MLLM-only segmentation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Challenges the dominant paradigm; first decoder-free single-token segmentation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-task validation with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear problem motivation and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Simplifies the MLLM segmentation pipeline and inspires future architecture design.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Rethinking Visual Token Reduction in LVLMs under Cross-Modal Misalignment](../../AAAI2026/multimodal_vlm/rethinking_visual_token_reduction_in_lvlms_under_cross-modal_misalignment.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] Rethinking VLMs for Image Forgery Detection and Localization](rethinking_vlms_for_image_forgery_detection_and_localization.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)
- [\[CVPR 2026\] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings](from_observation_to_action_latent_action-based_primitive_segmentation_for_vla_pr.md)

<!-- RELATED:END -->
