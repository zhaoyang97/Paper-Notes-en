---
title: >-
  [Paper Note] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation
description: >-
  [CVPR 2026][Medical Imaging][Medical Image Segmentation] This paper proposes a lightweight Temporal Gated Adapter (T-Gated Adapter) that injects adjacent-slice context into the 2D vision-language model CLIPSeg. Trained on only 30 annotated CT volumes, the method achieves an average Dice of 0.704 (+0.206), with consistent improvements on cross-domain zero-shot evaluation and CT-to-MRI cross-modal evaluation.
tags:
  - CVPR 2026
  - Medical Imaging
  - Medical Image Segmentation
  - Vision-Language Models
  - Temporal Adapter
  - Cross-Modal Generalization
  - 3D Volumetric Awareness
date: 2026-05-08
content_hash: c3a6ae97f4a37091
---

# T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation

**Conference**: CVPR 2026
**arXiv**: [2604.08167](https://arxiv.org/abs/2604.08167)
**Code**: [GitHub](https://github.com/pranzalkhadka/T-Gated-Adapter)
**Area**: Medical Imaging / Vision-Language Models
**Keywords**: Medical Image Segmentation, Vision-Language Models, Temporal Adapter, Cross-Modal Generalization, 3D Volumetric Awareness

## TL;DR

This paper proposes a lightweight Temporal Gated Adapter (T-Gated Adapter) that injects adjacent-slice context into the 2D vision-language model CLIPSeg. Trained on only 30 annotated CT volumes, the method achieves an average Dice of 0.704 (+0.206), with consistent improvements on cross-domain zero-shot evaluation and CT-to-MRI cross-modal evaluation.

## Background & Motivation

**Background**: Vision-language models (VLMs) such as CLIPSeg enable zero-shot/few-shot segmentation via text prompts, making them highly attractive for medical imaging where annotations are scarce. Conventional fully supervised 3D architectures (U-Net, nnU-Net, Swin UNETR) achieve high in-domain accuracy but require extensive dense voxel-level annotations and are constrained to fixed organ taxonomies.

**Limitations of Prior Work**: VLMs are inherently 2D models, whereas CT/MRI data are 3D volumetric. The prevailing approach decomposes 3D volumes into 2D axial slices for independent processing. However, this slice-by-slice inference discards critical volumetric context: (1) anatomical continuity across adjacent slices is ignored, leading to inconsistent segmentation results between slices (e.g., false-positive pancreas predictions on slices where it is absent); (2) the absence of a cross-slice validation mechanism prevents the model from leveraging the important prior that "the target structure does not appear in neighboring slices" to suppress noisy predictions.

**Key Challenge**: There exists a fundamental domain gap between the powerful semantic generalization capability of VLMs (derived from billions of images during CLIP pretraining) and the volumetric continuity requirements of 3D medical imaging.

**Goal**: How can volumetric (3D) context be injected in a lightweight manner—without modifying the underlying VLM—to bridge the gap between 2D foundation models and 3D medical imaging?

**Key Insight**: Rather than retraining a 3D model, the authors design a temporal adapter that operates purely at the token level, enabling each spatial token to attend to tokens at the same position in neighboring slices. This allows the model to learn the pattern: "if the corresponding position in neighboring slices shows no evidence of the target structure, suppress the current prediction."

**Core Idea**: Inject temporal attention over a 5-slice context window into CLIP visual token representations, combined with an adaptive gate to balance temporal fusion and single-slice features, achieving lightweight 3D-aware VLM segmentation.

## Method

### Overall Architecture

A temporal adapter module is inserted into CLIPSeg (CLIP ViT-B/16 encoder + CLIP text encoder + Transformer decoder). The input consists of the center slice along with 2 adjacent slices on each side (5 slices total), all processed in parallel through the CLIP visual encoder to obtain token sequences. The temporal adapter sequentially performs: (1) temporal Transformer cross-slice attention → (2) spatial self-attention refinement → (3) adaptive gated fusion. The adapter output is blended with the original single-slice features before being passed to the CLIPSeg decoder to generate the segmentation map.

### Key Designs

1. **Temporal Transformer**:

    - Function: Enables each spatial position to independently aggregate contextual information along the slice dimension.
    - Mechanism: The CLIP encoder processes 5 slices in parallel, yielding output of shape $(5, L, D_v)$. This is reshaped to $(L, 5, D_{proj})$ ($D_{proj}=256$), and self-attention is applied independently at each spatial token position across the 5-slice dimension. A 4-layer pre-norm Transformer encoder is used, with linearly increasing stochastic depth regularization (drop-path rate from 0 to 0.1) and learned temporal positional encodings $\mathbf{e}_{pos} \in \mathbb{R}^{1 \times 5 \times D_{proj}}$. The output is projected back to $D_v$ dimensions and the center-slice features are extracted.
    - Design Motivation: Per-spatial-position cross-slice attention enables the model to learn to suppress detections on the current slice when there is no evidence of the target structure at the same spatial position in neighboring slices—directly addressing the root cause of false positives.

2. **Spatial Context Block**:

    - Function: Refines spatial relationships within a slice and propagates temporal information to spatial neighbors.
    - Mechanism: Since the temporal Transformer operates independently at each spatial position without modeling intra-slice spatial relationships, a spatial self-attention block performs self-attention across all $L$ token positions. This allows spatially adjacent tokens to share volumetric information each has gathered from the temporal Transformer, producing globally consistent representations. A pre-norm architecture is used with a two-layer MLP feed-forward network and GELU activation.
    - Design Motivation: The propagation of temporal information requires spatial coordination—neighboring spatial positions may need to "negotiate" in order to form consistent organ boundary judgments.

3. **Adaptive Gate**:

    - Function: Content-adaptively balances temporally fused features against the original single-slice features.
    - Mechanism: A learned gate $g = \sigma(\mathbf{W}_g \mathbf{h}_{temporal} + \mathbf{b}_g)$ controls the output $\mathbf{h}_{center} = g \cdot \mathbf{h}_{temporal} + (1-g) \cdot \mathbf{h}_{single}$. A key design choice is that $\mathbf{W}_g$ is initialized to zero and $\mathbf{b}_g$ is initialized to $-5.0$, yielding an initial gate value near zero—making the model equivalent to the original CLIPSeg baseline at the start of training. A binary gate penalty $\lambda(g \odot (1-g))$ ($\lambda=0.001$) is added to encourage decisive binary gate decisions.
    - Design Motivation: Different organs and slices have different demands for temporal context. For large organs (e.g., the liver), single-slice features are already sufficient, and aggressive temporal fusion may introduce noise. The adaptive gate allows the model to leverage temporal information only when needed.

### Loss & Training

Loss function: binary cross-entropy + Dice loss (equal weights). Differential learning rates: visual/text encoders $10^{-6}$, CLIPSeg decoder $10^{-5}$, adapter parameters $5 \times 10^{-5}$. Training for 30 epochs, batch size 8, full float32 precision, on a single NVIDIA T4 GPU. Key training strategies: (1) Negative sample sampling—slices where the target organ is absent are included at a 1:3 ratio, providing direct supervision for false-positive suppression; (2) Class-imbalance sampling—small organs (adrenal gland, duodenum, esophagus) receive 8× weight, medium organs (pancreas, stomach, gallbladder) receive 2× weight; (3) Data augmentation—±5° rotation applied consistently across all 5 slices, with horizontal flipping disabled for lateralized organs to preserve left-right anatomical identity.

## Key Experimental Results

### Main Results

| Method | FLARE22 | BTCV (Zero-Shot) | AMOS22 CT (Zero-Shot) |
|--------|---------|-----------------|----------------------|
| CLIPSeg Baseline | 0.497 | 0.334 | 0.283 |
| **CLIPSeg + T-Gated Adapter** | **0.704** | **0.544** | **0.513** |
| Gain | +0.206 | +0.210 | +0.230 |

### Ablation Study (Per-Organ Dice, FLARE22)

| Organ | Baseline | + Temporal Adapter | Δ Dice |
|-------|----------|--------------------|--------|
| Pancreas | 0.243 | 0.647 | +0.404 |
| Right Kidney | 0.499 | 0.836 | +0.337 |
| Gallbladder | 0.442 | 0.715 | +0.273 |
| Stomach | 0.581 | 0.849 | +0.268 |
| Liver | 0.911 | 0.960 | +0.049 |
| Esophagus | 0.524 | 0.381 | -0.143 |

**Cross-Modal Generalization (AMOS22 MRI, Zero-Shot, No MRI Training Data)**:

| Method | AMOS22 MRI |
|--------|-----------|
| DynUNet (3D Fully Supervised Baseline) | 0.224 |
| **CLIPSeg + T-Gated Adapter** | **0.366** |

### Key Findings

- Average Dice improves from 0.497 to 0.704 (+41% relative gain), with consistent improvements maintained under zero-shot cross-domain evaluation (+0.210/+0.230).
- Cross-domain performance degradation decreases from 38.0% to 24.9%, indicating that the adapter has learned genuine volumetric understanding rather than dataset-specific patterns.
- The organs with the largest improvements are precisely those suffering most from temporal inconsistency—the pancreas (+0.404), which spans very few axial slices and exhibits high morphological variability.
- Regression is observed for the esophagus (-0.143), whose thin tubular structure frequently disappears within the 5-slice window, causing temporal attention to introduce background noise.
- In CT-to-MRI zero-shot cross-modal evaluation, the method surpasses the 3D fully supervised DynUNet baseline (0.366 vs. 0.224), demonstrating that CLIP's semantic representations are more modality-invariant than convolutional features.
- Text prompt sensitivity experiments: Dice drops to 0.005 with blank prompts and to 0.011 with incorrect prompts (-99%), confirming that the model genuinely conditions on language queries rather than learning positional priors.

## Highlights & Insights

- **Minimal design, substantial gains**: By adding only a lightweight adapter (temporal attention + spatial attention + gating) at the output of the CLIP encoder—without modifying any foundation model weights—the method raises Dice by over 20 points.
- **Elegant zero-initialization gating strategy**: Setting $W_g=0$ and $b_g=-5.0$ ensures the model begins training from a state equivalent to the original baseline, avoiding the risk of the adapter corrupting pretrained features.
- **Importance of negative sample sampling**: Systematically including slices where the target organ is absent as training negatives directly supervises the core failure mode of slice-by-slice inference—false positives.
- **Modality invariance of CLIP semantic representations**: The fact that CT-trained features achieve zero-shot MRI performance exceeding 3D supervised CNNs reveals that VLM features encode high-level anatomical concepts rather than modality-specific pixel statistics.

## Limitations & Future Work

- The fixed 5-slice context window does not account for inter-scan slice spacing variations (1 mm vs. 5 mm); window depth should be adapted based on DICOM metadata.
- CLIPSeg rescales native 512×512 slices to 352×352, causing information loss for small structures such as the adrenal glands—a bottleneck the temporal adapter cannot compensate for.
- Training and evaluation are conducted on only 30 annotated volumes, limiting statistical significance.
- The regression observed for thin tubular structures such as the esophagus remains unresolved; different temporal aggregation strategies may be required for different organ types.
- No comparison is made against 3D adaptation methods for other foundation models such as SAM and MedSAM.

## Related Work & Insights

- CLIPSeg [Lüddecke & Ecker, 2022] establishes the foundational paradigm for VLM-based segmentation; this paper extends it to 3D.
- 3DSAM-Adapter [Gong et al., 2024] extends SAM from 2D to 3D via an alternative approach and serves as a potential comparison baseline.
- The temporal adapter design is generalizable to 3D clinical applications of other 2D foundation models (e.g., SAM, BiomedCLIP).
- Cross-modal zero-shot results suggest that for new modalities with extremely scarce annotations (e.g., ultrasound, PET), VLMs may be a more practical path than training 3D models from scratch.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The temporal adapter with gating is concise and effective, and the zero-initialization strategy is clever; however, the overall idea has precedent (temporal adapters have been explored in the video domain).
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Coverage is comprehensive, including cross-domain zero-shot, cross-modal, per-organ analysis, and prompt sensitivity tests; however, the training data scale is small (30 volumes).
- **Writing Quality**: ⭐⭐⭐⭐ — The logic is clear, experimental analysis is thorough, and failure cases (esophagus) are discussed candidly.
- **Value**: ⭐⭐⭐⭐ — Provides a practical solution for applying 2D VLMs to 3D medical imaging; cross-modal results carry important implications.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] BiCLIP: Bidirectional and Consistent Language-Image Processing for Robust Medical Image Segmentation](biclip_bidirectional_and_consistent_language-image_processing_for_robust_medical.md)
- [\[NeurIPS 2025\] STAMP: Spatial-Temporal Adapter with Multi-Head Pooling](../../NeurIPS2025/medical_imaging/stamp_spatial-temporal_adapter_with_multi-head_pooling.md)
- [\[CVPR 2026\] Are General-Purpose Vision Models All We Need for 2D Medical Image Segmentation? A Cross-Dataset Empirical Study](are_generalpurpose_vision_models_all_we_need_for_2.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)

<!-- RELATED:END -->
