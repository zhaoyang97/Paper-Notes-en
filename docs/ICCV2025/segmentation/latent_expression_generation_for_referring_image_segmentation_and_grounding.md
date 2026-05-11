---
title: >-
  [Paper Note] Latent Expression Generation for Referring Image Segmentation and Grounding
description: >-
  [ICCV 2025][Segmentation][Referring image segmentation] This paper proposes the Latent-VG framework, which generates multiple latent expressions from a single textual description — each sharing the same subject but highl…
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "Referring image segmentation"
  - "visual grounding"
  - "latent expression generation"
  - "contrastive learning"
  - "multimodal fusion"
date: 2026-05-08
content_hash: 5f190247a99a03bb
---

# Latent Expression Generation for Referring Image Segmentation and Grounding

**Conference**: ICCV 2025
**arXiv**: [2508.05123](https://arxiv.org/abs/2508.05123)
**Code**: None
**Area**: Image Segmentation
**Keywords**: Referring image segmentation, visual grounding, latent expression generation, contrastive learning, multimodal fusion

## TL;DR

This paper proposes the Latent-VG framework, which generates multiple latent expressions from a single textual description — each sharing the same subject but highlighting distinct visual attributes — to bridge the semantic gap between sparse text and rich visual information via complementary visual details. The method achieves state-of-the-art performance on both referring image segmentation and referring expression comprehension tasks.

## Background & Motivation

Referring image segmentation (RIS) and referring expression comprehension (REC) require localizing target objects based on textual descriptions. A visual region can be described in many ways — for example, the same target may be referred to as "the man in jeans," "the older man," or "the man on the left" — yet models typically receive only a single, brief text input that captures only a fraction of the rich visual information pertaining to the target.

The paper identifies two core problems with existing approaches:

**Text–visual semantic gap**: A single sparse textual expression cannot cover all visual details of the target region (color, texture, spatial position, contextual relations, etc.), causing model predictions to be heavily dependent on specific textual cues.

**Confusion among similar objects**: When visually similar objects are present in a scene, sparse textual descriptions can lead to incorrect localization — e.g., "the man in jeans" may inadvertently match a woman also wearing jeans.

The core idea of this paper is: if multiple diverse latent expressions can be generated from a single text input — each preserving the same subject while emphasizing different visual attributes — and these expressions are collectively used for prediction, the semantic gap can be effectively bridged. The guiding principle is **shared subject and distinct attributes**.

## Method

### Overall Architecture

Latent-VG is built upon the BEiT-3 single-encoder architecture. Given an image and a text input, multiple initial latent text sequences are generated via latent expression initialization. At each encoder layer, the Subject Distributor and Visual Concept Injector modules enforce the shared-subject and distinct-attributes principle. At inference, the predicted masks from all expressions are averaged to produce the final segmentation result.

### Key Designs

1. **Latent Expression Initialization**: Starting from the original text token embeddings, $N$ initial latent expressions are generated. Each expression is processed by a Latent Attribute Initializer that (a) randomly drops semantic tokens with expression-specific probability $p_i$ to reduce dependence on the original text, and (b) transforms the token count from $m$ to a predefined length $k^i$ via a length transformation layer $\phi^i \in \mathbb{R}^{k^i \times m}$. A subject token $\mathbf{s}$ is automatically selected from the text tokens (via a linear layer followed by Gumbel-softmax) and prepended to each latent expression:

$$\mathbf{Z}_0^i = [\mathbf{z}_{[cls]}^i, \mathbf{s}, \mathbf{A}_0^i] \in \mathbb{R}^{(k^i+2) \times d}$$

The same subject token is also injected into the visual sequence, replacing the visual class token. The design motivation is to introduce diversity through varying dropout rates and sequence lengths, while maintaining consistent target reference through the shared subject token.

2. **Subject Distributor**: After self-attention at each encoder layer, the subject tokens within each latent expression may have drifted from the original subject semantics. This module re-synchronizes all expression-level subject tokens $\{\mathbf{s}^i\}_{i=1}^N$ with the visual-domain subject token $\mathbf{s}^{\mathcal{V}}$, ensuring that all latent expressions consistently refer to the same target subject throughout encoding — functioning as an anchor mechanism. The design motivation is to prevent subject drift caused by self-attention mixing.

3. **Visual Concept Injector (VCI)**: At each encoder layer, unique visual concepts are injected into the attribute tokens to realize differentiated attributes. The procedure is: (a) initialize $N_c$ orthogonal concept tokens $\mathbf{C}$; (b) select target-relevant patches $\mathbf{V}^{tr}$ from visual patches based on similarity to the textual class token (thresholded at the mean); (c) concept tokens retrieve visual concepts from the target-relevant patches via attention: $\mathbf{C}^{\mathcal{V}} = \mathbf{W} \mathbf{V}^{tr}$; (d) visual concepts are injected into the attribute tokens $\tilde{\mathbf{A}}$ via **column-normalized attention**, enabling each attribute token to competitively bind to distinct visual concepts (analogous to Slot Attention). The design motivation is to extract complementary information from the visual domain that belongs to the target but is absent from the original text.

### Loss & Training

The loss function consists of two components:

1. **Positive-Margin Contrastive Loss**:

$$\mathcal{L}_{\text{pos-cont}} = -\frac{1}{N}\sum_{i=1}^{N}\log\frac{\exp(\min(1, \gamma + s_i)/\tau)}{\sum_{k \in \mathcal{N}_i}\exp(s_k/\tau)}$$

where $s_i = \mathbf{t}_o^\top \mathbf{z}_o^i / (\|\mathbf{t}_o\| \|\mathbf{z}_o^i\|)$ and $\gamma$ is the margin parameter (set to 0.2). Standard contrastive learning forces positive samples into full alignment, which would cause all latent expressions to collapse into representations identical to the original text. The positive margin allows a degree of divergence among positive samples, encouraging diversity while preserving semantic consistency.

2. **Segmentation Loss**:

$$\hat{\mathbf{p}} = \frac{\sigma(\mathbf{F}^{\mathcal{V}} \cdot \mathbf{t}_o) + \sum_{i=1}^{N}\sigma(\mathbf{F}^{\mathcal{V}} \cdot \mathbf{z}_o^i)}{N+1}$$

BCE and Dice losses are applied to the averaged prediction map, with $\lambda_{\text{bce}}=2$ and $\lambda_{\text{dice}}=0.5$.

Training uses AdamW with batch size 64 on 4 A6000 GPUs. All dropout rates are set to 0 during inference.

## Key Experimental Results

### Main Results

**RIS Results (Combined Dataset training, mIoU)**

| Method | Encoder | RefCOCO val | RefCOCO+ val | RefCOCOg val(U) |
|--------|---------|-------------|--------------|-----------------|
| PolyFormer | Swin-B + BERT | 75.96 | 70.65 | 69.36 |
| EEVG | ViT-B + BERT | 79.49 | 71.86 | 73.56 |
| One-Ref | BEiT3-B | 79.83 | 74.68 | 74.06 |
| **Latent-VG (Ours)** | BEiT3-B | **81.01** | **76.92** | **76.10** |

**GRES Results (gRefCOCO dataset)**

| Method | val mIoU | val N-acc | testA mIoU | testB mIoU |
|--------|---------|----------|-----------|-----------|
| GSVA-7B (SAM+CLIP-L) | 66.47 | 62.43 | 71.08 | 62.23 |
| HDC (Swin-B) | 68.23 | 63.38 | 72.52 | 63.85 |
| **Latent-VG (Ours)** | **72.45** | **70.42** | **74.51** | **66.12** |

On GRES, Latent-VG substantially surpasses prior state of the art by simply adding an empty token to handle the no-target case.

### Ablation Study

**Component Contributions (RefCOCO+ val)**

| Configuration | SD | VCI | mIoU | Gain |
|---------------|----|-----|------|------|
| No latent expressions (baseline) | - | - | 68.76 | +0.00 |
| + Naive latent expressions | × | × | 71.03 | +2.27 |
| + Subject Distributor | ✓ | × | 71.59 | +2.83 |
| + Visual Concept Injector | × | ✓ | 71.86 | +3.10 |
| + SD + VCI | ✓ | ✓ | 72.63 | +3.87 |
| + Positive-margin contrastive loss | ✓ | ✓ | **73.19** | **+4.43** |

**Contrastive Learning Strategy Comparison**

| Loss | mIoU | oIoU | Pr@0.9 |
|------|------|------|--------|
| InfoNCE | 72.03 | 69.72 | 24.60 |
| Triplet | 72.57 | 70.42 | 23.55 |
| ArcFace | 72.73 | 70.43 | 24.54 |
| **Positive-margin (Ours)** | **73.19** | **70.68** | **26.45** |

### Key Findings

- Performance peaks at $N=2$ latent expressions with token lengths $\{4, 10\}$; additional expressions introduce redundancy rather than further gains.
- Subject token selection is effective: Gumbel-softmax automatically identifies the core referential word from the text.
- Column normalization in the Visual Concept Injector (analogous to Slot Attention) is critical for attribute differentiation; removing it causes all expressions to converge to identical representations.
- Performance gains are most pronounced on RefCOCO+ and RefCOCOg, suggesting the method is particularly effective for complex textual descriptions.
- On REC, directly extracting bounding boxes from RIS masks — without any task-specific decoder — outperforms LLM-based methods.

## Highlights & Insights

- **Strong originality**: Addressing semantic sparsity in visual grounding from a text augmentation perspective; latent expression generation in the latent space constitutes a novel research direction.
- **Unified framework**: A single model simultaneously achieves state-of-the-art performance on RIS, REC, and GRES without task-specific architectural modifications.
- **Positive-margin contrastive loss**: Elegantly resolves the latent expression collapse problem by preserving diversity while maintaining semantic consistency.
- **Slot Attention-inspired VCI**: The competitive mechanism enables attribute tokens to automatically bind to distinct visual concepts.

## Limitations & Future Work

- The computational overhead is modest (approximately 12M additional parameters and 3 GFLOPs), yet multiple latent expressions still increase inference time.
- Only BEiT3-B is used as the backbone; applicability to larger models or alternative architectures (e.g., CLIP-L) remains unverified.
- The number of visual concept tokens $N_c = 100$ is determined via ablation; an adaptive selection mechanism is lacking.
- Extension to video referring segmentation or 3D scene understanding is a promising direction for future work.
- Subject token selection via Gumbel-softmax may be insufficiently robust for complex expressions that lack an explicit referential subject.

## Related Work & Insights

- CRIS and CGFormer employ pixel-level contrastive losses for fine-grained alignment; the positive-margin contrastive loss proposed here represents an interesting refinement of conventional contrastive learning.
- SimCSE generates textual variants in latent space via dropout; this work extends that idea by incorporating visually conditioned attribute injection.
- One-Ref serves as the same-backbone prior state of the art; Latent-VG consistently improves upon it by 1–3 points, validating the value of latent expression generation.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Addressing semantic sparsity in visual grounding from a text augmentation perspective; latent expression generation with shared-subject and distinct-attributes design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive validation across RIS, REC, and GRES; ablation studies cover every component and hyperparameter.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation figures are intuitive, method descriptions are clear, and mathematical derivations are complete.
- **Value**: ⭐⭐⭐⭐⭐ A unified framework that advances the state of the art on three tasks; the positive-margin contrastive loss has broad applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ReferDINO: Referring Video Object Segmentation with Visual Grounding Foundations](referdino_referring_video_object_segmentation_with_visual_grounding_foundations.md)
- [\[ICCV 2025\] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy](deris_decoupling_perception_and_cognition_for_enhanced_referring_image_segmentat.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](../../NeurIPS2025/segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)
- [\[NeurIPS 2025\] ARGenSeg: Image Segmentation with Autoregressive Image Generation Model](../../NeurIPS2025/segmentation/argenseg_image_segmentation_with_autoregressive_image_generation_model.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)

</div>

<!-- RELATED:END -->
