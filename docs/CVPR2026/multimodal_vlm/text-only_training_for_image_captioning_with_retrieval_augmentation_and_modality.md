---
title: >-
  [Paper Note] Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction
description: >-
  [CVPR 2026][Multimodal VLM][Image captioning] This paper proposes TOMCap — a text-only training approach for image captioning that combines retrieval augmentation, modality gap correction…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Image captioning"
  - "text-only training"
  - "retrieval augmentation"
  - "modality gap correction"
  - "CLIP"
date: 2026-05-08
content_hash: fc6c3096e181e59f
---

# Text-Only Training for Image Captioning with Retrieval Augmentation and Modality Gap Correction

**Conference**: CVPR 2026
**arXiv**: [2512.04309](https://arxiv.org/abs/2512.04309)
**Code**: To be confirmed
**Area**: Multimodal VLM
**Keywords**: Image captioning, text-only training, retrieval augmentation, modality gap correction, CLIP

## TL;DR
This paper proposes TOMCap — a text-only training approach for image captioning that combines retrieval augmentation, modality gap correction, and LoRA fine-tuning. The model trains exclusively on text yet processes images at inference time, surpassing existing training-free and text-only methods.

## Background & Motivation
**Background**: Image captioning conventionally relies on large-scale manually annotated image-text pairs for supervised training. Two categories of low-resource methods have emerged recently: training-free methods (e.g., ZeroCap) that leverage pre-trained models for zero-shot inference, and text-only methods that train solely on text corpora and switch to image inputs at inference time.

**Limitations of Prior Work**: Training-free methods are prone to hallucination, while text-only methods are constrained by the CLIP modality gap — image embeddings and text embeddings are not perfectly aligned within the same space, causing a distributional mismatch when text features are used during training but image features are used during inference.

**Key Challenge**: The core assumption of text-only training — that text embeddings can serve as proxies for image embeddings — is not fully valid due to the CLIP modality gap. Existing methods rely only on Gaussian noise injection to bridge this gap, yielding limited effectiveness.

**Goal**: Integrate retrieval augmentation, modality gap correction, and latent representation decoding into a unified and stronger text-only training framework.

**Key Insight**: Rather than correcting only the mean, the proposed approach also aligns the standard deviation to reduce the modality gap, while incorporating retrieved similar captions as prompts to guide generation.

**Core Idea**: Jointly leverage retrieval-augmented prompt construction, mean-and-standard-deviation-aligned modality gap correction, and cross-attention latent guidance to achieve high-quality image captioning under text-only training.

## Method

### Overall Architecture
During training: text captions are encoded by the CLIP text encoder to obtain embeddings → modality gap correction → retrieval of similar captions for prompt construction → GPT-2 decoder (with cross-attention and LoRA) is trained. During inference: images are encoded by the CLIP image encoder → retrieval → prompt + cross-attention → caption generation.

### Key Designs

1. **Modality Gap Correction**:

    - Function: Reduces the distributional discrepancy between CLIP image and text embeddings.
    - Mechanism: Performs dimension-wise normalization alignment: $e_d^{T'_n} = (e_d^{T_n} - \mu_d^T) \times \frac{\sigma_d^I}{\sigma_d^T} + \mu_d^I$, correcting not only the mean shift (as in prior work) but also aligning the standard deviation.
    - Design Motivation: Aligning only the mean ignores differences in distributional shape, causing the corrected text embeddings to have a different spread than image embeddings. Aligning the standard deviation further reduces the modality gap radius.

2. **Retrieval Augmentation**:

    - Function: Retrieves semantically similar captions as contextual prompts.
    - Mechanism: SigLIP2 is used to encode approximately 16M captions in the database; nearest-neighbor retrieval is performed on the input embedding, and the top-$K$ results are used to construct the prompt: "Similar images have the following captions: {c1}...{ck}. Write a caption:"
    - Design Motivation: Retrieved similar captions provide stylistic and semantic references, helping the model capture patterns relevant to the input image.

3. **Cross-Attention + LoRA Fine-tuning**:

    - Function: Cross-attention layers are inserted into each GPT-2 layer to process corrected CLIP embeddings, while LoRA fine-tunes the attention layers.
    - Mechanism: Cross-attention layers take the corrected CLIP embeddings (input + $K$ retrieved results) as keys/values and GPT-2 hidden states as queries. LoRA (rank=32) fine-tunes only the attention projection matrices.
    - Design Motivation: Cross-attention provides latent-level visual guidance; LoRA avoids catastrophic forgetting associated with full-parameter fine-tuning.

4. **Training Target Design**:

    - Function: Uses the most similar retrieved caption (rather than the original ground truth) as the training target.
    - Mechanism: The caption in the database most similar to the input embedding is selected as the teacher-forcing target.
    - Design Motivation: This encourages the model to learn the mapping "similar embeddings produce the same caption," improving generalization.

### Loss & Training
Standard cross-entropy loss is used to predict the token sequence of the most similar retrieved caption. The CLIP and GPT-2 backbone parameters are frozen; only the cross-attention layers and LoRA parameters are trained. Total training time is approximately 6 hours on an NVIDIA RTX 6000.

## Key Experimental Results

### Main Results (MSCOCO Karpathy test)

| Category | Method | B@4 | METEOR | CIDEr |
|---------|--------|-----|--------|-------|
| Training-free | LMCap | 19.9 | 22.0 | 75.9 |
| Text-only | CapDec | 26.4 | 25.1 | 91.8 |
| Text-only | ViECap | 27.2 | 24.8 | 92.9 |
| Text-only | EntroCap | 27.6 | 25.3 | 94.3 |
| **Text-only** | **TOMCap (ours)** | **28.8** | **25.5** | **97.8** |

### NoCaps Validation Set (CIDEr)

| Method | In-domain | Near-domain | Out-domain | Overall |
|------|-----------|-------------|------------|---------|
| ViECap | 61.1 | 64.3 | 65.0 | 66.2 |
| EntroCap | 62.5 | - | - | - |
| **TOMCap** | **71.2** | **70.8** | **68.5** | **70.4** |

### Key Findings
- TOMCap outperforms all text-only and training-free methods on both MSCOCO and NoCaps.
- Retrieval augmentation is the most critical component; removing it results in the largest CIDEr drop.
- Mean + standard deviation alignment yields approximately 2 CIDEr points improvement over mean-only alignment.
- $K=4$ retrieved captions achieves the best performance; larger $K$ introduces noise.
- TOMCap shows a clear advantage on NoCaps Out-domain, demonstrating strong generalization.

## Highlights & Insights
- **Improved Modality Gap Correction**: Extending alignment from first-order moments (mean) to second-order moments (standard deviation) is simple yet effective. This idea is transferable to other cross-modal alignment scenarios.
- **Retrieval as Training Target**: Using the most similar retrieved caption — rather than the original annotation — as the training target elegantly elevates retrieval from an input-side auxiliary to a target construction mechanism, enhancing generalization.
- **Extremely Low Training Cost**: Requires only text data and a single GPU for 6 hours of training, making it well-suited for resource-constrained settings.

## Limitations & Future Work
- Performance still falls short of fully supervised methods, with a gap of approximately 10–15 CIDEr points.
- Relies on an external database of 16M captions; database quality and coverage directly impact performance.
- The CLIP modality gap shift may vary across domains, making uniform correction potentially suboptimal.
- Only GPT-2-base is used as the decoder; larger language models may yield better results but require more computation.

## Related Work & Insights
- **vs. SmallCap**: SmallCap also employs retrieval augmentation with cross-attention but requires image-text pair training; TOMCap eliminates the dependency on image training data.
- **vs. CapDec**: CapDec uses Gaussian noise to bridge the modality gap; TOMCap achieves more precise correction via statistical moment alignment.

## Rating
- Novelty: ⭐⭐⭐ The method is a combinatorial optimization of existing techniques without fundamental innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers MSCOCO and NoCaps with detailed ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear and the experimental section is well-organized.
- Value: ⭐⭐⭐ Text-only training for image captioning is a meaningful but relatively niche direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICLR 2026\] Closing the Modality Gap Aligns Group-Wise Semantics](../../ICLR2026/multimodal_vlm/closing_the_modality_gap_aligns_group-wise_semantics.md)
- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](../../ACL2026/multimodal_vlm/tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[CVPR 2026\] ReCALL: Recalibrating Capability Degradation for MLLM-based Composed Image Retrieval](recall_recalibrating_capability_degradation_for_mllm-based_composed_image_retrie.md)
- [\[CVPR 2026\] EagleNet: Energy-Aware Fine-Grained Relationship Learning Network for Text-Video Retrieval](eaglenet_energy-aware_fine-grained_relationship_learning_network_for_text-video_.md)

</div>

<!-- RELATED:END -->
