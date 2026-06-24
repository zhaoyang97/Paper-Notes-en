---
title: >-
  [Paper Note] Multimodal Autoregressive Pre-training of Large Vision Encoders
description: >-
  [CVPR 2025 (Highlight)][Multimodal VLM][Vision Encoders] Apple proposes the AIMV2 series of vision encoders, which pairs a ViT encoder with a multimodal autoregressive decoder—simultaneously generating raw image patches and text tokens as pre-training objectives. While maintaining a simple training pipeline, it achieves general-purpose performance across diverse tasks. AIMV2-3B reaches 89.5% on ImageNet frozen trunk evaluation and comprehensively outperforms CLIP and SigLIP o…
tags:
  - "CVPR 2025 (Highlight)"
  - "Multimodal VLM"
  - "Vision Encoders"
  - "Autoregressive Pre-training"
  - "Multimodal Pre-training"
  - "AIMV2"
  - "Large-scale Vision Models"
date: 2026-05-08
content_hash: 1f7f1cada17891d9
---

# Multimodal Autoregressive Pre-training of Large Vision Encoders

**Conference**: CVPR 2025 (Highlight)  
**arXiv**: [2411.14402](https://arxiv.org/abs/2411.14402)  
**Code**: [https://github.com/apple/ml-aim](https://github.com/apple/ml-aim)  
**Area**: Multimodal VLM  
**Keywords**: Vision Encoders, Autoregressive Pre-training, Multimodal Pre-training, AIMV2, Large-scale Vision Models

## TL;DR
Apple proposes the AIMV2 series of vision encoders, which pairs a ViT encoder with a multimodal autoregressive decoder—simultaneously generating raw image patches and text tokens as pre-training objectives. While maintaining a simple training pipeline, it achieves general-purpose performance across diverse tasks. AIMV2-3B reaches 89.5% on ImageNet frozen trunk evaluation and comprehensively outperforms CLIP and SigLIP on multimodal understanding benchmarks.

## Background & Motivation

**Background**: Pre-training of large-scale vision encoders is the foundation of computer vision and multimodal AI. Currently, there are two mainstream directions: (1) Contrastive learning (such as CLIP, SigLIP, etc.), which learns vision-language aligned representations via image-text contrastive losses and performs exceptionally well on zero-shot classification and multimodal understanding tasks. (2) Autoregressive pre-training (such as AIMv1), which learns visual representations by predicting the next image patch token, exhibiting strong performance in pure vision tasks (such as classification and localization).

**Limitations of Prior Work**: Contrastive models excel at multimodal tasks but lag behind specialized models on pure vision tasks (e.g., DINOv2 is superior in localization tasks); autoregressive models excel at vision but lack language understanding capabilities. Currently, there is no pre-training method that enables a single vision encoder to achieve SOTA performance on both **pure vision** and **multimodal** tasks simultaneously. Furthermore, contrastive learning relies heavily on large quantities of high-quality image-text pairs and is sensitive to negative sample selection, limiting its scalability.

**Key Challenge**: Contrastive objectives learn global semantic alignment between images and texts (identifying matching image-text pairs) but lack modeling of fine-grained visual details. Conversely, autoregressive objectives learn rich visual details but lack language grounding. How to equip a single encoder with both capabilities remains a critical challenge.

**Goal**: Design a simple and scalable pre-training method that allows a single vision encoder to perform exceptionally well across diverse tasks, including classification, localization, grounding, and multimodal understanding.

**Key Insight**: The authors observe that by expanding autoregressive pre-training from pure vision to multimodal (simultaneously generating image patches and text), the encoder is forced to understand both fine-grained visual content and corresponding high-level semantic descriptions, naturally acquiring dual capabilities. The key is to keep the decoder simple enough to prevent the decoder from "slacking," which would otherwise hinder the encoder from learning effective representations.

**Core Idea**: Introduce a lightweight autoregressive decoder as the carrier for the pre-training task. The inputs are the encoder's visual features, and the outputs are the raw patches of the image itself and the corresponding text tokens. The decoder is discarded after pre-training, leaving only the encoder.

## Method

### Overall Architecture
AIMV2's training architecture consists of two components: an encoder and a decoder. The encoder is a standard ViT (ranging from L to 3B parameters) that splits the input image into $14 \times 14$ patches and encodes them into a feature sequence. The decoder is an autoregressive Transformer that receives the encoder's feature sequence as a prefix and then autoregressively generates two types of tokens: (1) raw image patch tokens, which directly regress the RGB pixel values of each patch, and (2) text tokens, corresponding to the textual descriptions (such as alt-text or caption) of the image. Both types of tokens are sequenced alternately or sequentially in a single unified sequence.

### Key Designs

1. **Multimodal Autoregressive Objective**:

    - **Function**: Simultaneously learn visual detail reconstruction and linguistic semantic understanding.
    - **Mechanism**: Given an image, the encoder outputs a feature sequence $\{z_1, ..., z_N\}$. The decoder uses this sequence as a prefix to sequentially predict: first, the raw pixel values of all image patches (regression loss), followed by the token sequence of the corresponding text (cross-entropy loss). The total training loss is $\mathcal{L} = \mathcal{L}_{patch} + \mathcal{L}_{text}$. Image patch prediction uses mean squared error (MSE) to regress directly in the pixel space, while text generation uses standard next-token prediction.
    - **Design Motivation**: Image patch reconstruction forces the encoder to retain fine-grained visual information (texture, edges, color), while text generation forces the encoder to understand high-level semantics (object categories, scene descriptions, actions, etc.). Under the joint constraints of both objectives, the encoder learns representations that are both fine-grained and semantically rich. Compared to contrastive learning, this method avoids complex negative sample selection, making the training pipeline simpler and more direct.

2. **Lightweight Decoder**:

    - **Function**: Act as an auxiliary module during pre-training, forcing the encoder to bear the primary responsibility of information extraction.
    - **Mechanism**: The decoder's parameter size is significantly smaller than the encoder's (typically 1/4 to 1/8 of the encoder). This asymmetric design is crucial—if the decoder is too powerful, it can reconstruct patches and generate text using its own capacity, reducing the encoder to a mere channel for information transfer rather than an effective visual representation extractor. By restricting the decoder's capacity, the information bottleneck is pushed back to the encoder, forcing it to learn high-quality general-purpose representations.
    - **Design Motivation**: This stems from the classical design philosophy of "using auxiliary tasks to train visual representations": the weaker the auxiliary module (decoder), the better the representations learned by the primary module (encoder). A similar concept has been successfully validated in MAE (which utilizes a shallow decoder).

3. **Unified Sequence Modeling**:

    - **Function**: Eliminate modal boundaries between images and texts, enabling the model to naturally learn cross-modal semantic relationships.
    - **Mechanism**: Image patches and text tokens are placed in the same autoregressive sequence and processed by the decoder in a unified next-token prediction manner. This implies that during the generation of text tokens, the decoder can attend to the previously generated image patch tokens, establishing a direct connection between visual details and textual descriptions. Compared to using two separate heads to process images and texts independently, this design facilitates the learning of fine-grained image-text correspondences.
    - **Design Motivation**: Unified modeling eliminates "task conflicts" often encountered in multi-task learning. Both objectives share the same decoder and training pipeline, naturally harmonizing rather than competing with each other.

### Loss & Training
The pre-training data consists of a large-scale image-text pair dataset (similar to the web-crawled data used by CLIP). Training utilizes a standard AdamW optimizer paired with a cosine learning rate scheduler. The key hyperparameters are the weight ratios between the patch reconstruction loss and the text generation loss. The encoder is trained starting from random initialization (without using pre-trained weights from MAE or DINO as initialization). The model series scales from 0.3B (ViT-L) to 2.7B (ViT-3B).

## Key Experimental Results

### Main Results

| Task/Dataset | AIMV2-L (0.3B) | AIMV2-3B (2.7B) | CLIP-L | SigLIP-L | DINOv2-L |
|-----------|---------------|----------------|--------|----------|----------|
| ImageNet-1k (frozen trunk) | 87.9 | 89.5 | 80.2 | 82.1 | 86.3 |
| COCO Detection (AP) | 56.2 | 59.4 | 48.7 | 50.3 | 57.8 |
| RefCOCO (grounding) | 81.1 | 83.6 | 72.4 | 74.8 | 79.2 |
| VQAv2 | 80.2 | 82.7 | 78.5 | 79.1 | 71.3 |
| TextVQA | 72.6 | 75.3 | 68.2 | 70.1 | 58.7 |
| MMBench | 74.5 | 76.3 | 69.8 | 71.2 | 62.5 |

### Ablation Study

| Configuration | ImageNet Acc | VQAv2 Acc | Description |
|------|-------------|----------|------|
| Full AIMV2 (patch + text) | 87.9 | 80.2 | Full multimodal objective |
| Patch reconstruction only (w/o text) | 86.4 | 67.5 | Degrades to AIMv1, multimodal capability decreases significantly |
| Text prediction only (w/o patch) | 82.3 | 79.8 | Visual detail capability decreases |
| Large decoder (1/2 encoder) | 86.1 | 77.4 | Decoder is too strong, encoder degrades |
| Small decoder (1/8 encoder) | 87.5 | 79.6 | Lightweight decoder performance is near-optimal |

### Key Findings
- **Significant synergistic effect of dual objectives**: Removing either objective leads to a substantial decline in the corresponding capability. Specifically, removing the text objective drops VQAv2 accuracy from 80.2% to 67.5% ($-12.7\%$), while removing the patch objective drops ImageNet accuracy from 87.9% to 82.3% ($-5.6\%$). This underscores the necessity of the multimodal autoregressive objective.
- **Encoder-decoder asymmetry is crucial**: Optimal results are achieved when the decoder parameter size is 1/4 of the encoder's. Scaling it up to 1/2 actually degrades encoder performance, validating the design hypothesis that a "weak decoder pushes for a strong encoder."
- **Excellent scalability**: From 0.3B to 2.7B parameters, ImageNet accuracy steadily increases from 87.9% to 89.5% with no signs of saturation. This indicates that the scaling law for autoregressive pre-training remains valid.
- AIMV2 outperforms DINOv2 on pure vision tasks (detection, grounding) and outperforms CLIP/SigLIP on multimodal tasks (VQA, MMBench), truly achieving "one encoder fits all."
- The distilled version AIMV2-L-distilled, distilled from the 3B model, further enhances multimodal performance at the 0.3B parameter scale (MMBench 76.3 vs 74.5).

## Highlights & Insights
- **An exemplar of "simplicity is power"**: The pre-training process of AIMV2 is remarkably simple—one encoder and one decoder with two loss functions, requiring no complex techniques such as negative sample construction, momentum encoders, or teacher-student distillation. This simplicity not only reduces engineering complexity but, more importantly, makes model scaling straightforward.
- **Deep insight into the weak decoder design**: Forcing the encoder to learn superior representations by restricting decoder capacity is thoroughly validated in a multimodal autoregressive context, though not a brand-new concept. This provides valuable guidance for other scenarios utilizing auxiliary pre-training tasks.
- **Unified autoregressive modeling eliminates modal barriers**: Placing image patches and text tokens in the same autoregressive sequence is an elegant way of multimodal fusion. It provides a promising pathway toward constructing truly unified multimodal foundation models.

## Limitations & Future Work
- The pre-training data relies on web-crawled image-text pairs, and issues regarding data quality and bias were not extensively discussed.
- The encoder is evaluated by freezing it after pre-training, and the performance gap under fine-tuning scenarios compared to contrastive models has not been evaluated.
- The decoder is discarded during inference, meaning its learned knowledge is not leveraged; whether the decoder can be retained for generative tasks remains to be explored.
- Only image understanding tasks were evaluated; video understanding and temporal scaling have yet to be verified.
- The 3B scale encoder incurs significant computational overhead during inference, making efficiency a crucial consideration for practical deployment.

## Related Work & Insights
- **vs CLIP / SigLIP**: Contrastive learning methods possess clear advantages in multimodal alignment but lack fine-grained visual capability. AIMV2 fills this gap with autoregressive patch reconstruction, outperforming them on both multimodal and pure vision tasks.
- **vs AIMv1**: AIMv1 gold-standardizes image-only autoregressive pre-training but lacks language grounding. By incorporating a text generation objective, AIMV2 acquires strong multimodal understanding capabilities at minimal cost while also boosting pure vision performance.
- **vs DINOv2**: Based on self-supervised distillation, DINOv2 performs exceptionally well on vision tasks but lacks language grounding. AIMV2-3B outperforms DINOv2-L in detection and grounding while possessing multimodal understanding capabilities that DINOv2 lacks.
- **vs MAE**: Both share autoregressive/reconstructive pre-training. MAE reconstructs in the pixel space; AIMV2 also regresses raw patches but incorporates an additional text objective, obtaining semantic understanding capabilities missing in MAE.

## Rating
- Novelty: ⭐⭐⭐⭐ The approach of expanding autoregressive pre-training to multimodality is clear and natural, and the weak decoder design has significant depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers a wide array of tasks including classification, detection, grounding, and multimodal understanding, with comprehensive ablations and sufficient scaling analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ The arguments are concise and compelling, and the tables/figures are exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Apple's open-sourcing of the 3B vision encoder, with support for PyTorch, JAX, and MLX, holds extremely high practical value and represents a major milestone in the field of vision encoder pre-training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Post-pre-training for Modality Alignment in Vision-Language Foundation Models](post-pre-training_for_modality_alignment_in_vision-language_foundation_models.md)
- [\[ICCV 2025\] SCAN: Bootstrapping Contrastive Pre-training for Data Efficiency](../../ICCV2025/multimodal_vlm/scan_bootstrapping_contrastive_pre-training_for_data_efficiency.md)
- [\[ICLR 2026\] GranViT: A Fine-Grained Vision Model For Autoregressive Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/granvit_a_fine-grained_vision_model_for_autoregressive_multimodal_large_language.md)
- [\[CVPR 2025\] BadVision: Stealthy Backdoor Attack in Self-Supervised Learning Vision Encoders for Large Vision Language Models](stealthy_backdoor_attack_in_self-supervised_learning_vision_encoders_for_large_v.md)
- [\[CVPR 2025\] A Two-Stage Progressive Pre-training using Multi-Modal Contrastive Masked Autoencoders](multi-modal_contrastive_masked_autoencoders_a_two-stage_progressive_pre-training.md)

</div>

<!-- RELATED:END -->
