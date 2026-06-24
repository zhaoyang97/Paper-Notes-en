---
title: >-
  [Paper Note] Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval
description: >-
  [ECCV 2024][LLM Evaluation][zero-shot composed image retrieval] A Slerp-based ZS-CIR method is proposed, which directly fuses the image and text embeddings of VLP models via Spherical Linear Interpolation (Slerp) to construct composed query representations. Combined with Text-Anchored-Tuning (TAT), which fine-tunes the image encoder using LoRA to narrow the modality gap, this method achieves state-of-the-art (SOTA) performance on CIRR, CIRCO, and FashionIQ.
tags:
  - "ECCV 2024"
  - "LLM Evaluation"
  - "zero-shot composed image retrieval"
  - "spherical linear interpolation"
  - "text-anchored tuning"
  - "LoRA"
  - "CLIP/BLIP"
date: 2026-05-08
content_hash: 1622a7012c0c5b5d
---

# Spherical Linear Interpolation and Text-Anchoring for Zero-shot Composed Image Retrieval

**Conference**: ECCV 2024  
**arXiv**: [2405.00571](https://arxiv.org/abs/2405.00571)  
**Code**: Not provided  
**Area**: Vision-Language / Image Retrieval  
**Keywords**: zero-shot composed image retrieval, spherical linear interpolation, text-anchored tuning, LoRA, CLIP/BLIP

## TL;DR
A Slerp-based ZS-CIR method is proposed, which directly fuses the image and text embeddings of VLP models via Spherical Linear Interpolation (Slerp) to construct composed query representations. Combined with Text-Anchored-Tuning (TAT), which fine-tunes the image encoder using LoRA to narrow the modality gap, this method achieves state-of-the-art (SOTA) performance on CIRR, CIRCO, and FashionIQ.

## Background & Motivation

**Background**: Composed Image Retrieval (CIR) is a task that retrieves a target image using a reference image and a modification text. Supervised CIR methods rely on expensive triplet annotations $\langle\text{reference image}, \text{text intent}, \text{target image}\rangle$, which limits scalability. Zero-shot CIR (ZS-CIR) methods use general image-text pairs for training and exhibit better generalization ability.

**Limitations of Prior Work**: Current mainstream ZS-CIR methods (e.g., Pic2Word, SEARLE, LinCIR) adopt a "pseudo-word token" strategy—mapping the image to text word tokens via a projection module, concatenating them with the text prompts, and passing them through the text encoder to generate the composed representation. This approach has two issues: (1) the projection module distorts the original image representation; (2) the composed embedding is restricted to the text encoder's output space, failing to fully capture joint image-text information.

**Key Challenge**: Image and text embeddings of VLP models lie on the same hypersphere (due to being trained with cosine similarity), but there is a significant "modality gap" between the two modalities, which limits the effectiveness of direct interpolation between them.

**Goal**: To design a simple yet effective image-text composition scheme that avoids the defects of pseudo-word tokens and narrows the modality gap to improve composed retrieval performance.

**Key Insight**: Since the embeddings of VLP models lie on a hypersphere, Spherical Linear Interpolation (Slerp) is the most natural way of fusion. The modality gap can be narrowed by freezing the text encoder and fine-tuning the image encoder with LoRA.

**Core Idea**: Direct Slerp interpolation of image and text embeddings combined with text-anchored tuning to narrow the modality gap, realizing zero-shot CIR simply and effectively.

## Method

### Overall Architecture
The method consists of two phases: (1) **Slerp-based ZS-CIR**: during inference, spherical linear interpolation is used to fuse the query image embedding and text embedding to obtain the composed representation; (2) **Text-Anchored-Tuning (TAT)**: during the training phase, the text encoder is frozen, and only the image encoder is fine-tuned using LoRA to bring the image embeddings closer to their corresponding text embeddings. The combination of both yields the final high-performance ZS-CIR model.

### Key Designs

1. **Spherical Linear Interpolation (Slerp) Retrieval**

    - **Function**: Finding an intermediate representation of image and text embeddings on the VLP embedding hypersphere to serve as the composed query.
    - **Mechanism**: Given an image embedding $\mathbf{v}$ and a text embedding $\mathbf{w}$ (both $L_2$ normalized), the composed embedding is constructed via Slerp:
    $$\mathbf{c} = \text{Slerp}(\mathbf{v}, \mathbf{w}; \alpha) = \frac{\sin((1-\alpha)\theta)}{\sin(\theta)}\mathbf{v} + \frac{\sin(\alpha\theta)}{\sin(\theta)}\mathbf{w}$$
      where $\theta = \cos^{-1}(\mathbf{v} \cdot \mathbf{w})$ is the angle between the two embeddings, and $\alpha \in [0,1]$ controls the trade-off ratio between the image and text.
    - **Design Motivation**: VLP models are trained using scaled cosine similarity, meaning that image and text embeddings naturally distribute on a hypersphere. Slerp interpolates along the spherical arc path (instead of simple linear interpolation), preserving the distribution characteristics of embeddings on the hypersphere. Key observation: text-only retrieval typically outperforms image-only retrieval, so setting $\alpha \geq 0.8$ (biased toward text) yields the best performance.
    - **No training required**: This operation is performed purely during inference, with no additional projection modules or training required.

2. **Text-Anchored-Tuning (TAT)**

    - **Function**: Narrowing the modality gap between image and text embeddings in VLP models.
    - **Mechanism**: Freeze the text encoder $E_T$ (retaining the strong representation capability of text as an "anchor"), and only use LoRA to add a small number of trainable parameters $\mathcal{P}_{lora}$ (LoRA_α=16, rank=16, dropout=0.1) on the image encoder $E_I$ to realign image embeddings with their corresponding text embeddings. The training objective is the standard batch contrastive loss:
    $$\mathcal{L}_{cont.} = \mathcal{L}_{I2T} + \mathcal{L}_{T2I}$$
      where the temperature $\tau$ is fixed to $1/0.07$ to ensure training stability.
    - **Design Motivation**: (1) Text plays a dominant role in CIR (text-only even outperforms some methods), making it crucial to preserve the original capability of the text encoder; (2) LoRA retains the prior knowledge of the image encoder while allowing fine-grained alignment; (3) aligning requires training only <0.5% of the parameters and converges in a single epoch.
    - **Training Efficiency**: Training takes less than 0.5 hours with C-B32 on the LLaVA-Align dataset.

3. **$\alpha$ Parameter Dataset Adaptation**

    - **Function**: Adjusting the weights of text and image in Slerp based on dataset characteristics.
    - **Mechanism**: The text intent is more critical in the CIRR dataset, hence $\alpha=0.9$ (more biased toward text) is set; the image is also important in CIRCO and FashionIQ, so $\alpha=0.8$ is set.
    - **Design Motivation**: Different retrieval domains differ in their reliance on images and text, and $\alpha$ provides a user-adjustable control knob. Experiments show that $\alpha=1.0$ (text only) leads to a significant performance drop, proving that image information is indispensable.

4. **Inference Flow**

    - **Function**: Completing composed image retrieval.
    - **Mechanism**: (1) Encode gallery images through $E_I$ to obtain $\mathbf{v}_g$; (2) Obtain $\mathbf{v}_q$ and $\mathbf{w}_q$ by passing the query image and text through $E_I$ and $E_T$, respectively; (3) Fuse them using Slerp to obtain $\mathbf{c}_q$; (4) Calculate the cosine similarity between $\mathbf{c}_q$ and all $\mathbf{v}_g$ for ranking.
    - **Design Motivation**: No specific text template (e.g., "a photo of [$]") is required, and the query text is used directly, preventing template selection from becoming a performance bottleneck.

### Loss & Training
- **Loss Function**: Standard batch-wise contrastive loss (same as VLP pre-training)
  $$\mathcal{L}_{I2T} = -\frac{1}{N_B}\sum_{i=1}^{N_B}\log\frac{\exp(\mathbf{v}_i^T \cdot \mathbf{w}_i / \tau)}{\sum_{j=1}^{N_B}\exp(\mathbf{v}_i^T \cdot \mathbf{w}_j / \tau)}$$
- **Training Data**: Laion-2M (default) / LLaVA-Align (585K) / CC3M (2.3M)
- **Training Configuration**: 8×A100-80GB, batch size 1024, AdamW lr=1e-4, single epoch
- **Trainable Parameters**: < 0.5% (LoRA parameters only)

## Key Experimental Results

### CIRR Main Results

| Backbone | Method | R@1 | R@5 | R@10 | Rs@1 | Rs@2 | Rs@3 |
|----------|------|-----|------|------|------|------|------|
| CLIP-L/14 | Pic2Word | 23.90 | 51.70 | 65.30 | - | - | - |
| CLIP-L/14 | SEARLE | 24.22 | 52.41 | 66.29 | 53.71 | 74.63 | 87.61 |
| CLIP-L/14 | LinCIR | 25.04 | 53.25 | 66.68 | 57.11 | 77.37 | 88.89 |
| CLIP-L/14 | Slerp (No Training) | 24.43 | 49.93 | 62.29 | 57.71 | 77.59 | 88.80 |
| CLIP-L/14 | **Slerp+TAT** | **30.94** | **59.40** | **70.94** | **64.70** | **82.92** | **92.31** |
| BLIP-L/16 | Slerp | 28.60 | 55.37 | 65.66 | 65.16 | 83.90 | 92.05 |
| BLIP-L/16 | **Slerp+TAT** | **33.98** | **61.74** | **72.70** | **68.55** | **85.11** | **93.21** |

### CIRCO Main Results

| Backbone | Method | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|----------|------|-------|--------|--------|--------|
| CLIP-L/14 | SEARLE | 11.68 | 12.73 | 14.33 | 15.12 |
| CLIP-L/14 | LinCIR | 12.59 | 13.58 | 15.00 | 15.85 |
| CLIP-L/14 | **Slerp+TAT** | **18.46** | **19.41** | **21.43** | **22.41** |
| BLIP-L/16 | **Slerp+TAT** | **17.84** | **18.44** | **20.24** | **21.07** |

### TAT Ablation Study

| Configuration | mAP@5 | mAP@10 | mAP@25 | mAP@50 |
|------|-------|--------|--------|--------|
| (a) LLaVA-Align (0.58M) | 17.05 | 18.23 | 20.11 | 21.05 |
| (b) CC3M (2.3M) | 16.98 | 17.82 | 19.62 | 20.58 |
| (c) Laion-2M (Default) | **18.46** | **19.41** | **21.43** | **22.41** |
| (e) None-anchoring (both towers fine-tuned) | 8.26 | 8.90 | 10.07 | 10.71 |
| (f) Image-anchoring (freeze image, tune text) | 7.54 | 7.73 | 8.79 | 9.30 |
| (g) Pic2Word-Laion-2M | 8.93 | 9.96 | 11.50 | 12.02 |

### Key Findings
- **Slerp only (no training)** already matches or surpasses pseudo-word token methods that require training (e.g., SEARLE), verifying the soundness of spherical interpolation.
- **TAT requires only 0.58M data and a single epoch** to significantly outperform methods trained on millions of data over dozens of epochs.
- **Text anchoring is critical**: None-anchoring (8.26 mAP@5) and Image-anchoring (7.54) perform far worse than Text-anchoring (18.46), proving the importance of keeping the text encoder fixed.
- **$\alpha=1.0$ (text only) leads to a significant performance drop**, showing that the image embedding still plays a crucial role in Slerp.
- Models trained with TAT can also serve as better initialization checkpoints for supervised CIR methods.

## Highlights & Insights
- **Elegant simplicity returning to the core**: While prior works designed complex projection modules and pseudo-word tokens, this paper points out that the geometric properties of VLP embeddings on the hypersphere can be directly exploited—Slerp is indeed the most natural interpolation method on a hypersphere. This philosophy of "simplifying after grasping the essence" is highly elegant.
- **Insight on asymmetric fine-tuning**: Text is more important than images in CIR $\rightarrow$ keep the text encoder fixed as an anchor $\rightarrow$ tune only the image encoder to align toward text. This design logic has a clear chain and is heavily validated by ablation experiments (text-anchoring $\gg$ none-anchoring $\gg$ image-anchoring).

## Limitations & Future Work
- $\alpha$ needs to be manually set for each dataset, which lacks adaptive adjustment.
- TAT uses a fixed temperature $\tau$, leaving the effect of dynamic temperature adjustment on modality gap convergence unexplored.
- Slerp+TAT is not consistently optimal under some FashionIQ settings (Slerp is better on C-B32 R@10), possibly because the weight of image information in the fashion domain differs from text.
- Only two VLP models, CLIP and BLIP, have been evaluated, and the applicability to newer models (such as SigLIP, EVA-CLIP) remains unverified.

## Related Work & Insights
- **vs Pic2Word [Saito et al.]**: Pic2Word converts images into pseudo-word tokens via a projection module and feeds them to the text encoder, whereas Slerp directly interpolates in the embedding space, preventing information loss.
- **vs SEARLE [Baldrati et al.]**: SEARLE requires 3M image-text pairs and 5.5M text samples for training, while TAT outperforms it with only 0.58M data in a single epoch.
- **vs LinCIR [Gu et al.]**: LinCIR performs linear composition in the embedding space, while Slerp performs spherical linear interpolation—the latter being more geometrically sound on the hypersphere.
- **vs LoRA [Hu et al.]**: TAT realizes parameter-efficient fine-tuning based on LoRA, but the key innovation lies in the "text-anchoring" strategy rather than LoRA itself (ablation experiments demonstrate that simply adding LoRA to Pic2Word yields very marginal improvements).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The direct Slerp interpolation approach is extremely elegant and supported by geometric insights, and the Text-Anchored-Tuning design is exquisite.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three benchmarks (CIRR/CIRCO/FashionIQ), three backbones, and detailed ablations (datasets, anchoring strategies, $\alpha$, supervised CIR initialization).
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of method, coherent motivation and mathematical derivation.
- Value: ⭐⭐⭐⭐⭐ Extremely high training efficiency (<0.5h), with a massive performance lead, showcasing outstanding practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] UniGoal: Towards Universal Zero-shot Goal-oriented Navigation](../../CVPR2025/llm_evaluation/unigoal_towards_universal_zero-shot_goal-oriented_navigation.md)
- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zerosh.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](../../ACL2026/llm_evaluation/zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2025\] EditInspector: A Benchmark for Evaluation of Text-Guided Image Edits](../../ACL2025/llm_evaluation/editinspector_a_benchmark_for_evaluation_of_text-guided_image_edits.md)

</div>

<!-- RELATED:END -->
