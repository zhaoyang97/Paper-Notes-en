---
title: >-
  [Paper Note] lbGen: Low-Biased General Annotated Dataset Generation
description: >-
  [CVPR 2025][Image Generation][dataset bias] The lbGen framework is proposed to fine-tune Stable Diffusion through bi-level semantic alignment (global adversarial + individual cosine similarity) and quality assurance losses. Using only category names, it generates a low-biased general annotated dataset. Backbones pretrained on this dataset outperform those trained on real ImageNet data by 1.7%~2.1% in average transfer accuracy.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "dataset bias"
  - "synthetic dataset"
  - "diffusion model"
  - "CLIP"
  - "bi-level semantic alignment"
  - "quality assurance"
  - "transfer learning"
date: 2026-05-08
content_hash: b7ab617df69515a2
---

# lbGen: Low-Biased General Annotated Dataset Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.10831](https://arxiv.org/abs/2412.10831)  
**Code**: [https://github.com/vvvvvjdy/lbGen](https://github.com/vvvvvjdy/lbGen)  
**Area**: Image Generation  
**Keywords**: dataset bias, synthetic dataset, diffusion model, CLIP, bi-level semantic alignment, quality assurance, transfer learning

## TL;DR

The lbGen framework is proposed to fine-tune Stable Diffusion through bi-level semantic alignment (global adversarial + individual cosine similarity) and quality assurance losses. Using only category names, it generates a low-biased general annotated dataset. Backbones pretrained on this dataset outperform those trained on real ImageNet data by 1.7%~2.1% in average transfer accuracy.

## Background & Motivation

**Background**: Pretraining backbone networks on general annotated datasets (such as ImageNet) is a foundational step for various vision tasks. Recent advances in diffusion models have enabled the direct synthesis of annotated image data.

**Limitations of Prior Work**: (1) Manually collected datasets like ImageNet exhibit implicit data biases (e.g., fixed backgrounds, styles, and object locations for specific categories). Backbone networks capture these non-transferable shortcut features during pretraining, leading to degraded cross-domain/cross-category generalization. (2) Existing synthetic datasets (e.g., GenRobust, RealFake) mainly mimic the ImageNet distribution without considering bias reduction. (3) Re-collecting low-biased data manually is extremely expensive and inevitably introduces new biases.

**Key Challenge**: High accuracy on the ImageNet validation set does not equate to strong generalization capability — bias forces the model to rely on shortcut features instead of transferable semantic features.

**Key Insight**: Leveraging the low-biased semantic space defined by CLIP, this work fine-tunes a diffusion model via reinforcement learning to directly generate low-biased images that align with the semantic distribution, without utilizing any external biased images.

## Method

### Overall Architecture

Based on Stable Diffusion 1.5 + LoRA fine-tuning, the method takes only the 1000 category names of ImageNet-1K as input. The training framework consists of two modules: a bi-level semantic alignment module (core) and a quality assurance module (auxiliary), optimized using a reinforcement learning paradigm.

### Key Designs

**1. Entire Dataset Alignment**
- **Function**: Aligns the CLIP feature distribution of all generated images with the overall semantic distribution of the 1000 text categories.
- **Mechanism**: Using a Linear-ReLU-Linear discriminator $\mathcal{D}_\phi$, adversarial learning is performed by randomly selecting text features of categories **different** from the current image as positive samples, and the generated image features as negative samples:
  $$\mathcal{L}_{en} = \log(\mathcal{D}_\phi(f_{c_j})) + \log(1 - \mathcal{D}_\phi(f_{im_i}))$$
- **Design Motivation**: By avoiding matching text features of the same class, the entire synthetic dataset distribution is encouraged to approach the global distribution of the semantic space rather than performing category-level alignment.

**2. Individual Image Alignment**
- **Function**: Ensures each generated image precisely matches the semantic description of its corresponding category.
- **Mechanism**: Using a simple "photo of $c_i$" as the low-biased semantic description, the CLIP image-text cosine similarity is maximized:
  $$\mathcal{L}_{in} = 1 - \frac{f_{im_i} \cdot f_{p_{c_i}}}{\|f_{im_i}\| \cdot \|f_{p_{c_i}}\|}$$
- **Design Motivation**: While global alignment guarantees distribution consistency, it cannot precisely control the specific category of each image, which necessitates individual-level constraints.

**3. Quality Assurance**
- **Function**: Prevents image quality degradation caused by the semantic alignment training.
- **Mechanism**: Converts the score $Q(im_i)$ (ranging from [1, 5]) of the Q-ALIGN image quality assessment model into a loss:
  $$\mathcal{L}_q = 1 - \frac{Q(im_i)}{5}$$
- **Design Motivation**: Relying solely on semantic constraints leads to style or quality degradation; the quality assurance loss establishes a baseline for image fidelity.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{bi} + \lambda_1 \mathcal{L}_q$$

where $\mathcal{L}_{bi} = \mathcal{L}_{en} + \mathcal{L}_{in}$. The model is trained using a reinforcement learning paradigm, where gradients are computed only at 5 out of 50 denoising steps to save GPU memory.

## Key Experimental Results

### Main Results — Average Top-1 Accuracy on Eight Transfer Learning Datasets

| Backbone | Pretraining Data | IN-val | 8 Datasets Avg. |
|---|---|---|---|
| ResNet50 | IN-Real | 76.2 | 71.5 |
| ResNet50 | IN-RealFake | 69.8 | 71.8 |
| ResNet50 | **IN-lbGen** | 46.1 | **73.2** |
| ViT-S | IN-Real | 78.7 | 72.3 |
| ViT-S | IN-RealFake | 72.3 | 70.8 |
| ViT-S | **IN-lbGen** | 46.3 | **74.4** |

**Key Findings**: Although the IN-val accuracy of the backbone pretrained on lbGen is only ~46%, its transfer accuracy significantly outperforms the baselines, demonstrating that ImageNet validation accuracy is not positively correlated with generalization capability.

### Visual Perception Tasks (COCO Detection / ADE20K Segmentation)

| Pretraining Data | COCO AP^box (0.2×) | ADE20K mIoU (0.2×) |
|---|---|---|
| IN-Real | 29.14 | 32.10 |
| IN-lbGen | **30.68 (+1.54)** | **33.57 (+1.47)** |

lbGen achieves the most significant advantages when using 20% downstream data.

### Bias Metric Experiments

| Backbone | Pretraining Data | TI↓ (Texture Bias) | CB_avg↑ (Context) | BG_Gap↓ (Background) |
|---|---|---|---|---|
| ResNet50 | IN-Real | 60.9 | 60.0 | 6.8 |
| ResNet50 | IN-lbGen | **56.1** | **64.7** | **6.4** |
| ViT-S | IN-Real | 67.0 | 61.8 | 6.7 |
| ViT-S | IN-lbGen | **57.2** | **66.0** | **6.1** |

Under all three bias metrics, the proposed method comprehensively outperforms models pretrained on real data.

### Key Findings

1. **Data bias is quantifiable**: High IN-val accuracy $\neq$ high generalization; bias is the root cause.
2. **Greater benefits in few-shot scenarios**: The less downstream data available, the more prominent the advantages of lbGen become (Figure 3).
3. **Efficacy of semantic space alignment**: The text semantic space of CLIP indeed provides a low-biased representation anchor.

## Highlights & Insights

1. **First to directly generate a low-biased dataset**: Diverging from the traditional "collect then de-bias" paradigm, this work directly addresses the bias problem from the generation end.
2. **Zero-image training**: The diffusion model is fine-tuned using only 1000 category names, without introducing any external biased images.
3. **Counter-intuitive finding**: Synthetic data with 46% IN-val accuracy performs stronger in transfer learning than real data with 76% IN-val accuracy.
4. **Lightweight**: Training costs are kept highly manageable through LoRA fine-tuning and a 5-step gradient strategy.

## Limitations & Future Work

1. Extremely low IN-val accuracy (46%) demands caution when applying it to in-domain scenarios.
2. Validated only on 1K categories; the scaling behavior to larger categories (e.g., 21K) remains to be verified.
3. Relies heavily on the quality of CLIP's semantic space — CLIP itself may exhibit inherent biases.
4. The quality assurance module utilizes the scoring model Q-ALIGN, which might introduce implicit quality-preference bias.
5. Evaluated only on ResNet50 and ViT-S; whether the advantages persist for larger models (e.g., ViT-L) remains unknown.

## Related Work & Insights

- **RealFake (Yuan et al.)**: Synthesizes data after learning the ImageNet distribution but does not mitigate bias $\rightarrow$ essentially replicating the biases.
- **GenRobust (Bansal et al.)**: Fine-tunes a diffusion model on ImageNet and uses carefully designed prompts $\rightarrow$ still constrained by the original distribution.
- **CLIP align**: Utilizing CLIP's multimodal alignment capability as a "de-biasing" tool is a paradigm worth promoting — it can scale to other scenarios requiring low-bias representations (e.g., fairness, domain adaptation).

## Rating

⭐⭐⭐⭐ — High novelty in perspective (addressing data bias from the generation end for the first time), with comprehensive experiments covering bias metrics; the counter-intuitive results are convincing. However, the assumption of relying on CLIP's semantic space requires deeper theoretical support.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] A Bias-Free Training Paradigm for More General AI-generated Image Detection](a_bias-free_training_paradigm_for_more_general_ai-generated_image_detection.md)
- [\[ICCV 2025\] VIGFace: Virtual Identity Generation for Privacy-Free Face Recognition Dataset](../../ICCV2025/image_generation/vigface_virtual_identity_generation_for_privacy-free_face_recognition_dataset.md)
- [\[ICCV 2025\] A0: An Affordance-Aware Hierarchical Model for General Robotic Manipulation](../../ICCV2025/image_generation/a0_affordance_aware_hierarchical_model_robotic_manipulation.md)
- [\[CVPR 2025\] ORIDa: Object-Centric Real-World Image Composition Dataset](orida_object-centric_real-world_image_composition_dataset.md)
- [\[NeurIPS 2025\] CaMiT: A Time-Aware Car Model Dataset for Classification and Generation](../../NeurIPS2025/image_generation/camit_a_time-aware_car_model_dataset_for_classification_and_generation.md)

</div>

<!-- RELATED:END -->
