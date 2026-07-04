---
title: >-
  [Paper Note] RORem: Training a Robust Object Remover with Human-in-the-Loop
description: >-
  [CVPR 2025][Image Generation][Object Removal] RORem proposes a "Human-in-the-Loop" semi-supervised data generation paradigm. It first generates removal results using an initial model, leverages human annotators to filter high-quality samples, and then trains a discriminator to automate subsequent filtering. This iteratively constructs a 200K+ high-quality paired object removal dataset, leading to a fine-tuned SDXL model that outperforms prior methods by 18%+ in success rate.…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Object Removal"
  - "Human-in-the-Loop"
  - "Semi-Supervised Learning"
  - "Image Inpainting"
  - "Diffusion Model Distillation"
date: 2026-05-08
content_hash: cad5db04b81c6c87
---

# RORem: Training a Robust Object Remover with Human-in-the-Loop

**Conference**: CVPR 2025  
**arXiv**: [2501.00740](https://arxiv.org/abs/2501.00740)  
**Code**: [https://github.com/leeruibin/RORem](https://github.com/leeruibin/RORem)  
**Area**: Image Generation  
**Keywords**: Object Removal, Human-in-the-Loop, Semi-Supervised Learning, Image Inpainting, Diffusion Model Distillation

## TL;DR
RORem proposes a "Human-in-the-Loop" semi-supervised data generation paradigm. It first generates removal results using an initial model, leverages human annotators to filter high-quality samples, and then trains a discriminator to automate subsequent filtering. This iteratively constructs a 200K+ high-quality paired object removal dataset, leading to a fine-tuned SDXL model that outperforms prior methods by 18%+ in success rate. After distillation, the model requires only 4 steps (<1 second) for generation.

## Background & Motivation

**Background**: Object removal is a core requirement of image editing. Existing methods mainly rely on a self-supervised training paradigm—randomly masking image regions and training the model to reconstruct the original content. Methods based on pre-trained diffusion models (e.g., PowerPaint, SDXL-Inpainting) have achieved promising results.

**Limitations of Prior Work**: The self-supervised paradigm exhibits a fundamental ambiguity: during training, the model learns to "reconstruct the masked content" (including the object itself), but during testing, the goal is to "remove the object and restore the background". Consequently, models frequently reconstruct objects in the masked area (e.g., re-drawing a masked bird), or produce incomplete removals and blurry compositions, resulting in a low success rate.

**Key Challenge**: Resolving this ambiguity demands high-quality paired data (images with and without objects), which is extremely difficult to obtain. ObjectDrop employed photographers to capture physical pairs, obtaining only 2K pairs, which are not publicly released; synthetic data lacks quality and diversity.

**Goal**: Design an iteratively scalable data construction framework that starting from a small initial dataset, progressively accumulates a large-scale, high-quality paired dataset for object removal via a human feedback loop, in order to train a robust object removal model.

**Key Insight**: Leverage a positive loop of "self-training + human feedback"—model generates removal samples $\rightarrow$ humans filter high-quality samples $\rightarrow$ a discriminator is trained to automate filtering $\rightarrow$ the dataset expands $\rightarrow$ model is retrained $\rightarrow$ model becomes stronger and generates more high-quality samples.

**Core Idea**: Train a removal quality discriminator using human feedback to automate the filtering process of high-quality paired data, achieving efficient scale-up of the dataset from 60K to 200K.

## Method

### Overall Architecture
A three-stage iterative pipeline: (1) Initialization: Fine-tune SDXL-Inpainting on a 60K initial dataset composed of RORD (15K real pairs from videos) and MULAN (45K synthetic pairs); (2) Human Labeling: Use the current model to generate removal results on OpenImages, then collect labels from 10 annotators to filter high-quality samples; (3) Auto Labeling: Train a discriminator $D_\phi$ using the human feedback data to automatically replace human annotation. Stages 2 and 3 are iterated for 3 rounds, resulting in 200K+ paired data. There is also a final fine-tuning stage: use 1200 high-resolution pairs from DIV2K/Flickr2K to enhance output quality.

### Key Designs

1. **Human Feedback-driven Positive Data Loop**:

    - **Function**: Build a large-scale, high-quality paired object removal dataset starting from limited initial data.
    - **Mechanism**: In each round, sample images and masks from OpenImages (excluding domains like clothes/body, limiting to maximum 500 per category), generate removal results using the current model, and manually label them as "success/failure". Successful samples are added to the training set to retrain the model. The 3 rounds of human annotation filtered out 4182 / 7008 / 6133 valid samples respectively.
    - **Design Motivation**: The model strengthens as training progresses $\rightarrow$ the success rate increases $\rightarrow$ more high-quality samples are filtered in each round $\rightarrow$ the dataset becomes larger and more diverse $\rightarrow$ the model becomes stronger, forming a positive feedback loop.

2. **Object Removal Quality Discriminator (Discriminator $D_\phi$)**:

    - **Function**: Automatically replace human annotation.
    - **Mechanism**: Utilize the down and middle blocks of SDXL as a backbone, append LoRA (rank=4) and convolutional layers to output a 0-1 confidence score. The input is a triplet $(\mathbf{x}_e, \mathbf{x}_s, \mathbf{m})$, trained with yes/no human annotations as supervision signals. Setting a threshold of 0.9, only samples with confidence over 0.9 are automatically annotated as high-quality. The discriminator is re-fine-tuned after each round of human annotation.
    - **Design Motivation**: Human annotators can only process ~10K samples per round, whereas the discriminator can quickly process 30K-95K samples, dramatically increasing data collection efficiency. Experiments show that the discriminator's success rate estimation deviates from human judgment by less than 3%.

3. **LoRA + LCM Distillation Acceleration**:

    - **Function**: Compress 50-step inference down to 4 steps.
    - **Mechanism**: Introduce LoRA (rank=64) into the trained RORem, and train it with the LCM distillation loss $\mathcal{L}_\psi = \|f_\psi(\mathbf{x}_e^{t+k}, t+k) - f_\theta(\hat{\mathbf{x}}_e^t, t)\|_2^2$, where $\hat{\mathbf{x}}_e^t$ is sampled via the original DDIM model with $k=20$ steps. Set the text condition to empty and the CFG scale to 1 (since the removal task does not require text guidance), reducing memory and computation.
    - **Design Motivation**: The original 50-step model requires 4s+, whereas the distilled version takes only 0.5s for 4 steps, with only a marginal 1.4% drop in success rate.

### Loss & Training
Standard diffusion denoising loss $\mathcal{L}_\theta = \mathbb{E}[\|\epsilon - G_\theta(\mathbf{x}_e^t, \bar{\mathbf{x}}_s, \mathbf{m}, t)\|_2^2]$. Key design: Instead of directly concatenating the full original image, the masked region of the source image is set to zero before concatenation ($\bar{\mathbf{x}}_s = \mathbf{x}_s \cdot (1-\mathbf{m})$). This prevents residual artifacts from transparent objects. Training lasts 50K steps per round, batch size 192, using 16×A100 GPUs.

## Key Experimental Results

### Main Results (512×512)

| Method | Success Rate (Human) ↑ | Success Rate ($D_\phi$) ↑ | PSNR ↑ | LPIPS ↓ | DINO ↑ | Time (s) |
|------|-----------|-------------------|-------|--------|-------|---------|
| Lama | 55.4% | 48.6% | **33.06** | 2.99 | 0.77 | 0.15 |
| PowerPaint | 55.8% | 56.8% | 28.41 | 6.06 | 0.75 | 1.98 |
| SDXL-INP | 15.8% | 16.0% | 26.03 | 4.72 | 0.76 | 4.52 |
| **RORem** | **76.2%** | **75.6%** | 31.10 | **2.49** | **0.80** | 4.03 |
| **RORem-4S** | **74.8%** | **73.2%** | 29.33 | 3.65 | **0.80** | **0.50** |

### Ablation Study (Iterative Data Pipeline)

| Stage | Training Set Size | Success Rate ↑ | PSNR ↑ |
|------|-----------|--------|-------|
| Base Model | 0 | 7.6% | 25.72 |
| Initialization (RORD+Mulan) | 61,565 | 38.6% | 28.41 |
| Human + Auto Round 1 | 86,381 | 55.6% | 28.60 |
| Human + Auto Round 2 | 144,488 | 67.2% | 28.75 |
| Human + Auto Round 3 | 199,934 | 75.4% | 28.78 |
| + High-Resolution Fine-tuning | 201,134 | **76.2%** | **31.10** |

### Key Findings
- The positive data loop is highly effective: each round of iteration yields a success rate improvement (38.6% $\rightarrow$ 55.6% $\rightarrow$ 67.2% $\rightarrow$ 75.4%), demonstrating the positive feedback of "stronger model $\rightarrow$ better data $\rightarrow$ even stronger model".
- Masking the source image ($\bar{\mathbf{x}}_s$ vs $\mathbf{x}_s$) is critical for robustness, as it prevents transparent object remnants.
- The distilled model RORem-4S achieves an 8x speedup with only a 1.4% drop in success rate, proving highly practical.
- The discriminator $D_\phi$ shows high consistency with human judgment (deviation < 3%), proving the reliability of automated annotation.

## Highlights & Insights
- **Data Flywheel Effect**: Starting with 60K data and a 38.6% success rate, the dataset grew to 200K and the success rate improved to 76.2% after 3 rounds of iteration. This bootstrap-style data construction paradigm can be generalized to other generative editing tasks that require high-quality paired data.
- **Discriminator Replacing Humans**: Training a discriminator with human feedback to automate subsequent data filtering is key to low-cost data scaling. Three rounds required only ~30K manual labels to acquire 200K data.
- **High Practicality**: RORem-4S completes high-quality removal in 0.5 seconds, outperforming all competitors.

## Limitations & Future Work
- The source data is primarily from OpenImages, which might bias the category distribution of the dataset.
- The 0.9 threshold of the discriminator is fixed, which could theoretically be dynamically adjusted based on the current model's capability.
- Currently, it only processes single-object removal; multi-object interactions and complex occlusion scenarios remain to be explored.
- There is still room for improvement in high-resolution (>1024) performance.

## Related Work & Insights
- **vs PowerPaint**: PowerPaint uses learnable tokens to control inpainting/removal modes, but is still based on the self-supervised paradigm. RORem fundamentally resolves ambiguity using high-quality paired data.
- **vs ObjectDrop**: ObjectDrop suggests capturing real paired data, but obtains only 2K pairs and is not publicly released. RORem's data construction framework can scale to arbitrary sizes at low cost.
- The paradigm of "data flywheel + automated discriminator" can be generalized to other visual editing tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The human-in-the-loop iterative data construction approach is practical and novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Detailed records of the iterative process, human evaluations, and distillation analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear framework diagram and comprehensive process descriptions.
- Value: ⭐⭐⭐⭐⭐ Constructed usable datasets and models, contributing directly to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions](intermimic_towards_universal_whole-body_control_for_physics-based_human-object_i.md)
- [\[CVPR 2025\] InterAct: Advancing Large-Scale Versatile 3D Human-Object Interaction Generation](interact_advancing_large-scale_versatile_3d_human-object_interaction_generation.md)
- [\[ICLR 2026\] ReFocusEraser: Refocusing for Small Object Removal with Robust Context-Shadow Repair](../../ICLR2026/image_generation/refocuseraser_refocusing_for_small_object_removal_with_robust_context-shadow_rep.md)
- [\[CVPR 2025\] HOI-IDiff: An Image-like Diffusion Method for Human-Object Interaction Detection](an_image-like_diffusion_method_for_human-object_interaction_detection.md)
- [\[ICML 2026\] AdaEraser: Training-Free Object Removal via Adaptive Attention Suppression](../../ICML2026/image_generation/adaeraser_training-free_object_removal_via_adaptive_attention_suppression.md)

</div>

<!-- RELATED:END -->
