---
title: >-
  [Paper Note] Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders
description: >-
  [ECCV 2024][Self-Supervised Learning][Masked Autoencoders] CropMAE is proposed: a siamese masked autoencoder trained by replacing video frame pairs with two randomly cropped views of the same image. With an extremely high masking ratio of 98.5%, it learns object boundary-aware representations using only 2 visible patches. This accelerates training by up to 23.8× compared to SiamMAE while achieving competitive performance on video propagation tasks.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Masked Autoencoders"
  - "Siamese Networks"
  - "Video Segmentation"
  - "Label Propagation"
date: 2026-05-08
content_hash: 6a949a0d102dbed5
---

# Efficient Image Pre-Training with Siamese Cropped Masked Autoencoders

**Conference**: ECCV 2024  
**arXiv**: [2403.17823](https://arxiv.org/abs/2403.17823)  
**Code**: [https://github.com/alexandre-eymael/CropMAE](https://github.com/alexandre-eymael/CropMAE)  
**Area**: Self-Supervised Learning  
**Keywords**: Self-Supervised Learning, Masked Autoencoders, Siamese Networks, Video Segmentation, Label Propagation

## TL;DR

CropMAE is proposed: a siamese masked autoencoder trained by replacing video frame pairs with two randomly cropped views of the same image. With an extremely high masking ratio of 98.5%, it learns object boundary-aware representations using only 2 visible patches. This accelerates training by up to 23.8× compared to SiamMAE while achieving competitive performance on video propagation tasks.

## Background & Motivation

Self-supervised pre-training is central to visual representation learning, where Masked Autoencoders (MAE) learn semantic features by reconstructing masked patches. SiamMAE further introduces a siamese architecture that establishes correspondence between two video frames, achieving SOTA on video propagation tasks (e.g., object segmentation, pose propagation).

**However, SiamMAE suffers from two key bottlenecks**:

**Dependency on video datasets**: Image datasets are typically much larger than video datasets and have lower decoding costs; video training is limited by data availability and computational overhead.

**Low training efficiency**: It requires 2000 epochs to converge because it must implicitly learn object conceptual understanding from video motion, and the pre-training task (reconstructing the entire frame from a few visible patches) demands deep semantic knowledge.

**Key Insight**: Does the object boundary understanding learned by SiamMAE truly originate from explicit motion in videos? If the **implicit image transformations** (cropping, flipping, etc.) between two views are the actual driving force, different crops of static images can completely replace video frame pairs.

**Core Idea**: Replace video frame pairs with random crops of the same image (Global-to-Local strategy) to construct an explicitly solvable pretext task (where the local view is always contained within the global view). Reconstructions can be completed without learning conceptual knowledge of the world, thereby supporting higher masking ratios (98.5%) and faster convergence.

## Method

### Overall Architecture

The training pipeline of CropMAE: Input an image $I$ $\rightarrow$ generate two cropped views $V_1$ (global, unmasked) and $V_2$ (local, extremely high masking ratio) $\rightarrow$ encode both views separately using siamese ViT encoders $\rightarrow$ reconstruct $V_2$ from $V_1$ using a Transformer decoder via cross-attention $\rightarrow$ minimize the L2 reconstruction loss. After pre-training, the decoder is discarded, and the encoder is used as a feature extractor for downstream tasks.

### Key Designs

1. **Cropping Strategy (Global-to-Local Views)**: Four cropping methods were explored: Same Views (same crop, worst performance $\mathcal{J\&F}_m=36.6$), Random Views (two independent random crops, 60.0), Local-to-Global ($V_1$ cropped from $V_2$, 55.9), and Global-to-Local ($V_2$ cropped from $V_1$, **60.4**, optimal). The key to the optimality of Global-to-Local is: the local view $V_2$ is always entirely contained within the global view $V_1$, making the reconstruction task **always solvable without prior conceptual knowledge**. The model only needs to (i) locate the position of the local view within the global view using a small number of visible patches, and (ii) determine the transformation required for reconstruction.

2. **Extremely High Masking Ratio (98.5%)**: Conventional MAE uses a 75% masking ratio, VideoMAE uses 90%, and SiamMAE uses 95% (9/196 visible patches). CropMAE pushes the masking ratio to 98.5%, keeping only **2 visible patches**. This stems from the fundamental difference in pretext tasks: other MAE variants need to learn conceptual understanding by "hallucinating" masked content, whereas CropMAE's task is directly solvable, necessitating a higher masking ratio to introduce challenge. Moving from 95% to 98.5% reduces the number of visible patches from 9 to 2 (a 4.5× reduction), significantly reducing the computational load of the attention layers.

3. **Decoder Architecture**: A 4-layer Transformer ($d_{model}=256$, $d_{ff}=2048$) that alternates between self-attention (among masked image tokens) and cross-attention (masked tokens attending to visible image tokens). The L2 loss is applied to normalized pixel values. The decoder is intentionally kept smaller than the encoder ($256$-d vs $384$-d) to prevent the decoder from being overly powerful, which would stop the encoder from learning good representations.

### Loss & Training

- **Reconstruction Loss**: $\mathcal{L} = \| V_2 - R \|_2^2$, where $R$ is the reconstruction output and $V_2$ is pixel-value normalized.
- Optimizer: AdamW, base learning rate $1.5 \times 10^{-4}$
- Encoder: ViT-S/16 (main experiments) or ViT-B/16
- No Color Jitter or Gaussian Blur (proven harmful in experiments), with only optional horizontal flipping.
- Only requires training for 400 epochs (SiamMAE requires 2000 epochs).

## Key Experimental Results

### Main Results

Comparison on three video propagation downstream tasks (label propagation evaluation, without fine-tuning):

| Method | Backbone | Dataset | Epochs | DAVIS $\mathcal{J\&F}_m$ | VIP mIoU | JHMDB PCK@0.1 |
|------|------|--------|--------|-------------------------|----------|---------------|
| SiamMAE (paper) | ViT-S/16 | K400 | 2000 | 62.0 | 37.3 | 47.0 |
| SiamMAE (reproduction) | ViT-S/16 | K400 | 400 | 57.9 | 33.2 | **46.1** |
| **CropMAE** | ViT-S/16 | K400 | 400 | 58.6 (+0.7) | 33.7 (+0.5) | 42.9 |
| **CropMAE** | ViT-S/16 | IN Sub | 400 | **60.4 (+2.5)** | 33.3 | 43.6 |
| **CropMAE** | ViT-B/16 | IN Sub | 400 | **60.9** | 32.8 | 44.3 |
| MAE | ViT-B/16 | IN | 1600 | 53.5 | 28.1 | 44.6 |
| VideoMAE | ViT-S/16 | K400 | 800 | 39.3 | 23.3 | 41.0 |

**Key Findings**: Given the same budget of 400 epochs, CropMAE outperforms SiamMAE by 2.5% on DAVIS (using ImageNet) and converges faster (reaching 58.0 at 150 epochs). Performance on JHMDB is slightly inferior to SiamMAE, as this task involves human pose deformation, where real video motion is more helpful.

### Ablation Study

| Configuration | DAVIS $\mathcal{J\&F}_m$ | Description |
|------|--------------------------|------|
| **Cropping Strategy** | | |
| Same Views | 36.6 | Unable to learn propagation capability |
| Random Views | 60.0 | Sometimes solvable |
| Local-to-Global | 55.9 | Global reconstruction requires conceptual knowledge, which is difficult |
| **Global-to-Local** | **60.4** | Always solvable, optimal |
| **Masking Ratio** | | |
| 75% (49 patches) | 45.3 | Task is too simple; encoder fails to learn useful features |
| 90% (19 patches) | 47.1 | Still too simple |
| 95% (9 patches) | 51.2 | Chosen by SiamMAE |
| **98.5% (2 patches)** | **60.4** | Optimal, extreme but effective |
| 99% (1 patch) | 58.6 | Slightly too extreme |
| **Decoder Depth** | | |
| 2 layers | 59.1 | Slightly shallow |
| **4 layers** | **60.4** | Optimal |
| 8 layers | 57.0 | Overly large decoder is harmful |
| **Data Augmentation** | | |
| + Color Jitter | 56.2 | Significantly harmful |
| + Gaussian Blur | 59.6 | Slightly harmful |
| No horizontal flip | 60.3 | Almost no impact |

### Training Speed

| Method | Dataset | Frame Count | Masking Ratio | GFLOPs | Speedup |
|------|--------|------|--------|--------|---------|
| SiamMAE | K400 | 2 | 95% | 5.8 | ×1.0 |
| CropMAE | K400 | 1 | 98.5% | 5.6 | **×1.29** |
| CropMAE | IN Subset | 1 | 98.5% | 5.6 | **×23.8** |

Reasons for the 23.8× speedup when training on ImageNet: (1) No video decoding is required, making image loading much faster; (2) The higher masking ratio reduces the token count, significantly decreasing the quadratic complexity of attention computations.

### Key Findings

- Object boundary understanding **does not require explicit motion**: The attention maps of CropMAE trained on ImageNet clearly capture object boundaries, consistent with what SiamMAE learns from videos.
- ImageNet outperforms K400: This is attributed to ImageNet images being more diverse and centered on objects, leading to higher-quality pretext tasks generated by cropping.
- CropMAE surpasses the performance of SiamMAE at 350 epochs within only 150 epochs, validating the fast convergence advantage brought by the solvable pretext task.

## Highlights & Insights

- **Core Counter-Intuitive Finding**: Learning object boundaries and propagation capabilities does not require motion information from video; random crops of static images are sufficient.
- **The Victory of Extremism**: A masking ratio of 98.5% (only 2 visible patches) seems wild but is indeed optimal, because the pretext task itself is simple enough.
- Exceptional simplicity of the method: It requires no negative sample construction as in contrastive learning, no momentum encoder, and no meticulously designed data augmentations.
- Reformulates the video pre-training problem into an image pre-training problem, significantly lowering the data and computational barriers.

## Limitations & Future Work

- Inferior to SiamMAE on pose propagation (JHMDB) because random crops cannot simulate the complex deformations of human body motion.
- The scalability of the model and data (larger ViTs, more data) has not yet been fully explored.
- Not evaluated on mainstream benchmarks such as image classification (ImageNet linear probing); the performance on non-propagation tasks remains unclear.
- The unique contribution of video frames compared to static images still requires in-depth investigation.

## Related Work & Insights

- Relationship with SiamMAE: CropMAE is a "simplified version" of SiamMAE, whose core contribution lies in demonstrating that video frame pairs are not necessary.
- Relationship with MAE/VideoMAE: By employing a higher masking ratio and dual-view cropping, it significantly outperforms standard MAE on propagation tasks.
- Connection to contrastive learning (SimCLR/BYOL): Random cropping is also a core augmentation in contrastive learning, but CropMAE does not rely on a careful selection of data augmentations and is free from the representation collapse issue.
- Insight: For specific downstream tasks, the "solvability" of a pretext task might be more important than its difficulty.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of replacing video frames with image crops is simple yet counter-intuitive, and the discovery of the 98.5% masking ratio is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three downstream tasks + exhaustive ablations + training speed + attention visualization, though lacking evaluation on classification.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentation logic and deep analysis of "task solvability".
- Value: ⭐⭐⭐⭐ Dramatically lowers the data and computational barriers for self-supervised pre-training, offering practical significance for resource-constrained scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[CVPR 2025\] From Prototypes to General Distributions: An Efficient Curriculum for Masked Image Modeling](../../CVPR2025/self_supervised/from_prototypes_to_general_distributions_an_efficient_curriculum_for_masked_imag.md)
- [\[CVPR 2026\] In Pursuit of Pixel Supervision for Visual Pre-training](../../CVPR2026/self_supervised/in_pursuit_of_pixel_supervision_for_visual_pre-training.md)
- [\[CVPR 2026\] Recurrent Video Masked Autoencoders](../../CVPR2026/self_supervised/recurrent_video_masked_autoencoders.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](../../CVPR2026/self_supervised/com_pt_chain_of_models_pretraining.md)

</div>

<!-- RELATED:END -->
