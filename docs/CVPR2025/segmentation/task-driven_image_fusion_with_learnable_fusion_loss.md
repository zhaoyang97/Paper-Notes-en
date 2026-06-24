---
title: >-
  [Paper Note] Task-driven Image Fusion with Learnable Fusion Loss
description: >-
  [CVPR 2025][Segmentation][Multimodal Image Fusion] This paper proposes TDFusion, which trains a loss generation module via meta-learning to adaptively adjust the fusion loss function based on downstream tasks (semantic segmentation or object detection), thereby achieving optimal performance for infrared-visible fusion images on downstream tasks.
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Multimodal Image Fusion"
  - "Meta-learning"
  - "Learnable Loss Function"
  - "Semantic Segmentation"
  - "Object Detection"
date: 2026-05-08
content_hash: f43564209cbc7e17
---

# Task-driven Image Fusion with Learnable Fusion Loss

**Conference**: CVPR 2025  
**arXiv**: [2412.03240](https://arxiv.org/abs/2412.03240)  
**Code**: [https://github.com/HaowenBai/TDFusion](https://github.com/HaowenBai/TDFusion)  
**Area**: Segmentation/Image Fusion  
**Keywords**: Multimodal Image Fusion, Meta-learning, Learnable Loss Function, Semantic Segmentation, Object Detection

## TL;DR

This paper proposes TDFusion, which trains a loss generation module via meta-learning to adaptively adjust the fusion loss function based on downstream tasks (semantic segmentation or object detection), thereby achieving optimal performance for infrared-visible fusion images on downstream tasks.

## Background & Motivation

**Background**: Multimodal image fusion aggregates information from infrared and visible images into a single fused image, which is widely applied in downstream tasks such as semantic segmentation and object detection. Existing methods typically use predefined fusion losses (e.g., intensity loss, gradient loss) or introduce task losses by cascading downstream task networks to constrain the fusion.

**Limitations of Prior Work**: Even with the introduction of downstream tasks, existing frameworks still rely on fixed fusion loss terms and lack dynamic adaptability. Manually defined loss functions impose preset prior constraints, failing to flexibly adjust fusion preferences according to specific image pairs and task requirements. This limits the adaptability of fusion results to specific tasks.

**Key Challenge**: There is a gap between the design objective of fusion loss (visual quality) and the requirements of downstream tasks (semantic features, detection accuracy). Predefined losses cannot dynamically capture different preferences for source image information across different tasks—segmentation tasks favor boundaries and texture, while detection tasks favor the contrast of target regions.

**Goal**: To design a framework that makes the fusion loss itself learnable, allowing the downstream task loss to directly drive the optimization direction of the fusion process.

**Key Insight**: The authors observe that the essence of fusion loss is to control "how much intensity information from each source image is retained," parameterizing this as pixel-wise weights $w_a, w_b$, and predicting these weights using a neural network (loss generation module).

**Core Idea**: Use meta-learning (MAML-style) to train the loss generation module—the inner-loop update uses the fusion loss to update a replica of the fusion network, while the outer-loop update uses downstream task loss to update the loss generation module, ensuring that the generation of the fusion loss consistently evolves toward minimizing the task loss.

## Method

### Overall Architecture

TDFusion consists of three modules: a fusion network $\mathcal{F}$ (responsible for merging infrared and visible images into a single image), a downstream task network $\mathcal{T}$ (such as SegFormer or YOLOv8), and a loss generation module $\mathcal{G}$ (which outputs parameters of the learnable fusion loss). During training, the fusion network and the loss generation module are updated alternately: the loss generation module is optimized first through a meta-learning pipeline (inner loop + outer loop), and then the fusion network is trained using the optimized fusion loss.

### Key Designs

1. **Learnable Fusion Loss**:

    - Function: Adaptively generates pixel-wise fusion weights according to downstream task requirements.
    - Mechanism: The fusion loss consists of an intensity term and a gradient term $\mathcal{L}_f = \mathcal{L}_f^{int} + \alpha \mathcal{L}_f^{grad}$. In the intensity term, the loss generation module predicts pixel-wise weights $\{w_a, w_b\} = \mathcal{G}(I_a, I_b)$ for the input infrared and visible images, with a Softmax constraint ensuring $w_a^{ij} + w_b^{ij} = 1$ to control which source image the fused image should resemble more closely. The gradient term extracts gradients using a Sobel operator, requiring the fused image to preserve larger gradient values from the source images.
    - Design Motivation: Traditional fusion loss weights are fixed (e.g., equally split at $1/2$), which fails to differentiate information preferences across different tasks and regions. Learnable weights enable the model to determine whether to prefer infrared or visible information for each individual pixel.

2. **Meta-learning Training**:

    - Function: Allows downstream task loss to drive the optimization of the fusion loss.
    - Mechanism: Uses a MAML-style two-stage update. **Inner update**: Clones the fusion network $\mathcal{F}'$ and the task network $\mathcal{T}'$, and performs a single update step on the meta-training set using the current fusion loss, obtaining intermediate parameters $\theta_{\mathcal{F}'}$ and $\theta_{\mathcal{T}'}$. **Outer update**: Generates fused images using $\mathcal{F}'$ on the meta-test set, computes the downstream task loss $\mathcal{L}_t$, and backpropagates to update the loss generation module $\theta_{\mathcal{G}}$. The key is that the inner update retains the computation graph of $\theta_{\mathcal{F}'}$ with respect to $\theta_{\mathcal{G}}$, allowing second-order gradients to propagate back to the loss generation module.
    - Design Motivation: Directly training the fusion network with task loss degenerates current fusion into task-specific feature extraction, sacrificing the visual quality of the fused image. The meta-learning strategy allows the task loss to guide fusion indirectly by optimizing the fusion loss, preserving the generality of the fusion loss framework.

3. **Data Splitting and Alternating Training**:

    - Function: Avoids overfitting and ensures the generalization capability of the loss generation module.
    - Mechanism: In each epoch, non-overlapping meta-training and meta-test sets (each with $M$ image pairs) are randomly sampled from the fusion training set. After the loss generation module completes $M$ meta-learning steps on these subsets, the fusion network is trained for $N$ steps on the full training set using the updated fusion loss. This alternating process runs for $L$ epochs.
    - Design Motivation: The separation of meta-train and meta-test ensures that the loss generation module is learning general task preference patterns rather than memorizing specific image pairs. Alternating updates ensure the fusion loss remains optimal throughout different training stages of the fusion network.

### Loss & Training

The fusion loss $\mathcal{L}_f$ consists of a learnable intensity loss and a fixed gradient loss, with weight $\alpha=1$. The downstream task loss $\mathcal{L}_t$ depends on the specific task: cross-entropy loss for semantic segmentation and YOLO loss for object detection. The Adam optimizer is used with a learning rate of $1 \times 10^{-4}$ and a batch size of 2. The fusion network is constructed based on Restormer Blocks, and the loss generation module likewise uses Restormer Blocks with a Softmax output layer.

## Key Experimental Results

### Main Results

Fusion and downstream tasks are evaluated on four datasets (MSRS, FMB, M3FD, and LLVIP):

| Method | MSRS mIoU↑ | FMB mIoU↑ | M3FD mAP50↑ | LLVIP AP50↑ |
|---|---|---|---|---|
| TarDAL | 71.35 | 55.33 | 83.16 | 93.79 |
| SegMIF | 74.25 | 58.41 | 83.61 | 93.95 |
| EMMA | 74.48 | 56.28 | 83.71 | 94.00 |
| MRFS | 74.50 | 55.71 | 83.28 | 93.03 |
| TIMFusion | 73.58 | 57.24 | 83.22 | 93.76 |
| **TDFusion** | **75.09** | **60.50** | **86.27** | **95.00** |

### Ablation Study

| Configuration | EN↑ | SF↑ | SCD↑ | VIF↑ | $Q^{AB/F}$↑ | SSIM↑ |
|---|---|---|---|---|---|---|
| Fixed $w_a=w_b=0.5$ | 6.60 | 13.73 | 1.58 | 0.39 | 0.60 | 0.72 |
| w/o gradient loss | 6.77 | 11.65 | 1.63 | 0.37 | 0.64 | 0.73 |
| $\theta_\mathcal{F}$ directly affected by task loss | 6.80 | 13.85 | 1.70 | 0.41 | 0.66 | 0.73 |
| w/o independent fusion learning | 6.82 | 14.07 | 1.72 | 0.41 | 0.67 | 0.72 |
| Replacing fusion network with weighted average | 6.75 | 11.49 | 1.65 | 0.38 | 0.62 | 0.73 |
| **Full TDFusion** | **6.86** | **14.16** | **1.76** | **0.43** | **0.68** | **0.75** |

### Key Findings

- Compared to learnable weights, the fixed weights $w_a=w_b=0.5$ cause the SCD to drop from 1.76 to 1.58, which demonstrates that adaptive weights are the key contribution.
- Removing the gradient loss causes SF (spatial frequency) to drop sharply from 14.16 to 11.65, showing that the gradient term is crucial for preserving texture details.
- Directly training $\theta_\mathcal{F}$ with task loss (Exp III) performs worse than indirect optimization via meta-learning, validating the necessity of the meta-learning strategy.
- Visualizations show that semantic segmentation favors visible texture and low-light infrared information, while detection favors highlighted target regions in infrared data, with a clear discrepancy in fusion weight distributions between the two tasks.

## Highlights & Insights

1. **"Learning to Learn Loss Functions" Paradigm**: Instead of manually designing better loss functions, this paper lets meta-learning automatically generate the most suitable loss function for downstream tasks. This concept possesses high generality and can be transferred to other ground-truth-free tasks.
2. **Interpretability of Fusion Weights**: Pixel-wise visualizations of $w_a$ and $w_b$ clearly illustrate the differences in information preference across different tasks, providing an intuitive tool for understanding multimodal fusion.
3. **Architecture Agnosticism**: The framework imposes no requirements on the architectures of the fusion network and task network, permitting plug-and-play replacement with any arbitrary network.

## Limitations & Future Work

- Second-order gradient calculations in meta-learning increase training costs, and single-GPU (RTX 3090) training relies on small batch sizes.
- The method has only been validated in infrared-visible fusion scenarios and has not yet been extended to medical image fusion such as CT-MRI.
- The Softmax constraint in the loss generation module limits $w_a + w_b = 1$, which cannot represent scenarios where "both source images are emphasized simultaneously."
- Future work can explore extending the learnable loss concept to more unsupervised tasks (e.g., super-resolution, dehazing, etc.).

## Related Work & Insights

- **vs TIMFusion**: TIMFusion searches for an optimal fusion architecture initialization via NAS but still uses a fixed fusion loss, whereas TDFusion directly learns the loss function, offering greater flexibility.
- **vs SegMIF**: SegMIF embeds high-level vision task features into the fusion process, which belongs to feature-level guidance; TDFusion guides at the loss level, making the two approaches complementary.
- **vs ReFusion**: Previous work by the same authors, which learns the fusion loss by reconstructing source images; TDFusion is instead directly driven by the task loss, resulting in a more end-to-end paradigm.
- The idea of meta-learning-driven loss design can be transferred to other multimodal learning scenarios, such as the alignment loss design in vision-language models.

## Rating

- Novelty: 8/10 — The idea of using meta-learning to learn fusion loss functions is novel and elegant, though the MAML framework itself has mature applications.
- Experimental Thoroughness: 8/10 — Four datasets + two downstream tasks + comprehensive ablations, presenting a thorough experimental design.
- Writing Quality: 7/10 — The formula derivations are detailed but the paper is long; the core ideas could be stated more concisely.
- Value: 7/10 — Valuable to the multimodal fusion field, though limited in application scope to infrared-visible fusion.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Semantic Library Adaptation: LoRA Retrieval and Fusion for Open-Vocabulary Semantic Segmentation](semantic_library_adaptation_lora_retrieval_and_fusion_for_open-vocabulary_semant.md)
- [\[AAAI 2026\] CtrlFuse: Mask-Prompt Guided Controllable Infrared and Visible Image Fusion](../../AAAI2026/segmentation/ctrlfuse_mask-prompt_guided_controllable_infrared_and_visible_image_fusion.md)
- [\[NeurIPS 2025\] HCLFuse: Revisiting Generative Infrared and Visible Image Fusion Based on Human Cognitive Laws](../../NeurIPS2025/segmentation/revisiting_generative_infrared_and_visible_image_fusion_based_on_human_cognitive.md)
- [\[CVPR 2025\] G2HFNet: GeoGran-Aware Hierarchical Feature Fusion Network for Salient Object Detection in Optical Remote Sensing Images](binwang2hfnet_geogran-aware_hierarchical_feature_fusion_network_for_salient_obje.md)
- [\[CVPR 2025\] Frequency Dynamic Convolution for Dense Image Prediction](frequency_dynamic_convolution_for_dense_image_prediction.md)

</div>

<!-- RELATED:END -->
