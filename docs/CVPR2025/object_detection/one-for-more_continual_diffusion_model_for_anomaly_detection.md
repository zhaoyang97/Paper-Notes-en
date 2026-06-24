---
title: >-
  [Paper Note] One-for-More: Continual Diffusion Model for Anomaly Detection
description: >-
  [CVPR 2025][Object Detection][Continual Learning] The CDAD framework is proposed to achieve stable continual learning for diffusion models via gradient projection. Supported by iterative SVD (iSVD), the memory consumption is reduced from 157GB to 17GB. Additionally, an anomaly-masked network is designed to enhance the conditioning mechanism, achieving first place in 17 out of 18 settings across MVTec and VisA.
tags:
  - "CVPR 2025"
  - "Object Detection"
  - "Continual Learning"
  - "Diffusion Model"
  - "Anomaly Detection"
  - "Gradient Projection"
  - "Singular Value Decomposition"
date: 2026-05-08
content_hash: 92636fe290092751
---

# One-for-More: Continual Diffusion Model for Anomaly Detection

**Conference**: CVPR 2025  
**arXiv**: [2502.19848](https://arxiv.org/abs/2502.19848)  
**Code**: [https://github.com/FuNz-0/One-for-More](https://github.com/FuNz-0/One-for-More)  
**Area**: Image Generation/Anomaly Detection  
**Keywords**: Continual Learning, Diffusion Model, Anomaly Detection, Gradient Projection, Singular Value Decomposition

## TL;DR

The CDAD framework is proposed to achieve stable continual learning for diffusion models via gradient projection. Supported by iterative SVD (iSVD), the memory consumption is reduced from 157GB to 17GB. Additionally, an anomaly-masked network is designed to enhance the conditioning mechanism, achieving first place in 17 out of 18 settings across MVTec and VisA.

## Background & Motivation

Anomaly detection has evolved from "one-for-one" (training a dedicated model for each class) to "one-for-many" (detecting all classes with a single model). However, in realistic scenarios, class increments are unpredictable, requiring models to possess continual learning capabilities (the "one-for-more" paradigm).

Diffusion-model-based anomaly detection methods generate normal images via image-to-image translation and utilize the difference between input and output as the anomaly score. However, they face two major challenges in continual learning:

1. **Catastrophic Forgetting**: After training on new classes, the model's ability to reconstruct old classes degrades severely. For example, ControlNet performs excellently on 10 base classes, but its performance drops sharply as new tasks are added.
2. **Fidelity Hallucination**: Diffusion models tend to "overfit" to generating normal images rather than reconstructing anomalous regions, leading to output inconsistency with original samples of old classes after continual learning.

Meanwhile, existing continual anomaly detection methods (e.g., DNE, UCAD) rely on an expanding memory buffer to store old knowledge, which is not scalable. The semantic compression loss in IUF still suffers from significant forgetting during large-step class increments.

## Method

### Overall Architecture

CDAD constructs a continual diffusion model based on a pre-trained VAE and U-Net. During testing, the input image is encoded into the latent space and undergoes diffusion. Then, the anomaly-masked network (AMN) guides the denoising reconstruction as a condition. The feature distance between the reconstruction result and the original image is used as the anomaly score.

### Key Designs

**Design 1: Continual Diffusion Model (CDM) based on Gradient Projection**

- **Function**: Protects the feature space of old tasks from being corrupted while learning new tasks.
- **Mechanism**: For each layer in the U-Net, after calculating the gradient of the new task, it is projected onto the orthogonal complement of the input feature space of old tasks. Let $\hat{U}_i$ be the k-rank column basis of the old task, the projected gradient is: $\nabla_{W_i}^{orth} = \nabla_{W_i} - \hat{U}_i \hat{U}_i^T \nabla_{W_i}$
- **Design Motivation**: Since $X_{pre} \cdot \nabla_W^{orth} \approx 0$, the updated model's output for old tasks remains approximately unchanged, fundamentally eliminating forgetting. It does not require storing old data and introduces no extra inference overhead.

$$\hat{O}_{pre} = X_{pre} W_i - \eta X_{pre} \nabla_W^{orth} \approx O_{pre}$$

**Design 2: Iterative Singular Value Decomposition (iSVD)**

- **Function**: Reduces the memory consumption of computing the salience representation from ~157GB to ~17GB.
- **Mechanism**: Utilizing the transitivity of linear representation (Lemma 1), a large matrix $M$ is partitioned into $\{M_1, M_2, ..., M_n\}$. First, a k-rank approximation $\hat{U}_1$ is obtained by performing SVD on $M_1$. Then, $\hat{U}_i$ and $M_{i+1}$ are iteratively concatenated and subjected to SVD, ultimately yielding the global salience representation $\hat{U}_n$.
- **Design Motivation**: The Markovian denoising process of diffusion models generates a massive amount of intermediate features. Traditional SVD requires loading all features into memory at once. Based on the theory of linear transitivity, iSVD proves that iterative computation is equivalent to global computation, making memory overhead comparable to a single batch.

**Design 3: Anomaly-Masked Network (AMN)**

- **Function**: Enhances the conditioning mechanism of the diffusion model to focus reconstruction on anomalous regions.
- **Mechanism**: CNNs are used to encode local features, while Transformers encode global features. Neighbor-masked self-attention (from UniAD) and anomaly-masked loss are introduced to mask features of anomalous regions, providing only normal-region features as conditional inputs to the U-Net.
- **Design Motivation**: Traditional image-to-image diffusion models tend to replicate the entire input rather than reconstruct the anomalous regions (referring to the "overfitting" issue). By using a masking mechanism, the model is forced to focus on normalizing reconstruction in anomalous regions.

### Loss & Training

The primary loss is the standard latent-space diffusion loss:

$$\mathcal{L}_{CDM} = \mathbb{E}_{\mathcal{E}(x), \tilde{x}, \epsilon \sim \mathcal{N}(0,I), t} \|\epsilon - \epsilon_\theta(z_t, t, \tau_\theta(\tilde{x}))\|_2^2$$

Where $\tilde{x}$ is the original image with random patch perturbations used as the conditional input. The anomaly-masked loss additionally supervises the AMN to learn to distinguish between normal and anomalous features.

## Key Experimental Results

### MVTec Continual Anomaly Detection (10+1+1+1+1 Setting)

| Method | Image AUROC | Pixel AUROC |
|------|------------|------------|
| IUF | 83.2 | 90.1 |
| UniAD+EWC | 80.5 | 88.7 |
| DNE | 85.1 | 91.3 |
| **CDAD (Ours)** | **91.4** | **95.2** |

### iSVD Memory Consumption Comparison

| Method | 10 Images | 100 Images | 1000 Images |
|------|---------|----------|-----------|
| Traditional SVD | 157 GB | 1570 GB | OOM |
| **iSVD (Ours)** | **17 GB** | **17 GB** | **17 GB** |

### Ablation Study (MVTec, 5+10 Setting)

| Component | Image AUROC | Pixel AUROC |
|------|------------|------------|
| Baseline (ControlNet) | 75.3 | 85.6 |
| + Gradient Projection | 87.2 | 92.1 |
| + iSVD (replacing traditional SVD) | 87.0 | 91.9 |
| + AMN | **91.4** | **95.2** |

### Key Findings

1. CDAD ranks first in 17 out of 18 continual learning settings across MVTec and VisA.
2. iSVD reduces memory consumption by about 9 times with only a ~0.2 AUROC performance loss, verifying the effectiveness of the linear transitivity theory.
3. Gradient projection alone brings a +11.9 Image AUROC improvement, pointing to it as the core mechanism for resolving the forgetting issue.
4. AMN contributes an additional +4.2 Image AUROC, demonstrating the importance of enhancing the conditioning mechanism for anomaly detection.

## Highlights & Insights

1. **Introduction of the One-for-more Paradigm**: Clearly distinguishes continual anomaly detection from "one-for-one" and "one-for-many", focusing on more realistic incremental settings.
2. **Theoretical Contribution of iSVD**: The proposed iterative SVD based on the linear transitivity theorem not only addresses the memory bottleneck of diffusion models but also inspires other scenarios requiring large-scale SVD.
3. **Precise Problem Diagnosis**: The identification and analysis of "fidelity hallucination" are highly accurate, and the design of AMN directly addresses this issue.

## Limitations & Future Work

1. Each task still requires training for multiple epochs; continual learning efficiency can be further improved.
2. Gradient projection may lead to an overly restricted gradient space when the number of tasks is very large, limiting the capability to learn new tasks.
3. Only validated on industrial defect detection scenarios, without extension to other anomaly detection fields like medical imaging.
4. AMN relies on random patch perturbations to simulate anomalies, whereas real anomaly patterns are much more diverse.

## Related Work & Insights

- **UniAD**: A pioneer in multi-class anomaly detection, whose neighbor-masked self-attention is borrowed by AMN.
- **DiAD**: The first to introduce latent diffusion models into multi-class anomaly detection, but does not support continual learning.
- **GPM**: A gradient projection method for continual learning in classification tasks, which is extended to generative models in this paper.
- Insight: The iSVD method can be extended to other scenarios in large models requiring continual learning, such as continual diffusion generation and continual text-to-image.

## Rating

⭐⭐⭐⭐ — The problem definition is clear (one-for-more), and the method design is systematic (the trinity of CDM, iSVD, and AMN), combining both theory and experiments. The theoretical contribution of iSVD is particularly prominent. The limitation lies in the narrow application domain and the still-limited number of tasks in continual learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Odd-One-Out: Anomaly Detection by Comparing with Neighbors](odd-one-out_anomaly_detection_by_comparing_with_neighbors.md)
- [\[CVPR 2025\] Distribution Prototype Diffusion Learning for Open-set Supervised Anomaly Detection](distribution_prototype_diffusion_learning_for_open-set_supervised_anomaly_detect.md)
- [\[CVPR 2025\] Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)
- [\[NeurIPS 2025\] Scalable, Explainable and Provably Robust Anomaly Detection with One-Step Flow Matching](../../NeurIPS2025/object_detection/scalable_explainable_and_provably_robust_anomaly_detection_with_one-step_flow_ma.md)
- [\[CVPR 2025\] UniVAD: A Training-free Unified Model for Few-shot Visual Anomaly Detection](univad_a_training-free_unified_model_for_few-shot_visual_anomaly_detection.md)

</div>

<!-- RELATED:END -->
