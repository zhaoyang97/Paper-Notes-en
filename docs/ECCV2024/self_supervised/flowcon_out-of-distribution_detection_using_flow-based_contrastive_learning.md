---
title: >-
  [Paper Note] FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning
description: >-
  [ECCV 2024][Self-Supervised Learning][OOD detection] Proposes FlowCon, a density estimation-based OOD detection method that innovatively combines normalizing flows with supervised contrastive learning. By using a contrastive loss based on the Bhattacharyya coefficient in the latent space of the flow model to learn class-conditional Gaussian distributions, it achieves efficient OOD detection without requiring external OOD data or retraining the classifier.
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "OOD detection"
  - "Normalizing Flows"
  - "Contrastive Learning"
  - "Density Estimation"
  - "Bhattacharyya Coefficient"
date: 2026-05-08
content_hash: b3e705e7e61b9b78
---

# FlowCon: Out-of-Distribution Detection using Flow-Based Contrastive Learning

**Conference**: ECCV 2024  
**arXiv**: [2407.03489](https://arxiv.org/abs/2407.03489)  
**Code**: [https://github.com/saandeepa93/FlowCon_OOD](https://github.com/saandeepa93/FlowCon_OOD)  
**Area**: Self-Supervised Learning  
**Keywords**: OOD detection, Normalizing Flows, Contrastive Learning, Density Estimation, Bhattacharyya Coefficient

## TL;DR

Proposes FlowCon, a density estimation-based OOD detection method that innovatively combines normalizing flows with supervised contrastive learning. By using a contrastive loss based on the Bhattacharyya coefficient in the latent space of the flow model to learn class-conditional Gaussian distributions, it achieves efficient OOD detection without requiring external OOD data or retraining the classifier.

## Background & Motivation

Deep learning models are trained under the closed-world assumption, which assumes that the input distribution at test time aligns with the training distribution. However, during real-world deployment, models inevitably encounter **out-of-distribution (OOD) samples**, including semantic shifts (far-OOD, featuring new classes) and covariate shifts (near-OOD, where the input space changes but the label space remains unchanged). Models can generate arbitrarily high confidence predictions for OOD samples, leading to severe consequences in safety-critical domains such as medical diagnosis and autonomous driving.

**Limitations of Prior Work**:

**Post-hoc methods** (MSP, ODIN, Energy, ReAct, etc.) directly manipulate the softmax scores of pre-trained classifiers, which are simple and effective but suffer from **significant performance degradation in near-OOD scenarios**.

**Outlier-based methods** (Heatmap, etc.) require external OOD datasets for training, but the space of OOD data is extremely vast, and assuming a specific OOD distribution can introduce bias.

**Density-based methods** (Mahalanobis, ResFlow, etc.) are theoretically more robust as they explicitly model the ID data distribution, but they suffer from severe practicality issues:
   - **ResFlow** requires training an independent flow model for **each class and each network layer** (e.g., CIFAR-100 + ResNet18 = 400 flow models), where training costs explode with dataset and model complexity.
   - **Zhang et al.** proposed jointly training the classifier and the flow model, but this requires **retraining the original classifier**, which is unsuitable for practical deployment.
   - Traditional flow models learn a single Gaussian distribution and **ignore class information**, which can assign high likelihoods to OOD samples.

**Key Challenge**: How to efficiently learn density estimation with class information using a **single model** without **using external OOD data or retraining the classifier**, thereby achieving robust performance across various OOD scenarios (far/near/mixed)?

**Core Idea**: Train a flow model on the penultimate layer features of a pre-trained classifier, while optimizing two loss functions simultaneously—the flow loss $\mathcal{L}_{flow}$ (maximizing log-likelihood) and a newly proposed contrastive loss $\mathcal{L}_{con}$ (performing supervised contrastive learning using the Bhattacharyya coefficient as the similarity function). This forces the flow model to learn a **class-conditional joint multimodal Gaussian distribution** instead of a single Gaussian.

## Method

### Overall Architecture

Training pipeline of FlowCon: Given an input image $x$, a pre-trained (frozen) classifier extracts the penultimate deep feature $z_{emb}$. The flow model maps $z_{emb}$ to a latent embedding $z_{flow}$ with its corresponding distribution parameters $\mathcal{N}(\mu, \sigma)$. During training, both the flow loss and the contrastive loss are optimized. During inference, the likelihood of a test sample is calculated across all class distributions, and the maximum value is used as the OOD score.

### Key Designs

1. **Flow-Based Contrastive Similarity**: Traditional contrastive learning utilizes the dot product or cosine similarity of feature vectors. FlowCon innovatively defines a new similarity function leveraging the likelihood values of the flow model:

$$S_{flow}(z_i, z_j, \mathcal{N}_i) = \exp\left(\left(p_Z(z_i|\mathcal{N}_i) \cdot p_Z(z_j|\mathcal{N}_i)\right)^{\tau_1}\right)$$

where $p_Z(z_i|\mathcal{N}_i) = \frac{1}{\sigma\sqrt{2\pi}} \exp\left[-\frac{1}{2}\left(\frac{z_i - \mu_i}{\sigma_i}\right)^2\right]$.

When $\tau_1 = 0.5$, the product inside the exponential is a generalized form of the **Bhattacharyya coefficient**, which is a classic statistic specifically designed to measure similarity between two probability distributions.

**Design Motivation**: Reducing the high-dimensional vector dot product to a product of scalar likelihood values simplifies the computation and enables contrast at the probability distribution level rather than the feature space, aligning the learning objectives more consistently with the density estimation nature of OOD detection.

2. **FlowCon Loss**: Integrating the new similarity function into the supervised contrastive loss framework yields:

$$\mathcal{L}_{con} = \sum_{i \in I} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{S_{flow}(z_i, z_p, \mathcal{N}_i) / \tau_2}{\sum_{a \in A(i)} S_{flow}(z_i, z_a, \mathcal{N}_i) / \tau_2}$$

Unlike traditional supervised contrastive loss (SCL), the anchor in FlowCon includes not only the latent vector $z_i$ but also the **distribution $\mathcal{N}_i$**. The total loss is defined as: $\mathcal{L} = \mathcal{L}_{con} + \lambda \mathcal{L}_{flow}$, where $\lambda = 0.07$.

**Design Motivation**: $\mathcal{L}_{con}$ operates in the distribution space, pulling similar class distributions closer while pushing different ones apart. Meanwhile, $\mathcal{L}_{flow}$ ensures that the latent embedding of each data point belongs to its corresponding class distribution. The two losses work in synergy, prompting the latent space to form **distinct class-conditional multimodal Gaussian distributions** (as demonstrated intuitively by the toy experiment in Fig. 1).

3. **OOD Inference**: After training, empirical distribution parameters are computed for each class $c$:

$$\mu_c = \frac{1}{|\mathcal{X}_c|}\sum_{i \in \mathcal{X}_c} \mu_i, \quad \sigma_c = \frac{1}{|\mathcal{X}_c|}\sum_{i \in \mathcal{X}_c} \sigma_i$$

The OOD score for a test sample is calculated as $S(x_{test}) = \max_{i \in \{1,...,k\}} p_Z(z_{test}|\mathcal{N}_{y=i}$)—representing the maximum likelihood across all class distributions. ID samples should yield high likelihood, whereas OOD samples should yield low likelihood.

**Design Motivation**: Simplifying the distributions of $n$ training samples to $k$ class distributions dramatically reduces computational overhead during inference while maintaining class discriminative capacity.

### Loss & Training

- The flow model adopts the **RealNVP** architecture with 8 coupling blocks, consisting of a single flow layer.
- Trained on the 512-dimensional penultimate features of ResNet18 and the 128-dimensional penultimate features of WideResNet.
- Adam optimizer with a learning rate of $1\times10^{-5}$ and a weight decay of $1\times10^{-5}$.
- Trained for 700 epochs with a batch size of 64 and image size of $32 \times 32$.
- Hyperparameters: $\lambda = 0.07$, $\tau_1 = 1.5$, $\tau_2 = 0.1$.

## Key Experimental Results

### Main Results: Far-OOD Detection Performance

The ID data is CIFAR-10/CIFAR-100, and OOD results are averaged over 6 external datasets:

| ID Dataset (Model) | Method | AUROC↑ | AUPR-S↑ | AUPR-E↑ | FPR-95↓ |
|---------------|------|--------|---------|---------|---------|
| CIFAR-10 (ResNet18) | MSP | 90.72 | 97.89 | 63.48 | 55.21 |
| CIFAR-10 (ResNet18) | Energy | 91.72 | 97.90 | 72.12 | 37.97 |
| CIFAR-10 (ResNet18) | ResFlow‡ | 95.60 | 99.35 | 82.82 | 13.22 |
| CIFAR-10 (ResNet18) | Heatmap† | 96.47 | 99.17 | 83.73 | 15.37 |
| **CIFAR-10 (ResNet18)** | **FlowCon** | **97.19** | **99.43** | **85.65** | 16.26 |
| CIFAR-100 (ResNet18) | MSP | 79.29 | 95.04 | 40.34 | 76.58 |
| CIFAR-100 (ResNet18) | Heatmap† | 86.74 | 96.49 | 58.78 | 52.73 |
| **CIFAR-100 (ResNet18)** | **FlowCon** | **88.22** | **96.85** | **67.89** | **41.85** |

### Near-OOD / Mixed-OOD Detection

| Scenario | Method | AUROC↑ | FPR-95↓ | Description |
|------|------|--------|---------|------|
| C10→C100 Mixed (ResNet) | Energy | 85.60 | 55.20 | post-hoc baseline |
| C10→C100 Mixed (ResNet) | ResFlow | 76.40 | 67.20 | Flow model, poor performance |
| **C10→C100 Mixed (ResNet)** | **FlowCon** | **93.97** | **35.95** | **Best across all metrics** |
| C100→C10 Near (ResNet) | Energy | 77.06 | 81.15 | Difficult near-OOD |
| C100→C10 Near (ResNet) | ResFlow | 58.29 | 79.00 | Flow model collapse |
| **C100→C10 Near (ResNet)** | **FlowCon** | **82.80** | **67.60** | **Best across all metrics** |

In the highly challenging Mixed-OOD scenario, FlowCon achieves an AUROC of 93.97%, representing an improvement of 8.37% over Energy and 17.57% over ResFlow.

### Ablation Study: Impact of $\lambda$ on Performance

In the CIFAR-100 (WideResNet) Far-OOD scenario:

| $\lambda$ Value | AUROC↑ | AUPR-S↑ | AUPR-E↑ | FPR-95↓ | Description |
|-----|--------|---------|---------|---------|------|
| 0.05 | 75.62 | 92.70 | 41.84 | 72.58 | Flow loss weight too low |
| **0.07** | **83.62** | **96.60** | **53.34** | **60.28** | **Optimal balance** |
| 0.30 | 75.75 | 92.76 | 48.61 | 63.67 | Flow loss too large |
| 0.50 | 78.60 | 93.96 | 49.07 | 65.92 | Performance degradation |
| 1.00 | 78.57 | 93.24 | 45.94 | 67.85 | Contrastive loss suppressed |

### Classification Preservation Validation

| Dataset | Model | Original Classifier | FlowCon | Difference |
|--------|------|-----------|---------|------|
| CIFAR-10 | ResNet18 | 94.3% | 94.2% | -0.1% |
| CIFAR-10 | WideResNet | 93.3% | 93.8% | +0.5% |
| CIFAR-100 | ResNet18 | 75.8% | 74.9% | -0.9% |
| CIFAR-100 | WideResNet | 70.9% | 71.1% | +0.2% |

The class distributions learned by FlowCon can be directly used for classification (Bayes decision), with accuracy almost identical to that of the original classifier.

### Key Findings

- FlowCon achieves state-of-the-art or near-state-of-the-art performance across all OOD scenarios on ResNet18 and shows comparable robustness on CIFAR-100 (100 classes).
- Compared to ResFlow which requires 400 models (100 classes $\times$ 4 layers), FlowCon only needs to train **1 model** to perform OOD detection on the penultimate layer features.
- Likelihood histogram analysis reveals that the maximum likelihood of OOD samples in FlowCon **never exceeds** the maximum likelihood of ID samples, solving the classic problem of flow models assigning high likelihood to OOD data.
- UMAP visualizations demonstrate that FlowCon learns elegant class clustering structures, with near-OOD samples overlapping with semantically similar ID classes, which aligns with performance degradation trends.

## Highlights & Insights

- **Elegant Probabilistic Fusion**: Replaces cosine similarity with the Bhattacharyya coefficient as the similarity function for contrastive learning, realizing contrastive learning in the probability distribution space rather than conventional feature space contrast.
- **Single Model Alternative to Multi-Models**: Compared to ResFlow's brute-force approach of training one model per class and layer, FlowCon achieves superior performance on a single features layer using just one model.
- **Classification Preservation**: The contrastive loss not only assists OOD detection but also fully preserves the classification performance of the original classifier—solving both OOD detection and ID classification through a single branch.
- **No OOD Data Required**: The entire training pipeline relies solely on ID data, removing the need for any assumptions about potential OOD data distributions.

## Limitations & Future Work

- **Relatively Weaker Performance on WideResNet**: 128-dimensional features are somewhat too low-dimensional for coupling-layer flow models, as RealNVP/Glow typically perform better on higher-dimensional data. Future work could explore flow architectures tailored for low-dimensional features.
- **Dimensionality Constraints**: Normalizing flows require identical input and output dimensions, restricting the model's flexibility across different classifiers.
- **Longer Training Time**: Training for 700 epochs is still computationally inefficient.
- **Room for Improvement in Near-OOD Scenarios**: As shown by its UMAP visualization, class overlapping in near-OOD remains a common challenge across all baseline methods.

## Related Work & Insights

- **Normalizing Flows (Dinh et al., RealNVP)**: Invertible generative models that provide exact log-likelihood computations.
- **ResFlow (Zisselman et al.)**: Class-level residual flows for OOD detection, which suffer from high training costs.
- **SupCon (Khosla et al.)**: A classical framework for supervised contrastive learning; FlowCon extends it by replacing its similarity function.
- **Kirichenko et al.**: Revealed the issue of flow models assigning high likelihoods to OOD data, which FlowCon tackles via class-conditional distributions.
- **Insight**: The similarity function in contrastive learning can be customized and designed based on specific task characteristics, rather than being restricted to cosine similarity or dot product.

## Rating

- Novelty: ⭐⭐⭐⭐ Utilizing the Bhattacharyya coefficient as the similarity function for flow-based contrastive learning is conceptually novel and mathematically elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Highly comprehensive, covering three OOD scenarios, two classifiers, four evaluation metrics, likelihood histograms, UMAP visualization, classification preservation validation, and $\lambda$ ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, compelling intuitive demonstration from toy experiments, and rigorous equation derivations.
- Value: ⭐⭐⭐⭐ Provides an efficient and feasible new solution for density-based OOD detection, with the single-model design carrying significant engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Adaptive Multi-head Contrastive Learning](adaptive_multihead_contrastive_learning.md)
- [\[ECCV 2024\] Rethinking Unsupervised Outlier Detection via Multiple Thresholding](rethinking_unsupervised_outlier_detection_via_multiple_thresholding.md)
- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[ICLR 2026\] InfoNCE Induces Gaussian Distribution](../../ICLR2026/self_supervised/infonce_induces_gaussian_distribution.md)
- [\[ICML 2026\] Zero-Flow Encoders](../../ICML2026/self_supervised/zero-flow_encoders.md)

</div>

<!-- RELATED:END -->
