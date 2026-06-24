---
title: >-
  [Paper Note] UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning
description: >-
  [CVPR 2025][Multimodal VLM][Transductive few-shot learning] UNEM is proposed to unroll each iteration of the Generalized EM (GEM) algorithm as a neural network layer. It automatically optimizes the **class-balance hyperparameter** $\lambda$ and **temperature scaling** $T$ through end-to-end learning. It achieves an average accuracy of 77.8% under the vision-language setting across 11 fine-grained datasets (vs. 73.6% of EM-Dirichlet) and up to a 10% gain under the vision-only…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Transductive few-shot learning"
  - "EM algorithm unrolling"
  - "Hyperparameter learning"
  - "CLIP"
  - "Dirichlet distribution"
date: 2026-05-08
content_hash: 0803e49fc7219e59
---

# UNEM: UNrolled Generalized EM for Transductive Few-Shot Learning

**Conference**: CVPR 2025  
**arXiv**: [2412.16739](https://arxiv.org/abs/2412.16739)  
**Code**: [https://github.com/ZhouLong0/UNEM-Transductive](https://github.com/ZhouLong0/UNEM-Transductive)  
**Area**: Multimodal VLM  
**Keywords**: Transductive few-shot learning, EM algorithm unrolling, Hyperparameter learning, CLIP, Dirichlet distribution

## TL;DR
UNEM is proposed to unroll each iteration of the Generalized EM (GEM) algorithm as a neural network layer. It automatically optimizes the **class-balance hyperparameter** $\lambda$ and **temperature scaling** $T$ through end-to-end learning. It achieves an average accuracy of 77.8% under the vision-language setting across 11 fine-grained datasets (vs. 73.6% of EM-Dirichlet) and up to a 10% gain under the vision-only setting.

## Background & Motivation
Transductive few-shot learning (FSL) exploits statistical information from unlabeled data by jointly inferring classes of a batch of query samples, significantly outperforming inductive methods that infer each sample independently. However, existing transductive methods introduce key hyperparameters (especially $\lambda$ controlling the degree of class balance), whose optimal values vary across datasets and pre-trained models.

**Limitations of Prior Work**: As shown in Fig. 1, in the EM-Dirichlet algorithm, the class-balance hyperparameter $\lambda$ heavily impacts accuracy, and its optimal value can differ by several orders of magnitude across datasets (e.g., the optimal $\lambda$ is approximately 4 for Food101 but 2000 for SUN397). The current practice relies on exhaustive grid search on validation sets, which is not only **computationally infeasible** (especially with combinations of multiple hyperparameters) but also prone to being **suboptimal** due to coarse search ranges.

**Key Challenge**: The massive impact of hyperparameters on performance vs. the infeasibility of exhaustive search. Especially when considering both hyperparameters $\lambda$ (class balance) and $T$ (softness/hardness of prediction) simultaneously, 2D grid search becomes increasingly intractable.

**Key Insight**: Introduce the "learning to optimize" paradigm—unrolling the iterative optimization algorithm into a neural network, where hyperparameters become learnable parameters optimized automatically via backpropagation.

## Method

### Overall Architecture
UNEM unrolls $L$ iterations of the generalized EM algorithm into an $L$-layer neural network. Each layer corresponds to one EM iteration containing three steps: updating distribution parameters $\theta_k$, updating class proportions $\pi_k$, and updating assignment vectors $u_n$. Each layer has independent hyperparameters $(\lambda^{(\ell)}, T^{(\ell)})$, which are learned by minimizing the cross-entropy loss on the validation set. It supports both Gaussian distribution (for vision-only models) and Dirichlet distribution (for vision-language models like CLIP).

### Key Designs

1. **Generalized EM Formulation**
    - Function: Unifies existing transductive few-shot methods into a general framework as special cases
    - Mechanism: The optimization objective is $\min_{u,\theta} \mathcal{L}(u,\theta) + \lambda\Psi(u) + T\Phi(u)$, where:
        - $\mathcal{L}$: Negative log-likelihood (clustering term) which has an implicit bias toward class balance
        - $\Psi$: Shannon entropy of the class distribution, $\Psi(u) = -\sum_k \pi_k \ln \pi_k$, which controls class balance
        - $\Phi$: Entropy barrier of assignment vectors, $\Phi(u) = \sum_n \sum_k u_{n,k} \ln u_{n,k}$, which controls the softness/hardness of predictions
    - Design Motivation: Recovers standard EM when $T=1, \lambda=|\mathbb{Q}|$, and recovers EM-Dirichlet when $T=1$. Explicitly formulated hyperparameters $\lambda$ and $T$ enable them to be learned
    - Closed-form solution for assignment updates: $u_n^{(\ell+1)} = \text{softmax}\left(\frac{1}{T}\left(\ln p(z_n|\theta_k^{(\ell+1)}) + \frac{\lambda}{|\mathbb{Q}|}\ln(\pi_k^{(\ell+1)})\right)_k\right)$

2. **Unrolling Architecture**
    - Function: Converts the iterative algorithm into an end-to-end trainable neural network to automatically learn hyperparameters
    - Mechanism: $L$ iterations $\rightarrow$ $L$-layer network, where each layer $\mathcal{L}^{(\ell)}$ executes one update of GEM, and the hyperparameters $(\lambda^{(\ell)}, T^{(\ell)})$ of the $\ell$-th layer are independently learnable. The network has only $2L$ learnable parameters (only 20 parameters for $L=10$)
    - Design Motivation:
        - Hyperparameters can **vary layer by layer**—early and later iterations might require different class-balance intensities and temperatures
        - Learning via gradient descent is more efficient than grid search and can find globally optimal configurations
        - Parameter constraints: $\lambda^{(\ell)} = \text{Softplus}(a^{(\ell)})$ guarantees non-negativity; $T^{(\ell)} = 1 + \text{Softplus}(b^{(\ell)})$ guarantees $\geq 1$ to avoid vanishing gradients
    - Training Loss: Standard cross-entropy $L_c(w) = \sum_{n \in \mathbb{Q}} \sum_k y_{n,k} \log(u_{n,k}^{(L)})$

3. **Support for Dual Distribution Models (Gaussian + Dirichlet)**
    - Function: Enables UNEM to support both traditional vision-only pre-trained models and vision-language models such as CLIP
    - Mechanism:
        - **UNEM-Gaussian**: Assumes features $z_n$ follow a Gaussian distribution $p(z_n|\theta_k) \propto \exp(-\frac{1}{2}\|z_n-\theta_k\|^2)$, where parameters $\theta_k$ denote class prototypes, used for backbones like ResNet/WRN
        - **UNEM-Dirichlet**: Assumes CLIP's softmax probability output follows a Dirichlet distribution $p(z_n|\theta_k) = \frac{1}{\mathcal{B}(\theta_k)}\prod_i z_{n,i}^{\theta_{k,i}-1}$, used for CLIP
    - Design Motivation: CLIP outputs probability vectors on a simplex, making the Gaussian assumption inappropriate. The Dirichlet distribution naturally models data on a simplex, and EM-Dirichlet has been validated as effective

## Key Experimental Results

### Vision-Only: mini-ImageNet (Tab. 1, WRN28-10 backbone)

| Method | 5-shot | 10-shot | 20-shot |
|------|--------|---------|---------|
| PADDLE | 62.6 | 73.0 | 79.2 |
| α-TIM | 71.5 | 75.2 | 78.3 |
| **UNEM-Gaussian** | **71.6** | **79.2** | **83.7** |
- Outperforms PADDLE by 4.5 percentage points in the 20-shot setting

### Vision-Only: tiered-ImageNet (160 classes, Tab. 1, WRN28-10)

| Method | 5-shot | 10-shot | 20-shot |
|------|--------|---------|---------|
| PADDLE | 43.9 | 59.4 | 69.9 |
| **UNEM-Gaussian** | **54.1** | **66.8** | **74.7** |
- A 10.2 percentage point gain in the 5-shot setting!

### Vision-Language: 11 Fine-Grained Datasets (Tab. 3, CLIP, 4-shot)

| Method | Food101 | Flowers | Cars | SUN397 | ImageNet | Average |
|------|---------|---------|------|--------|----------|------|
| Tip-Adapter (Inductive) | 76.7 | 83.2 | 63.9 | 66.7 | 62.7 | 68.3 |
| EM-Dirichlet (Transductive) | 88.7 | 91.3 | 73.5 | 80.9 | 78.4 | 73.6 |
| **UNEM-Dirichlet** | **91.4** | **95.6** | **80.0** | **88.5** | **83.1** | **77.8** |
- An average gain of 4.2 percentage points, with a 6.5 gain on Cars and 7.6 gain on SUN397

### CUB Fine-Grained Bird Classification (Tab. 2)

| Method | 5-shot | 10-shot | 20-shot |
|------|--------|---------|---------|
| PADDLE | 71.2 | 81.8 | 86.8 |
| **UNEM-Gaussian** | **78.5** | **85.3** | **88.6** |

### Key Findings
- Layer-wise variable hyperparameters vs. global fixed hyperparameters: Layer-wise variable hyperparameters consistently perform better, validating the hypothesis that hyperparameters should change programmatically across iterations
- Learned $\lambda$ values vary significantly across datasets (e.g., ImageNet $\sim 300$ vs. SUN397 $\sim 2000$), but UNEM automatically adapts to each
- Extremely lightweight training with only 20 learnable parameters ($L=10$ layers $\times 2$ hyperparameters)

## Highlights & Insights
- **First Application of Algorithm Unrolling to Hyperparameter Optimization in Few-Shot Learning**: Introduces the "learning to optimize" paradigm to the FSL field, addressing the long-standing hyperparameter search dilemma
- **Extremely Lightweight**: The entire network requires only 20 learnable parameters (contrasting with millions in standard deep neural networks) while bringing massive performance improvements
- **Unified Framework**: The GEM formulation unifies standard EM, EM-Dirichlet, and $\alpha$-TIM under a single umbrella as special cases, making a solid theoretical contribution
- **High Practicality**: Eliminates the need to perform grid searches for every new dataset, generalizing directly after learning—substantially lowering actual deployment costs
- **Insights into Layer-wise Hyperparameter Dynamics**: Early iterations require stronger class-balance constraints, while later iterations demand harder assignments—a finding of theoretical value for understanding EM optimization dynamics

## Limitations & Future Work
- The number of unrolled layers $L=10$ is static, which might be redundant or insufficient for certain tasks
- Validated only on Gaussian and Dirichlet distribution models; generalization to other exponential family distributions remains unverified
- Training hyperparameters on the validation set requires a moderate amount of labeled data (though significantly less than grid search)
- Focuses solely on image classification; extensions to other visual tasks like object detection and segmentation have not been explored
- The unrolled network needs to be retrained when the number of actual classes $K_{eff}$ in the query set changes

## Related Work & Insights
- EM-Dirichlet $\rightarrow$ Direct baseline of this work; UNEM automates its hyperparameters
- Algorithm Unrolling (LISTA, ADMM-Net) $\rightarrow$ A general paradigm turning optimization iterations into learnable network layers, applied to FSL of this nature for the first time
- Scarcity of transductive approaches for CLIP $\rightarrow$ This work fills the methodological gap of CLIP + transductive FSL
- Insight: Hyperparameters in iterative optimization algorithms should not be static—different stages of iteration may demand distinct values, and the unrolling paradigm naturally enables this flexibility

## Rating
⭐⭐⭐⭐ — The proposed approach is clear and elegant, innovatively applying the classic algorithm unrolling paradigm to few-shot learning and achieving substantial performance gains with negligible learnable parameters. The unified GEM framework is theoretically sound, and the experiments are thorough. It makes a significant contribution toward filling the research gap in transductive few-shot learning using CLIP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Generalized Few-Shot 3D Point Cloud Segmentation with Vision-Language Model](generalized_few-shot_3d_point_cloud_segmentation_with_vision-language_model.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[CVPR 2025\] Rethinking Few-Shot Adaptation of Vision-Language Models in Two Stages](rethinking_few-shot_adaptation_of_vision-language_models_in_two_stages.md)
- [\[ICCV 2025\] Causal Disentanglement and Cross-Modal Alignment for Enhanced Few-Shot Learning](../../ICCV2025/multimodal_vlm/causal_disentanglement_and_cross-modal_alignment_for_enhanced_few-shot_learning.md)
- [\[CVPR 2025\] Single Domain Generalization for Few-Shot Counting via Universal Representation Matching](single_domain_generalization_for_few-shot_counting_via_universal_representation_.md)

</div>

<!-- RELATED:END -->
