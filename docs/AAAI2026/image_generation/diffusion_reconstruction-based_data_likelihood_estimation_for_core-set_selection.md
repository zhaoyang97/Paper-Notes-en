---
title: >-
  [Paper Note] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection
description: >-
  [AAAI 2026][Image Generation][Core-set selection] This paper proposes using the partial reverse denoising reconstruction bias of diffusion models as a theoretically grounded approximation of data likelihood, combined with information bottleneck theory for optimal reconstruction timestep selection, enabling distribution-aware core-set selection that achieves near-full-dataset training performance on ImageNet with only 50% of the data.
tags:
  - AAAI 2026
  - Image Generation
  - Core-set selection
  - diffusion models
  - data likelihood
  - reconstruction bias
  - information bottleneck
date: 2026-05-08
content_hash: 320b0ce80d1594ea
---

# Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection

**Conference**: AAAI 2026
**arXiv**: [2511.19274](https://arxiv.org/abs/2511.19274)
**Code**: [GitHub](https://github.com/mchen725/DRD)
**Area**: Image Generation
**Keywords**: Core-set selection, diffusion models, data likelihood, reconstruction bias, information bottleneck

## TL;DR

This paper proposes using the partial reverse denoising reconstruction bias of diffusion models as a theoretically grounded approximation of data likelihood, combined with information bottleneck theory for optimal reconstruction timestep selection, enabling distribution-aware core-set selection that achieves near-full-dataset training performance on ImageNet with only 50% of the data.

## Background & Motivation

Core-set selection aims to identify representative subsets from large-scale datasets for efficient model training. Existing score-based methods (e.g., Forgetting Score, EL2N, AUM) rely on heuristic proxy signals—forgetting events, model uncertainty, loss values—but suffer from fundamental limitations:

**Lack of explicit data likelihood modeling**: Heuristic scores fail to capture subtle yet critical structural features of the data distribution.

**Insufficient discriminability**: As shown by t-SNE visualizations, Forgetting Score and EL2N stratify data in a manner similar to random sampling, with samples from different score intervals heavily intermixed in feature space.

**Lack of theoretical grounding**: Score selection resembles empirical tuning rather than principled design.

Diffusion models are generative models that explicitly learn the data distribution. The core insight is that **reconstruction error is inversely proportional to data likelihood**—samples in high-density regions incur small reconstruction errors, while low-density or outlier samples incur large errors. Therefore, diffusion reconstruction bias serves as a natural, theoretically justified signal for data filtering.

## Method

### Overall Architecture

The method consists of two core steps:

1. **Diffusion Reconstruction Bias (DRD) Score**: For each sample $x_0$, partial forward noising → reverse denoising → reconstruction bias computation.
2. **Information Bottleneck Timestep Selection**: An information-theoretic method selects the optimal reconstruction timestep $t^*$ to maximize the discriminability of the bias signal.

A **Best Window Selection (BWS)** strategy then slides a window over the sorted score list to identify the optimal subset.

### Key Designs

#### 1. Theoretical Connection Between Reconstruction Bias and Data Likelihood

**Theorem 1** (Inverse Dependence of Reconstruction Bias on Log-Likelihood): Given $x_0$ noised to $x_t$ and denoised via DDPM to obtain $x_{0:t}'$, the expected reconstruction bias satisfies the lower bound:

$$\mathbb{E}_\epsilon[\|\Delta x_0(t)\|^2] \geq -\kappa(t) \log q(x_0) + \mathcal{C}_{\text{noise}}(t)$$

where $\kappa(t) = \frac{1}{t}\sum_{s=1}^{t}\frac{1}{\sigma_s^2}$. Intuitively, high-density samples yield small reconstruction bias while low-density samples yield large bias.

In practice, DDIM is used for deterministic reconstruction, and bias is measured by LPIPS (perceptual similarity), which aligns better with human judgment than pixel-level $L_2$.

#### 2. Information Bottleneck-Based Optimal Timestep Selection

Timestep selection is critical—a too-small $t$ yields $x_0' \approx x_0$ (uninformative bias), while a too-large $t$ drives $x_t$ toward pure noise (reconstruction becomes independent of $x_0$).

The problem is formulated within the Information Bottleneck (IB) framework:

$$\min_t \mathbb{E}_{x_0}[\mathcal{I}(x_0; x_t) - \beta \mathcal{I}(x_t; c)]$$

where $\mathcal{I}(x_0; x_t)$ is approximated by SNR$(t)$, and $\mathcal{I}(x_t; c)$ denotes the mutual information between the noised data and class labels.

The optimal timestep is defined as the point of maximum rate of mutual information decrease:

$$t^* = \arg\max_t \left|\frac{\partial \mathcal{I}(x_t; c)}{\partial t}\right|, \quad \text{s.t.} \; \text{SNR}(t) \in [0.05, 1]$$

**Lemma 1** proves this derivative is equivalent to $|\mathbb{E}_{x_0,c}[\frac{\partial}{\partial t}\log p(c|x_t)]|$, which can be approximated via a diffusion classifier. In practice, finite differences are used:

$$\frac{\partial}{\partial t}\log p_\theta(c|x_t) \approx \frac{\log p_\theta(c|x_{t+1}) - \log p_\theta(c|x_{t-1})}{2}$$

Monte Carlo estimation is performed by sampling $B=20$ samples per class, completing timestep selection within minutes.

#### 3. Moderate-Likelihood Data Is Optimal

Experiments reveal an important finding: **data of moderate likelihood is most effective for core-set construction**. Sorting samples by DRD score into quintiles, models trained on the middle quintile generalize best. This parallels the effect of data augmentation (e.g., CutMix)—moderate-likelihood samples often contain ambiguous or partial semantic cues, encouraging the model to learn more robust features.

### Loss & Training

- A pretrained DiT (Diffusion Transformer) model is used with a full schedule of $T=1000$ steps.
- Inference uses DDIM sampling with $T=50$ steps, performing partial reconstruction from a selected $t < 50$.
- 20 random noise samples are drawn to approximate $p_\theta(c|x_t)$ in the diffusion classifier.
- BWS with a step size of 5% is used by default to find the optimal window.
- All results are reproducible on a single RTX 4090.

## Key Experimental Results

### Main Results

**ImageNet-1K Core-Set Selection (ResNet-50, Test Accuracy %)**

| Method | 10% | 30% | 50% | 75% |
|--------|-----|-----|-----|-----|
| Random | ~60 | ~70 | ~74 | ~76 |
| Forgetting | ~62 | ~71 | ~74 | ~76 |
| EL2N | ~60 | ~70 | ~74 | ~76 |
| BWS (Forgetting) | ~64 | ~73 | ~75 | ~77 |
| **DRD (Ours)** | **~66** | **~75** | **~76** | **~77** |

DRD consistently outperforms all baselines across all selection ratios, achieving near-full-dataset performance with 50% of the data.

**Table 1: Comparison of Score-Guided Selection Strategies on ImageWoof**

| Strategy | Score | 10% | 30% | 75% |
|----------|-------|-----|-----|-----|
| CCS | Forgetting | 68.4 | 80.9 | 86.4 |
| CCS | DRD | 68.9 | 80.1 | 86.9 |
| BWS | Forgetting | 70.1 | 81.5 | 87.3 |
| **BWS** | **DRD** | **74.5** | **83.3** | **88.7** |

BWS+DRD achieves the highest accuracy at all ratios, with a particularly pronounced advantage at low ratios (10%): +4.4 over BWS+Forgetting.

### Ablation Study

**Table 2: Sensitivity to Timestep Selection Hyperparameters (ImageWoof)**

| B | #ε | 10% | 30% | 75% |
|---|----|-----|-----|-----|
| 20 | 5 | 67.8 | 81.1 | 86.3 |
| 20 | 20 | **74.5** | **83.3** | 88.5 |
| 40 | 20 | 74.7 | 83.3 | 88.5 |

Using too few noise samples (#ε = 5) leads to significant performance degradation; performance stabilizes at #ε ≥ 20.

### Key Findings

1. **DRD is more discriminative than Forgetting and EL2N**: t-SNE visualizations show that DRD stratification is semantically coherent, whereas other methods are indistinguishable from random sampling.
2. **Moderate-likelihood data is optimal**: The BWS optimal window typically starts in the 20%–40% range; samples at extreme likelihood levels (too high or too low) contribute less to training.
3. **IB timestep selection vs. grid search**: The IB method completes in minutes, whereas grid search requires full training at each candidate timestep—an enormous difference in computational cost.
4. **Cross-architecture generalization**: DRD achieves top performance on ResNet-18, ResNet-50, EfficientNet-B0, and ViT.
5. The selected timestep typically clusters around $t \approx 20$, precisely where the signal-to-noise ratio is balanced.

## Highlights & Insights

1. **Theory-driven**: Rather than introducing yet another heuristic score, this work rigorously derives the connection between reconstruction bias and data likelihood from the diffusion model ELBO (Theorem 1) and employs information bottleneck theory (Lemma 1) to guide timestep selection.
2. **Distribution-aware scoring**: The DRD score carries genuine physical meaning—directly reflecting a sample's position on the data manifold, where high score = low likelihood = deviation from the main distribution.
3. **Explanation for moderate-likelihood preference**: This finding unifies with curriculum learning and data augmentation under a common explanatory framework—the most valuable training samples are neither too easy nor too hard.
4. **Computationally accessible**: All results are reproducible on a single GPU; timestep selection requires only a few minutes.

## Limitations & Future Work

1. **Dependency on pretrained diffusion models**: A DiT pretrained on the target dataset is required, which may not be directly applicable to novel-domain data.
2. **Scalability to large class counts**: Computing classification probabilities via the diffusion classifier is costly when the number of classes is very large (e.g., ImageNet with 1,000 classes).
3. **Validation limited to image classification**: Generalizability to other tasks such as detection, segmentation, and NLP has not been verified.
4. **BWS requires a validation set**: Optimal window search requires evaluation on a full validation set and is therefore not entirely unsupervised.
5. Future work may explore combining DRD with active learning and continual learning.

## Related Work & Insights

- **BWS (Choi et al., 2024)**: Proposes the optimal window search strategy, using Forgetting Score as the default ranking criterion.
- **EL2N / Forgetting Score**: Classical heuristic filtering methods; this work demonstrates their insufficient discriminability.
- **Diffusion Classifier (Li et al., 2023)**: Approximates class posterior probabilities via ELBO; adopted here to compute $p_\theta(c|x_t)$.
- **Information Bottleneck Theory (Tishby et al., 2000)**: Elegantly applied in this work to the timestep selection problem.
- Insight: The "reconstruction capability" of diffusion models can serve not only for generation but also as a data quality measurement tool—opening a new application paradigm for diffusion models.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (theoretically grounded, entirely new scoring criterion)
- Technical Depth: ⭐⭐⭐⭐⭐ (theorem proofs + information bottleneck + systematic analysis)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 datasets + 4 architectures + strategy comparison + hyperparameter analysis + visualization)
- Value: ⭐⭐⭐⭐ (single-GPU reproducible, substantial gains)
- Overall: 9.0/10

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Realistic Face Reconstruction from Facial Embeddings via Diffusion Models](realistic_face_reconstruction_from_facial_embeddings_via_diffusion_models.md)
- [\[CVPR 2026\] DMin: Scalable Training Data Influence Estimation for Diffusion Models](../../CVPR2026/image_generation/dmin_scalable_training_data_influence_estimation_for_diffusion_models.md)
- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](../../ICLR2026/image_generation/sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)
- [\[ICLR 2026\] Learning a Distance Measure from the Information-Estimation Geometry of Data](../../ICLR2026/image_generation/learning_a_distance_measure_from_the_information-estimation_geometry_of_data.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)

</div>

<!-- RELATED:END -->
