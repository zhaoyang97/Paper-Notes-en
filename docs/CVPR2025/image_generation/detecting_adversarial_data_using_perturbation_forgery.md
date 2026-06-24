---
title: >-
  [Paper Note] Detecting Adversarial Data Using Perturbation Forgery
description: >-
  [CVPR 2025][Image Generation][Adversarial Detection] By modeling the Gaussian distribution of adversarial noise and proving its proximity relation, the Perturbation Forgery method is proposed to continuously perturb the noise distribution during training to form an open cover. Combined with sparse masks to generate pseudo-adversarial data to train a binary classifier, using only the noise distribution from a single attack (FGSM) can generalize to detect various unseen attacks…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Adversarial Detection"
  - "Perturbation Forgery"
  - "Noise Distribution"
  - "Open Cover"
  - "Generalization Detection"
date: 2026-05-08
content_hash: 817c10f248784cdf
---

# Detecting Adversarial Data Using Perturbation Forgery

**Conference**: CVPR 2025  
**arXiv**: [2405.16226](https://arxiv.org/abs/2405.16226)  
**Code**: [https://github.com/cc13qq/PFD](https://github.com/cc13qq/PFD)  
**Area**: Image Generation  
**Keywords**: Adversarial Detection, Perturbation Forgery, Noise Distribution, Open Cover, Generalization Detection

## TL;DR
By modeling the Gaussian distribution of adversarial noise and proving its proximity relation, the Perturbation Forgery method is proposed to continuously perturb the noise distribution during training to form an open cover. Combined with sparse masks to generate pseudo-adversarial data to train a binary classifier, using only the noise distribution from a single attack (FGSM) can generalize to detect various unseen attacks, including gradient-based, GAN-based, diffusion-based, and physical attacks, achieving an AUROC of 0.99+ with extremely low inference overhead.

## Background & Motivation

**Background**: Adversarial detection is a defense strategy against adversarial attacks by filtering adversarial samples based on identifying distribution discrepancies between natural and adversarial data. Existing methods primarily design detectors for gradient-based attacks (such as PGD and FGSM).

**Limitations of Prior Work**: (1) Existing detectors suffer from poor generalization, only detecting attack types seen during training and failing against unseen attacks. (2) New adversarial attacks based on generative models (GAN-based, Diffusion-based) generate non-uniform and anisotropic perturbations, which are difficult for existing methods to detect. (3) Some high-performance methods (e.g., EPSAD) have massive inference overhead, taking nearly 400 seconds to process 100 ImageNet images, making them impractical.

**Key Challenge**: A low-overhead detector capable of generalizing to both gradient-based and generative adversarial attacks is needed, but existing methods are either poorly generalized or computationally expensive.

**Goal**: To design a model-agnostic, low-overhead adversarial detector that can generalize to various unseen attacks, including gradient-based, GAN-based, diffusion-based, and physical attacks.

**Key Insight**: Mathematically, by modeling adversarial noise as a truncated Gaussian distribution, the authors discovered a proximity relation (bounded Wasserstein distance) between different attack noise distributions, thereby deriving that all adversarial noise distributions can be contained within an "open cover".

**Core Idea**: By continuously perturbing the noise distribution of a known attack to form an open cover, the detector trained on this open cover can generalize to detect all types of unseen attacks.

## Method

### Overall Architecture
The overall pipeline of Perturbation Forgery is as follows: Input natural images $\to$ first estimate the Gaussian distribution parameters of the adversarial noise using a common attack (FGSM) $\to$ randomly perturb the distribution parameters for each training batch to generate new proximal distributions $\to$ sample noise from the perturbed distributions $\to$ generate sparse masks using saliency detection and GradCAM $\to$ inject local noise by converting global noise using the masks into half of the natural data to generate pseudo-adversarial data $\to$ train a binary classifier using both pseudo-adversarial data and clean data. At inference time, the trained detector is directly used for binary classification to determine whether the input is an adversarial sample.

### Key Designs

1. **Noise Distribution Perturbation**:

    - **Function**: Generates a large number of proximal noise distributions by continuously perturbing the parameters of a known attack's noise distribution, forming an open cover of adversarial noise distributions.
    - **Mechanism**: First, generate adversarial noise using FGSM, flatten the noise, and estimate the mean $\hat\mu$ and covariance $\hat\Sigma$ of the multivariate Gaussian distribution. For each training batch, a random perturbation is added to the mean: $\hat\mu_i = \hat\mu + \alpha_i \cdot m_i$ ($m_i \sim \mathcal{N}(0, I)$, $\alpha_i \sim U(-\epsilon_\mu, \epsilon_\mu)$); the covariance is processed similarly. Under $n$ batches, a set of distributions approximating the open cover is acquired.
    - **Design Motivation**: The authors prove via Theorem 1 that under the same $\ell^p$ norm constraint, all adversarial noise distributions are proximal to each other, thus an open cover exists. By training on this open cover, the detector learns to distinguish between the distributions of natural and adversarial data, thereby generalizing to unseen attacks.

2. **Sparse Mask Generation**:

    - **Function**: Converts global noise into local sparse noise to simulate the non-uniform perturbation patterns of generative-based attacks.
    - **Mechanism**: Map natural samples using a saliency detection model and GradCAM to generate attention maps. Take the union of both to represent high-frequency/salient regions ($\text{Mask}_1$). Use the Sobel operator to extract high-frequency points from the gradient map and sparsify them ($\text{Mask}_2$). The final mask is the intersection of both: $\text{Mask} = \text{Mask}_1 \cap \text{Mask}_2$.
    - **Design Motivation**: Perturbations of generative attacks (e.g., Diff-PGD) are concentrated in high-frequency and salient regions, while being weak in low-frequency and background regions. Physical attacks also tend to add patches in localized areas. The sparse mask forces the detector to focus on these local noise patterns, overcoming the limitation where pure distribution perturbation can only simulate global noise.

3. **Pseudo-Adversarial Data Production**:

    - **Function**: Integrates sampled noise with sparse masks to inject local noise into natural samples, producing pseudo-adversarial data for training.
    - **Mechanism**: Sample noise $\hat\eta_{i,k}$ from the perturbed distribution, exclude low-likelihood samples using a probability density threshold $\gamma_p$, and perform a Hadamard product with the mask to get local noise, which is then added to natural images: $\hat{x}_{i,k} = x_{i,k} + \hat\eta_{i,k} \otimes \text{Mask}(x_{i,k})$.
    - **Design Motivation**: Replacing real adversarial data with pseudo-adversarial data to train the detector avoids the need for real attack data, realizing model-agnostic detection.

### Loss & Training
For each batch, half of the natural data is converted to pseudo-adversarial data and combined with the other half of clean data to form the training set. A standard cross-entropy loss is used to train the binary classifier. Throughout the training process, the noise distribution parameters are randomly perturbed for each batch, ensuring coverage of sufficient distribution variants.

## Key Experimental Results

### Main Results

| Dataset/Attack | Metric | Ours (PFD) | EPSAD | SPAD | Description |
|------------|------|---------|-------|------|------|
| ImageNet100 Gradient Attacks (8 types) | Avg. AUROC | 0.992 | 0.996 | 0.983 | Comparable to EPSAD but ~80× faster inference |
| ImageNet100 Generative Attacks (5 types) | Avg. AUROC | 0.947 | 0.472 | 0.903 | Significantly outperforms EPSAD |
| Face Physical Attacks (4 types) | Avg. AUROC | 0.992 | 0.959 | 0.983 | Comprehensively leading |
| Inference Time (100 images) | Seconds | 4.85 | 396.81 | 4.56 | ~82× faster than EPSAD |

### Ablation Study

| Configuration | Gradient Attack AUROC | Generative Attack AUROC | Description |
|------|---------------|---------------|------|
| Full model | 0.992 | 0.947 | Full model |
| w/o Noise Distribution Perturbation | 0.971 | 0.867 | Removing distribution perturbation significantly drops generalization |
| w/o Sparse Mask | 0.990 | 0.891 | Removing mask significantly drops generative attack detection |
| w/o Probability Density Filtering | 0.988 | 0.932 | Filtering low-likelihood noise has minor contribution |

### Key Findings
- Noise distribution perturbation is the core contributor to generalization; without it, generative attack detection drops by approximately 8 percentage points.
- Sparse masks contribute the most to generative attack detection (+5.6 AUROC) by simulating the local perturbation characteristics of generative attacks.
- EPSAD severely fails under generative attacks (AUROC only 0.47), whereas the proposed method maintains 0.94+, showing that diffusion model-guided detection schemes are inherently disadvantaged against generative attacks.
- The proposed method also performs remarkably well in detecting physical attacks on face datasets (AUROC 0.99+), reflecting that sparse masks can effectively simulate patch-like physical attacks.

## Highlights & Insights
- **Theory-Driven Method Design**: By modeling adversarial noise as a Gaussian distribution and using Wasserstein distance to prove proximity, a solid mathematical foundation is provided, elevating it beyond purely empirical methods. This concept of "first modeling noise distribution, then leveraging topological properties" can be applied to other detection tasks requiring generalization to unseen distributions.
- **Forgery as a Substitute for Truth**: Completely replacing real adversarial samples with pseudo-adversarial data for detector training makes it model-agnostic and attack-independent, which is much more practical than traditional methods relying on specific attack algorithms for training data.
- **Trade-off between Inference Speed and Performance**: Achieving AUROC of 0.99+ while maintaining an inference time of only 4.85 seconds/100 images, which is 82 times faster than EPSAD.

## Limitations & Future Work
- The theoretical assumption models adversarial noise as an isotropic Gaussian distribution ($\sigma^2 I_d$), but actual adversarial noise is often anisotropic; the degree of relaxation for this assumption warrants further analysis.
- Sparse mask generation relies on pre-trained saliency detection models and GradCAM, introducing additional computation and model dependencies.
- Validation is mainly conducted on ImageNet100 and physical face datasets; evaluation on larger-scale datasets (e.g., full ImageNet) is missing.
- The detector itself is an auxiliary binary classification model, which increases system complexity during actual deployment.

## Related Work & Insights
- **vs EPSAD**: EPSAD uses diffusion models to boost detection, achieving 0.999 AUROC on gradient attacks, but it is 80× slower in inference and fails under generative attacks (0.15-0.47). PFD avoids the diffusion process at inference by using forged training data.
- **vs SPAD**: SPAD uses manually designed pseudo-noise for data augmentation but lacks theoretical guidance, resulting in weaker generative attack detection (0.90) compared to our model (0.95).
- **vs LID/LiBRe**: These methods rely on target-model internal features, meaning they are not model-agnostic and lack sufficient generalization.

## Rating
- Novelty: ⭐⭐⭐⭐ The theoretical derivation is novel, but the concept of "forged training data" is not entirely new in adversarial detection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 types of attacks (gradient/GAN/diffusion/physical) with comprehensive multi-dataset validation and thorough ablation studies.
- Writing Quality: ⭐⭐⭐⭐ The theoretical section is clear, but containing many mathematical symbols, requiring some mathematical background.
- Value: ⭐⭐⭐⭐ Highly practical, with fast inference and strong generalization capabilities, making it ideal for actual deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] ForgeLens: Data-Efficient Forgery Focus for Generalizable Forgery Image Detection](../../ICCV2025/image_generation/forgelens_data-efficient_forgery_focus_for_generalizable_forgery_image_detection.md)
- [\[CVPR 2025\] Instant Adversarial Purification with Adversarial Consistency Distillation](instant_adversarial_purification_with_adversarial_consistency_distillation.md)
- [\[CVPR 2025\] Uncertainty-guided Perturbation for Image Super-Resolution Diffusion Model](uncertainty-guided_perturbation_for_image_super-resolution_diffusion_model.md)
- [\[CVPR 2025\] Divide and Conquer: Heterogeneous Noise Integration for Diffusion-based Adversarial Purification](divide_and_conquer_heterogeneous_noise_integration_for_diffusion-based_adversari.md)
- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](../../NeurIPS2025/image_generation/detecting_generated_images_by_fitting_natural_image_distributions.md)

</div>

<!-- RELATED:END -->
