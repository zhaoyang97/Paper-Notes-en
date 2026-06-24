---
title: >-
  [Paper Note] Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models
description: >-
  [CVPR 2025][Image Generation][Image Protection] This paper proposes FastProtect, the first latency-focused image protection framework. By replacing traditional image-by-image iterative optimization with pre-trained Mixture-of-Perturbations (MoP), combined with Multi-Layer Protection Loss to enhance training effects, as well as Adaptive Targeted Protection and Adaptive Protection Strength to optimize inference, FastProtect achieves real-time protection that is 175× faster than…
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Image Protection"
  - "Adversarial Perturbation"
  - "Diffusion Models"
  - "Personalized Defense"
  - "Real-time Protection"
  - "Mixture of Perturbations"
date: 2026-05-08
content_hash: 2ea0d12df2a8cc53
---

# Nearly Zero-Cost Protection Against Mimicry by Personalized Diffusion Models

**Conference**: CVPR 2025  
**arXiv**: [2412.11423](https://arxiv.org/abs/2412.11423)  
**Code**: [https://webtoon.github.io/impasto](https://webtoon.github.io/impasto)  
**Area**: Image Generation/Image Protection  
**Keywords**: Image Protection, Adversarial Perturbation, Diffusion Models, Personalized Defense, Real-time Protection, Mixture of Perturbations

## TL;DR

This paper proposes FastProtect, the first latency-focused image protection framework. By replacing traditional image-by-image iterative optimization with pre-trained Mixture-of-Perturbations (MoP), combined with Multi-Layer Protection Loss to enhance training effects, as well as Adaptive Targeted Protection and Adaptive Protection Strength to optimize inference, FastProtect achieves real-time protection that is 175× faster than the existing fastest method PhotoGuard (0.04s vs 7s for processing a $512^2$ image on an A100 GPU), while maintaining comparable protection efficacy and superior invisibility.

## Background & Motivation

**Background**: Personalization technologies for diffusion models (e.g., DreamBooth, LoRA, Textual Inversion) have enabled malicious users to mimic others' artistic styles or generate deepfakes using a small number of reference images. Existing protection methods (AdvDM, PhotoGuard, Mist, Glaze, Anti-DB, Impasto) disable personalized fine-tuning by adding adversarial perturbations to images, but all of them rely on inference-time iterative optimization (PGD), which takes 7-225 seconds to protect a $512^2$ image (on an A100 GPU).

**Limitations of Prior Work**: (1) **Latency** is the biggest obstacle—processing an image on a CPU takes 5-120 minutes, making it completely unusable for ordinary users; (2) The trade-off between perturbation invisibility and protection efficacy is still suboptimal, with noticeable artifacts particularly on cartoon/illustration styles with simple textures; (3) As the image resolution increases (e.g., to $2048^2$), the latency increases exponentially, whereas modern artistic works are mostly high-resolution. These three factors collectively hinder the practical adoption of protection technologies.

**Key Challenge**: Existing Universal Adversarial Perturbation (UAP) techniques can eliminate inference-time optimization through pre-training, but the single perturbation of UAP is image-agnostic. Applying it directly to image protection tasks severely degrades the protection efficacy (dropping the FID from 227.6 to 207.6). How to restore the protection efficacy of image-specific optimization while maintaining the speed advantage of UAP?

**Goal**: Design an image protection framework that satisfies a three-fold demand—effective protection, invisibility, and real-time latency—enabling ordinary users to protect their images on low-compute devices.

**Key Insight**: (1) Train multiple perturbations (instead of one) and adaptively select them based on the VAE latent code of the input image; (2) Calculate the protection loss in the multi-layer feature space of VAE to enhance the training effect; (3) Predictively select the target image and perturbation strength adaptively based on the input texture complexity during inference.

**Core Idea**: Replace image-by-image PGD iteration with pre-trained Mixture-of-Perturbations (MoP), and achieve semi-image-specific protection through VAE latent-based clustering allocation, reducing the inference cost from seconds to milliseconds.

## Method

### Overall Architecture

FastProtect consists of two phases: training and inference. **Training phase**: First, K-means++ is performed on the VAE latents of training images to establish K group assignment functions $\mathcal{A}$. Then, a Multi-Layer Protection Loss is used to train a global perturbation $\delta_g$ and K group perturbations $\Delta = \{\delta_1,...,\delta_K\}$. Three sets of MoP are trained separately for target images with three different levels of mode repetition. **Inference phase**: The input image is fed into the VAE encoder to obtain its latent code. The most matching target image and its corresponding MoP are selected via entropy distance. A perturbation is selected from the K groups, and the regional protection strength is adaptively adjusted using an LPIPS distance map to finally output the protected image.

### Key Designs

1. **Mixture-of-Perturbations (MoP)**:
    - **Function**: Provides semi-image-specific protection capabilities while maintaining zero inference optimization cost.
    - **Mechanism**: Pre-trains K=4 perturbations $\Delta=\{\delta_1,...,\delta_4\}$ and one global perturbation $\delta_g$. For each input image, the VAE encoder first extracts the latent $\mathbf{z}$, and the corresponding group perturbation $\Delta_k$ is selected via the pre-trained K-means++ assignment function $\mathcal{A}$. The protected image is $\hat{\mathbf{x}} = \mathbf{x} + \delta_g + \Delta_k$, where $k = \mathcal{A}(\mathcal{E}(\mathbf{x}))$. Both perturbations are constrained within an $(\eta/2)$-ball.
    - **Design Motivation**: The single perturbation of UAP cannot cover the diversity of images (simple vs. complex textures, natural photos vs. cartoons). MoP allows similar images to share perturbations through clustering—increasing the total capacity while retaining some image adaptability.

2. **Multi-Layer Protection (MLP) Loss**:
    - **Function**: Enhances the protection potency of perturbations during pre-training without adding to the inference cost.
    - **Mechanism**: Traditional texture loss only computes $\|\mathbf{z} - \mathbf{z}_y\|_2^2$ on the VAE latent $\mathbf{z}$. MLP Loss additionally leverages the intermediate layer features $\mathcal{F} = \{\mathbf{f}^1,...,\mathbf{f}^L\}$ of the VAE encoder: $\mathcal{L}_T = -\|\mathbf{z} - \mathbf{z}_y\|_2^2 - \frac{\lambda}{L}\sum_{l=1}^L \|\mathcal{F}^l - \mathcal{F}_y^l\|_2^2$.
    - **Design Motivation**: A loss solely in the z-space is not always sufficient to push $\mathbf{z}$ towards $\mathbf{z}_y$. Auxiliary losses from the multi-layer feature space impose constraints across multiple levels of abstraction, strengthening the destructive power of perturbations during subsequent fine-tuning.

3. **Adaptive Targeted Protection + Adaptive Protection Strength**:
    - **Function**: Adaptively selects the optimal target image and the perturbation strength of each region during inference.
    - **Mechanism**: Prepare three target images with low, medium, and high texture repetitions respectively (corresponding to three sets of MoP). During inference, the latent entropy distance $t = \arg\min_{i\in\{l,m,h\}} \|\mathcal{H}(\mathbf{z}) - \mathcal{H}(\mathbf{z}_y^i)\|_1$ is used to select the best matching target. Then, LPIPS is used to generate a perceptual distance map $\mathbf{M} = \text{LPIPS}(\mathbf{x}, \hat{\mathbf{x}})$, and the perturbation is adjusted as $\hat{\mathbf{x}} = \mathbf{x} + \mathcal{S}(1-\mathbf{M}) \cdot (\delta_g^t + \Delta_k^t)$—strengthening the perturbation in texture-complex regions (where human eyes are insensitive) and weakening it in flat regions.
    - **Design Motivation**: Simple texture images require low-repetition targets, while complex textures require high-repetition ones (as discovered in the experiments in Fig.3). LPIPS distance maps are more accurate and faster than the traditional JND maps used by Impasto.

### Loss & Training

- Training: MLP-enhanced texture loss $\mathcal{L}_T = -\|\mathbf{z} - \mathbf{z}_y\|_2^2 - \frac{\lambda}{L}\sum_{l=1}^L \|\mathcal{F}^l - \mathcal{F}_y^l\|_2^2$
- The Adam optimizer is used to update perturbation parameters, with an initial resolution of $512 \times 512$, and bilinear interpolation is used during inference for other resolutions.

## Key Experimental Results

### Main Results

**Comparison of $512^2$ Image Protection (LoRA Personalization Attack)**:

| Method | CPU Latency | GPU Latency | Object DISTS↓/FID↑ | Face DISTS↓/FID↑ | Cartoon DISTS↓/FID↑ |
|------|---------|---------|---------------------|-------------------|---------------------|
| PhotoGuard | 370s | 7s | 0.203/223.0 | 0.189/308.7 | 0.209/219.1 |
| Mist | 1440s | 40s | 0.185/217.2 | 0.154/307.5 | 0.223/223.7 |
| Anti-DB | 7278s | 225s | 0.239/214.4 | 0.162/301.4 | 0.294/225.4 |
| **FastProtect** | **2.9s** | **0.04s** | **0.155/223.0** | **0.149/308.9** | **0.186/220.3** |

FastProtect is 175× faster on GPU than the second fastest method, PhotoGuard, while achieving better visual quality (DISTS) in most domains.

### Ablation Study

**Contribution of Each Component in MoP (FID↑ in Object Domain)**:

| Configuration | FID↑ |
|------|------|
| PhotoGuard (iterative PGD) | 227.6 |
| UAP (baseline) | 207.6 |
| MoP (w/o assignment $\mathcal{A}$) | 214.5 |
| MoP (with $\mathcal{A}$) | 225.9 |
| + MLP Loss | 234.6 |
| + Adaptive Target | **238.8** |

MoP with the assignment function improves UAP from 207.6 to 225.9 (close to PGD's 227.6). MLP Loss and Adaptive Target further boost it to 238.8, surpassing PGD.

### Key Findings

- FastProtect maintains real-time performance on $2048^2$ images (GPU ~0.04s), whereas for other methods, the latency grows exponentially—which is crucial for protecting high-resolution artworks.
- The assignment function $\mathcal{A}$ of MoP is the core: average with assignment (225.9) vs. without assignment (214.5), the latter is even worse than a single UAP with assignment.
- There is a matching relationship between the texture repetition of the target image and the input image (Fig. 3)—simple textures are more effectively matched with low-repetition targets.
- In black-box scenarios (SD-v2.1, SD-XL, Textual Inversion, DreamStyler), the protection efficacy of FastProtect is comparable to or better than PhotoGuard.
- Inference VRAM requires only 1.7GB vs. >8GB for other methods—making it usable on consumer GPUs or even CPUs.

## Highlights & Insights

1. **First focus on the latency issue**: All previous protection methods ignored the most practical obstacle of "users cannot afford to wait". Dropping the inference time from minutes to milliseconds is a true paradigm shift.
2. The MoP design cleverly finds the **optimal balance** between being image-agnostic (UAP) and image-specific (PGD)—using VAE latent clustering for coarse-grained allocation, which incurs minimal cost but approaches the effect of image-by-image optimization.
3. The **"free lunch" effect of MLP Loss**: The multi-layer feature constraints only increase costs during training and are entirely free during inference, representing an efficient way of knowledge injection.
4. The design of Adaptive Protection Strength using the LPIPS perceptual map to regulate regional perturbation strength is **highly aligned with human visual perception**.

## Limitations & Future Work

- The number of perturbations K=4 is manually set; whether more perturbations can continue to improve performance remains to be studied.
- The resolution of pre-trained perturbations is fixed at $512^2$, and other resolutions via bilinear interpolation may introduce artifacts.
- It is only trained and primarily evaluated on Stable Diffusion v1.5; its transferability to generative models of the new generation (SD3, FLUX) is not verified.
- The robustness against countermeasures such as JPEG compression still needs improvement in some domains (e.g., cartoon) (Table 4).
- The clustering assignment of MoP is pre-computed at once and cannot adapt to out-of-distribution new image types.

## Related Work & Insights

- **Impasto** [Ahn et al.]: Proposes perception-guided invisible protection, but relies on traditional JND maps and still requires iterative optimization. This paper replaces it with LPIPS, which is faster and more accurate.
- **Mist** [Liang & Wu]: A protection method that unifies texture and semantic losses. The MLP Loss in this paper inherits the texture loss and extends it to multiple layers.
- **UAP** [Moosavi-Dezfooli et al.]: The concept of Universal Adversarial Perturbations. This paper transfers it from classification attacks and improves it into MoP for image protection.
- **PhotoGuard** [Salman et al.]: Previously the fastest protection method (GPU 7s). FastProtect is 175× faster.
- **Insights**: The paradigm of pre-trained perturbation + adaptive inference may be applicable to a broader range of adversarial attack and defense scenarios—any task requiring per-instance optimization could potentially benefit from a similar "offline pre-training + online selection" strategy.

## Rating

⭐⭐⭐⭐ — For the first time, the latency of image protection is brought down to a real-time level, solving the most critical practical obstacle in this field. The MoP design is simple yet effective, and the ablation study clearly verifies the contribution of each component. The industrial background from NAVER WEBTOON AI makes the research motivation highly grounded. The only minor drawback is the lack of generalization validation on next-generation diffusion models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit](../../NeurIPS2025/image_generation/blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)
- [\[CVPR 2025\] Enhancing Facial Privacy Protection via Weakening Diffusion Purification](enhancing_facial_privacy_protection_via_weakening_diffusion_purification.md)
- [\[CVPR 2025\] Personalized Preference Fine-tuning of Diffusion Models](personalized_preference_fine-tuning_of_diffusion_models.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](../../NeurIPS2025/image_generation/perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [\[CVPR 2025\] Implicit Bias Injection Attacks against Text-to-Image Diffusion Models](implicit_bias_injection_attacks_against_text-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
