---
title: >-
  [Paper Note] DRAG: Data Reconstruction Attack using Guided Diffusion
description: >-
  [ICML 2025][Image Generation][Data Reconstruction Attack] This paper proposes DRAG, which leverages the image prior knowledge of pre-trained Latent Diffusion Models (LDMs) to reconstruct the original input images with high fidelity from deep Intermediate Representations (IRs) in Split Inference (SI) via a guided diffusion process, revealing severe privacy vulnerabilities of vision foundation models (such as CLIP and DINOv2) under SI scenarios.
tags:
  - "ICML 2025"
  - "Image Generation"
  - "Data Reconstruction Attack"
  - "Guided Diffusion"
  - "Split Inference Privacy"
  - "Vision Foundation Models"
  - "Latent Diffusion Models"
date: 2026-05-08
content_hash: a5d358fa337833fd
---

# DRAG: Data Reconstruction Attack using Guided Diffusion

**Conference**: ICML 2025  
**arXiv**: [2509.11724](https://arxiv.org/abs/2509.11724)  
**Code**: [ntuaislab/DRAG](https://github.com/ntuaislab/DRAG)  
**Area**: Image Generation  
**Keywords**: Data Reconstruction Attack, Guided Diffusion, Split Inference Privacy, Vision Foundation Models, Latent Diffusion Models

## TL;DR

This paper proposes DRAG, which leverages the image prior knowledge of pre-trained Latent Diffusion Models (LDMs) to reconstruct the original input images with high fidelity from deep Intermediate Representations (IRs) in Split Inference (SI) via a guided diffusion process, revealing severe privacy vulnerabilities of vision foundation models (such as CLIP and DINOv2) under SI scenarios.

## Background & Motivation

**Split Inference (SI)** is an inference paradigm that splits a neural network into a client model $f_c$ and a server model $f_s$. The client processes raw data $\mathbf{x}^*$ on an edge device to obtain an intermediate representation (IR) $\mathbf{h}^* = f_c(\mathbf{x}^*)$, which is then sent to the cloud to complete the computation. SI is widely believed to balance privacy and computational efficiency.

**Limitations of Prior Work**:
1. Existing Data Reconstruction Attacks (DRAs) mainly target small CNN classification models (e.g., ResNet18), leaving the privacy risks of large vision foundation models understudied.
2. The patch tokenization and attention mechanisms of Vision Transformers (ViTs) are fundamentally different from those of CNNs, and the effectiveness of attacks on them has not been fully explored.
3. ViTs exhibit **token order invariance**, a property not present in CNNs, which significantly impacts the effectiveness of reconstruction attacks.
4. Reconstructing raw data from deep IRs is much more difficult due to the highly abstract nature of deep representations.

**Key Insight**: Pre-trained LDMs (such as Stable Diffusion) have learned rich image priors on large-scale datasets, which can serve as strong regularization constraints to restrict reconstruction within the natural image manifold.

## Method

### Overall Architecture

The core idea of DRAG is to use a diffusion model as an image prior $R_\mathcal{I}$ to constrain the solution space of the optimization problem. Specifically, the attacker operates under a white-box threat model: the architecture and parameters of the client model $f_c$ are known (which is a reasonable assumption in the era of foundation models, as networks like CLIP and DINOv2 are publicly released frozen models).

**Two Attack Paradigms**:

1. **Optimization-based Methods**: Directly minimize the distance between the reconstructed image and the target IR:
$$\mathbf{x}' = \arg\min_{\mathbf{x} \in \mathcal{X}} d_\mathcal{H}(f_c(\mathbf{x}), \mathbf{h}^*) + \lambda R_\mathcal{I}(\mathbf{x})$$

2. **Learning-based Methods**: Train an inverse network $f_c^{-1}: \mathcal{H} \to \mathcal{X}$ to learn the mapping from public datasets.

DRAG adopts the optimization-based route, replacing traditional gradient descent optimization with the iterative sampling process of the diffusion model.

### Key Designs

**Design 1: Guided Diffusion Sampling**

Based on the DDIM sampling framework, unconditional sampling is transformed into conditional sampling. The core is to define the reconstruction objective function:

$$L(\hat{\mathbf{x}}_0, \mathbf{c}) = d_\mathcal{H}(f_c(\hat{\mathbf{x}}_0), \mathbf{h}^*)$$

where $\hat{\mathbf{x}}_0$ is the clean image estimated in a single step from the current noisy timestep via Tweedie's formula:

$$\hat{\mathbf{x}}_0 = \frac{\mathbf{x}_t - \sqrt{1-\alpha_t}\,\epsilon_\theta(\mathbf{x}_t)}{\sqrt{\alpha_t}}$$

Note that the guidance cannot be directly computed using the noisy image $\mathbf{x}_t$, because $f_c$ is trained only on clean images, and noisy inputs would yield unreliable gradients.

**Design 2: Diffusion with Spherical Gaussian constraint (DSG)**

DSG (Diffusion with Spherical Gaussian constraint) is used to fuse the guidance gradient $\mathbf{g}_t$ with the noise $\epsilon_t$:

$$\epsilon_t \leftarrow r \cdot \text{Unit}((1-w)\sigma_t\epsilon_t + wr \cdot \text{Unit}(\mathbf{g}_t))$$

where $r = \sqrt{n}\sigma_t$, and $n$ is the dimension of $\mathbf{x}_t$. This design reduces the required number of denoising steps and improves generation quality.

**Design 3: Self-Recurrence**

As $f_c$ is typically non-convex, a single guidance step is insufficient for high-quality reconstruction. DRAG adopts a self-recurrence strategy that performs $k$ denoising-noising loops at each timestep:

$$\mathbf{x}_t = \sqrt{\alpha_t/\alpha_{t-1}} \cdot \mathbf{x}_{t-1} + \sqrt{1 - \alpha_t/\alpha_{t-1}} \cdot \epsilon$$

This allows the model to receive multiple gradient guidance steps at each timestep, significantly boosting reconstruction accuracy.

**Design 4: Gradient Optimization Techniques**

- Gradient clipping to prevent gradient explosion.
- Using the Adam optimizer to maintain historical guidance vectors and improve convergence stability.

### DRAG++ Variant

DRAG++ combines the strengths of learning-based and optimization-based methods:
1. First, use an inverse network $f_c^{-1}$ to obtain a coarse estimate $\mathbf{x}_{\text{coarse}} = f_c^{-1}(\mathbf{h}^*)$ from the IR.
2. Then, refine $\mathbf{x}_{\text{coarse}}$ through the diffusion-denoising process.

This two-stage strategy provides a better initialization for diffusion sampling, accelerating convergence and improving reconstruction quality.

### Loss & Training

**Distance Metric $d_\mathcal{H}$**: Measures the distance between the IR of the reconstructed image after passing through $f_c$ and the target $\mathbf{h}^*$ (such as MSE or cosine distance).

**Regularization $R_\mathcal{I}$**: Implicitly provided by the diffusion model—the LDM's denoising process naturally constrains reconstruction within the domain of natural image distribution, eliminating the need for extra regularization terms (such as Total Variation or Deep Image Prior).

**Three Key Differences from Existing Methods**:
1. $f_c$ is typically non-convex, requiring stronger optimization strategies.
2. Defense mechanisms may exist, requiring attackers to work under adversarial settings.
3. The client can embed randomness in $f_c$ (e.g., token shuffling), further increasing the difficulty.

## Key Experimental Results

### Main Results

Evaluated on CLIP and DINOv2 vision foundation models, reconstructing natural images from deep IRs:

| Method | Target Model | Image Quality | Perceptual Similarity | Advantage |
|------|---------|---------|-----------|------|
| rMLE (He et al.) | ResNet/ViT | Low | Poor | Only TV regularization |
| LM (Singh et al.) | ResNet/ViT | Medium | Fair | Incorporates Deep Image Prior |
| GLASS (Li et al.) | ResNet/ViT | High | Good | StyleGAN2 constraints |
| **DRAG (Ours)** | **CLIP/DINOv2** | **Significantly Higher** | **Best** | **LDM image prior** |
| **DRAG++ (Ours)** | **CLIP/DINOv2** | **Highest** | **Best** | **Inverse network + LDM** |

### Ablation Study

| Configuration | Key Effect | Description |
|------|---------|------|
| Without DSG constraint | Quality degradation | Spherical Gaussian constraint is key to improving quality |
| Without self-recurrence ($k=1$) | Blurry reconstruction | Multiple denoising-noising loops are crucial for non-convex optimization |
| Without gradient clipping | Unstable training | Gradient explosion causes generation collapse |
| Without Adam history guidance | Slow convergence | Historical gradient information speeds up optimization |
| DRAG vs DRAG++ | DRAG++ is superior | Inverse network provides better initialization |
| Shallow IR vs Deep IR | Shallow is easier | Deep IR information is more abstract, making reconstruction harder |

### Key Findings

1. **ViT's Token Order Invariance**: The order of tokens in ViTs does not affect the output, meaning that reconstruction from IR requires extra handling of token correspondences—a unique challenge not present in CNN models.
2. **Limited Defense Effectiveness**: Existing defense methods (such as NoPeek and DisP) remain highly vulnerable to DRAG, as input data can still be successfully reconstructed.
3. **Privacy Risks of Foundation Models**: Widely used public vision encoders like CLIP and DINOv2 exhibit severe privacy leakage risks under SI scenarios.
4. **Powerful Effect of LDM Priors**: The image priors provided by pre-trained LDMs are far superior to traditional TV regularization or GAN constraints, especially when reconstructing from deep IRs.

## Highlights & Insights

1. **Clever Dual-Role Utilization of Diffusion Models**: LDMs provide strong image priors (regularization) while simultaneously driving the optimization via iterative denoising—elegantly and effectively transforming traditional gradient optimization into guided diffusion sampling.
2. **Ingenious Use of Tweedie's Formula**: Estimating the clean image $\hat{\mathbf{x}}_0$ through single-step denoising to calculate the guidance gradient elegantly solves the problem where $f_c$ cannot process noisy inputs.
3. **DRAG++ Hybrid Strategy**: Combining the fast coarse estimation from an inverse network with the fine-grained optimization from a diffusion model achieves a balance between efficiency and quality.
4. **Wake-up Call to the Security Community**: Revealing the urgency of privacy protection for SI in the era of foundation models—public model weights make white-box attacks a practical threat.

## Limitations & Future Work

1. **High Computational Overhead**: The iterative sampling of guided diffusion coupled with self-recurrence loops makes the reconstruction speed relatively slow, hindering real-time attacks.
2. **Dependency on Pre-trained LDMs**: When the target image distribution differs heavily from the LDM's training set (e.g., medical imaging, remote sensing images), the attack effectiveness may drop.
3. **Limitations of the White-Box Assumption**: Although the public availability of foundation models makes the white-box assumption reasonable, clients may perform fine-tuning or adaptation, making the model parameters not fully known.
4. **Insufficient Exploration of Defenses**: The paper primarily demonstrates attack capabilities, with relatively shallow discussion on how to defend effectively (such as via differential privacy or adversarial training).
5. **Scalability**: Whether this can scale to higher-resolution images, video data, or 3D data modalities remains to be validated.

## Related Work & Insights

- **GLASS (Li et al., 2023)**: DRA using StyleGAN2 as an image prior, but GAN priors are less diverse than LDM priors.
- **UGD (Bansal et al., 2024)**: A universal guided diffusion framework, which DRAG adapts to the DRA scenario.
- **DSG (Yang et al., 2024)**: Spherical Gaussian constraints to accelerate guided diffusion, which is directly adopted by DRAG.
- **DPS (Chung et al., 2023)**: Diffusion Posterior Sampling, whose idea of using Tweedie's formula to estimate clean images is adopted by DRAG.
- **Insight**: This attack framework has the potential to be extended to gradient reconstruction attacks in federated learning (replacing IRs with gradient information).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying guided diffusion to DRA is a natural yet highly effective combination, and the DRAG++ hybrid strategy is quite creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers multiple foundation models and defense methods with relatively comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, complete mathematical derivations, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐⭐ — Serves as an important warning to the AI security community, exposing the privacy risks of SI in foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Diffusion Reconstruction-Based Data Likelihood Estimation for Core-Set Selection](../../AAAI2026/image_generation/diffusion_reconstruction-based_data_likelihood_estimation_for_core-set_selection.md)
- [\[ICCV 2025\] ScoreHOI: Physically Plausible Reconstruction of Human-Object Interaction via Score-Guided Diffusion](../../ICCV2025/image_generation/scorehoi_physically_plausible_reconstruction_of_human-object_interaction_via_sco.md)
- [\[CVPR 2025\] UIBDiffusion: Universal Imperceptible Backdoor Attack for Diffusion Models](../../CVPR2025/image_generation/uibdiffusion_universal_imperceptible_backdoor_attack_for_diffusion_models.md)
- [\[NeurIPS 2025\] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data](../../NeurIPS2025/image_generation/geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)
- [\[ICML 2025\] Origin Identification for Text-Guided Image-to-Image Diffusion Models](origin_identification_for_text-guided_image-to-image_diffusion_models.md)

</div>

<!-- RELATED:END -->
