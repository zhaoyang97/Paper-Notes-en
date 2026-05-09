---
title: >-
  [Paper Note] Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement
description: >-
  [Image Restoration] This paper proposes Latent Harmony, a two-stage framework that constructs a degradation-robust LH-VAE via latent space regularization, and subsequently applies high-frequency-guided LoRA fine-tuning to independently optimize the encoder (fidelity) and decoder (perceptual quality), achieving a unified solution to the generalization–reconstruction–perception trilemma in all-in-one UHD image restoration.
tags:
  - Image Restoration
date: 2026-05-08
content_hash: 92a60f61c31ae4be
---

# Latent Harmony: Synergistic Unified UHD Image Restoration via Latent Space Regularization and Controllable Refinement

## Basic Information

- **arXiv**: 2510.07961
- **Conference**: NeurIPS 2025
- **Authors**: Yidi Liu, Xueyang Fu, Jie Huang, Jie Xiao, Dong Li, Wenlong Zhang, Lei Bai, Zheng-Jun Zha
- **Affiliations**: University of Science and Technology of China, Shanghai AI Laboratory
- **Code**: https://github.com/lyd-2022/Latent-Harmony

## TL;DR

This paper proposes Latent Harmony, a two-stage framework that constructs a degradation-robust LH-VAE via latent space regularization, and subsequently applies high-frequency-guided LoRA fine-tuning to independently optimize the encoder (fidelity) and decoder (perceptual quality), achieving a unified solution to the generalization–reconstruction–perception trilemma in all-in-one UHD image restoration.

## Background & Motivation

### Problem Definition

Ultra-high-definition (UHD/4K) image restoration requires balancing computational efficiency with detail preservation. Existing VAE-based methods can transfer the restoration process into a low-dimensional latent space to improve efficiency; however, the Gaussian variational constraint imposed by standard VAEs, while preserving semantic information during compression, discards high-frequency information closely associated with degradation characteristics, thereby compromising reconstruction fidelity.

### Root Cause

Through systematic experimental analysis, the authors identify three core contradictions:

1. **Latent Space: Generalization vs. Reconstruction** — Enhancing the VAE's reconstruction capability renders the latent space more sensitive to degradations (lower Cross-Degradation Cosine Similarity). t-SNE visualizations reveal that the latent representations of a high-reconstruction-capacity VAE cluster by degradation type rather than by content. Frequency-domain analysis further demonstrates that such VAEs encode an excessively high proportion of high-frequency components in the latent space, while cross-degradation consistency is precisely lowest for high-frequency components, leading to degraded generalization.

2. **Joint VAE Optimization: Downstream Adaptation vs. Structural Preservation** — Directly back-propagating restoration losses to update VAE parameters causes rapid initial PSNR gains followed by severe oscillation, as the encoder is prematurely forced to perform direct degradation removal, collapsing the latent space into a simple bottleneck structure. Freezing the VAE, on the other hand, results in an insurmountable performance ceiling.

3. **High-Frequency Restoration: Perceptual Quality vs. Fidelity** — Pixel-level losses reduce systematic error (SE) to achieve high PSNR but suppress the variance effect (VE), yielding flat textures; generative perceptual losses improve visual naturalness but may introduce hallucinations. These two objectives fundamentally pursue conflicting optimization directions.

### Key Findings

The authors identify high-frequency information as the key linking factor across all three contradictions: (1) an excessive proportion of high-frequency components in the latent space impairs generalization; (2) back-propagating a high-frequency alignment loss (rather than a full-frequency restoration loss) to update the VAE stabilizes training while breaking through the performance ceiling; and (3) independently fine-tuning the encoder with a fidelity-oriented high-frequency loss and the decoder with a perceptual high-frequency loss yields independent improvements on the respective metrics.

## Method

### Overall Architecture

Latent Harmony adopts a two-stage training strategy:

- **Stage 1**: Train LH-VAE to construct a degradation-robust universal latent space via latent space regularization.
- **Stage 2**: Jointly train a restoration network on top of LH-VAE with HF-LoRA fine-tuning to achieve high-frequency detail compensation and controllable output.

### Stage 1: Building a Generalizable Latent Space (LH-VAE)

Building upon the standard VAE's L1 reconstruction loss and KL divergence regularization, three additional regularization mechanisms are introduced:

**1. Progressive Degradation Perturbation Strategy (PDPS)**

During training, clean images are subjected to progressively increasing degradation perturbations over time. Perturbations are sampled according to probabilities $p_0, p_1, p_2$: no perturbation; synthetic degradations (Gaussian noise, blur, JPEG compression, etc., with intensity increasing over training); or interpolation with the paired degraded image (with mixing coefficient $\beta(t)$ increasing over time). The progressive strategy ensures the encoder gradually learns to resist degradations, avoiding training instability.

**2. Degradation-Invariant Visual Semantic Loss $L_{Inv}$**

Pre-trained DINOv2 is used to extract semantic features of clean images as anchor points, constraining the encoder to align its representations of perturbed images toward these semantic references. This encourages latent representations to be organized by content rather than by degradation type.

**3. Latent Space Equivariance Loss $L_{Eqv}$**

Latent codes are randomly downsampled before decoding, and the decoded output is required to be consistent with the corresponding downsampled image. This constraint enhances scale robustness, reduces the decoder's over-reliance on high-frequency components, and promotes a more balanced frequency distribution in the learned features.

Stage 1 total loss: $L_{Stage1} = L_{VAE} + \lambda_{Inv} L_{Inv} + \lambda_{Eqv} L_{Eqv}$

### Stage 2: High-Frequency-Guided Controllable LoRA Fine-Tuning

Stage 2 proceeds in three steps:

**Step 1: Training the Latent Space Restoration Network**

The LH-VAE is frozen. A restoration network $R_\theta$ is trained to take degraded latent codes $z_{deg}$ as input and produce restored latent codes $z_{res}$, supervised by an L1 restoration loss. SFHformer is adopted as the restoration backbone.

**Step 2: Fidelity-Oriented Encoder HF-LoRA (FHF-LoRA)**

LoRA parameters $\Delta\phi_{LoRA}$ are injected into the encoder, while the decoder's base parameters $\psi^*$ remain frozen. The high-frequency fidelity loss is:

$$L_{HF_{Fid}} = \|HF(D_{\psi^*}(E_{\phi^*+\Delta\phi_{LoRA}}(I_{deg}))) - HF(I_{clean})\|_1$$

This loss guides the encoder LoRA to accurately extract genuine high-frequency structures from degraded inputs, corresponding to the optimization of the SE term in Equation (1).

**Step 3: Perceptual-Oriented Decoder HF-LoRA (PHF-LoRA)**

LoRA parameters $\Delta\psi_{LoRA}$ are injected into the decoder, while the encoder's base parameters $\phi^*$ remain frozen. The high-frequency adversarial loss is:

$$L_{HF_{GAN}} = -\mathbb{E}_{I_{deg}}[\log D_{HF}(HF(D_{\psi^*+\Delta\psi_{LoRA}}(R_\theta(E_{\phi^*}(I_{deg})))))]$$

A high-frequency discriminator $D_{HF}$ drives the decoder LoRA to generate visually natural high-frequency textures, corresponding to the preservation and shaping of the VE term in Equation (1).

The two LoRA modules are optimized in an alternating fashion, with gradients flowing only through the corresponding LoRA parameters without perturbing the pre-trained weights, thereby preserving the structural integrity of the latent space.

### Inference-Time Control

At inference, a scalar $\alpha \in [0,1]$ flexibly controls the trade-off between fidelity and perceptual quality:

$$\phi = \phi^* + \alpha \cdot \Delta\phi_{LoRA}, \quad \psi = \psi^* + (1-\alpha) \cdot \Delta\psi_{LoRA}$$

Larger $\alpha$ favors fidelity (higher PSNR), while smaller $\alpha$ favors perceptual quality (lower LPIPS).

## Key Experimental Results

### Table 1: Comparison on All-in-One UHD Restoration Across Four Degradations

| Method | Full-Res Inference | FLOPs | Params | Avg. PSNR↑ | Avg. SSIM↑ | Avg. LPIPS↓ |
|------|:---:|------:|-------:|-----:|-----:|------:|
| PromptIR | ✗ | 158G | 33M | 24.68 | 0.803 | 0.2571 |
| HAIR | ✗ | 41G | 29M | 27.36 | 0.847 | 0.2822 |
| UHDprocesser | ✓ | 4G | 1.6M | 29.23 | 0.868 | 0.2541 |
| **Latent Harmony** | **✓** | **3.6G** | **1.2M** | **29.70** | **0.877** | **0.2502** |

Latent Harmony achieves the highest performance with the fewest parameters (1.2M) and the lowest computational cost (3.6G FLOPs), improving PSNR by 0.47 dB over UHDprocesser while supporting full-resolution 4K inference.

### Table 2: Ablation Study

| Configuration | PSNR↑ | SSIM↑ | LPIPS↓ |
|------|------:|------:|-------:|
| Latent Harmony (full) | 29.77 | 0.88 | 0.250 |
| w/o $L_{Inv}$ | 24.28 | 0.79 | 0.292 |
| w/o $L_{Eqv}$ | 25.68 | 0.82 | 0.302 |
| w/o PDPS | 27.82 | 0.84 | 0.287 |
| w/o FHF-LoRA | 28.12 | 0.86 | 0.286 |
| w/o PHF-LoRA | 29.02 | 0.84 | 0.306 |
| w/o LoRA fine-tuning | 28.68 | 0.85 | 0.298 |

Removing $L_{Inv}$ has the largest impact (5.49 dB PSNR drop), demonstrating that semantic alignment is critical for generalization. LoRA fine-tuning contributes approximately 1 dB improvement in total, with FHF-LoRA primarily benefiting PSNR/SSIM and PHF-LoRA primarily benefiting LPIPS.

## Highlights & Insights

1. **Systematic Motivation Analysis**: Rather than simply stacking modules, the paper employs t-SNE, CDCS, and frequency-domain analysis to systematically identify three core contradictions inherent to applying VAEs in UHD restoration, providing clear empirical justification for every design choice.

2. **High-Frequency Loss as a "Bridge" for Joint Optimization**: The finding that using a high-frequency alignment loss (rather than a full-frequency restoration loss) to back-propagate updates into the VAE maintains training stability is a valuable empirical discovery with potential implications for other VAE joint-training scenarios.

3. **Encoder–Decoder Divide-and-Conquer LoRA**: Decoupling fidelity and perceptual quality into independent LoRA modules for the encoder and decoder, optimized in alternation to avoid gradient conflicts, while providing an inference-time controllable parameter $\alpha$—this design is both engineering-pragmatic and theoretically elegant.

4. **Extreme Efficiency**: With 1.2M parameters, 3.6G FLOPs, and 0.43-second inference time (compared to 12.3 seconds for DreamUIR), the method enables full-resolution 4K inference on consumer-grade GPUs.

5. **Plug-and-Play Validation**: LH-VAE can replace the VAE component in PromptIR, Diff-Plugin, and CosAE with consistent performance gains, validating the generality of the proposed framework.

## Limitations & Future Work

1. **High Training Complexity**: The two-stage pipeline—Stage 1 for LH-VAE followed by Stage 2 for the restoration network and two LoRA modules—entails a lengthy overall training process and significant engineering complexity.

2. **Manual Selection of $\alpha$**: The optimal value of $\alpha$ at inference depends on the specific application scenario; no adaptive selection mechanism is provided.

3. **Unclear Generalization Boundaries**: Although generalization to unseen and composite degradations is validated, robustness to out-of-distribution degradations (e.g., sensor-specific noise) is not thoroughly discussed.

4. **Dependency on DINOv2**: Stage 1 relies on pre-trained DINOv2 to provide semantic anchor points, introducing a dependency on an external model.

## Related Work & Insights

- **UHDprocesser (CVPR 2025)**: A prior work by the same authors and the direct baseline for improvement, employing a degradation-aware prompt branch.
- **DreamUHD (AAAI 2025)**: A prior work by the same authors, featuring frequency-enhanced VAE and high-frequency injection.
- **REPA-E**: Proposes representation alignment for end-to-end joint training of VAE and LDM, inspiring the use of alignment losses for joint optimization in this work.
- **SE/VE Decomposition Framework**: Drawn from perception–fidelity trade-off theory, providing the theoretical foundation for the divide-and-conquer design of FHF-LoRA and PHF-LoRA.

**Implications for Future Work**: The idea of using high-frequency information as a bridge loss for VAE joint optimization, as well as the paradigm of assigning fidelity and perceptual quality to the encoder and decoder respectively, can be generalized to downstream tasks requiring VAE compression, such as video restoration and 3D reconstruction.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: ⭐⭐⭐⭐ — Systematic solution to three core contradictions; the HF-LoRA divide-and-conquer design is novel, though the core contributions lean more toward engineering integration.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Rigorous motivation experiments, comprehensive ablations, multi-setting validation, and thorough plug-and-play generalization verification.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure; the motivation–method–validation three-part argumentation is convincing.
- **Value**: ⭐⭐⭐⭐ — Establishes a strong baseline for UHD restoration; the plug-and-play nature of LH-VAE is likely to see wide adoption.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Implicit Augmentation from Distributional Symmetry in Turbulence Super-Resolution](implicit_augmentation_from_distributional_symmetry_in_turbulence_super-resolutio.md)
- [\[NeurIPS 2025\] Audio Super-Resolution with Latent Bridge Models](audio_super-resolution_with_latent_bridge_models.md)
- [\[NeurIPS 2025\] Rethinking Nighttime Image Deraining via Learnable Color Space Transformation](rethinking_nighttime_image_deraining_via_learnable_color_space_transformation.md)
- [\[NeurIPS 2025\] MoDEM: A Morton-Order Degradation Estimation Mechanism for Adverse Weather Image Restoration](modem_a_morton-order_degradation_estimation_mechanism_for_adverse_weather_image_.md)
- [\[NeurIPS 2025\] MS-BART: Unified Modeling of Mass Spectra and Molecules for Structure Elucidation](ms-bart_unified_modeling_of_mass_spectra_and_molecules_for_structure_elucidation.md)

<!-- RELATED:END -->
