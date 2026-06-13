---
title: >-
  [Paper Note] Sim-to-Real: An Unsupervised Noise Layer for Screen-Camera Watermarking Robustness
description: >-
  [AAAI 2026][Robotics][Screen-camera watermarking] This paper proposes the Simulation-to-Real (S2R) framework, which introduces a novel two-stage noise approximation strategy of "mathematical modeling → unsupervised domai…
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Screen-camera watermarking"
  - "noise approximation"
  - "unsupervised learning"
  - "domain transfer"
  - "GAN"
  - "robust watermarking"
date: 2026-05-08
content_hash: b2d21a43da480b5d
---

# Sim-to-Real: An Unsupervised Noise Layer for Screen-Camera Watermarking Robustness

**Conference**: AAAI 2026
**arXiv**: [2504.18906](https://arxiv.org/abs/2504.18906)  
**Code**: [GitHub](https://github.com/ttz0523/S2R-main)  
**Area**: AI Security / Digital Watermarking
**Keywords**: Screen-camera watermarking, noise approximation, unsupervised learning, domain transfer, GAN, robust watermarking

## TL;DR

This paper proposes the Simulation-to-Real (S2R) framework, which introduces a novel two-stage noise approximation strategy of "mathematical modeling → unsupervised domain transfer": a mathematical transform $T$ first maps clean images to a known noise domain $\mathcal{C}$, and an unsupervised image-to-image network $G$ then maps $\mathcal{C}$ to the real screen-camera (SC) noise domain $\mathcal{U}$. Without requiring paired data, S2R accurately approximates real SC noise and achieves state-of-the-art watermarking robustness (BER reduced by 30–60%) and image quality (PSNR 42.27 dB / SSIM 0.962) across multiple devices, angles, and distances.

## Background & Motivation

- **Background**: Screen-camera (SC) capture is one of the primary means of unauthorized content acquisition, and robust watermarking is a core tool for post-hoc copyright tracing. The key to existing SC watermarking methods lies in the noise layer design during training—the noise layer simulates SC degradations so that the watermarking network learns to resist them through adversarial training.
- **Limitations of Prior Work**: Two existing noise approximation strategies both suffer from fundamental flaws:
  1. **Mathematical modeling** (StegaStamp, PIMoG, SSDS): SC noise is decomposed into independent components—perspective transformation, blur, illumination, Moiré patterns, Gaussian noise—and linearly superimposed. This assumes independence among noise components, ignoring their coupling in real scenes, and struggles to model fine-grained, localized distortions.
  2. **Supervised neural network fitting** (CDTF): Paired data is used to train a network to directly learn the mapping from clean to SC images. Obtaining high-quality paired data is extremely difficult (requiring manual alignment prone to spatial misregistration), and the network's limited capacity hinders coverage of the full diversity of SC noise.
- **Key Challenge**: Mathematical modeling offers controllable priors but large approximation bias; neural network fitting achieves high fidelity but depends on paired data and generalizes poorly. Neither can fundamentally achieve effective approximation of real SC noise.
- **Key Insight**: Rather than directly learning the intractable mapping $\mathcal{S} \to \mathcal{U}$ (clean → real noise), the problem is decomposed as $\mathcal{S} \xrightarrow{T} \mathcal{C} \xrightarrow{G} \mathcal{U}$—first obtaining a coarse noise approximation via existing mathematical models, then using unsupervised methods to bridge the remaining distributional gap. Learning the difference between noise domains is far simpler than learning the noise mapping from scratch.

## Method

### Overall Architecture

The core formula of S2R is $F_{\mathcal{U}}(\cdot) = T * G$, where the noise approximation function is composed of two components:

1. **Mathematical modeling transform $T$**: Maps a clean image $x^s$ to a known-noise-domain image $y^c = T(x^s)$. By default, the paper adopts PIMoG's noise model (perspective transformation + illumination variation + Moiré patterns + Gaussian noise).
2. **Unsupervised domain transfer network $G$**: Maps $y^c$ to the real SC noise domain, yielding $y^u = G(y^c)$ as the final noise-approximated image.

During training, given a clean image set $\mathcal{S}$ and an unpaired real SC image set $\mathcal{U}$, simulated noisy images $y^c$ are first generated via $T$, and $G$ is then trained to align the distribution of $y^c$ with that of $\mathcal{U}$. During inference, $G$'s weights are fixed, and clean images are passed sequentially through $T$ and $G$.

### Key Designs

1. **Unsupervised Noise Domain Transfer (Core Contribution)**

    - **Function**: Learns a mapping from the known noise domain $\mathcal{C}$ to the unknown noise domain $\mathcal{U}$ using unpaired data.
    - **Mechanism**: No paired correspondence between $y^c$ and $y^u$ is required—only two sets of images drawn from their respective distributions. $G$ learns a distribution-level transformation rather than an image-level correspondence.
    - **Design Motivation**: Collecting real SC images is straightforward (any screen photograph suffices), whereas precisely pairing them with clean originals is extremely difficult. The unsupervised approach entirely bypasses the paired-data bottleneck.
    - **Theoretical Support**: The authors derive via a noise decomposition formula that $y^u = k^{(c \to u)} \cdot y^c + n^{(c \to u)}$ (when $n^s = 0$), i.e., real noisy images can be expressed as multiplicative and additive transformations of simulated noisy images. This reduces the problem from learning the full $\mathcal{S} \to \mathcal{U}$ mapping to learning residual biases $k_\delta$ and $n_\delta$.

2. **Image-to-Image Network Architecture**

    - **Function**: An improved MIMO-UNet serves as generator $G$.
    - **Mechanism**: Multi-Input Single Encoder (MISE) combined with Asymmetric Feature Fusion (AFF) enables multi-scale feature extraction and fusion. The encoder receives downsampled noisy images at different scales along with Gaussian noise as inputs; the decoder outputs multi-scale deblurred/noise-transformed images.
    - **Design Motivation**: Multi-scale processing simultaneously captures global noise characteristics (illumination shifts, color casts) and local fine-grained noise (Moiré patterns, pixel-level distortions). Injecting random Gaussian noise $z$ mitigates mode collapse and promotes diverse noise generation.

3. **Modular and Replaceable Design**

    - **Function**: The mathematical modeling module $T$ and the domain transfer network $G$ are decoupled and independently replaceable.
    - **Experimental Validation**: Replacing PIMoG with the noise models of StegaStamp and SSDS as $T$ still allows the S2R framework to function and improve performance.
    - **Design Motivation**: Different application scenarios may involve different SC noise characteristics; flexible replacement of $T$ accommodates diverse requirements. CycleGAN and DualGAN can likewise be substituted for $G$.

### Loss & Training

**Generator Loss**: $L_G = L_{\text{cGAN}}(G, D) + \lambda_G L_P(G)$

- **Adversarial Loss** $L_{\text{cGAN}}$: Standard GAN loss; discriminator $D$ distinguishes real SC images from generated ones, while $G$ attempts to fool $D$.
- **Multi-scale Perceptual Loss** $L_P$: Reconstruction error computed in the feature space of a pretrained VGG network, with scale-decreasing weights $\frac{1}{2^{k-1}}$ for coarse-to-fine content reconstruction. Avoids over-smoothing caused by pixel-level constraints.

**Discriminator Loss**: $L_D = -L_{\text{cGAN}}(G, D) + \lambda_{\text{grad}} L_{\text{grad}}^D(D)$

- **Gradient Penalty** $L_{\text{grad}}^D$: Imposes a gradient norm constraint on interpolated samples to enforce Lipschitz continuity and stabilize GAN training.

**Hyperparameters**: $\lambda_G = 1.0$, $\lambda_{\text{grad}} = 0.005$ (following Blur2Blur).

**Training Details**:
- Watermarking framework: MCFN; 10,000 images from COCO resized to 128×128 with 64-bit random watermarks embedded.
- S2R training data: 900 SC images per device pair across 3 device combinations (Samsung+Lenovo / iPhone+Envision / MEIZU+ASUS), merged into the SIM+LEA dataset.
- Hardware: NVIDIA RTX 4090 GPU, batch size = 8.

## Key Experimental Results

### Performance Comparison of Different Noise Layers Under the Same Watermarking Framework (Distance: 30 cm)

| Method | PSNR (dB) | SSIM | BER 0° | BER 20° | BER 40° |
|--------|-----------|------|--------|---------|---------|
| StegaStamp | 39.89 | 0.948 | 5.5% | 7.1% | 7.3% |
| PIMoG | 41.41 | 0.950 | 6.2% | 8.8% | 9.5% |
| SSDS | 41.05 | 0.956 | 5.1% | 6.0% | 7.6% |
| **S2R** | **42.27** | **0.962** | **2.1%** | **3.3%** | **6.0%** |

S2R achieves comprehensive superiority in both image quality and watermarking robustness: PSNR improves by 1–2.4 dB, and 0° BER decreases by 59% relative to SSDS.

### BER Comparison Across Capture Distances and Angles (%)

| Method | 20cm | 25cm | 30cm | 35cm | 40cm | L-60° | L-40° | L-20° | R20° | R40° | R60° |
|--------|------|------|------|------|------|-------|-------|-------|------|------|------|
| StegaStamp | 2.9 | 3.9 | 4.6 | 4.7 | 4.4 | 5.9 | 7.2 | 4.1 | 5.8 | 7.7 | 7.6 |
| PIMoG | 1.5 | 1.4 | 3.3 | 3.2 | 2.6 | 9.0 | 8.7 | 5.2 | 5.3 | 9.3 | 9.7 |
| SSDS | 2.4 | 2.7 | 2.1 | 2.7 | 4.1 | 7.5 | 5.1 | 3.9 | 4.2 | 6.1 | 6.2 |
| **S2R** | **1.2** | **1.1** | **2.1** | **2.5** | **2.2** | **5.8** | **3.9** | **3.2** | **3.3** | **6.0** | **5.9** |

S2R is particularly advantageous at close range (20–25 cm), achieving BER as low as 1.1–1.2%, and maintains leading performance at large angles.

### Ablation Study

| Variant | PSNR (dB) | SSIM | BER 0° | BER 20° | BER 40° |
|---------|-----------|------|--------|---------|---------|
| StegaStamp-based (SIM+LEA) | 40.47 | 0.952 | 2.4% | 3.7% | 7.1% |
| SSDS-based (SIM+LEA) | 41.25 | 0.967 | 5.0% | 8.1% | 10.6% |
| S2R-supervised (I+E) | 41.29 | 0.959 | 3.8% | 5.5% | 7.9% |
| S2R-CycleGAN (SIM+LEA) | 41.85 | 0.960 | 2.9% | 4.5% | 6.9% |
| S2R-DualGAN (SIM+LEA) | 41.55 | 0.958 | 3.5% | 5.2% | 7.6% |
| S2R (I+E) | 42.57 | 0.964 | 1.6% | 3.1% | 5.1% |
| **S2R (SIM+LEA)** | **42.27** | **0.962** | **2.1%** | **3.3%** | **6.0%** |

Key findings: (1) The S2R framework can plug-and-play with different mathematical models $T$, consistently outperforming the original methods; (2) unsupervised S2R significantly outperforms the supervised variant, validating the unpaired-data strategy; (3) S2R's image-to-image network outperforms CycleGAN and DualGAN.

### Robustness Under Extreme Conditions (Limitations Analysis)

| Capture Condition | BER (%) |
|-------------------|---------|
| Standard (0°, 30 cm) | 1.6 |
| Extreme angle +80° | 30.0 |
| Extreme angle −80° | 26.0 |
| Long distance 100 cm | 3.6 |
| Local glare | 2.5 |
| Dark screen | 30.0 |
| Partial crop (center 75% retained) | 50.0 |

## Highlights & Insights

1. **Elegant Problem Decomposition**: The near-intractable direct mapping from "clean → real noise" is decomposed into the two-step pipeline "clean → simulated noise → real noise." This leverages mathematical modeling priors to shrink the search space from the entire noise space to a residual noise space, substantially reducing learning difficulty. The sim-to-real paradigm is well-established in robotics and autonomous driving, but its application to watermarking is a first.

2. **Rigorous Theoretical Feasibility Proof**: Through the noise decomposition formula $y^u = k_\delta \cdot y^c + n_\delta$, the paper theoretically establishes the feasibility of inter-domain transfer—when the input image is noise-free, real noisy images can be fully expressed as mappings of simulated noisy images. This provides not merely empirical validity but a theoretical foundation.

3. **Practical Value of the Unsupervised Approach**: In the watermarking domain, acquiring paired SC data is prohibitively expensive due to the need for precise alignment. S2R requires only "a few hundred arbitrarily captured screen photographs" to train the noise model, greatly lowering the barrier to practical deployment. Cross-device generalization experiments further demonstrate universality across different devices.

4. **Quantitative Link Between Noise Fidelity and Watermarking Performance**: Through intermediate-model experiments at different training epochs (Table 8), the authors demonstrate that more realistic noise approximation leads to lower watermark BER, establishing a causal chain between noise fidelity and downstream watermarking performance.

## Limitations & Future Work

1. **Fragility Under Extreme Conditions**: BER reaches 26–30% at ±80° extreme viewing angles and under dark-screen conditions, and 50% under partial cropping. These extreme degradations are insufficiently represented in training data, constituting out-of-distribution failures.
2. **Non-End-to-End Training**: The noise model $G$ and the watermarking network are trained separately. Although cross-source training experiments show limited performance degradation, end-to-end joint optimization could further improve the performance upper bound—a direction the authors identify for future work.
3. **Fixed Training Resolution**: The default training resolution is 128×128. While a resolution scaling strategy (following TrustMark) enables inference at arbitrary resolutions, this remains a post-processing solution rather than native support.
4. **Dependence on the Mathematical Modeling Module**: Although $T$ is replaceable, the system still requires a reasonable initial mathematical model as a starting point. If $T$'s noise simulation deviates too far from real noise, the unsupervised $G$ may lack sufficient bridging capacity.
5. **Limited Evaluation Scenarios**: Testing is primarily conducted on 3 device pairs, without covering print-camera scenarios, video watermarking, or the effects of different screen refresh rates on watermarking—among other broader application settings.

## Related Work & Insights

- **Mathematical Modeling SC Watermarking**: StegaStamp (differentiable physical degradation pipeline), PIMoG (perspective + illumination + Moiré + Gaussian), SSDS (additionally incorporating grayscale bias)—all superimpose independent noise components, ignoring coupling relationships.
- **Supervised Fitting SC Watermarking**: CDTF (Wengrowski et al.)—trains a supervised noise-fitting network on a 1.9 TB real dataset; data acquisition is extremely costly and generalization is limited.
- **Watermarking Frameworks**: HiDDeN, MBRS, Adaptor, MCFN, and other end-to-end watermark encoder-decoder frameworks; S2R uses MCFN as the default watermarking backbone.
- **Unsupervised Image Translation**: CycleGAN, DualGAN (cycle-consistency loss for unpaired translation), Pix2Pix (supervised translation, extended to unsupervised settings such as Blur2Blur).
- **Sim-to-Real Transfer**: A classic paradigm in robotics and autonomous driving; S2R introduces it into the domain of watermark noise modeling.

## Rating

| Dimension | Score (1–10) | Notes |
|-----------|--------------|-------|
| Novelty | 8 | First to introduce the sim-to-real paradigm for SC watermark noise approximation; the combination of mathematical modeling and unsupervised learning is original. |
| Technical Depth | 7 | Theoretical derivation is complete (noise decomposition + feasibility proof), but the core techniques (GAN + perceptual loss) are relatively standard. |
| Experimental Thoroughness | 9 | Covers multiple devices, angles, distances, cross-dataset and cross-training-source settings, and comprehensive ablations. |
| Writing Quality | 8 | Clear structure, well-motivated, rich figures and tables, with a complete Problem → Insight → Solution logical chain. |
| Value | 8 | No paired data required, easy data collection, modular and replaceable components, low barrier to practical deployment. |
| Overall | 8.0 | An elegantly engineered solution that successfully transplants the sim-to-real transfer idea into watermark noise modeling, supported by comprehensive and convincing experiments. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] D-REX: Differentiable Real-to-Sim-to-Real Engine for Learning Dexterous Grasping](../../ICLR2026/robotics/d-rex_differentiable_real-to-sim-to-real_engine_for_learning_dexterous_grasping.md)
- [\[NeurIPS 2025\] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training](../../NeurIPS2025/robotics/generalizable_domain_adaptation_for_sim-and-real_policy_co-training.md)
- [\[CVPR 2026\] Rethinking Camera Choice: An Empirical Study on Fisheye Camera Properties in Robotic Manipulation](../../CVPR2026/robotics/rethinking_camera_choice_an_empirical_study_on_fisheye_camera_properties_in_robo.md)
- [\[CVPR 2026\] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer](../../CVPR2026/robotics/gecosrt_geometryaware_continual_adaptation_for_rob.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](../../ICML2026/robotics/towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)

</div>

<!-- RELATED:END -->
