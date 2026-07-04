---
title: >-
  [Paper Note] Reversible Decoupling Network for Single Image Reflection Removal
description: >-
  [CVPR 2025][Image Restoration][Image Reflection Removal] RDNet proposes a single image reflection removal method based on a reversible decoupling architecture, which ensures lossless transmission of multi-scale semantic information during forward propagation through a multi-column reversible encoder, and designs a transmission-rate-aware prompt generator to adaptively handle varying reflection intensities. It comprehensively outperforms SOTAs on five benchmark datasets and wo…
tags:
  - "CVPR 2025"
  - "Image Restoration"
  - "Image Reflection Removal"
  - "Reversible Network"
  - "Information Preservation"
  - "Multi-column Encoder"
  - "Transmission-Aware"
date: 2026-05-08
content_hash: ccb12b33c4cc96bd
---

# Reversible Decoupling Network for Single Image Reflection Removal

**Conference**: CVPR 2025  
**arXiv**: [2410.08063](https://arxiv.org/abs/2410.08063)  
**Code**: [https://github.com/lime-j/RDNet](https://github.com/lime-j/RDNet)  
**Area**: Image Restoration  
**Keywords**: Image Reflection Removal, Reversible Network, Information Preservation, Multi-column Encoder, Transmission-Aware

## TL;DR

RDNet proposes a single image reflection removal method based on a reversible decoupling architecture, which ensures lossless transmission of multi-scale semantic information during forward propagation through a multi-column reversible encoder, and designs a transmission-rate-aware prompt generator to adaptively handle varying reflection intensities. It comprehensively outperforms SOTAs on five benchmark datasets and won the NTIRE 2025 challenge.

## Background & Motivation

1. **Background**: Single image reflection removal (SIRR) aims to separate the transmission layer $T$ and reflection layer $R$ from a blended image $I=T+R$ captured through glass. Deep learning methods mainly rely on two technical routes: utilizing hierarchical semantic features of pre-trained models (e.g., VGG hypercolumn) and dual-stream interaction networks.
2. **Limitations of Prior Work**: (1) According to the information bottleneck principle, high-level semantic cues are compressed or discarded during layer-by-layer propagation—stacking high-dimensional hierarchical features and then mapping them to low-dimensional space inevitably causes a loss of semantic information; (2) In dual-stream interaction networks (such as the linear assumption in YTMT and multiplicative gating in DSRNet), feature interactions follow a fixed pattern, and information preservation cannot be fully guaranteed.
3. **Key Challenge**: The SIRR task is inherently ill-posed, requiring as much information as possible to resolve decomposition ambiguity, but existing architectures continuously lose information during feature transmission.
4. **Goal**: To design an architecture that guarantees lossless propagation of multi-scale semantic information while flexibly decoupling transmission- and reflection-related features.
5. **Key Insight**: Reversible networks naturally possess information-preservation properties—inputs can be precisely reconstructed from outputs, thereby preventing information loss. Combining this with the multi-column hierarchical structure of GLOM/RevCol allows cross-scale interaction while preserving information.
6. **Core Idea**: Replace the traditional U-Net-like structure with a multi-column reversible encoder to ensure lossless information propagation via reversible connections, while dynamically adapting to different scene reflection intensities using a transmission-rate-aware prompt generator.

## Method

### Overall Architecture

RDNet consists of three core modules: a Pre-trained Hierarchical Feature Extractor (PHE) extracts multi-scale semantic features from the input image, feeding them into the first column of the Multi-column Reversible Encoder (MCRE); a Transmission-Aware Prompt Generator (TAPG) estimates transmission rate parameters and generates channel-wise prompts to modulate features; within MCRE, transmission/reflection features are progressively decoupled across multiple columns via intra-layer reversible connections and cross-layer bidirectional interactions; the hierarchical encoding of each column is decoded by a Hierarchical Decoder (HDec), with the output of the final column yielding the final separation result.

### Key Designs

1. **Multi-column Reversible Encoder (MCRE)**:
    - **Function**: Progressively decouple multi-scale transmission/reflection features while ensuring lossless information propagation.
    - **Mechanism**: Inspired by RevCol, it employs multiple subnetworks ("columns"), with each column processing multi-scale information. Column-to-column propagation involves two mechanisms: intra-layer reversible connections (ensuring lossless information) and cross-layer connections (bidirectional interaction). Specifically, the feature of the $j$-th layer in the $i$-th column is expressed as $F_j^i = \omega(\theta(F_{j-1}^i) + \delta(F_{j+1}^{i-1})) + \gamma F_j^{i-1}$, where $\gamma$ is a learnable reversible channel scaling operation. Its reverse operation is $F_j^{i-1} = \gamma^{-1}[F_j^i - \omega(\theta(F_{j-1}^i) + \delta(F_{j+1}^{i-1}))]$, ensuring that information from the previous column can be accurately reconstructed. The embedding layer uses a $7 \times 7$ convolution (stride=2) to generate $2 \times 2$ overlapping patches.
    - **Design Motivation**: Traditional dual-stream interaction networks (such as linear operations in YTMT and gating mechanisms in DSRNet) lose information during interaction. Reversible connections fundamentally solve this issue—learning "separation and reorganization" rather than "selection and discarding." The multi-column design accommodates the need for cross-scale interactions.

2. **Transmission-Aware Prompt Generator (TAPG)**:
    - **Function**: Estimate the transmission rate parameters of the scene and generate channel-wise prompts to guide the decoupling network.
    - **Mechanism**: A simplified pre-trained ConvNext model is used to estimate six parameters $\alpha_{\{R,G,B\}}, \beta_{\{R,G,B\}}$ that minimize $\|\alpha_i T + \beta_i - I\|_2$. A three-layer MLP is then used to convert these parameters into a prompt $P$ of dimension $C \times H \times W$, which modulates the features of the column embedding layer via element-wise multiplication $P \circ F$. Training is performed in two stages: first training the transmission-rate estimator, then fixing it while training the main network.
    - **Design Motivation**: Reflection patterns in real-world scenes vary based on multiple factors like refractive index, color granularity, and viewing angle. Directly adjusting the input image (by dividing by the estimated $\alpha$) introduces irreparable bias when estimates are inaccurate. Indirectly modulating features via channel-wise prompts is more flexible and robust—estimating just coarse adjustment parameters with 24.34dB already outperforms previous SOTAs.

3. **Hierarchical Decoder (HDec)**:
    - **Function**: Integrate and decode multi-scale hierarchical encodings into the final transmission/reflection images.
    - **Mechanism**: Multiple Level Decoders (LDs) are employed to decode layer-by-layer using pixel-shuffle upsampling (an information-preserving operation), fusing multi-scale information via multiplicative modulation. The final output layer produces residuals $\hat{T}_{res}$ and $\hat{R}_{res}$, which are added to the original input to obtain the separated results $\hat{T}$ and $\hat{R}$.
    - **Design Motivation**: Pixel-shuffle is an information-consistent scaling operation (unlike standard upsampling which loses information), which effectively fuses cross-scale features when paired with multiplicative modulation. Residual learning reduces the learning burden on the network.

### Loss & Training

Two-stage training: In the first stage, the transmission-rate estimator is trained (using only MSE loss). In the second stage, the estimator is fixed, and the main network + prompt generator are trained.

Total loss $\mathcal{L} = \mathcal{L}_{\text{cont}} + 0.01 \cdot \mathcal{L}_{\text{per}}$:
- Content loss: $\mathcal{L}_{\text{cont}} = 0.3\|\hat{T}-T\|^2 + 0.9\|\hat{R}-R\|^2 + 0.6\|\nabla\hat{T}-\nabla T\|_1$, constraining both transmission and reflection components, along with gradient domain regularization.
- Perceptual loss: $\ell_1$ difference of VGG-19 multi-scale features (from conv2_2 to conv5_2).
- Adam optimizer, learning rate $10^{-4}$, batch size 2, trained for 20 epochs on an RTX 3090.

## Key Experimental Results

### Main Results

| Dataset | Metric | RDNet (Ours) | DSRNet (Prev. SOTA) | Gain |
|--------|------|-------------|-------------------|------|
| Real20 | PSNR/SSIM | **25.58**/0.846 | 24.23/0.820 | +1.35 |
| Objects | PSNR/SSIM | **26.78**/0.921 | 26.74/0.920 | +0.04 |
| Postcard | PSNR/SSIM | **26.33**/0.922 | 24.83/0.911 | +1.50 |
| Wild | PSNR/SSIM | **27.70**/0.915 | 26.11/0.906 | +1.59 |
| Average | PSNR/SSIM | **26.65**/0.917 | 25.75/0.910 | +0.90 |

### Ablation Study

| Configuration | PSNR | SSIM | Explanation |
|------|------|------|------|
| w/o Transmission Tech (A) | 25.52 | 0.909 | Without any transmission awareness, drops 1.13dB |
| Direct Input Adjust (B) | 25.99 | 0.910 | Naively dividing by the estimated $\alpha$ |
| Input Adjust + Prompt (C) | 26.03 | 0.913 | Adding Prompt after input adjustment gives no further gain |
| **Full Model (Ours)** | **26.65** | **0.917** | Using Prompt only yields the best performance |
| Dual-stream Design (D) | 26.37 | 0.917 | DSRNet-style dual-stream, doubles computation but drops 0.28dB |
| w/o Reflection Loss (E) | 25.99 | 0.914 | Removing loss on R, drops 0.66dB |
| U-Net Connection (F) | 24.05 | 0.884 | Replacing reversible connections with U-Net, drops **2.6dB** |
| 2 Columns | 26.25 | 0.914 | Insufficient columns |
| 4 Columns | **26.65** | **0.917** | Optimal columns |
| 6 Columns | 26.19 | 0.910 | Excess columns lead to degradation |

### Key Findings

- **Reversibility is crucial**: Replacing reversible connections with U-Net connections leads to a massive 2.6dB drop, making it the most critical component.
- The addition of reflection loss contributes 0.66dB—supervising both components simultaneously helps the network distinguish them more clearly.
- Transmission-aware prompts contribute 1.13dB, and indirect modulation (via prompt) is superior to direct input adjustment.
- The optimal number of columns is 4; too few leads to insufficient decoupling, while too many may introduce redundancy.
- In user studies, the method beats Zhu et al. with a 77.2% win rate, and beats DSRNet with a 64.4% win rate.

## Highlights & Insights

- **Philosophy of Reversible Decoupling**: Instead of "selecting and discarding" information, it "classifies and reorganizes" it. Reversible connections ensure no information is irreversibly discarded during the decomposition process. This design philosophy can be transferred to other tasks requiring signal decomposition (e.g., white balancing, shadow removal, image harmonization).
- **Clever Design of Transmission-Aware Prompt**: Only 6 scalar parameters capture the global reflection properties of the scene, which are then expanded into spatial prompts via an MLP. This avoids introducing irreparable bias caused by directly manipulating the input.
- **NTIRE 2025 Challenge Champion**: Achieved the best performance in both fidelity and perceptual quality dimensions, demonstrating the practical competitiveness of the method.

## Limitations & Future Work

- The physical model used by the authors (linear overlay) may fail in complex scenarios such as overexposure.
- PHE uses a pre-trained FocalNet, which incurs extra computational overhead during inference.
- The computational and memory requirements of the 4-column structure remain high.
- Possible improvement directions: introducing diffusion models for generative reflection removal, combining multi-frame information to enhance robustness, and handling non-linear reflection models.

## Related Work & Insights

- **vs YTMT**: YTMT relies on a dual-stream interaction based on the "you trans, me trans" strategy, but the interaction module depends on a linear assumption, and the information bottleneck limits performance. RDNet fundamentally resolves the information loss issue using a reversible structure.
- **vs DSRNet**: DSRNet is also a dual-stream mutual guidance design, but the multiplicative gating mechanism yields information decay, and progressive hierarchical fusion occurs only in the first stage. RDNet's reversible connections span the entire inter-column propagation, ensuring information integrity.
- **vs RevCol/GLOM**: RDNet borrows the multi-column reversible design from RevCol but innovatively applies it to reflection removal—a task requiring dual decoupling—and adds transmission-aware dynamic modulation across columns.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of reversible networks to reflection removal, combined with a clever transmission-aware prompt, although the core is a combination of existing technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensively leading across five datasets, detailed ablations, user studies, and NTIRE championship.
- Writing Quality: ⭐⭐⭐⭐ Clear motivational derivation and rigorous formulas, though the paper structure is slightly verbose.
- Value: ⭐⭐⭐⭐ NTIRE winning solution with high practical representation, demonstrating a new paradigm of reversible decoupling for reflection removal.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LightRR: A Lightweight Network for Single Image Reflection Removal](../../CVPR2026/image_restoration/lightrr_a_lightweight_network_for_single_image_reflection_removal.md)
- [\[CVPR 2026\] GFRRN: Explore the Gaps in Single Image Reflection Removal](../../CVPR2026/image_restoration/gfrrn_explore_the_gaps_in_single_image_reflection_removal.md)
- [\[CVPR 2025\] Gyro-based Neural Single Image Deblurring](gyro-based_neural_single_image_deblurring.md)
- [\[CVPR 2026\] Rectifying Latent Space for Generative Single-Image Reflection Removal](../../CVPR2026/image_restoration/rectifying_latent_space_for_generative_single-image_reflection_removal.md)
- [\[CVPR 2025\] PolarFree: Polarization-based Reflection-Free Imaging](polarfree_polarization-based_reflection-free_imaging.md)

</div>

<!-- RELATED:END -->
