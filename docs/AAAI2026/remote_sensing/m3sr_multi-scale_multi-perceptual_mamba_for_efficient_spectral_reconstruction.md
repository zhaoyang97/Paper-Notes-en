---
title: >-
  [Paper Note] M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction
description: >-
  [AAAI 2026][Remote Sensing][Spectral Reconstruction] This paper proposes M3SR, a Mamba-based multi-scale multi-perceptual architecture that integrates spatial, frequency…
tags:
  - "AAAI 2026"
  - "Remote Sensing"
  - "Spectral Reconstruction"
  - "Mamba"
  - "State Space Model"
  - "Multi-Scale"
  - "Hyperspectral Imaging"
date: 2026-05-08
content_hash: 3bab6b9af0d40e18
---

# M3SR: Multi-Scale Multi-Perceptual Mamba for Efficient Spectral Reconstruction

**Conference**: AAAI 2026
**arXiv**: [2601.08293](https://arxiv.org/abs/2601.08293)
**Code**: [https://github.com/zhangyuzecn/M3SR](https://github.com/zhangyuzecn/M3SR)
**Area**: Remote Sensing / Hyperspectral Image Reconstruction
**Keywords**: Spectral Reconstruction, Mamba, State Space Model, Multi-Scale, Hyperspectral Imaging

## TL;DR
This paper proposes M3SR, a Mamba-based multi-scale multi-perceptual architecture that integrates spatial, frequency, and spectral branches in parallel within a U-Net multi-scale structure. With only 2.17M parameters and 100.9G FLOPs, M3SR surpasses existing state-of-the-art methods on four spectral reconstruction benchmarks.

## Background & Motivation
Hyperspectral imaging (HSI) captures rich spatial-spectral information through narrow spectral bands and is widely applied in environmental monitoring, medical imaging, and agriculture. However, direct HSI acquisition is costly and complex, making **spectral reconstruction (SR)**—generating HSI from RGB images—an important alternative.

Existing SR methods exhibit a clear developmental trajectory and associated limitations:
- **Traditional methods** (sparse dictionaries, Gaussian processes, low-rank representations) struggle to identify complex patterns.
- **CNN-based methods** (HSCNN+, HRNet) improve performance but fail to capture long-range dependencies.
- **Transformer-based methods** (MST++, ESSAformer) model long-range relationships but suffer from computational complexity that scales sharply with image size.

The **Mamba** architecture, based on state space models (SSMs), processes long sequences with linear complexity. Nevertheless, existing Mamba-based SR methods face two core challenges:
1. **Single spatial perception** limits comprehensive understanding of hyperspectral images.
2. **Single-scale feature extraction** struggles to simultaneously capture complex structures and fine-grained details.

The core idea of M3SR is to design a Multi-Perceptual Fusion (MPF) block that integrates spatial, frequency, and spectral perception into a unified module, which is then embedded within a U-Net for multi-scale feature extraction and fusion.

## Method

### Overall Architecture
M3SR adopts a U-Net-based encoder-decoder structure:
1. **Input stage**: Receives an RGB image and extracts shallow features.
2. **Encoder path**: Progressively extracts multi-scale semantic features via downsampling across three scales:
   - **Global scale**: Captures overall structural information.
   - **Intermediate scale**: Focuses on contextual information.
   - **Local scale**: Recovers fine-grained texture details.
3. **Decoder path**: Restores spatial resolution through upsampling, fusing multi-scale features via skip connections.
4. **Output stage**: Generates the reconstructed hyperspectral image.

The core building block at each scale is the **Multi-Perceptual Fusion (MPF) Block**, comprising three parallel branches for spatial, frequency, and spectral perception.

### Key Designs

**Design 1: Multi-Perceptual Fusion Block (MPF Block) — Three-Branch Parallel Perception**

The MPF Block contains three parallel branches targeting different information dimensions:

**(1) Spatial perception branch**: Based on VMamba's 2D Selective Scan (SS2D), this branch unfolds 2D images into 1D sequences and scans in four directions to capture long-range spatial dependencies. The VSS block is defined as:

$$VSS(x) = Lin(LN(SS2D(x')))$$
$$x' = SiLU(DWConv(Lin(x)))$$

Spatial features are enhanced via reshape and concatenation:

$$\mathbf{F}_a^2 = Reshape(Concat(VSS(LN(\mathbf{F}_a^1)), \mathbf{F}_a^1))$$

**(2) Frequency perception branch**: Applies the Discrete Wavelet Transform (DWT) to decompose the input into a low-frequency component ($I_{LL}$) and three high-frequency components ($I_{LH}, I_{HL}, I_{HH}$) to capture texture details:

$$\mathbf{F}_f^1 = DWT(\mathbf{F}_{in})$$
$$\mathbf{F}_f^2 = IDWT(Concat(VSS(LN(\mathbf{F}_f^1)), \mathbf{F}_f^1))$$

**(3) Spectral perception branch**: Based on the original Mamba block, this branch models continuous dependencies along the spectral dimension. The channel dimension $C$ is expanded to $C \times G$, and grouped features are processed by the Mamba block to extract spectral interaction features:

$$\mathbf{F}_e^1 = Mamba(Reshape(Conv(\mathbf{F}_{in})))$$
$$\mathbf{F}_e^2 = Conv(Reshape(\mathbf{F}_e^1))$$

The Mamba block is defined as: $Mamba(x) = Lin(x' + x'')$, where $x' = SiLU(Lin(x))$ and $x'' = S6(SiLU(DWConv(Lin(x))))$, with S6 denoting the selective SSM.

**Design 2: Adaptive Fusion Mechanism**

The outputs of the three branches are fused via learnable weights with a residual connection:

$$\mathbf{F}_{out} = \omega_a \cdot \mathbf{F}_a^2 + \omega_f \cdot \mathbf{F}_f^2 + \omega_e \cdot \mathbf{F}_e^2 + \mathbf{F}_{in}$$

where $\omega_a, \omega_f, \omega_e$ are randomly initialized and updated through backpropagation. Unlike simple uniform or sequential fusion, adaptive weighting allows the model to dynamically adjust the relative importance of each perceptual branch.

**Design 3: Multi-Scale U-Net Integration**

MPF Blocks are embedded into a symmetric U-Net structure. Through downsampling, upsampling, and skip connections, the architecture performs feature extraction and fusion at global, intermediate, and local scales, balancing global semantic consistency with local texture fidelity.

### Loss & Training
Mean Absolute Error (MAE) is used as the loss function:

$$L = \frac{1}{H \times W \times C} \sum_{i=1}^{H} \sum_{j=1}^{W} \sum_{k=1}^{C} |Z_{i,j,k} - \hat{Z}_{i,j,k}|$$

Training uses the Adam optimizer ($\beta_1=0.9, \beta_2=0.999$) with an initial learning rate of 0.0004 and cosine annealing over 100 epochs. Data augmentation includes random rotation and flipping. Batch size is 32 with 128×128 patch cropping. Training is performed on a single NVIDIA 4090 GPU.

## Key Experimental Results

### Main Results

**Results on NTIRE2022 & CAVE Datasets:**

| Method | Params(M) | FLOPs(G) | NTIRE2022 PSNR↑ | NTIRE2022 RMSE↓ | CAVE PSNR↑ | CAVE RMSE↓ |
|--------|-----------|----------|-----------------|----------------|-----------|-----------|
| HSCNN+ | 1.642 | 808.0 | 25.26 | 0.058 | 33.81 | 0.0227 |
| HRNet | 31.705 | 1249.0 | 25.22 | 0.0577 | 34.53 | 0.0205 |
| MST++ | 1.62 | 177.7 | 30.18 | 0.035 | 34.65 | 0.0205 |
| GMSR | 0.019 | 8.0 | 26.92 | 0.0492 | 34.58 | 0.0206 |
| **M3SR** | **2.166** | **100.9** | **31.40** | **0.0343** | **35.61** | **0.0184** |

**Results on NTIRE2020 Dataset:**

| Method | NTIRE2020-Clean PSNR↑ | NTIRE2020-Real PSNR↑ | Clean RMSE↓ | Real RMSE↓ |
|--------|-----------------------|---------------------|------------|------------|
| MST++ | 36.32 | 35.63 | 0.0198 | 0.0185 |
| HSRNet | 37.17 | 34.55 | 0.0198 | 0.0213 |
| GMSR | 33.97 | 31.90 | 0.0239 | 0.0278 |
| **M3SR** | **37.71** | **36.35** | **0.0196** | **0.0171** |

M3SR achieves 31.40 dB PSNR on NTIRE2022 (surpassing MST++'s 30.18 dB by approximately 1.2 dB), while requiring only 100.9G FLOPs (vs. 177.7G for MST++) and 2.166M parameters.

### Ablation Study

**Ablation on Perceptual Branches (NTIRE2022):**

| Variant | Spatial | Frequency | Spectral | PSNR↑ | RMSE↓ | SAM↓ | MSSIM↑ |
|---------|---------|-----------|----------|-------|-------|------|--------|
| M3SR-V1 | ✗ | ✓ | ✓ | 30.49 | 0.0381 | 12.55 | 0.8827 |
| M3SR-V2 | ✓ | ✗ | ✓ | 30.59 | 0.0365 | 6.52 | 0.9315 |
| M3SR-V3 | ✓ | ✓ | ✗ | 30.36 | 0.0369 | 6.28 | 0.9241 |
| **M3SR** | **✓** | **✓** | **✓** | **31.40** | **0.0343** | **6.62** | **0.9351** |

Removing the spatial branch causes a sharp degradation in SAM from 6.62 to 12.55 and a PSNR drop of approximately 0.9 dB.

**Ablation on Group Number G:**

| G | PSNR↑ | RMSE↓ | Params(M) | FLOPs(G) |
|---|-------|-------|-----------|----------|
| 2 | 31.23 | 0.0351 | 2.066 | 91.3 |
| **4** | **31.40** | **0.0343** | **2.166** | **100.9** |
| 8 | 30.64 | 0.0372 | 2.368 | 120.1 |
| 16 | 30.76 | 0.0361 | 2.770 | 158.6 |

$G=4$ achieves the best trade-off between performance and efficiency.

### Key Findings
- The spatial perception branch has the greatest impact on spectral angle (SAM); its removal causes SAM to surge from 6.62 to 12.55.
- The frequency perception branch contributes most significantly to MSSIM; its removal reduces MSSIM from 0.9351 to 0.9315.
- M3SR achieves higher performance than FMNet (which requires 5819.5G FLOPs) using only 100.9G FLOPs, representing approximately a 57× efficiency improvement.

## Highlights & Insights
- The **three-branch parallel design** unifies spatial (SS2D), frequency (DWT), and spectral (vanilla Mamba) perception into a single module, representing a systematic extension of Mamba to low-level vision tasks.
- **Adaptive weighted fusion** is more flexible than simple concatenation or summation, enabling dynamic adjustment of each perceptual branch's contribution across different scales and datasets.
- Achieving state-of-the-art performance with extremely low computational overhead makes M3SR particularly well-suited for resource-constrained hyperspectral imaging applications.

## Limitations & Future Work
- Only MAE loss is employed; perceptual losses, frequency-domain losses, and other strategies that may further improve reconstruction quality remain unexplored.
- Ablation studies are conducted solely on NTIRE2022, lacking cross-dataset ablation validation.
- DWT is fixed to the Haar wavelet; the effect of alternative wavelet bases is not investigated.
- The spectral group number $G$ requires manual tuning as a hyperparameter, with no adaptive selection mechanism proposed.

## Related Work & Insights
- **vs. MST++**: M3SR surpasses MST++ by 1.2 dB in PSNR while requiring only 56.8% of its FLOPs, demonstrating the advantage of Mamba's linear complexity.
- **vs. GMSR**: Both are Mamba-based SR methods, but GMSR relies solely on a single spatial SSM (PSNR 26.92), whereas M3SR's multi-perceptual fusion raises this to 31.40—a gain of 4.5 dB.
- **vs. HRNet**: HRNet has approximately 15× more parameters (31.7M vs. 2.17M) than M3SR, yet achieves only 25.22 dB PSNR compared to M3SR's 31.40 dB.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The three-branch multi-perceptual Mamba fusion constitutes a systematic innovation in applying SSMs to low-level vision.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Evaluation spans four datasets, ten state-of-the-art baselines, comprehensive ablation studies, and parameter analysis.
- **Writing Quality**: ⭐⭐⭐ — Structure is clear, though notation is occasionally inconsistent across formulations.
- **Value**: ⭐⭐⭐⭐ — Provides an efficient and practical Mamba-based solution for hyperspectral reconstruction with open-source code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Exploring Spatiotemporal Feature Propagation for Video-Level Compressive Spectral Reconstruction](../../CVPR2026/remote_sensing/exploring_spatiotemporal_feature_propagation_for_video-level_compressive_spectra.md)
- [\[AAAI 2026\] Consistency-based Abductive Reasoning over Perceptual Errors of Multiple Pre-trained Models in Novel Environments](consistency-based_abductive_reasoning_over_perceptual_errors_of_multiple_pre-tra.md)
- [\[CVPR 2026\] Joint and Streamwise Distributed MIMO Satellite Communications with Multi-Antenna Ground Users](../../CVPR2026/remote_sensing/joint_and_streamwise_distributed_mimo_satellite_communications_with_multi-antenn.md)
- [\[ACL 2026\] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems](../../ACL2026/remote_sensing/moneta_multimodal_industry_classification_through_geographic_information_with_mu.md)
- [\[ICLR 2026\] Spectral Gaps and Spatial Priors: Studying Hyperspectral Downstream Adaptation Using TerraMind](../../ICLR2026/remote_sensing/spectral_gaps_and_spatial_priors_studying_hyperspectral_downstream_adaptation_us.md)

</div>

<!-- RELATED:END -->
