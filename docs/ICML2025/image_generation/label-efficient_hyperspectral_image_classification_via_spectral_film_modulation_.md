---
title: >-
  [Paper Note] Label-Efficient Hyperspectral Image Classification via Spectral FiLM Modulation of Low-Level Pretrained Diffusion Features
description: >-
  [ICML2025][Image Generation][Diffusion Models] The GeoDiffNet-F framework is proposed, which leverages frozen pre-trained diffusion models to extract low-level spatial features and adaptively injects hyperspectral spectral signatures into these spatial features through the FiLM (Feature-wise Linear Modulation) mechanism, realizing highly efficient hyperspectral image land-cover classification with very limited annotations.
tags:
  - "ICML2025"
  - "Image Generation"
  - "Diffusion Models"
  - "Hyperspectral Imaging"
  - "Remote Sensing"
  - "Label-Efficient Learning"
  - "FiLM Modulation"
date: 2026-05-08
content_hash: a9a55e2486064879
---

# Label-Efficient Hyperspectral Image Classification via Spectral FiLM Modulation of Low-Level Pretrained Diffusion Features

**Conference**: ICML2025  
**arXiv**: [2512.03430](https://arxiv.org/abs/2512.03430)  
**Authors**: Yuzhen Hu, Biplab Banerjee, Saurabh Prasad  
**Code**: TBC  
**Area**: Image Generation  
**Keywords**: Diffusion Models, Hyperspectral Imaging, Remote Sensing, Label-Efficient Learning, FiLM Modulation  

## TL;DR

The GeoDiffNet-F framework is proposed, which leverages frozen pre-trained diffusion models to extract low-level spatial features and adaptively injects hyperspectral spectral signatures into these spatial features through the FiLM (Feature-wise Linear Modulation) mechanism, realizing highly efficient hyperspectral image land-cover classification with very limited annotations.

## Background & Motivation

**Problem Definition**: Hyperspectral image (HSI) land-cover classification is a fundamental task in the remote sensing domain, but it faces three key challenges:

**Label Scarcity**: Pixel-scale labelling requires domain experts and is highly expensive and time-consuming, resulting in extremely sparse available training samples.

**Low Spatial Resolution**: There is a trade-off between spectral fidelity and spatial resolution; HSIs usually contain weaker texture information and worse spatial details.

**Curse of Dimensionality**: Hundreds of contiguous spectral bands bring heavy computational overhead and a high risk of overfitting, which is especially severe when labels are limited.

**Limitations of Prior Work**:
- Traditional multi-modal fusion methods (e.g., HSI+RGB/SAR) and Transformer architectures (e.g., SpectralFormer) heavily rely on a large volume of supervised labels.
- Prior works on diffusion models in remote sensing (e.g., SpectralDiff) train 3D diffusion models from scratch, which is computationally expensive and requires large training sets.
- Previous feature extraction research using diffusion models assumes alignment between source and target domains, leaving cross-domain transferability unexplored.
- Self-supervised methods like MAE are less flexible than probabilistic frameworks in handling uncertainty and degraded inputs.

**Core Motivation**: Pre-trained diffusion models learn rich spatial structures and pixel-level contextual dependencies through iterative denoising, and their low-level features (edges, textures) exhibit cross-domain transferability. The core questions are: (1) How to transfer a diffusion model pre-trained on natural images to the remote sensing domain? (2) How to adaptively fuse spatial and spectral bi-modal information under sparse supervision?

## Method

### Overall Architecture: GeoDiffNet and GeoDiffNet-F

The framework consists of two complementary branches:
- **Spatial Branch (GeoDiffNet)**: Extracts spatial features using a frozen pre-trained diffusion model trained on ImageNet.
- **Spectral Branch**: Encodes the full spectral signature of each pixel to generate FiLM modulation parameters.
- **Fusion Module (GeoDiffNet-F)**: Injects spectral conditions into spatial features through FiLM layers to achieve adaptive multi-modal fusion.

### Key Designs

#### Design 1: Spatial Feature Extraction from Diffusion Model

1. **Pseudo-RGB Construction**: Selects three spectral bands corresponding to red, green, and blue wavelengths from the hyperspectral image to construct pseudo-RGB images.
2. **Patch Division**: Segments images into $64 \times 64$ overlapping patches (stride 32) to match the training scale of the diffusion model.
3. **Frozen Inference**: Employs a pre-trained diffusion model based on the U-Net architecture (containing a 12-layer decoder) with completely frozen parameters.
4. **Forward Process Feature Extraction**: Computes $x_t$ directly from $x_0$ in a single step using the forward diffusion formulation $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$, avoiding the high computational overhead of iterative sampling.
5. **Low Timestep Sampling**: Extracts features at early timesteps such as $t=0, 50, 100$ to preserve more local details of the original image.

#### Design 2: Cross-Domain Transferability Analysis of Low-Level Features

The paper proposes a **dual-hierarchy analysis framework**:
- **Spatial Hierarchy** (U-Net decoder layers):
    - Deep layers (layers 2-5): Encode high-level semantic features, strongly tied to the pre-training domain.
    - Shallow layers (layers 9-11): Preserve low-level spatial details (edges, textures) with stronger domain independence.
- **Temporal Hierarchy** (denoising timesteps):
    - High-noise timesteps: Capture coarse-grained global structure.
    - Low-noise timesteps: Restore fine-grained local details.

**Core Finding**: Despite the significant domain gap between natural and remote sensing images (differences in spatial scale, imaging geometry, spectral coverage, etc.), the low-level features corresponding to the upper layers of the U-Net decoder (layers 9-11) can still be effectively transferred to geospatial image analysis tasks.

#### Design 3: FiLM Spectral Conditional Modulation

The core workflow of the FiLM (Feature-wise Linear Modulation) fusion mechanism is as follows:
1. **Spectral Encoding**: Passes the hyperspectral signature $s_i \in \mathbb{R}^b$ (where $b$ is the number of bands) of each pixel through a lightweight spectral encoder to obtain a compact embedding.
2. **Parameter Regression**: Regresses scaling vectors $\gamma(s_i) \in \mathbb{R}^d$ and shift vectors $\beta(s_i) \in \mathbb{R}^d$ through MLPs.
3. **Feature Modulation**: Conditions and modulates spatial features in a pixel-wise manner: $\hat{f}_i = \gamma(s_i) \cdot f_i^{\text{spatial}} + \beta(s_i)$.
4. **Classification Output**: The modulated features are passed through a 2-layer MLP for pixel-level land-cover classification.

**Advantages over traditional fusion methods**: Compared to concatenation or summation, FiLM offers more flexible learnable cross-modal interactions, achieving dynamic adaptation through feature-level scaling and shifting.

### Loss & Training

- **Freeze Strategy**: Complete freezing of diffusion model parameters, training only the spectral encoder, FiLM parameter regression MLP, and classification head.
- **Parameter-Efficient**: Extremely few trainable parameters (only lightweight MLPs), suitable for small-label scenarios.
- **Input Processing**: Overlapping patching of $64 \times 64$ patches with a stride of 32, balancing computational efficiency and coverage completeness.
- **Evaluation Metrics**: Overall Accuracy (OA), Average Accuracy (AA), Kappa Coefficient (KC).

## Key Experimental Results

### Datasets and Experimental Settings

The experiments are validated on two recent hyperspectral remote sensing datasets:

| Dataset | Scene | Characteristics | Evaluation Metrics |
|--------|------|------|----------|
| Augsburg | Augsburg, Germany | Multi-class land-cover, mixed urban & agricultural areas | OA / AA / KC |
| Berlin | Berlin, Germany | Urban remote sensing scenes, fine classification | OA / AA / KC |

### Decoder Layers and Timestep Ablation Analysis

The paper systematically evaluates the joint performance of decoder layers 2-11 with different timesteps ($t=0, 50, 100$):

| Configuration | Best Layer for Augsburg | Best Layer for Berlin | Explanation |
|------|----------------|--------------|------|
| $t=0$ | Layer 10 (Best) | Layer 11 | Preserves original features most completely |
| $t=50$ | Layer 10 | Layer 11 (Best) | Slight noise may enhance features |
| $t=100$ | Layer 10 | Layer 11 | More noise, performance decreases slightly |

**Key Conclusions**:
- Performance on both datasets peaks at upper decoder layers (layers 10-11), verifying the assumption of strong cross-domain transferability for low-level features.
- Transfer results of high-level semantic features corresponding to lower decoder layers (layers 2-5) are poor due to strong coupling with the pre-training domain.
- The optimal timestep is relatively low ($t=0$ or $t=50$), indicating that preserving the local structure of the original image is crucial for remote sensing tasks.

### Comparison with SOTA Methods

The paper compares the proposed method with state-of-the-art methods under the condition of using only sparse training labels:

| Method Type | Representative Methods | Characteristics | Compared to GeoDiffNet-F |
|----------|----------|------|-------------------|
| Fully-supervised multi-modal fusion | Joint HSI+RGB/SAR | Requires a loss of labels + multi-source data | GeoDiffNet-F requires only sparse labels |
| Transformer architectures | SpectralFormer | Relies on full supervision | GeoDiffNet-F outperforms SOTA under limited labels |
| Remote sensing diffusion models | SpectralDiff (3D) | Trained from scratch, computationally expensive | Zero extra training, direct transfer |
| Self-supervised methods | MAE, etc. | Deterministic framework | Probabilistic framework is more robust |

Experiments show that GeoDiffNet-F outperforms existing methods on both datasets, achieving optimal performance using only the provided sparse training labels.

## Highlights & Insights

1. **New Paradigm for Cross-Domain Transfer**: First systematic validation of the feature transferability of general pre-trained diffusion models (ImageNet) on remote sensing hyperspectral images, challenging the common assumption that "remote sensing requires specialized models."
2. **Dual-Hierarchy Analysis**: Proposes a 2-D grid search framework of decoder layers $\times$ timesteps, systematically revealing the advantages of low-level features in cross-domain scenarios, providing methodological guidance for cross-domain feature extraction.
3. **Clever Application of FiLM Fusion**: Introduces the FiLM conditioning mechanism into hyperspectral-spatial fusion, achieving a lightweight yet effective cross-modal interaction that is far more flexible than simple concatenation/addition.
4. **Extreme Parameter Efficiency**: Freezes the entire diffusion model and trains only a lightweight MLP and spectral encoder, showing significant practical value in remote sensing scenarios with extremely limited annotations.
5. **Efficient Utilization of the Forward Process**: Avoids the iterative sampling overhead of diffusion models, directly extracting multi-step features through a single forward computation.

## Limitations & Future Work

1. **Pseudo-RGB Information Loss**: Selecting only 3 bands from hundreds of spectral bands to construct pseudo-RGB inputs for the diffusion model results in a significant loss of spectral information; although the FiLM branch compensates for this, the input utilization of the spatial branch remains low.
2. **Limited Dataset Scale**: Only validated on two datasets (Augsburg and Berlin), lacking tests on a broader range of remote sensing scenes (e.g., different geographical regions) and different sensor data.
3. **Fixed Resolution Constraint**: The fixed patch size of $64 \times 64$ may not be suitable for hyperspectral data of all spatial resolutions, lacking a multi-scale processing mechanism.
4. **Lack of Comparison with Latest Foundation Models**: No comparison was conducted with specialized foundation models in the remote sensing domain (such as SatMAE, ScaleMAE, etc.).
5. **Heuristic Timestep Selection**: Only explored three discrete timesteps ($t=0, 50, 100$), lacking an adaptive timestep selection mechanism.
6. **Single Diffusion Model**: Only used DDPM pre-trained on ImageNet, without exploring the effects of larger-scale pre-trained models like Stable Diffusion.

## Related Work & Insights

- **Baranchuk et al. (2021)**: First to prove that pre-trained diffusion models can provide strong pixel-level representations, outperforming self-supervised methods under limited labels $\rightarrow$ Direct source of inspiration for this work.
- **FiLM (Perez et al., 2018)**: The original work on Feature-wise Linear Modulation, proposing conditioning through affine transformations $\rightarrow$ Introduced into remote sensing multi-modal fusion in this work.
- **DDPM feature extraction (Xu et al., 2023; Zhang et al., 2023)**: Feature extraction from different U-Net layers and timesteps for segmentation $\rightarrow$ Extended to cross-domain remote sensing scenarios in this work.
- **SpectralFormer (Hong et al., 2021)**: Application of Transformers in hyperspectral classification $\rightarrow$ This work demonstrates that diffusion features combined with FiLM can outperform such methods.
- **Transfer Learning Theory (Long et al., 2015; Yosinski et al., 2014)**: Low-level features have strong cross-domain transferability $\rightarrow$ This classic theory is validated on diffusion models in this work.

**Inspiration for Future Research**: This work demonstrates that the low-level features of general large-scale pre-trained models (even if not specifically trained for remote sensing) can still be effectively transferred to specialized scientific imaging domains. This provides theoretical and practical groundings for applying the "frozen pre-training + lightweight adaptation" paradigm to other scientific imaging tasks, such as medical imaging and astronomical observations.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First systematic analysis of the transferability of diffusion model low-level features in remote sensing cross-domain tasks; the application of the FiLM fusion mechanism in the hyperspectral domain is novel.
- **Experimental Thoroughness**: ⭐⭐⭐ — The ablation studies are comprehensive (layers and timesteps 2D grid), but with only 2 datasets, it lacks comparisons with the latest foundation models.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, systematic description of the method, and the dual-hierarchy analysis framework is well-presented.
- **Value**: ⭐⭐⭐⭐ — Provides a practical pre-trained model transfer scheme for low-label remote sensing scenarios, contributing to cross-domain feature transfer theories.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] IntLoRA: Integral Low-rank Adaptation of Quantized Diffusion Models](intlora_integral_low-rank_adaptation_of_quantized_diffusion_models.md)
- [\[NeurIPS 2025\] FocalCodec: Low-Bitrate Speech Coding via Focal Modulation Networks](../../NeurIPS2025/image_generation/focalcodec_low-bitrate_speech_coding_via_focal_modulation_networks.md)
- [\[ICCV 2025\] Spectral Image Tokenizer](../../ICCV2025/image_generation/spectral_image_tokenizer.md)
- [\[CVPR 2026\] Advancing Image Classification with Discrete Diffusion Classification Modeling](../../CVPR2026/image_generation/advancing_image_classification_with_discrete_diffusion_classification_modeling.md)
- [\[ICML 2026\] Spectral Guidance for Flexible and Efficient Control of Diffusion Models](../../ICML2026/image_generation/spectral_guidance_for_flexible_and_efficient_control_of_diffusion_models.md)

</div>

<!-- RELATED:END -->
