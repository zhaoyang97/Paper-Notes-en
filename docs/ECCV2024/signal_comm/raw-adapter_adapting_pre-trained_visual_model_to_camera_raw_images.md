---
title: >-
  [Paper Note] RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images
description: >-
  [ECCV 2024][Signal & Communication][RAW images] This paper proposes RAW-Adapter, which efficiently adapts sRGB pre-trained models to camera RAW images with extremely small parameter overhead (0.2–0.8M) via an input-level adapter (learnable ISP stages) and a model-level adapter (injecting ISP intermediate features into the backbone). It achieves SOTA performance on detection and segmentation tasks under various lighting conditions, including normal, low-light…
tags:
  - "ECCV 2024"
  - "Signal & Communication"
  - "RAW images"
  - "Image Signal Processor"
  - "adapter"
  - "object detection"
  - "semantic segmentation"
date: 2026-05-08
content_hash: 57b06888ecc2713e
---

# RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images

**Conference**: ECCV 2024  
**arXiv**: [2408.14802](https://arxiv.org/abs/2408.14802)  
**Code**: [Available](https://github.com/cuiziteng/RAW-Adapter)  
**Area**: Signal Communication  
**Keywords**: RAW images, Image Signal Processor, adapter, object detection, semantic segmentation

## TL;DR

This paper proposes RAW-Adapter, which efficiently adapts sRGB pre-trained models to camera RAW images with extremely small parameter overhead (0.2–0.8M) via an input-level adapter (learnable ISP stages) and a model-level adapter (injecting ISP intermediate features into the backbone). It achieves SOTA performance on detection and segmentation tasks under various lighting conditions, including normal, low-light, and overexposure.

## Background & Motivation

Current computer vision models are mainly pre-trained on sRGB images. However, RAW images possess unique advantages: they are **uncompressed by ISP**, retaining rich physical information (such as noise distribution), show a linear relationship with radiant energy, and offer a higher dynamic range. Under real-world extreme lighting conditions (where solar irradiance ranges from $1.3 \times 10^3 W/m^2$ to $2.0 \times 10^{-6} W/m^2$), the benefits of RAW images are particularly pronounced.

However, existing methods leveraging RAW images for vision tasks present three core issues:

**ISP Objective Mismatch**: Traditional ISPs aim to generate "visually pleasing" images rather than **optimizing for downstream vision tasks**. Hand-crafted ISPs sometimes perform even worse than directly using RAW data.

**Limitations of Joint Training Methods**: Existing schemes either utilize evolutionary strategies to optimize traditional ISP parameters (e.g., Hardware-in-the-loop) or employ neural networks to substitute the ISP (e.g., Dirty-Pixel using a residual UNet). However, the former is non-differentiable, and the latter introduces heavy computational burdens (Dirty-Pixel adds 4.28M parameters).

**Lack of Interaction**: The aforementioned methods treat the ISP and the downstream network as two independent modules, lacking **information interaction** between the ISP stages and the downstream pipeline. Meanwhile, training on RAW data from scratch discards the knowledge from large-scale sRGB pre-trained models (Fig.2 demonstrates that using pre-trained weights yields significant performance improvements).

**Core Problem**: How to efficiently adapt information-rich RAW images to knowledge-rich sRGB pre-trained models?

## Method

### Overall Architecture

RAW-Adapter contains two layers of adapters: the **input-level adapter** (Input-level Adapters) maps the RAW image $\mathbf{I}_1$ into a machine-vision-friendly image $\mathbf{I}_5$; the **model-level adapter** (Model-level Adapters) injects intermediate features from various ISP stages into different stages of the backbone network, enhancing their information interaction. The overall framework maintains the modular design of a traditional ISP but makes all key parameters learnable and differentiable via Query Adaptive Learning (QAL).

### Key Designs

1. **Query Adaptive Learning (QAL)**: Used to adaptively predict the key parameters of various ISP stages. The design is inspired by the attention mechanism in Transformers: the input image is downsampled via two convolutional layers to extract features, generating a key $\mathbbm{k}$ and a value $\mathbbm{v}$, while the query $\mathbbm{q}$ is a set of **learnable dynamic parameters**. Self-attention is computed to predict ISP parameters:

$$parameters = FFN\left(softmax\left(\frac{\mathbbm{q} \cdot \mathbbm{k}^T}{\sqrt{d_k}}\right) \cdot \mathbbm{v}\right)$$

This allows each image to adaptively predict distinct ISP parameters based on its own content. Two QAL blocks are defined: $\mathbb{P_K}$ (predicting gain/denoising parameters) and $\mathbb{P_M}$ (predicting white balance/CCM parameters).

2. **ISP Stages of Input-level Adapter**: The modular design of traditional ISP is retained, but the key parameters of each stage are adaptively predicted by QAL:

    - **Gain & Denoise**: $\mathbb{P_K}$ predicts the gain ratio $g$ (to adapt to different lighting) and the anisotropic Gaussian kernel parameters $\{r_1, r_2\}$ (for adaptive denoising), as well as the sharpening parameter $\sigma$:
    $$\mathbf{I}_2' = (g \cdot \mathbf{I}_1) \circledast k, \quad \mathbf{I}_2 = \mathbf{I}_2' + (g \cdot \mathbf{I}_1 - \mathbf{I}_2') \cdot \sigma$$
    - **White Balance & CCM**: $\mathbb{P_M}$ predicts the Minkowski distance parameter $\rho$ (unifying gray-world and Max-RGB white balance) and the $3\times3$ color correction matrix $\mathbf{E}_{ccm}$.
    - **Color Manipulation**: Uses NILUT (Neural Implicit 3D LUT) for end-to-end learnable color mapping.

3. **Model-level Adapter**: The intermediate ISP stages $\mathbf{I}_1 \sim \mathbf{I}_4$ contain rich prior information, which has been completely overlooked by previous methods. The model-level adapter extracts and concatenates features from each ISP stage via convolutional layers: $\mathbb{C}(\mathbf{I}_{1\sim4}) = \mathbb{C}(c_1(\mathbf{I}_1), c_2(\mathbf{I}_2), c_3(\mathbf{I}_3), c_4(\mathbf{I}_4))$. After being processed by residual blocks, the features are step-by-step injected into stages 1-3 of the backbone network via a merge block (concatenation + residual connection). This enables the downstream task network to **utilize prior knowledge from all ISP stages**.

### Loss & Training

- No additional human-vision-oriented loss functions are required—only the loss of the downstream task itself is used (focal loss for detection / cross-entropy for segmentation, etc.). The ISP parameters are optimized end-to-end through the task loss.
- All comparison methods use the same data augmentation and training settings.
- Extremely small parameter footprint: input-level adapter $\mathbb{P_K}$ (37.57K) + $\mathbb{P_M}$ (37.96K) + $\mathbb{L}$ (1.97K), model-level adapter $\mathbb{M}$ (114.87K ~ 687.59K), totaling **0.2M ~ 0.8M**, which is significantly smaller than SID (11.99M) and Dirty-Pixel (4.28M).

## Key Experimental Results

### Main Results

**PASCAL RAW Object Detection (RetinaNet, mAP)**:

| Method | ResNet-18 Normal | ResNet-18 Dark | ResNet-50 Normal | ResNet-50 Dark |
|------|-----------------|---------------|-----------------|---------------|
| Default ISP | 88.3 | - | 89.6 | - |
| Demosaicing | 87.7 | 80.3 | 89.2 | 82.6 |
| Dirty-Pixel | 88.6 | 80.8 | 89.7 | 83.6 |
| **RAW-Adapter** | **88.7** | **82.5** | **89.7** | **86.6** |

**LOD Low-light Detection (ResNet-50, mAP)**:

| Method | RetinaNet | Sparse-RCNN |
|------|-----------|-------------|
| Demosaicing | 58.5 | 57.7 |
| Dirty-Pixel | 61.6 | 58.8 |
| **RAW-Adapter** | **62.1** | **59.2** |

**ADE20K RAW Semantic Segmentation (Segformer, mIoU)**:

| Method | Backbone | Params(M) | Normal | Overexposed | Dark |
|------|----------|-----------|-------|------|------|
| Demosaicing | MIT-B5 | 82.01 | 47.47 | 45.69 | 37.55 |
| Dirty-Pixel | MIT-B5 | 86.29 | 47.86 | 46.50 | 38.02 |
| **RAW-Adapter** | **MIT-B5** | **82.31** | **47.95** | **46.62** | **38.75** |
| **RAW-Adapter** | **MIT-B3** | **45.16** | 46.57 | 44.19 | **37.62** |

### Ablation Study

**Effectiveness of Model-level Adapter (LOD Dataset, mAP)**:

| Configuration | RetinaNet | Sparse-RCNN | Description |
|------|-----------|-------------|------|
| RAW-Adapter (w/o $\mathbb{M}$) | 61.6 | 58.6 | Input-level adapter only |
| **RAW-Adapter (Full)** | **62.1** | **59.2** | With model-level adapter |

**ADE20K RAW Ablation (MIT-B5, mIoU)**:

| Configuration | Normal | Overexposed | Dark | Description |
|------|-------|------|------|------|
| w/o $\mathbb{M}$ | 47.83 | 46.48 | 38.41 | Input-level adapter only |
| **Full** | **47.95** | **46.62** | **38.75** | Full RAW-Adapter |

### Key Findings

- **Most Significant Improvement in Low Light**: On PASCAL RAW (dark), ResNet-50 improves from 83.6 of Dirty-Pixel to 86.6 (+3.0), showing that the high dynamic range advantage of RAW is more prominent under extreme illumination.
- **Small Backbone Outperforms Large Backbone**: RAW-Adapter + ResNet-18 low-light detection (82.5) outperforms certain ISP methods with ResNet-50.
- **MIT-B3 + RAW-Adapter low-light segmentation (37.62) achieves performance close to Dirty-Pixel + MIT-B5 (38.02)**, using less than half the parameters.
- Traditional ISPs sometimes actually **degrade** downstream task performance, especially under non-normal lighting conditions.
- Even without constraints from human vision loss, RAW-Adapter still generates visually pleasing intermediate images.

## Highlights & Insights

- **Precise Design Philosophy**: Instead of pursuing complex input processing, it focuses on "simplifying the input stage + enhancing the connection between ISP and the network."
- **High Parameter Efficiency**: Only 0.2–0.8M extra parameters, far smaller than existing approaches, and with faster inference speed.
- **Strong Generalizability**: The framework can be adapted to various detectors (RetinaNet, Sparse-RCNN) and segmentors (Segformer), as well as backbones of different sizes.
- Cleverly applies adapter concepts from NLP/CV to the RAW-to-sRGB domain adaptation problem.

## Limitations & Future Work

- Different lighting conditions (normal, dark, overexposed) require **separate training**, and a unified model cannot handle multiple lighting conditions simultaneously.
- Only validation on CNN-based backbones (ResNet, MIT) has been carried out; the adaptation performance on pure Transformer architectures such as ViT remains unexplored.
- The selection and simplification of ISP stages are hand-crafted. An automated selection process for ISP stages could potentially yield further improvements.
- The noise models used for synthesizing low-light and overexposed data are relatively simple; robustness under real-world extreme lighting conditions remains to be verified.

## Related Work & Insights

- Core difference from Dirty-Pixel: Dirty-Pixel completely replaces the ISP with a UNet, acting as a "black-box" method. RAW-Adapter retains the modular ISP design as a "white-box" approach, yielding much better interpretability.
- The design methodology of QAL can be extended to other scenarios requiring adaptive parameter prediction (e.g., adaptive enhancement, adaptive kernel prediction).
- Inspiration from Model-level Adapters: Features from intermediate processing stages should not be discarded; multi-level feature fusion is crucial in domain adaptation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Applying the adapter paradigm to RAW-sRGB domain adaptation is a novel angle; the dual-level design of input-level + model-level showcases strong originality.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers detection and segmentation across multiple lighting conditions and backbones, though the ablation studies could be more detailed.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, well-motivated, with standard formulation.
- **Value**: ⭐⭐⭐⭐ — Significantly drives forward the practical application of RAW images; the parameter-efficient design has strong engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images](querycdr_query-based_controllable_distortion_rectification_network_for_fisheye_i.md)
- [\[ICLR 2026\] TS-DDAE: A Novel Temporal-Spectral Denoising Diffusion AutoEncoder for Wireless Signal Recognition Model Pre-training](../../ICLR2026/signal_comm/ts-ddae_a_novel_temporal-spectral_denoising_diffusion_autoencoder_for_wireless_s.md)
- [\[CVPR 2026\] CLAY: Conditional Visual Similarity Modulation in Vision-Language Embedding Space](../../CVPR2026/signal_comm/clay_conditional_visual_similarity.md)
- [\[ICML 2026\] Joint Model and Data Sparsification via the Marginal Likelihood](../../ICML2026/signal_comm/joint_model_and_data_sparsification_via_the_marginal_likelihood.md)
- [\[ICLR 2026\] Synchronizing Probabilities in Model-Driven Lossless Compression](../../ICLR2026/signal_comm/synchronizing_probabilities_in_model-driven_lossless_compression.md)

</div>

<!-- RELATED:END -->
