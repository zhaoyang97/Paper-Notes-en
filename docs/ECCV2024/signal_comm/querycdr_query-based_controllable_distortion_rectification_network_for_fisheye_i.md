---
title: >-
  [Paper Note] QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images
description: >-
  [ECCV 2024][Signal & Communication][Fisheye image rectification] The proposed QueryCDR network, utilizing a distortion-aware learnable query mechanism (DLQM) and two controllable modulation modules (CCMB/CAMB), achieves high-quality controllable rectification for fisheye images with various distortion degrees **without retraining** for the first time.
tags:
  - "ECCV 2024"
  - "Signal & Communication"
  - "Fisheye image rectification"
  - "controllable distortion correction"
  - "learnable queries"
  - "hybrid CNN-Transformer"
  - "generative rectification"
date: 2026-05-08
content_hash: d6966ed090607764
---

# QueryCDR: Query-based Controllable Distortion Rectification Network for Fisheye Images

**Conference**: ECCV 2024  
**arXiv**: [2412.13496](https://arxiv.org/abs/2412.13496)  
**Code**: [https://github.com/PbGuo/QueryCDR](https://github.com/PbGuo/QueryCDR)  
**Area**: Signal and Communication (Fisheye Image Rectification/Low-level Vision)  
**Keywords**: Fisheye image rectification, controllable distortion correction, learnable queries, hybrid CNN-Transformer, generative rectification

## TL;DR

The proposed QueryCDR network, utilizing a distortion-aware learnable query mechanism (DLQM) and two controllable modulation modules (CCMB/CAMB), achieves high-quality controllable rectification for fisheye images with various distortion degrees **without retraining** for the first time.

## Background & Motivation

Fisheye cameras are widely utilized in security surveillance and autonomous driving due to their ultra-wide Field of View (FoV). However, the image distortion introduced by fisheye lenses significantly degrades the performance of downstream visual tasks. Existing fisheye rectification methods are categorized into two main paradigms:

**Regression-based methods**: Reconstruct the image after predicting distortion parameters, which require additional annotations and are not end-to-end.

**Generative methods**: Directly generate the rectified image using an encoder-decoder network, which is end-to-end but suffers from poor generalization.

**Limitations of Prior Work**: Existing methods (including DR-GAN, PCN, SimFIR, etc.) can only perform well on distortion degrees similar to those in the training data. **When the distortion degree changes without retraining, the rectification quality degrades significantly.** This is because the models learn fixed spatial mapping relationships during training, failing to generalize to new distortion distributions. Furthermore, collecting fisheye image datasets and retraining models for every distortion degree is extremely costly.

**Why can we not directly apply controllable mechanisms from the image restoration field?**

There are two critical gaps:
- **Difference in Optimization Objectives**: Restoration tasks learn pixel-level detail recovery, whereas fisheye rectification learns spatial-level position mapping. Control mechanisms for restoration that lack spatial positional information cannot effectively control rectification.
- **Difference in Control Conditions**: Fisheye distortion increases from the center to the edges, introducing spatial variation. Single scalar control conditions in restoration tasks (e.g., CFSNet, MM-RealSR) cannot handle such spatially-varying distortion.

**Key Insight**: Design high-dimensional learnable queries containing spatial positional information as control conditions instead of scalar controls. Simultaneously, design specialized modulation blocks to guide the rectification using these control conditions on both CNN and Transformer scales.

**Core Idea**: Project the implicit spatial mapping relationships of different distortion degrees into a low-dimensional latent space of a set of learnable queries. Users can control the rectification output simply by selecting different queries. Interpolation between queries also achieves continuous and smooth distortion rectification.

## Method

### Overall Architecture

QueryCDR consists of three core components: (1) a flow estimation module (inherited from PCN) for coarse warping; (2) a DLQM to extract control conditions from user-specified queries and feed them layer-by-layer into the rectification network; and (3) a U-shaped hierarchical rectification network composed of CCMB and CAMB, which generates the rectified image under the guidance of control conditions.

### Key Designs

1. **Distortion-aware Learnable Query Mechanism (DLQM)**:

   **Function**: Maintains a set of learnable queries representing different distortion degrees to extract position-related control conditions.

   **Mechanism**: Construct a query set $\mathbf{Q}_s = \{Q_i \mid Q_i \in \mathbb{R}^{C_{in} \times H_{in} \times W_{in}}, i=1,...,N\}$, where each query is of the same size as the input image, encoding the spatial position mapping of the corresponding distortion degree. A control extractor (CE) with three layers of $3\times3$ convolutions is first used to extract features:

    $Q_{ex} = \text{CE}(Q_i)$

   Then, control conditions are generated layer-by-layer through multiple fully connected layers:

    $Q_c^l = \text{FC}_2^l(\text{FC}_1^l(Q_c^{l-1}))$

   The output $Q_c^l$ of each layer serves as both the control condition for the rectification block at that layer and the input to the DLQM of the next layer.

   **Key Properties**: Interpolation between queries enables continuous rectification, e.g., $Q_{1.25} = 0.75 Q_1 + 0.25 Q_2$.

   **Design Motivation**: Compared with scalar control conditions, queries of the same size as the input naturally incorporate spatial positional information, allowing them to express the radially increasing distortion distribution from the center to the edges.

2. **Controllable Convolution Modulating Block (CCMB)**:

   **Function**: Implements dynamic feature modulation at the CNN level to preserve local texture details.

   **Mechanism**: Receiving the input feature $F_{in}$ and the control condition $Q_c$, the block first computes the control feature $F_c = F_{in} \otimes Q_c$ (element-wise multiplication) and then predicts the dynamic fusion ratio using a coefficient predictor:

    $\theta = \text{CP}(F_{in}, Q_c)$
    $F_{out} = \theta F_c \oplus (1-\theta) F_{in}$

   **Design Motivation**: Directly using control features or a fixed ratio fusion degrades the rectification quality. The dynamic fusion ratio allows the model to automatically find the optimal balance between preserving original features and applying control features.

3. **Controllable Attention Modulating Block (CAMB)**:

   **Function**: Captures long-range distortion mapping relationships using the global attention mechanism of Transformers.

   **Mechanism**: Design a controllable attention mechanism (CTRL-ATTN), which projects the control features $F_c$ to Query $\mathcal{Q}$, and projects the input features $F_{in}$ to Key $\mathcal{K}$ and Value $\mathcal{V}$:

    $\text{CTRL-ATTN}(\mathcal{Q},\mathcal{K},\mathcal{V}) = \text{softmax}\left(\frac{\mathcal{Q}\mathcal{K}^T}{\sqrt{m}}\right)\mathcal{V}$

   Finally, the output is produced via a residual connection and an FFN:
    $F_a = \text{CTRL-ATTN}(\text{LN}(\mathcal{Q},\mathcal{K},\mathcal{V})) \oplus F_{in}$
    $F_{out} = \text{Conv}_{1\times 1}(\text{FFN}(\text{LN}(F_a)) \oplus F_a)$

   **Design Motivation**: CNNs struggle to capture continuous, non-rigid, long-range distortion patterns in fisheye images. CAMB perceives the spatial mapping relationships in the control conditions via global attention, ensuring global consistency in the rectification.

**Hybrid Architecture**: CCMB is used in the first 3 encoder layers and last 3 decoder layers ($l=\{1,2,3,9,10,11\}$) where feature maps are larger to preserve textures, while CAMB is utilized in the middle layers ($l=\{4,5,6,7,8\}$) with smaller feature maps to capture global dependencies, achieving an optimal configuration of 6C+5A.

### Loss & Training

**Two-stage Training**:

- **Coarse Pre-training**: Trained only on a single distortion degree $d$ using a single query $Q$. The loss is a combination of reconstruction L1 loss and multi-scale loss:
  $$\mathcal{L}_{pre} = \|I_{out}^d - I_{gt}^d\|_1 + \sum_{j=1}^{Z-1}\|S(I_{gt}^d, j) - C(F_{out}^j)\|_1$$

- **Fine-tuning**: Duplicate the pre-trained query weights to all 9 queries, then fine-tune on data spanning 9 distortion degrees:
  $$\mathcal{L}_{fine}^{d_i} = \mathcal{L}_r^{d_i} + \mathcal{L}_m^{d_i}$$

Pre-training uses 40,000 images, fine-tuning uses only 18,000 images (2,000 per distortion), and testing uses 3,600 images. Input size is 256×256. Adam optimizer is used with a learning rate of 1e-4.

## Key Experimental Results

### Main Results

**Comparison of PSNR (dB) across 9 distortion degrees on the COCO Fisheye dataset:**

| Method | d1 | d5 | d9 | Average PSNR | Average SSIM |
|------|-----|-----|-----|---------|---------|
| SC (Traditional) | 10.05 | 11.50 | 9.14 | 10.73 | 0.151 |
| DR-GAN | 15.68 | 18.50 | 17.47 | 17.74 | 0.323 |
| PCN | 14.93 | 18.86 | 18.26 | 17.97 | 0.575 |
| DDA | 16.39 | 20.12 | 18.22 | 18.33 | 0.591 |
| SimFIR | 16.57 | 19.31 | 18.48 | 18.53 | 0.601 |
| **QueryCDR** | **20.01** | **20.72** | **20.53** | **20.32** | **0.676** |

QueryCDR surpasses the previous state-of-the-art SimFIR by **1.79 dB** in average PSNR and **0.075** in SSIM. Even on d5, where previous methods perform best, it still outperforms DDA by 0.60 dB.

### Ablation Study

**Comparison of different control mechanisms (PSNR):**

| Control Mechanism | d1 | d5 | d9 | Average | Description |
|---------|-----|-----|-----|------|------|
| No Control (PCN) | 14.93 | 18.86 | 18.26 | 17.97 | Baseline |
| Scalar Control | 19.52 | 19.89 | 20.15 | 19.78 | +1.81 dB |
| Fixed Query | 20.13 | 20.16 | 19.84 | 19.93 | +1.96 dB |
| Learnable Query (Ours) | 20.01 | 20.72 | 20.53 | **20.32** | **+2.35 dB** |

**Comparison of different modulation types:**

| Modulation Type | PSNR | SSIM | Description |
|---------|------|------|------|
| Directly use $F_c$ | 20.14 | 0.655 | Baseline |
| Fixed 1:1 ratio fusion | 20.17 | 0.658 | +0.03 dB |
| Dynamic mechanism (CCMB) | 20.20 | 0.669 | +0.06 dB |
| Attention mechanism (CAMB) | 20.26 | 0.671 | +0.12 dB |

**Comparison of network architectures:**

| Architecture | FLOPs(G) | Params(M) | PSNR | SSIM |
|------|----------|---------|------|------|
| PCN | 12.305 | 35.637 | 17.97 | 0.575 |
| 11C+0A | 12.736 | 37.701 | 20.20 | 0.669 |
| 6C+5A (Ours) | 12.353 | 43.244 | 20.32 | 0.676 |
| 0C+11A | 15.190 | 51.795 | 20.26 | 0.671 |

### Key Findings

1. **Controllable mechanism is critical**: The progressive performance gain from No Control (17.97) $\rightarrow$ Scalar Control (19.78) $\rightarrow$ Learnable Query (20.32) verifies that higher-dimensional control conditions yield better results.
2. **CCMB+CAMB hybrid architecture is optimal**: 6C+5A achieves the best balance between performance (20.32/0.676) and computational efficiency (12.353G FLOPs).
3. **Requires only a small amount of fine-tuning data**: Pre-training on 40K and fine-tuning on 18K allows generalization to 9 distortion degrees, preventing the need for retraining on each distortion.
4. **Query interpolation enables continuous rectification**: Distortion degrees missing from the training set can be covered via linear interpolation of queries, e.g., $Q_{1.25} = 0.75Q_1 + 0.25Q_2$.
5. **Effective on real fisheye images**: Despite being trained on synthetic data, QueryCDR exhibits robust rectification performance on real-world fisheye datasets.

## Highlights & Insights

- **Novel perspective on using learnable queries as control conditions**: Unlike scalar or one-hot encodings, full-resolution queries naturally encode spatially-varying distortion distribution information.
- **Elegant design of query interpolation for continuous control**: Similar to latent space interpolation in StyleGAN, the query space exhibits excellent smoothness.
- **Pragmatic two-stage training strategy**: First pre-train on single-distortion data (easy to obtain), then fine-tune with a small amount of multi-distortion data (expensive but efficient), significantly reducing data requirements.
- **Dynamic fusion ratio prediction in CCMB**: Adaptively trades off between preserving original features and applying control features, which is more flexible than fixed ratios.

## Limitations & Future Work

- Currently, control still requires users to manually select the query index. The paper mentions that an **automatic control mechanism** should be implemented in the future to automatically estimate the distortion degree of the input image.
- Quantitative evaluation is only conducted on synthetic fisheye datasets (COCO/Places2), whereas only qualitative results are provided for real data.
- The number of queries is fixed to 9, corresponding to 9 discrete distortion degrees; while interpolation is possible, the precision may be limited.
- No comparison with the latest diffusion-based rectification methods.
- No performance evaluation on downstream tasks (e.g., object detection, semantic segmentation) using the rectified images.

## Related Work & Insights

- The flow estimation module proposed in PCN [Yang et al.] is directly adopted in this work, on top of which QueryCDR introduces a controllable mechanism.
- The scalar-controllable restoration concept of CFSNet [Wang et al.] is generalized to high-dimensional query spaces.
- The dynamic fusion concept in CCMB is similar to gating mechanisms in attention mechanisms, which can be transferred to other controllable generation tasks.
- The design in CAMB, which projects control features as Query and original features as Key/Value, is also inspiring for controllable image editing.

## Rating

- Novelty: ⭐⭐⭐⭐ The approach of using learnable queries to control distortion rectification is novel, and query interpolation for continuous control is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 9 distortion degrees plus detailed ablation studies, but lacks quantitative evaluation on real-world data.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, well-detailed method descriptions, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Addresses the generalization pain point in fisheye rectification, showing solid practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] RAW-Adapter: Adapting Pre-trained Visual Model to Camera RAW Images](raw-adapter_adapting_pre-trained_visual_model_to_camera_raw_images.md)
- [\[ICLR 2026\] Lossy Common Information in a Learnable Gray-Wyner Network](../../ICLR2026/signal_comm/lossy_common_information_in_a_learnable_gray-wyner_network.md)
- [\[ICML 2025\] Large Language Model (LLM)-enabled In-context Learning for Wireless Network Optimization](../../ICML2025/signal_comm/large_language_model_llm-enabled_in-context_learning_for_wireless_network_optimi.md)
- [\[ECCV 2024\] PYRA: Parallel Yielding Re-Activation for Training-Inference Efficient Task Adaptation](pyra_parallel_yielding_re-activation_for_training-inference_efficient_task_adapt.md)
- [\[ECCV 2024\] Defect Spectrum: A Granular Look of Large-Scale Defect Datasets with Rich Semantics](defect_spectrum_a_granular_look_of_large-scale_defect_datasets_with_rich_semanti.md)

</div>

<!-- RELATED:END -->
