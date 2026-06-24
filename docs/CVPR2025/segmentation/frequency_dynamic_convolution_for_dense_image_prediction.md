---
title: >-
  [Paper Note] Frequency Dynamic Convolution for Dense Image Prediction
description: >-
  [CVPR 2025][Segmentation][Dynamic Convolution] FDConv redesigns dynamic convolution from a frequency domain perspective. By leveraging Fourier Disjoint Weights (FDW), it constructs frequency-diverse convolutional kernels without increasing parameters. Combined with Kernel Spatial Modulation (KSM) and Frequency Band Modulation (FBM) for fine-grained frequency adaptation, FDConv outperforms existing dynamic convolution methods requiring 65-90M extra parameters while only introd…
tags:
  - "CVPR 2025"
  - "Segmentation"
  - "Dynamic Convolution"
  - "Frequency Domain"
  - "Dense Prediction"
  - "Object Detection"
date: 2026-05-08
content_hash: a0613a5124367f8e
---

# Frequency Dynamic Convolution for Dense Image Prediction

**Conference**: CVPR 2025  
**arXiv**: [2503.18783](https://arxiv.org/abs/2503.18783)  
**Code**: [https://github.com/Linwei-Chen/FDConv](https://github.com/Linwei-Chen/FDConv)  
**Area**: Segmentation  
**Keywords**: Dynamic Convolution, Frequency Domain, Dense Prediction, Segmentation, Object Detection

## TL;DR
FDConv redesigns dynamic convolution from a frequency domain perspective. By leveraging Fourier Disjoint Weights (FDW), it constructs frequency-diverse convolutional kernels without increasing parameters. Combined with Kernel Spatial Modulation (KSM) and Frequency Band Modulation (FBM) for fine-grained frequency adaptation, FDConv outperforms existing dynamic convolution methods requiring 65-90M extra parameters while only introducing 3.6M parameters.

## Background & Motivation

**Background**: Dynamic convolution adaptively selects weights input-dependently by learning multiple sets of parallel weights and linearly combining them using attention mechanisms, showing promising performance on dense prediction tasks such as object detection and segmentation. Representative methods include CondConv, DY-Conv, and ODConv.

**Limitations of Prior Work**: Parallel weights of existing dynamic convolution methods are highly similar in their frequency responses. The authors' analysis reveals that the cosine similarity of the 4 sets of weights in ODConv exceeds 0.88, and t-SNE visualization shows tightly clustered filters. This indicates that despite a 4x increase in parameter count, the learned frequency information is highly redundant, limiting the model's frequency adaptation capability.

**Key Challenge**: Traditional dynamic convolution learns multiple sets of weights in the spatial domain. Without explicit frequency constraints, the optimization process naturally converges to similar frequency responses, leading to high parameter cost but low diversity. Low-frequency information helps suppress noise, while high-frequency information captures boundaries and details; the lack of frequency diversity limits the model’s ability to adaptively extract features of different frequencies.

**Goal**: (1) How to construct convolutional weights with diverse frequency responses without increasing parameters? (2) How to allow the convolution to adaptively adjust its frequency response in both spatial and frequency dimensions?

**Key Insight**: The authors observe that in the Fourier domain, different frequency components correspond to different spatial patterns. If the parameters are grouped by frequency index such that each group contains disjoint frequency components, the weights obtained by applying the inverse Discrete Fourier Transform (iDFT) to these groups must have distinct frequency responses, guaranteed by mathematical properties.

**Core Idea**: Group fixed parameter budgets by disjoint frequency indices in the Fourier domain, construct frequency-diverse convolutional weights via iDFT, and enhance frequency adaptation capability with kernel spatial modulation and frequency band modulation.

## Method

### Overall Architecture
FDConv consists of three core modules: (1) **Fourier Disjoint Weight (FDW)** groups parameters by frequency in the Fourier domain to construct multiple sets of weights with diverse frequency responses; (2) **Kernel Spatial Modulation (KSM)** dynamically modulates the frequency response of each filter element at the kernel-space level; (3) **Frequency Band Modulation (FBM)** decomposes weights into different frequency bands in the frequency domain to achieve spatially-varying frequency modulation. FDConv can be directly integrated into various architectures such as ResNet, ConvNeXt, and Swin Transformer by replacing standard convolutional layers.

### Key Designs

1. **Fourier Disjoint Weight (FDW)**:

    - **Function**: Constructs $n > 10$ sets of weights with diverse frequency responses using a fixed parameter budget of $k \times k \times C_{in} \times C_{out}$, whereas traditional methods require $n$ times the parameters and can only generate a small number ($n < 10$) of similar weights.
    - **Mechanism**: Reshapes the parameters into $\mathbf{P} \in \mathbb{R}^{kC_{in} \times kC_{out}}$, where each parameter corresponds to a Fourier index $(u,v)$. Sorted by the $L_2$ norm of the index from low to high frequencies, they are evenly divided into $n$ disjoint groups. Each group of parameters is transformed into the spatial domain via iDFT, then cropped into $k \times k$ patches and reorganized into standard weight tensors. Since each group only contains Fourier coefficients of specific frequency bands, the transformed weights mathematically must have distinct frequency responses (cosine similarity of 0), which are then linearly combined via attention coefficients.
    - **Design Motivation**: Traditional methods learning weights in the spatial domain cannot guarantee frequency diversity. FDW fundamentally addresses this issue by utilizing the mathematical properties of the Fourier transform: disjoint frequency indices $\to$ distinct frequency responses. The actual number of parameters remains unchanged; it simply performs different frequency slicing on the same set of parameters.

2. **Kernel Spatial Modulation (KSM)**:

    - **Function**: Performs independent dynamic modulation on each filter element within each weight to finely adjust the frequency response.
    - **Mechanism**: Predicts a dense modulation matrix $\alpha \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$ to perform element-wise modulation on the weights. It contains two branches: the **local channel branch** uses a lightweight 1D convolution to capture local channel information and predict the full dense modulation matrix; the **global channel branch** uses a fully connected layer to capture global information and predict sparse modulation values across three dimensions (input channels, output channels, and kernel space). The two branches are fused to obtain the final modulation matrix.
    - **Design Motivation**: Weight-level mixing in FDW (Eq.1) is too coarse to independently adjust the frequency response of each $k \times k$ filter. KSM provides fine-grained element-wise control. Replacing fully connected layers with 1D convolutions significantly reduces the parameter overhead of predicting dense matrices.

3. **Frequency Band Modulation (FBM)**:

    - **Function**: Decomposes convolutional weights into different frequency bands and independently modulates each band at different spatial locations, achieving spatially-varying frequency adaptation.
    - **Mechanism**: Decomposes the frequency response of the weights into $B$ frequency bands (default 4, divided by octaves: $\{0, 1/16, 1/8, 1/4, 1/2\}$) using a binary mask $\mathcal{M}_b$ in the frequency domain. It performs convolution (element-wise multiplication) in the frequency domain to avoid the infinite support hazard of ideal filters in the spatial domain. For the output of each frequency band, a standard convolution and a sigmoid function are used to predict a spatial modulation map $\mathbf{A}_b \in \mathbb{R}^{h \times w}$, leading to the final output $\mathbf{Y} = \sum_{b=0}^{B-1} \mathbf{A}_b \odot \mathbf{Y}_b$. By duality, decomposing kernel frequency bands is equivalent to decomposing feature frequency bands.
    - **Design Motivation**: Both FDW and KSM are spatially invariant (weights are shared across the entire feature map). However, in natural images, different spatial regions require different frequency processing—backgrounds need high-frequency noise suppression, whereas boundaries require high-frequency detail preservation. FBM allows the frequency response to dynamically vary across spatial locations, achieving true content-adaptive frequency modulation.

### Loss & Training
As a plug-and-play module, FDConv follows the standard training strategies of downstream tasks. The number of weights $n$ is set to 64 by default. Training follows the settings of the respective original frameworks (e.g., training for 12 epochs using a 1x schedule for Mask R-CNN).

## Key Experimental Results

### Main Results

| Model/Framework | Method | Extra Params | AP_box | AP_mask |
|-----------|------|---------|--------|---------|
| Faster R-CNN/R50 | Baseline | - | 37.2 | - |
| Faster R-CNN/R50 | +CondConv (8×) | +90.0M | 38.1 | - |
| Faster R-CNN/R50 | +ODConv (4×) | +65.1M | 39.2 | - |
| Faster R-CNN/R50 | **+FDConv** | **+3.6M** | **39.4** | - |
| Mask R-CNN/R50 | Baseline | - | 39.6 | 36.4 |
| Mask R-CNN/R50 | +KW (4×) | +76.5M | 42.4 | 38.9 |
| Mask R-CNN/R50 | **+FDConv** | **+3.6M** | **42.4** | 38.6 |
| UPerNet/R50 | Baseline | - | mIoU 40.7 | - |
| UPerNet/R50 | +ODConv (4×) | +65M | mIoU 43.3 | - |
| UPerNet/R50 | **+FDConv** | **+4M** | **mIoU 43.8** | - |

### Ablation Study

| Architecture/Method | Extra Params | AP_box | AP_mask |
|-----------|---------|--------|---------|
| ConvNeXt-T + KW | +4M | 44.8 | 40.6 |
| **ConvNeXt-T + FDConv** | **+3M** | **45.2** | **40.8** |
| Swin-T + FDConv | +3M | 44.5 | 40.5 |
| Mask2Former-R50 | - | mIoU 79.4 | - |
| **Mask2Former-R50 + FDConv** | - | **mIoU 80.4** | - |
| MaskDINO-Swin-L† + FDConv | - | mIoU 57.2 (+0.5) | - |

### Key Findings
- **Extremely High Parameter Efficiency**: FDConv achieves or exceeds the performance of CondConv (+90M), ODConv (+65.1M), and KW (+76.5M) while only adding 3.6M parameters, improving parameter efficiency by approximately 20 times.
- **Frequency Diversity is Key**: The cosine similarity of the 4 sets of weights in ODConv is >0.88, whereas for FDConv it is 0, which directly translates to superior representation capacity.
- **FBM visualization** shows that high-frequency modulation values are concentrated on object boundaries, while low-frequency modulation values are concentrated inside objects, which is intuitive and confirms the effectiveness of spatially-varying frequency modulation.
- **FDConv exhibits strong cross-architecture generalization**, seamlessly integrating into both CNN (ResNet, ConvNeXt) and Transformer (Swin) architectures.

## Highlights & Insights
- The core idea of **"creating diversity with the same parameters"** is extremely elegant—instead of adding parameters, it performs different frequency slicing on the same set of parameters in the Fourier domain, with diversity mathematically guaranteed. This concept can be extended to any scenario requiring diverse basis functions.
- **Clever Application of the Convolution Theorem**: Performing convolution in the frequency domain avoids the infinite support issue of ideal filters in the spatial domain, while invoking duality to prove that decomposing kernel frequency bands is equivalent to decomposing feature frequency bands, providing implementation flexibility.
- **FBM's spatially-varying frequency modulation** bridges the gap between dynamic convolution (spatially invariant weights) and deformable convolution (spatially-varying sampling).

## Limitations & Future Work
- FBM introduces extra frequency-domain computations (FFT and iFFT), increasing FLOPs by 1.8G, which may not be negligible in real-time inference scenarios.
- The number of frequency bands $B=4$ and frequency division thresholds are manually set; adaptive learning could be considered.
- Validated only on vision tasks; application in other frequency-sensitive areas such as audio and signal processing could be explored.
- The combination with deformable convolution is worth exploring—FDConv addresses frequency adaptation but with fixed sampling locations.

## Related Work & Insights
- **vs ODConv**: ODConv learns 4 sets of weights in the spatial domain and modulates them using channel/filter/spatial 3D attention, but the frequency responses of the weights are highly similar. FDConv approaches from the frequency domain, achieving better diversity and performance with fewer parameters.
- **vs KW**: KW reduces parameters by decomposing weights into shareable small units, but fundamentally remains in the spatial domain. FDConv performs grouping in the frequency domain with mathematical guarantees of diversity. While KW (4×) requires 76.5M parameters to achieve its performance, FDConv matches it with only 3.6M.
- **vs FADC**: FADC adjusts dilation rates according to feature frequency characteristics, whereas FDConv directly performs band decomposition and spatial modulation on the convolutional kernels, offering finer granularity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reinterpreting dynamic convolution from the frequency domain is a completely fresh perspective. The design of using Fourier disjoint grouping in FDW to guarantee diversity is highly ingenious.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple tasks including detection, instance segmentation, and semantic segmentation, validated across diverse architectures with in-depth analysis.
- Writing Quality: ⭐⭐⭐⭐ Frequency analysis visualizations (Fig.1, Fig.5) are intuitive and convincing, and the methodology is clearly articulated.
- Value: ⭐⭐⭐⭐⭐ A significant breakthrough in the field of dynamic convolution. The 20x parameter efficiency improvement holds immense practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DeCLIP: Decoupled Learning for Open-Vocabulary Dense Perception](declip_decoupled_learning_for_open-vocabulary_dense_perception.md)
- [\[CVPR 2025\] OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels](overlock_an_overview-first-look-closely-next_convnet_with_context-mixing_dynamic.md)
- [\[ICCV 2025\] Dynamic Dictionary Learning for Remote Sensing Image Segmentation](../../ICCV2025/segmentation/dynamic_dictionary_learning_for_remote_sensing_image_segmentation.md)
- [\[CVPR 2025\] HFP-SAM: Hierarchical Frequency Prompted SAM for Efficient Marine Animal Segmentation](hfp-sam_hierarchical_frequency_prompted_sam_for_efficient_marine_animal_segmenta.md)
- [\[CVPR 2025\] Exploring CLIP's Dense Knowledge for Weakly Supervised Semantic Segmentation](exploring_clips_dense_knowledge_for_weakly_supervised_semantic_segmentation.md)

</div>

<!-- RELATED:END -->
