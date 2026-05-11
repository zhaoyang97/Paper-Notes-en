---
title: >-
  [Paper Note] GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule
description: >-
  [ICCV 2025][Medical Imaging][echocardiography segmentation] This paper proposes GDKVM, an echocardiography video segmentation architecture based on linear key-value association and the gated delta rule…
tags:
  - "ICCV 2025"
  - "Medical Imaging"
  - "echocardiography segmentation"
  - "linear attention"
  - "key-value memory"
  - "gated delta rule"
  - "video segmentation"
date: 2026-05-08
content_hash: fcad391b99cca4a0
---

# GDKVM: Echocardiography Video Segmentation via Spatiotemporal Key-Value Memory with Gated Delta Rule

**Conference**: ICCV 2025
**arXiv**: [2512.10252](https://arxiv.org/abs/2512.10252)
**Code**: [https://github.com/wangrui2025/GDKVM](https://github.com/wangrui2025/GDKVM)
**Area**: Medical Imaging
**Keywords**: echocardiography segmentation, linear attention, key-value memory, gated delta rule, video segmentation

## TL;DR

This paper proposes GDKVM, an echocardiography video segmentation architecture based on linear key-value association and the gated delta rule, achieving state-of-the-art performance on CAMUS and EchoNet-Dynamic through efficient memory management and multi-scale feature fusion while maintaining real-time inference speed.

## Background & Motivation

Accurate segmentation of echocardiography videos is critical for quantitative cardiac function analysis. However, the task poses multiple challenges:

**Poor image quality**: Ultrasound images suffer from severe speckle noise, low contrast, blurred tissue structures, and incomplete boundaries.

**Large non-rigid deformation**: The heart undergoes drastic shape and scale changes within a single cardiac cycle (systole vs. diastole).

**Limitations of existing methods**:
   - CNNs (e.g., U-Net) have limited local receptive fields and cannot directly model inter-frame temporal dependencies.
   - Vision Transformers offer global receptive fields but incur quadratic computational complexity.
   - RNNs/ConvLSTMs suffer from error propagation and high computational overhead.
   - Spatiotemporal memory networks (XMem, Cutie) rely on fully annotated reference frames and lack cardiac cycle recognition.

There is a need for a model that can efficiently integrate global temporal context while preserving local spatial precision.

## Method

### Overall Architecture

GDKVM consists of three core modules: Linear Key-Value Association (LKVA) for linear-complexity temporal modeling across frames; Gated Delta Rule (GDR) for controlling memory updates and forgetting; and Key-Pixel Feature Fusion (KPFF) for integrating multi-scale features. A key encoder based on ResNet-50 extracts features, while a value encoder encodes raw frames and the previous prediction mask. Online segmentation is performed via a recurrent state $S_t$.

### Key Designs

1. **Linear Key-Value Association (LKVA)**: Reduces the complexity of conventional softmax attention from $O(t^2 C_d)$ to $O(tC_v C_k)$ by replacing the exponential kernel with a kernel approximation: $\exp(K_i^T Q_t) \rightarrow \phi(K_i)^T \phi(Q_t)$, reformulating the matching as a linear RNN. The hidden state $S_t = \sum V_i \phi(K_i)^T \in \mathbb{R}^{C_v \times C_k}$ is a fixed-size 2D matrix that accumulates contextual information frame by frame without explicitly constructing an attention matrix. This design is particularly suited to modeling the continuous and periodic motion of cardiac structures.

2. **Gated Delta Rule (GDR)**: Addresses the information "blurring" problem in pure linear attention, where all historical information is accumulated with equal weight. Two data-dependent matrices projected from the previous state $S_{t-1}$ are designed:

    - **$\beta_t$ (write strength)**: Controls the learning intensity of new information. The old association corresponding to the current frame key is first removed via $V_t^{old} = S_{t-1} K_t$, and then new and old values are interpolated using $\beta_t$. $\beta_t$ increases during rapid cardiac motion or at frames with clear boundaries to reinforce learning from the current frame.
    - **$\alpha_t$ (decay factor)**: Controls the decay of past memory to prevent memory saturation. The update rule is: $S_t = S_{t-1}(\alpha_t(I - \beta_t K_t K_t^T)) + \beta_t V_t K_t^T$. This allows the model to release memory capacity at anomalous rhythm frames while retaining long-term important information.

3. **Key-Pixel Feature Fusion (KPFF)**: Fuses three scales of features to enhance robustness against noise and artifacts:

    - **Local key features $F_K$**: Local semantics extracted via convolution.
    - **Global key features $F_{Global}$**: Global context obtained through global average pooling and expansion.
    - **Pixel-level features $F_{Pix}$**: High-frequency detail information.
    - Fusion via a gating mechanism: $G = \sigma(\text{Conv}_{gate}(F_K + F_{Global}))$, $F_{fused} = G \cdot (F_K + F_{Global}) + (1-G) \cdot F_{Pix}$. When local features are unreliable due to noise, global features provide a stable complement; pixel features prevent the loss of fine-grained information such as valve motion.

### Loss & Training

- Loss function: Cross-entropy + Soft Dice Loss, combined with equal weights.
- Clinical scenario simulation: Ground-truth labels are not used during prediction; end-diastolic and end-systolic frames are predicted first, then the loss is computed.
- Training: Single RTX 3090, 1500 iterations, AdamW with lr=$10^{-4}$, batch size=10.
- Data augmentation: gamma enhancement, random scaling/rotation/contrast adjustment (each with $p=0.5$).
- Gradient clipping $\lambda=3$ for stable training.
- Resolution: $256\times256$ for CAMUS, $128\times128$ for EchoNet-Dynamic; 10 frames uniformly sampled per video.

## Key Experimental Results

### Main Results

Comparison on CAMUS and EchoNet-Dynamic:

| Method | CAMUS mDice ↑ | CAMUS mIoU ↑ | CAMUS HD ↓ | CAMUS ASD ↓ | Echo mDice ↑ | Echo mIoU ↑ |
|--------|--------------|--------------|------------|-------------|--------------|-------------|
| XMem++ | 89.38 | 85.81 | 4.03 | 4.87 | 87.51 | 83.57 |
| Cutie | 91.09 | 87.97 | 3.89 | 3.74 | 88.96 | 85.63 |
| VideoMamba | 91.96 | 89.04 | 3.48 | 3.31 | 90.22 | 87.03 |
| PKEchoNet | 93.49 | 90.95 | 3.42 | 2.93 | 92.60 | 89.89 |
| DSA | 94.25 | 91.80 | 3.27 | 2.37 | 92.91 | 90.26 |
| MemSAM | 93.63 | 90.97 | 3.47 | 2.60 | 92.71 | 89.90 |
| **GDKVM** | **95.11** | **92.97** | **3.05** | **1.98** | **93.46** | **90.86** |

Clinical metric LVEF (CAMUS):

| Method | corr ↑ | bias ± std (%) |
|--------|--------|----------------|
| DSA | 0.891 | 0.86 ± 13.4 |
| MemSAM | 0.878 | -0.89 ± 12.3 |
| SimLVSeg | 0.895 | 1.83 ± 13.8 |
| **GDKVM** | **0.904** | **-0.19 ± 11.3** |

### Ablation Study

Contribution of each module (CAMUS):

| LKVA | GDR | KPFF | mDice | mIoU | HD | ASD |
|------|-----|------|-------|------|----|-----|
| ✓ | - | - | 93.10 | 90.46 | 3.65 | 2.85 |
| ✓ | ✓ | - | 94.49 | 92.11 | 3.21 | 2.19 |
| ✓ | - | ✓ | 93.30 | 90.78 | 3.55 | 2.74 |
| ✓ | ✓ | ✓ | **95.11** | **92.97** | **3.05** | **1.98** |

Internal ablation of GDR (state update strategies):

| Strategy | mDice | Inference Time (ms) |
|----------|-------|---------------------|
| Baseline (simple accumulation) | 93.30 | 151.61 |
| Full replacement (no $\beta_t$) | 74.68 | 155.09 |
| $\beta_t$ only (no $\alpha_t$) | 94.57 | 158.77 |
| $\alpha_t$ only (no $\beta_t$) | 94.26 | 156.90 |
| **Full GDR ($\alpha_t$ + $\beta_t$)** | **95.11** | 160.62 |

### Key Findings

- GDKVM achieves 37 FPS with 35.2M parameters, balancing accuracy and speed.
- Both gating mechanisms in GDR ($\alpha_t$ decay + $\beta_t$ write strength) are indispensable: completely replacing old information (no gating) causes Dice to drop sharply to 74.68.
- KPFF is critical for handling the scale variation and spatial transformation characteristic of ultrasound imaging.
- The clinical metric LVEF achieves a correlation of 0.904 with a bias of only $-0.19\%$, demonstrating significant clinical utility.
- Gradient analysis of $\alpha_t$ and $\beta_t$ reveals a coordinated regulation pattern (block and diagonal structures), indicating that the gating mechanism has learned meaningful memory management strategies.

## Highlights & Insights

- **Linear-complexity temporal modeling**: LKVA reduces inter-frame matching from $O(t^2)$ to $O(t)$, enabling real-time processing of long cardiac video sequences.
- **Biologically inspired memory management**: The dual-gating design of GDR resembles selective human attention—maintaining mild updates during stable frames and making aggressive adjustments at critical frames (structural transitions, high noise).
- **Clinically oriented design**: The model simulates clinical scenarios by not using ground-truth labels during inference; LVEF estimation aligns closely with clinical gold standards.
- With only 35.2M parameters, the model can be trained and deployed on a single RTX 3090, presenting an extremely low deployment barrier.

## Limitations & Future Work

- The intermediate state matrix is non-square, limiting block-level parallelism and hardware acceleration.
- The model occasionally over-relies on its own predictions, leading to contour deviations on difficult samples (failure cases in Figure 10).
- Validation is limited to echocardiography; generalization to other medical video segmentation scenarios (e.g., endoscopy, CT sequences) remains unexplored.
- Long-sequence consistency under fully supervised and semi-supervised benchmarks has not been evaluated.
- Adaptive boundary definition or correction mechanisms could be explored to handle artifacts in difficult samples.

## Related Work & Insights

- Compared to STM-based methods such as XMem/XMem++, GDKVM does not require fully annotated reference frames and is computationally more efficient.
- The GDR design is inspired by the delta rule in linear RNNs such as DeltaNet, but extends it with data-dependent dual-gating control.
- KPFF compensates for the spatial reasoning limitations of linear models by integrating local features from xLSTM-style designs with pixel-level information.
- The proposed methodology is generalizable: the combination of linear key-value association and gated memory management is applicable to any medical video analysis task requiring real-time processing of long sequences.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of the gated delta rule and linear key-value association is novel in the context of medical video segmentation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, clinical metrics, detailed ablations, and visualization analyses are provided.
- **Writing Quality**: ⭐⭐⭐⭐ The method derivation is clear, with a natural progression from softmax attention to its linear approximation.
- **Value**: ⭐⭐⭐⭐ Real-time, high-accuracy ultrasound segmentation has direct clinical application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] T-Gated Adapter: A Lightweight Temporal Adapter for Vision-Language Medical Segmentation](../../CVPR2026/medical_imaging/t-gated_adapter_a_lightweight_temporal_adapter_for_vision-language_medical_segme.md)
- [\[ICCV 2025\] PVChat: Personalized Video Chat with One-Shot Learning](pvchat_personalized_video_chat_with_one-shot_learning.md)
- [\[ICCV 2025\] SciVid: Cross-Domain Evaluation of Video Models in Scientific Applications](scivid_cross-domain_evaluation_of_video_models_in_scientific_applications.md)
- [\[ICCV 2025\] ProGait: A Multi-Purpose Video Dataset and Benchmark for Transfemoral Prosthesis Users](progait_a_multi-purpose_video_dataset_and_benchmark_for_transfemoral_prosthesis_.md)
- [\[NeurIPS 2025\] Multimodal Disease Progression Modeling via Spatiotemporal Disentanglement and Multiscale Alignment](../../NeurIPS2025/medical_imaging/multimodal_disease_progression_modeling_via_spatiotemporal_disentanglement_and_m.md)

</div>

<!-- RELATED:END -->
