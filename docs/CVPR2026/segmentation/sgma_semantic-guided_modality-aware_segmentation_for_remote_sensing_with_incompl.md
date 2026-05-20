---
title: >-
  [Paper Note] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data
description: >-
  [CVPR 2026][Segmentation][Incomplete multimodal semantic segmentation] This paper proposes the SGMA framework, which constructs global semantic prototypes via a Semantic-Guided Fusion (SGF) module for adaptive cross-moda…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Incomplete multimodal semantic segmentation"
  - "remote sensing"
  - "modality imbalance"
  - "semantic prototypes"
  - "adaptive fusion"
date: 2026-05-08
content_hash: aab09b4faf628c2c
---

# SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data

**Conference**: CVPR 2026
**arXiv**: [2603.02505](https://arxiv.org/abs/2603.02505)  
**Code**: N/A  
**Area**: Semantic Segmentation
**Keywords**: Incomplete multimodal semantic segmentation, remote sensing, modality imbalance, semantic prototypes, adaptive fusion

## TL;DR

This paper proposes the SGMA framework, which constructs global semantic prototypes via a Semantic-Guided Fusion (SGF) module for adaptive cross-modal fusion, and dynamically increases the training frequency of fragile modalities through a Modality-Aware Sampling (MAS) module. The framework addresses three core challenges in incomplete multimodal semantic segmentation for remote sensing: modality imbalance, large intra-class variance, and cross-modal heterogeneity.

## Background & Motivation

### 1. State of the Field

Multimodal semantic segmentation (MSS) integrates information from multiple sensors—RGB, NIR, DSM, SAR—to achieve more accurate scene understanding in remote sensing Earth observation. In practice, sensor failures or incomplete coverage frequently cause modality absence, motivating research on incomplete multimodal semantic segmentation (IMSS).

### 2. Limitations of Prior Work

IMSS faces three major challenges:

- **Modality imbalance**: Dominant modalities (e.g., RGB) overwhelm the learning of fragile modalities (e.g., DSM/NIR/SAR) during training.
- **Large intra-class variance**: Instances of the same semantic category vary significantly in scale, orientation, and shape (e.g., buildings of different sizes).
- **Cross-modal heterogeneity**: Different modalities produce conflicting responses for the same semantic region (e.g., rooftops and ground surfaces appear similar in RGB but differ greatly in DSM elevation).

### 3. Root Cause

Existing methods rely on contrastive learning or joint optimization, but forced alignment discards modality-specific information (over-alignment), and imbalanced training biases the model toward robust modalities. Modality dropout cannot sufficiently train fragile modalities, and MAE-based approaches focus on low-level reconstruction rather than high-level semantics.

### 4. Paper Goals

To design a unified framework that maintains robust performance under arbitrary modality-missing scenarios while explicitly addressing modality imbalance, intra-class variance, and cross-modal heterogeneity.

### 5. Starting Point

Class-level semantic prototypes serve as cross-modal semantic anchors, circumventing the drawbacks of pixel-wise contrastive alignment. Attention weights quantify modality robustness to guide adaptive fusion and sampling.

### 6. Core Idea

Multimodal features are compressed into global semantic prototypes (one vector per class), which serve as queries to adaptively aggregate per-modality features via attention. The attention weights simultaneously reflect modality reliability, driving the MAS module to increase training frequency for fragile modalities and achieve balanced learning.

## Method

### Overall Architecture

SGMA establishes a dual-branch optimization structure containing two plug-and-play modules:

1. **Semantic-Guided Fusion (SGF)**: Reduces intra-class variance and reconciles cross-modal heterogeneity through semantic prototype-guided attention fusion.
2. **Modality-Aware Sampling (MAS)**: Dynamically adjusts modality sampling probabilities using robustness scores from SGF to alleviate modality imbalance.

All modalities share a single weight-tied encoder $F$ that independently extracts features at 4 scales. During training, SGF and MAS generate segmentation predictions in parallel for joint optimization; at inference, only the SGF branch is retained.

### Key Designs

#### 1. Modality-specific Projector (MP)

- **Function**: Maps per-modality features into a unified semantic space.
- **Mechanism**: Employs three depthwise separable convolutions of different kernel sizes (11×11, 7×7, 3×3) to capture multi-scale context, followed by a 1×1 convolution for semantic projection.
- **Design Motivation**: Multi-scale receptive fields allow features from different modalities to retain their respective scale information within the unified space.

#### 2. Class-aware Semantic Filter (CSF) + Global Semantic Prototypes

- **Function**: Compresses per-modality features from $C_i$ channels to $K$ channels (number of classes) to extract class-level semantic representations; generates global semantic prototypes $p_{se}^{i,k} \in \mathbb{R}^{C}$ via matrix multiplication.
- **Mechanism**: $\{p_{se}^{i,k}\}_{k=1}^K = [c_m^i] \otimes [f_{m \to se}^i]^T$, computing a matrix product between compact and semantic features across all modalities to obtain a global prototype vector per class.
- **Design Motivation**: Global prototypes provide a full receptive field, anchoring dispersed pixel representations to class centers and naturally reducing intra-class variance.

#### 3. Spatial Perceptron (SP)

- **Function**: Uses global semantic prototypes as queries to retrieve multimodal features at each spatial location via multi-head attention.
- **Mechanism**: $a_{se}^{i,k} = \text{MHA}_{SP}(q_i, k_i, v_i)$, where $q_i$ is the prototype broadcast to every spatial position and $k_i = v_i$ are the rearranged multimodal features.
- **Design Motivation**: Enables each pixel to selectively aggregate the most relevant cross-modal information according to its semantic class, enhancing class consistency.

#### 4. Robustness Perceptron (RP)

- **Function**: Performs a second multi-head attention pass with semantically guided features as queries, producing both the fused feature $f_{SGF}^i$ and modality robustness maps $\{r_m^i\}_{m \in \mathcal{M}}$.
- **Mechanism**: Attention weights reflect the alignment of each modality with the semantic prototypes—higher alignment indicates greater reliability for a given spatial location and semantic class.
- **Design Motivation**: Provides class-dependent and scale-dependent modality reliability estimates; DSM receives high weights for structural classes such as buildings, NIR for vegetation, with adaptive adjustment across scales.

#### 5. Modality-Aware Sampling (MAS)

- **Function**: Inverts robustness scores into sampling probabilities, sampling one modality per training iteration for independent training.
- **Mechanism**: $\hat{r}_m^i = \frac{1/r_m^i}{\sum_{m'} 1/r_{m'}^i}$, assigning higher sampling probability to modalities with lower robustness scores.
- **Design Motivation**: Directly addresses modality imbalance by providing fragile modalities with more training opportunities, preventing them from being dominated by robust modalities. This is equivalent to applying SoftMin to pre-softmax values, making it computationally efficient.

### Loss & Training

- SGF and MAS each produce segmentation predictions and are supervised independently with cross-entropy loss: $\mathcal{L}_{IMSS} = \lambda_{SGF} \mathcal{L}_{SGF} + \lambda_{MAS} \mathcal{L}_{MAS}$
- $\lambda_{SGF} = 2$, $\lambda_{MAS} = 1$
- Modality dropout is applied during training to simulate missing-modality scenarios; all modality combinations participate in training.
- AdamW optimizer, lr = 6e-5, polynomial decay (power 0.9), 200 epochs, 10-epoch warm-up.

## Key Experimental Results

### Main Results

**Datasets**: ISPRS Potsdam (RGB+DSM+NIR, 5 classes), DFC2023 (RGB+DSM+SAR, building), DELIVER (RGB+Depth+Event+LiDAR, 25 classes)

**Table 1: ISPRS Dataset mIoU (%) — PVT-v2-b2 backbone**

| Method | R | D | N | R+D | R+N | D+N | R+D+N | Average | Last-1 |
|--------|------|------|------|------|------|------|-------|---------|--------|
| MuSS | 40.21 | 17.13 | 1.36 | 83.75 | 57.71 | 31.52 | 86.50 | 45.45 | 1.36 |
| M3L | 30.72 | 10.41 | 20.99 | 81.31 | 78.54 | 72.76 | 84.07 | 54.12 | 10.41 |
| IMLT | 69.57 | 38.78 | 69.82 | 80.03 | 81.29 | 67.82 | 85.12 | 70.35 | 38.78 |
| MAGIC | 81.39 | 34.34 | 46.97 | 83.27 | 77.99 | 63.30 | 84.75 | 67.43 | 34.34 |
| **SGMA** | **83.51** | **57.05** | **76.06** | **86.62** | **84.25** | **82.56** | **86.84** | **79.55** | **57.05** |

Average mIoU gain: +9.20%; Last-1 (worst single modality) gain: **+18.26%**.

**Table 2: DFC2023 Dataset mIoU (%) — PVT-v2-b2 backbone**

| Method | R | D | S | R+D | R+S | D+S | R+D+S | Average | Last-1 |
|--------|------|------|------|------|------|------|-------|---------|--------|
| IMLT | 90.54 | 53.73 | 32.53 | 90.81 | 90.61 | 49.98 | 91.12 | 71.33 | 32.53 |
| MAGIC | 88.98 | 65.96 | 37.51 | 89.20 | 83.29 | 43.75 | 81.93 | 70.09 | 37.51 |
| **SGMA** | **90.84** | **76.70** | **53.13** | **91.95** | **90.98** | **77.47** | **92.29** | **81.91** | **53.13** |

Average mIoU gain: +7.66%; Last-1 gain: **+15.54%** (SAR single modality).

### Ablation Study

**Table 3: Progressive Ablation of SGF and MAS — ISPRS (PVT-v2-b2)**

| Variant | SGF | MAS | Average mIoU | Last-1 mIoU |
|---------|-----|-----|-------------|-------------|
| (a) Baseline | ✗ | ✗ | 46.51 | 2.61 |
| (b) SGF only | ✓ | ✗ | 49.13 | 7.01 |
| (c) SGF+MAS | ✓ | ✓ | **79.55** | **57.05** |

- SGF alone yields limited improvement over the baseline (+2.62%), primarily because fragile modalities remain insufficiently trained without MAS.
- Adding MAS produces a dramatic improvement: Average +**30.42%**, Last-1 +**50.04%**, validating the critical role of MAS in training fragile modalities.

### Key Findings

1. **Substantial gains for fragile modalities**: Across all datasets, fragile modalities (DSM/SAR/Event/LiDAR) achieve the largest single-modality improvements (+10–18%); fragile-plus-fragile combinations even surpass the single robust modality.
2. **Consistent improvement with more modalities**: SGMA is the only method that monotonically improves as more modalities are added; baseline methods can degrade when additional modalities are introduced (MAGIC drops 3.4% when NIR is added).
3. **Cross-backbone generalizability**: Equivalent gains are observed on ResNet-50 (ISPRS Average +10.21%), confirming the plug-and-play property.
4. **Computational efficiency**: Only 9.47 GFLOPs and 4.79M additional parameters are introduced (1.1% and 1.7% relative to the backbone), far lower than MAGIC (98.11G / 22.29M).
5. **Interpretable robustness maps**: Visualizations reveal adaptive shifts in per-scale modality contributions—shallow layers show balanced contributions across RGB/DSM/NIR, while deep layers are RGB-dominant (0.66).

## Highlights & Insights

1. The idea of using **semantic prototypes as cross-modal anchors** is elegant: rather than performing pixel-wise alignment, the method builds semantic bridges through class-level prototypes, simultaneously reducing intra-class variance and avoiding over-alignment.
2. **Dual use of attention weights**: RP attention weights serve concurrently for weighted fusion and robustness estimation—one mechanism addressing two problems.
3. **Efficient SoftMin implementation**: Inverting post-softmax values is equivalent to applying SoftMin to the original pre-softmax logits, eliminating the need to store pre-softmax activations.
4. **Decoupled training via MAS**: Rather than forcing all modalities to compete within a shared loss, MAS independently samples and trains fragile modalities, addressing imbalance at its root.
5. **Success of fragile-plus-fragile combinations**: All-fragile combinations such as DSM+SAR and Event+LiDAR yield meaningful segmentation results, demonstrating maximal exploitation of complementary information.

## Limitations & Future Work

1. **Limited interpretability**: The authors acknowledge the absence of an explicit mechanism for quantifying modality learning dynamics; robustness scores are visualizable but have limited semantic interpretability.
2. **Temporal multimodal dynamics not modeled**: Dynamic changes in modality reliability due to temporal variation in remote sensing (e.g., seasonal changes, post-disaster scenes) are not addressed.
3. **Prototype quality instability in early training**: Global semantic prototypes may be unstable during early training, particularly for rare classes; the warm-up strategy may be insufficient.
4. **MAS absent at inference**: MAS is used only during training; robustness under completely unseen fragile-modality combinations at inference is limited.
5. **Sensitivity to class count $K$**: CSF compresses features to $K$ channels, yet the performance difference between coarse-grained (e.g., DELIVER with 25 classes) and fine-grained category settings has not been thoroughly analyzed.

## Related Work & Insights

- **IMLT [5]**: The first IMSS method for remote sensing, combining contrastive learning with masked pretraining; forced alignment, however, discards modality-specific information.
- **MAGIC [65]**: Partitions modalities into robust/fragile groups with joint optimization and cosine alignment; the group assignment is static.
- **M3L [33]**: Employs random modality dropout with learnable parameters to preserve modality representations, but cannot sufficiently train fragile modalities.
- **Insights**: The semantic prototype paradigm is extensible to other multimodal tasks (e.g., CT+MRI+PET missing-modality scenarios in medical imaging); the robustness-guided sampling strategy of MAS is applicable to any multimodal learning problem exhibiting data imbalance.

## Rating

⭐⭐⭐⭐ A systematic and practical IMSS framework. The combination of semantic prototypes and robustness-guided sampling is elegantly designed, yielding consistent and significant improvements across three datasets and two backbones. The substantial gains for fragile modalities (Last-1 +18%) carry real deployment value. The plug-and-play design with negligible computational overhead enhances engineering feasibility. The primary limitation is that novelty lies mainly in the combination of components rather than in any single novel contribution, and extensions to temporal dynamics and more complex missing-modality patterns remain unexplored.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [\[AAAI 2026\] S5: Scalable Semi-Supervised Semantic Segmentation in Remote Sensing](../../AAAI2026/segmentation/s5_scalable_semi-supervised_semantic_segmentation_in_remote_sensing.md)
- [\[CVPR 2026\] RDNet: Region Proportion-Aware Dynamic Adaptive Salient Object Detection Network in Optical Remote Sensing Images](rdnet_region_proportionaware_dynamic_adaptive_sali.md)
- [\[CVPR 2026\] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training](data_warmup_complexity-aware_curricula_for_efficient_diffusion_training.md)
- [\[CVPR 2026\] SemLayer: Semantic-aware Generative Segmentation and Layer Construction for Abstract Icons](semlayer_semantic-aware_generative_segmentation_and_layer_construction_for_abstr.md)

</div>

<!-- RELATED:END -->
