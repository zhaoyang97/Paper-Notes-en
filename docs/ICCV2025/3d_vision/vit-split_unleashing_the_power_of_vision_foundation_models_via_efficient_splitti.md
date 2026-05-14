---
title: >-
  [Paper Note] ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads
description: >-
  [ICCV 2025][3D Vision][Vision Foundation Models] Grounded in the key observation that VFM layers can be partitioned into low-level feature extractors and high-level task adapters, this paper proposes ViT-Split…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "Vision Foundation Models"
  - "Efficient Adapters"
  - "DINOv2"
  - "Hierarchical Features"
  - "Semantic Segmentation"
date: 2026-05-08
content_hash: a3586c83737b9449
---

# ViT-Split: Unleashing the Power of Vision Foundation Models via Efficient Splitting Heads

**Conference**: ICCV 2025
**arXiv**: [2506.03433](https://arxiv.org/abs/2506.03433)
**Code**: [jackyfl.github.io/vitsplit.github.io](https://jackyfl.github.io/vitsplit.github.io/)
**Area**: 3D Vision
**Keywords**: Vision Foundation Models, Efficient Adapters, DINOv2, Hierarchical Features, Semantic Segmentation

## TL;DR

Grounded in the key observation that VFM layers can be partitioned into low-level feature extractors and high-level task adapters, this paper proposes ViT-Split, which freezes the VFM backbone and introduces a task head (replicating the last $K_t$ layers) and a prior head (a lightweight CNN aggregating multi-scale prior features). On ADE20K, ViT-Split achieves 58.2 mIoU (DINOv2-L) with only a linear head, offers 4× faster training, and requires only 1/4–1/5 of the trainable parameters compared to conventional adapters.

## Background & Motivation

### Problem Definition

How to efficiently transfer pre-trained knowledge from vision foundation models (VFMs, e.g., DINOv2) to downstream tasks (segmentation, detection, depth estimation, VQA, etc.), while substantially reducing training overhead without sacrificing—or even surpassing—state-of-the-art performance?

### Limitations of Prior Work

**Inefficiency of VFM adapters (ViT-Adapter, ViT-CoMer)**:
- Dual-branch architectures (CNN + ViT) require gradient backpropagation through all layers, with computational and memory costs scaling linearly with model size.
- All components (VFM + CNN + adapter + task head) require fine-tuning, resulting in large parameter counts.
- The pre-trained prior features of the VFM are altered, preventing full exploitation of pre-trained knowledge.

**Limitations of PEFT methods**:
- VPT (prompt tuning), AdaptFormer (adapter tuning), and LoRA (low-rank fine-tuning) still insert learnable parameters at every layer, triggering gradient backpropagation through early layers.
- Without low-level features (as provided by the CNN branch in VFM adapters), performance is typically on par with or slightly below full fine-tuning.
- Pre-trained prior features of the VFM are underutilized.

**Low efficiency in multi-task deployment**: Conventional VFM adapters maintain a complete pipeline (VFM + CNN + adapter + head) per task, incurring severe memory and computational redundancy.

### Core Motivation

**Key Observation**: CKA (Centered Kernel Alignment) analysis reveals that layers in VFMs such as DINOv2 can be explicitly divided into two groups—**early layers learn low-level features** (textures, edges, etc., similar across tasks) and **later layers learn task-specific features** (segmentation focuses on semantics, detection on corners, with large cross-task divergence). This implies:
1. An additional CNN branch for low-level features is unnecessary—the VFM's own early layers already capture them.
2. Only the later layers need fine-tuning for downstream adaptation—the early layers can be frozen.
3. Prior features from all VFM layers should be leveraged rather than overwritten by fine-tuning.

## Method

### Overall Architecture

The entire VFM backbone is frozen. Two "split heads" are introduced: (1) **Task head**: a copy of the last $K_t$ VFM layers, receiving the output of layer $L-K_t$ and learning task-specific features; (2) **Prior head**: a lightweight CNN that uniformly samples features from $K_p$ frozen VFM layers and aggregates them. The outputs of both heads are fused via a fusion net and passed to the downstream task head.

### Key Designs

#### 1. **VFM Layer Grouping Observation & Task Head**

- **Function**: Copies the last few VFM layers as a trainable task-specific adapter.
- **Mechanism**:

  **Observation validation**: CKA analysis and feature visualization reveal that:
  - Early layers (e.g., L1–L6 of DINOv2-S): cross-task features (pre-training/segmentation/detection) are similar, focusing on textures and edges.
  - Later layers (L7–L12): features diverge across tasks—segmentation biases toward semantic information, detection toward object corners.

  **Task head design**: The output $f_{L-K_t}$ of VFM layer $L-K_t$ is fed into the copied last $K_t$ layers:
  $$f_t = g_{\theta_t}(f_{L-K_t})$$
  The class token is then discarded and the output is reshaped into a spatial feature map $f_t' \in \mathbb{R}^{h \times w \times D}$.

  $K_t$ is a key hyperparameter: it controls the trade-off between adapter capacity and training efficiency. For segmentation, gains diminish as $K_t$ increases, so a smaller value suffices.

- **Design Motivation**: Since early layers are shared across tasks, fine-tuning them (and their associated gradient backpropagation) is unnecessary. Copying the later layers as the task head provides a naturally good initialization while preserving the original VFM prior features intact. This also implies that large segmentation heads (Mask2Former, UperNet) may be unnecessary.

#### 2. **Prior Head: Multi-Scale Prior Feature Aggregation**

- **Function**: Extracts multi-scale prior features from the frozen VFM to enhance task-specific features.
- **Mechanism**:

  **Layer selection—uniform sampling**: $K_p$ layers are uniformly sampled from $L$ layers with interval $\delta = \frac{L-b-1}{K_p-1}$:
  $$\mathcal{S} = \{b + \text{round}(i \cdot \delta) | i = 0, ..., K_p-1\}$$
  where $b=2$ or $b=3$ (skipping the noisiest features in the very first layers).

  **Prior head architecture**: Features from sampled layers are concatenated as $f_p \in \mathbb{R}^{h \times w \times (K_p \cdot D)}$ and processed by two CNN layers:
  $$f_p' = g_{\theta_p}(f_p)$$
  The CNN consists of a $1 \times 1$ convolution (channel compression) followed by a $3 \times 3$ deformable convolution (enhancing low-level features and modeling geometric transformations).

- **Design Motivation**: Uniform sampling reduces redundancy among highly similar adjacent-layer features while increasing diversity. VFM prior features, learned from large-scale data, encode rich general-purpose knowledge—directly leveraging them enhances task-specific features and mitigates overfitting in the task head. Deformable convolutions are better suited for irregular spatial structures than standard convolutions.

#### 3. **Fusion Net & Multi-Task Adaptation**

- **Function**: Fuses task-specific and prior features for adaptation to different downstream tasks.
- **Mechanism**:

  Task and prior feature maps are concatenated (preserving more information):
  $$f_o = g_{\theta_f}([f_p'; f_t'])$$
  The fusion net shares the same structure as the prior head ($1 \times 1$ + $3 \times 3$ deformable convolution).

  **Task-specific transformations**:
  - Segmentation: $4\times$ upsampling (two transposed convolution layers).
  - Detection: generates 4 scales ($4\times, 2\times, 1\times, 0.5\times$) to match MaskRCNN input.
  - VQA: reshaped into sequence dimension $(h \cdot w) \times D$ and fed into the LLM decoder.

- **Design Motivation**: Concatenation outperforms addition and attention-based fusion (verified by ablation), as it retains all information. A single frozen VFM backbone can be shared across multiple tasks, with each task training only its own task head + prior head + fusion net, substantially reducing multi-task deployment overhead.

### Loss & Training

- **Segmentation**: Standard cross-entropy loss, AdamW (lr=2e-4, wd=1e-2), batch size 16, 40k iterations.
- **Detection**: Standard MaskRCNN losses, AdamW (lr=1e-4, wd=5e-2), 12 epochs.
- Task head learning rate is scaled by 0.1× (to avoid rapid corruption of the initialization).
- The backbone is completely frozen; no gradients are backpropagated into the early VFM layers.

## Key Experimental Results

### Main Results

**ADE20K Semantic Segmentation** (512×512):

| Method | Head | Trainable Params (M) | mIoU |
|--------|------|----------------------|------|
| ViT-Adapter-S | UperNet | 57.6 | 46.2 |
| ViT-CoMer-S | UperNet | 61.4 | 46.5 |
| DINOv2-S (Linear) | Linear | 22.1 | 49.6 |
| **ViT-Split-S** | **Linear** | **10.2** | **51.6** |
| ViT-Adapter-B | UperNet | 133.9 | 48.8 |
| DINOv2-B (UperNet) | UperNet | 120.7 | 54.8 |
| **ViT-Split-B** | **Linear** | **40.5** | **55.7** |
| ViT-Adapter-L | UperNet | 363.8 | 53.4 |
| DINOv2-L (UperNet) | UperNet | 341.2 | 57.1 |
| **ViT-Split-L** | **Linear** | **88.6** | **58.2** |

**Cityscapes Semantic Segmentation** (896×896):

| Method | Head | Trainable Params (M) | mIoU (SS/MS) |
|--------|------|----------------------|--------------|
| ViT-Adapter-L | Mask2Former | 571 | 84.9/85.8 |
| DINOv2-L | Linear | 312.9 | 83.5/84.3 |
| **ViT-Split-L** | **Linear** | **164.1** | **85.8/86.7** |

### Ablation Study

**VQA Performance (LLaVA-1.5 + ViT-Split vs. Baseline)**:

| Benchmark | LLaVA-1.5 | + ViT-Split | Change |
|-----------|-----------|-------------|--------|
| VQAv2 | 78.5 | 78.2 | -0.3 |
| LLaVA-Wild | 65.4 | 71.1 | **+5.7** |
| SciQA-IMG | 66.8 | 70.4 | **+3.6** |
| POPE (adv) | 84.2 | 86.1 | **+1.9** |
| MMBench | 64.3 | 66.4 | **+2.1** |

**Training Efficiency Comparison**:

| Method | Training Time (10k iters) | Ratio vs. ViT-Split |
|--------|--------------------------|---------------------|
| ViT-Adapter-S | ~38 min | ~4× slower |
| ViT-CoMer-S | ~37 min | ~3.9× slower |
| **ViT-Split-S** | **~9 min 25 s** | **baseline** |

**Fusion Strategy Ablation**:

| Fusion Strategy | mIoU |
|----------------|------|
| Addition | Lower |
| Concatenation | **Best** |

### Key Findings

1. **A simple linear head suffices to outperform complex segmentation heads**: ViT-Split-L (Linear, 88.6M params) surpasses ViT-Adapter-L (UperNet, 363.8M params) by 4.8 mIoU.
2. **Significant parameter efficiency**: Trainable parameters are only 1/4–1/5 of conventional adapters, with only 1/4 of the training iterations required.
3. **4× training speedup**: Achieved by eliminating gradient backpropagation through early layers and removing the CNN branch.
4. **Value of VFM prior features**: The prior head yields an average improvement of ~2% mIoU over the no-prior baseline, confirming the utility of frozen multi-scale prior features.
5. **Multi-task generality**: Effective across segmentation, detection, depth estimation, and VQA; integrating ViT-Split into LLaVA-1.5 yields a +5.7-point gain on LLaVA-Wild.

## Highlights & Insights

1. **Concise and powerful core observation**: VFM layers = feature extractor (shared across tasks) + task adapter (task-specific), validated through both CKA analysis and feature visualization.
2. **"Less is more" philosophy**: Freezing the majority of parameters and training only a small subset outperforms full fine-tuning by preventing prior knowledge from being overwritten.
3. **Elegant design**: The entire method reduces to "copy the last few layers + add a CNN to aggregate priors," yet substantially outperforms complex multi-branch adapter architectures.
4. **Efficient multi-task inference**: A single frozen VFM is shared; each task requires only its own lightweight heads, making the approach GPU-memory friendly.
5. **Deep understanding of VFMs**: The paper reveals the intuitive divergence between DINOv2 segmentation features (semantic-focused) and detection features (corner-focused).

## Limitations & Future Work

1. **VFM dependency**: The approach works best with DINOv2 (where layer grouping is most pronounced); layer grouping is less distinct for other VFMs (e.g., CLIP).
2. **Detection task gap**: Due to the large discrepancy between detection and DINOv2 pre-training objectives, ViT-Split only matches ViT-CoMer on COCO detection.
3. **Manual tuning of $K_t$ and $K_p$**: Optimal values vary across tasks and model scales.
4. **Resolution limitations**: Inherits ViT's patch size (14×14 or 16×16), leading to reduced efficiency at high input resolutions.
5. **Video/temporal extension unexplored**: The current framework supports only single-frame input.

## Related Work & Insights

- **Relation to ViT-Adapter**: ViT-Adapter employs a CNN branch for local features with cross-attention interaction; ViT-Split demonstrates that the VFM's own early layers already capture local features, rendering an additional CNN branch unnecessary.
- **Comparison with LoRA/VPT**: PEFT methods modify every layer without leveraging low-level features; ViT-Split modifies only high-level layers while utilizing prior features from all layers.
- **Encoder-decoder nature of DINOv2**: Self-supervised pre-training (reconstructing masked patches) naturally induces a MAE-like hierarchical "encode-then-decode" layer structure.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The core observation is insightful and the design is clean and effective, though the adapter paradigm itself is not entirely new.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers four tasks (segmentation/detection/depth/VQA) with comprehensive ablations and training efficiency analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Observations and method descriptions are clear, with highly informative figures.
- **Value**: ⭐⭐⭐⭐⭐ — Provides an efficient paradigm for VFM adaptation with strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DAViD: Data-efficient and Accurate Vision Models from Synthetic Data](david_data-efficient_and_accurate_vision_models_from_synthetic_data.md)
- [\[AAAI 2026\] Parameter-Free Fine-tuning via Redundancy Elimination for Vision Foundation Models](../../AAAI2026/3d_vision/parameter-free_fine-tuning_via_redundancy_elimination_for_vision_foundation_mode.md)
- [\[ICLR 2026\] GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](../../ICLR2026/3d_vision/giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)
- [\[AAAI 2026\] VGGT-DP: Generalizable Robot Control via Vision Foundation Models](../../AAAI2026/3d_vision/vggt-dp_generalizable_robot_control_via_vision_foundation_models.md)
- [\[AAAI 2026\] FoundationSLAM: Unleashing the Potential of Deep Foundation Models in End-to-End Dense Visual SLAM](../../AAAI2026/3d_vision/foundationslam_unleashing_the_power_of_depth_foundation_models_for.md)

</div>

<!-- RELATED:END -->
