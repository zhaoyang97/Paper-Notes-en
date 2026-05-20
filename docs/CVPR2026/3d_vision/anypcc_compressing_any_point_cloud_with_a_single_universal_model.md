---
title: >-
  [Paper Note] AnyPcc: Compressing Any Point Cloud with a Single Universal Model
description: >-
  [CVPR 2026][3D Vision][point cloud compression] AnyPcc proposes a Universal Context Model (UCM) that integrates dual-granularity spatial and channel priors, combined with an Instance-Adaptive Fine-Tuning (IAFT) strategy…
tags:
  - "CVPR 2026"
  - "3D Vision"
  - "point cloud compression"
  - "universal context model"
  - "instance-adaptive fine-tuning"
  - "occupancy code"
  - "lossless/lossy compression"
date: 2026-05-08
content_hash: 127ce85398d88e7e
---

# AnyPcc: Compressing Any Point Cloud with a Single Universal Model

**Conference**: CVPR 2026
**arXiv**: [2510.20331](https://arxiv.org/abs/2510.20331)  
**Code**: [anypcc.github.io](https://anypcc.github.io)  
**Area**: 3D Vision
**Keywords**: point cloud compression, universal context model, instance-adaptive fine-tuning, occupancy code, lossless/lossy compression

## TL;DR

AnyPcc proposes a Universal Context Model (UCM) that integrates dual-granularity spatial and channel priors, combined with an Instance-Adaptive Fine-Tuning (IAFT) strategy, to achieve state-of-the-art point cloud geometry compression across 15 diverse datasets using a single model, yielding approximately 12% bitrate reduction over G-PCC v23.

## Background & Motivation

**Pressing demand for point cloud compression**: The widespread adoption of 3D applications such as autonomous driving and VR has made point clouds a standard 3D data format, where efficient geometry compression is critical for reducing storage and transmission costs.

**Poor generalization of existing methods**: Learning-based methods perform well on standard benchmarks but degrade sharply in real-world scenarios, struggling with varying density (sparse LiDAR vs. dense reconstruction) and out-of-distribution (OOD) data.

**Density sensitivity of context models**: Spatial-prior methods (e.g., Unicorn) are unreliable for sparse scenes, while channel-wise methods (e.g., RENO) are robust to sparse data but neglect coarse-grained structural information; each category has its own blind spot.

**OOD data as the core bottleneck**: Even models claiming universality, such as Unicorn-U, fail on novel point cloud types including medical scans, 3D Gaussian Splats, and Dust3R/VGGT reconstructions, due to the lack of an effective distribution adaptation mechanism.

**Implicit compression is too slow**: INR-based methods train a network from scratch per instance, offering strong generalization but unacceptable encoding times for practical deployment.

**Lack of a unified framework**: Existing methods target either dense objects or sparse LiDAR; no single model simultaneously handles all point cloud types while supporting both lossless and lossy compression.

## Method

### Overall Architecture

AnyPcc adopts a multi-scale octree representation and formulates point cloud geometry compression as a scale-by-scale occupancy code probability prediction problem. The framework comprises three components: (1) a Universal Context Model (UCM) for robust cross-density context modeling; (2) Instance-Adaptive Fine-Tuning (IAFT) to address OOD generalization via instance-level adaptation; and (3) a probability threshold mechanism that seamlessly extends lossless compression to lossy scenarios. During encoding, UCM predicts occupancy code probability distributions coarse-to-fine at each scale, which are then fed to an arithmetic coder; for OOD data, IAFT rapidly fine-tunes a small subset of parameters and encodes the weight delta into the bitstream.

### Key Design 1: Universal Context Model (UCM) — Dual-Granularity Spatial-Channel Context

**Function**: A recursively parameter-shared context model that, at each scale, draws context from three sources: parent-scale occupancy codes, fine-grained channel priors, and coarse-grained spatial priors.

**Mechanism**:
- **Spatial Grouping (SG)**: A 3D checkerboard pattern partitions the current-scale occupancy codes into two groups (even/odd coordinate sums), forming a two-step autoregressive process. When predicting the second group, decoded neighbor information from the first group is available.
- **Channel Grouping (CG)**: Each 8-bit occupancy code is split into two 4-bit sub-symbols (low 4 bits + high 4 bits), enabling cascaded prediction — the low 4 bits are predicted first, then the high 4 bits are predicted conditioned on them.
- **Cooperative Aggregation**: When predicting the second spatial group, sparse convolution aggregates features from the decoded first-group codes, which are then fused with the original context via a fusion network, enabling deep interaction between coarse and fine-grained information.

**Design Motivation**: The authors formally prove two theorems: (1) channel-wise autoregression and spatial sub-voxel autoregression are information-theoretically equivalent (Theorem 1); (2) sparse convolution in occupancy code space has an effective receptive field equivalent to a convolution with twice the kernel width in the fine-grained voxel space (Theorem 2), which is especially critical for sparse data. Ablation studies show that using channel grouping alone (as in RENO) yields negligible improvement and must be combined with spatial priors to be effective.

### Key Design 2: Instance-Adaptive Fine-Tuning (IAFT) — Bridging Explicit and Implicit Compression

**Function**: For each input point cloud instance, IAFT rapidly fine-tunes a small subset of UCM parameters and encodes the resulting weight delta into the bitstream.

**Mechanism**:
- UCM parameters are partitioned into a frozen subset $\Theta_{\text{frozen}}$ (feature extraction and sparse convolution, comprising the vast majority) and a tunable subset $\Theta_{\text{tune}}$ (linear layers in the Prediction Head only).
- During encoding, a single forward pass caches the frozen component outputs; only $\Theta_{\text{tune}}$ is then iteratively optimized over the cached features (~200 iterations, converging in seconds).
- The optimized weights are uniformly scalar-quantized and encoded via DeepCABAC into the bitstream.

**Design Motivation**: INR methods are prohibitively slow to train from scratch, while fixed pretrained models cannot adapt to OOD data. IAFT combines the strengths of both: the pretrained model's strong priors accelerate convergence, and fine-tuning only the final layer keeps overhead minimal. Experiments on the GS dataset show that weight transmission adds only 0.319 bpp while saving 1.883 bpp in geometry coding, yielding a substantial net gain.

### Key Design 3: Unified Lossless-Lossy Compression

**Function**: A probability threshold mechanism extends the lossless framework to lossy compression scenarios.

**Mechanism**:
- For sparse LiDAR point clouds: the finest $n$ scales are simply omitted during encoding.
- For dense point clouds: the encoder transmits only the target number of points $k$; the decoder reconstructs geometry by selecting the $k$ highest-probability locations according to the model's predictions.
- A single model supports both lossless and lossy compression without additional training.

## Loss & Training

- **Pretraining stage**: UCM is trained on a large-scale mixed dataset (KITTI, Ford, 8iVFB, MVUB, ScanNet, GausPcc-1K, Thuman, etc.) using the negative log-likelihood (cross-entropy) of occupancy codes as the loss.
- **IAFT stage**: The instance-level fine-tuning loss is negative log-likelihood $+ \lambda_{L1} \cdot \|\Theta_{\text{tune}}\|_1$, where the $L_1$ regularization promotes weight sparsity to reduce transmission overhead.
- **Two variants**: Ours (category-specific models trained separately) and Ours-U (a single unified model with shared weights applied to all test sets).

## Key Experimental Results

### Table 1: Lossless Compression Performance on 15 Datasets (bpp↓)

| Dataset | Difficulty | OOD | RENO | SparsePCGC | OctAttention | TopNet | GPCC v23 | **Ours** | **Ours-U** |
|--------|------|-----|------|------------|--------------|--------|----------|----------|------------|
| 8iVFB | E | ✗ | 0.70 | 0.57 | 0.68 | 0.59 | 0.76 | **0.54** | 0.57 |
| MVUB | E | ✗ | 1.00 | 0.69 | 0.76 | 0.69 | 0.94 | **0.67** | 0.75 |
| Thuman | E | ✗ | 1.64 | 1.70 | 2.31 | 2.20 | 2.00 | **1.58** | 1.64 |
| KITTI | E | ✗ | 7.06 | 6.80 | 7.21 | 6.85 | 8.19 | **6.18** | 6.45 |
| GS | M | ✗ | 13.89 | 15.82 | 11.31 | 10.95 | 14.46 | **11.65** | 11.74 |
| VGGT | M | ✓ | 8.24 | 7.84 | 8.22 | 7.83 | 7.33 | 7.30 | **7.06** |
| S3DIS | M | ✓ | 13.06 | 11.88 | 11.52 | 10.84 | 10.66 | 10.93 | **10.79** |
| CS | H | ✓ | 3.94 | 4.94 | 3.40 | 3.21 | 3.23 | 3.18 | **3.08** |
| **CR-Gain vs GPCC** | | | 2.96% | 2.07% | 1.32% | -4.04% | 0% | **-11.93%** | -10.75% |

### Table 2: UCM Ablation Study (Contribution of Each Component)

| Configuration | Spatial Conv (SC) | Spatial Grouping (SG) | Channel Grouping (CG) | CR-Gain | Params (M) |
|------|:-----------:|:-----------:|:-----------:|---------|----------|
| Baseline | ✗ | ✗ | ✗ | 0.00% | 5.15 |
| SG only | ✗ | ✓ | ✗ | -6.56% | 5.68 |
| CG only | ✗ | ✗ | ✓ | +0.13% | 5.15 |
| SC+SG | ✓ | ✓ | ✗ | -7.74% | 9.78 |
| SC+CG | ✓ | ✗ | ✓ | -5.33% | 7.19 |
| **Full (Ours)** | ✓ | ✓ | ✓ | **-9.88%** | 9.77 |

Key finding: Using CG alone (as in RENO) yields virtually no benefit (+0.13%); it must act in concert with SC/SG to be effective.

### Table 3: Bitrate Breakdown of IAFT on GS Dataset

| Metric | UCM only | UCM + IAFT |
|------|--------|------------|
| Entropy coding bpp | 13.307 | 11.424 |
| Weight transmission bpp | 0 | 0.319 |
| **Total bpp** | 13.307 | **11.743** |

Weight transmission adds only 0.319 bpp while geometry coding saves 1.883 bpp, yielding a net reduction of 1.564 bpp.

## Highlights & Insights

1. **Theory-practice coherence**: Two formal theorems rigorously establish the equivalence of channel-spatial modeling and the receptive field advantage, directly informing the UCM design — a notably principled contribution.
2. **Elegant fusion of explicit and implicit compression**: IAFT transplants the instance-adaptive capability of INR methods onto a pretrained model; encoding takes seconds rather than tens of minutes, making the approach practically deployable.
3. **Universality of a single model**: Ours-U applies a single set of weights across 15 datasets spanning vastly different characteristics (sparse LiDAR, dense human bodies, 3DGS, noisy point clouds), even outperforming specialized models on OOD data.
4. **Comprehensive evaluation**: A benchmark comprising 15 datasets — including standard sets and extreme scenarios (noise/dropout/deformation) — far exceeds the evaluation scale common in this field.
5. **Competitive decoding efficiency**: Decoding time is comparable to the fastest baseline RENO (0.46s vs. 0.23s), while encoding time can be flexibly controlled between 0.44s and 2.84s by adjusting the number of IAFT iterations.

## Limitations & Future Work

1. **Encoding time overhead**: Enabling IAFT increases encoding time from 0.44s to ~12s (800 iterations), which may be insufficient for latency-sensitive applications such as live streaming.
2. **Model size**: The full model contains 68.39M parameters; although Ours-U has only 9.77M, this offers no clear advantage over RENO (9.03M), and IAFT requires backpropagation on the encoder side.
3. **Naive lossy compression strategy**: The lossy scheme for dense point clouds (selecting the $k$ highest-probability locations) lacks rate-distortion optimization and may not represent the optimal R-D trade-off.
4. **Training data dependency**: UCM's generalization still relies on the diversity of training data; performance on entirely unseen point cloud types (e.g., molecular structures) remains to be validated.

## Related Work & Insights

- **Unicorn/Unicorn-U**: The closest competitor, which attempts a unified architecture but still relies on a non-unified attention–convolution hybrid design with poor OOD generalization. AnyPcc addresses this thoroughly via a pure-convolution UCM combined with IAFT.
- **RENO**: A pioneer of channel-wise occupancy code prediction; however, ablation results show that channel grouping alone is nearly ineffective and must be combined with spatial priors.
- **INR-based compression (NeRF/3DGS related)**: IAFT can be viewed as a lightweight variant of INR compression — fine-tuning only the final layer rather than the entire network — suggesting that parameter-efficient fine-tuning holds considerable promise in the compression domain.
- **DeepCABAC**: A context-adaptive binary arithmetic coder used for weight encoding, which is a key technology for efficiently transmitting network parameters as data; it merits broader adoption in other model-in-the-loop scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-granularity spatial-channel context fusion and the IAFT explicit-implicit hybrid strategy are both field-first contributions, further strengthened by formal theoretical proofs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 15 datasets, 6 baselines, comprehensive ablations, lossless and lossy evaluation, and detailed time/parameter analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rigorous theoretical derivations, informative figures and tables, and well-motivated problem formulation.
- **Value**: ⭐⭐⭐⭐ — Universal compression with a single model addresses a critical practical deployment need, and the IAFT strategy is broadly transferable to other 3D data compression tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Deformation-based In-Context Learning for Point Cloud Understanding](deformation-based_in-context_learning_for_point_cloud_understanding.md)
- [\[CVPR 2026\] APC: Transferable and Efficient Adversarial Point Counterattack for Robust 3D Point Cloud Recognition](apc_adversarial_point_counterattack.md)
- [\[CVPR 2026\] Adapting Point Cloud Analysis via Multimodal Bayesian Distribution Learning](adapting_point_cloud_analysis_via_multimodal_bayesian_distribution_learning.md)
- [\[CVPR 2026\] PCSTracker: Long-Term Scene Flow Estimation for Point Cloud Sequences](pcstracker_long-term_scene_flow_estimation_for_point_cloud_sequences.md)
- [\[CVPR 2026\] QD-PCQA: Quality-Aware Domain Adaptation for Point Cloud Quality Assessment](qd-pcqa_quality-aware_domain_adaptation_for_point_cloud_quality_assessment.md)

</div>

<!-- RELATED:END -->
