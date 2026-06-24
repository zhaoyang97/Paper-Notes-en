---
title: >-
  [Paper Note] TaCo: A Benchmark for Lossless and Lossy Codecs of Heterogeneous Tactile Data
description: >-
  [ICLR 2026][Robotics][Tactile data compression] This paper proposes TaCo—the first comprehensive benchmark for tactile data codecs. It systematically evaluates lossless and lossy compression across 5 heterogeneous tactile datasets, 30 codecs, and 4 types of downstream tasks. The authors train TaCo-LL (lossless) and TaCo-L (lossy), the first codecs driven purely by tactile data, which achieve new SOTA results across all tasks.
tags:
  - "ICLR 2026"
  - "Robotics"
  - "Tactile data compression"
  - "Lossless/Lossy codecs"
  - "Heterogeneous tactile sensors"
  - "Dexterous grasping"
  - "Neural codecs"
date: 2026-05-08
content_hash: 1d82df1bff9ca0f0
---

# TaCo: A Benchmark for Lossless and Lossy Codecs of Heterogeneous Tactile Data

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=1PYXFkS6Hy](https://openreview.net/forum?id=1PYXFkS6Hy)  
**Code**: https://github.com/sjtu-medialab/RwkvCompress (TaCo-L is based on LALIC and reuses this repository's setup)  
**Area**: Robotics / Embodied AI / Tactile Perception / Data Compression / Benchmark  
**Keywords**: Tactile data compression, Lossless/Lossy codecs, Heterogeneous tactile sensors, Dexterous grasping, Neural codecs

## TL;DR
This paper proposes TaCo—the first comprehensive benchmark for tactile data codecs. It systematically evaluates lossless and lossy compression across 5 heterogeneous tactile datasets, 30 codecs, and 4 types of downstream tasks. The authors train TaCo-LL (lossless) and TaCo-L (lossy), the first codecs driven purely by tactile data, which achieve new SOTA results across all tasks.

## Background & Motivation

**Background**: Tactile perception is a key modality for embodied intelligence, providing fine-grained force and deformation information for dexterous manipulation and environmental interaction. However, as the resolution and sampling rates of tactile sensors increase, the raw data volume has exploded. For example, the raw bitrate of a GelSight video (640×480×30fps×24bit/pixel) reaches approximately 200 Mbps. In scenarios such as remote teleoperation, real-time tactile feedback for dexterous hands, and large-scale data storage for robot model training, bandwidth and storage act as hard constraints, making tactile data compression essential.

**Limitations of Prior Work**: Although the need for compression is well-recognized, existing work is highly fragmented. One category involves classic signal processing (dimensionality reduction, wavelet transforms, thresholding + delta encoding, etc.), which typically only exploits simple signal sparsity or quantization strategies, lacks rigorous compression metrics, and generalizes poorly to narrow scenarios. Another category—data-driven neural codecs—has surpassed traditional methods in image and video domains but has **never been trained or systematically evaluated on tactile data**. The field lacks unified datasets, standard metrics, and baseline models.

**Key Challenge**: Tactile data is inherently **heterogeneous**. Visuo-tactile sensors (e.g., GelSight, DIGIT) output RGB images/videos of elastomer surface deformations via internal cameras. In contrast, mechanical tactile sensors (e.g., Paxini) output sequences of 3D force vectors from multiple contact points, which are structurally distinct from images. This heterogeneity, combined with complex spatio-temporal redundancy, makes it difficult for neural encoders pre-trained on one type of tactile data to transfer to another, leaving the trade-off between compression efficiency and downstream task performance unexplored.

**Goal**: The paper decomposes the problem of tactile data compression into three quantifiable sub-problems: (1) defining a unified representation to make off-the-shelf codecs interoperable across heterogeneous data; (2) evaluating the performance of 30 existing codecs across 5 datasets and 4 task types; and (3) determining the performance upper bound for codecs trained purely on tactile data.

**Key Insight**: The authors observe that most tactile signals can be **naturally converted into image-like formats**, allowing for the direct reuse of mature image/video codecs with adjustable rate-distortion. This approach, largely unexplored in the tactile domain, enables the unification of heterogeneous data under a single evaluation framework.

**Core Idea**: By unifying heterogeneous tactile data into image representations, the authors build a 3D evaluation matrix (5 datasets × 30 codecs × 4 tasks). They also perform the first end-to-end training of lossless (TaCo-LL) and lossy (TaCo-L) codecs on tactile data to reveal the key trade-offs between compression efficiency and task performance.

## Method

### Overall Architecture
TaCo is not just a new model but a system comprising a benchmark and two data-driven baselines. It consists of three layers: the **bottom layer** is a unified representation for heterogeneous tactile data (mapping both visuo-tactile and mechanical tactile data to three-channel images); the **middle layer** is an evaluation pool of 30 codecs (17 off-the-shelf and 13 neural codecs, including 14 lossless and 16 lossy); the **top layer** includes 4 types of downstream tasks (lossless storage, human visualization, semantic classification, and dexterous grasping). Within this framework, the authors retrain the SOTA lossless image compressor DualComp-I and the SOTA lossy image compressor LALIC on tactile data to derive **TaCo-LL (Lossless)** and **TaCo-L (Lossy)**.

For data partitioning, 70% of Touch and Go and ObjectFolder are used to train TaCo-LL/TaCo-L. The remaining 30%, along with the entirety of SSVTP, YCB-Slide, and ObjTac, are used for evaluation. These latter three datasets represent "unseen distributions" for TaCo, testing generalization across sensors and datasets. Training was conducted on 2 A100 GPUs.

### Key Designs

**1. Unified Image-based Representation for Heterogeneous Tactile Data**

To address structural inconsistencies, sensors are processed based on their principles. Visuo-tactile sensors (GelSight/DIGIT) output RGB sequences directly compatible with image/video codecs. For mechanical tactile sensors (ObjTac with $N=60$ contact points), the authors map **each 3D force vector to an RGB pixel** and stack these readings over time $T$ to generate a $T \times 60$ "image." This allows standard and neural codecs to be evaluated fairly within the same coordinate system.

**2. Three-dimensional Evaluation Matrix**

The benchmark covers:
- **Datasets**: 5 datasets across sensor types (vision/mechanical), resolutions (120×160 to 640×480), and scales (250K+ frames).
- **Codecs**: Standard lossless (gzip, PNG, JPEG-XL, etc.), standard lossy (JPEG-XL, VTM/HM intra and SCC modes, VVenC), and neural codecs (P2LLM/DualComp-I, LLM-based LMIC, ELIC/LALIC, DCVC).
- **Tasks**: Lossless compression for storage; lossy compression for human visualization (PSNR/MS-SSIM), machine semantics (Top-1 accuracy for material/object classification), and robotic control (dexterous grasping success rate).

**3. TaCo-LL: First Purely Tactile-trained Lossless Codec**

TaCo-LL retrains DualComp-I on tactile data. During tokenization, the input is divided into $16 \times 16 \times 3$ patches and flattened via raster-scan. For visuo-tactile data, sub-pixels $(R_1, G_1, B_1, \dots)$ are expanded sequentially; for force signals, components are treated as color channels $(x_1, y_1, z_1, \dots)$. The network $f_a$ autoregressively predicts the distribution $p(x_i \mid x_{<i})$, and an arithmetic encoder generates the bitstream. The loss is the entropy:

$$L = \mathbb{E}\big[-\log_2 p_{\hat x}(x_i\mid x_{<i})\big]$$

**4. TaCo-L: First Purely Tactile-trained Lossy Codec**

TaCo-L retrains LALIC using a VAE architecture. Inputs are randomly cropped or zero-padded to $256 \times 256$. Since the data is already in three channels, **no tokenization is required**. The analysis transform $g_a$ maps signals to a latent representation $y$, which is quantized to $\hat y$. A hyper-prior branch $h_a/h_s$ generates side information $z$ to estimate the density model of $\hat y$, and the synthesis transform $g_s$ reconstructs $\hat x$. The training uses a rate-distortion loss:

$$L = \lambda \cdot D(x,\hat x) + \mathbb{E}\big[-\log_2 p_{\hat y\mid\hat z}(\hat y\mid\hat z)\big]$$

## Key Experimental Results

### Main Results

**Lossless Compression (bits/Byte, lower is better; raw is 8)**: TaCo-LL-96M achieved the best performance across all 5 datasets.

| Dataset | gzip | JPEG-XL | DualComp-I | TaCo-LL-96M | Comp. Ratio |
|--------|------|---------|-----------|-------------|--------|
| Touch and Go | 2.298 | 0.739 | 0.948 | **0.447** | 18× |
| ObjectFolder | 3.969 | 3.657 | 3.126 | **2.709** | 3× |
| SSVTP | 2.234 | 1.478 | 1.442 | **1.066** | 8× |
| ObjTac | 0.571 | 0.382 | 0.540 | **0.360** | 22× |
| YCB-Slide | 2.185 | 1.431 | 1.388 | **1.073** | 8× |

**Lossy Compression (BD-Rate %, HM-Intra as anchor, lower is better)**: TaCo-L was superior across all datasets, reaching −61.8% on Touch and Go.

| Dataset | ELIC | LALIC | VTM-Intra | TaCo-L |
|--------|------|-------|-----------|--------|
| Touch and Go | −40.2% | −51.6% | −21.7% | **−61.8%** |
| ObjectFolder | 0.6% | 0.2% | −19.7% | **−24.3%** |
| SSVTP | −5.8% | 4.3% | −16.0% | **−19.2%** |
| YCB-Slide | −9.2% | −4.6% | −24.4% | **−27.4%** |
| ObjTac | 44.5% | 32.8% | −22.0% | **−27.0%** |

### Downstream Task Experiments

**Classification (top-1 acc)**: Performance remains close to "Raw" even at extremely low bitrates.

| Dataset | Classifier | Uncompressed | TaCo-L | TaCo-L Bitrate |
|--------|--------|--------|--------|-------------|
| Touch and Go | SVM | 76.63% | 75.12% | 0.193 bpp (124×) |
| YCB-Slide | SVM | 98.75% | 98.01% | 0.126 bpp (190×) |

**Dexterous Grasping (Isaac Sim, Success Rate)**: At 0.0251 bpp, TaCo-L achieved a lifting success rate $s_{lift}$ of 62.2% (vs. 63.8% uncompressed).

| Codec | BPP | $s_{lift}$ Avg | $s_{disturb}$ Avg |
|---------|-----|--------------|------------------|
| Uncompressed | 24 | 63.8% | 61.7% |
| TaCo-L | **0.0251** | 62.2% (−1.6%) | 59.9% (−1.8%) |

### Key Findings
- **In-domain training is decisive**: Retraining on tactile data (TaCo-LL/L) significantly outperforms pre-trained versions and off-the-shelf codecs.
- **Physical tactile compression has a ceiling**: Real-world tactile data limits are around 22×, whereas simulated signals can reach 1000× due to extreme sparsity.
- **Grasping is more sensitive than classification**: While classification tasks tolerate 100×+ compression, closed-loop control drops by ~1.6-1.8% even with optimal codecs.
- **Complexity is controllable**: TaCo-LL achieves better results than 8B-parameter LLM compressors with only 12M-96M parameters.

## Highlights & Insights
- **Force Vector to RGB Mapping**: Treating 3D force components as channels is a clever trick to apply mature image codecs to non-visual modalities.
- **Task-specific focus**: Explicitly evaluating "who we are compressing for" reveals asymmetric resilience (classification is robust; grasping is fragile).
- **Proving the value of domain training**: The sharp contrast between LALIC's failure (BD-Rate +32.8%) and TaCo-L's success (−27.0%) on mechanical data confirms that specialized tactile compression is a necessity.

## Limitations & Future Work
- **Architectural innovation**: TaCo-LL/L are retrained SOTA models rather than entirely new architectures designed for tactile physics.
- **Inter-frame redundancy**: TaCo currently focuses on intra-frame compression; future work should integrate temporal modeling.
- **Simulation gap**: Grasping experiments in Isaac Sim involve very sparse signals; real-world verification is required.

## Related Work & Insights
- Compared to classic signal-processing methods, TaCo provides a rigorous benchmarking framework.
- Compared to generic LLM compressors (e.g., LMIC), TaCo's specialized small models are faster and more efficient for structured tactile data.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AnyTouch 2: General Optical Tactile Representation Learning For Dynamic Tactile Perception](anytouch_2_general_optical_tactile_representation_learning_for_dynamic_tactile_p.md)
- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ICLR 2026\] Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning](memory_benchmark_robots_a_benchmark_for_solving_complex_tasks_with_reinforcement.md)
- [\[ICLR 2026\] DexMove: Learning Tactile-Guided Non-Prehensile Manipulation with Dexterous Hands](dexmove_learning_tactile-guided_non-prehensile_manipulation_with_dexterous_hands.md)
- [\[ICLR 2026\] RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification](rf-matid_dataset_and_benchmark_for_radio_frequency_material_identification.md)

</div>

<!-- RELATED:END -->
