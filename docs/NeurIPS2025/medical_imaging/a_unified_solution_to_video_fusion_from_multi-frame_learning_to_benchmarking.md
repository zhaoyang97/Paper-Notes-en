---
title: >-
  [Paper Note] A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking
description: >-
  [NeurIPS 2025][Medical Imaging][video fusion] This paper proposes UniVF, the first unified video fusion framework based on multi-frame learning, optical flow feature warping, and temporal consistency loss, along with VF-Bench, the first video fusion benchmark covering four major fusion tasks (multi-exposure, multi-focus, infrared-visible, and medical), achieving state-of-the-art performance across all sub-tasks.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - video fusion
  - multi-frame learning
  - optical flow alignment
  - temporal consistency
  - benchmark
date: 2026-05-08
content_hash: d0ea6d152644209e
---

# A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking

**Conference**: NeurIPS 2025
**arXiv**: [2505.19858](https://arxiv.org/abs/2505.19858)
**Code**: [vfbench.github.io](https://vfbench.github.io)
**Area**: Medical Imaging
**Keywords**: video fusion, multi-frame learning, optical flow alignment, temporal consistency, benchmark
**Authors**: Zixiang Zhao (ETH Zürich), Haowen Bai, Bingxin Ke, Yukun Cui, Lilun Deng, Yulun Zhang, Kai Zhang, Konrad Schindler

## TL;DR

This paper proposes UniVF, the first unified video fusion framework based on multi-frame learning, optical flow feature warping, and temporal consistency loss, along with VF-Bench, the first video fusion benchmark covering four major fusion tasks (multi-exposure, multi-focus, infrared-visible, and medical), achieving state-of-the-art performance across all sub-tasks.

---

## Background & Motivation

**Limitations of image fusion**: Existing image fusion methods (multi-exposure, multi-focus, infrared-visible, and medical fusion) operate on static frames, ignoring inter-frame temporal dependencies in video. Frame-by-frame independent processing leads to flickering and temporal inconsistency.

**Video fusion as a natural extension**: The real world is dynamic. Video provides continuous, temporally consistent scene representations that capture motion, transient events, and contextual changes. With advances in video hardware and the growth of video data, extending image fusion to the temporal domain has become an inevitable trend.

**Insufficient exploitation of temporal information**: Processing frames independently ignores the inherent temporal continuity of video, causing flickering and motion discontinuities. Effective video fusion must integrate information from neighboring frames to improve per-frame quality while ensuring temporal consistency.

**Limited dataset scale**: Compared to paired images, collecting perfectly aligned, temporally synchronized, and diverse video pairs is significantly more difficult and costly, constraining the development and evaluation of data-driven fusion methods.

**Lack of evaluation protocols**: Existing evaluation metrics are designed for images and completely ignore consistency assessment along the temporal axis, making them unable to measure the temporal quality of video fusion.

**Absence of a unified framework and benchmark**: Although a small number of infrared-RGB video fusion works have emerged recently, a unified video fusion framework and a multi-task benchmark remain absent.

---

## Method

### Overall Architecture: UniVF

UniVF adopts a Transformer-based (Restormer) encoder-decoder architecture. The core mechanism is as follows: at each time step $t$, three consecutive frames ($t-1, t, t+1$) are taken from each of the two input video streams. Optical flow estimation and feature warping are used to align neighboring frame features to the current frame, which are then fused and decoded. The model is trained with a temporal consistency loss.

The framework comprises four core components:

#### Key Design 1: Dual-Stream Feature Extractor

For two input video streams $\mathcal{V}_1, \mathcal{V}_2$, a 3-frame clip $\{I_{t-1}^k, I_t^k, I_{t+1}^k\}$, $k \in \{1,2\}$, is extracted at each time step. Each stream has an independent encoder $\mathcal{E}_k$ (composed of multiple Restormer blocks with 8 attention heads, feature dimension 32, and 4 stacked layers), with parameters shared across the 3 frames within the same stream:

$$\Phi_{t-1}^k, \Phi_t^k, \Phi_{t+1}^k = \mathcal{E}_k(I_{t-1}^k, I_t^k, I_{t+1}^k)$$

#### Key Design 2: Optical Flow Estimation and Feature Warping

This is the core mechanism that distinguishes UniVF from per-frame image fusion. SEA-RAFT (a state-of-the-art optical flow estimator) is used to compute bidirectional optical flow between adjacent frames:

$$\mathcal{O}_{s \to t}^k = \mathcal{S}(I_s^k, I_t^k), \quad s \in \{t-1, t+1\}$$

Differentiable bilinear sampling is then applied to warp deep features from neighboring frames to the current time step using the estimated flow:

$$\widetilde{\Phi}_{s \to t}^k = \mathcal{W}(\Phi_s^k, \mathcal{O}_{s \to t}^k)$$

The warped features are temporally aligned with the current frame features and fed into subsequent fusion as motion-compensated inputs.

#### Key Design 3: Fusion Decoder

Six sets of features (three frames from each of the two streams, i.e., the current frame features plus forward/backward warped features) are concatenated along the channel dimension:

$$\Phi_t^F = \text{Concat}(\Phi_t^1, \Phi_t^2, \widetilde{\Phi}_{t-1 \to t}^1, \widetilde{\Phi}_{t+1 \to t}^1, \widetilde{\Phi}_{t-1 \to t}^2, \widetilde{\Phi}_{t+1 \to t}^2)$$

A Restormer-based decoder $\mathcal{D}$ models long-range dependencies in both spatial and temporal dimensions, producing the fused output for the current frame: $I_t^F = \mathcal{D}(\Phi_t^F)$.

### Loss & Training

Training employs a composite loss of three terms: $\mathcal{L} = \mathcal{L}_{\text{spatial}} + \alpha_1 \mathcal{L}_{\text{grad}} + \alpha_2 \mathcal{L}_{\text{temp}}$

- **Spatial similarity loss $\mathcal{L}_{\text{spatial}}$**: Different strategies are adopted for different tasks. IVF/MVF use $\|I_t^F - \max(I_t^1, I_t^2)\|_1$; MEF uses an intensity loss combined with MEF-SSIM; MFF uses a mean intensity loss.
- **Gradient preservation loss $\mathcal{L}_{\text{grad}}$**: Based on the Sobel operator, this term preserves structural details and edges from the source images: $\||\nabla I_t^F| - \max(|\nabla I_t^1|, |\nabla I_t^2|)\|_1$.
- **Temporal consistency loss $\mathcal{L}_{\text{temp}}$**: The core innovation. The current fused frame is compared against temporally warped fused frames from neighboring time steps, penalizing inconsistent regions. A validity mask based on forward-backward optical flow consistency checking is introduced to compute the loss only in reliable regions, excluding occluded areas and motion boundaries. Threshold $\epsilon=1.0$.

Loss weight configurations: $\{\alpha_1, \alpha_2\} = \{10,2\}, \{1,0.5\}, \{5,2\}, \{1,1\}$ for MEF, MFF, IVF, and MVF, respectively.

---

## Key Experimental Results

### VF-Bench Dataset Construction

| Task | Data Source | Construction Method | Train/Test Scenes | Avg. Frames |
|------|------------|--------------------|--------------------|-------------|
| MEF (Multi-Exposure) | YouTube-HDR 10-bit | EOTF linear-domain exposure adjustment ±3 EV | 450/50 | ~150 |
| MFF (Multi-Focus) | DAVIS | Video depth estimation + CoC blur | 120/30 | ~70 |
| IVF (Infrared-Visible) | VTMOT | Three-stage filtering (quality + complementarity + alignment) | 75/15 | ~300 |
| MVF (Medical) | Harvard Medical | MRI+CT/PET/SPECT consecutive slices | 49/8 | ~27 |

### Main Results (Tab 1–3)

**Multi-Exposure Video Fusion (MEF, 2K resolution)**:

| Method | VIF↑ | SSIM↑ | MI↑ | Q_abf↑ | BiSWE↓ | MS2R↓ |
|--------|------|-------|-----|--------|--------|-------|
| FILM | 0.78 | 0.98 | 4.39 | 0.71 | 8.27 | 0.34 |
| TC-MoA | 0.76 | 0.98 | 2.94 | 0.71 | 7.78 | 0.34 |
| **UniVF** | **0.82** | **0.99** | **4.45** | **0.72** | **6.40** | **0.33** |

**Infrared-Visible Video Fusion (IVF)**:

| Method | VIF↑ | SSIM↑ | MI↑ | Q_abf↑ | BiSWE↓ | MS2R↓ |
|--------|------|-------|-----|--------|--------|-------|
| TDFusion | 0.45 | 0.64 | 2.34 | 0.67 | 4.35 | 0.36 |
| ReFusion | 0.42 | 0.64 | 2.27 | 0.67 | 4.64 | 0.36 |
| **UniVF** | **0.44** | **0.64** | **2.47** | **0.68** | **3.94** | **0.35** |

**Medical Video Fusion (MVF)**:

| Method | VIF↑ | SSIM↑ | MI↑ | Q_abf↑ | BiSWE↓ | MS2R↓ |
|--------|------|-------|-----|--------|--------|-------|
| CDDFuse | 0.29 | 0.76 | 1.80 | 0.59 | 26.33 | 1.34 |
| FILM | 0.33 | 0.36 | 1.83 | 0.67 | 32.04 | 1.59 |
| **UniVF** | **0.35** | **0.76** | **2.00** | **0.68** | **29.61** | **1.30** |

### Ablation Study (IVF task, Tab 4)

| Configuration | VIF↑ | SSIM↑ | MI↑ | Q_abf↑ | BiSWE↓ | MS2R↓ |
|--------------|------|-------|-----|--------|--------|-------|
| w/o feature warping | 0.40 | 0.63 | 2.44 | 0.66 | 4.18 | 0.36 |
| w/o warping + w/o multi-frame input | 0.38 | 0.61 | 2.07 | 0.64 | 4.46 | 0.37 |
| w/o temporal consistency loss | 0.42 | 0.65 | 2.38 | 0.65 | 5.79 | 0.39 |
| **Full UniVF** | **0.44** | **0.64** | **2.47** | **0.68** | **3.94** | **0.35** |

### Key Findings

1. **Temporal consistency loss is critical**: Removing $\mathcal{L}_{\text{temp}}$ degrades BiSWE from 3.94 to 5.79 (+47%) and MS2R from 0.35 to 0.39, demonstrating that this loss is key to suppressing flickering.
2. **Optical flow feature warping is effective**: Removing warping degrades both spatial and temporal metrics, confirming that cross-frame feature alignment makes a substantive contribution to fusion quality.
3. **Multi-frame input is necessary**: Degrading to single-frame processing causes a comprehensive drop across all metrics, validating the value of multi-frame learning.
4. **Unified framework generalizes across tasks**: The same architecture achieves state-of-the-art or highly competitive results across all four distinct tasks.

---

## Highlights & Insights

1. **First unified video fusion framework**: UniVF covers four fusion task categories with a single unified architecture, eliminating the redundancy of designing task-specific networks and demonstrating the feasibility of cross-task fusion.
2. **First video fusion benchmark**: VF-Bench fills a critical gap in the field and introduces two innovative data generation paradigms: multi-exposure (HDR → linear-domain exposure simulation) and multi-focus (depth estimation + CoC physical blur).
3. **Elegant temporal consistency design**: The validity mask is based on forward-backward optical flow consistency checking, computing the temporal loss only in reliable regions to avoid noisy gradients from occlusions and motion boundaries.
4. **New temporal evaluation metrics**: BiSWE (Bidirectional Self-Warping Error) and MS2R (Motion-Smoothness-to-Signal Ratio) address the absence of temporal dimension assessment in video fusion evaluation.
5. **Methodological value of data construction**: Multi-exposure data is generated by adjusting exposure in the linear light domain via EOTF, achieving far higher fidelity than direct manipulation in the gamma domain; multi-focus data is based on a physical CoC model rather than semantic segmentation, better reflecting real optical processes.

---

## Limitations & Future Work

1. **Only a 3-frame window**: The current design exploits only one preceding and one following frame (3-frame window), which may be insufficient for fast motion or long-range temporal dependencies; larger windows could further improve performance at the cost of additional computation.
2. **Frozen optical flow estimator**: SEA-RAFT is used as a pre-trained, frozen optical flow estimator without end-to-end joint training, potentially limiting adaptation to specific fusion scenarios (e.g., non-natural motion in medical images).
3. **Limited medical fusion data**: MVF contains only 49 training and 8 test scenes with an average of 27 frames, constraining data scale and diversity, which may limit generalization.
4. **Computational overhead**: Encoding three frames from each of two streams, plus four optical flow estimations and four warping operations, substantially increases inference cost compared to single-frame methods.
5. **Limitations of evaluation metrics**: BiSWE and MS2R still rely on optical flow estimation, whose accuracy is bounded by the optical flow estimator itself.
6. **Alternative temporal modeling not explored**: Approaches such as 3D convolutions, temporal attention, or state space models are not compared as alternatives.

---

## Related Work & Insights

- **Discriminative fusion methods** (CDDFuse, EMMA, TC-MoA, FILM, ReFusion): Most existing state-of-the-art image fusion methods are discriminative models (CNN/Transformer); UniVF extends these by introducing the temporal dimension.
- **Generative fusion** (DDFM, GAN-based): Diffusion models and GANs model latent space manifolds to produce richer detail, but ensuring temporal consistency is more challenging.
- **Optical flow utilization in video restoration/enhancement**: UniVF's optical flow warping strategy directly draws on experience from video super-resolution and video inpainting (e.g., BasicVSR++).
- **Restormer**: As the backbone network, Restormer's transposed self-attention mechanism has proven effective in image restoration tasks.
- **Trend toward unified fusion models** (TC-MoA, FILM, ReFusion): Recent works have begun exploring cross-task unified fusion; UniVF extends this trend to the video domain.
- **Insights**: The data construction methodology presented in this paper (physics-based synthesis + rigorous filtering of existing data) offers strong reference value for other domains lacking paired video data. The VF-Bench evaluation protocol (joint spatial and temporal assessment) is generalizable to other video generation and editing tasks.

---

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First unified video fusion framework and benchmark; the temporal consistency loss design (validity mask) and data generation paradigms are both innovative, though the core technical components (optical flow warping + Restormer) are combinations of existing modules.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covers 4 sub-tasks, compares against 7–10 methods, ablation study is well-designed, spatial and temporal metrics are comprehensive, and testing is conducted at both 2K and lower resolutions.
- **Writing Quality**: ⭐⭐⭐⭐ — Structure is clear, mathematical derivations are complete, figures and tables are rich, and the data construction process is described in detail.
- **Value**: ⭐⭐⭐⭐⭐ — VF-Bench has high field-level value as the first video fusion benchmark, laying a solid foundation for subsequent video fusion research and likely to drive systematic progress in this direction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] CGBench: Benchmarking Language Model Scientific Reasoning for Clinical Genetics Research](cgbench_benchmarking_language_model_scientific_reasoning_for_clinical_genetics_r.md)
- [\[NeurIPS 2025\] MedAgentBoard: Benchmarking Multi-Agent Collaboration with Conventional Methods for Diverse Medical Tasks](medagentboard_benchmarking_multi-agent_collaboration_with_conventional_methods_f.md)
- [\[NeurIPS 2025\] Unified All-Atom Molecule Generation with Neural Fields](unified_all-atom_molecule_generation_with_neural_fields.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](../../ICLR2026/medical_imaging/human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)

</div>

<!-- RELATED:END -->
