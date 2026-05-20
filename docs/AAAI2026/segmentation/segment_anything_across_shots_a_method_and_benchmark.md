---
title: >-
  [Paper Note] Segment Anything Across Shots: A Method and Benchmark
description: >-
  [AAAI 2026][Segmentation][multi-shot video segmentation] This paper proposes SAAS, a method for Multi-shot Video Object Segmentation (MVOS)…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "multi-shot video segmentation"
  - "SAM2"
  - "data augmentation"
  - "shot transition detection"
  - "benchmark"
date: 2026-05-08
content_hash: 7f9d7fc6b05b5882
---

# Segment Anything Across Shots: A Method and Benchmark

**Conference**: AAAI 2026
**arXiv**: [2511.13715](https://arxiv.org/abs/2511.13715)  
**Code**: [https://henghuiding.com/SAAS/](https://henghuiding.com/SAAS/)  
**Area**: Segmentation
**Keywords**: multi-shot video segmentation, SAM2, data augmentation, shot transition detection, benchmark

## TL;DR

This paper proposes SAAS, a method for Multi-shot Video Object Segmentation (MVOS), along with the Cut-VOS benchmark. SAAS achieves robust cross-shot segmentation via transition-simulating data augmentation (TMA), shot transition detection and understanding modules (TDM+TCH), and a local memory bank.

## Background & Motivation

Semi-supervised Video Object Segmentation (VOS) tracks and segments target objects in subsequent frames given a mask in the first frame. However, existing methods (XMem, Cutie, SAM2, etc.) almost exclusively focus on **single-shot videos**, neglecting the prevalence of **multi-shot videos** in real-world scenarios—creating a significant gap between academic research and practical deployment.

### Core Challenges in Multi-Shot Videos

Shot transitions in multi-shot videos introduce **drastic changes** in target appearance, spatial position, and background:
- SAM2-B+ suffers a **21.4% drop** in $\mathcal{J\&F}$ on the multi-shot benchmark Cut-VOS compared to the single-shot benchmark MOSE.
- On transition types such as delayed cut in, close-up view, and scene change, SAM2's tracking accuracy falls **below 27%**.
- Existing methods can detect target disappearance but **fail to correctly re-associate the target upon reappearance**.

### Insufficiency of Data and Benchmarks

- The only existing MVOS dataset, YouMVOS, has numerous shortcomings: sparse shot transitions, limited object categories (predominantly people), and unreleased mask annotations.
- The lack of native multi-shot training data hinders model development.
- No benchmark adequately reflects the challenges of multi-shot segmentation.

The authors address these issues via: (1) TMA, a strategy for synthesizing multi-shot training samples from single-shot data; (2) the SAAS model with dedicated modules for detecting and understanding shot transitions; and (3) the Cut-VOS benchmark for evaluating cross-shot segmentation.

## Method

### Overall Architecture

SAAS is built upon SAM2 and introduces three novel components:

1. **Transition-Simulating Data Augmentation (TMA)**: A training strategy that synthesizes multi-shot training samples from single-shot data.
2. **Transition Detection Module (TDM) + Transition Comprehension and Handling Module (TCH)**: Runtime modules for detecting and understanding shot transitions.
3. **Local Memory Bank $\mathcal{B}_{local}$**: Stores fine-grained local features of the target to facilitate cross-shot matching.

### Key Designs

#### 1. Transition-Simulating Data Augmentation (TMA)

TMA is the key innovation addressing the scarcity of multi-shot training data. Building upon standard 8-frame consecutive sampling, TMA applies transition simulation with probability $p_{trans}$, encompassing four main modes:

- **Mode (a) Random Strong Augmentation**: Applies horizontal flipping, random scaling, and random affine transforms to the latter half of an 8-frame clip, simulating close-up/long-shot transitions.
- **Mode (b) Intra-video Temporal Skip**: Samples frames from a different temporal segment of the same video, simulating transitions with large temporal gaps (changes in target pose and viewpoint).
- **Mode (c) Cross-video Multiple Transitions**: Cuts to an unrelated video and back, simulating cut away + cut in scenarios.
- **Mode (d) Cross-video with Copy-Paste**: Cuts to an unrelated video and copies the target with random translation, simulating scene change + delayed cut in.

Different modes are combined by controlling random variables $p_{trans}$, $p_{once}$, $p_{cut}$, $p_{same}$, $p_{copy}$, and $p_{hflip}$.

**Design Motivation**: Existing VOS datasets (e.g., YTVOS) consist entirely of single-shot videos; training on them directly does not improve—and may even degrade—multi-shot performance (a 0.3–0.9% drop on Cut-VOS). TMA bridges this gap synthetically, compensating for the absence of real annotated multi-shot data.

#### 2. Transition Detection Module (TDM)

A lightweight dilated convolutional pyramid predicts the transition probability for each frame:

$$\hat{p}_{i,tr} = \text{Sigmoid}(\mathcal{F}_{\text{TDM}}(F^t, F^{t-i}_{i=1,2,...,N}))$$

When $\hat{p}_{i,tr} < \tau_{tr}$, the standard SAM2 segmentation pipeline is followed; otherwise, a shot is identified and the transition segmentation strategy is activated.

- Memory from non-transition frames is encoded into $\mathcal{B}_{adj}$ (adjacent memory bank).
- Memory from transition frames is encoded into $\mathcal{B}_{scene}$ (scene memory bank) for scene-level understanding.

**Design Motivation**: Inspired by shot boundary detection methods (e.g., TransNet), transition detection must precede the application of transition-specific strategies. The dilated convolutional pyramid captures inter-frame differences at multiple temporal scales.

#### 3. Transition Comprehension and Handling Module (TCH)

TCH first reads scene information from $\mathcal{B}_{cond}$ and $\mathcal{B}_{scene}$ and integrates it into the current frame features via stacked attention layers. Learnable query vectors $Q_{init}$ then interact with both the previous and current frame features through multiple cross-attention layers:

$$Q_i^n = \text{Attn}(\text{Attn}(Q_i^{n-1}, F_{l3}^{\prime t}), F_{l3}^{t-1})$$

Two auxiliary training objectives are incorporated:
- **Existence Prediction**: Predicts whether the target appears in the next frame from $Q_i$ (BCE loss $\mathcal{L}_{exis}$).
- **Bounding Box Regression**: Predicts the target bounding box after the transition from $Q_i$ and the previous frame's box (MCE loss $\mathcal{L}_{box}$).

An **aggregator** decodes $Q_i$ to refine the previous memory $\mathcal{M}_{adj}^{t-1}$; the refined memory is concatenated with $\mathcal{B}_{cond}$ and $\mathcal{B}_{local}$ and fed into SAM2's memory attention module.

**Design Motivation**: Detecting transitions alone is insufficient; the model must also understand the type of transition and the change in target state. The auxiliary objectives compel the model to establish cross-transition mappings. The cross-attention aggregator ensures compatibility between the transition-aware features and SAM2's segmentation head.

#### 4. Local Memory Bank $\mathcal{B}_{local}$

- A **Minimum Spanning Tree (MST)** is constructed on the deep feature map $M_0 \odot F_{l3}^0$ of the conditioning frame.
- Low-weight edges are pruned to yield semantically coherent sub-region partitions (unsupervised segmentation).
- The centroid of each sub-region serves as a positive point prompt, with all remaining centroids as negative prompts; SAM extracts high-resolution, fine-grained features for each region.
- Features are compressed into complementary object pointers and stored in $\mathcal{B}_{local}$.
- A ratio threshold $\tau_p = 2.5\%$ filters out overly small regions to prevent over-segmentation.

**Design Motivation**: After a shot transition, local details of the target (e.g., clothing, vehicle markings) are critical cues for re-identification. MST-based segmentation unsupervisedly captures part-level features, addressing the inability of prior methods to actively leverage fine-grained local features.

### Loss & Training

Total loss = SAM2 original losses (focal + dice + iou + CE) + $0.5 \cdot \mathcal{L}_{box}$ + $0.5 \cdot \mathcal{L}_{exis}$

Training proceeds in two stages:
1. All other parameters are frozen; TDM is trained on the IACC.3 and ClipShots datasets.
2. All parameters are unfrozen; the full model is trained on YTVOS with TMA enabled for 30 epochs.

AdamW optimizer, learning rate decayed from 5e-6 to 5e-7, using 4 × NVIDIA RTX-A6000 GPUs.

## Key Experimental Results

### Main Results

| Method | Source | YouMVOS $\mathcal{J\&F}$ | YouMVOS $\mathcal{J}_t$ | Cut-VOS $\mathcal{J\&F}$ | Cut-VOS $\mathcal{J}_t$ |
|--------|--------|---------|---------|---------|---------|
| XMem | ECCV'22 | 61.9 | 54.2 | 49.9 | 35.5 |
| DEVA | ICCV'23 | 63.9 | 55.2 | 49.1 | 35.3 |
| Cutie | CVPR'24 | 67.7 | 63.4 | 52.3 | 40.8 |
| SAM2-B+ | ICLR'25 | 67.6 | 63.7 | 55.2 | 47.2 |
| SAM2-L | ICLR'25 | 70.1 | 68.5 | 59.4 | 50.7 |
| Cutie+TMA | - | 69.6 | 65.4 | 53.5 | 43.1 |
| **SAAS-B+** | **AAAI'26** | **73.5** | **68.9** | **60.7** | **53.1** |
| **SAAS-L** | **AAAI'26** | **74.2** | **69.6** | **62.0** | **54.0** |

SAAS-B+ vs. SAM2-B+: YouMVOS +5.9% $\mathcal{J\&F}$; Cut-VOS +5.5% $\mathcal{J\&F}$, +5.9% $\mathcal{J}_t$.

### Ablation Study

| ID | $\mathcal{B}_{local}$ | TMA | TCH | Cut-VOS $\mathcal{J\&F}$ | Cut-VOS $\mathcal{J}_t$ |
|----|---|-----|-----|---------|---------|
| I (Baseline) | ✗ | ✗ | ✗ | 55.2 | 47.2 |
| II | ✓ | ✗ | ✗ | 57.6 | 49.4 |
| III | ✗ | ✓ | ✗ | 58.0 | 50.7 |
| IV | ✓ | ✓ | ✗ | 58.8 | 52.0 |
| V | ✗ | ✓ | ✓ | 60.1 | 52.8 |
| VI (Full) | ✓ | ✓ | ✓ | **60.7** | **53.1** |

**TCH Internal Ablation (Tab. 5, Appendix)**:

| Config | Aggregator | $Q_i$ | $\mathcal{B}_{scene}$ | $\mathcal{J\&F}$ | $\mathcal{J}_t$ |
|--------|-----------|------|---------|---------|---------|
| I | Linear | ✗ | - | 59.2 | 50.1 |
| VII | **Cross-attn** | **✓** | **✓** | **60.6** | **52.9** |

### Key Findings

- **Generalizability of TMA**: TMA benefits not only SAAS but also Cutie+TMA, yielding consistent gains on both benchmarks (+1.2% $\mathcal{J\&F}$), demonstrating its general applicability.
- **Training on single-shot data without TMA is harmful**: SAM2-B+★ (fine-tuned without TMA) drops 0.3% on Cut-VOS compared to the untuned SAM2-B+, confirming that multi-shot scenarios require dedicated training strategies.
- **Three modules are complementary**: $\mathcal{B}_{local}$ contributes fine-grained matching (+2.4%), TMA provides richer training distribution (+2.8%), and TCH contributes transition understanding (TMA+TCH outperforms TMA+$\mathcal{B}_{local}$ by 1.3%).
- **Transition type analysis**: Delayed cut in, close-up view, and scene change are the most challenging types (SAM2 accuracy < 27%); Cut-VOS has lower expected accuracy than YouMVOS (38.8% vs. 44.7%).
- **Negligible inference overhead**: SAAS-B+ achieves 21 FPS vs. SAM2-B+ at 22 FPS.

## Highlights & Insights

1. **Identifying the "single-shot blind spot" in VOS research**: This widely overlooked yet practically important problem is compellingly substantiated by a 21.4% performance drop.
2. **TMA elegantly resolves the chicken-and-egg problem** of lacking multi-shot training data by combining six probability control variables to synthesize diverse transition patterns.
3. **Cut-VOS** is a high-quality benchmark: 1.6× higher transition frequency, 3× more object categories, a nine-category transition taxonomy, and a dual-review annotation process.
4. **The auxiliary objective design is well-motivated**: Existence prediction and bounding box regression compel TCH to genuinely "understand" transitions rather than merely detect them.
5. **The MST-based local memory bank** extracts part-level features in an unsupervised manner, offering an elegant annotation-free solution.

## Limitations & Future Work

- Extreme appearance changes (e.g., clothing or hairstyle changes) remain challenging—TMA cannot adequately simulate such cases, and local feature cues also fail.
- The method relies on purely visual feature matching and lacks high-level reasoning (e.g., it cannot distinguish "person A in white" from "person B in white").
- Cut-VOS is relatively small in scale (100 videos), which may not cover all real-world scenarios.
- TMA requires tuning six probability hyperparameters (though ablations show robustness across multiple configurations).
- Audio cues are not considered—sound is an important signal for understanding shot transitions in multi-shot videos.

## Related Work & Insights

- The TMA strategy is transferable to other video understanding tasks (e.g., video tracking, video instance segmentation).
- The two-stage design of transition detection followed by transition understanding may inspire other methods that must handle temporal discontinuities.
- The nine-category transition taxonomy of Cut-VOS can serve as a general analytical framework for multi-shot video understanding.
- The MST-based local memory bank is applicable to tasks requiring part-level features (e.g., fine-grained recognition, Re-ID).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First method and benchmark specifically targeting multi-shot VOS; TMA+TDM+TCH+local memory bank design is complete and original)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two benchmarks, comprehensive ablations, transition-type analysis, TMA generalizability validation, hyperparameter experiments)
- Writing Quality: ⭐⭐⭐⭐⭐ (Compelling problem framing, clear transition-type visualizations, complete algorithmic pseudocode)
- Value: ⭐⭐⭐⭐⭐ (Opens a new MVOS research direction; Cut-VOS will drive future work; TMA strategy has broad applicability)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Segment and Matte Anything in a Unified Model (SAMA)](segment_and_matte_anything_in_a_unified_model.md)
- [\[AAAI 2026\] SAQ-SAM: Semantically-Aligned Quantization for Segment Anything Model](saq-sam_semantically-aligned_quantization_for_segment_anything_model.md)
- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [\[ICCV 2025\] OmniSAM: Omnidirectional Segment Anything Model for UDA in Panoramic Semantic Segmentation](../../ICCV2025/segmentation/omnisam_omnidirectional_segment_anything_model_for_uda_in_panoramic_semantic_seg.md)
- [\[AAAI 2026\] Tracking and Segmenting Anything in Any Modality](tracking_and_segmenting_anything_in_any_modality.md)

</div>

<!-- RELATED:END -->
