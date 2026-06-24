---
title: >-
  [Paper Note] Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection
description: >-
  [ECCV 2024][Video Understanding][Video Anomaly Detection] A framework for unsupervised video anomaly detection is proposed, which interleaves the training of a weighted one-class classification (wOCC) model and a weakly-supervised (WS) model. It mitigates training fluctuations using soft labels and progressively optimizes the segmentation threshold via an adaptive thresholding strategy, achieving performance close to weakly-supervised methods without requiring any manual anno…
tags:
  - "ECCV 2024"
  - "Video Understanding"
  - "Video Anomaly Detection"
  - "Unsupervised Learning"
  - "One-Class Classification"
  - "Weakly-Supervised Learning"
  - "Adaptive Thresholding"
date: 2026-05-08
content_hash: 0adc648e412c5681
---

# Interleaving One-Class and Weakly-Supervised Models with Adaptive Thresholding for Unsupervised Video Anomaly Detection

**Conference**: ECCV 2024  
**arXiv**: [2401.13551](https://arxiv.org/abs/2401.13551)  
**Code**: [github.com/benedictstar/Joint-VAD](https://github.com/benedictstar/Joint-VAD)  
**Area**: Human Understanding  
**Keywords**: Video Anomaly Detection, Unsupervised Learning, One-Class Classification, Weakly-Supervised Learning, Adaptive Thresholding

## TL;DR

A framework for unsupervised video anomaly detection is proposed, which interleaves the training of a weighted one-class classification (wOCC) model and a weakly-supervised (WS) model. It mitigates training fluctuations using soft labels and progressively optimizes the segmentation threshold via an adaptive thresholding strategy, achieving performance close to weakly-supervised methods without requiring any manual annotations.

## Background & Motivation

Video Anomaly Detection (VAD) primarily follows two paradigms:
- **One-Class Classification (OCC)**: Trained on normal data only, but requires manual removal of anomalous data.
- **Weakly-Supervised (WS)**: Trained on video-level annotations, which are costly and cover bounded anomaly categories.

Both rely on manual annotations and fail to cover all anomaly types. **Unsupervised VAD (UVAD)** aims to operate completely without annotations, but existing methods (such as GCL) that adopt autoencoders and fully connected classifiers have limited capacity.

Key Insight of this work: While OCC and WS methods have rapidly progressed individually, can the latest OCC and WS models be directly combined into a single UVAD framework? The challenges lie in: (1) training fluctuations caused by the randomness of pseudo-labels; (2) the reliance on manual hyperparameters for the threshold required to split pseudo-labels into normal and anomalous classes.

## Method

### Overall Architecture

The framework contains an iterative **wOCC-WS alternating training module**:
1. The wOCC model generates anomaly scores -> partitions pseudo-labels for the WS model.
2. The WS model generates anomaly scores -> used as soft labels for the wOCC model.
3. Upon convergence of the module, the threshold is adjusted using an adaptive threshold update mechanism, initiating the next round.
4. The process is repeated until the stopping criterion is met.

### Key Designs

1. **Weighted One-Class Classification (wOCC)**: The core solution to the training fluctuation issue. Traditional OCC uses hard labels $\{0,1\}$ to distinguish between normal and anomalous data for training. Hard labels can mutate abruptly (e.g., $0 \to 1$ or $1 \to 0$), causing training instability. wOCC introduces a soft label $w_{X_i} \in [0,1]$ as a weight:
    $\mathcal{L}_{wocc} = -(1 - w_{X_i}) \log(p_Z(f_{STG\text{-}NF}(X_i)))$
   A smaller weight indicates a more normal sample, which receives a larger weight in the likelihood modeling. Soft labels change continuously (e.g., $0.7 \to 0.6$ instead of $1 \to 0$), significantly reducing training fluctuations. Taking STG-NF as an example, this extends the original negative log-likelihood on only normal data to a weighted version over all data.

2. **Pseudo-Label Interaction Mechanism**: In the wOCC -> WS direction, hard labels are generated using ranking and thresholding: snippets are sorted in descending order of anomaly scores, and the top $T_{ws}$ snippets are labeled as anomalous. In the WS -> wOCC direction, the anomaly scores from the WS model are directly used as soft labels $w_{X_i} = \hat{x}_i$ without needing an additional threshold.

3. **Adaptive Thresholding Strategy**: Solves the threshold dependency issue. The core ensures that the threshold is monotonically decreasing: $T_{ws}^1 \geq T_{ws}^2 \geq \cdots$:

    - The initial threshold $T_{ws}^1 = R\% \times N$ (e.g., $R=30$) is set to a sufficiently large value.
    - Each round produces multiple wOCC models, each identifying a set $A_j$ representing the top $R\%$ of snippets with the highest anomaly scores.
    - The threshold for the next round is obtained by counting the size of the intersection: $T_{ws}^{i+1} = \text{Num}(A_1 \cap A_2 \cap \cdots \cap A_{M_i})$.
    - The intersection operation ensures monotonic decrease: fewer early models yield a larger intersection, which tightens as more models join the consensus.
    - **Stopping Criterion**: The process stops when the threshold rate of change drops below $Q\%$ (default: 10%) of the initial rate of change.

4. **Initialization Strategy**: In the first module, soft labels are randomly initialized by sampling from a Beta distribution $w_X \sim \text{Beta}(1, 5)$, which ensures most weights are close to 0 (normal) and a few are close to 1 (anomalous), aligning with the data prior. Subsequent modules are initialized using the final WS model output from the previous module.

### Loss & Training

- **wOCC Loss**: Weighted negative log-likelihood (taking STG-NF as an example).
- **WS Loss**: Top-k MIL ranking loss + BCE classification loss (taking RTFM as an example).
- **Alternating Training**: Switches after training one model for one epoch. The entire training takes about 30 epochs, costing around 2.5 hours.
- Model parameters are re-initialized for each new module (not inherited) to avoid error accumulation.

## Key Experimental Results

### Main Results

**ShanghaiTech Dataset (AUC %)**:

| Method | Supervision Type | Feature | AUC |
|------|---------|------|-----|
| GCL | Unsupervised | I3D | 76.14 |
| STG-NF (full data) | Unsupervised | - | 80.29 |
| **OurwOCC** | **Unsupervised** | - | **82.57** |
| **OurWS** | **Unsupervised** | I3D | **88.18** |
| STG-NF | OCC | - | 85.90 |
| RTFM | WS | I3D | 96.10 |

**UBnormal Dataset (AUC %)**:

| Method | AUC |
|------|-----|
| STG-NF (full data) | 70.48 |
| **OurwOCC** | **74.76** |
| **OurWS** | 63.10 |

### Ablation Study

**wOCC vs OCC + Adaptive Thresholding Ablation (ShanghaiTech)**:

| Weighted OCC | Adaptive Thresholding | RTFM AUC | STG-NF AUC |
|:---:|:---:|:---:|:---:|
| ✗ | ✗ | 82.06 | 80.52 |
| ✓ | ✗ | 83.48 | 81.78 |
| ✗ | ✓ | 85.86 | 81.94 |
| **✓** | **✓** | **88.18** | **82.57** |

**Different OCC Model Combinations (WS=RTFM)**:

| OCC Model | OurwOCC | OurWS(RTFM) |
|---------|---------|-------------|
| AE | 70.99 | 78.90 |
| Jigsaw | 81.23 | 85.35 |
| **STG-NF** | **82.57** | **88.18** |

### Key Findings

- wOCC significantly reduces training fluctuations compared to directly using OCC, achieving a higher converged AUC.
- The adaptive threshold converges after about 6 modules, with different initial $R\%$ values ultimately converging to similar thresholds.
- Heterogeneous combinations (OCC+WS) outperform homogeneous combinations (OCC+OCC or WS+WS).
- When degenerated into supervised settings, wOCC outperforms the original OCC (86.37 vs 85.90), demonstrating the inherent value of the weighting mechanism.
- The overall training time is only about 2.5 hours, which is comparable to the training time of a single original model.

## Highlights & Insights

- **Flexibility of the Framework**: Allows plug-and-play integration of the latest OCC and WS models, enabling automatic upgrades alongside advances in the field.
- **Stability Insight of Soft Labels**: Identifies that abrupt hard label mutation is the root cause of training fluctuations; using soft labels elegantly resolves this issue in a simple yet effective manner.
- **Mathematical Guarantee of Adaptive Thresholding**: The intersection operation guarantees monotonic decrease, without relying on specific ranges of anomaly scores.
- **Convergence Analysis**: Provides an analysis of why wOCC can still learn meaningful representations starting from random initialization—the prior that normal data significantly outnumbers anomalous data is sufficient to bootstrap learning.

## Limitations & Future Work

- Although the $R\%$ parameter is insensitive, it still needs to be roughly set according to the dataset.
- On the UBnormal dataset, the AUC of the WS model (63.10%) is lower than that of the wOCC model (74.76%), indicating limited adaptability to datasets with a high proportion of anomalies.
- Combinations of OCC/WS models based on Transformer architectures have not been explored.
- The stopping criterion is based on the rate of threshold change rather than direct performance metrics, which may not be the optimal stopping point under extreme conditions.
- Validated only on pose-based anomaly detection, without testing on appearance-based anomaly scenarios.

## Related Work & Insights

- GCL was the only prior UVAD method, but its network architecture has limited capacity.
- STG-NF, as an OCC model, models the normal pose distribution through normalizing flows, which inspired the design of the weighted likelihood.
- RTFM's Top-k MIL mechanism provides a strong baseline for the WS module.
- Self-training and pseudo-labeling have been widely used in semi-supervised learning; this work extends them to completely unsupervised scenarios.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The alternating wOCC-WS framework combined with adaptive thresholding is novel, and the soft label approach is simple and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Extensive ablations and combinations with various OCC/WS models are provided, though evaluated only on two datasets.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear logic, complete mathematical derivations, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ — Proposes a continuously upgradeable UVAD framework with high practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Learning Anomalies with Normality Prior for Unsupervised Video Anomaly Detection](learning_anomalies_with_normality_prior_for_unsupervised_video_anomaly_detection.md)
- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/video_understanding/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[ECCV 2024\] ActionSwitch: Class-agnostic Detection of Simultaneous Actions in Streaming Videos](actionswitch_class-agnostic_detection_of_simultaneous_actions_in_streaming_video.md)
- [\[CVPR 2026\] The Road Less Seen: Segment Exploration for Weakly Supervised Video Anomaly Detection](../../CVPR2026/video_understanding/the_road_less_seen_segment_exploration_for_weakly_supervised_video_anomaly_detec.md)
- [\[AAAI 2026\] RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection](../../AAAI2026/video_understanding/refinevad_semantic-guided_feature_recalibration_for_weakly_supervised_video_anom.md)

</div>

<!-- RELATED:END -->
