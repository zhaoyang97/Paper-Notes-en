---
title: >-
  [Paper Note] RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection
description: >-
  [AAAI 2026][LLM Evaluation][Weakly supervised video anomaly detection] This paper proposes RefineVAD, a framework comprising two modules — Motion-aware Temporal Attention Recalibration (MoTAR) and Category-Oriented REfinement (CORE) — that jointly models temporal motion dynamics and anomaly category semantics, achieving precise localization and interpretable detection of anomalous events in weakly supervised video anomaly detection.
tags:
  - AAAI 2026
  - LLM Evaluation
  - Weakly supervised video anomaly detection
  - multiple instance learning
  - semantic guidance
  - temporal modeling
  - category prototypes
date: 2026-05-08
content_hash: 0e9a4193a9491bb6
---

# RefineVAD: Semantic-Guided Feature Recalibration for Weakly Supervised Video Anomaly Detection

**Conference**: AAAI 2026
**arXiv**: [2511.13204](https://arxiv.org/abs/2511.13204)
**Code**: [GitHub](https://github.com/VisualScienceLab-KHU/RefineVAD)
**Area**: LLM Evaluation
**Keywords**: Weakly supervised video anomaly detection, multiple instance learning, semantic guidance, temporal modeling, category prototypes

## TL;DR

This paper proposes RefineVAD, a framework comprising two modules — Motion-aware Temporal Attention Recalibration (MoTAR) and Category-Oriented REfinement (CORE) — that jointly models temporal motion dynamics and anomaly category semantics, achieving precise localization and interpretable detection of anomalous events in weakly supervised video anomaly detection.

## Background & Motivation

Weakly supervised video anomaly detection (WVAD) relies solely on video-level labels to identify anomalous events, striking a balance between annotation efficiency and practical applicability. Existing methods typically follow the multiple instance learning (MIL) paradigm, treating videos as "bags" of segments and assuming that at least one anomalous segment exists in an anomalous video.

However, existing WVAD methods suffer from two critical limitations:

**Shallow and rigid temporal modeling**: Most methods rely on fixed pooling or simple aggregation schemes, failing to adapt to the diverse motion characteristics present in real-world anomalies. Many anomalies are defined by dynamic, non-uniform, or context-dependent motion patterns, and temporal rigidity severely limits localization precision.

**Neglect of semantic diversity**: Most frameworks treat all anomalous events as a single generic category, ignoring the semantic differences among distinct anomaly types. For instance, fighting involves sudden bidirectional motion, whereas an explosion manifests as a sudden flash and spatial burst — overlooking such distinctions constrains the model's ability to learn discriminative features.

Human perception of anomalies leverages two complementary dimensions simultaneously: (1) the temporal evolution of contextual motion dynamics, and (2) prior knowledge of anomaly types. RefineVAD emulates this dual-process reasoning by jointly modeling "how motion evolves" and "what semantic category applies."

## Method

### Overall Architecture

RefineVAD follows the MIL setting, dividing input videos into $T$ fixed-length segments. Each segment is independently encoded by a pretrained visual encoder (CLIP ViT-L/14) and a text encoder (InternVideo2.5), and concatenated to form a joint multimodal representation. This representation is processed sequentially by MoTAR and CORE, and a lightweight classifier then computes segment-level anomaly scores.

### Key Designs

1. **MoTAR (Motion-aware Temporal Attention Recalibration)**: The core mechanism adaptively adjusts the channel shift ratio of temporal features according to motion intensity. Given the input sequence $\mathbf{X} = [\mathbf{x}_1, \dots, \mathbf{x}_T] \in \mathbb{R}^{T \times D}$, adjacent frame feature differences are first computed as $\Delta_t = \mathbf{x}_t - \mathbf{x}_{t-1}$, and the variance $\mathbf{v}_t = \text{Var}(\Delta_t)$ is used to measure local motion intensity. Higher variance indicates more salient motion, requiring broader temporal context aggregation. The variance vector is passed through a lightweight MLP to predict a shift ratio $r_t = \sigma(W_3 \cdot \phi(W_2 \cdot \phi(W_1 v_t)))$, from which the number of shifted channels is computed as $s_t = \lfloor r_t \cdot D/K \rfloor$. The dynamically constructed shifted output is $\mathbf{y}_t = [\mathbf{x}_{t-1}^{(1:s_t)}, \mathbf{x}_{t+1}^{(s_t:2s_t)}, \mathbf{x}_t^{(2s_t:D)}]$. A lightweight Transformer then encodes long-range temporal dependencies. **Design Motivation**: Conventional TSM uses a fixed shift ratio and cannot adapt to frames with varying motion intensities; MoTAR addresses this through variance-driven adaptive shifting.

2. **CORE (Category-Oriented REfinement)**: CORE consists of two stages: soft category classification and category prototype injection. First, temporal contextual features output by MoTAR are coarsely scored via a fully connected layer, normalized, and aggregated into a video-level representation, which is then passed through a soft category classifier to obtain logits $\mathbf{z} \in \mathbb{R}^{C \times 2}$ ($C$ being the number of anomaly categories), where each row corresponds to "normal"/"anomalous" scores. The anomaly probability $p_c^a$ is computed for each category $c$, and category weights $w_c$ are derived via softmax. These weights are then used to compute a weighted sum over learnable category prototype embeddings $\mathbf{E} \in \mathbb{R}^{C \times d_{\text{emb}}}$, yielding a soft category embedding $\mathbf{v} = \sum_{c=1}^{C} w_c \mathbf{e}_c$. This embedding is injected into segment-level features via cross-attention: $\mathbf{x}_t^{\text{ca}} = \text{CrossAttn}(\mathbf{v}, \mathbf{x}_t^{\text{tc}}, \mathbf{x}_t^{\text{tc}})$. **Design Motivation**: Defining the "normal" state as the absence of strong anomalous features — rather than as the first anomaly class — avoids distortion of the representation space; soft classification, as opposed to hard classification, captures ambiguous or overlapping anomalous features.

### Loss & Training

The total loss comprises three components:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{MIL}} + \lambda_1 \mathcal{L}_{\text{GMM}} + \lambda_2 \mathcal{L}_{\text{cat}}$$

- $\mathcal{L}_{\text{MIL}}$: Top-$k$ MIL ranking loss, encouraging anomalous segments to score higher than normal segments.
- $\mathcal{L}_{\text{GMM}}$: An improved GMM smoothing loss that incorporates weighted category embeddings to more clearly reflect category-specific features.
- $\mathcal{L}_{\text{cat}}$: Category classification loss using BCE to encourage semantic embeddings to remain category-discriminative.

During training, the ground-truth category embedding is added to the soft category embedding as $\mathbf{v}_{\text{train}} = \mathbf{v} + \mathbf{e}_y$ to promote category-aware specialization. Loss weights are set to $\lambda_1 = 0.1$ and $\lambda_2 = 0.2$. The AdamW optimizer is used with a batch size of 64, up to 30 epochs, on a single NVIDIA A5000 GPU.

## Key Experimental Results

### Main Results

| Dataset | Metric | RefineVAD | Prev. SOTA | Gain |
|--------|------|-----------|----------|------|
| UCF-Crime | AUC (%) | 88.92 | 90.33 (π-VAD) | -1.41 |
| XD-Violence | AP (%) | **88.66** | 86.52 (Ex-VAD) | **+2.14** |
| UCF-Crime | mAP@0.1 (%) | **20.90** | 16.51 (Ex-VAD) | **+4.39** |

RefineVAD achieves state of the art among all weakly supervised methods on XD-Violence (88.66% AP) and also leads significantly on the fine-grained mAP@0.1 metric on UCF-Crime (20.90%). The UCF-Crime AUC is slightly below π-VAD but remains competitive.

### Ablation Study

| Configuration | AUC (%) | Notes |
|------|---------|------|
| Base (MLP + MIL) | 84.60 | Baseline model |
| + MoTAR | 85.43 | Motion-aware temporal adjustment: +0.83 |
| + Category-Injection | 87.28 | Category injection: largest gain, +1.85 |
| + Category-Injection + Soft-Classification | 87.85 | Soft classification: further +0.57 |
| + MoTAR + CORE (Full) | **88.89** | All modules combined: optimal, +4.29 |

### Key Findings

1. **Category injection is the largest contributor**: Adding Category-Injection alone yields a +2.68% AUC improvement (84.60→87.28), the largest gain among all modules, demonstrating the critical importance of semantic category information for anomaly detection.
2. **Strong cross-dataset semantic transfer**: The CORE module trained on UCF-Crime transfers directly to XD-Violence, achieving 87.52% AP (vs. 88.66% with full training); zero-shot cross-domain evaluation still attains 77.56% AP.
3. **t-SNE visualization reveals semantically coherent clustering**: Arson/explosion/traffic accidents cluster together (scene-level abrupt changes); arrest/assault/fighting cluster together (multi-person interaction); shoplifting/robbery cluster together (single-person behavior).

## Highlights & Insights

- Defining "normal" as the absence of anomalous features rather than as an independent class avoids distortion of the representation space — a clever and practically meaningful design choice.
- The soft classification mechanism, rather than hard classification, simultaneously captures semantic cues from multiple categories, offering greater flexibility in handling boundary-ambiguous anomalies.
- The variance-based motion intensity estimation in MoTAR is parameter-free and noise-robust, with negligible computational overhead, making it suitable for real-time applications.
- The modular architecture allows each component's contribution to be independently validated and facilitates component replacement or upgrading in practice.

## Limitations & Future Work

- RefineVAD does not surpass π-VAD on UCF-Crime AUC (88.92% vs. 90.33%), suggesting a structural disadvantage of video-level category prediction for precise frame-level boundary localization at high IoU thresholds.
- The number of category prototypes must be predefined and is dataset-specific, limiting scalability to open-world novel anomaly types.
- Evaluation is conducted on only two datasets (UCF-Crime and XD-Violence), without validation on larger-scale or more diverse scenarios.
- The introduction of the InternVideo2.5 text encoder increases model complexity, and its individual contribution is not sufficiently discussed.

## Related Work & Insights

- Compared to CLIP-based methods such as VadCLIP and PEMIL, RefineVAD does not rely on discrete category labels or hand-crafted prompts; instead, it achieves soft semantic injection through learnable prototypes, offering greater adaptability.
- The design paradigm of soft prototypes combined with cross-attention is transferable to other weakly supervised tasks (e.g., weakly supervised action detection and object detection), where category priors can be injected into the feature space to guide localization.
- Cross-dataset transfer experiments demonstrate that the learned semantic space exhibits strong generalizability, offering new directions for label-efficient cross-domain deployment.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The dual-process reasoning framework emulates human cognition; both MoTAR's adaptive shifting and CORE's soft prototype injection are original contributions.
- **Technical Depth**: ⭐⭐⭐⭐ — Mathematical derivations are complete, module designs are well-motivated, and the loss function incorporates fine-grained considerations.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Ablation studies, cross-domain transfer, and visualization analyses are comprehensive, though the number of evaluated datasets is limited.
- **Practicality**: ⭐⭐⭐⭐ — Single-GPU training, low computational overhead, and open-source code availability.
- **Overall**: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Weakly Supervised Video Anomaly Detection with Anomaly-Connected Components and Intention Reasoning](../../CVPR2026/llm_evaluation/weakly_supervised_video_anomaly_detection_with_anomaly-connected_components_and_.md)
- [\[ICLR 2026\] Towards Anomaly-Aware Pre-Training and Fine-Tuning for Graph Anomaly Detection](../../ICLR2026/llm_evaluation/towards_anomaly-aware_pre-training_and_fine-tuning_for_graph_anomaly_detection.md)
- [\[NeurIPS 2025\] Normal-Abnormal Guided Generalist Anomaly Detection](../../NeurIPS2025/llm_evaluation/normal-abnormal_guided_generalist_anomaly_detection.md)
- [\[AAAI 2026\] Sampling Control for Imbalanced Calibration in Semi-Supervised Learning](sampling_control_for_imbalanced_calibration_in_semi-supervised_learning.md)
- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)

</div>

<!-- RELATED:END -->
