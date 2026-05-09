---
title: >-
  [Paper Note] Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning
description: >-
  [CVPR 2026][Medical Imaging][epileptic seizure forecasting] This work introduces the first purely vision-based epileptic seizure forecasting task, leveraging large-scale rodent epilepsy videos for cross-species self-supervised pre-training via the VideoMAE framework, achieving >70% forecasting accuracy within a 3–10 second prediction window.
tags:
  - CVPR 2026
  - Medical Imaging
  - epileptic seizure forecasting
  - video analysis
  - cross-species transfer learning
  - VideoMAE
  - few-shot learning
date: 2026-05-08
content_hash: 7f231cd471ac7fb3
---

# Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning

**Conference**: CVPR 2026
**arXiv**: [2603.12887](https://arxiv.org/abs/2603.12887)
**Code**: N/A
**Area**: Medical Imaging
**Keywords**: epileptic seizure forecasting, video analysis, cross-species transfer learning, VideoMAE, few-shot learning

## TL;DR

This work introduces the first purely vision-based epileptic seizure forecasting task, leveraging large-scale rodent epilepsy videos for cross-species self-supervised pre-training via the VideoMAE framework, achieving >70% forecasting accuracy within a 3–10 second prediction window.

## Background & Motivation

Epileptic seizure forecasting is clinically valuable yet technically challenging. Existing approaches rely predominantly on neural signals such as EEG, requiring specialized equipment that is ill-suited for long-term deployment. Video data offers non-invasive and accessible alternatives; however, prior video-based studies focus mainly on post-ictal detection, leaving pre-ictal forecasting largely unexplored. The core challenges are: (1) annotated human epilepsy video data is extremely scarce due to privacy constraints; and (2) general-purpose video pre-trained models lack epilepsy-relevant behavioral representations. Rodent epilepsy models are widely used in epilepsy research, and their ictal behaviors exhibit cross-species consistency with humans, providing an opportunity for knowledge transfer.

## Method

### Overall Architecture

A two-stage framework is proposed: Stage 1 performs VideoMAE self-supervised pre-training on a cross-species mixed dataset; Stage 2 transfers the pre-trained encoder to few-shot classification on human epilepsy videos.

### Key Designs

1. **Cross-Species Self-Supervised Pre-training**: A mixed dataset $D_{pt} = \{v_r^{(1)}, \ldots, v_r^{(m)}, v_h^{(1)}, \ldots, v_h^{(n)}\}$ is constructed, comprising rodent epilepsy videos (2,952 seizure + 3,000 normal clips from the RodEpil dataset) and 1,870 inter-ictal video clips from 6 human patients. A tube masking strategy (optimal masking ratio = 0.3) is applied, with MSE loss supervising spatiotemporal patch reconstruction: $\mathcal{L}_{MSE} = \frac{1}{\Omega}\sum_{i \in \Omega}(I_i - \hat{I}_i)^2$. The design motivation is to compensate for the scarcity of human epilepsy videos through cross-species data.

2. **Few-Shot Fine-Tuning**: The decoder is discarded; a lightweight classification head is appended to the CLS token to predict seizure probability: $\hat{y} = \sigma(\mathbf{W} \cdot \mathbf{z}_{\text{cls}} + b)$. Model adaptation under data-scarce conditions is evaluated in 2/3/4-shot settings. Gradient checkpointing and 16-bit mixed-precision training are employed to ensure training stability and memory efficiency.

3. **Pre-training Data Ablation Design**: Different data combinations are systematically compared — human-only (+H), seizure rodents only (+R(Y)), normal rodents only (+R(N)), mixed rodents (+R(Y/N)), and the full cross-species combination (+R(Y/N)+H) — to validate the effectiveness of cross-species transfer and the contribution of each component.

### Loss & Training

- Stage 1: MSE reconstruction loss, Adam optimizer with LR = $1 \times 10^{-4}$, 8 × NVIDIA L40 GPUs, tube masking ratio = 0.3
- Stage 2: Binary cross-entropy classification loss, fine-tuned for 20 epochs
- Input sampling: $T=16$ frames, temporal stride 2, resolution $224 \times 224$

## Key Experimental Results

### Datasets

- **Pre-training data**: RodEpil rodent dataset (2,952 seizure + 3,000 normal, 10-second clips) + 1,870 inter-ictal 5-second clips from 6 human patients
- **Evaluation benchmark**: 40 video sequences (20 pre-ictal + 20 inter-ictal), sourced from 2 public and 1 private data sources
- **Evaluation protocol**: 2/3/4-shot independent sampling with non-overlapping support and query sets

### Main Results

| Method | Metric | 2-shot | 3-shot | 4-shot | Avg. |
|--------|--------|--------|--------|--------|------|
| CSN | bacc | 0.339 | 0.588 | 0.656 | 0.528 |
| SlowFast | bacc | 0.578 | 0.680 | 0.728 | 0.662 |
| Human-only | bacc | 0.744 | 0.694 | 0.706 | 0.715 |
| **Ours** | **bacc** | **0.739** | **0.718** | **0.713** | **0.723** |
| **Ours** | **roc_auc** | **0.768** | **0.737** | **0.762** | **0.756** |

### Ablation Study

| Configuration | avg bacc | avg roc_auc | Note |
|---------------|---------|-------------|------|
| Base (Human-only) | 0.715 | 0.749 | Human-data-only baseline |
| +H (unlabeled human) | 0.716 | 0.742 | Marginal gain from unlabeled human data |
| +R(Y) (seizure rodents) | 0.696 | 0.733 | Performance drops with seizure rodents only |
| +R(Y/N) (all rodents) | 0.697 | 0.750 | Mixed rodent data |
| +R(Y/N)+H (full) | **0.723** | **0.756** | Full cross-species combination is optimal |

### Key Findings

- Cross-species transfer learning is effective: the full cross-species configuration achieves the best performance across all averaged metrics.
- The optimal masking ratio is 0.3, substantially lower than the standard VideoMAE setting (0.75–0.9), as seizure forecasting requires preserving richer spatiotemporal context.
- Using seizure rodent data alone (+R(Y)) degrades performance, highlighting the regularization role of normal behavioral data.

## Highlights & Insights

- This work is the first to define a purely vision-based epileptic seizure forecasting task (predicting seizure occurrence within 5 seconds using a 3–10 second window), representing a clinically pioneering contribution.
- The cross-species transfer learning paradigm is novel, exploiting pathological consistency between rodent and human epilepsy for knowledge transfer.
- The finding of a low optimal masking ratio (0.3) reveals a fundamental difference in information density between medical and natural videos.
- The performance degradation when pre-training on seizure samples alone underscores the importance of normal behavior as a contrastive baseline.

## Limitations & Future Work

- The evaluation set comprises only 40 video sequences, limiting statistical power.
- A fixed 5-second prediction window is used; varying prediction horizons are not explored.
- The method is purely visual, without incorporating audio or wearable device signals.
- The theoretical basis of cross-species transfer — specifically which behavioral patterns are truly transferable — lacks in-depth analysis.
- Clinical deployment requires validation on larger-scale, more diverse longitudinal datasets.

## Related Work & Insights

- VideoMAE, as a powerful foundation model for video self-supervised learning, warrants further exploration in medical domains.
- The RodEpil dataset provides a new data resource for cross-species learning research.
- The few-shot learning paradigm is well-suited to data-scarce medical settings, though broader sample diversity is needed to validate robustness.
- The cross-species consistency hypothesis may extend beyond epilepsy to other neurological and behavioral disorders, warranting generalizability studies.
- Complementary fusion with EEG-based methods may represent an important future direction.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First formulation of video-based seizure forecasting; cross-species transfer paradigm is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐ Ablation design is well-conceived, but the dataset size (40 videos) is limited and statistical significance is questionable.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definition is clear and the framework is described thoroughly.
- **Value**: ⭐⭐⭐⭐ Opens a new direction for non-invasive seizure warning systems with significant clinical potential.

## Additional Remarks

The core assumption underlying cross-species transfer learning — that pre-ictal behavioral prodromes share commonalities across species — has partial support in the neuroscience literature. Although validated at limited scale, this work opens new avenues for large-scale follow-up studies. Future integration of multimodal signals (video + HRV + audio) is expected to further improve forecasting performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[CVPR 2026\] Parameter-efficient Prompt Tuning and Hierarchical Textual Guidance for Few-shot Whole Slide Image Classification](parameter-efficient_prompt_tuning_and_hierarchical_textual_guidance_for_few-shot.md)
- [\[CVPR 2026\] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] CRFT: Consistent-Recurrent Feature Flow Transformer for Cross-Modal Image Registration](crft_consistent-recurrent_feature_flow_transformer_for_cross-modal_image_registr.md)

</div>

<!-- RELATED:END -->
