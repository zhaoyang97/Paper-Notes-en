---
title: >-
  [Paper Note] Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition
description: >-
  [NEURIPS2025][Video Understanding][action recognition] A cross-attention multimodal architecture is proposed that integrates V-JEPA 2 visual context features with CoMotion 3D skeletal pose data…
tags:
  - "NEURIPS2025"
  - "Video Understanding"
  - "action recognition"
  - "multimodal fusion"
  - "3D human pose"
  - "cross-attention"
  - "V-JEPA 2"
  - "CoMotion"
date: 2026-05-08
content_hash: f8b7744a1a01419e
---

# Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition

**Conference**: NEURIPS2025
**arXiv**: [2511.05622](https://arxiv.org/abs/2511.05622)  
**Code**: [nbabey20/groundactrec](https://github.com/nbabey20/groundactrec)  
**Area**: Video Understanding
**Keywords**: action recognition, multimodal fusion, 3D human pose, cross-attention, V-JEPA 2, CoMotion

## TL;DR

A cross-attention multimodal architecture is proposed that integrates V-JEPA 2 visual context features with CoMotion 3D skeletal pose data, outperforming unimodal baselines on standard and high-occlusion action recognition benchmarks.

## Background & Motivation

- Action recognition is a core capability for embodied AI, yet existing methods exhibit clear modality-specific limitations.
- **RGB video methods** (e.g., V-JEPA 2) capture rich scene context and object interaction information, but spatial reasoning degrades in heavily occluded scenarios where key body parts are hidden.
- **Skeleton-based methods** (e.g., CoMotion) provide explicit 3D joint coordinates and offer some robustness to visual noise and occlusion, but lack environmental context and person–object interaction information.
- The two paradigms have complementary blind spots: the visual stream conveys "what is being done and in what environment," while the skeleton stream conveys "how the body is configured and moving."
- The central hypothesis of this work is that combining contextual visual understanding with precise geometric skeletal representations yields more robust and spatially aware action recognition.

## Core Problem

1. How can high-level semantic features from pretrained visual foundation models be effectively fused with low-level geometric features from 3D human poses?
2. Can the fusion architecture significantly outperform unimodal approaches in high-occlusion scenarios?
3. How much does cross-attention fusion gain over early fusion (concatenation) and late fusion (score averaging)?

## Method

### Overall Architecture

The model consists of two feature extraction branches, a cross-attention fusion Transformer, and a classification head.

### Visual Feature Extraction

- $T=64$ frames are uniformly sampled from each video (TSN sampling: 8 segments × 8 frames/segment).
- Each frame is processed by the ViT-g/384 encoder of V-JEPA 2, extracting the [CLS] token as a frame-level feature.
- This yields a visual feature sequence $F_V \in \mathbb{R}^{T \times 1408}$.

### Skeletal Feature Extraction

- CoMotion predicts SMPL parameters (pose $\theta$, translation $t$, shape $\beta$) for each frame.
- The SMPL layer decodes $J=24$ 3D joint coordinates.
- Root-relative normalization is applied (subtracting pelvis coordinates) to remove global position influence.
- Features are flattened into a $D_S = 3 \times 24 = 72$-dimensional vector.
- TSN sampling is applied in alignment with the visual stream, yielding $F_S \in \mathbb{R}^{T \times 72}$.

### Modality Embedding and Positional Encoding

- Two independent linear projection layers map $F_V$ and $F_S$ to a shared dimension $D_{model}=512$.
- A learnable [CLS] token is prepended to each stream.
- Sinusoidal positional encoding is added to preserve temporal information.

### Cross-Attention Fusion Transformer

- $L=4$ fusion layers are used; each layer comprises:
    - **Bidirectional cross-attention**: the visual stream attends to the skeleton stream as K/V, and the skeleton stream symmetrically attends to the visual stream as K/V.
    - **Self-attention refinement**: each stream independently performs self-attention to consolidate fused features.
    - Residual connections and LayerNorm are applied after each sub-layer.
- 8 attention heads are used.

### Classification Head

- The [CLS] tokens from both streams in the final layer are concatenated and passed through an MLP with softmax to produce action class probabilities.

## Key Experimental Results

### Datasets

| Dataset | Description | Scale |
|--------|------|------|
| InHARD | Industrial action recognition; 16 subjects, 14 action classes, 2M+ frames | Standard train/val split |
| UCF-19-Y-OCC | High-occlusion subset of UCF-101; 19 classes, 1,732 videos | Real-world natural occlusion |

### Main Results (InHARD)

| Model | Top-1 Acc (%) | Macro mAP (%) | Macro F1 (%) |
|------|-------------|--------------|-------------|
| V-JEPA 2 baseline | 80.76 | 80.93 | 76.24 |
| CoMotion baseline | 75.92 | 74.60 | 69.52 |
| Gated recursive fusion | 79.25 | 76.90 | 73.69 |
| **Fusion (cross-attn)** | **83.47** | **84.96** | **80.21** |

### Main Results (UCF-19-Y-OCC, High Occlusion)

| Model | Top-1 Acc (%) | Macro mAP (%) | Macro F1 (%) |
|------|-------------|--------------|-------------|
| V-JEPA 2 baseline | 31.83 | **58.48** | 14.23 |
| CoMotion baseline | 6.20 | 8.84 | 1.72 |
| Gated recursive fusion | 29.54 | 50.07 | 11.44 |
| **Fusion (cross-attn)** | **38.62** | 54.10 | **16.30** |

### Ablation Study (Fusion Strategy Comparison)

| Fusion Method | InHARD Acc | UCF-19-Y-OCC Acc |
|---------|-----------|-----------------|
| Early fusion (concat) | 79.52 | 33.34 |
| Late fusion (score avg) | 80.24 | 34.42 |
| **Cross-attention** | **83.47** | **38.62** |

### Training Configuration

- GPU: NVIDIA A100 SXM; 30 epochs
- AdamW optimizer, lr = $3 \times 10^{-4}$, weight decay = 0.05
- Cosine learning rate decay with 5% warmup
- Batch size = 128, dropout = 0.1, gradient clipping norm = 1.0
- Results averaged over 3 random seeds with reported standard deviation

## Highlights & Insights

1. **Clear complementarity design rationale**: Combining V-JEPA 2's implicit world-model understanding with CoMotion's explicit geometric skeletal representation is well-motivated and conceptually coherent.
2. **Substantial gains under high occlusion**: On UCF-19-Y-OCC, the fusion model surpasses the V-JEPA 2 baseline by 6.79% in top-1 accuracy, while CoMotion alone nearly collapses under occlusion (6.20%), confirming genuine complementarity.
3. **Cross-attention outperforms naive fusion**: Ablation experiments clearly demonstrate that cross-attention captures complex cross-modal relationships more effectively than early or late fusion.
4. **Concise and reproducible architecture**: With 4-layer fusion Transformer, shared dimension of 512, and standard components, the architecture is moderately straightforward to implement.

## Limitations & Future Work

1. **Reliance on frozen feature extractors**: Both V-JEPA 2 and CoMotion are used as fixed feature extractors without end-to-end fine-tuning, which may result in suboptimal features.
2. **Small dataset scale**: InHARD covers only 14 industrial action classes, and UCF-19-Y-OCC contains only 19 classes and 1,732 videos; generalizability on larger benchmarks (e.g., Kinetics-400, AVA) remains to be verified.
3. **Low absolute performance under high occlusion**: The best top-1 accuracy on UCF-19-Y-OCC is only 38.62%, indicating that occluded action recognition remains an open problem.
4. **Fusion does not dominate on mAP in the occlusion set**: V-JEPA 2 achieves a higher mAP (58.48%) than the fusion model (54.10%) on UCF-19-Y-OCC, showing that fusion does not provide uniform advantages.
5. **Limited fusion baselines**: Only gated recursive fusion is compared against; recent methods such as ATFusion and MMAct are absent from the experimental comparison.
6. **Computational cost not analyzed**: Simultaneously running ViT-g and CoMotion inference is computationally expensive, yet the paper provides no analysis of latency or computational overhead.

## Related Work & Insights

- **V-JEPA 2**: A self-supervised video pretraining model offering strong contextual understanding but without explicit human pose modeling; used in this work as the visual feature backbone.
- **CoMotion**: A multi-person 3D pose tracker capable of recovering skeletons under occlusion but lacking scene context; used as the skeletal feature backbone.
- **Gated Recursive Fusion**: A gated recursive fusion architecture that underperforms cross-attention fusion in this paper's experiments.
- **ST-GCN / Skeleton GCN Methods**: Graph convolution-based skeleton action recognition methods that do not involve multimodal fusion.
- **ATFusion / Other Multimodal Methods**: Mentioned in related work but not included as experimental baselines.

The core insight of this work—combining a world model's implicit understanding with explicit geometric data—generalizes to other tasks requiring spatial reasoning, such as hand manipulation recognition and human–robot collaboration. Cross-attention fusion is a general paradigm applicable to other multimodal combinations such as video+depth or video+optical flow. The challenges posed by high-occlusion scenarios suggest that 2D view information alone is insufficient, and that 3D geometric priors play an irreplaceable role in occlusion reasoning. As a workshop paper, the scope and experiments are limited, but the approach has potential to develop into a full-length contribution.

## Rating

- Novelty: 3/5 — The idea of fusing visual foundation models with 3D poses is reasonable but not entirely novel; cross-attention fusion is a standard technique.
- Experimental Thoroughness: 2.5/5 — Dataset scale is small, baselines are limited, and computational cost analysis is absent.
- Writing Quality: 3.5/5 — Structure is clear, motivation is well articulated, and mathematical notation is consistent.
- Value: 3/5 — Workshop-paper level; validates the complementarity hypothesis but requires larger-scale experiments for broader support.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-Based Human Action Recognition](../../ICCV2025/video_understanding/adaptive_hyper-graph_convolution_network_for_skeleton-based_human_action_recogni.md)
- [\[ICCV 2025\] Adaptive Hyper-Graph Convolution Network for Skeleton-based Human Action Recognition with Virtual Connections](../../ICCV2025/video_understanding/adaptive_hyper_graph_convolution_network_skeleton_action_recognition.md)
- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[NeurIPS 2025\] egoEMOTION: Egocentric Vision and Physiological Signals for Emotion and Personality Recognition in Real-World Tasks](egoemotion_egocentric_vision_and_physiological_signals_for_emotion_and_personali.md)
- [\[NeurIPS 2025\] Seeing Beyond the Scene: Analyzing and Mitigating Background Bias in Action Recognition](seeing_beyond_the_scene_analyzing_and_mitigating_background_bias_in_action_recog.md)

</div>

<!-- RELATED:END -->
