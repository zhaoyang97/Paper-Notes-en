---
title: >-
  [Paper Note] EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal Sentiment Analysis] This paper proposes EBMC, a two-stage framework that first improves the representation quality of weak modalities via semantic disentanglement and cross-modal enhancement, then achieves balanced multimodal sentiment analysis through energy-guided modality coordination and instance-aware trust distillation, maintaining strong robustness under missing-modality scenarios.
tags:
  - CVPR 2026
  - Multimodal VLM
  - Multimodal Sentiment Analysis
  - Modality Imbalance
  - Energy-Based Model
  - Modality Trust Distillation
  - Robustness
date: 2026-05-08
content_hash: 6ac4e4634dedc629
---

# EBMC: Enhance-then-Balance Modality Collaboration for Robust Multimodal Sentiment Analysis

**Conference**: CVPR 2026
**arXiv**: [2604.12518](https://arxiv.org/abs/2604.12518)
**Code**: [https://github.com/kangverse/EBMC](https://github.com/kangverse/EBMC)
**Area**: Multimodal Learning / Sentiment Analysis
**Keywords**: Multimodal Sentiment Analysis, Modality Imbalance, Energy-Based Model, Modality Trust Distillation, Robustness

## TL;DR

This paper proposes EBMC, a two-stage framework that first improves the representation quality of weak modalities via semantic disentanglement and cross-modal enhancement, then achieves balanced multimodal sentiment analysis through energy-guided modality coordination and instance-aware trust distillation, maintaining strong robustness under missing-modality scenarios.

## Background & Motivation

**Background**: Multimodal Sentiment Analysis (MSA) infers sentiment by fusing textual, acoustic, and visual signals, and a large body of work has explored representation learning and multimodal fusion strategies.

**Limitations of Prior Work**: The textual modality consistently dominates prediction, while acoustic and visual signals are underweighted due to weaker or noisier sentiment cues. The dominant modality accumulates larger gradients and reinforces its own representations, leaving weak modalities insufficiently updated — a "Matthew effect" in multimodal learning.

**Key Challenge**: Modality competition progressively marginalizes weaker modalities, especially under noisy or real-world conditions. Existing methods implicitly assume all modalities are equally reliable.

**Goal**: (1) Enhance the representation quality of weak modalities; (2) balance modality contributions to prevent competitive suppression; (3) maintain robustness under missing-modality scenarios.

**Key Insight**: An enhance-then-balance two-stage paradigm — first strengthen weak modalities, then ensure that strong modalities do not suppress weaker ones.

**Core Idea**: Stage I enhances weak modalities through disentanglement and compensation; Stage II employs an energy-based model to coordinate gradients and instance-aware trust distillation to dynamically adjust fusion weights.

## Method

### Overall Architecture

Stage I (Enhancement): Modality Semantic Disentanglement (MSD) separates shared and modality-specific semantics; Cross-modal Complementary Enhancement (CCE) strengthens weak modalities. Stage II (Balance): Energy-guided Modality Coordination (EMC) aligns optimization dynamics; Instance-aware Modality Trust Distillation (IMTD) adaptively adjusts fusion weights.

### Key Designs

1. **Energy-guided Modality Coordination (EMC)**:

    - Function: Rebalances modality optimization via energy potentials and gradient flow dynamics.
    - Mechanism: Introduces energy-based models (EBMs) into modality coordination for the first time. The learning state of each modality is mapped to an energy potential, and energy differentials across modalities drive implicit gradient rebalancing. A differentiable balancing objective equalizes gradient contributions across modalities.
    - Design Motivation: Existing methods adjust learning rates or gradients heuristically; EMC provides a principled, physically-intuitive alternative.

2. **Instance-aware Modality Trust Distillation (IMTD)**:

    - Function: Estimates modality reliability at the sample level to adaptively adjust fusion weights.
    - Mechanism: Estimates per-modality reliability for each sample from probabilistic teacher signals, and dynamically modulates fusion weights accordingly. Unreliable modalities receive reduced weights on noisy or missing samples.
    - Design Motivation: Modality reliability varies across samples (e.g., clear visual but noisy audio), necessitating instance-level adaptation rather than static weighting.

3. **Modality Semantic Disentanglement + Cross-modal Complementary Enhancement (MSD + CCE)**:

    - Function: Separates and enhances modality-specific and shared semantics.
    - Mechanism: MSD decomposes each modality into shared and modality-specific semantic components. CCE leverages complementary cues from strong modalities (typically text) to enhance weak modalities (audio, visual) via cross-modal attention, transferring discriminative information.
    - Design Motivation: Weak modalities must attain sufficient representational capacity before the balancing stage can be effective.

### Loss & Training

Multi-task loss: sentiment prediction loss + disentanglement orthogonality constraint + energy balancing objective + trust distillation KL divergence. The two stages are trained alternately.

## Key Experimental Results

### Main Results

| Method | MOSI Acc7↑ | MOSI MAE↓ | MOSEI Acc7↑ | IEMOCAP Acc↑ |
|--------|-----------|----------|------------|-------------|
| MISA | 42.3 | 0.783 | 52.1 | 68.5 |
| Self-MM | 43.5 | 0.768 | 53.2 | 69.7 |
| UniMSE | 44.1 | 0.752 | 54.3 | 70.8 |
| **EBMC** | **45.8** | **0.731** | **55.6** | **72.3** |

### Ablation Study

| Configuration | MOSI Acc7 | Note |
|---------------|----------|------|
| Full EBMC | 45.8 | All components |
| w/o EMC | 43.9 | No energy coordination |
| w/o IMTD | 44.2 | No trust distillation |
| w/o CCE | 44.5 | No cross-modal compensation |
| w/o MSD | 44.8 | No semantic disentanglement |

### Key Findings

- EMC contributes the most (1.9% drop upon removal), confirming that modality coordination is the central challenge.
- EBMC degrades significantly less than baselines under missing-modality conditions, demonstrating robustness.
- Consistent improvements are observed when transferring to the Emotion Recognition in Conversation (ERC) task.

## Highlights & Insights

- Introducing EBMs into modality coordination is a physically-intuitive innovation: energy potentials naturally encode the learning state of each modality.
- The enhance-then-balance two-stage paradigm is generalizable and transferable to other multimodal learning settings.
- Instance-aware trust distillation addresses the inherent limitations of static fusion weights.

## Limitations & Future Work

- Performance gains on small datasets such as IEMOCAP are limited.
- Hyperparameter tuning of the energy-based model may affect training stability.
- Scenarios involving more than four modalities remain unexplored.
- EMC could be applied to modality balancing in vision-language pre-training.

## Related Work & Insights

- **vs. MISA**: MISA performs modality disentanglement but does not address modality imbalance; EBMC extends disentanglement with energy-guided coordination.
- **vs. OGM-GE**: OGM-GE balances modalities via gradient manipulation; EBMC's EBM-based approach is more principled.

## Rating

- Novelty: ⭐⭐⭐⭐ EBM-based modality coordination is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three datasets + missing-modality evaluation + ERC transfer.
- Writing Quality: ⭐⭐⭐⭐ The two-stage structure is clearly articulated.
- Value: ⭐⭐⭐⭐ Provides meaningful reference for robust multimodal learning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Purify-then-Align: Towards Robust Human Sensing under Modality Missing with Knowledge Distillation from Noisy Multimodal Teacher](purify-then-align_towards_robust_human_sensing_under_modality_missing_with_knowl.md)
- [\[CVPR 2026\] MASQuant: Modality-Aware Smoothing Quantization for Multimodal Large Language Models](masquant_modality-aware_smoothing_quantization_for_multimodal_large_language_mod.md)
- [\[CVPR 2026\] Disentangle-then-Align: Non-Iterative Hybrid Multimodal Image Registration via Cross-Scale Feature Disentanglement](disentangle-then-align_non-iterative_hybrid_multimodal_image_registration_via_cr.md)
- [\[CVPR 2026\] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)
- [\[CVPR 2026\] ChartNet: A Million-Scale, High-Quality Multimodal Dataset for Robust Chart Understanding](chartnet_a_million-scale_high-quality_multimodal_dataset_for_robust_chart_unders.md)

</div>

<!-- RELATED:END -->
