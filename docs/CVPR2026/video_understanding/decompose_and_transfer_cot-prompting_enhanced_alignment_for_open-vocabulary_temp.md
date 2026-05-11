---
title: >-
  [Paper Note] Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection
description: >-
  [CVPR 2026][Video Understanding][open-vocabulary temporal action detection] This paper proposes the Phase-wise Decomposition and Alignment (PDA) framework…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "open-vocabulary temporal action detection"
  - "chain-of-thought prompting"
  - "action phase decomposition"
  - "cross-modal alignment"
  - "knowledge transfer"
date: 2026-05-08
content_hash: b6f8289f22012787
---

# Decompose and Transfer: CoT-Prompting Enhanced Alignment for Open-Vocabulary Temporal Action Detection

**Conference**: CVPR 2026
**arXiv**: [2603.24030](https://arxiv.org/abs/2603.24030)
**Code**: N/A
**Area**: Video Understanding
**Keywords**: open-vocabulary temporal action detection, chain-of-thought prompting, action phase decomposition, cross-modal alignment, knowledge transfer

## TL;DR
This paper proposes the Phase-wise Decomposition and Alignment (PDA) framework, which leverages the CoT reasoning capability of LLMs to decompose action labels into start–middle–end phase descriptions. Through text-guided foreground filtering and adaptive phase-wise alignment, PDA achieves fine-grained action pattern transfer, attaining an Avg mAP of 46.9 on THUMOS14 OV-TAD, surpassing the previous SOTA Ti-FAD (41.2).

## Background & Motivation
**Background**: Open-vocabulary temporal action detection (OV-TAD) requires localizing and classifying unseen action categories, with knowledge transfer from seen categories as the central challenge.

**Limitations of Prior Work**: Existing methods perform only label-level global text–visual alignment, making it difficult to capture fine-grained temporal patterns shared across different actions. For instance, "LongJump" and "PoleVault" exhibit low label-level similarity, yet their run-up and take-off phases are visually highly similar.

**Key Challenge**: Label-level semantic alignment fails to discover transferable visual patterns across categories, limiting generalization to unseen classes.

**Goal**: To extract and transfer shared phase-level visual priors across different actions for improved open-vocabulary generalization.

**Key Insight**: Simulating human cognition—understanding an action as a sequential unfolding (initiation → execution → completion)—by exploiting the CoT capability of LLMs to automatically decompose actions into multiple phases.

**Core Idea**: Decompose action labels into phase descriptions → perform text–visual alignment independently for each phase → adaptively aggregate the alignment results across phases.

## Method

### Overall Architecture
Three core modules: CSD (CoT-Prompted Semantic Decomposition) → TIF (Text-Infused Foreground Filtering) → APA (Adaptive Phase-wise Alignment). Input videos are processed by a visual encoder for feature extraction; action labels are decomposed by GPT-4o into four phase descriptions—start, middle, end, and global—with each phase undergoing independent visual–text matching followed by adaptive aggregation.

### Key Designs
1. **CoT-Prompting Semantic Decomposition (CSD)**: GPT-4o's CoT reasoning is employed to decompose each action label into four phase descriptions: {start, middle, end, global}. For example, "LongJump" yields: start = "accelerating along the runway," mid = "planting and taking off," end = "landing in the sandpit." The CLIP text encoder extracts phase embeddings $t_c^p = \Phi_{txt}(s_c^p)$. **Design Motivation**: Label-level semantics cannot express phase patterns shared across categories, whereas phase decomposition naturally exposes these transferable knowledge structures.

2. **Text-Infused Foreground Filtering (TIF)**: For each phase $p$, the cosine similarity between the phase text embedding and video features is computed; a max-then-Softmax operation produces a phase-level foreground confidence score $S_{fg}^p$, which is binarized to filter phase-relevant video segments: $F_v^p = \hat{S}_{fg}^p \cdot F_v$. **Design Motivation**: Naively partitioning video into uniform segments cannot handle real-world scenarios with multiple actions and variable durations; semantic phase-aware segment selection is required.

3. **Adaptive Phase-wise Alignment (APA)**: Cross-attention fusion is performed independently for each phase, $\bar{F}_v^p = \text{CrossAttn}(F_v^p, F_t^p)$, followed by classification score computation $C_{cls}^p = \bar{F}_v^p \cdot F_t^{p\top}$. **Adaptive aggregation** uses a Sigmoid network to predict per-phase weights $\omega_p$, yielding the final classification $C_{cls} = \sum_{p} \omega_p \cdot C_{cls}^p$. **Design Motivation**: The discriminability of each phase varies across actions—some actions are identifiable from the beginning while others require the ending—making adaptive weighting preferable to simple averaging.

### Loss & Training
- Total loss: $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{fg} + \mathcal{L}_{loc}$, comprising classification (cross-entropy), foreground awareness, and DIoU localization losses.
- At inference, the same LLM-based phase decomposition is applied to test categories, and SoftNMS is used for redundancy suppression.

## Key Experimental Results

### Main Results (THUMOS14, 50% Seen / 50% Unseen)

| Method | 0.3 | 0.5 | 0.7 | Avg mAP |
|--------|-----|-----|-----|---------|
| Ti-FAD (NeurIPS'24) | 57.0 | 43.3 | 21.2 | 41.2 |
| STOV (WACV'25) | 56.3 | 34.4 | 11.3 | 34.0 |
| **PDA (Ours)** | **65.4** | **49.7** | **24.3** | **46.9** |

**ActivityNet v1.3**

| Method | 0.5 | 0.75 | Avg mAP |
|--------|-----|------|---------|
| Ti-FAD | 50.6 | 32.2 | 32.0 |
| **PDA (Ours)** | **53.1** | **35.3** | **34.6** |

### Ablation Study

| Configuration | Avg mAP | Notes |
|---------------|---------|-------|
| Global alignment baseline | ~41.2 | Label-level alignment only |
| + CSD | Improved | Phase decomposition exposes transferable patterns |
| + CSD + TIF | Further improved | Adaptive foreground filtering vs. static temporal partitioning |
| + CSD + TIF + APA | **46.9** | Adaptive weighting outperforms average aggregation |

### Key Findings
- On the THUMOS14 50/50 split, PDA achieves a 5.7-point improvement in Avg mAP over the strongest baseline, Ti-FAD.
- In the cross-category transfer case of LongJump → PoleVault, phase decomposition enables the model to identify shared "run-up acceleration" and "take-off" patterns, substantially improving detection performance on unseen categories.
- Adaptive aggregation demonstrates greater flexibility compared to simple averaging.

## Highlights & Insights
- CoT reasoning is extended from NLP to action understanding: rather than mere text augmentation, this constitutes structured temporal decomposition that directly corresponds to the cognitive process of action understanding.
- Phase decomposition naturally exposes transferable cross-category knowledge that label-level methods cannot access.
- TIF's text-guided foreground filtering outperforms static temporal partitioning and handles multi-action and variable-duration scenarios.
- In the LongJump → PoleVault transfer case, phase decomposition enables the model to recognize shared "run-up acceleration" and "take-off" patterns.
- Evaluation under the 75/25 split confirms the method's robustness across different seen/unseen ratios.
- On THUMOS14, mAP at IoU@0.5 improves from 43.3 to 49.7 (+6.4%), indicating that fine-grained alignment also enhances localization precision.

## Limitations & Future Work
- The method relies on GPT-4o for phase decomposition, incurring high cost, and decomposition quality is bounded by the LLM's action knowledge.
- The fixed three-phase decomposition (start/middle/end) may lack flexibility for certain action types (e.g., cyclic actions).
- Adaptive determination of the number of phases remains unexplored.
- The quality of phase description encoding by the CLIP text encoder may become a bottleneck.
- Validation on larger-scale video datasets (e.g., Kinetics) has not been conducted.
- CoT decomposition quality may vary considerably across different LLMs.

## Related Work & Insights
- **Distinction from DeTAL and Ti-FAD**: These methods employ global alignment or simple text augmentation, whereas the proposed framework achieves fine-grained knowledge transfer through structured phase decomposition.
- The application of CoT prompting to visual tasks is an emerging direction; this work demonstrates its potential for temporal understanding.
- The design of adaptive phase weights is generalizable to other tasks requiring multi-granularity alignment.
- Under the 75/25 split, Avg mAP improves from 42.9 (Ti-FAD) to 47.3, demonstrating generalization across different ratios.

## Technical Details
- **GPT-4o Prompt**: "Decompose the action of ⟨Action⟩ into coherent three phases based on the natural temporal progression"
- **Phase text template**: 'a video of people's motion that [Description]'
- **Cross-attention fusion**: $\bar{F}_v^p = \text{Softmax}(\frac{Q(F_v^p)K(F_t^p)^\top}{\sqrt{D}})V(F_t^p)$
- **Adaptive weights**: $\omega_p = \text{Sigmoid}(W_p(F_v^p))$, allowing different phases to carry different importance across actions
- **Localization branch**: Concatenation of all phase visual features → MLP projection → foreground-aware head + regression head
- **Inference**: Test categories are similarly decomposed into phases by LLM; SoftNMS is applied for redundancy removal
- **75/25 split results**: THUMOS14 Avg mAP 47.3 (vs. Ti-FAD 42.9); ActivityNet Avg mAP 36.6 (vs. DeTAL 25.5)
- **Training objective**: $\mathcal{L} = \mathcal{L}_{cls} + \mathcal{L}_{fg} + \mathcal{L}_{loc}$, combining classification, foreground awareness, and DIoU localization
- **Phase set**: $\mathcal{P} = \{start, middle, end, glob\}$, comprising 4 phases
- **Foreground binarization threshold**: The mean similarity across all temporal positions serves as the binarization threshold
- **Compatible visual encoders**: Compatible with standard visual encoders such as CLIP ViT-B/16

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of CoT prompting and phase decomposition is novel and cognitively well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated on two benchmarks (THUMOS14 and ActivityNet) under multiple split settings.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation figures are intuitive, though the paper is notation-heavy.
- **Value**: ⭐⭐⭐⭐ Achieves significant gains on OV-TAD, though the task's application scope is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SAVA-X: Ego-to-Exo Imitation Error Detection via Scene-Adaptive View Alignment and Bidirectional Cross View Fusion](savax_egotoexo_imitation_error_detection_via_scene.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[ICCV 2025\] Learning to Generalize Without Bias for Open-Vocabulary Action Recognition](../../ICCV2025/video_understanding/learning_to_generalize_without_bias_for_open-vocabulary_action_recognition.md)
- [\[CVPR 2026\] OpenMarcie: Dataset for Multimodal Action Recognition in Industrial Environments](openmarcie_dataset_for_multimodal_action_recognition_in_industrial_environments.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)

</div>

<!-- RELATED:END -->
