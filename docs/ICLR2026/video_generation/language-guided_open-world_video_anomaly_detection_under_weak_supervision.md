---
title: >-
  [Paper Note] Language-guided Open-world Video Anomaly Detection under Weak Supervision
description: >-
  [ICLR 2026][Video Generation][Video Anomaly Detection] This paper proposes LaGoVAD, a language-guided open-world video anomaly detection paradigm that models anomaly definitions as random variables provided in natural la…
tags:
  - "ICLR 2026"
  - "Video Generation"
  - "Video Anomaly Detection"
  - "Open-world"
  - "Language-guided"
  - "Concept Drift"
  - "Weak Supervision"
date: 2026-05-08
content_hash: e49d88ffe42fe911
---

# Language-guided Open-world Video Anomaly Detection under Weak Supervision

**Conference**: ICLR 2026
**arXiv**: [2503.13160](https://arxiv.org/abs/2503.13160)  
**Code**: [GitHub](https://github.com/Kamino666/LaGoVAD-PreVAD)  
**Area**: Video Generation
**Keywords**: Video Anomaly Detection, Open-world, Language-guided, Concept Drift, Weak Supervision

## TL;DR

This paper proposes LaGoVAD, a language-guided open-world video anomaly detection paradigm that models anomaly definitions as random variables provided in natural language. Combined with dynamic video synthesis and contrastive learning regularization, it achieves zero-shot state-of-the-art performance across seven datasets.

## Background & Motivation

1. **Background**: Video Anomaly Detection (VAD) aims to identify video frames that deviate from expected patterns and is widely applied in intelligent surveillance. Weakly supervised methods have achieved strong performance under closed-set settings in recent years.

2. **Limitations of Prior Work**: Existing methods assume a fixed anomaly definition and cannot handle scenarios in open-world settings where the definition may shift with changing requirements. For example, not wearing a mask is anomalous during a flu outbreak but normal otherwise — a manifestation of concept drift.

3. **Key Challenge**: Open-set and domain generalization methods can detect novel anomaly categories outside the training set, but still assume a static anomaly definition. They fail when the label of the same behavior changes across contexts (e.g., a pedestrian walking on a road is normal in a crime dataset but anomalous in highway surveillance).

4. **Goal**: To propose an open-world VAD paradigm that allows users to dynamically define anomalies in natural language at inference time, fundamentally eliminating concept drift.

5. **Key Insight**: The anomaly definition $Z$ is explicitly modeled as a random variable, and prediction is conditioned on the joint function $\Phi:(V,Z)\rightarrow Y$ of video $V$ and definition $Z$, keeping $P(Y|V,Z)$ invariant and theoretically eliminating concept drift.

6. **Core Idea**: By treating the anomaly definition as an input condition, the model learns a joint mapping over video–text–label triplets, supported by large-scale diverse datasets for generalization.

## Method

### Overall Architecture

LaGoVAD adopts a dual-branch architecture. The video branch extracts video features $v^t = \mathcal{F}(v)$ via a pretrained CLIP image encoder and a Transformer temporal encoder; the text branch extracts anomaly definition features $z^t = \mathcal{G}(z)$ via a CLIP text encoder. The two feature sets are fused by a Transformer fusion module $\mathcal{U}$ and passed into a binary detection head $\mathcal{H}^{\text{bin}}$ and a multi-class head $\mathcal{H}^{\text{mul}}$. Training employs four loss functions: MIL loss, MIL-align loss, dynamic video synthesis loss $\mathcal{L}_{\text{dvs}}$, and contrastive learning loss $\mathcal{L}_{\text{neg}}$.

### Key Designs

**1. Dynamic Video Synthesis**

- **Function**: Increases diversity in the proportion of anomaly duration in training data.
- **Mechanism**: Videos of varying lengths are synthesized dynamically at runtime. The module first determines whether to generate a normal or anomalous video, then decides on the number of segments, and concatenates semantically similar clips selected from K-nearest neighbors. Anchor positions are converted into binary pseudo-labels $y^p \in \{0,1\}^L$, supervised by $\mathcal{L}_{\text{dvs}}$.
- **Design Motivation**: In real scenes, anomalies typically occupy only a small portion of a video, whereas web-sourced datasets tend to have a higher anomaly ratio. Synthesizing videos with varying temporal anomaly proportions alleviates this distributional bias.

**2. Contrastive Learning with Hard Negative Mining**

- **Function**: Enhances the discriminability of normal/anomalous frame features and achieves fine-grained cross-modal alignment.
- **Mechanism**: Frame-level video features are aggregated into video-level features weighted by anomaly scores. Normal segments within anomalous videos serve as hard negatives and are contrasted against the corresponding anomaly descriptions. The loss includes bidirectional contrastive terms in both text→video and video→text directions.
- **Design Motivation**: Boundaries between normal and anomalous frames within anomalous videos are ambiguous, necessitating contrastive learning to enhance discriminability. The exponentially decaying sample density in the joint multimodal space also calls for a stronger alignment strategy.

**3. PreVAD Large-scale Pretraining Dataset**

- **Function**: Provides diverse $(v, z, y)$ triplets to support training of the language-guided paradigm.
- **Mechanism**: A scalable data pipeline aggregates videos from video-text datasets, web sources, and surveillance streams, with automated cleaning and annotation via MLLMs. The dataset contains 35,279 videos, 35 subcategories across 7 major anomaly categories, and textual descriptions for each anomalous video.
- **Design Motivation**: Existing datasets are small (up to 5K videos), have limited domain coverage, and lack semantic description annotations, rendering them insufficient for training an open-world paradigm.

### Loss & Training

The total loss is the sum of four terms: $\mathcal{L} = \mathcal{L}_{\text{MIL}} + \mathcal{L}_{\text{MIL-align}} + \mathcal{L}_{\text{dvs}} + \mathcal{L}_{\text{neg}}$

- $\mathcal{L}_{\text{MIL}}$: Multiple instance learning loss for temporal binary classification detection.
- $\mathcal{L}_{\text{MIL-align}}$: MIL alignment loss for video-level multi-class classification.
- $\mathcal{L}_{\text{dvs}}$: Dynamic video synthesis loss (Eqs. 7–8), supervised by pseudo-labels from synthesized videos.
- $\mathcal{L}_{\text{neg}}$: Contrastive loss (Eqs. 10–11), bidirectional contrastive learning with hard negative mining.

## Key Experimental Results

### Main Results

Protocol 1: Zero-shot cross-dataset binary anomaly detection (AUC/AP)

| Dataset | Metric | Ours (LaGoVAD) | Prev. SOTA | Gain |
|---------|--------|----------------|------------|------|
| UCF-Crime | AUC | 82.81 | 82.42 (OVVAD) | +0.39 |
| XD-Violence | AP | 76.28 | 63.74 (OVVAD) | +12.54 |
| MSAD | AUC | 88.09+ | — | — |
| DoTA | AUC | Outperforms all baselines | — | — |
| TAD | AUC | Outperforms all baselines | — | — |

Protocol 2: Concept drift evaluation (drift@5). LaGoVAD performs consistently across varying anomaly definitions, outperforming VadCLIP and LLM-based methods.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Full model | Best | All components working jointly |
| w/o Dynamic Video Synthesis | Degraded | Insufficient anomaly duration diversity leads to overfitting |
| w/o Contrastive Learning | Degraded | Reduced feature alignment quality |
| w/o Text Branch | Significantly degraded | Degenerates to fixed-definition mode; cannot handle concept drift |

### Key Findings

- Detection and classification on XD-Violence improve by 20% and 32% respectively, demonstrating the substantial advantage of the language-guided paradigm for cross-domain generalization.
- The scale and diversity of PreVAD are critical to model performance; the 35K diverse training videos are key to generalization.
- The concept drift evaluation protocol (drift@5) confirms that the model effectively handles dynamic changes in anomaly definitions.

## Highlights & Insights

- **Solid Theoretical Contribution**: Concept drift is formally characterized through a probabilistic framework, with proof that conditioning on the anomaly definition as input eliminates concept drift, tightly coupling theory and practice.
- **Paradigm Innovation**: The shift from fixed anomaly definitions to dynamic language-guided definitions establishes a new paradigm in the VAD field.
- **Large-scale Dataset**: PreVAD is currently the largest and most diverse video anomaly detection dataset, holding independent evaluation value.
- **Strong Practicality**: Users can flexibly define anomalies in natural language to adapt to diverse scenario requirements.

## Limitations & Future Work

- Reliance on CLIP as the backbone may inherit its limitations in fine-grained visual understanding.
- Although large, the dataset is predominantly composed of web videos, creating a domain gap with real surveillance scenarios.
- Inference requires users to supply appropriate anomaly definition text, and detection quality is directly affected by definition quality.
- Replacing CLIP feature extraction with stronger video understanding models (e.g., VideoLLMs) is worth exploring.

## Related Work & Insights

- Open-vocabulary methods such as OVVAD can detect novel categories but still assume fixed definitions; the "definition as input" approach in this work is more flexible.
- Compared to LLM-based methods such as LAVAD, LaGoVAD achieves better performance while remaining lightweight.
- The dynamic video synthesis strategy is generalizable to data augmentation in other video understanding tasks.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Pioneering formalization of the concept drift problem and proposal of a language-guided VAD paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Zero-shot evaluation across seven datasets with two evaluation protocols — highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear logic and rigorous theoretical derivation.
- Value: ⭐⭐⭐⭐⭐ A paradigm-level contribution; the PreVAD dataset also holds independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] MiVE: Multiscale Vision-language features for reference-guided video Editing](../../ICML2026/video_generation/mive_multiscale_vision-language_features_for_reference-guided_video_editing.md)
- [\[NeurIPS 2025\] DisMo: Disentangled Motion Representations for Open-World Motion Transfer](../../NeurIPS2025/video_generation/dismo_disentangled_motion_representations_for_openworld_moti.md)
- [\[ICLR 2026\] LoRA-Edit: Controllable First-Frame-Guided Video Editing via Mask-Aware LoRA Fine-Tuning](lora-edit_controllable_first-frame-guided_video_editing_via_mask-aware_lora_fine.md)
- [\[CVPR 2026\] LAMP: Language-Assisted Motion Planning for Controllable Video Generation](../../CVPR2026/video_generation/lamp_language-assisted_motion_planning_for_controllable_video_generation.md)
- [\[AAAI 2026\] GenVidBench: A 6-Million Benchmark for AI-Generated Video Detection](../../AAAI2026/video_generation/genvidbench_a_6-million_benchmark_for_ai-generated_video_detection.md)

</div>

<!-- RELATED:END -->
