---
title: >-
  [Paper Note] Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning
description: >-
  [ACL 2025][Object Detection][noisy label learning] This paper proposes the Delora framework, which constructs a noisy label detector by introducing clean LoRA and noisy LoRA modules. By decoupling sample selection from model training, Delora breaks the vicious catch-22 cycle of mutual influence between sample selection and training in traditional "small-loss" approaches.
tags:
  - "ACL 2025"
  - "Object Detection"
  - "noisy label learning"
  - "LoRA"
  - "sample selection"
  - "parameter-efficient fine-tuning"
  - "LLM fine-tuning"
date: 2026-05-08
content_hash: 1f2856178959705d
---

# Weed Out, Then Harvest: Dual Low-Rank Adaptation is an Effective Noisy Label Detector for Noise-Robust Learning

**Conference**: ACL 2025  
**arXiv**: [2510.10208](https://arxiv.org/abs/2510.10208)  
**Code**: None (No public link provided in the paper)  
**Area**: Noisy Label Learning/Parameter-Efficient Fine-Tuning  
**Keywords**: noisy label learning, LoRA, sample selection, parameter-efficient fine-tuning, LLM fine-tuning

## TL;DR
This paper proposes the Delora framework, which constructs a noisy label detector by introducing clean LoRA and noisy LoRA modules. By decoupling sample selection from model training, Delora breaks the vicious catch-22 cycle of mutual influence between sample selection and training in traditional "small-loss" approaches.

## Background & Motivation
**Background**: PEFT (especially LoRA) has become the dominant method for downstream task adaptation in LLMs, but training data in real-world scenarios often contains noisy labels, which has received insufficient attention in existing work.

**Limitations of Prior Work**: Mainstream LNL (learning with noisy labels) methods rely on a "small-loss" strategy to select clean samples based on training loss. However, the selected samples in turn affect the loss computation of subsequent training cycles, creating a catch-22 loop: accurate sample selection requires strong generalization, whereas strong generalization depends on accurate sample selection.

**Key Challenge**: Sample selection relies on the in-training model, leading to selection bias that accumulates and amplifies during training, resulting in severe performance degradation especially under high noise ratios.

**Goal**: To completely decouple noisy label detection (sample selection) from classification model training to avoid the catch-22 loop.

**Key Insight**: Utilize the limited capacity property of PEFT modules to design a dual-LoRA framework that separately memorizes clean and noisy samples.

**Core Idea**: Use two distinct LoRA modules to absorb the memory of clean data and noisy data respectively, thereby constructing a learnable threshold for noisy label detection.

## Method

### Overall Architecture
A two-stage framework:
- **Stage 1 (Noisy Label Detection)**: Construct a noisy label detector using a dual LoRA setup (clean LoRA + noisy LoRA) to perform sample selection.
- **Stage 2 (Classification Model Training)**: Train the classifier using the selected clean samples and re-annotated noisy samples.

### Key Designs
1. **Dual LoRA Detector**: Introduces a clean LoRA ($\Delta w_c$) to memorize clean samples and a noisy LoRA ($\Delta w_n$) to memorize noisy samples. For sample $x_i$, if its cross-entropy loss on clean LoRA is smaller than that on noisy LoRA (using a learnable threshold $\phi_i$), it is identified as a clean sample.
2. **Dynamic Regularization Constraints**: Based on the memorization effect of deep networks (memorizing clean samples first and noisy samples later), time-dependent regularization is designed: $\tau_1(t) = t^{h_1}$ (increasing, gradually constraining clean LoRA updates to prevent memorization of noise) and $\tau_2(t) = t^{-h_2}$ (decreasing, gradually relaxing constraints on noisy LoRA to allow absorption of noise).
3. **Detector Optimization**: Formulates the threshold comparison as a paper-level binary classification problem—the clean probability $p_i^c$ is obtained by normalizing the CE values of the two LoRAs via softmax. Positive samples are constructed by using GPT-4o to generate pseudo-labels that match the original labels, while negative samples are constructed by randomly flipping the labels.
4. **Recycling Noisy Samples**: Instead of discarding noisy samples in Stage 2, clean samples are used as demonstrations to prompt GPT-4o to re-annotate the noisy samples, which are then trained with a robust loss.

### Loss & Training
- Warm-up phase: $L_{\text{warm}} = L_{\text{ce}} + L_{\text{LoRA}}$
- Standard training: $L = L_{\text{ce}} + L_{\text{LoRA}} + L_{\text{Detector}}$
- $L_{\text{ce}}$: Standard cross-entropy (jointly optimizing both LoRAs)
- $L_{\text{LoRA}}$: Dynamic regularization to constrain the update magnitude of both LoRA parameters
- $L_{\text{Detector}}$: Negative log-likelihood optimization of the detector based on positive and negative samples

## Key Experimental Results

### Noisy Label Detection Performance (Precision/Recall %)

| Dataset | Method | 20%S (P/R) | 40%S (P/R) | 20%A (P/R) | 40%A (P/R) |
|--------|------|------------|------------|------------|------------|
| Trec | LLMs-detection | 70.4/70.3 | 70.2/70.2 | 70.0/70.0 | 69.0/68.8 |
| Trec | Small-loss | 81.2/88.6 | 60.2/87.8 | 81.5/74.0 | 59.4/96.2 |
| Trec | **Delora** | **99.5/95.3** | **99.3/96.4** | **99.2/98.1** | **99.1/97.3** |
| SST-2 | **Delora** | **100.0/89.1** | **100.0/86.3** | **100.0/88.6** | **99.8/86.8** |

### Text Classification Accuracy (%) — Synthetic Noise

| Model | Trec 20%S | Trec 40%S | SST-5 20%S | SST-5 40%A |
|------|-----------|-----------|------------|------------|
| Base (Clean) | 98.60 | - | 58.05 | - |
| Base (Noisy) | 95.20 | 90.20 | 54.08 | 47.70 |
| NoiseAL | 97.30 | 96.54 | 55.00 | 48.12 |
| **Delora** | **98.46** | **97.60** | **57.39** | **55.39** |

### Key Findings
- Delora significantly outperforms existing methods in terms of detection accuracy under all noise settings: precision on Trec is generally >99%, far exceeding the 60-81% of the small-loss method.
- The advantage is even more pronounced under a high noise rate (40%), where the small-loss method's precision drops to ~60% while Delora maintains >99%.
- Classification accuracy closely approaches the upper bound of training with clean data (Trec: 98.46% vs 98.60%).
- Delora also clearly outperforms all baselines on real-world noisy datasets (Hausa with 50.37% noise, Yorùbá with 33.28% noise).

## Highlights & Insights
- **Elegant Decoupling Philosophy**: Separating sample selection from model training is a fundamental way to break the catch-22 cycle, offering a very clear methodology.
- **Learnable Threshold**: Using the prediction of the noisy LoRA as a sample-level dynamic threshold is more practical and flexible than manually setting a global threshold.
- **Clever Exploitation of the Memorization Effect**: By controlling the learning pace of both LoRA modules via dynamic regularization, the framework naturally achieves split memorization of clean and noisy samples.
- **Recycling Noisy Samples**: Instead of throwing away noisy samples, they are re-annotated and reused, maximizing data utility.

## Limitations & Future Work
- Stage 1 requires calling GPT-4o to generate pseudo-labels for constructing the positive sample set, introducing dependencies on external APIs and extra computational overhead.
- The hyperparameters $h_1$ and $h_2$ in dynamic regularization still require manual tuning.
- The method has only been validated on NLP text classification tasks, without expansion to other modalities or task types.
- The paper does not explore the robustness boundary of the method under extremely high noise rates (>60%).

## Related Work & Insights
- Comparison with CleaR by Kim et al. (2024): CleaR also leverages LoRA to handle noise but still relies on the in-training model's loss, whereas Delora completely resolves this issue through decoupling.
- Inspiration from Liu et al. (2024), which uses two LoRAs to separate style and subject in images; this concept is elegantly migrated to the noise separation scenario in NLP.
- Insight: The limited capacity of PEFT modules is both a constraint and an advantage—it can be exploited to control and separate the memorization of different types of data.

## Rating
⭐⭐⭐⭐ (4/5)
- **Novelty**: ⭐⭐⭐⭐ — The decoupling idea and dual-LoRA detector design are novel and practical.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Covering synthetic + real-world noise, multiple noise types and ratios, and evaluating across both detection and classification dimensions.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation and well-described methodology.
- **Value**: ⭐⭐⭐⭐ — Establishes a new paradigm for learning with noisy labels, with practical significance for addressing data quality issues in real-world LLM deployments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EW-DETR: Evolving World Object Detection via Incremental Low-Rank DEtection TRansformer](../../CVPR2026/object_detection/ew-detr_evolving_world_object_detection_via_incremental_low-rank_detection_trans.md)
- [\[CVPR 2025\] Generalized Diffusion Detector: Mining Robust Features from Diffusion Models for Domain-Generalized Detection](../../CVPR2025/object_detection/generalized_diffusion_detector_mining_robust_features_from_diffusion_models_for_.md)
- [\[NeurIPS 2025\] Semi-supervised Graph Anomaly Detection via Robust Homophily Learning](../../NeurIPS2025/object_detection/semi-supervised_graph_anomaly_detection_via_robust_homophily_learning.md)
- [\[CVPR 2026\] FSLoRA: Harmonizing Detection and Re-Identification via Freq-Spatial Low-Rank Adapter for One-Stage Person Search](../../CVPR2026/object_detection/fslora_harmonizing_detection_and_re-identification_via_freq-spatial_low-rank_ada.md)
- [\[CVPR 2026\] CD-Buffer: Complementary Dual-Buffer Framework for Test-Time Adaptation in Adverse Weather Object Detection](../../CVPR2026/object_detection/cd-buffer_complementary_dual-buffer_framework_for_test-time_adaptation_in_advers.md)

</div>

<!-- RELATED:END -->
