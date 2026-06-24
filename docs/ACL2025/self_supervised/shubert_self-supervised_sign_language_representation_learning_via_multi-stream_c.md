---
title: >-
  [Paper Note] SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction
description: >-
  [ACL 2025][Self-Supervised Learning][Sign Language Representation Learning] Proposes SHuBERT (Sign Hidden-Unit BERT), migrating the masked cluster prediction paradigm of the speech self-supervised learning model HuBERT to sign language video. By clustering hand, face, and body pose streams separately and simultaneously predicting the cluster labels of masked frames, the model is pre-trained on approximately 984 hours of ASL video, achieving state-of-the-art (SOTA) on public b…
tags:
  - "ACL 2025"
  - "Self-Supervised Learning"
  - "Sign Language Representation Learning"
  - "Self-Supervised Pre-training"
  - "Multi-Stream Cluster Prediction"
  - "Masked Prediction"
  - "HuBERT"
date: 2026-05-08
content_hash: fee3b907849d8edf
---

# SHuBERT: Self-Supervised Sign Language Representation Learning via Multi-Stream Cluster Prediction

**Conference**: ACL 2025  
**arXiv**: [2411.16765](https://arxiv.org/abs/2411.16765)  
**Code**: [http://shubert.pals.ttic.edu](http://shubert.pals.ttic.edu)  
**Area**: Self-Supervised Learning / Sign Language Processing  
**Keywords**: Sign Language Representation Learning, Self-Supervised Pre-training, Multi-Stream Cluster Prediction, Masked Prediction, HuBERT

## TL;DR
Proposes SHuBERT (Sign Hidden-Unit BERT), migrating the masked cluster prediction paradigm of the speech self-supervised learning model HuBERT to sign language video. By clustering hand, face, and body pose streams separately and simultaneously predicting the cluster labels of masked frames, the model is pre-trained on approximately 984 hours of ASL video, achieving state-of-the-art (SOTA) on public benchmarks across translation, isolated recognition, and fingerspelling detection tasks.

## Background & Motivation

**Background**: Sign language processing (translation/recognition) traditionally relies on task-specific models. Existing pre-training methods can be categorized into supervised pre-training (requiring large amounts of annotated data, e.g., 6,600 hours) and self-supervised pre-training (e.g., MAE). However, current self-supervised methods either learn context-independent frame/segment representations or model only a subset of modalities (e.g., hand only).

**Limitations of Prior Work**: (1) Sign language data is scarce, and annotation costs are extremely high; (2) Sign language is multi-channel, where hands, facial expressions, and body poses simultaneously convey semantic information, causing single-channel models to lose key information; (3) Existing self-supervised methods (like MAE in SSVP-SLT) require massive computational resources (64×A100 for 14 days of training) and only process 128-frame/8-second clips, failing to model long-range dependencies.

**Key Challenge**: A self-supervised representation learning method is needed that can simultaneously handle the multi-channel nature of sign language, model long-range context, and remain computationally efficient.

**Goal**: To learn unified, contextual, and multi-channel self-supervised representations for sign language videos.

**Key Insight**: HuBERT in the speech domain has successfully learned contextual speech representations via masked cluster prediction. Sign language and speech share similar challenges (no predefined tokens, variable-length units, no explicit boundaries), suggesting that the HuBERT paradigm can be adapted to multi-stream sign language inputs.

**Core Idea**: Four-stream input (left hand/right hand/face/body pose) $\rightarrow$ independent k-means clustering per stream $\rightarrow$ masked prediction to simultaneously predict cluster labels of the four streams $\rightarrow$ a single pre-trained model applicable to multiple downstream tasks.

## Method

### Overall Architecture
Video $\rightarrow$ MediaPipe keypoint extraction $\rightarrow$ cropping hands/face/body pose $\rightarrow$ DINOv2 feature extraction $\rightarrow$ four-stream k-means clustering to generate pseudo-labels $\rightarrow$ Transformer encoder for masked cluster prediction $\rightarrow$ fine-tuning on downstream tasks.

### Key Designs

1. **Four-Stream Feature Preprocessing**:

    - **Left/Right Hand**: MediaPipe detects hand keypoints $\rightarrow$ crop and resize to 224×224 $\rightarrow$ DINOv2 (fine-tuned on hand data) extracts 384-dimensional features.
    - **Face**: MediaPipe detects face $\rightarrow$ retain mouth and eye regions, grayscale the rest $\rightarrow$ Gaussian blur (for privacy protection) $\rightarrow$ DINOv2 (fine-tuned on facial data) extracts 384-dimensional features.
    - **Body Pose**: 7 upper-body keypoints (nose, shoulders, elbows, wrists) normalized to a 14-dimensional vector.
    - **Design Motivation**: Hand keypoint estimation is inaccurate in capturing hand shapes, making DINOv2 features superior; facial processing balances linguistic information retention and privacy protection.

2. **Self-Supervised Pre-Training (Masked Cluster Prediction)**:

    - **Function**: Predict the four-stream cluster labels for masked frames.
    - **Mechanism**: The four-stream features are linearly projected to 256 dimensions each $\rightarrow$ concatenated to 1024 dimensions per frame $\rightarrow$ random span masking (span = 3 frames $\approx$ 200 ms, roughly the duration of a fingerspelled letter) $\rightarrow$ 12-layer Transformer encoder $\rightarrow$ four linear classification heads separately predict the k-means cluster labels ($k=256$) of the masked positions.
    - **Design Motivation**: Clustering each stream independently but predicting them jointly allows the model to learn cross-stream dependencies. Random masking is more effective than channel-wise or purely temporal masking.

3. **Multi-Task Fine-Tuning**:

    - **Translation (SLT)**: SHuBERT + ByT5 decoder, trained in two stages (YouTube-ASL pre-training $\rightarrow$ target dataset fine-tuning).
    - **Isolated Sign Language Recognition (ISLR)**: SHuBERT + linear classification head.
    - **Fingerspelling Detection**: SHuBERT + binary classification head (detecting active fingerspelling).
    - All Transformer layer outputs are combined using learned layer weights.

### Loss & Training
- Pre-training: 984 hours of ASL video, 8×A6000 GPUs, approx. 7 days, 400K steps.
- Adam optimizer, peak lr = 5e-4, cosine schedule + linear warmup.
- 86M parameters (12-layer Transformer, $d=768$, $h=12$).

## Key Experimental Results

### Main Results (Sign Language Translation on Public Data)

| Method | Self-Supervised | Pre-Training Duration | How2Sign BLEU↑ | OpenASL BLEU↑ | FLEURS-ASL BLEU↑ |
|------|--------|-----------|----------------|--------------|-------------------|
| Uthus 2023 | × | 984h | 12.4 | - | - |
| SSVP (Rust 2024) | ✓ | 1054h | 15.5 | - | - |
| Tanzer 2024 | × | 3207h | 15.4 | - | 4.4 |
| Uni-Sign | × | 984h | 14.9 | 23.1* | - |
| **SHuBERT** | **✓** | **984h** | **16.2** | **23.2** | **4.7** |

*Uni-Sign pre-training contains >72% of the OpenASL test set, making it not fully comparable.

### Ablation Study

| Configuration | How2Sign BLEURT |
|------|----------------|
| Random masking (Default) | **49.9** |
| Channel masking | 48.7 |
| Time masking | 49.1 |
| Frozen SHuBERT | 49.1 |
| Fine-tuned SHuBERT | 49.9 |

### Key Findings
- **Single Pre-trained Model Achieves Multi-Task SOTA**: The exact same SHuBERT model achieves SOTA on all public benchmarks for translation, ISLR, and fingerspelling detection, demonstrating the generality of the representations.
- **Computational Efficiency Advantage**: 8×A6000 training for 7 days vs SSVP's 64×A100 for 14 days (approx. $50 \times$ difference in computational cost), benefiting from multi-stream features and compact representations.
- **Strong Frozen SHuBERT**: Under the frozen setting, translation quality drops only slightly (BLEURT 49.1 vs 49.9), indicating the extremely high quality of pre-trained representations.
- **Necessity of Multi-Stream Joint Modeling**: Joint prediction across all four streams outperforms separate modeling, successfully capturing cross-stream dependencies (e.g., the combination of hand shape and facial expression to express negation).
- **Natural Sign Language > Translated Sign Language**: The largest gain is observed on OpenASL, which contains natural ASL (+10 BLEU vs baseline), indicating that pre-training is more effective on domain-similar data.

## Highlights & Insights
- **Modality Migration from HuBERT to SHuBERT**: Adapting the self-supervised speech paradigm to visual sign language is a natural yet non-trivial transfer. The key innovation is using multi-stream clustering and joint prediction to replace the single-stream setup in speech HuBERT.
- **Privacy-Friendly Facial Representation**: Grayscaling and blurring the face while retaining the mouth and eye regions balances privacy concerns with the retention of linguistic information.
- **DINOv2 as a Sign Language Feature Extractor**: Task-specific continued pre-training on DINOv2 (using 5M hand/face crops) produces features that are more accurate than keypoints.

## Limitations & Future Work
- Validated only on ASL; generalization across diverse sign languages (e.g., DGS/BSL/CSL) remains to be explored.
- The model has 86M parameters (base model size); scaling up might yield further improvements.
- Relies on MediaPipe hand detection (approx. 95% accuracy); interpolation is required when detection fails.
- Combined auxiliary losses (e.g., contrastive learning, multi-task joint training) have not been explored.
- Facial blurring might lose subtle non-manual markers (e.g., eyebrow movements).

## Related Work & Insights
- **vs SSVP-SLT (MAE)**: SSVP reconstructs image pixels via MAE, which is computationally expensive (64×A100 × 14 days) and only handles 128 frames. SHuBERT replaces pixel reconstruction with cluster prediction, which is far more efficient.
- **vs SignBERT+**: SignBERT+ only models hand pose, lacking facial and body information, and its pre-training data overlaps with downstream test data. SHuBERT joint-models four streams, with complete separation of pre-training and test data.
- **vs HuBERT (Speech)**: SHuBERT is a natural extension of HuBERT to sign language, with the core difference being multi-stream clustering and random span masking.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The transfer design from HuBERT to sign language is elegant, and the multi-stream cluster prediction is a successful adaptation tailored to sign language characteristics.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across six benchmarks spanning three major tasks, featuring detailed ablation studies and comparisons against approaches utilizing private datasets.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, well-defined motivation, and standard high-quality figures and tables.
- Value: ⭐⭐⭐⭐⭐ A breakthrough foundation model for sign language processing, highly computational-efficient, and open-source/reproducible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Scaling Language-Free Visual Representation Learning](../../ICCV2025/self_supervised/scaling_languagefree_visual_representation_learning.md)
- [\[NeurIPS 2025\] M-GRPO: Stabilizing Self-Supervised Reinforcement Learning for Large Language Models with Momentum-Anchored Policy Optimization](../../NeurIPS2025/self_supervised/m-grpo_stabilizing_self-supervised_reinforcement_learning_for_multimodal_underst.md)
- [\[NeurIPS 2025\] Adv-SSL: Adversarial Self-Supervised Representation Learning with Theoretical Guarantees](../../NeurIPS2025/self_supervised/adv-ssl_adversarial_self-supervised_representation_learning_with_theoretical_gua.md)
- [\[ACL 2025\] Improving Low-Resource Morphological Inflection via Self-Supervised Objectives](improving_low-resource_morphological_inflection_via_self-supervised_objectives.md)
- [\[ACL 2025\] WhiSPA: Semantically and Psychologically Aligned Whisper with Self-Supervised Contrastive and Student-Teacher Learning](whispa_semantically_and_psychologically_aligned_whisper_with_self-supervised_con.md)

</div>

<!-- RELATED:END -->
