---
title: >-
  [Paper Note] BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy
description: >-
  [CVPR 2026][Human Understanding][Ambivalence & Hesitancy Recognition] This paper proposes a heavily regularized multimodal fusion pipeline that achieves robust video-level recognition of Ambivalence/Hesitancy (A/H) behaviors in naturalistic settings. The framework employs a heterogeneous classifier committee across four modalities — visual (SigLip2), audio (HuBERT), text (F2LLM), and statistical features — combined with PSO-based hard-voting ensemble regularized by a train-validation gap penalty, achieving Macro F1 = 0.7465 on the ABAW10 test set.
tags:
  - CVPR 2026
  - Human Understanding
  - "Ambivalence & Hesitancy Recognition"
  - Multimodal Fusion
  - Ensemble Learning
  - Particle Swarm Optimization
  - Behavioral Analysis
date: 2026-05-08
content_hash: 00e1dfebfffb70bd
---

# BROTHER: Behavioral Recognition Optimized Through Heterogeneous Ensemble Regularization for Ambivalence and Hesitancy

**Conference**: CVPR 2026
**arXiv**: [2603.14361](https://arxiv.org/abs/2603.14361)
**Code**: Not released
**Area**: Human Understanding
**Keywords**: Ambivalence & Hesitancy Recognition, Multimodal Fusion, Ensemble Learning, Particle Swarm Optimization, Behavioral Analysis

## TL;DR

This paper proposes a heavily regularized multimodal fusion pipeline that achieves robust video-level recognition of Ambivalence/Hesitancy (A/H) behaviors in naturalistic settings. The framework employs a heterogeneous classifier committee across four modalities — visual (SigLip2), audio (HuBERT), text (F2LLM), and statistical features — combined with PSO-based hard-voting ensemble regularized by a train-validation gap penalty, achieving Macro F1 = 0.7465 on the ABAW10 test set.

## Background & Motivation

**Importance of A/H Recognition**: Ambivalence and Hesitancy are primary psychological barriers to health behavior change, and their automatic detection carries significant implications for digital behavioral intervention.

**Fundamental Distinction from Basic Emotions**: Unlike discrete emotions such as "happiness" or "anger," A/H represents a subtle inner conflict state occupying a gray zone between positive and negative attitudes — a nuance that traditional fixed-category emotion recognition systems struggle to capture.

**Necessity of Multimodal Cues**: A/H manifests across facial expressions, prosody/speech rate, and word choice simultaneously; single-modality approaches cannot comprehensively model such cross-channel behavioral conflict signals.

**Challenges in Naturalistic Settings**: The BAH dataset, sourced from the ABAW10 competition, was recorded by participants in uncontrolled everyday environments, where lighting variation and background noise further complicate detection.

**Limitations of Pretrained Emotion Classifiers**: Constraining model outputs to fixed emotion categories impedes the organic emergence of more complex, fine-grained multimodal feature relationships.

**Overfitting Risk**: The limited training set size, combined with the high-dimensional feature space of multiple modalities, makes models prone to memorizing the training distribution rather than learning generalizable patterns, necessitating strong regularization strategies.

## Method

### Overall Architecture

The pipeline consists of four stages: (1) four-modality feature extraction → (2) training 3 classifier types across 15 modality combinations → (3) optimal model selection based on validation BCE loss → (4) PSO-based hard-voting ensemble. The core idea treats A/H as multimodal temporal conflict rather than static emotion categories, leverages a heterogeneous classifier committee to preserve the unique advantages of each modality subset, and employs PSO-regularized ensemble to prevent overfitting.

### Key Design 1: Four-Modality Feature Extraction

- **Function**: Extract feature vectors independently from visual, audio, text, and statistical modalities.
- **Mechanism**:
    - **Visual**: RetinaFace face detection → SigLip2 embedding extraction → MAD-based noisy frame filtering → concatenation of raw/first-order/second-order derivative means (2304-dim) → PCA reduction to 512-dim.
    - **Audio**: HuBERT extracts audio embeddings at 1-second intervals, capturing non-verbal cues such as prosody and rhythm.
    - **Text**: F2LLM extracts global embeddings from the full transcript, preserving complete linguistic context.
    - **Statistical Modality**: Aggregates min/max/mean/std statistics from temporal/sentence-level sequences of the first three modalities, supplemented by librosa audio features (RMS, spectral centroid/bandwidth, zero-crossing rate, silence ratio, fundamental frequency mean/variance).
- **Design Motivation**: Avoiding fixed-category outputs of pretrained emotion classifiers allows the model to organically discover complex relationships from raw embedding spaces. The statistical modality further supplements temporal and structural patterns that direct sequence processing may discard.

### Key Design 2: Behavioral Linguistic Strategy (Text Statistics)

- **Function**: Engineer text features within the statistical modality that are specifically tailored to Hesitancy and Ambivalence.
- **Mechanism**:
    - **Hesitancy Detection** (local/sentence-level): Computes cosine similarity between each sentence and four lexical categories — filler words, filler sounds, hedges, and self-corrections — to quantify local hesitancy.
    - **Ambivalence Detection** (global/text-level): Constructs prompt embeddings across six dimensions (affect, ability, excuse, success, motivation, opportunity) and computes temperature-scaled softmax similarities over four polarity axes (neutral/negative/positive/both) to capture attitudinal conflict in the transcript.
- **Design Motivation**: Hesitancy is highly localized (pauses and hedges within a single sentence), whereas Ambivalence is global (alternating positive and negative attitudes across the full utterance); the two phenomena require modeling at different granularities.

### Key Design 3: Heterogeneous Classifier Committee

- **Function**: Train MLP, Random Forest, and GBDT classifiers on each of the 15 modality combinations ($2^4 - 1$), retaining the best-performing one per combination.
- **Mechanism**: For each combination, the optimal classifier is selected based on validation set BCE loss (rather than hard classification metrics), prioritizing probability calibration. This yields 15 heterogeneous base models in total.
- **Design Motivation**: MLPs excel on low-dimensional, simple configurations; Random Forests provide stronger regularization on high-dimensional multimodal concatenations. Since different modality subsets favor different architectures, the committee mechanism preserves this diversity.

### Key Design 4: PSO Hard-Voting Ensemble with Generalization Penalty

- **Function**: Apply Particle Swarm Optimization to search for optimal hard-voting weights in a 15-dimensional continuous weight space.
- **Mechanism**: Each base model produces a binary vote based on its individually optimized threshold; the weighted sum exceeding half constitutes a positive prediction. The PSO fitness function is defined as the harmonic mean of training and validation F1 minus a gap penalty:
  $$\text{Fitness} = \frac{2 \cdot F1_{val} \cdot F1_{train}}{F1_{val} + F1_{train}} - (\lambda \cdot |F1_{train} - F1_{val}|)^2$$
- **Design Motivation**: Hard voting avoids information loss from probability distribution averaging. The harmonic mean enforces jointly high performance on both splits. The squared gap penalty actively suppresses overfitting and redundant classifiers. As $\lambda$ increases, PSO drives 9 out of 15 model weights to zero, retaining only the 6 most reliable models.

## Loss & Training

- **Classifier Training**: MLP employs Gaussian noise injection, Batch Normalization, and Dropout regularization; Random Forest uses balanced class weights and a maximum depth of 50; GBDT uses an extremely low learning rate (1e-3).
- **Model Selection**: The optimal classifier per modality combination is selected based on validation BCE loss, prioritizing probability calibration over hard classification boundaries.
- **Ensemble Optimization**: PSO runs with 50 particles × 100 iterations; inertia weight $w=0.9$ (encouraging exploration), cognitive parameter $c_1=1.5$, social parameter $c_2=2.1$; $\lambda$ is searched over $\{0.0, 0.2, 0.4, 0.6, 0.8\}$ with independent runs per setting.

## Key Experimental Results

### Table 1: Optimal Classifier Selection per Modality Combination (Validation BCE Loss & Macro F1)

| Modality Combination | MLP BCE / F1 | RF BCE / F1 | GBDT BCE / F1 | Winner |
|---|---|---|---|---|
| Text | 0.573 / 0.728 | 0.623 / 0.661 | 0.631 / 0.678 | MLP |
| Audio | 0.675 / 0.632 | 0.695 / 0.599 | 0.692 / 0.597 | MLP |
| Video | 0.747 / 0.523 | 0.696 / 0.464 | 0.696 / 0.470 | GBDT |
| Stats | 0.650 / 0.693 | 0.634 / 0.641 | 0.640 / 0.632 | RF |
| Text+Audio | 0.593 / 0.701 | 0.632 / 0.641 | 0.639 / 0.654 | MLP |
| Text+Video | 0.688 / 0.536 | 0.624 / 0.669 | 0.632 / 0.669 | RF |
| All Modalities | 0.696 / 0.595 | 0.627 / 0.660 | 0.636 / 0.654 | RF |

**Key Findings**: Text is the strongest single modality (F1=0.728), while video is the weakest (F1=0.470). MLP wins on simpler configurations, whereas RF is superior for high-dimensional multimodal combinations (winning 7/15 cases).

### Table 2: PSO Ensemble Performance under Different Penalty Coefficients $\lambda$ (Macro F1)

| Penalty $\lambda$ | Train F1 | Val F1 | **Test F1** |
|---|---|---|---|
| 0.0 (no penalty) | 0.974 | 0.736 | 0.740 |
| **0.2 (20%)** | **0.982** | **0.736** | **0.747** |
| 0.4 (40%) | 0.965 | 0.758 | 0.741 |
| 0.6 (60%) | 0.965 | 0.758 | 0.741 |
| 0.8 (80%) | 0.978 | 0.749 | 0.742 |

**Key Findings**: $\lambda=0.2$ achieves the best test Macro F1 (0.7465) and Weighted F1 (0.7559). Moderate regularization outperforms both no penalty and excessive penalization.

## Highlights & Insights

1. **Innovative Design of the Statistical Modality**: The fourth modality goes beyond simple handcrafted features; it employs distinct granularity strategies for Hesitancy (local, sentence-level) and Ambivalence (global, text-level), drawing strong inspiration from psychological theory.
2. **Effectiveness of the PSO Generalization Penalty**: The gap penalty not only improves generalization but also performs implicit model selection — under high $\lambda$, PSO drives 9/15 model weights to zero, consistently assigning the highest weights to the text and Text+Video+Stats combinations.
3. **Committee vs. End-to-End**: Rather than training a complex end-to-end multimodal Transformer, the method achieves competitive performance through a heterogeneous classifier committee and intelligent ensemble, yielding a concise and interpretable pipeline.

## Limitations & Future Work

1. **Absence of End-to-End Temporal Modeling**: All features are compressed to video-level vectors via statistical pooling, discarding fine-grained temporal dynamics and limiting the modeling of hesitancy-to-ambivalence transitions within long videos.
2. **Weak Video Modality Performance**: The visual-only F1 of 0.47 indicates that SigLip2, despite being a strong general-purpose visual model, has limited capacity for capturing subtle facial ambivalence signals; dedicated facial action unit (AU) modeling may be required.
3. **Limited Dataset Scale**: The relatively small BAH dataset makes the 15-dimensional PSO search space prone to instability, potentially resulting in high variance across independent runs.
4. **Underutilization of LLM Reasoning**: The text modality relies solely on embeddings for classification, leaving the semantic reasoning capability of LLMs regarding hesitancy and ambivalence unexploited — in contrast to approaches such as the Video-LLaVA baseline.

## Related Work & Insights

- **González-González et al. (ICLR 2026)**: Proposers of the BAH dataset; achieved F1=0.634 with Video-LLaVA in a zero-shot setting, a result substantially surpassed by the proposed method.
- **HSEmotion (ABAW-8)**: Employs EmotiEffLib with lightweight MLP fusion — a conceptually similar approach, though the proposed method offers more refined feature extraction and ensemble strategies.
- **Insights**: (1) For ambiguous or mixed affective states, circumventing the fixed-category constraints of pretrained classifiers is more important than pursuing stronger backbones. (2) The PSO gap penalty constitutes a general-purpose ensemble regularization strategy transferable to other small-data multimodal tasks.

## Rating

- **Novelty**: ⭐⭐⭐ — Core contributions lie in the engineering design of the statistical modality and PSO-regularized ensemble; practically motivated but with limited theoretical novelty.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Systematic ablation across 15 modality combinations × 3 classifiers, combined with 5 $\lambda$ value comparisons, yields comprehensive analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly articulated, methodology is well-structured, and experimental analysis is thorough.
- **Value**: ⭐⭐⭐ — A competition-oriented contribution with a reproducible methodology and a transferable ensemble strategy, though broader generalizability remains to be validated.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Team LEYA in 10th ABAW Competition: Multimodal Ambivalence/Hesitancy Recognition Approach](team_leya_in_10th_abaw_competition_multimodal_ambivalencehesitancy_recognition_a.md)
- [\[ICLR 2026\] BAH Dataset for Ambivalence/Hesitancy Recognition in Videos for Digital Behaviour Analysis](../../ICLR2026/human_understanding/bah_dataset_for_ambivalencehesitancy_recognition_in_videos_for_digital_behaviour.md)
- [\[CVPR 2026\] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)
- [\[CVPR 2026\] MMGait: Towards Multi-Modal Gait Recognition](mmgait_multi_modal_gait_recognition.md)
- [\[CVPR 2026\] FusionAgent: A Multimodal Agent with Dynamic Model Selection for Human Recognition](fusionagent_a_multimodal_agent_with_dynamic_model_selection_for_human_recognitio.md)

<!-- RELATED:END -->
