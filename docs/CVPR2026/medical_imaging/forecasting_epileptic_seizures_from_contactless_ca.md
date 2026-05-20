---
title: >-
  [Paper Note] Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning
description: >-
  [CVPR 2026][Medical Imaging][Epileptic seizure forecasting] This paper is the first to systematically define the task of video-based epileptic seizure forecasting (predicting whether a seizure will occur within the next…
tags:
  - "CVPR 2026"
  - "Medical Imaging"
  - "Epileptic seizure forecasting"
  - "video analysis"
  - "cross-species transfer learning"
  - "contactless monitoring"
  - "VideoMAE"
date: 2026-05-08
content_hash: dde9ad6caa9b4927
---

# Forecasting Epileptic Seizures from Contactless Camera via Cross-Species Transfer Learning

**Conference**: CVPR 2026
**arXiv**: [2603.12887](https://arxiv.org/abs/2603.12887)  
**Code**: To be confirmed  
**Area**: Medical Imaging
**Keywords**: Epileptic seizure forecasting, video analysis, cross-species transfer learning, contactless monitoring, VideoMAE

## TL;DR

This paper is the first to systematically define the task of video-based epileptic seizure forecasting (predicting whether a seizure will occur within the next 5 seconds using 3–10-second pre-ictal clips), and proposes a two-stage cross-species transfer learning framework — self-supervised pre-training of VideoMAE on a mixed dataset of rodent and human videos, followed by few-shot fine-tuning on a very limited set of human epilepsy videos. Under 2/3/4-shot settings, the framework achieves an average balanced accuracy (bacc) of 72.30% and ROC-AUC of 75.58%, outperforming all video understanding baselines.

## Background & Motivation

**Background**: Epileptic seizure forecasting is a clinically valuable problem. Existing methods rely primarily on neural signals such as EEG, which require specialized equipment and complex wearable setups, severely limiting long-term deployment in everyday settings. Video data offers advantages of non-invasiveness, accessibility, and support for continuous recording; however, existing video-based studies largely focus on post-onset detection, and pre-onset seizure forecasting (i.e., issuing warnings before a seizure occurs) remains almost entirely unexplored.

**Limitations of Prior Work**: (1) Video-based seizure forecasting has never been defined as a task — detection exists but forecasting does not; (2) large-scale annotated human epilepsy video datasets are extremely scarce due to privacy concerns and collection difficulties (this paper collected only 40 short videos); (3) general-purpose video pre-trained models such as those trained on Kinetics-400 lack knowledge of epilepsy-related behavioral dynamics and perform poorly when directly fine-tuned.

**Key Challenge**: Seizure forecasting requires models to understand subtle pre-ictal behavioral dynamics, yet annotated human data is extremely scarce, and general-purpose video models possess no such domain knowledge.

**Goal**: (1) Define the task of video-based seizure forecasting; (2) train effective forecasting models under conditions of extremely limited human data.

**Key Insight**: Rodent epilepsy models provide abundant data, and seizure characteristics exhibit cross-species consistency between rodents and humans. Pre-training on cross-species video data enables the model to learn epilepsy-related spatiotemporal behavioral patterns, compensating for the scarcity of human data.

**Core Idea**: Use rodent epilepsy videos combined with human normal videos for self-supervised pre-training to establish priors on epileptic behavioral dynamics, then perform few-shot fine-tuning on a small number of human epilepsy videos to enable seizure forecasting.

## Method

### Overall Architecture

The framework consists of two stages. **Stage 1** — Domain-Specific Continual Pre-training: VideoMAE-Base (pre-trained on Kinetics-400) is used as initialization, and tube-masked self-supervised reconstruction is performed on a cross-species mixed dataset. **Stage 2** — Few-shot Fine-tuning: the decoder is discarded, the encoder is retained, and a lightweight classification head is added for binary classification.

### Key Designs

1. **Cross-Species Pre-training Data Construction**

    - **Function**: Construct a pre-training dataset by mixing rodent epilepsy videos with human normal videos.
    - **Mechanism**: The RodEpil dataset (13,000+ 10-second rodent clips) is used, with 2,952 seizure samples and 3,000 normal samples selected via balanced sampling. Additionally, 1,870 5-second human non-ictal videos (from 6 patients) are included. Both sources are simply concatenated to form $D_{pt}$.
    - **Design Motivation**: Rodent data provides knowledge of epileptic motor dynamics (subtle pre-ictal behavioral patterns), while human data preserves the model's ability to represent human body poses. The two sources are complementary. Ablation experiments confirm that the mixed data configuration (+R(Y/N)+H) achieves the best performance across all shot settings.

2. **VideoMAE Self-Supervised Pre-training**

    - **Function**: Perform masked video autoencoder pre-training on the mixed dataset to learn epilepsy-related spatiotemporal representations.
    - **Mechanism**: Tube masking is applied to 3D video patches, which are then reconstructed. The MSE reconstruction loss is: $\mathcal{L}_{MSE} = \frac{1}{\Omega}\sum_{i\in\Omega}(I_i - \hat{I}_i)^2$. **Key finding**: The optimal masking ratio is 0.3, rather than the 0.75–0.9 commonly used for general videos, as pre-ictal movements are subtle and require more spatiotemporal context to be preserved for meaningful representation learning.
    - **Design Motivation**: Self-supervised pre-training requires no labels, making full use of large-scale unlabeled data. The low masking ratio is a domain-specific finding — the high information redundancy of general videos (where 90% masking still allows reconstruction in Kinetics) does not apply to medical behavioral videos.

3. **Few-Shot Classification Fine-tuning**

    - **Function**: Perform binary classification using the encoder CLS token and a linear classification head under $N \in \{2, 3, 4\}$-shot settings.
    - **Mechanism**: The decoder is discarded and the pre-trained encoder weights are retained. The CLS token is passed through a linear layer followed by a sigmoid to output seizure probability: $\hat{y} = \sigma(\mathbf{W} \cdot \mathbf{z}_{cls} + b)$. The model is trained with cross-entropy loss for 20 epochs. Gradient checkpointing and 16-bit mixed precision are used to reduce memory overhead.
    - **Design Motivation**: Under extreme data scarcity (40 videos), complex classification heads are prone to overfitting; linear probing is the most reliable choice.

### Loss & Training

- Stage 1: MSE reconstruction loss; Adam optimizer with lr=1e-4; 16-frame sampling at stride 2; resolution 224×224; 8× NVIDIA L40 GPUs with DDP.
- Stage 2: BCE classification loss; 20 epochs of fine-tuning; gradient checkpointing + FP16.

## Key Experimental Results

### Main Results — Performance of Different Methods under 2/3/4-Shot Settings

| Method | avg bacc | avg roc_auc | avg pr_auc |
|--------|----------|-------------|------------|
| CSN | 0.5278 | 0.5722 | 0.5837 |
| X3D | 0.5540 | 0.7045 | 0.7105 |
| SlowFast | 0.6620 | 0.7065 | 0.6812 |
| Linear Probing | 0.4742 | 0.4994 | 0.5274 |
| Human-only | 0.7149 | 0.7491 | 0.6943 |
| Pretrained zeroshot | 0.5250 | 0.5500 | 0.4944 |
| **Ours** | **0.7230** | **0.7558** | **0.7091** |

### Ablation Study — Pre-training Data Combinations

| Configuration | avg bacc | avg roc_auc | avg pr_auc |
|---------------|----------|-------------|------------|
| Base (direct fine-tuning from Kinetics-400) | 0.7149 | 0.7491 | 0.6943 |
| +H (human normal only) | 0.7163 | 0.7419 | 0.6995 |
| +R(Y) (rodent seizure only) | 0.6961 | 0.7327 | 0.6765 |
| +R(N) (rodent normal only) | 0.7097 | 0.7422 | 0.6594 |
| +R(Y/N) (rodent seizure + normal) | 0.6965 | 0.7500 | 0.7078 |
| **+R(Y/N)+H (full framework)** | **0.7230** | **0.7558** | **0.7091** |

### Key Findings

- **Cross-species pre-training is effective**: The full framework (+R(Y/N)+H) achieves the best performance on all three averaged metrics, improving over Base by +0.81% bacc, +0.67% roc_auc, and +1.48% pr_auc.
- **Low masking ratio is critical**: The optimal mask ratio is 0.3, rather than the 0.75–0.9 commonly used in VideoMAE. This is because pre-ictal movements are subtle, and high masking discards key behavioral cues.
- **Conventional video models perform poorly in few-shot settings**: CSN achieves only 0.34 bacc under 2-shot and X3D only 0.53, demonstrating that general-purpose video models are ill-suited for extreme few-shot medical scenarios.
- **2-shot is the most challenging setting**: The model achieves the highest roc_auc (0.7682) and pr_auc (0.7269) under 2-shot — even surpassing Human-only — highlighting that cross-species pre-training is most valuable when data is most scarce.
- Using rodent seizure data alone (+R(Y)) performs worse than Base, indicating that mixing normal behavioral data is important for regularization.

## Highlights & Insights

1. **Pioneering task definition**: This paper is the first to advance video-based epilepsy analysis from *detection* (post-hoc determination of whether a seizure occurred) to *forecasting* (proactive warning of whether a seizure will occur), representing a qualitative shift in clinical value — a 5-second warning window enables timely intervention.
2. **Validation of cross-species transfer**: The paper demonstrates that rodent epileptic behavioral dynamics can indeed transfer to human seizure prediction tasks, opening a new avenue of leveraging animal data for small-data medical computer vision problems.
3. **Domain-specific finding on masking ratio**: While high masking ratios (0.75–0.9) work well for general videos in VideoMAE, low masking (0.3) is optimal for medical behavioral videos — a finding with broader implications for self-supervised learning in medical video analysis.

## Limitations & Future Work

1. **Extremely small dataset**: Only 40 evaluation videos (20 pre-ictal + 20 normal) are used, limiting statistical reliability; generalization of the conclusions requires validation at a larger scale.
2. **Fixed 5-second prediction horizon**: Different prediction time horizons (Seizure Prediction Horizon) are not explored, whereas clinicians need to know the maximum effective prediction lead time.
3. **Pure video setting**: In practical deployment, multimodal signals such as audio and heart rate variability (HRV) could be incorporated to reduce false alarm rates.
4. **Cross-species consistency is not quantified**: The paper relies on literature citations to justify rodent–human epileptic behavioral consistency, without providing quantitative analysis.
5. **Minimal model design**: Only the CLS token with a linear head is used for classification; more sophisticated temporal modeling approaches (e.g., temporal attention) are not explored.

## Related Work & Insights

- **vs. EEG-based methods**: EEG requires specialized equipment and contact-based sensors, whereas the video-based approach is entirely contactless — a qualitative improvement in deployability (e.g., home monitoring scenarios).
- **vs. CNN-LSTM detection methods (Pérez-García et al.)**: These methods perform post-onset detection, whereas this paper addresses pre-onset forecasting — a fundamentally different task definition.
- **vs. SlowFast/X3D**: General action recognition models perform far worse than the proposed cross-species pre-training approach under extreme few-shot medical settings (2–4 shots).
- **Generalizability of the cross-species paradigm**: The framework is not limited to epilepsy — any medical video task where human behavioral data is scarce but animal models are abundant (e.g., Parkinsonian gait analysis) can benefit from this approach.

## Rating

⭐⭐⭐⭐

- **Novelty** ⭐⭐⭐⭐⭐: Pioneering task definition; unique angle via cross-species transfer learning.
- **Experimental Thoroughness** ⭐⭐⭐: Multiple baselines, data ablations, and masking ratio ablations are provided, but the dataset is extremely small (40 videos), raising concerns about statistical reliability.
- **Writing Quality** ⭐⭐⭐⭐: Problem motivation is clearly articulated; the two-stage framework is concisely presented.
- **Value** ⭐⭐⭐⭐⭐: Opens a new direction for contactless seizure forecasting; the cross-species learning paradigm serves as a model for small-data medical computer vision tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Cross-Slice Knowledge Transfer via Masked Multi-Modal Heterogeneous Graph Contrastive Learning for Spatial Gene Expression Inference](cross-slice_knowledge_transfer_via_masked_multi-modal_heterogeneous_graph_contra.md)
- [\[CVPR 2026\] Unlocking Positive Transfer in Incrementally Learning Surgical Instruments: A Self-reflection Hierarchical Prompt Framework](unlocking_positive_transfer_in_incrementally_learning_surgical_instruments_a_sel.md)
- [\[ICCV 2025\] G2PDiffusion: Cross-Species Genotype-to-Phenotype Prediction via Evolutionary Diffusion](../../ICCV2025/medical_imaging/g2pdiffusion_cross-species_genotype-to-phenotype_prediction_via_evolutionary_dif.md)
- [\[CVPR 2026\] Interpretable Cross-Domain Few-Shot Learning with Rectified Target-Domain Local Alignment](interpretable_cross-domain_few-shot_learning_with_rectified_target-domain_local_.md)
- [\[CVPR 2026\] STEPH: Sparse Task Vector Mixup with Hypernetworks for Efficient Knowledge Transfer in WSI Prognosis](sparse_task_vector_mixup_wsi_prognosis.md)

</div>

<!-- RELATED:END -->
