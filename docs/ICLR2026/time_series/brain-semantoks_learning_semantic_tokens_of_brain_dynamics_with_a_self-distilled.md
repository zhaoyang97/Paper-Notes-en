---
title: >-
  [Paper Note] Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model
description: >-
  [ICLR2026][Time Series][fMRI foundation model] This paper proposes Brain-Semantoks, an fMRI foundation model based on a semantic tokenizer and a self-distillation objective. It aggregates functional network signals into robust semantic tokens and learns abstract brain dynamic representations through cross-temporal view consistency, achieving state-of-the-art performance under a linear probing setting.
tags:
  - ICLR2026
  - Time Series
  - fMRI foundation model
  - self-distillation
  - semantic tokenizer
  - brain dynamics representation learning
  - linear probing
date: 2026-05-08
content_hash: bb8e1141f239de06
---

# Brain-Semantoks: Learning Semantic Tokens of Brain Dynamics with a Self-Distilled Foundation Model

**Conference**: ICLR2026  
**arXiv**: [2512.11582](https://arxiv.org/abs/2512.11582)  
**Code**: [https://github.com/SamGijsen/Brain-Semantoks](https://github.com/SamGijsen/Brain-Semantoks)  
**Area**: Time Series  
**Keywords**: fMRI foundation model, self-distillation, semantic tokenizer, brain dynamics representation learning, linear probing

## TL;DR
This paper proposes Brain-Semantoks, an fMRI foundation model based on a semantic tokenizer and a self-distillation objective. It aggregates functional network signals into robust semantic tokens and learns abstract brain dynamic representations through cross-temporal view consistency, achieving state-of-the-art performance under a linear probing setting.

## Background & Motivation

**State of the Field**: fMRI foundation models have advanced rapidly in recent years. Pioneering works such as BrainLM, Brain-JEPA, and NeuroSTORM all adopt mask-and-reconstruct objectives. These methods focus on low-level signal prediction — BrainLM directly reconstructs BOLD signals in input space, Brain-JEPA performs prediction in latent space to avoid modeling noise, and NeuroSTORM conducts spatiotemporal reconstruction over 4D voxels.

**Root Cause**: There is a fundamental mismatch between reconstruction objectives and downstream task objectives. Downstream tasks such as disease diagnosis and cognitive assessment require stable, high-level phenotypic signatures, whereas representations learned through reconstruction objectives are sensitive to noise and temporal fluctuations, necessitating extensive fine-tuning for adaptation. This dependence on fine-tuning undermines the practical utility of foundation models, particularly in the fMRI domain where datasets vary considerably in subject populations, hardware, and acquisition protocols.

**Key Assumption**: Effective prediction of stable phenotypes requires a shift from "reconstruction" to "abstraction" — the goal is not to precisely encode BOLD signals but to extract underlying phenotypic features from them.

**Starting Point**: (1) Time series from individual ROIs are noisy and semantically ambiguous, making them unsuitable as input tokens for a Transformer. (2) The functional organization of the brain (e.g., the default mode network) provides strong neuroscientific priors that can be leveraged to construct semantic tokens.

**Core Idea**: A functional-network-level semantic tokenizer aggregates noisy regional signals into robust tokens, after which a self-distillation objective learns temporally stable, abstract representations.

## Method

### Overall Architecture
Brain-Semantoks employs a student–teacher architecture. Given an input fMRI time series $X \in \mathbb{R}^{C \times T}$ (C = 457 brain regions, T = number of time points), two long temporal segments are first cropped as distinct views, encoded into functional-network-level tokens by the semantic tokenizer, and then processed by a Transformer encoder. The teacher network weights are an exponential moving average (EMA) of the student, and the training objective enforces cross-view representation consistency.

### Key Designs

1. **Semantic Tokenizer**:

    - Based on the Yeo 7-network cortical parcellation plus subcortical and cerebellar regions, yielding 9 functional networks in total.
    - Each network has an independent tokenization module $g_n$ that processes the time series of all ROIs within that network.
    - The temporal dimension is divided into $P$ longer temporal patches; each patch is processed by a dual-branch convolution (standard convolution + structured convolution) to extract multi-scale temporal patterns.
    - The output token tensor is $Z \in \mathbb{R}^{N \times P \times D}$ (9 networks × P patches × 768 dimensions), yielding a sequence of length only $N \times P$ — far shorter than the original 457-ROI sequence.
    - **Core Value**: Aggregates noisy ROI-level signals into semantically rich network-level tokens, providing the Transformer with higher-quality inputs.

2. **Slice Masking**:

    - Tokens are arranged as an $N \times P$ 2D matrix.
    - One of two strategies is randomly selected: network slicing (masking entire rows) or temporal slicing (masking contiguous column blocks).
    - A high masking ratio ($\mathcal{U}[0.65, 0.85]$) compels the model to learn complex inter-network and cross-temporal relationships.
    - This design prevents the model from completing predictions through simple interpolation.

3. **Triple Loss Function**:

    - **$\mathcal{L}_{CLS}$ (Global Cross-View Loss)**: Bidirectional distillation between the student and teacher [CLS] tokens across two temporal views, learning stable global representations; coding rate regularization is applied to prevent representation collapse.
    - **$\mathcal{L}_{Tok}$ (Network Token Loss)**: Within each view, the student reconstructs masked network tokens to match teacher outputs, learning temporally sensitive local features.
    - **$\mathcal{L}_{TTR}$ (Teacher-guided Temporal Regularization)**: The $P$ patch embeddings of each network are averaged into a single summary token, which is then distilled across views. This loss is activated during the first 5% of training steps and then decayed to zero via a cosine schedule — guiding the model to first learn temporally averaged network signatures before modeling more complex temporal variations.

4. **Design Motivation for TTR**:

    - Directly applying a self-distillation objective to low-SNR fMRI data is prone to training instability and representation collapse.
    - TTR helps the model find good initial representations by constraining the initial token space (from $N \times P + 1$ to $N + 1$).
    - Activating TTR only in the early training phase avoids over-constraining the final learned solution.

### Training Details
- Temporal crop length $T_{crop} = 100$, patch length 20, 9 functional networks, 5 patches per network.
- Transformer: $D_f = 768$, 8 layers, projection head with 2 hidden layers ($D_h = 1024$), output $D_{proj} = 128$.
- Pre-trained on 39,139 resting-state fMRI scans from UK BioBank; training takes less than 2 hours on a single GPU (<20 GB VRAM).
- Z-score normalization replaces robust scaling, improving cross-dataset transfer.

## Key Experimental Results

### Linear Probing (Frozen Weights + Single Linear Layer)

| Dataset / Task | BrainLM | Brain-JEPA | **Brain-Semantoks** |
|----------------|---------|------------|---------------------|
| ABIDE (ASD) | 53.84 | 52.92 | **65.13** |
| HBN CELF | 42.03 | 41.50 | **42.18** |
| HBN WISC | 38.26 | 38.34 | **40.87** |
| UKB Sex | 86.71 | 83.23 | **87.52** |
| UKB Age | 30.16 | 30.60 | **31.15** |
| SRPBS SZ | 57.61 | 57.63 | **69.26** |
| SRPBS MDD | 55.72 | 52.72 | **62.60** |

- Achieves the highest score on 8 out of 9 tasks, with particularly pronounced advantages on clinical diagnosis tasks (ASD +12, SZ +12, MDD +7).

### Comparison with Supervised Methods
- Linear probing alone surpasses all fully supervised end-to-end trained baselines (FC, BNT, BolT, BrainMass) as well as fine-tuned BrainLM/Brain-JEPA.
- Average of 52.72% across 12 tasks vs. the best supervised baseline at 50.68%, demonstrating that the learned representations generalize broadly without fine-tuning.

### Generalization to Task-Based fMRI (Hariri Emotion Task)
- Brain-Semantoks achieves 93.84–96.50% under linear probing, substantially outperforming Brain-JEPA at 81.06–82.29%.
- The masked distillation framework combined with the patch construction strategy resolves the temporal scale mismatch between pre-training and inference.

### Scaling Laws
- Provides the first detailed scaling analysis for fMRI foundation models.
- Linear probing performance follows a power-law relationship with the logarithm of pre-training data volume.
- Consistent scaling gains are observed on OOD tasks with no performance plateau.
- Continuous improvement is observed even when the HBN dataset exhibits an age gap of >20 years relative to UKB.

### Ablation Study

| Configuration | Avg. Score | Notes |
|---------------|-----------|-------|
| Full Brain-Semantoks | 52.39 | Semantic tokenizer + TTR + CLS + Tok |
| Remove TTR (0%) | 50.88 | Training instability; performance drops by 1.5 |
| TTR always active (100%) | 49.60 | Over-constrained; worse performance |
| Remove CLS loss | 47.32 | Global representation loss is critical |
| Linear projection replaces semantic tokenizer | Large gap | Induces partial collapse; cosine similarity rapidly rises to 0.95 |
| Random masking replaces slice masking | 51.03 | Slice masking reduces shortcut learning via interpolation |

## Highlights & Insights

- **Paradigm shift from reconstruction to abstraction**: Unlike all prior fMRI foundation models that use reconstruction objectives, Brain-Semantoks explicitly targets learning abstract representations. This paradigm shift yields substantial gains in linear probing performance.
- **Elegant design of the semantic tokenizer**: Neuroscientific priors (functional networks) are incorporated into the model architecture, simultaneously compressing sequence length (457→45) and aggregating noisy regional signals into semantic tokens — analogous to aggregating characters into words in NLP.
- **TTR as curriculum learning**: Addresses training instability arising from the combination of low-SNR data and self-distillation objectives. Activating TTR only during the first 5% of training steps precisely balances stability and flexibility.
- **Z-scoring > Robust Scaling**: This seemingly simple change in normalization strategy resolves DC offset inconsistencies across datasets during cross-dataset transfer, representing an important engineering contribution.
- **First fMRI scaling law analysis**: Demonstrates reliable OOD performance improvements with increasing pre-training data volume, bolstering community confidence in fMRI foundation models.

## Limitations & Future Work

- The functional network parcellation relies on the fixed Yeo 7-network atlas; future work could explore learning ROI groupings from data.
- Pre-training uses only resting-state fMRI (UKB); incorporating task-based fMRI data may further improve representation quality.
- Continuous targets are discretized into multi-class labels in downstream evaluations, which may not fully reflect the representations' capacity on regression tasks.
- Although the scaling analysis shows no plateau, it is limited by the UKB dataset size (~39K); the effects of larger-scale pre-training remain unknown.
- Single-GPU training efficiency is high (<2 h), but whether the 8-layer Transformer capacity constrains the learning of more complex patterns has not been investigated.

## Related Work & Insights

- **vs. BrainLM / Brain-JEPA**: Both adopt ROI-level mask-and-reconstruct objectives. Brain-Semantoks performs network-level semantic distillation and comprehensively outperforms them under linear probing, with gaps exceeding 10% on OOD clinical tasks.
- **vs. BrainMass**: BrainMass uses static functional connectivity matrices, ignoring temporal dynamics; Brain-Semantoks explicitly models temporal information.
- **vs. DINO / iBOT / SimDINO**: Brain-Semantoks successfully transfers the visual self-distillation paradigm to fMRI while introducing a domain-specific semantic tokenizer and TTR stabilization curriculum.
- **Insights**: The functional-network-as-semantic-tokenizer idea is generalizable to other neuroimaging modalities (EEG, MEG); the TTR strategy of "learn averages first, then details" may be broadly useful for self-distillation on other low-SNR data.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Paradigm shift (reconstruction → abstraction); semantic tokenizer and TTR are both original designs.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 downstream tasks × 6 datasets; linear probing + fine-tuning + ablation + scaling + interpretability — extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, logical progression, systematic ablations.
- Value: ⭐⭐⭐⭐⭐ Establishes a new paradigm for fMRI foundation models; surpasses supervised methods via linear probing alone; high practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] The Human Brain as a Combinatorial Complex](../../NeurIPS2025/time_series/the_human_brain_as_a_combinatorial_complex.md)
- [\[ICLR 2026\] Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning](towards_generalizable_pde_dynamics_forecasting_via_physics-guided_invariant_lear.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning of Time-Series Data](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-ser.md)
- [\[ICLR 2026\] GTM: A General Time-series Model for Enhanced Representation Learning](gtm_a_general_time-series_model_for_enhanced_representation_learning_of_time-series.md)
- [\[ICLR 2026\] Adapt Data to Model: Adaptive Transformation Optimization for Domain-shared Time Series Foundation Models](adapt_data_to_model_adaptive_transformation_optimization_for_domain-shared_time_.md)

<!-- RELATED:END -->
