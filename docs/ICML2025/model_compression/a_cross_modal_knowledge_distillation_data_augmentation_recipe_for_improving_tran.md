---
title: >-
  [Paper Note] A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features
description: >-
  [ICML 2025][Model Compression][Cross-modal knowledge distillation] Proposes Semi-Clipped (a CLIP-based cross-modal distillation method) and PEA (Perturbation Embedding Augmentation). In weakly paired data scenarios, these methods distill rich morphological features from microscopy images into transcriptomics representations, significantly improving their predictive power while maintaining the interpretability of gene expression.
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Cross-modal knowledge distillation"
  - "Data augmentation"
  - "Transcriptomics"
  - "Microscopy imaging"
  - "Weakly paired learning"
date: 2026-05-08
content_hash: 9c5ec30b6fceb8fb
---

# A Cross Modal Knowledge Distillation & Data Augmentation Recipe for Improving Transcriptomics Representations through Morphological Features

**Conference**: ICML 2025  
**arXiv**: [2505.21317](https://arxiv.org/abs/2505.21317)  
**Code**: None  
**Area**: Interpretability  
**Keywords**: Cross-modal knowledge distillation, Data augmentation, Transcriptomics, Microscopy imaging, Weakly paired learning

## TL;DR

Proposes Semi-Clipped (a CLIP-based cross-modal distillation method) and PEA (Perturbation Embedding Augmentation). In weakly paired data scenarios, these methods distill rich morphological features from microscopy images into transcriptomics representations, significantly improving their predictive power while maintaining the interpretability of gene expression.

## Background & Motivation

### Core Problem

Understanding cellular responses to various stimuli is fundamental to biological discovery and drug development. Currently, there are two complementary biological data modalities:

- **Transcriptomics**: Provides gene-level interpretable insights directly, but has relatively weak predictive performance.
- **Microscopy Imaging**: Contains rich visual phenotypic features and powerful predictive capabilities, but is difficult to interpret.

Each modality has its strengths and limitations; ideally, they should be combined to build a more comprehensive representation of biological systems. However, **collecting sample-level matched multi-modal data is extremely expensive**, making it technically and financially infeasible.

### Challenges of Weakly Paired Data

In practical scenarios, only **weakly paired datasets** can be acquired—where samples from different modalities do not originate from the same biological replicate experiment, but share key metadata such as cell lines and perturbation conditions. Even weakly paired datasets are extremely scarce, severely limiting training and inference in multi-modal learning.

### Motivation

- Most cross-modal distillation techniques rely on supervised objectives requiring precise labels, which are often unavailable in biological modalities.
- Unsupervised alignment methods attempt to discover shared structures across modalities, which is challenging because the biological relationships captured by the two modalities are inherently different.
- There is a need for a method that works under data scarcity, requires only weakly paired information, and uses only a single modality during inference.

## Method

### Overall Architecture

The proposed framework consists of two core components:

1. **Semi-Clipped**: A cross-modal knowledge distillation method that transfers knowledge from microscopy images (teacher modality $T$) to transcriptomics (student modality $S$).
2. **PEA (Perturbation Embedding Augmentation)**: A biologically-inspired data augmentation technique specifically designed for transcriptomics representations.

**Problem Formulation**: Given a teacher modality dataset $\mathcal{X}_T$ and a student modality dataset $\mathcal{X}_S$, samples $x_T^{(i)}$ and $x_S^{(i)}$ correspond to the same biological perturbation and cell type, but are not strongly paired due to biological variability. Each sample is annotated with weak labels $p$ (perturbation type) and $l$ (cell type). The data is organized in experimental batches (biological batches), where each batch contains perturbed samples and control (unperturbed) samples.

### Key Designs

#### Semi-Clipped: Distillation with Frozen Teacher via CLIP

The core idea is to leverage frozen pre-trained unimodal encoders to achieve unidirectional knowledge transfer via a trainable adapter:

1. **Frozen Encoders**: Use pre-trained unimodal encoders $E_T$ and $E_S$ to generate teacher embeddings $z_T^{(i)} = E_T(x_T^{(i)})$ and student embeddings $z_S^{(i)} = E_S(x_S^{(i)})$, respectively.
2. **Trainable Adapter**: Learn a mapping function $f_S: \mathbb{R}^{d_S} \to \mathbb{R}^{d_T}$ to align student embeddings into the teacher embedding space, generating $h_S^{(i)} = f_S(z_S^{(i)})$.
3. **CLIP Loss Alignment**: Optimize the CLIP loss between $h_S$ and $z_T$.

**Key Differences from Standard CLIP**:

- Standard CLIP trains both encoders bidirectionally, which easily leads to modality drift when shared information is limited.
- Semi-Clipped **freezes the teacher side** and only trains the student adapter $f_S$.
- This ensures **unidirectional knowledge transfer** (teacher $\to$ student) and avoids the bidirectional drift problem.
- Weak labels are only used to establish pairing relationships, rather than as learning targets.

#### PEA: Perturbation Embedding Augmentation

The key innovation of PEA is to **repurpose batch effect correction as a data augmentation strategy**:

**Biological Background**: Batch effects (variations introduced by differences in experimental conditions) in biological datasets are typically treated as noise that needs to be eliminated. Traditional batch correction techniques denoise the data by centering embeddings on the control samples of each batch.

**Mechanism**:

1. Randomly select a transformation $A$ from a predefined set of batch correction transformations $\mathcal{A}$.
2. Apply augmentation to the student embedding: $z_{S,A}^{(i)} = A(z_S^{(i)}, X_S^{(c)})$, where $X_S^{(c)}$ represents the control samples.
3. The augmented embeddings are passed to the student adapter $f_S$ for cross-modal distillation.
4. The teacher side uses a fixed batch correction $B$ (TVN).

**Key Advantages of PEA**:

- Exploits variations generated by different batch correction methods as a source of "augmentation".
- Different correction methods eliminate different noise patterns; randomly switching between them is equivalent to introducing biologically meaningful variations.
- The augmented data retains core biological information (perturbation signals) while enriching training data diversity.
- Particularly suitable for representation-level augmentation, distinguishing it from image-level rotations/flips.

### Loss & Training

**Loss Function**: A standard CLIP contrastive loss is computed between $h_S = f_S(z_{S,A})$ (augmented student embedding mapped via the adapter) and $z_T$ (frozen teacher embedding after batch correction). Positive pairs consist of cross-modal samples sharing the same weak labels (perturbation + cell line).

**Key Training Strategies**:

- Both the teacher encoder $E_T$ and the student encoder $E_S$ are **frozen** and do not participate in training.
- Only the student adapter $f_S$ (a lightweight MLP) is trained.
- PEA augmentation is applied randomly online during training.
- The teacher side consistently uses Typical Variation Normalization (TVN) for batch correction.
- Inference requires only the student modality (transcriptomics) data.

## Key Experimental Results

### Main Results

The paper evaluates on biological downstream tasks related to drug discovery, mainly including perturbation retrieval and gene relationship discovery.

| Method | Modality | Weak Pairing Dependency | Retrieval Performance | Characteristics |
|------|------|-----------|---------|------|
| scVI (baseline) | Transcriptomics | No | Baseline | Unimodal foundation model |
| Standard CLIP | Bi-modal | Yes | Medium | Bidirectional drift issue |
| CSA | Bi-modal | Yes | Better | few-shot alignment |
| **Semi-Clipped** | Distilled $\to$ Unimodal | Yes | **SOTA** | Frozen teacher, prevents drift |
| **Semi-Clipped + PEA** | Distilled $\to$ Unimodal | Yes | **Best** | Augmentation brings further gains |

### Ablation Study

| Configuration | Key Metric Change | Explanation |
|------|------------|------|
| Semi-Clipped (No Augmentation) | Baseline | Distillation alone already achieves SOTA |
| + Traditional Augmentation (Cutout/Mixup) | No significant gain or drop | Traditional augmentations are not suitable for biological representations |
| + Gaussian Noise | Slight gain | Random noise acts as a regularizer to some extent |
| + PEA (Single Correction) | Moderate gain | Augmentation from a single correction method is limited |
| + **PEA (Random Multi-Correction)** | **Max Gain** | Combination of multiple correction methods performs best |
| Bidirectional CLIP (Train both sides) | Performance drop | Confirms the modality drift issue |
| Freeze student, train teacher only | Significant performance drop | Erroneous distillation direction |

### Key Findings

1. **Semi-Clipped significantly outperforms standard CLIP and other alignment methods under data scarcity**—freezing the teacher side is key to success.
2. **PEA performs best among all augmentation methods**—traditional augmentations like Cutout, Mixup, and Gaussian noise have limited or even negative effects on biological representations.
3. **Distilled transcriptomics representations surpass original microscopy features in downstream tasks**—indicating that distillation not only transfers teacher knowledge but also complements the student modality's original information.
4. **PEA is particularly outstanding in discovering novel gene relationships**—significantly outperforming other augmentation techniques.

## Highlights & Insights

1. **Repurposing batch effects as augmentation**: Traditionally, batch effects are considered noise that needs to be eliminated. This work creatively transforms them into a source of augmentation. Different batch correction methods produce different "perspectives"; randomly switching between them is equivalent to observing biological signals from multiple angles.
2. **Unidirectional distillation strategy**: By freezing the teacher side, Semi-Clipped elegantly solves the modality drift challenge in weakly paired scenarios. This design concept is also inspiring for other weakly paired cross-modal learning scenarios.
3. **Inference requires only a single modality**: Multi-modal data is used during training, but only the cheaper, more interpretable transcriptomics data is needed during inference. This "asymmetric training-inference" design is highly practical in real-world applications.
4. **Method generality**: The concepts of Semi-Clipped and PEA can be extended to other weakly paired multi-modal scenarios (e.g., proteomics-transcriptomics, imaging-genomics, etc.).

## Limitations & Future Work

1. **Selection of the augmentation set**: PEA relies on a predefined set of batch correction methods; how to automatically select or learn the optimal combination of correction methods is worth exploring.
2. **Limited to weakly paired scenarios**: Whether there is still an advantage when strongly paired data is available has not been fully discussed in the paper.
3. **Adapter architecture**: Currently, a simple MLP is used as the adapter; whether more complex architectures (such as cross-attention) can bring further improvements remains to be investigated.
4. **Scalability**: Whether the freezing strategy of Semi-Clipped remains optimal as the scale of the weakly paired dataset increases.
5. **Biological validation**: New gene relationships discovered by the distilled representations still require experimental validation.

## Related Work & Insights

- **CLIP (Radford et al., 2021)**: The direct inspiration for Semi-Clipped, which adapts it as a distillation tool by freezing one side.
- **CSA (Li et al., 2024)**: Uses few-shot alignment of pre-trained unimodal models, but bidirectional training limits its performance under data scarcity.
- **scVI (Lopez et al., 2018)**: A foundation model in transcriptomics used as the student encoder.
- **VICReg (Bardes et al., 2022)**: A self-supervised alignment method, but requires significant shared information between modalities.
- **Batch correction literature**: Batch correction methods such as TVN (Ando et al., 2017) form the technical foundation of PEA.

**Insight**: The concept of "repurposing pre-processing/post-processing steps as training-time augmentations" has broad general significance—similar "wasted transformational diversity" might exist and be exploited in other fields.

## Rating

| Dimension | Score (1-5) | Explanation |
|------|-----------|------|
| Innovation | 4 | Semi-Clipped is simple yet effective; PEA's "batch effect $\to$ augmentation" concept is highly novel |
| Technical Depth | 3.5 | The method itself is not complex, but the problem modeling and biological integration are well executed |
| Experimental Thoroughness | 4 | Multiple downstream tasks and ablation studies, with biologically meaningful evaluations |
| Writing Quality | 4 | Clear structure and well-formulated problem motivation |
| Value | 4.5 | Solves real pain points in drug discovery with low inference costs |
| **Overall** | **4.0** | A highly practical cross-modal distillation work with valuable biological insights |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ATLAS: Autoformalizing Theorems through Lifting, Augmentation, and Synthesis of Data](../../NeurIPS2025/model_compression/atlas_autoformalizing_theorems_through_lifting_augmentation_and_synthesis_of_dat.md)
- [\[AAAI 2026\] Asymmetric Cross-Modal Knowledge Distillation: Bridging Modalities with Weak Semantic Consistency](../../AAAI2026/model_compression/asymmetric_cross-modal_knowledge_distillation_bridging_modalities_with_weak_sema.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](../../AAAI2026/model_compression/distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[CVPR 2025\] Multi-modal Knowledge Distillation-based Human Trajectory Forecasting](../../CVPR2025/model_compression/multi-modal_knowledge_distillation-based_human_trajectory_forecasting.md)
- [\[ACL 2025\] Data Laundering: Artificially Boosting Benchmark Results through Knowledge Distillation](../../ACL2025/model_compression/data_laundering_artificially_boosting_benchmark_results_through_knowledge_distil.md)

</div>

<!-- RELATED:END -->
