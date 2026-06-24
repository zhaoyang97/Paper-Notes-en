---
title: >-
  [Paper Note] Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)
description: >-
  [ICML 2025][Medical Imaging][ECG signal interpretation] D-BETA proposes a contrastive learning framework that integrates generative masked autoencoders with enhanced discriminative capabilities. Through the ECG-Text Sigmoid (ETS) loss and Nearest Neighbor Negative Sampling (N3S) strategy, it significantly outperforms existing methods in cross-modal ECG-text representation learning, achieving a 15% average AUC improvement in linear probing with only 1% of the training data…
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "ECG signal interpretation"
  - "multimodal representation learning"
  - "contrastive masked autoencoders"
  - "zero-shot classification"
  - "cardiovascular diagnosis"
date: 2026-05-08
content_hash: ce0132b55ac1b8e5
---

# Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)

**Conference**: ICML 2025  
**arXiv**: [2410.02131](https://arxiv.org/abs/2410.02131)  
**Code**: [https://github.com/manhph2211/D-BETA](https://github.com/manhph2211/D-BETA)  
**Area**: Medical Imaging  
**Keywords**: ECG signal interpretation, multimodal representation learning, contrastive masked autoencoders, zero-shot classification, cardiovascular diagnosis

## TL;DR

D-BETA proposes a contrastive learning framework that integrates generative masked autoencoders with enhanced discriminative capabilities. Through the ECG-Text Sigmoid (ETS) loss and Nearest Neighbor Negative Sampling (N3S) strategy, it significantly outperforms existing methods in cross-modal ECG-text representation learning, achieving a 15% average AUC improvement in linear probing with only 1% of the training data, and a 2% improvement in zero-shot performance.

## Background & Motivation

Electrocardiograms (ECGs) are a core tool for diagnosing cardiovascular diseases, with standard 12-lead ECGs playing a critical role in diagnosing diseases such as arrhythmias. Although deep learning has progressed in automated ECG interpretation, it faces two major bottlenecks:

**Scarcity of Annotated Data**: Supervised methods rely on large amounts of expert-annotated data, which is highly expensive to acquire. Self-supervised learning (SSL) can learn robust representations from unlabeled data, but existing methods are split into contrastive and generative pathways, lacking effective integration.

**Underutilized Cross-Modal Information**: Clinical text reports contain rich diagnostic clues, but most existing ECG SSL methods ignore textual information. The few methods attempting cross-modal learning (such as MERL) primarily adopt standard ResNet + BERT architectures and rely solely on contrastive learning, which suffers from difficulties in negative sample selection and insufficient modeling of cross-modal relationships.

**Inherent Limitations of Contrastive Learning**: In medical datasets, random negative sampling easily yields false negatives (e.g., in MIMIC-IV ECG, there are only about 180,000 unique texts out of 800,000 records), which severely degrades contrastive learning effectiveness.

**Core Motivation**: Unify the generative (masked modeling) and discriminative (contrastive learning) paradigms into a single framework, while addressing negative sample quality issues, to achieve stronger cross-modal ECG-text representation.

## Method

### Overall Architecture

The D-BETA framework consists of two main branches and four learning objectives:

- **ECG Encoder ($\mathcal{F}_x$)**: A Transformer-based ECG signal encoder that outputs $H_x \in \mathbb{R}^{L_x \times d}$
- **Text Encoder ($\mathcal{F}_t$)**: Uses a pre-trained Flan-T5-base model, outputting $H_t \in \mathbb{R}^{L_t \times d}$
- **Fusion Module**: Inherently fuses both modalities via a cross-attention mechanism, outputting $H_f \in \mathbb{R}^{(L_x+L_t) \times d}$
- **Three Task Heads**: MLM decoder (text reconstruction), MEM decoder (ECG reconstruction), and ETM head (cross-modal matching)
- **Two Projection Heads**: $g_x$ and $g_t$, which work with the ETS loss to learn discriminative representations

The total loss is the sum of four terms: $\mathcal{L} = \mathcal{L}_{MLM} + \mathcal{L}_{MEM} + \mathcal{L}_{ETM} + \mathcal{L}_{ETS}$

### Key Designs

#### 1. ECG Encoder

- Input: $X \in \mathbb{R}^{L \times C}$ ($L$ is signal length, $C$ is number of channels)
- Preprocessing: Random lead masking ($p=0.5$) + Input Dropout ($p=0.1$) to achieve masked modeling
- Feature Extraction: Multi-layer convolution + GELU activation + Group Normalization $\rightarrow$ 768-dimensional projection
- Positional Encoding: Convolutional positional encoding preserves temporal information
- Backbone: 8-layer Transformer encoder layers (multi-head self-attention)

#### 2. Text Encoder

- Employs Flan-T5-base (first applied to the ECG domain), outputting a 768-dimensional embedding
- Flan-T5 is pre-trained on large-scale multi-task data and possesses strong text understanding capabilities
- Fine-tuned during the pre-training phase

#### 3. Fusion Module

- Linear projection maps outputs from both encoders to a 768-dimensional space
- Adds modality-specific embeddings to distinguish between ECG and text data
- **Cross-Attention Mechanism**: Allows each modality to attend to relevant features of the other modality to fully exploit complementary information

#### 4. ETS Loss Function (Core Innovation)

The traditional ETM loss in masked autoencoder architectures is based on a binary classification task of fused features and cannot directly enhance the discriminative capabilities of individual encoders. D-BETA proposes the ETS loss, inspired by SigLIP:

$$\mathcal{L}_{ETS} = -\frac{1}{\mathcal{B}} \sum_{i=1}^{\mathcal{B}} \sum_{j=1}^{\mathcal{B}} \log \frac{1}{1 + e^{-y_{ij} \cdot \mathbf{x}'_i{}^\top \mathbf{t}'_j}}$$

where $y_{ij}=1$ denotes a matched pair, and $y_{ij}=-1$ denotes a mismatched pair.

Key advantages of ETS:
- Based on Sigmoid instead of Softmax, avoiding the high computational cost of global normalization
- Computed independently for each ECG-text pair, improving memory efficiency and scalability
- Directly enhances the discriminative capacity of encoders via independent projection heads (Pooling $\rightarrow$ Tanh $\rightarrow$ Dense $\rightarrow$ 768-dimensional)

#### 5. Nearest Neighbor Negative Sampling Strategy (N3S)

To address the problem of false negatives generated by random negative sampling due to the high volume of duplicate/similar texts in medical datasets:

- Uses pre-trained Flan-T5 (small) to generate a 512-dimensional vector representation $v_t$ for each text
- During training, for a given positive sample pair $(x_k, t_k^+)$, negative samples $t_k^-$ are selected from the top-64 reports with the largest cosine distance
- N3S is applied to only half of the samples in a batch
- Efficient vector retrieval is implemented using the FAISS library to support large-scale datasets

Effect of N3S: ETM accuracy improves from $\sim 75\%$ without N3S to $>96\%$.

### Loss & Training

**Four Loss Functions**:

| Loss Function | Type | Formula | Role |
|---------|------|------|------|
| $\mathcal{L}_{MLM}$ | Cross-Entropy | Predict masked text tokens | Learn contextualized word embeddings |
| $\mathcal{L}_{MEM}$ | MSE | Reconstruct masked ECG signals | Capture ECG temporal structure |
| $\mathcal{L}_{ETM}$ | Binary Cross-Entropy | ECG-text pair matching classification | Align fusion feature spaces |
| $\mathcal{L}_{ETS}$ | Sigmoid Contrastive | Directly align ECG/text encoder outputs | Enhance encoder discriminative capability |

**Training Configuration**:

- Pre-training Dataset: MIMIC-IV-ECG v1.0 ($\sim 780,000$ paired ECG-text samples from 161,352 subjects)
- Optimizer: Adam ($lr=5\times10^{-5}$, $\beta_1=0.9$, $\beta_2=0.98$, $\varepsilon=10^{-6}$, weight decay=0.01)
- LR Scheduler: Three-stage scheduler (ratio 0.1:0.4:0.5)
- Training Steps: 300,000 steps, batch size=128
- Hardware: Single NVIDIA H100-80GB GPU

## Key Experimental Results

### Main Results

**Experiment 1: Full Fine-Tuning (PhysioNet 2021)**

| Method | 12-lead (Dx.) | 1-lead (Dx.) | 12-lead (Id.) |
|------|:---:|:---:|:---:|
| W2V+CMSC+RLM | 73.2 | 55.4 | 57.7 |
| **D-BETA** | **85.7** | **76.5** | **65.4** |
| Gain | +12.5 | +21.1 | +7.7 |

With only 1 lead, D-BETA (76.5%) outperforms the previous SOTA using all 12 leads (73.2%).

**Experiment 2: Linear Probing (Frozen Encoder)**

| Dataset | Data Ratio | D-BETA | MERL | Gain |
|--------|:---:|:---:|:---:|:---:|
| PTBXL-Rhythm | 1% | 86.61 | 53.33 | +33.28 |
| CSN | 1% | 70.10 | 58.26 | +11.84 |
| CPSC2018 | 1% | 85.46 | 70.33 | +15.13 |
| PTBXL-Rhythm | 100% | 96.71 | 88.34 | +8.37 |
| CPSC2018 | 100% | 94.92 | 90.57 | +4.35 |

**Experiment 3: Zero-Shot Classification**

| Dataset | D-BETA | MERL | Gain |
|--------|:---:|:---:|:---:|
| PTBXL-Super | 76.2 | 74.2 | +2.0 |
| PTBXL-Sub | 75.9 | 75.7 | +0.2 |
| PTBXL-Form | 66.1 | 65.9 | +0.2 |
| CSN | 88.6 | 78.5 | +10.1 |
| CODE-test | 80.1 | 82.8 | -2.7 |
| Average | **77.1** | 75.3 | **+1.8** |

**Highlight: On the CODE-test dataset, D-BETA zero-shot achieves 96.79% AUC, surpassing human cardiologists (92-94%) and supervised DNNs (96.59%).**

### Ablation Study

**Component Ablation (Table 6)**:

| Configuration | Fine-Tuning (Dx.) | Linear Probe (1%) | Zero-Shot | Description |
|------|:---:|:---:|:---:|------|
| Baseline (Bert, w/o ETS, w/o N3S) | 76.81 | 63.50 | – | Baseline model |
| + ETS | 78.29 | 67.19 | – | ETS improves $\sim 4\%$ |
| + N3S | 80.93 | 78.29 | 70.61 | N3S enables zero-shot |
| + Flan-T5 (Full D-BETA) | **85.70** | **80.93** | **72.82** | Flan-T5 adds another $4\%+$ |

**Text Encoder Ablation (Table 7)**:

| Encoder| Fine-Tuning | Linear Probe (1%) | Zero-Shot |
|--------|:---:|:---:|:---:|
| Bert | 78.08 | 77.58 | 69.14 |
| Deberta | 79.23 | 78.24 | 70.67 |
| Med-CPT | 81.02 | 79.57 | 71.81 |
| **Flan-T5** | **85.70** | **80.93** | **72.82** |

### Key Findings

1. **ETS loss is the largest contributor**: Leading to $\sim 15\%$ performance improvement, confirming the necessity of introducing an additional discriminative loss in masked autoencoders.
2. **N3S is crucial for zero-shot classification**: ETM accuracy improves from $\sim 75\%$ to $>96\%$, effectively addressing the problem of highly repetitive text in medical datasets.
3. **Flan-T5 is signficantly superior to the BERT series**: Leading across all experimental settings, indicating that stronger pre-trained language models are equally effective in the ECG domain.
4. **Lead combination experiments**: A 3-lead configuration (I, II, V2) achieves performance close to 12-lead (only 1.5% difference), offering significant clinical value.
5. **Zero-shot performance surpasses human experts**: On CODE-test, D-BETA zero-shot (96.79%) outperforms cardiologists (90.5-93.6%) and supervised models (96.59%).

## Highlights & Insights

1. **Unified Generative + Discriminative Framework**: D-BETA elegantly resolves the tension between masked autoencoders (biased towards reconstruction) and contrastive learning (biased towards discrimination), allowing them to complement each other through independent projection heads and the ETS loss.
2. **Sigmoid vs. Softmax Contrastive Loss**: Drawing inspiration from SigLIP's design, it avoids the overhead of Softmax global normalization, making it particularly suitable for large-scale pre-training.
3. **Domain-Aware Negative Sampling via N3S**: Leverages the Flan-T5 feature space and FAISS indexing for efficient negative sample selection, substantially outperforming random sampling in clinical scenarios where duplicate data is highly prevalent.
4. **Zero-Shot Enhancement with GPT-4o**: Employs a concise prompt to instruct GPT-4o to generate clinical descriptions, thereby augmenting class text encoding in a more controllable manner than MERL's database retrieval approach.
5. **Excellent Performance under Extreme Low-Data Regimes**: Obtains significant advantages using only 1% of the training data, demonstrating high clinical utility in label-sparse medical scenarios.

## Limitations & Future Work

1. **Single Pre-training Dataset**: Pre-training was only conducted on the MIMIC-IV-ECG dataset, which may introduce data distribution bias. Scaling up to multi-centric, multi-ethnic datasets is worth exploring.
2. **Reliance on Text Quality**: The pre-training effect is constrained by the quality of raw clinical reports; the brevity and highly repetitive nature of clinical notes might limit the performance upper bound.
3. **Computational Cost**: Although ETS is more efficient than Softmax contrastive loss, the 300,000 steps of pre-training still require an H100 GPU, limiting reproducibility for resource-constrained researchers.
4. **Zero-Shot Dependency on GPT-4o**: Optimal zero-shot results rely on GPT-4o to generate class descriptions, increasing the dependency on external LLMs.
5. **ECG-Specific Signal Processing**: Currently, ECG preprocessing is relatively simple (masking + dropout). Exploring more physiological signal-specific augmentation strategies is a potential avenue.
6. **Larger Models Unexplored**: Only the base version of Flan-T5 was used; larger versions might yield further improvements.

## Related Work & Insights

- **SigLIP (Zhai et al., 2023)**: The direct inspiration for the ETS loss, extending the Sigmoid pairwise loss from vision-language to ECG-text.
- **MERL (Liu et al., 2024b)**: The strongest baseline, which employs ResNet + BERT + cross-modal alignment + test-time knowledge augmentation. D-BETA improves upon it in both architecture and training strategy.
- **M3AE (Chen et al., 2022)**: A pioneering work in multimodal masked autoencoders, upon which D-BETA enhances discriminative capabilities.
- **CLIP (Radford et al., 2021)**: The classic paradigm for cross-modal contrastive learning. D-BETA combines the contrastive philosophy of CLIP with the generative philosophy of MAEs.
- **Inspiration for Future Work**: This framework can be extended to cross-modal learning between other medical signals (such as EEG, EMG) and clinical text reports.

## Rating

| Dimension | Score (1-5) | Description |
|------|:---:|------|
| Novelty | 4 | The unification of generative + discriminative paradigms, along with N3S negative sampling, represents substantial innovation |
| Technical Depth | 4 | Multiple carefully designed components with thorough ablation studies |
| Experimental Thoroughness | 5 | Evaluated across 5 datasets under 3 paradigms with extensive ablations |
| Practical Value | 4 | Extremely low data and zero-shot performance hold significant clinical value |
| Writing Quality | 4 | Well-structured with strong motivation |
| **Overall Score** | **4.2** | A solid work in multimodal medical representation learning |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining](from_token_to_rhythm_a_multi-scale_approach_for_ecg-language_pretraining.md)
- [\[NeurIPS 2025\] A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](../../NeurIPS2025/medical_imaging/a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)
- [\[CVPR 2026\] Masked-Diffusion Autoencoders for 3D Medical Vision Representation Learning](../../CVPR2026/medical_imaging/masked-diffusion_autoencoders_for_3d_medical_vision_representation_learning.md)
- [\[ICML 2026\] PaCX-MAE: Physiology-Augmented Chest X-Ray Masked Autoencoder](../../ICML2026/medical_imaging/pacx-mae_physiology-augmented_chest_x-ray_masked_autoencoder.md)
- [\[ICLR 2026\] Boosting Medical Visual Understanding From Multi-Granular Language Learning](../../ICLR2026/medical_imaging/boosting_medical_visual_understanding_from_multi-granular_language_learning.md)

</div>

<!-- RELATED:END -->
