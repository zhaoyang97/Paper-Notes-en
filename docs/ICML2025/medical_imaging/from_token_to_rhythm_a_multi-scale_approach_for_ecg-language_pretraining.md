---
title: >-
  [Paper Note] From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining
description: >-
  [ICML 2025][Medical Imaging][ECG Pretraining] MELP proposes a multi-scale ECG-language pretraining model. By utilizing cross-modal supervisory signals at three levels (Token, Beat, and Rhythm) combined with domain-specific cardiology language model pretraining, it comprehensively outperforms existing self-supervised and multimodal ECG methods in zero-shot classification, linear probing, and transfer learning.
tags:
  - "ICML 2025"
  - "Medical Imaging"
  - "ECG Pretraining"
  - "Multimodal Learning"
  - "Multi-scale Representation"
  - "Contrastive Learning"
  - "ECG-Text Alignment"
  - "Zero-shot Classification"
  - "Self-Supervised Learning"
date: 2026-05-08
content_hash: 5ff7219530da1f1b
---

# From Token to Rhythm: A Multi-Scale Approach for ECG-Language Pretraining

**Conference**: ICML 2025  
**arXiv**: [2506.21803](https://arxiv.org/abs/2506.21803)  
**Code**: [https://github.com/HKU-MedAI/MELP](https://github.com/HKU-MedAI/MELP)  
**Area**: Medical AI / ECG Analysis  
**Keywords**: ECG Pretraining, Multimodal Learning, Multi-scale Representation, Contrastive Learning, ECG-Text Alignment, Zero-shot Classification, Self-Supervised Learning

## TL;DR

MELP proposes a multi-scale ECG-language pretraining model. By utilizing cross-modal supervisory signals at three levels (Token, Beat, and Rhythm) combined with domain-specific cardiology language model pretraining, it comprehensively outperforms existing self-supervised and multimodal ECG methods in zero-shot classification, linear probing, and transfer learning.

## Background & Motivation

Electrocardiograms (ECGs) are a core tool for diagnosing cardiovascular diseases. While deep learning has significantly improved ECG analysis, it faces the challenge of high costs associated with large-scale manual annotation. Consequently, self-supervised learning (SSL) has emerged as a promising alternative.

Limitations of Prior Work:

**Limitations of Unimodal SSL**: Contrastive (e.g., SimCLR, CLOCS) and generative (e.g., ST-MEM, HeartLang) methods only utilize ECG signals, ignoring the rich semantic information embedded in clinical texts.

**Insufficiency of Global Alignment**: The few existing ECG-language alignment methods (e.g., MERL) only focus on global ECG-to-text alignment, neglecting the multi-scale structure of ECG signals.

**Omission of Hierarchical Structure**: Cardiologists interpret ECGs in a hierarchical manner—from waveform components (token level) to heartbeat cycles (beat level) and finally to overall rhythms (rhythm level). Existing models fail to capture this multi-scale nature.

Core Motivation: To mimic the multi-scale interpretation process of clinicians by establishing cross-modal supervision between ECG and text across three levels: token, beat, and rhythm.

## Method

### Overall Architecture

MELP (Multi-scale ECG-Language Pretraining) consists of the following components (Figure 2):

1. **Cardiology Language Model Pretraining** (Stage 1)
2. **Multimodal Pretraining** (Stage 2) — Three-level Supervision

### Key Designs

#### 1. Cardiology Language Pretraining

Based on the MedCPT query encoder, masked language modeling (MLM) pretraining is conducted using cardiology corpora from three sources:
- PubMed cardiology-related literature
- Wikipedia cardiology articles
- Clinical reports from the MIMIC-IV-ECG training set

The objective is to equip the text encoder with rich domain knowledge in cardiology, laying a high-quality semantic representation foundation for subsequent cross-modal alignment.

#### 2. Token-level: ECG Description Generation

Adopting an encoder-decoder framework, the ECG encoder generates token-level embeddings $E \in \mathbb{R}^{L_t \times D}$, which are aggregated into $\tilde{E} \in \mathbb{R}^{128 \times D}$ via attention pooling with 128 learnable query tokens. The text decoder autoregressively generates paired reports in a GPT-style manner:

$$\mathcal{L}_{\mathrm{LM}}(\zeta) = -\sum_{i=1}^{N} \log p(w_i | w_{0:i-1}, \tilde{E})$$

Through the report generation task, the model is forced to capture fine-grained waveform features (e.g., absence of P-wave, QRS duration), implicitly learning the relationships between these local indicators and clinical diagnoses.

#### 3. Beat-level: Heartbeat-Sentence Alignment

Introducing 10 learnable tokens, token-level features are aggregated into beat-level representations $B \in \mathbb{R}^{N_B \times D}$ via attention pooling. On the text side, word tokens are averaged per sentence to generate sentence embeddings $S \in \mathbb{R}^{S \times D}$.

Attention-weighted beat embedding:

$$\hat{B}(l) = \sum_{l=1}^{N_B} \alpha_l S(l), \quad \alpha(l) = \frac{\exp(\langle S(l), B(l) \rangle / \tau_1)}{\sum_j \exp(\langle S(l), B(j) \rangle / \tau_1)}$$

Local contrastive loss:

$$\mathcal{L}_{\mathrm{Local}} = \frac{1}{2}(\mathcal{L}_{\mathrm{Local}}^{e \to t} + \mathcal{L}_{\mathrm{Local}}^{t \to e})$$

Through the beat-sentence matching mechanism, the model captures correspondences between transient abnormal heartbeats and specific descriptive sentences.

#### 4. Rhythm-level: Global ECG-Report Alignment

The global ECG embedding $X_i^g$ is computed as the average of all beat embeddings, while the global text embedding $T_i^g$ is represented by the [CLS] token. A standard InfoNCE contrastive loss is employed:

$$\mathcal{L}_g^{e \to t} = -\frac{1}{B}\log \frac{\exp(\langle X_i^g, T_i^g \rangle / \tau)}{\sum_{j=1}^{B} \exp(\langle X_i^g, T_j^g \rangle / \tau)}$$

#### 5. Total Loss

$$\mathcal{L} = \mathcal{L}_g + \lambda_1 \mathcal{L}_{\mathrm{LM}} + \lambda_2 \mathcal{L}_{\mathrm{Local}}$$

Where $\lambda_1 = 2$ and $\lambda_2 = 0.2$.

## Key Experimental Results

### Linear Probing (AUC%, 100% Training Data)

| Method | PTBXL-Rhythm | PTBXL-Super | CPSC2018 | CSN |
|------|-------------|-------------|----------|-----|
| SimCLR | 77.73 | 73.53 | 76.54 | 73.20 |
| Wav2Vec2+CMSC+RLM | 92.05 | 85.53 | **92.61** | 87.87 |
| MERL | 92.31 | 88.01 | 92.48 | 87.39 |
| **MELP** | **93.66** | **89.44** | 92.49 | **90.25** |

### Linear Probing (AUC%, 1% Training Data)

| Method | PTBXL-Rhythm | PTBXL-Super | CPSC2018 | CSN |
|------|-------------|-------------|----------|-----|
| SimCLR | 51.41 | 63.41 | 59.78 | 59.02 |
| MERL | 84.85 | 84.46 | 81.49 | 73.23 |
| **MELP** | **87.72** | **87.63** | **82.05** | **77.65** |

Key Findings:
- The advantages of MELP are most prominent in extremely low-resource annotation scenarios (1%), where it outperforms MERL by 2-4 percentage points across multiple datasets.
- With 100% data, MELP consistently maintains SOTA performance on average across six evaluation sets.

### Zero-Shot Classification

MELP achieves top performance in zero-shot ECG classification, demonstrating the efficacy of multi-scale supervision in enhancing the quality of cross-modal alignment.

### Ablation Study

- Removal of token-level supervision $\to$ Largest performance drop
- Removal of rhythm-level supervision $\to$ Loss of zero-shot capability
- Removal of beat-level supervision $\to$ Moderate performance drop
- None of the three components are dispensable, verifying the complementary nature of multi-scale supervision.

## Highlights & Insights

1. **Clinically-Inspired Design**: The supervisory signals are designed by mimicking the multi-scale interpretation process of cardiologists (waveform $\to$ heartbeat $\to$ rhythm), which offers strong physiological plausibility.
2. **Cardiology-Specific Language Model**: Pretraining the text computer on domain-specific corpora prior to cross-modal alignment proves significantly superior to directly utilizing generic language models.
3. **Integration of Generative and Contrastive Learning**: Token-level captioning (generative) and Beat/Rhythm-level contrastive learning (discriminative) complement each other, avoiding the limitations of a single paradigm.
4. **Flexible Downstream Adaptation**: Generative pretraining allows the model to naturally extend to tasks such as ECG report generation and ECG question-answering.

## Limitations & Future Work

1. The beat level uses a fixed number of 10 learnable tokens, lacking adaptive adjustment based on the actual heart rate.
2. The pretraining data (MIMIC-IV-ECG) originates from a single source, and cross-institutional generalization remains to be verified.
3. The text decoder is initialized randomly, potentially requiring more data for sufficient training.
4. The computational overhead of token-level captioning is relatively high.

## Related Work & Insights

- **ECG Representation Learning**: SimCLR, CLOCS, Wav2Vec 2.0, ST-MEM, HeartLang
- **ECG-Language Pretraining**: MERL, C-MELT, ECG-Chat, ESI
- **General Multimodal Contrastive Learning**: CLIP, CoCa

## Rating

⭐⭐⭐⭐ — The multi-scale design concept is elegant and physiologically grounded, with comprehensive experimental coverage (six datasets, multiple evaluation protocols). The introduction of cardiology language pretraining provides an important practical insight. The open-source code facilitates reproducibility and subsequent research. The fixed number of beat-level tokens and single-source pretraining are the primary limitations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] EEG-Language Pretraining for Highly Label-Efficient Clinical Phenotyping](eeg-language_pretraining_for_highly_label-efficient_clinical_phenotyping.md)
- [\[NeurIPS 2025\] A Novel Approach to Classification of ECG Arrhythmia Types with Latent ODEs](../../NeurIPS2025/medical_imaging/a_novel_approach_to_classification_of_ecg_arrhythmia_types_with_latent_odes.md)
- [\[ICML 2025\] Boosting Masked ECG-Text Auto-Encoders as Discriminative Learners (D-BETA)](boosting_masked_ecg-text_auto-encoders_as_discriminative_learners.md)
- [\[NeurIPS 2025\] PhysioWave: A Multi-Scale Wavelet-Transformer for Physiological Signal Representation](../../NeurIPS2025/medical_imaging/physiowave_a_multi-scale_wavelet-transformer_for_physiological_signal_representa.md)
- [\[CVPR 2026\] TAMER: A Tri-Modal Contrastive Alignment and Multi-Scale Embedding Refinement Framework for Zero-Shot ECG Diagnosis](../../CVPR2026/medical_imaging/tamer_a_tri-modal_contrastive_alignment_and_multi-scale_embedding_refinement_fra.md)

</div>

<!-- RELATED:END -->
