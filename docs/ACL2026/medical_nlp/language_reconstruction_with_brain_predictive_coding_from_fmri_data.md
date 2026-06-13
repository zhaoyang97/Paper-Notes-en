---
title: >-
  [Paper Note] Language Reconstruction with Brain Predictive Coding from fMRI Data
description: >-
  [ACL 2026][Medical NLP][fMRI Language Reconstruction] This paper proposes PredFT, an end-to-end fMRI-to-Text decoding model that combines a main network (language decoding) and a side network (brain predictive coding rep…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "fMRI Language Reconstruction"
  - "Predictive Coding"
  - "Brain Signal Decoding"
  - "Neurolinguistics"
  - "Side Network"
date: 2026-05-08
content_hash: 05d5e4acbf6919ff
---

# Language Reconstruction with Brain Predictive Coding from fMRI Data

**Conference**: ACL 2026  
**arXiv**: [2405.11597](https://arxiv.org/abs/2405.11597)  
**Code**: None  
**Area**: Brain-Computer Interface / Language Decoding  
**Keywords**: fMRI Language Reconstruction, Predictive Coding, Brain Signal Decoding, Neurolinguistics, Side Network

## TL;DR

This paper proposes PredFT, an end-to-end fMRI-to-Text decoding model that combines a main network (language decoding) and a side network (brain predictive coding representation). By extracting prospective semantic representations from prediction-related brain regions (PTO areas) and fusing them into the decoding process, it achieves a BLEU-1 of 34.95% on the LeBel dataset (Sub-1), a 7.84 percentage point improvement over the strongest baseline, MapGuide.

## Background & Motivation

**Background**: Reconstructing natural language from fMRI signals is a crucial window for understanding human brain language formation mechanisms. Recent studies have utilized pre-trained language models to achieve open-vocabulary fMRI-to-Text decoding: Tang et al. use GPT to generate semantic candidates and then use brain signals to select matches, while Xi et al. frame the problem as sequence-to-sequence translation.

**Limitations of Prior Work**: Existing research focuses on model architecture design and the utilization of language models but ignores a key neuroscientific foundation—how natural language is encoded in the human brain. Specifically, the brain naturally makes multi-time-scale predictions of future content while perceiving current speech stimuli (Predictive Coding Theory), yet this information has never been used to guide language reconstruction.

**Key Challenge**: Brain signals contain rich prospective predictive information, but existing decoding models only utilize brain activity representations from the current moment, wasting naturally generated predictive signals from the brain.

**Goal**: (1) Validate the feasibility of predictive coding theory in fMRI-to-Text decoding; (2) Design a decoding model that effectively utilizes brain predictive representations; (3) Analyze the impact of different brain regions, prediction distances, and lengths on decoding performance.

**Key Insight**: Predictive coding theory states that the brain naturally predicts future words when hearing speech. Caucheteux et al. proved that linear mapping between language model activations and brain responses is enhanced if predictive content is used to construct language model representations. This inspires: Can predictive representations be extracted from brain signals to assist language reconstruction?

**Core Idea**: Design a dual-network architecture—the main network is responsible for standard fMRI-to-Text decoding, while the side network extracts prospective representations from prediction-related brain regions (PTO areas), merging predictive information into the decoding process via Predictive Coding Attention.

## Method

### Overall Architecture

PredFT is an end-to-end model comprising a main network $\mathcal{M}_\theta$ (encoder-decoder) and a side network $\mathcal{M}_\phi$ (encoder-decoder). The main network encodes fMRI sequences into spatio-temporal features and uses a Transformer decoder to generate text; the side network extracts representations from prediction-related brain regions (ROIs) and fuses them through self-attention, with its encoder output $H_{\phi_\text{Enc}}^M$ injected into the main network as predictive representations. The side network decoder is discarded during inference.

### Key Designs

1.  **Main Network Encoder (fMRI Feature Extraction + Temporal Modeling)**:
    -   **Function**: Extracts spatio-temporal features from raw fMRI signals.
    -   **Mechanism**: For 4D volume fMRI images $F_{i,j} \in \mathbb{R}^{w \times h \times d \times (k+1)}$, an $L$-layer 3D-CNN (with Group Normalization, ReLU, and residual connections) is used to step-wise reduce dimensionality to a 1D vector $x_{i,j}^t \in \mathbb{R}^{d_m}$; for 2D surface fMRI, linear layers are used. An FIR model $g_t$ is then applied to compensate for BOLD signal delay, concatenating $k-k^*$ future frames followed by linear fusion. Finally, temporal positional encoding is added before entering a Transformer encoder to capture temporal dependencies.
    -   **Design Motivation**: The BOLD signal of fMRI has a delay of approximately 4-6 seconds; temporal compensation via the FIR model is critical for correctly aligning brain activity with speech.

2.  **Side Network (Brain Predictive Representation Extraction)**:
    -   **Function**: Extracts prospective semantic representations from prediction-related brain regions.
    -   **Mechanism**: The side network encoder $\mathcal{M}_{\phi_\text{Enc}}$ receives sequences $R_{i,j}$ from prediction-related ROIs (concatenated from STS, IFG, SMG, and Angular Gyrus), reduces dimensionality via fully connected layers, applies FIR compensation and positional encoding, and feeds them into a Transformer encoder to output predictive representations $H_{\phi_\text{Enc}}^M$. The side network decoder takes future words $V_j$ (prediction targets at distance $d$ and length $l$) as input, and the encoder is trained to learn predictive representations via cross-entropy loss. The decoder is discarded during inference.
    -   **Design Motivation**: Predictive coding validation experiments show that prediction scores in PTO areas are significantly higher than whole-brain or random ROIs, making the selection of correct brain regions vital for extracting effective signals.

3.  **Predictive Coding Attention**:
    -   **Function**: Fuses predictive representations from the side network into the main network decoder.
    -   **Mechanism**: A PC-Attention module is added to each layer of the main network Transformer decoder, using the decoder hidden state $H_{\theta_\text{Dec}}^l$ as the query and the side network encoder output $H_{\phi_\text{Enc}}^M$ as the key and value. The key is the design of the attention mask: for each token in a text segment $u_j^t$, it is allowed to attend to all predictive representations at and after time step $t$, masking those before—since predictive information should come from brain activity following the current moment.
    -   **Design Motivation**: The mask design ensures causality—decoding the current word only utilizes future predictive information, matching the "prospective" nature of predictive coding.

### Loss & Training

Joint end-to-end training is employed with total loss $\mathcal{L} = \mathcal{L}_\text{Main} + \lambda \mathcal{L}_\text{Side}$. The two networks share a word embedding layer (updated only by main network gradients) and respectively use left-to-right autoregressive cross-entropy loss. During inference, the side network decoder is discarded, retaining only the encoder to provide predictive representations.

## Key Experimental Results

### Main Results

**Within-subject decoding on LeBel dataset (10 frames = 20 seconds)**

| Model | BLEU-1 | BLEU-4 | ROUGE1-F | BERTScore |
| :--- | :--- | :--- | :--- | :--- |
| Tang's | 22.25 | 0.00 | 19.44 | 80.84 |
| BrainLLM | 24.18 | 1.11 | 21.16 | 83.26 |
| MapGuide | 27.11 | 1.54 | 24.83 | 82.66 |
| PredFT w/o SideNet | 27.91 | 1.29 | 26.82 | 81.35 |
| **Ours (PredFT)** | **34.95** | **1.78** | **32.03** | 82.92 |

**Cross-subject decoding on Narratives dataset**

| Length | Model | BLEU-1 | ROUGE1-F | BERTScore |
| :--- | :--- | :--- | :--- | :--- |
| 10 frames | UniCoRN | 20.64 | 19.23 | 75.35 |
| 10 frames | **Ours** | **24.73** | **19.53** | **78.52** |
| 40 frames | UniCoRN | 21.76 | 25.30 | 74.40 |
| 40 frames | **Ours** | **27.80** | **25.96** | **78.63** |

### Ablation Study

**Impact of ROI selection on decoding performance (LeBel dataset)**

| ROI Type | Description | Relative Performance |
| :--- | :--- | :--- |
| BPC (Prediction ROIs) | STS, IFG, SMG, Angular Gyrus | **Optimal** |
| Whole (Whole Brain) | Entire cerebral cortex | Sub-optimal |
| Random | Randomly selected regions | Worst |

### Key Findings

-   The side network contributes significantly: PredFT improves BLEU-1 from 27.91 to 34.95 (+7.04 Gain) on Sub-1 compared to w/o SideNet, proving that brain predictive information substantially aids decoding.
-   ROI selection is crucial: BPC regions (PTO) consistently outperform whole-brain and random ROIs, validating the regional specificity of predictive coding.
-   Optimal intervals exist for prediction length and distance: Predictive lengths that are too short ($l=1,2$) or too long ($l=11,12$) are suboptimal; medium lengths ($l=6,7,8$) paired with appropriate distances ($d=3-5$) yield the best results.
-   Within-subject decoding is significantly better than cross-subject decoding, and all models still struggle with long-text generation (BLEU-3/4).

## Highlights & Insights

-   Transforming the neuroscientific predictive coding theory directly into model design is an elegant interdisciplinary innovation—the side network's "train with it, infer without it" strategy is similar to knowledge distillation.
-   The causal mask design of PC-Attention is simple yet powerful—it allows current words to focus only on future predictive representations, perfectly corresponding to the prospective nature of predictive coding.
-   The predictive coding validation experiment itself has independent value—systematically demonstrating the interaction between brain regions, prediction distance, and length.

## Limitations & Future Work

-   Validated only on fMRI data; the applicability to other brain signal modalities (MEG, EEG) has not been explored.
-   Content unexpected by the subject might interfere with brain prediction functions, affecting decoding performance.
-   There remains significant room for improvement in high-precision long-text generation for all models (BLEU-4 generally below 2%).
-   Could be extended to brain signal decoding scenarios for visual stimuli.

## Related Work & Insights

-   **vs Tang's**: Tang uses GPT beam search to generate candidates for selection; PredFT is end-to-end and utilizes brain predictive information.
-   **vs UniCoRN**: UniCoRN uses a three-stage training framework with BART; PredFT introduces predictive coding priors through a side network, achieving a 6+ percentage point Gain in BLEU-1 in cross-subject settings.
-   **vs BrainLLM**: BrainLLM fine-tunes Llama2 by concatenating fMRI embeddings with word embeddings; PredFT provides more targeted auxiliary signals through an independent prediction network.

## Rating

-   Novelty: ⭐⭐⭐⭐⭐ First systematic application of predictive coding theory to fMRI-to-Text decoding; outstanding interdisciplinary innovation.
-   Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive analysis across two datasets, multiple subjects, ROIs, and prediction parameters, though lacking human evaluation.
-   Writing Quality: ⭐⭐⭐⭐ Clear and fluid logical progression from predictive coding validation to model design.
-   Value: ⭐⭐⭐⭐ Provides a theoretically grounded new method for the brain-computer interface field, validating the practical value of brain predictive information.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Ryze: Evidence-Enriched Data Synthesis from Biomedical Papers](ryze_evidence-enriched_data_synthesis_from_biomedical_papers.md)
- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised RL Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[ACL 2026\] Dr. Assistant: Enhancing Clinical Diagnostic Inquiry via Structured Diagnostic Reasoning Data and Reinforcement Learning](dr_assistant_enhancing_clinical_diagnostic_inquiry_via_structured_diagnostic_rea.md)
- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](../../CVPR2026/medical_nlp/towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[ACL 2026\] Beyond the Leaderboard: Rethinking Medical Benchmarks for Large Language Models](beyond_the_leaderboard_rethinking_medical_benchmarks_for_large_language_models.md)

</div>

<!-- RELATED:END -->
