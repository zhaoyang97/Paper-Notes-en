---
title: >-
  [Paper Note] Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series
description: >-
  [ICLR 2026][Medical Imaging][Medical Time Series] This paper proposes the TeCh framework, whose core contribution is the CoTAR (Core Token Aggregation-Redistribution) module…
tags:
  - "ICLR 2026"
  - "Medical Imaging"
  - "Medical Time Series"
  - "Transformer"
  - "Channel Dependency"
  - "Core Token"
  - "Linear Complexity"
date: 2026-05-08
content_hash: d84dc5f6adc428dc
---

# Decentralized Attention Fails Centralized Signals: Rethinking Transformers for Medical Time Series

**Conference**: ICLR 2026
**arXiv**: [2602.18473](https://arxiv.org/abs/2602.18473)  
**Code**: [https://github.com/Levi-Ackman/TeCh](https://github.com/Levi-Ackman/TeCh)  
**Area**: Medical Imaging
**Keywords**: Medical Time Series, Transformer, Channel Dependency, Core Token, Linear Complexity

## TL;DR
This paper proposes the TeCh framework, whose core contribution is the CoTAR (Core Token Aggregation-Redistribution) module, which replaces standard attention in Transformers to model channel dependencies in medical time series. By introducing a global "core token" as a proxy — first aggregating information from all channels and then redistributing it back — the computational complexity is reduced from $O(n^2)$ to $O(n)$. On the APAVA dataset, TeCh achieves 86.86% accuracy (surpassing Medformer by 12.13%) while consuming only 33% of the memory and 20% of the inference time.

## Background & Motivation

**Background**: Medical time series analysis (EEG/ECG) requires modeling two critical patterns simultaneously — temporal dependencies (temporal dynamics within a single channel) and channel dependencies (interactions across multiple channels). Recent Transformers such as Medformer and PatchTST have excelled at temporal dependency modeling, but channel dependency modeling remains a significant weakness.

**Limitations of Prior Work**: Standard Transformer attention is "decentralized" — each token directly interacts with all other tokens (peer-to-peer). However, medical signals are inherently "centralized": EEG is centrally governed by thalamo-cortical circuits, and ECG is uniformly coordinated by the sinoatrial node. This structural mismatch causes attention mechanisms to dilute the dominant patterns driven by central control.

**Key Challenge**: The problem is not that attention is insufficiently powerful, but that its decentralized architecture is fundamentally mismatched with centrally organized signals. When every channel can be directly influenced by noisy channels, centrally coordinated signal patterns become obscured.

**Goal**: (a) Design a channel interaction mechanism that matches the centralized structure of physiological signals; (b) reduce computational complexity from quadratic to linear; (c) adaptively handle the varying importance of temporal and channel dependencies across different datasets.

**Key Insight**: Inspired by the star topology in distributed systems — traditional P2P communication is inefficient, and establishing a central server to aggregate and distribute information is more efficient and reliable. By analogy to medical signals, a single "core token" is introduced to proxy all inter-channel communication.

**Core Idea**: Replace peer-to-peer attention with a global core token proxy — all channels first aggregate into the core token, which then redistributes information back to each channel, simulating the signal propagation pattern of the central nervous system.

## Method

### Overall Architecture
TeCh takes medical time series input $X \in \mathbb{R}^{T \times C}$ ($T$ time steps, $C$ channels), generates Temporal embeddings and Channel embeddings via adaptive dual tokenization, and feeds them into $M$ and $N$ Transformer Encoders respectively (with attention replaced by CoTAR). The outputs of each branch are averaged along the channel dimension and summed, then projected by a linear layer to produce classification output $\hat{Y} \in \mathbb{R}^K$. $M$ and $N$ are tunable; setting either to 0 removes the corresponding branch.

### Key Designs

1. **CoTAR (Core Token Aggregation-Redistribution)**:

    - **Function**: Replaces standard attention; uses a single global core token as an intermediary to enable indirect inter-channel interaction.
    - **Mechanism**: Given input $O \in \mathbb{R}^{S \times D}$, it is first projected via MLP to $\tilde{O} \in \mathbb{R}^{S \times D_c}$; Softmax is applied along the token dimension to obtain weights $O_w$; a weighted summation yields the core token $\tilde{C_o} \in \mathbb{R}^{D_c}$ (aggregation stage). The core token is then repeated to each token position, concatenated with the original $O$, and passed through another MLP to produce $A \in \mathbb{R}^{S \times D}$ (redistribution stage). The entire process involves only matrix-vector operations, achieving **complexity $O(S)$**.
    - **Design Motivation**: The $QK^T$ operation in standard attention allows each token to directly interact with all others (decentralized), enabling noisy channels to directly corrupt others. CoTAR mediates all interactions through a core token — equivalent to a star topology — so noisy channels can only exert indirect influence, providing natural noise robustness.

2. **Adaptive Dual Tokenization**:

    - **Function**: Simultaneously extracts two types of token representations along the temporal and channel dimensions.
    - **Temporal Embedding**: Flattens $L$ time steps across all channels and embeds them, yielding $E \in \mathbb{R}^{P \times D}$ ($P = \lceil T/L \rceil$); well-suited for capturing temporal dependencies.
    - **Channel Embedding**: Embeds the complete time series of each channel as a whole, yielding $H \in \mathbb{R}^{C \times D}$; preserves the full semantic information of each channel and excels at capturing channel dependencies.
    - **Design Motivation**: Different datasets exhibit varying strengths of temporal versus channel dependencies. TDBrain is dominated by temporal dependencies (the Temporal-only branch alone achieves 93.21%), PTB is dominated by channel dependencies (the Channel-only branch achieves 85.96%), and APAVA relies heavily on both (Dual surpasses single-branch configurations by over 11%). Adjusting $M$/$N$ allows adaptive alignment with dataset characteristics.

3. **Classification Paradigm**:

    - **Function**: Fuses the representations from both branches for final classification.
    - **Mechanism**: The Temporal branch output $O_{te}$ is averaged along the token dimension to obtain $\tilde{O}_{te}$; the Channel branch is similarly processed to yield $\tilde{O}_{ch}$; their sum is linearly projected: $\hat{Y} = (\tilde{O}_{te} + \tilde{O}_{ch})W_y + b_y$.
    - **Design Motivation**: Simple additive fusion avoids introducing additional parameters and allows flexible degradation to a single branch by setting $M=0$ or $N=0$.

### Loss & Training
- A Subject-Independent protocol is adopted: training/validation/test sets are split by subject to ensure generalization to unseen patients.
- Results are reported as mean and standard deviation over 5 random seeds.
- Models are saved based on the best validation F1 score.

## Key Experimental Results

### Main Results

| Dataset | Task | TeCh Acc | Medformer Acc | Avg Gain |
|--------|------|---------|-------------|---------|
| APAVA (EEG, 2-class) | Alzheimer's Diagnosis | **86.86±1.09** | 78.74±0.64 | +9.59% |
| TDBrain (EEG, 2-class) | Parkinson's Diagnosis | **93.21±0.61** | 89.62±0.81 | +4.26% |
| ADFTD (EEG, 3-class) | Dementia Classification | **54.54±0.70** | 53.27±1.54 | ~on par |
| PTB (ECG, 2-class) | Myocardial Infarction Diagnosis | **85.96±2.52** | 83.50±2.01 | +5.92% |
| PTB-XL (ECG, 5-class) | Cardiac Disease Classification | **73.53±0.07** | 72.87±0.23 | +0.67% |
| FLAAP (HAR, 10-class) | Human Activity Recognition | **80.60±0.30** | 76.44±0.64 | +3.81% |
| UCI-HAR (HAR, 6-class) | Human Activity Recognition | **94.15±0.96** | 89.62±0.81 | +3.41% |

Efficiency comparison (APAVA, batch=128): TeCh uses only **33% of Medformer's memory** and **20% of its inference time**.

### Ablation Study

**Dual Tokenization Ablation (Table 4)**:

| Configuration | APAVA Acc | APAVA F1 | TDBrain Acc | PTB Acc |
|------|----------|---------|------------|--------|
| w/o (no tokenization) | 50.68 | 50.13 | 53.79 | 72.62 |
| Temporal only | 55.93 | 53.71 | **93.21** | 74.74 |
| Channel only | 75.68 | 73.54 | 67.58 | **85.96** |
| Dual (full) | **86.86** | **86.30** | 89.79 | 84.15 |

**CoTAR Ablation (Table 5)**:

| Configuration | APAVA Acc | APAVA F1 | TDBrain Acc | UCI-HAR Acc |
|------|----------|---------|------------|------------|
| w/o CoTAR | 83.31 | 81.99 | 92.69 | 92.40 |
| Attention replacement | 83.42 | 82.09 | 90.40 | 93.13 |
| CoTAR (full) | **86.86** | **86.30** | **93.21** | **94.15** |

### Key Findings
- **CoTAR outperforms standard attention more consistently**: CoTAR surpasses standard attention across all 5 datasets with lower variance (overall 0.86 vs. 0.96, a 10.42% reduction), demonstrating that the centralized structure is more stable.
- **Dual tokenization yields large gains on APAVA** (+31% Acc vs. Temporal only), indicating that EEG data relies on both temporal and channel patterns simultaneously, and single-branch tokenization discards critical information.
- **Different datasets favor different tokenizations**: TDBrain favors Temporal (93.21%) and PTB favors Channel (85.96%), validating the necessity of the adaptive dual-branch design.
- **Noise robustness experiment**: As Gaussian noise is progressively injected into the last channel of PTB ($\beta$ ranging from 0 to 20), the F1 of attention drops sharply while CoTAR degrades slowly — because decentralized attention allows noise to propagate directly, whereas the core token in CoTAR acts as a buffer.

## Highlights & Insights
- **A deep insight into structural mismatch**: The issue is not that attention is insufficiently powerful, but that its decentralized peer-to-peer architecture is fundamentally incompatible with physiological signals organized in a centralized manner. This observation generalizes to any signal with a centralized source (e.g., fMRI, sensor networks).
- **The "star proxy" design of CoTAR is remarkably elegant**: Using only an MLP, a Softmax-weighted summation, and a repeat-concatenation operation, it reduces complexity from quadratic to linear while simultaneously achieving substantial accuracy gains. This design principle is transferable to any scenario requiring efficient global interaction.
- **Interpretability of the core token**: t-SNE visualizations show that the core token occupies a central position in both temporal and channel spaces and is class-discriminative — it learns a representation akin to a "global physiological state summary," which aligns closely with the Global Workspace Theory of the brain and cardiac pacemaker synchronization mechanisms.

## Limitations & Future Work
- The core token dimension $D_c$ is a fixed hyperparameter that may require tuning for different datasets; no mechanism for adaptive determination of $D_c$ is provided.
- Validation is limited to classification tasks; other MedTS tasks such as forecasting and anomaly detection are not explored.
- On the ADFTD three-class task, TeCh merely matches Medformer (54.54 vs. 53.27); performance under severe class imbalance warrants further investigation.
- The $M$/$N$ values of the dual branches require manual tuning; NAS or adaptive gating mechanisms could be considered as future directions.

## Related Work & Insights
- **vs. Medformer**: Both address channel dependencies in MedTS, but Medformer retains standard attention. Replacing it with CoTAR in TeCh yields higher accuracy with substantially improved efficiency.
- **vs. iTransformer**: iTransformer introduced Channel embedding (whole-channel embedding); TeCh builds upon this by incorporating the core token proxy and the dual tokenization design.
- **vs. PatchTST**: PatchTST only performs Temporal embedding and lacks explicit inter-channel interaction modeling.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Reexamines the attention mechanism from the perspective of signal organizational structure; the insight is profound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 5 MedTS and 2 HAR datasets with comprehensive ablations, efficiency analysis, noise robustness tests, and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is precise, analogies are intuitive (decentralized vs. centralized), and the narrative is compelling.
- **Value**: ⭐⭐⭐⭐⭐ — Establishes a new paradigm for channel modeling in medical time series; CoTAR is generalizable to other centralized signals.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] RAxSS: Retrieval-Augmented Sparse Sampling for Explainable Variable-Length Medical Time Series Classification](../../NeurIPS2025/medical_imaging/raxss_retrieval-augmented_sparse_sampling_for_explainable_variable-length_medica.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](../../NeurIPS2025/medical_imaging/mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[ACL 2026\] Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](../../ACL2026/medical_imaging/detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)
- [\[ACL 2026\] Learning Dynamic Representations and Policies from Multimodal Clinical Time-Series with Informative Missingness](../../ACL2026/medical_imaging/learning_dynamic_representations_and_policies_from_multimodal_clinical_time-seri.md)
- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](../../NeurIPS2025/medical_imaging/towards_self-supervised_foundation_models_for_critical_care_time_series.md)

</div>

<!-- RELATED:END -->
