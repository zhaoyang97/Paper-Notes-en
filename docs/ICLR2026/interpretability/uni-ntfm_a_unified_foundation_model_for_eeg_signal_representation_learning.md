---
title: >-
  [Paper Note] Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning
description: >-
  [ICLR 2026][EEG] Uni-NTFM is grounded in first-principles neuroscience. It introduces a Heterogeneous Feature Projection Module (HFPM) for decoupled time-frequency encoding, a hierarchical Topological Embedding (TE) for unifying heterogeneous electrode configurations, and an MoE Transformer for functional modularity and sparse coding. A 1.9B-parameter model is pretrained on approximately 28,000 hours of EEG data, achieving state-of-the-art performance on 9 downstream tasks under both linear probing and fine-tuning protocols.
tags:
  - ICLR 2026
  - EEG
  - Foundation Model
  - Neural Topology
  - Mixture of Experts
  - Self-Supervised Learning
  - Brain-Computer Interface
date: 2026-05-08
content_hash: 53e95e747d57ba53
---

# Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning

**Conference**: ICLR 2026
**arXiv**: [2509.24222](https://arxiv.org/abs/2509.24222)
**Code**: [https://anonymous.4open.science/r/Uni-NTFM-0924](https://anonymous.4open.science/r/Uni-NTFM-0924)
**Area**: Interpretability
**Keywords**: EEG, Foundation Model, Neural Topology, Mixture of Experts, Self-Supervised Learning, Brain-Computer Interface

## TL;DR

Uni-NTFM is grounded in first-principles neuroscience. It introduces a Heterogeneous Feature Projection Module (HFPM) for decoupled time-frequency encoding, a hierarchical Topological Embedding (TE) for unifying heterogeneous electrode configurations, and an MoE Transformer for functional modularity and sparse coding. A 1.9B-parameter model is pretrained on approximately 28,000 hours of EEG data, achieving state-of-the-art performance on 9 downstream tasks under both linear probing and fine-tuning protocols.

## Background & Motivation

**State of the Field**: EEG foundation models have emerged as an active research direction over the past two years. Models such as LaBraM, EEGPT, and CBraMod have explored transferring pretraining paradigms from NLP and CV to the EEG domain, leveraging large-scale self-supervised learning to acquire generalizable representations.

**Limitations of Prior Work**: Existing EEG foundation models suffer from three fundamental architectural deficiencies:

1. **Coupled time-frequency encoding**: Standard architectures treat signals as a single homogeneous stream, ignoring the brain's use of decoupled mechanisms for transient temporal events (e.g., spike waves, K-complexes) and stationary spectral rhythms (e.g., alpha, beta bands). Forcing both into a unified encoding causes waveform morphology and spectral structure to interfere with each other.
2. **Non-unified electrode configurations**: Different datasets employ varying sensor layouts (e.g., clinical 19-channel 10-20 systems vs. high-density 64-channel 10-10 systems). Standard Transformer 1D positional encodings treat electrodes as simple sequences, discarding cortical surface geometry and impeding cross-dataset transfer.
3. **Lack of functional modularity**: Biological neural networks achieve efficient processing through functional specialization and sparse coding (e.g., V1 for vision, Broca's area for language), whereas standard dense Transformers activate all parameters for every input, leading to task interference when processing highly heterogeneous EEG signals.

**Core Idea**: Model architecture should be aligned with biological neural mechanisms—decoupled encoding, topology-awareness, and modular sparse processing—rather than naively porting CV/NLP architectures.

## Method

### Overall Architecture

Input EEG data undergoes probabilistic augmentation (Gaussian noise, channel dropout, random time shifts) and is reshaped into $X \in \mathbb{R}^{B \times R \times E \times T}$, where $R$ denotes the number of predefined brain regions, $E$ the maximum number of electrodes per region, and $T$ the temporal length. The signal is then processed sequentially through: (1) HFPM, which decouples the full time series of each electrode into temporal, spectral, and raw feature streams; (2) DCM dual-domain cross-attention for fusing time-frequency representations; (3) hierarchical topological embedding for injecting spatial priors; and (4) an MoE Transformer for high-level semantic encoding via sparse expert routing. The pretraining objective is dual-domain masked autoencoding reconstruction.

### Key Designs

1. **Heterogeneous Feature Projection Module (HFPM) + Dual-domain Cross-attention Module (DCM)**

    - **Temporal path**: The full time series $x_i \in \mathbb{R}^T$ of each electrode is treated as a single holistic patch (rather than split into temporal segments). A multi-layer 1D convolutional encoder $\Phi_T$ captures local waveform structure and non-stationary events, producing $h_{i,T} \in \mathbb{R}^D$.
    - **Spectral path**: A DFT is applied to compute the power spectral density, parameterized as a mean power vector $P_b \in \mathbb{R}^{N_b}$ over $N_b$ canonical frequency bands, which is then projected by an MLP $\Phi_F$ to yield $h_{i,F} \in \mathbb{R}^D$.
    - **Raw path**: A standard linear projection preserves complete signal information as a ground truth reference $H_R$ for self-supervised reconstruction.
    - **DCM fusion**: Temporal features serve as queries attending to spectral keys/values, with the symmetric operation applied in reverse: $H'_T = \text{LN}(H_T + \text{CrossAttn}(Q\!=\!H_T, K\!=\!H_F, V\!=\!H_F))$. The final fused representation $H_{\text{fused}}$ is generated by concatenating both directions and passing through an FFN.
    - **Design Motivation**: Unlike the mainstream approach of slicing along the time axis, whole-channel encoding preserves the signal's continuity and multi-scale time-frequency characteristics.

2. **Hierarchical Topological Embedding (TE)**

    - The spatial identity of each electrode is decomposed into three levels of neuroanatomical semantics: (a) **Region embedding** $E_{\text{region}} \in \mathbb{R}^{5 \times D}$, corresponding to five functional areas—Frontal (executive function), Central (sensorimotor), Temporal (auditory/memory), Parietal (spatial attention), and Occipital (visual processing); (b) **Intra-region embedding** $E_{\text{intra}}$, encoding the relative spatial arrangement of electrodes within a region (e.g., C3 and C1 are spatially adjacent on the motor cortex); (c) **Global absolute embedding** $E_{\text{abs}}$, assigning a unique global identifier to each IFCN-standard electrode.
    - The final spatial representation is the superposition of all embeddings: $H_{\text{in}}^{(i)} = H_{\text{fused}}^{(i)} + H_R^{(i)} + E_{\text{region}}[x_{\text{region}}^{(i)}] + E_{\text{intra}}[x_{\text{intra}}^{(i)}] + E_{\text{abs}}[x_{\text{abs}}^{(i)}]$
    - **Design Motivation**: Region-level embeddings enable the model to generalize by brain area rather than by channel index, addressing the fundamental bottleneck of cross-electrode-configuration transfer.

3. **MoE Transformer and Sparse Routing**

    - The dense FFN in standard Transformer layers is replaced by sparsely activated MoE layers. A gating network $g(h_i) = h_i W_g$ computes routing logits for each token over $N_e$ experts, and a Top-k gate selects a subset of experts.
    - Self-attention layers employ RoPE to encode the relative spatial ordering of electrodes.
    - An auxiliary load-balancing loss $L_{\text{aux}} = \alpha \cdot N_e \sum_j f_j \cdot \bar{p}_j$ prevents routing collapse.
    - **Design Motivation**: Distinct EEG signal patterns (motor rhythms, pathological discharges, cognitive events) are handled by specialized expert sub-networks, mitigating task interference while keeping computation tractable despite the large total parameter count.

### Loss & Training

- **Pretraining objective**: Dual-domain masked autoencoding. A proportion of tokens are randomly masked and replaced with a learnable embedding $e_{[\text{MASK}]}$, with reconstruction performed in both temporal and spectral domains: $L_{\text{total}} = \lambda_T L_{\text{time}} + \lambda_F L_{\text{freq}} + \lambda_{\text{aux}} L_{\text{aux}}$
- **Pretraining data**: 9 public datasets, 17,000+ subjects, approximately 28,000 hours of recordings, covering resting state, emotion induction, cognitive classification, BCI paradigms, and clinical recordings.
- **Model scales**: Four variants—Tiny (57M), Small (427M), Middle (912M), and Large (1.9B).
- **Training environment**: NVIDIA A100-80G GPUs, PyTorch 2.3.1 + CUDA 11.8.

## Key Experimental Results

### Main Results: Fine-tuning on 9 Downstream Tasks

Uni-NTFM is comprehensively compared against 7 traditional task-specific methods and 4 pretrained foundation models (LaBraM, CBraMod, BIOT, CSBrain, etc.) across 9 downstream EEG tasks. Representative fine-tuning results for Uni-NTFM$_\text{large}$ are summarized below:

| Task (Dataset) | #Classes | Metric | Uni-NTFM | Best Baseline | Baseline |
|---|---|---|---|---|---|
| Abnormal Detection (TUAB) | 2 | Bal. Acc. | **81.97** | 81.72 | CSBrain |
| Event Classification (TUEV) | 6 | Bal. Acc. | **69.91** | 69.03 | CSBrain |
| Emotion Recognition (SEED) | 3 | Bal. Acc. | **73.37** | 73.18 | LaBraM |
| Brain Age Classification (TDBrain) | 2 | Bal. Acc. | **83.69** | 82.81 | CBraMod |
| Dementia Detection (ADFTD) | 3 | Bal. Acc. | 76.61 | **77.63** | BIOT |
| Motor Imagery (BCIC-IV-2a) | 4 | Bal. Acc. | 56.08 | **56.57** | CSBrain |
| Cognitive Workload (Workload) | 2 | Bal. Acc. | 66.44 | **66.55** | BIOT |
| Sleep Staging (HMC) | 5 | Kappa | **68.32** | 68.18 | CSBrain |
| EEG Slowing Detection (TUSL) | 3 | Bal. Acc. | 78.44 | **85.71** | CSBrain |

Under the linear probing protocol (frozen encoder with a linear classification head), Uni-NTFM$_\text{large}$ achieves a Bal. Acc. of 78.44 on TUAB (surpassing the fine-tuned results of most traditional methods), a Cohen's Kappa of 66.11 on TUEV, and a Bal. Acc. of 73.14 on SEED, demonstrating the high quality of the pretrained representations.

### Ablation Study (TUAB + TUEV)

Components are incrementally added to Uni-NTFM$_\text{tiny}$ to quantify each module's contribution:

| ID | HFPM | DCM | TE | MoE | TUAB AUROC | TUEV Weighted F1 |
|---|---|---|---|---|---|---|
| A1 (baseline) | ✗ | ✗ | ✗ | ✗ | 71.16 | 73.66 |
| A2 | ✓ | ✗ | ✗ | ✗ | 78.05 (+6.89) | 76.69 |
| A3 | ✓ | ✓ | ✗ | ✗ | 79.76 | 78.72 |
| A6 | ✗ | ✗ | ✓ | ✓ | 80.03 | 78.94 |
| A7 | ✓ | ✗ | ✓ | ✓ | 81.10 | 80.81 |
| A9 | ✓ | ✓ | ✓ | ✗ | 81.40 | 79.39 |
| A10 (full) | ✓ | ✓ | ✓ | ✓ | **83.25** (+12.09) | **81.74** (+8.08) |

### Key Findings

- **HFPM is the single most impactful component**: Adding HFPM alone improves TUAB AUROC from 71.16 to 78.05 (+6.89), validating the central value of decoupled time-frequency encoding.
- **Synergistic gains from MoE**: MoE in isolation yields limited gains (+7.46), but its contribution is substantially amplified in combination with other components (full model vs. A9 without MoE: 83.25 vs. 81.40), suggesting that MoE requires high-quality multi-domain features to realize its modular routing advantage.
- **Scaling laws hold**: Performance increases monotonically across four model scales from 57M to 1.9B; under linear probing, TUAB Bal. Acc. improves from 71.36 to 78.44 and TUEV Kappa from 60.11 to 66.11.
- **Linear probing surpasses traditional fine-tuning**: Uni-NTFM$_\text{large}$ with a frozen encoder and linear head (TUAB Bal. Acc. 78.44) outperforms the fine-tuned results of most traditional methods (SPaRCNet 77.49, EEGNet 77.12), confirming the quality of the pretrained representations.
- **Underperformance on TUSL**: On the EEG slowing detection task, Uni-NTFM (78.44) falls considerably short of CSBrain (85.71), indicating that general-purpose representations may still be outpaced by task-specific designs on certain specialized tasks.

## Highlights & Insights

- **The counter-intuitive "whole-channel encoding" design**: Rather than slicing patches along the time axis, each electrode's complete time series is treated as a single holistic token—contrary to the patch-based paradigm in NLP/CV, but better suited to EEG, as it preserves signal continuity and multi-scale time-frequency structure.
- **Hierarchical topological embedding elegantly resolves cross-configuration transfer**: The three-level embedding hierarchy (region → intra-region → global) enables the model to generalize by brain functional area rather than memorizing channel indices, allowing seamless joint training and transfer between 19-channel and 64-channel data.
- **Ablations reveal a clear component hierarchy**: HFPM > DCM ≈ TE > MoE in terms of individual contribution, yet the synergistic effect of the full combination substantially exceeds simple additive gains.
- **A 1.9B-parameter EEG model**: The MoE architecture makes this scale feasible for the first time in the EEG domain, with sparse activation keeping computation tractable.

## Limitations & Future Work

- **Failure to reach SOTA on TUSL and BCIC-IV-2a**: Performance lags behind CSBrain by 7.27 and 0.49 points respectively, suggesting that general foundation models still have room for improvement on tasks requiring fine temporal resolution or subject-specific adaptation.
- **CSBrain results are not reproducible**: The paper notes that CSBrain's code is not publicly available and comparison figures are taken directly from its paper, raising concerns about fairness.
- **Only linear probing and fine-tuning are evaluated**: More nuanced transfer learning assessments such as few-shot and zero-shot evaluation are absent.
- **Insufficient analysis of expert specialization in MoE**: The paper does not demonstrate whether different experts have genuinely learned functionally distinct roles (e.g., one expert specializing in motor rhythms, another in pathological discharges), and routing visualizations are lacking.
- **Pretraining data skewed toward clinical recordings**: The large proportion of clinical data (e.g., TUEG) may introduce representational bias against cognitive task paradigms.
- **Deployment costs unaddressed**: Inference latency and memory requirements of the 1.9B-parameter model on practical BCI devices are not discussed.

## Related Work & Insights

- **vs. LaBraM (Jiang et al., 2024)**: LaBraM applies patch-based MAE pretraining to EEG without time-frequency decoupling or topological embedding. Uni-NTFM outperforms it across tasks such as TUAB and SEED (TUAB Bal. Acc. 81.97 vs. 81.40; SEED 73.37 vs. 73.18).
- **vs. CBraMod (Wang et al., 2024)**: CBraMod models spatial interactions via criss-cross attention but lacks an explicit topological structure. Uni-NTFM shows a clear advantage on TDBrain (83.69 vs. 82.81).
- **vs. BIOT (Yang et al., 2023)**: BIOT is designed for cross-dataset learning but uses a dense architecture without MoE. It slightly outperforms Uni-NTFM on ADFTD and Workload, suggesting that these tasks may be better suited to dense architectures.
- **vs. CSBrain (Zhou et al., 2025)**: CSBrain leads substantially on TUSL (85.71 vs. 78.44), but its code is not publicly available and results on TDBrain are not reported, limiting the fairness of the comparison.
- **Broader inspiration**: The "biologically aligned architecture" design paradigm is generalizable to other neural signals such as MEG and fNIRS; the hierarchical spatial embedding approach has potential implications for federated learning in multi-center clinical EEG settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Each of the three modules is well-motivated; whole-channel encoding and hierarchical topological embedding are genuinely novel, though replacing FFN with MoE is a standard operation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive coverage across 9 downstream tasks, 4 model scales, full ablations, and comparisons against 7 traditional methods and 4 foundation models.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear articulation of the correspondence between neuroscientific principles and architectural design choices; notation is rigorous.
- **Value**: ⭐⭐⭐⭐ Establishes the design principle that EEG foundation models should be aligned with neural mechanisms, and the 1.9B-parameter scale marks the entry of EEG foundation models into the large-model era.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ICLR 2026\] Information Shapes Koopman Representation](information_shapes_koopman_representation.md)
- [\[ICLR 2026\] NIMO: a Nonlinear Interpretable MOdel](nimo_a_nonlinear_interpretable_model.md)

<!-- RELATED:END -->
