---
title: >-
  [Paper Note] S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection
description: >-
  [NeurIPS 2025][Model Compression][Spiking Neural Networks] This paper proposes S2M-Former, a spiking-driven symmetric mixing Branchformer framework that achieves SOTA-level accuracy on EEG-based auditory attention detection with only 0.06M parameters, via complementary learning across spatial-frequency dual branches and lightweight 1D token representations, while reducing energy consumption to 1/5.8 of dual-branch ANN counterparts.
tags:
  - NeurIPS 2025
  - Model Compression
  - Spiking Neural Networks
  - Auditory Attention Detection
  - Symmetric Mixing Architecture
  - Energy-Efficient Computing
  - EEG
date: 2026-05-08
content_hash: 8689ebc71278a11b
---

# S2M-Former: Spiking Symmetric Mixing Branchformer for Brain Auditory Attention Detection

**Conference**: NeurIPS 2025
**arXiv**: [2508.05164](https://arxiv.org/abs/2508.05164)
**Code**: [GitHub](https://github.com/JackieWang9811/S2M-Former)
**Area**: Model Compression
**Keywords**: Spiking Neural Networks, Auditory Attention Detection, Symmetric Mixing Architecture, Energy-Efficient Computing, EEG

## TL;DR

This paper proposes S2M-Former, a spiking-driven symmetric mixing Branchformer framework that achieves SOTA-level accuracy on EEG-based auditory attention detection with only 0.06M parameters, via complementary learning across spatial-frequency dual branches and lightweight 1D token representations, while reducing energy consumption to 1/5.8 of dual-branch ANN counterparts.

## Background & Motivation

Auditory Attention Detection (AAD) aims to decode from EEG signals which speaker a listener attends to in complex acoustic environments, which is critical for developing neurally-driven hearing aids. Existing methods face the following core challenges:

**Isolated Learning Paradigm**: Recent dual-branch networks (e.g., DBPNet, M-DBPNet) employ multiple EEG features but fuse branches merely via simple concatenation or summation, overlooking the potential for complementary inter-branch learning.

**Excessive Computational Cost**: To model EEG-specific properties such as topological structure, these methods rely on high-cost operations like 3D convolutions, resulting in parameter counts of 0.88M–1.32M, which are unsuitable for deployment on low-power wearable devices.

**Energy Efficiency Bottleneck**: Hearing aids and BCI systems impose strict constraints on battery life, latency, and computational resources; the high energy consumption of existing ANN models is difficult to meet in practice.

The core idea of S2M-Former is to leverage the inherently low-power nature of spiking neural networks (SNNs) — replacing MAC operations with sparse AC operations — combined with a symmetric dual-branch design for complementary spatial-frequency feature learning, while replacing 3D operations with 1D token sequences to substantially reduce parameter count.

## Method

### Overall Architecture

S2M-Former consists of three components: (1) branch-specific spiking encoders (SBE/FBE) that extract spatial and frequency domain features respectively; (2) the Spiking Symmetric Mixing (S2M) module for complementary inter-branch fusion; and (3) a classification head for output prediction. The input EEG is first processed by CSP to extract spatial features $E_S \in \mathbb{R}^{C \times T}$ and by DE to extract frequency features $E_F \in \mathbb{R}^{5 \times H \times W}$, which are then expanded across time steps and spiking-encoded before further processing.

### Key Designs

1. **Channel-Parametric LIF (CPLIF) Neuron**: Building upon the standard Parametric LIF, CPLIF assigns independent membrane time constants $\tau_l[c]$ and biases $\beta[c]$ to each channel, enabling channel-level adaptive temporal modeling. The membrane potential update is formulated as:

$$H[t,c,n] = V[t-1,c,n] + \frac{1}{\tau_l[c]}(X[t,c,n] - (V[t-1,c,n] - V_{reset})) + \beta[c]$$

**Design Motivation**: Standard LIF shares a single time constant across all channels, making it unable to capture the heterogeneous temporal dynamics across different frequency bands and channels. CPLIF provides finer-grained control over spiking activation.

2. **Dual-Branch Spiking Encoders (SBE + FBE)**:

    - **SBE (Spatial Branch)**: Extracts temporal dependencies via cascaded temporal convolutions (kernel sizes 8→16), followed by dual-path spatial convolutions (kernel $C \times 1$) to aggregate cross-channel interactions with residual summation. Output dimension: $\mathbb{R}^{T_S \times D \times T}$.
    - **FBE (Frequency Branch)**: Processes 2D brain topological maps using three layers of dilated convolutions (dilation=2, kernel 3×3) with channel progression $D \to 4D \to 2D \to D$, accompanied by max pooling and 1×1 residual convolutions. The key innovation lies in flattening traditional 3D operations into 1D token sequences, reducing parameter count by 14.7×.

3. **S2M Module (Four Collaborative Sub-modules)**:

    - **SCSA (Spiking Channel Self-Attention)**: Reformulates the standard SSA attention matrix from $N \times N$ to $D \times D$ along the channel dimension, reducing complexity from $O(N^2D)$ to $O(ND^2)$. The spatial branch captures electrode correlations while the frequency branch captures multi-band relationships. QKV projections are implemented via 1D depthwise separable convolutions (kernel=3).
    - **SMSC (Spiking Multi-Scale Separable Convolution)**: Three parallel depthwise convolution paths (kernels 1/3/5) capture multi-scale local patterns. A pointwise convolution first expands channels to 3D; the three paths are summed and followed by channel shuffle to promote cross-scale information exchange without additional parameters.
    - **SGCM (Spiking Gated Channel Mixer)**: Concatenates tokens from the spatial and frequency branches, projects them linearly to 2D channels, and splits the result into query and key components. A channel attention vector $A_c \in \mathbb{R}^{T_S \times (N_S+N_F) \times 1}$ is generated by summing over the query, which then applies a channel-wise mask to the key for adaptive fusion.
    - **MPTM (Membrane Potential-aware Token Mixer)**: Aggregates global information via GAP, mixes global summaries from the fused and original branches at ratio $\alpha=0.5$ to construct a guidance representation $R$, and performs cross-branch fusion with residual preservation via spiking element-wise modulation: $F = \mathcal{SN}(X_G) \odot R + X_G$.

### Loss & Training

Binary cross-entropy loss is used for two-class classification (left/right auditory attention). A unified preprocessing pipeline (re-referencing, bandpass filtering, downsampling to 128 Hz) is applied across all datasets, with feature extraction performed separately on training/validation/test splits to prevent information leakage. EEG segments are obtained using a sliding window with 50% overlap. The model maintains a fixed size (0.06M) across all decision window lengths, in contrast to M-DBPNet whose parameter count varies with window size (1.32M/1.00M/0.88M).

## Key Experimental Results

### Main Results (Within-trial Setting)

| Dataset | Metric (2s Acc%) | S2M-Former | DBPNet | M-DBPNet | DARNet | Parameter Comparison |
|---------|-----------------|------------|--------|----------|--------|---------------------|
| KUL | Accuracy ± SD | 93.71 ± 8.14 | 93.66 ± 7.88 | 93.75 ± 6.34 | 92.81 ± 9.45 | **0.06M vs 0.88M** |
| DTU | Accuracy ± SD | **85.28 ± 6.01** | 83.93 ± 5.17 | 82.56 ± 8.01 | 81.30 ± 5.76 | 14.7× smaller |
| AV-GC | Accuracy ± SD | **91.83 ± 6.66** | 90.78 ± 4.91 | 87.04 ± 7.76 | 89.17 ± 6.94 | 5.8× lower energy |

### Ablation Study (DTU Within-trial / Cross-trial)

| Configuration | DTU Within-2s | DTU Cross-2s | Notes |
|---------------|--------------|--------------|-------|
| S2M-Former (Full) | **85.28** | **76.74** | All components |
| SM-Former (ANN replacement) | 80.94 (−4.34) | 73.49 (−3.25) | SNN→ANN, significant accuracy drop |
| CPLIF → LIF | 84.13 (−1.15) | 75.67 (−1.07) | Channel-wise parameterization is effective |
| w/o SGCM+MPTM | 82.98 (−2.30) | 74.81 (−1.93) | Complementary learning modules are critical |
| Spatial branch only | 82.28 (−3.00) | 73.58 (−3.16) | Dual-branch outperforms single-branch |
| Frequency branch only | 70.48 (−14.80) | 70.11 (−6.63) | Spatial features are more informative |

### Energy Efficiency Analysis

| Model | SNN | Params (M) | FLOPs (G) | Energy (mJ) |
|-------|-----|-----------|-----------|-------------|
| DBPNet | ✗ | 0.88 | 0.0984 | 0.4526 |
| M-DBPNet | ✗ | 1.32 | 0.1068 | 0.4913 |
| SM-Former (ANN) | ✗ | 0.06 | 0.0243 | 0.1116 |
| **S2M-Former** | ✓ | **0.06** | **0.0112** | **0.0779** |

### Key Findings

- S2M-Former achieves the highest accuracy in 11 out of 18 evaluation conditions (61.1%) with only 0.06M parameters, with a Top-3 coverage rate of 83.33%.
- Energy consumption is only 0.0779 mJ, which is 5.8× lower than DBPNet (0.4526 mJ) and 6.3× lower than M-DBPNet (0.4913 mJ).
- In cross-subject (LOSO) evaluation, S2M-Former achieves the best performance on both KUL (75.75%) and DTU (59.75%).
- The SNN version consistently outperforms the ANN counterpart (SM-Former), demonstrating the effectiveness of spiking-driven mechanisms for EEG representation learning.
- The frequency branch alone (DE features + FBE) already surpasses three SNN baselines: QKFormer, SDT, and Spikformer.

## Highlights & Insights

1. **Elegance of the Symmetric Design**: The spatial and frequency branches employ mirrored module structures, naturally promoting complementary learning without requiring manually designed complex fusion strategies.
2. **1D Token as a Substitute for 3D Operations**: This lightweight strategy is the key to dramatically reducing parameter count while maintaining or even improving performance.
3. **Practical Value of SNNs**: On EEG signals, which inherently exhibit temporal spiking characteristics, SNNs not only deliver energy efficiency advantages but also provide superior accuracy (+4.34% vs. ANN) — a result that departs from the common perception in many domains that SNNs underperform ANNs in accuracy.

## Limitations & Future Work

- Large standard deviations (>18%) are observed under cross-trial settings on KUL and AV-GC datasets, with some subjects performing below chance level (50%), leaving room for improved generalization.
- Evaluation is limited to binary classification (left/right attention) and has not been extended to multi-speaker scenarios.
- Deployment on real neuromorphic hardware (e.g., Loihi, SpiNNaker) has not been validated; energy estimates are based on theoretical calculations.
- Performance gaps relative to some baselines widen at very short decision windows (0.1s).

## Related Work & Insights

- Comparisons with DBPNet/M-DBPNet demonstrate that high parameter counts do not equate to high performance; lightweight design combined with well-motivated architectural innovations can achieve disproportionate gains.
- The channel-level parameterization concept in CPLIF is generalizable to other SNN architectures, offering a new direction for fine-grained temporal modeling.
- The symmetric mixing architecture paradigm provides a valuable reference for multi-modal fusion tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First application of an SNN-based symmetric mixing framework to AAD; CPLIF and S2M module designs are novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three datasets, three evaluation settings, complete ablation and energy consumption analysis
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure, well-presented equations
- **Value**: ⭐⭐⭐⭐⭐ Offers a highly promising solution for low-power BCI and hearing aid devices

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Spiking Brain Compression: Post-Training Second-Order Compression for Spiking Neural Networks](spiking_brain_compression_post-training_second-order_compression_for_spiking_neu.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[NeurIPS 2025\] AI-Generated Video Detection via Perceptual Straightening](ai-generated_video_detection_via_perceptual_straightening.md)
- [\[NeurIPS 2025\] BaRISTA: Brain-Scale Informed Spatiotemporal Representation of Human Intracranial EEG](barista_brain_scale_informed_spatiotemporal_representation_of_human_intracranial.md)
- [\[NeurIPS 2025\] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning](dragon_guard_llm_unlearning_in_context_via_negative_detection_and_reasoning.md)

<!-- RELATED:END -->
