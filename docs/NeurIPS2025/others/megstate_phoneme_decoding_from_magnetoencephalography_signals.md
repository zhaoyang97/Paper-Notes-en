---
title: >-
  [Paper Note] MEGState: Phoneme Decoding from Magnetoencephalography Signals
description: >-
  [NeurIPS 2025][MEG] This paper proposes MEGState, an architecture combining multi-resolution convolution and sensor-wise state space models (SSMs) for decoding phonemes from magnetoencephalography (MEG) signals…
tags:
  - "NeurIPS 2025"
  - "MEG"
  - "phoneme decoding"
  - "state space model"
  - "multi-resolution convolution"
  - "brain-computer interface"
date: 2026-05-08
content_hash: 765c302858dab998
---

# MEGState: Phoneme Decoding from Magnetoencephalography Signals

**Conference**: NeurIPS 2025
**arXiv**: [2512.17978](https://arxiv.org/abs/2512.17978)  
**Code**: None  
**Area**: Brain-Computer Interface / Speech Decoding
**Keywords**: MEG, phoneme decoding, state space model, multi-resolution convolution, brain-computer interface

## TL;DR

This paper proposes MEGState, an architecture combining multi-resolution convolution and sensor-wise state space models (SSMs) for decoding phonemes from magnetoencephalography (MEG) signals, achieving substantial improvements over baseline methods on the LibriBrain dataset.

## Background & Motivation

### Limitations of Prior Work

**Background**: Decoding speech representations from brain activity is of significant importance for restoring communicative ability in individuals with paralysis or severe speech disorders. Invasive brain-computer interfaces (e.g., ECoG) have demonstrated continuous speech reconstruction with word error rates below 5%; however, their reliance on neurosurgical implantation limits scalability and clinical viability.

Magnetoencephalography (MEG), as a non-invasive alternative, offers a safe and repeatable means of detecting speech-related neural activity, yet faces three major challenges:

**Low signal-to-noise ratio**: MEG signals are extremely weak.

**High temporal resolution**: This gives rise to high-dimensional time series.

**Sparse neural representations**: Speech information is sparsely encoded in MEG signals.

**Goal**: The paper is motivated by the need to design an architecture capable of simultaneously capturing fine-grained local temporal dynamics and long-range temporal dependencies to overcome the above challenges.

## Method

### Overall Architecture

MEGState takes MEG signals $\mathbf{X} \in \mathbb{R}^{M \times T}$ ($M$ sensors, $T$ time steps) as input, sequentially passes them through a multi-resolution convolution module to extract local temporal features, then through a sensor-wise SSM to model global temporal dependencies, and finally applies average pooling followed by a fully connected layer to output phoneme classification probabilities.

### Key Designs

1. **Multi-Resolution Convolution Module**: Four parallel one-dimensional convolutional layers with kernel sizes of $f_{sample}/2$, $f_{sample}/4$, $f_{sample}/8$, and $f_{sample}/16$ (where $f_{sample}$ denotes the sampling rate). Different kernel sizes capture cortical response features at different temporal scales. The four outputs are concatenated to yield $\mathbf{H} \in \mathbb{R}^{F \times M \times T}$. **Design Motivation**: Cortical responses to different phonemes exhibit distinct characteristics at different temporal scales; the multi-resolution design enables comprehensive capture of these differences.

2. **Sensor-wise SSM**: A state space model based on the S5 variant, modeling global temporal dependencies on a per-sensor basis. The module comprises $L$ hierarchically stacked blocks, each consisting of an SSM layer + LayerNorm + residual connection + FFN. S5 employs the HiPPO-N matrix as the state matrix $\mathbf{A}$, achieving efficient recurrence via diagonalization and zero-order hold discretization. **Design Motivation**: SSMs are well-suited for modeling long-range dependencies in continuous signals; sensor-wise processing preserves spatial information.

3. **Training Data Sampling Strategy**: A novel combination of smoothing and data augmentation is introduced to improve SNR and alleviate phoneme label imbalance. At each step, two phoneme labels $y_1, y_2$ are uniformly sampled; for each label, $N'$ samples are drawn and averaged to construct class-conditional prototypes (for denoising); the two prototypes and their labels are then mixed using a mixup coefficient $\alpha$. The prototype size is $N'=100$ and $\alpha=0.5$.

### Loss & Training

- Cross-entropy loss is used for training.
- AdamW optimizer with $\beta_1=0.9$, $\beta_2=0.999$, learning rate $10^{-4}$.
- Batch size 32, trained for 50 epochs.
- Number of SSM blocks $L=2$.
- Class-conditional prototype averaging combined with mixup augmentation simultaneously addresses low SNR and class imbalance.

## Key Experimental Results

### Main Results

Dataset: LibriBrain, comprising MEG recordings from a single subject listening to audiobooks, with 306 sensors at 1 kHz sampling rate, downsampled to 250 Hz after preprocessing, totaling 52.32 hours of data across 39 ARPAbet phoneme labels.

| Model | Accuracy↑ | Cohen's Kappa↑ | Macro-F1↑ | Leaderboard F1↑ |
|-------|-----------|---------------|-----------|-----------------|
| Baseline | 38.80±2.40 | 45.71±0.74 | 34.82±1.92 | — |
| w/o Multi-Resol Conv | 40.25±3.15 | 37.90±2.83 | 34.77±1.83 | — |
| w/o Sensor-wise SSM | 40.37±3.20 | 49.60±6.25 | 37.18±4.44 | — |
| **MEGState (Full)** | **45.53±1.88** | **54.19±2.42** | **41.11±2.20** | **55.74 (68.41)** |

### Ablation Study

| Configuration | Accuracy↑ | Kappa↑ | Macro-F1↑ | Notes |
|---------------|-----------|--------|-----------|-------|
| Full MEGState | **45.53** | **54.19** | **41.11** | Best with both modules |
| w/o Multi-Resol Conv | 40.25 (-5.28) | 37.90 (-16.29) | 34.77 (-6.34) | Largest Kappa drop |
| w/o Sensor-wise SSM | 40.37 (-5.16) | 49.60 (-4.59) | 37.18 (-3.93) | More moderate impact |

### Key Findings

- MEGState significantly outperforms the baseline on all metrics (p<0.05): Accuracy +6.73, Kappa +8.48, Macro-F1 +6.29.
- Both modules are indispensable: removing the multi-resolution convolution has a larger impact (Kappa drops sharply from 54.19 to 37.90).
- Phoneme-level analysis shows MEGState outperforms the baseline on 19 of 39 phonemes, with 10 reaching statistical significance.
- A 5-model ensemble on the leaderboard further improves Macro-F1 to 68.41%.
- Phonemes with more training samples generally yield better decoding performance, though some low-frequency phonemes are also decoded effectively.

## Highlights & Insights

- **Effective training data sampling strategy**: Class-conditional prototype averaging for denoising combined with mixup for class imbalance addresses two problems simultaneously, serving as a practical technique for low-SNR neural signal decoding.
- The design rationale for multi-resolution convolution is well-motivated: different phonemes exhibit distinct cortical response patterns at different temporal scales, consistent with neuroscientific knowledge of speech processing.
- SSMs are more suitable than Transformers for high-sampling-rate continuous signals (e.g., 250 Hz MEG) due to their superior scalability with respect to sequence length.
- The substantial leaderboard gain from 5-model ensembling (55.74 → 68.41) indicates considerable complementarity among individual models.

## Limitations & Future Work

- Validation is limited to a single subject; cross-subject generalization remains unexplored.
- The paper is relatively brief (workshop paper format), with concise method descriptions and limited in-depth analysis.
- Evaluation is restricted to phoneme classification; end-to-end speech reconstruction and word-level decoding are not attempted.
- The multi-resolution convolution kernel sizes are hard-coupled to the sampling rate, requiring adjustment for different sampling rates.
- No direct comparison is made with other MEG decoding methods (e.g., MAD-MEG, NeuSpeech).
- SSM variants such as Mamba may yield further improvements.
- While the 38–46% accuracy substantially exceeds the baseline, it remains far from the level required for practical brain-computer interfaces.

## Related Work & Insights

- Invasive BCI methods (ECoG-DCNet, Cortical-SSM) establish upper bounds for speech decoding performance.
- SSM variants including S4, S5, and Mamba provide a theoretical foundation for modeling continuous neural signals.
- Multi-resolution processing is a common strategy in both speech recognition (e.g., the wav2vec family) and EEG/MEG analysis.
- The training strategy of prototype averaging combined with mixup is generalizable to other low-SNR biosignal decoding tasks.

## Rating
- Novelty: ⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CAT-Net: A Cross-Attention Tone Network for Cross-Subject EEG-EMG Fusion Tone Decoding](../../AAAI2026/others/cat-net_a_cross-attention_tone_network_for_cross-subject_eeg-emg_fusion_tone_dec.md)
- [\[NeurIPS 2025\] Are Pixel-Wise Metrics Reliable for Sparse-View Computed Tomography Reconstruction?](are_pixel-wise_metrics_reliable_for_sparse-view_computed_tomography_reconstructi.md)
- [\[NeurIPS 2025\] An Empirical Investigation of Neural ODEs and Symbolic Regression for Dynamical Systems](an_empirical_investigation_of_neural_odes_and_symbolic_regression_for_dynamical_.md)
- [\[NeurIPS 2025\] Continuous Thought Machines](continuous_thought_machines.md)
- [\[NeurIPS 2025\] Hessian-guided Perturbed Wasserstein Gradient Flows for Escaping Saddle Points](hessian-guided_perturbed_wasserstein_gradient_flows_for_escaping_saddle_points.md)

</div>

<!-- RELATED:END -->
