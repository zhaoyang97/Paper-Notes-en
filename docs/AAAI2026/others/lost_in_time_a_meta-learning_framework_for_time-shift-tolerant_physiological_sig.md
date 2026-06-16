---
title: >-
  [Paper Note] Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation
description: >-
  [AAAI 2026][Meta-Learning] This paper proposes ShiftSyncNet, a bi-level meta-learning optimization framework that trains a SyncNet to learn temporal offsets between training signal pairs and leverages the Fourier shift t…
tags:
  - "AAAI 2026"
  - "Meta-Learning"
  - "Time-Shift Correction"
  - "Fourier Phase Shift"
  - "Noisy Label Learning"
  - "Physiological Signal Waveform Transformation"
date: 2026-05-08
content_hash: 660d9fc7d2299a65
---

# Lost in Time? A Meta-Learning Framework for Time-Shift-Tolerant Physiological Signal Transformation

**Conference**: AAAI 2026
**arXiv**: [2511.21500](https://arxiv.org/abs/2511.21500)  
**Code**: [GitHub](https://github.com/HQ-LV/ShiftSyncNet)  
**Area**: Physiological Signal Processing / Time Series Transformation
**Keywords**: Meta-Learning, Time-Shift Correction, Fourier Phase Shift, Noisy Label Learning, Physiological Signal Waveform Transformation

## TL;DR

This paper proposes ShiftSyncNet, a bi-level meta-learning optimization framework that trains a SyncNet to learn temporal offsets between training signal pairs and leverages the Fourier shift theorem to automatically correct label alignment, achieving waveform transformation accuracy improvements of 9.4%, 6.0%, and 12.8% across three datasets respectively.

## Background & Motivation

Continuous blood pressure monitoring is clinically critical, yet invasive arterial blood pressure (ABP) measurement carries infection risks and is unsuitable for daily use. Non-invasive signals such as photoplethysmography (PPG) and ballistocardiography (BCG) serve as alternatives; deep learning can transform these signals into ABP waveforms for low-cost continuous monitoring.

However, a long-overlooked yet critical issue in multimodal physiological signal acquisition is **time shift**. Factors such as clock asynchrony, system scheduling delays, device placement differences, and firmware faults introduce unknown temporal misalignment between source and target signals. This misalignment substantially degrades waveform transformation accuracy, particularly in capturing ABP peaks—a key feature for hypertension diagnosis.

Existing mitigation strategies fall into two categories, both with notable limitations: (1) traditional signal synchronization methods (e.g., DTW, cross-correlation) rely on assumptions of waveform similarity or require manual parameter tuning, and perform poorly on cross-modal signals with large morphological differences (e.g., PPG to ABP); (2) in learning with noisy labels (LNL), sample selection strategies such as Co-teaching discard excessive useful data at high corruption rates, while semi-supervised pseudo-labeling methods produce unreliable pseudo-labels under overparameterized models—potentially conflating peaks from different temporal offsets.

The core insight is that **time-shifted labels, though misaligned, retain complete physiological information in their waveform features** and should not be simply discarded or replaced with unreliable pseudo-labels as in conventional LNL approaches. The paper's starting point is to employ a meta-learning framework that enables the network to automatically learn temporal offsets and apply phase shifts in the Fourier domain to achieve differentiable label correction.

## Method

### Overall Architecture

ShiftSyncNet adopts a bi-level optimization architecture comprising two core networks:
- **TransNet** $f_\theta$: the waveform transformation network, mapping source signals (PPG/BCG) to target signals (ABP)
- **SyncNet** $h_\alpha$: the time-shift correction meta-network, learning temporal offsets between training pairs and generating aligned supervisory signals

Training data consists of a large time-shifted training set $D' = \{(x, y')\}^N$ and a small aligned meta-dataset $D = \{(x_m, y_m)\}^M$ ($M \ll N$).

### Key Designs

1. **Bi-Level Optimization Objective**:

    - Function: jointly optimizes TransNet and SyncNet
    - Mechanism: the upper-level objective minimizes TransNet's loss $\mathcal{L}_D(\theta^*_\alpha)$ on the aligned meta-dataset $D$; the lower-level objective minimizes TransNet's loss $\mathcal{L}_{D'}(\alpha, \theta)$ on the training set $D'$ after SyncNet-based correction
    - Design Motivation: if SyncNet provides high-quality aligned labels, TransNet should achieve low loss on the clean meta-dataset, which serves as the optimization signal to guide SyncNet's parameter updates

2. **K-Step Gradient Descent Lookahead Meta-Gradient**:

    - Function: efficiently approximates meta-gradient computation in bi-level optimization
    - Mechanism: $k$-step gradient descent approximates the inner-level optimal solution, yielding a recurrent meta-gradient $\frac{\partial \mathcal{L}_D(\theta^{\tau+1})}{\partial \alpha} \approx -\eta g_{\theta^{\tau+1}} H_{\theta,\alpha}^\tau + \lambda \frac{\partial \mathcal{L}_D(\theta^\tau)}{\partial \alpha}$, where $\lambda = 1-\eta$ is a discount factor, requiring only the most recent meta-gradient to be stored
    - Design Motivation: exactly solving the inner optimization is computationally prohibitive; the $k$-step approximation maintains computational efficiency while incorporating richer historical gradient information

3. **Fourier Phase-Shift Label Correction**:

    - Function: applies the temporal offset learned by SyncNet as a frequency-domain phase shift to generate aligned supervisory signals
    - Mechanism: exploiting the Fourier shift theorem—a time-domain shift of $t_0$ is equivalent to a linear phase shift $e^{-j\omega t_0}$ in the frequency domain. Given the offset $s$ predicted by SyncNet, the misaligned label $y'$ is transformed via FFT to obtain $Y'$, a phase shift is applied as $Y_c = Y' \cdot e^{-j2\pi f s}$, and the corrected signal $y_c$ is recovered by IFFT
    - Design Motivation: direct time-domain alignment via cropping requires rounding and slicing, rendering the loss non-differentiable. Frequency-domain phase shifting embeds the offset into an exponential term, which is naturally differentiable and allows gradients to backpropagate through SyncNet

4. **Sample-Selection-Based Training Strategy**:

    - Function: exploits aligned and misaligned samples in a staged manner
    - Mechanism: during the first $e$ epochs, a warm-up phase selects only the $1-r$ fraction of samples with the lowest loss for training (inspired by Co-teaching's curriculum of learning easy examples first); in subsequent phases, each batch is split into aligned and misaligned subsets—aligned samples use their original labels $y'$, while misaligned samples use SyncNet-corrected labels $y_c$—and the two contributions are fused via a soft-weighted loss $\mathcal{L}_{D'_{soft}} = \beta \mathcal{L}_{D'_a} + (1-\beta)\mathcal{L}_{D'_s}$
    - Design Motivation: early in training, the loss distribution over simple patterns can distinguish aligned from misaligned samples; stabilizing training during warm-up before introducing SyncNet correction maximizes data utilization

### Loss & Training

- Waveform transformation loss uses MSE
- TransNet is pre-trained on the meta-dataset for initialization
- The soft-weighted loss adaptively balances contributions from aligned and corrected samples, with weight $\beta = |D'_a| / (|D'_a| + |D'_s|)$

## Key Experimental Results

### Main Results

Under the setting $S=20$, $r=0.7$ (maximum shift of 20 points, 70% of samples corrupted), compared against 10 baseline methods:

| Dataset | Metric | ShiftSyncNet | Co-teaching (2nd best) | Gain |
|---------|--------|-------------|----------------------|------|
| VitalDB | MSE↓ | **0.009** | 0.010 | 6.0% |
| MIMIC II | MSE↓ | **0.016** | 0.019 | 12.8% |
| OML | MSE↓ | **0.023** | 0.025 | 9.4% |
| VitalDB | PRD↓ | **2.097** | 2.167 | 3.2% |
| MIMIC II | PRD↓ | **2.755** | 2.945 | 6.5% |
| OML | PRD↓ | **3.572** | 3.760 | 5.0% |

Compared to MLC, the second-best label correction method, MSE is reduced by 30.4%, 19.6%, and 32.2% respectively.

### Ablation Study

| Configuration | VitalDB ($r=0.7$) | OML ($r=0.7$) | Notes |
|---------------|-------------------|---------------|-------|
| w/o SL (no soft loss) | 0.0096 | 0.0227 | ignores direct supervision from aligned samples |
| w/o WU (no warm-up) | 0.0096 | 0.0235 | lacks early-stage stable training |
| Full model | **0.0095** | **0.0226** | soft loss + warm-up achieves best performance |

### Downstream Blood Pressure Prediction Task

| Method | VitalDB SBP MAE | VitalDB DBP MAE | MIMIC II SBP MAE | MIMIC II DBP MAE |
|--------|----------------|----------------|------------------|------------------|
| InceptionTime | 12.41 | 5.50 | 16.82 | 7.13 |
| Co-teaching | 3.22 | 2.44 | 5.82 | 2.99 |
| ShiftSyncNet | **2.43** | **1.49** | **4.83** | **2.36** |

ShiftSyncNet meets the AAMI standard (MAE < 5 mmHg), reducing SBP/DBP MAE by up to 80%/72% (VitalDB) and 71%/67% (MIMIC II) compared to the uncorrected InceptionTime baseline.

### Key Findings

- The offsets $\hat{s}$ predicted by SyncNet exhibit strong diagonal alignment with the ground-truth offsets $s$ ($\hat{s} \approx s$), validating the effectiveness of temporal offset learning
- Co-teaching progressively discards high-loss samples as training proceeds, reducing available data; ShiftSyncNet corrects the labels of these discarded samples, substantially lowers their loss, and recovers their utilization
- Semi-supervised pseudo-labeling methods (e.g., DivideMix, C2MT) generate erroneous pseudo-labels that conflate peaks from different temporal offsets under the time-shift scenario

## Highlights & Insights

- The time-shift problem is reframed from a signal processing perspective as a learnable label correction problem, elegantly leveraging the Fourier shift theorem to achieve differentiable correction
- Rather than simply discarding misaligned samples, the proposed approach adopts a "correct and reuse" strategy that maximizes data utilization—particularly valuable in practical scenarios where aligned signal annotations are scarce
- The meta-learning framework is highly generalizable: the SyncNet design is architecture-agnostic with respect to TransNet and is in principle extensible to other sequence-to-sequence tasks involving temporal alignment issues

## Limitations & Future Work

- Temporal shifts are simulated via artificial injection; in real-world scenarios, shifts may be more complex (non-uniform, time-varying)
- A small set of aligned meta-data is required as a guidance signal, which may be difficult to obtain in certain practical settings
- Validation is limited to physiological signals (PPG/BCG→ABP) and has not been extended to other cross-modal time series transformation tasks
- The Fourier phase-shift assumption presupposes a globally uniform offset and may not be suitable for locally nonlinear temporal distortions

## Related Work & Insights

- The "easy-first, hard-later" curriculum intuition from Co-teaching provides valuable insight into deep network behavior under noisy labels; the present work offers an elegant solution to the problem of excessive data discarding at high corruption rates
- The meta-learning approach to label correction (as in MLC and MSLC) is effectively extended here: rather than directly outputting corrected labels, the model outputs temporal offsets that are then corrected via physical priors
- The Fourier shift theorem is a classical result in signal processing; embedding it into a differentiable deep learning optimization pipeline constitutes this paper's key methodological innovation

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] TEMPORA: Characterising the Time-Contingent Utility of Online Test-Time Adaptation](../../ICML2026/others/tempora_characterising_the_time-contingent_utility_of_online_test-time_adaptatio.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[CVPR 2026\] Neural Collapse in Test-Time Adaptation](../../CVPR2026/others/neural_collapse_in_test-time_adaptation.md)
- [\[NeurIPS 2025\] Deep Learning for Continuous-Time Stochastic Control with Jumps](../../NeurIPS2025/others/deep_learning_for_continuous-time_stochastic_control_with_jumps.md)
- [\[CVPR 2026\] ViT3: Unlocking Test-Time Training in Vision](../../CVPR2026/others/vit3_unlocking_test_time_training_in_vision.md)

</div>

<!-- RELATED:END -->
