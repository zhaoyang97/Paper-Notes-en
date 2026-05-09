---
title: >-
  [Paper Note] Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data
description: >-
  [NeurIPS 2025][Medical Imaging][Self-supervised learning] This paper proposes FGNO (Flow-Guided Neural Operator), which combines Flow Matching with operator learning for self-supervised pre-training on time-series data. By leveraging STFT for resolution-invariant function-space learning and treating flow time and network layer depth as adjustable "knobs" for controlling feature granularity, FGNO substantially outperforms baselines such as MAE on biomedical tasks.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - Self-supervised learning
  - Flow Matching
  - Neural Operator
  - Time series
  - Biomedical signals
date: 2026-05-08
content_hash: 2636738a2cc2bab5
---

# Self-Supervised Learning via Flow-Guided Neural Operator on Time-Series Data

**Conference**: NeurIPS 2025
**arXiv**: [2602.12267](https://arxiv.org/abs/2602.12267)
**Code**: Unavailable
**Area**: Medical Imaging / Time-Series Self-Supervised Learning
**Keywords**: Self-supervised learning, Flow Matching, Neural Operator, Time series, Biomedical signals

## TL;DR

This paper proposes FGNO (Flow-Guided Neural Operator), which combines Flow Matching with operator learning for self-supervised pre-training on time-series data. By leveraging STFT for resolution-invariant function-space learning and treating flow time and network layer depth as adjustable "knobs" for controlling feature granularity, FGNO substantially outperforms baselines such as MAE on biomedical tasks.

## Background & Motivation

Self-supervised learning (SSL) for time series faces three core challenges:

**Resolution heterogeneity**: Real-world signals are acquired at varying sampling rates (e.g., 4 Hz–200 Hz for wearable devices), and up/downsampling distorts the intrinsic properties of the signal. Conventional methods process fixed-size inputs and generalize poorly across sampling rates.

**Multi-scale requirements**: Different downstream tasks demand representations at different granularities — sleep-stage classification relies on second-level local patterns, whereas apnea index regression requires information spanning an entire night. Existing SSL methods typically produce a single latent representation.

**Fixed pre-training objectives**: MAE reconstructs inputs under a fixed masking ratio, offering limited flexibility. Diffusion and flow models have demonstrated the ability to produce multi-scale feature hierarchies at different noise levels in image domains, yet this capability remains underexplored for time-series SSL.

The core insight of FGNO is to treat **the degree of corruption (noise level / flow time) as a new degree of freedom in representation learning**, rather than a fixed hyperparameter as in MAE. Combined with the resolution invariance of STFT and the function-space learning capacity of neural operators, FGNO constructs a unified multi-scale SSL framework.

## Method

### Overall Architecture

FGNO consists of two stages:
1. **Self-supervised pre-training**: A Transformer model is trained on unlabeled data using the Flow Matching objective.
2. **Downstream probing**: The pre-trained backbone is frozen; an optimal pair $(l, s)$ (layer index and flow time) is selected, and a lightweight probe head is trained.

### Key Designs

1. **STFT data embedding**: A 1D time series $x \in \mathbb{R}^T$ is transformed into a time-frequency spectrogram $f \in \mathbb{C}^{N_f \times N_t}$ via the Short-Time Fourier Transform, and the magnitude spectrum $\phi = |\Phi|$ is used as the model input. The key advantage of STFT is **resolution invariance** — signals sampled at different rates can be directly transformed without resampling, avoiding interpolation artifacts. Unlike Fourier Neural Operators (FNO) that apply a global FFT, STFT analyzes frequency content within local temporal windows, enabling the capture of time-varying characteristics.

2. **Flow Matching pre-training**: A time-conditioned network $u_\theta(s, g)$ is trained in the spectrogram space to map a simple prior distribution (Gaussian noise) to the complex data distribution. For flow time $s \sim \mathcal{U}[0,1]$, a noisy interpolant is constructed as $g = s\phi + \sigma_s \epsilon$, and the training objective is:

$$J(\theta) = \mathbb{E}_{s, \phi, g} \left[\|v_s^\phi(g) - u_\theta(s, g)\|^2\right]$$

where the target velocity field is $v_s^\phi(g) = \frac{(\sigma_s)'}{\sigma_s}(g - s\phi) + \phi$. The backbone $u_\theta$ adopts a Transformer architecture, and flow time $s$ is incorporated as a conditioning signal via sinusoidal positional encoding.

3. **Clean-input feature extraction**: After pre-training, features are extracted from **clean spectrograms** (rather than noisy inputs) — given a clean input $\phi$ and a specified flow time $s$, the $l$-th layer activation is extracted as $z_{l,s}(\phi) = u_\theta^{(l)}(s, \phi)$. Although this introduces a train-inference distribution shift, experiments demonstrate that lightweight probe heads can effectively bridge this gap. Clean inputs eliminate noise-induced stochasticity, yielding deterministic and stable features.

### Loss & Training

- **Pre-training**: Standard Flow Matching objective ($L_2$ velocity field regression); a single model simultaneously learns multi-scale representations.
- **Probing stage**: Grid search over $(l^*, s^*) = \arg\min_{l,s} \mathcal{L}_{\text{val}}(l, s)$.
- Shallow layers with low $s$ (high corruption) yield coarse-grained global features; deep layers with high $s$ (low corruption) capture fine-grained temporal details.
- The model contains only 370K parameters, far fewer than baselines (BrainBERT: 43M, PopT: 20M).

## Key Experimental Results

### Main Results: DREAMT Dataset (Wearable Device Data)

| Task | Metric | FGNO | MAE | Chronos |
|------|--------|------|-----|---------|
| Sleep/Wake classification | AUROC (%) ↑ | **96.5** | 95.8 | 96.3 |
| Skin temperature regression | RMSE (°C) ↓ | **0.600** | 0.735 | 0.954 |

### Main Results: BrainTreeBank (Neural Signal Decoding)

| Method | Parameters | Speech AUROC | Volume | Pitch |
|--------|------------|-------------|--------|-------|
| FGNO | **370K** | **Best** | **Best** | **Best** |
| BrainBERT | 43M | 2nd | 2nd | 2nd |
| PopT | 20M | — | — | — |

FGNO outperforms all baselines on 3 out of 4 tasks, with only 1/50 the parameter count of the baselines.

### Robustness under Label Scarcity (5% Labeled Data)

| Method | SleepEDF ACC | SleepEDF MF1 | Epilepsy ACC | Epilepsy MF1 |
|--------|-------------|-------------|-------------|-------------|
| FGNO (5%) | **93.5** | **89.0** | **94.1** | **90.3** |
| TS-TCC (5%) | 77.0 | 70.9 | 93.1 | 93.7 |
| Supervised (5%) | 60.5 | 54.8 | 83.4 | 80.4 |
| FGNO (100%) | 93.9 | 89.1 | 94.8 | 90.3 |

With only 5% labeled data, FGNO nearly matches the performance achieved with full supervision (SleepEDF: 93.5% vs. 93.9%).

### Ablation Study

| Ablation | Result | Note |
|----------|--------|------|
| Clean vs. noisy input | AUROC 96.40% vs. 95.86% | Clean input is superior and deterministic, requiring no noise sampling |
| Noisy input variance | std = 0.0039 (10 runs) | Noise introduces unnecessary stochasticity |
| Cross-resolution generalization (48× downsampling) | FGNO 74%+ vs. MAE ~52% | Function-space learning naturally supports resolution invariance |
| Computational efficiency | 60% reduction in probing time | 370K parameters + frozen backbone + lightweight probe |

### Key Findings

- Different tasks require different $(l, s)$ combinations: classification tasks favor high $s$ (low corruption, preserving local patterns), while regression tasks favor intermediate $s$ (requiring global features).
- Optimal $(l, s)$ pairs form continuous, structured regions in the hyperparameter space, facilitating practical selection.
- FGNO maintains 74%+ AUROC under 48× downsampling, whereas MAE degrades to ~52%.
- Pre-training time is comparable to MAE (~21 h), but downstream adaptation time is reduced by 60%.

## Highlights & Insights

- **Flow time as a control knob**: Elevating the noise level in flow/diffusion models from a "training hyperparameter" to a "tunable feature granularity" provides an elegant mechanism for a unified framework to produce multi-scale representations.
- **Pragmatic choice of clean-input inference**: Although a train-inference distribution shift exists in principle, in practice a lightweight probe head proves sufficient to bridge this gap, while simultaneously offering determinism and efficiency.
- **Synergy of STFT and operator learning**: STFT provides resolution-invariant time-frequency representations, and operator learning operates in function space; together, they naturally support multi-resolution generalization.
- **Extreme parameter efficiency**: A 370K-parameter model surpasses a 43M-parameter baseline, suggesting that the quality of the self-supervised objective matters more than model scale.

## Limitations & Future Work

- Selecting $(l, s)$ requires grid search on a validation set, adding computational overhead to downstream adaptation.
- Evaluation is primarily conducted on biomedical signals; generalization to other time-series domains (e.g., finance, industrial IoT) remains unverified.
- The Flow Matching pre-training stage offers no significant efficiency advantage over MAE.
- No comparison is made against more recent time-series foundation models (e.g., TimesFM).
- The claim of function-space learning requires more rigorous theoretical validation.

## Related Work & Insights

- **MAE / BrainBERT**: Masked autoencoders applied to time-series SSL.
- **Chronos**: A time-series foundation model based on autoregressive T5.
- **FNO (Fourier Neural Operator)**: Models in global FFT space; this work instead adopts local STFT.
- **REPA / CleanDIFT**: Pioneering works on extracting clean-input representations from diffusion models in the image domain.
- Insight: The potential of Flow Matching in SSL warrants broader exploration across domains; the $(l, s)$ selection mechanism is extensible to other generative SSL approaches.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ An original combination of Flow Matching, operator learning, and STFT; the concept of flow time as a feature-control knob is particularly innovative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-task evaluation with comprehensive ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ Clear figures and well-motivated method description.
- **Value**: ⭐⭐⭐⭐ Practical value for biomedical time-series analysis, especially in data-scarce settings.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Towards Self-Supervised Foundation Models for Critical Care Time Series](towards_self-supervised_foundation_models_for_critical_care_time_series.md)
- [\[NeurIPS 2025\] Ditch the Denoiser: Emergence of Noise Robustness in Self-Supervised Learning from Data Curriculum](ditch_the_denoiser_emergence_of_noise_robustness_in_self-supervised_learning_fro.md)
- [\[NeurIPS 2025\] Self-supervised Learning of Echocardiographic Video Representations via Online Cluster Distillation](self-supervised_learning_of_echocardiographic_video_representations_via_online_c.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)

<!-- RELATED:END -->
