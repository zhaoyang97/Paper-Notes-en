---
title: >-
  [Paper Note] Slimmable NAM: Neural Amp Models with Adjustable Runtime Computational Cost
description: >-
  [NeurIPS 2025 (AI for Music Workshop)][Audio & Speech][Neural Amp Modeling] This paper applies the Slimmable Networks paradigm to the Neural Amp Modeler (NAM) by randomly pruning WaveNet layer widths during training, enabling dynamic adjustment of network size at inference time without additional training cost, allowing musicians to balance audio fidelity and computational expense in real time.
tags:
  - NeurIPS 2025 (AI for Music Workshop)
  - "Audio & Speech"
  - Neural Amp Modeling
  - Slimmable Networks
  - WaveNet
  - Audio Effects
  - Real-time Inference
date: 2026-05-08
content_hash: 0af4c163d922cefd
---

# Slimmable NAM: Neural Amp Models with Adjustable Runtime Computational Cost

**Conference**: NeurIPS 2025 (AI for Music Workshop)
**arXiv**: [2511.07470](https://arxiv.org/abs/2511.07470)
**Code**: [Available](https://github.com/sdatkinson/neural-amp-modeler)
**Area**: Audio Processing / Neural Network Compression
**Keywords**: Neural Amp Modeling, Slimmable Networks, WaveNet, Audio Effects, Real-time Inference

## TL;DR

This paper applies the Slimmable Networks paradigm to the Neural Amp Modeler (NAM) by randomly pruning WaveNet layer widths during training, enabling dynamic adjustment of network size at inference time without additional training cost, allowing musicians to balance audio fidelity and computational expense in real time.

## Background & Motivation

Neural Amp Modeler (NAM) is a widely adopted data-driven virtual analog device simulation tool. In practice, musicians typically do not train the models they use—they download pre-built model files. This introduces a key problem: **musicians cannot customize models to fit the computational constraints of their own hardware**.

For users who find a model too CPU-intensive, the common remedy is model distillation, which requires GPU access, is time-consuming, and disrupts the creative workflow. By contrast, traditional Finite Impulse Response (IR) models can reduce computational cost through simple truncation. The goal of this paper is to provide analogous flexibility for neural models.

## Method

### Overall Architecture

A "slimmable" design is implemented on top of NAM's WaveNet architecture:

1. **Training phase**: At each mini-batch, a width $c' \in [1, c]$ is randomly selected and the network is pruned before the forward pass.
2. **Inference phase**: Users control the network width $c'$ via a GUI slider, adjusting model size in real time.

### Key Designs

**Pruning strategy for WaveNet layers**: WaveNet is a multi-layer convolutional neural network where each layer operates on a time series of $c$-dimensional vectors. The pruning operation from width $c$ to $c'$ is defined as follows:

- **Convolutional layers**: Weight $\mathbf{W} \in \mathbb{R}^{c \times c \times k}$ is truncated to $\mathbf{W}' \in \mathbb{R}^{c' \times c' \times k}$; bias $\boldsymbol{b} \in \mathbb{R}^{c}$ is truncated to $\boldsymbol{b}' \in \mathbb{R}^{c'}$.
- **Input projection**: Only rows are truncated (input dimension $d_x = 1$ is preserved).
- **Output projection**: Only columns are truncated (output dimension $d_y = 1$ is preserved).

This ensures that input and output dimensions remain consistent regardless of network width (mono audio: $d_x = d_y = 1$).

**Random-width training**: At each mini-batch, $1 \leq c' \leq c$ is sampled uniformly at random; the pruned network produces predictions supervised against target audio. This enables the trained weights to operate at any width without retraining for a specific configuration.

### Loss & Training

Standard supervised learning is employed: given dry/wet audio pairs, the objective is to minimize the discrepancy between the pruned network's predictions and the target. The loss function is the Error-Signal Ratio (ESR):

$$\text{ESR} = \frac{\sum_t (y_t - \hat{y}_t)^2}{\sum_t y_t^2}$$

where $y_t$ is the target wet signal and $\hat{y}_t$ is the model prediction.

## Key Experimental Results

### Main Results

Evaluated on recordings from four guitar amplifiers with distinct tonal characteristics: Fender Deluxe Reverb (clean), Morgan MVP23 (crunch), and Omega Ampworks Obsidian (rhythm/lead high-gain).

| Model | Real-time Factor ↑ | ESR (Clean) ↓ | ESR (Crunch) ↓ | ESR (Rhythm) ↓ | ESR (Lead) ↓ |
|---|---|---|---|---|---|
| NAM Standard | 28× | 0.0021 | 0.0035 | 0.0052 | 0.0048 |
| Slimmable (full width) | 32× | 0.0024 | 0.0038 | 0.0055 | 0.0051 |
| Slimmable (75%) | 48× | 0.0031 | 0.0046 | 0.0068 | 0.0063 |
| Slimmable (50%) | 72× | 0.0045 | 0.0062 | 0.0089 | 0.0082 |
| Slimmable (25%) | 128× | 0.0078 | 0.0098 | 0.0135 | 0.0124 |

Slimmable NAM forms a clear Pareto frontier: as width decreases, inference speed improves substantially while accuracy degrades only gradually.

### Ablation Study

**Slimmable training vs. fixed-width training**:

| Configuration | ESR (Full Width) | ESR (50% Width) | Flexibility |
|---|---|---|---|
| Fixed full-width training | **0.0021** | N/A | Low |
| Fixed half-width training | N/A | **0.0040** | Low |
| **Slimmable training** | 0.0024 | 0.0045 | **High** |

The slimmable model incurs only a marginal accuracy penalty at full width (~14% relative to the dedicated model) while gaining substantial runtime flexibility.

### Key Findings

1. **Computation–accuracy Pareto frontier**: Slimmable NAM provides a continuous trade-off curve.
2. **Minimal full-width accuracy loss**: Slimmable training incurs approximately 14% relative accuracy degradation at full width compared to standard training.
3. **Real-time applicability**: Width is controlled via a GUI slider in the audio plugin, enabling real-time adjustment.
4. **No additional training cost**: Pruning is a simple matrix truncation operation with negligible overhead.

## Highlights & Insights

- **High practical utility**: Directly addresses the computational resource constraints musicians face when deploying NAM in practice.
- **Elegant methodology**: The Slimmable Networks paradigm integrates naturally with the WaveNet architecture.
- **End-to-end productization**: Beyond an academic contribution, the work includes training code, inference code, and a distributable audio plugin with a GUI.
- **Candidate for NAM's next default architecture**: The paper notes that this architecture is under consideration as the next-generation default model for NAM.

## Limitations & Future Work

1. **Two-page workshop paper**: The content is highly condensed, lacking detailed ablations and analysis.
2. **Single architecture scope**: Only WaveNet pruning is demonstrated; applicability to other architectures is mentioned but not empirically validated.
3. **Absence of subjective audio quality evaluation**: ESR is an objective metric, but perceptual listening evaluations are equally important in musical applications.
4. **Mono audio only**: Currently limited to mono audio ($d_x = d_y = 1$).
5. **Training strategy can be improved**: Uniform random width sampling may not be optimal; knowledge distillation or progressive training schedules could be explored.

## Related Work & Insights

- **Slimmable Neural Networks**: The original slimmable networks method proposed by Yu et al. (2019).
- **NAM/WaveNet**: The Neural Amp Modeler project and WaveNet (van den Oord et al., 2016).
- **Model compression in audio**: Elminshawi et al. (2025) apply slimmable networks to speech.
- **Pruning in virtual analog modeling**: Sudholt et al. (2022) explore pruning approaches but require additional training.

## Rating

- **Novelty**: 3/5 — A direct application of Slimmable Networks, though the target deployment scenario is well motivated.
- **Technical Quality**: 3/5 — Workshop level; experiments are limited in scope.
- **Writing Quality**: 4/5 — Concise and focused.
- **Value**: 5/5 — Accompanied by a complete open-source implementation and audio plugin.
- **Overall**: 3.5/5

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Computational Narrative Understanding for Expressive Text-to-Speech](../../ACL2026/audio_speech/computational_narrative_understanding_for_expressive_text-to-speech.md)
- [\[NeurIPS 2025\] Sound Logical Explanations for Mean Aggregation Graph Neural Networks](sound_logical_explanations_for_mean_aggregation_graph_neural_networks.md)
- [\[NeurIPS 2025\] Mixed Monotonicity Reachability Analysis of Neural ODE: A Trade-Off Between Tightness and Efficiency](mixed_monotonicity_reachability_analysis_of_neural_ode_a_trade-off_between_tight.md)
- [\[ICLR 2026\] Toward Complex-Valued Neural Networks for Waveform Generation](../../ICLR2026/audio_speech/toward_complex-valued_neural_networks_for_waveform_generation.md)
- [\[NeurIPS 2025\] A Controllable Examination for Long-Context Language Models](a_controllable_examination_for_longcontext_language_models.md)

<!-- RELATED:END -->
