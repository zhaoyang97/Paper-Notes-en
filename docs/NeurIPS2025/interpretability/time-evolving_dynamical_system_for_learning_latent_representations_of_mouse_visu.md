---
title: >-
  [Paper Note] Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex
description: >-
  [NeurIPS 2025][Interpretability][latent variable models] This paper proposes TE-ViDS, a sequential latent variable model that decomposes visual neural activity into an external representation linked to visual stimuli and…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "latent variable models"
  - "visual neural activity"
  - "time-evolving dynamical systems"
  - "contrastive learning"
  - "mouse visual cortex"
date: 2026-05-08
content_hash: 6a4264d89e43e3d0
---

# Time-Evolving Dynamical System for Learning Latent Representations of Mouse Visual Cortex

**Conference**: NeurIPS 2025
**arXiv**: [2408.07908](https://arxiv.org/abs/2408.07908)  
**Code**: [Available](https://github.com/Grasshlw/Time-Evolving-Visual-Dynamical-System)  
**Area**: Interpretability
**Keywords**: latent variable models, visual neural activity, time-evolving dynamical systems, contrastive learning, mouse visual cortex

## TL;DR

This paper proposes TE-ViDS, a sequential latent variable model that decomposes visual neural activity into an external representation linked to visual stimuli and an internal representation reflecting internal states. By incorporating a time-evolving structure and contrastive learning, TE-ViDS achieves state-of-the-art decoding performance on natural scenes and videos.

## Background & Motivation

Latent variable models (LVMs) reveal intrinsic associations between neural activity and behavior or sensory stimuli by constructing low-dimensional representations, making them central to neural data analysis. However, three important gaps exist in the literature:

**State of the Field — Bias toward motor cortex**: Most LVM research focuses on motor regions (e.g., pre-planned movement), with relatively little work on the **visual cortex**.

**Limitations of Prior Work — Temporal relationships ignored**: Natural visual stimuli are inherently high-dimensional and temporally dependent, yet most LVMs do not explicitly model the temporal structure of neural activity.

**Limitations of Prior Work — Visual-specific properties underutilized**: Visual neural activity contains both stimulus-related and internal-state components, which existing methods do not specifically address.

**Key Challenge**: When mice passively observe natural scenes or videos, the neural dynamics in the visual cortex are driven by two factors:
- **External visual stimuli**: Scene or movie frame content
- **Internal states**: Attention, arousal level, etc., which may exert an even greater influence on neural activity than the visual stimuli themselves.

How to construct high-quality latent representations that disentangle these two components is therefore a critical open problem.

## Method

### Overall Architecture

TE-ViDS is a sequential latent variable model whose core components include:
- **Encoder**: Extracts spatial features from sequential spike data
- **Time-evolving system**: Evolves latent variables conditioned on RNN state factors
- **Decoder**: Maps latent variables to inferred firing rates
- **Disentangled design**: External latent variables (deterministic) + internal latent variables (stochastic)

The input is $\mathbf{x} = (\mathbf{x}_1, ..., \mathbf{x}_T) \in \mathbb{R}^{T \times N}$ (spike counts from $N$ neurons across $T$ time windows).

### Key Designs

#### 1. External Latent Variables (Deterministic, Stimulus-Related)

**Function**: Capture the component of neural activity associated with visual stimuli.

$$\mathbf{z}_t^{(e)} = f_{\text{enc}}^{(e)}(f_x(\mathbf{x}_t), \mathbf{h}_{t-1}^{(e)})$$

**Mechanism**: Designed as deterministic (non-stochastic) values, since stimulus-related components should be stable and variability should be attributed to internal states. Shaped via contrastive learning (NT-Xent loss) — temporally offset sequences serve as positive pairs (as adjacent-time visual stimuli are similar), while negative samples are drawn randomly from the training set.

**Design Motivation**: Positive pairs cover time segments with similar visual stimuli, naturally aligning external representations with stimulus content. A swap operation is also applied — exchanging external representations between positive pairs while preserving internal representations — to further enhance disentanglement.

#### 2. Internal Latent Variables (Stochastic, State-Related)

**Function**: Reflect the animal's internal dynamic states (attention, arousal, etc.), which exhibit high variability and noise.

Approximate posterior: $\mathbf{z}_t^{(i)} | \mathbf{x}_{1:t}, \mathbf{h}_{1:t-1}^{(i)} \sim \mathcal{N}(\boldsymbol{\mu}_{z,t}, \boldsymbol{\sigma}_{z,t}^2 \cdot \mathbf{I})$

Prior distribution: $\tilde{\mathbf{z}}_t^{(i)} | \mathbf{h}_{1:t-1}^{(i)} \sim \mathcal{N}(\tilde{\boldsymbol{\mu}}_{z,t}, \tilde{\boldsymbol{\sigma}}_{z,t}^2 \cdot \mathbf{I})$

**Mechanism**: Modeled as stochastic variables whose prior depends only on the previous state factor (capturing temporal spontaneity); KL divergence constrains the gap between posterior and prior.

**Design Motivation**: Internal states are inherently variable and noisy, making stochastic modeling more appropriate. A temporally dependent prior allows the model to capture the slow drift of internal states.

#### 3. Time-Evolving Mechanism (GRU State Factors)

Two independent GRUs maintain external and internal state factors, respectively:

$$\mathbf{h}_t^{(e)} = f_{\text{GRU}}^{(e)}(f_x(\mathbf{x}_t), \mathbf{h}_{t-1}^{(e)})$$
$$\mathbf{h}_t^{(i)} = f_{\text{GRU}}^{(i)}(f_x(\mathbf{x}_t), \mathbf{z}_t^{(e)}, \mathbf{z}_t^{(i)}, \mathbf{h}_{t-1}^{(i)})$$

A key distinction: the GRU for the internal state factor additionally receives the external latent variable as input, reflecting the fact that internal states are inevitably influenced by visual stimuli.

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{\text{recons}} + \beta \mathcal{L}_{\text{contrastive}} + \gamma \mathcal{L}_{\text{regular}}$$

- $\mathcal{L}_{\text{recons}}$: Poisson negative log-likelihood (spike count reconstruction)
- $\mathcal{L}_{\text{contrastive}}$: NT-Xent contrastive loss (shaping external representations)
- $\mathcal{L}_{\text{regular}}$: KL divergence + prior regularization (constraining internal representations)

## Key Experimental Results

### Main Results 1: Natural Scene Decoding (118 scene images)

| Model | Mouse 1 | Mouse 2 | Mouse 3 | Mouse 4 | Mouse 5 |
|-------|---------|---------|---------|---------|---------|
| PCA | 0.59% | 1.53% | 1.53% | 0.80% | 0.85% |
| LFADS | 30.76% | 16.46% | 22.20% | 19.69% | 4.69% |
| pi-VAE | 7.49% | 19.42% | 22.92% | 13.71% | 2.22% |
| Swap-VAE | 32.81% | 24.34% | 14.36% | 14.85% | 3.92% |
| CEBRA | 1.53% | 3.42% | 4.86% | 2.81% | 1.08% |
| TE-ViDS-small | 47.08% | 23.95% | 29.08% | 34.95% | 9.93% |
| **TE-ViDS** | **50.86%** | **27.24%** | **29.90%** | **38.05%** | **9.44%** |

TE-ViDS achieves the highest decoding accuracy across all five mice, with substantial margins over the second-best model (18% gain for Mouse 1, 23% for Mouse 4).

### Main Results 2: Natural Movie Frame Decoding (900 frames, 1-second windows)

| Model | Mouse 1 | Mouse 2 | Mouse 3 | Mouse 4 | Mouse 5 |
|-------|---------|---------|---------|---------|---------|
| PCA | 8.44% | 28.77% | 25.42% | 21.56% | 11.69% |
| LFADS | 8.94% | 26.57% | 26.77% | 24.76% | 12.69% |
| Swap-VAE | 12.19% | 51.31% | 45.96% | 41.53% | 22.70% |
| CEBRA | 10.62% | 52.76% | 61.01% | 42.11% | 22.33% |
| **TE-ViDS** | **13.88%** | **65.38%** | **59.88%** | **54.33%** | **30.18%** |

### Ablation Study

| Configuration | Key Metric | Remarks |
|--------------|-----------|---------|
| External vs. internal representations | External decoding score >> internal | Validates the hypothesis that external representations capture stimulus-related information |
| Temporal vs. non-temporal synthetic data | Performance drops sharply after time dimension shuffling | Demonstrates model sensitivity to temporal structure |
| TE-ViDS vs. TE-ViDS-small | Comparable or marginally better | Small model is also effective; gains are not due to parameter scaling |
| Comparison across 6 cortical areas | VISp highest, VISrl lowest | Provides computational evidence for a functional hierarchy in the visual cortex |

### Key Findings

1. **Mechanistic basis of individual differences**: RSA analysis reveals that Mouse 1's neural representations split into two distinct temporal epochs across scenes (attributable to internal state shifts), whereas Mouse 2 shows no such pattern. This explains the large variance in decoding performance across animals.
2. **Evidence for cortical hierarchy**: Primary and intermediate visual areas (VISp, VISl, VISal) show higher decoding performance than higher-order areas (VISpm, VISam), with the multisensory area VISrl scoring lowest — offering novel computational evidence for a functional hierarchy in the mouse visual cortex.
3. **Limitations of CEBRA**: CEBRA performs extremely poorly on natural scene decoding (~3%), indicating that its fixed-kernel temporal encoding is ill-suited for extracting temporal features under static stimulation.

## Highlights & Insights

1. **Stimulus-related and state-related disentanglement strategy**: The deterministic-external plus stochastic-internal design precisely matches the two components of visual neural activity.
2. **Natural application of contrastive learning**: Using temporally offset sequences as positive pairs is highly principled — visually similar stimuli naturally occur at adjacent time points.
3. **Rich biological insights**: Beyond methodological contributions, the work reveals the influence of internal states on visual coding and functional differences across cortical regions.
4. **Methodological generality**: The framework is not limited to the mouse visual cortex and can be extended to other species, brain regions, and modalities.

## Limitations & Future Work

1. **Lack of quantitative evaluation for internal representations**: No behavioral or internal state recordings are available to validate the interpretability of the internal latent variables.
2. **Large individual variability**: The substantial gap in decoding performance between Mouse 1 and Mouse 5 indicates that the model does not fully overcome inter-individual variability.
3. **Passive viewing paradigm only**: Mice do not perform any task, precluding direct links between representations and task-related behavior.
4. **Computational cost not thoroughly discussed**: The time complexity of sequential GRU processing may become a bottleneck for very long time series.

## Related Work & Insights

- **vs. CEBRA (Schneider 2023)**: CEBRA encodes temporal features via fixed convolutional kernels, whereas TE-ViDS uses RNN-based dynamic evolution, which is better suited to visual neural activity.
- **vs. Swap-VAE (Liu 2021)**: TE-ViDS inherits the swap operation and split architecture but augments them with a time-evolving mechanism.
- The influence of internal states on perception is consistent with the behavioral findings of Ashwood (2022).
- Future work could integrate brain–computer interface applications, leveraging the disentangled representations to handle stimulus and state information separately.

## Rating
- Novelty: ⭐⭐⭐⭐ (Time-evolving + disentangled design is well-motivated but not revolutionary)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Synthetic and real neural data; comprehensive multi-animal, multi-region analysis)
- Writing Quality: ⭐⭐⭐⭐ (Methods are clearly presented; biological discussion is in-depth)
- Value: ⭐⭐⭐⭐ (Fills a gap in LVM research on visual neural activity; provides valuable neuroscientific insights)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SEED-SET: Scalable Evolving Experimental Design for System-level Ethical Testing](../../ICLR2026/interpretability/seed-set_scalable_evolving_experimental_design_for_system-level_ethical_testing.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](../../ICLR2026/interpretability/decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[NeurIPS 2025\] Latent Principle Discovery for Language Model Self-Improvement](latent_principle_discovery_for_language_model_self-improvement.md)
- [\[NeurIPS 2025\] Partial Information Decomposition via Normalizing Flows in Latent Gaussian Distributions](partial_information_decomposition_via_normalizing_flows_in_latent_gaussian_distr.md)
- [\[ICLR 2026\] MATA: A Trainable Hierarchical Automaton System for Multi-Agent Visual Reasoning](../../ICLR2026/interpretability/mata_a_trainable_hierarchical_automaton_system_for_multi-agent_visual_reasoning.md)

</div>

<!-- RELATED:END -->
