---
title: >-
  [Paper Note] Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning
description: >-
  [AAAI 2026][Interpretability][Auditory midbrain modelling] This paper proposes ψ-ICNet, a variationally conditioned deep neural network that encodes the effects of hearing loss via only 6 learnable conditioning parameter…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Auditory midbrain modelling"
  - "hearing loss"
  - "variational conditioning"
  - "neural coding"
  - "Bayesian optimization"
date: 2026-05-08
content_hash: a081486393697b6c
---

# Modelling the Effects of Hearing Loss on Neural Coding in the Auditory Midbrain with Variational Conditioning

**Conference**: AAAI 2026
**arXiv**: [2506.03088](https://arxiv.org/abs/2506.03088)  
**Code**: None  
**Area**: Audio & Speech
**Keywords**: Auditory midbrain modelling, hearing loss, variational conditioning, neural coding, Bayesian optimization

## TL;DR

This paper proposes ψ-ICNet, a variationally conditioned deep neural network that encodes the effects of hearing loss via only 6 learnable conditioning parameters ψ. The model directly learns a low-dimensional representation space of hearing loss from real neural recordings, achieving accuracy comparable to animal-specific models in predicting auditory midbrain responses in both normal-hearing and hearing-impaired animals, and can be rapidly fitted to unseen animals via Bayesian optimization.

## Background & Motivation

Computational modelling of the auditory system is essential for understanding hearing and hearing loss. Current auditory models face several key limitations:

**Parametric cochlear models (e.g., Verhulst 2018)**: These parameterize hearing loss based on known biophysical mechanisms (e.g., inner/outer hair cell damage), but:
   - They make potentially incomplete assumptions about the effects of hearing loss
   - They are computationally slow and non-differentiable
   - They cannot capture central auditory pathway changes induced by brain plasticity

**DNN approximation models (e.g., Baby 2021)**: Deep learning is used to approximate such analytical models, yielding fast and differentiable surrogates, but their training data are drawn from the original models, inheriting their biases and assumptions.

**ICNet (Drakopoulos 2025)**: A DNN trained directly on real neural recordings from the auditory midbrain (inferior colliculus, IC), accurately simulating both normal and hearing-impaired brain activity. However, a separate model must be trained from scratch for each animal, precluding generalization to new animals. The multi-branch ICNet can share dynamics across multiple normal-hearing animals but cannot handle the distinct distortion patterns specific to each impaired auditory system.

**Core Motivation**: Can a general model be learned that encodes the space of hearing loss with a small number of parameters, enabling the generation of realistic neural activity for unseen hearing conditions? This has direct application prospects for future development of parametric hearing compensation models (hearing aids).

## Method

### Overall Architecture

ψ-ICNet introduces a variational conditioning module on top of ICNet's encoder–decoder architecture. The core assumption is that the effects of hearing loss on neural coding can be parameterized by a small set of parameters ψ, with the model approximating $f(s|\theta, \psi)$, where θ denotes parameters shared across animals and ψ encodes individual hearing loss effects.

**Architecture components** (Figure 1):
- **Convolutional encoder**: SincNet layer (48 filters) + 5 causal convolutional layers (246 kernels, size 60) + final layer (64 kernels), extracting shared acoustic feature representation $\hat{r}_b$
- **Variational conditioning module**: Expands ψ into a feature map $\hat{r}_\psi$ and concatenates it with the shared representation
- **Fusion layers**: 4 convolutional layers merging shared and animal-specific features, reducing dimensionality back to 320×64
- **Shared decoder**: Outputs a categorical probability distribution $p(\hat{R}|s, \psi)$, predicting spike counts per channel per time step

### Key Designs

#### 1. **Variational Conditioning Module**: Encoding the Hearing Loss Space

The conditioning parameters ψ are the only animal-specific components in the model. Key design aspects:

- **Variational sampling**: The mean $\mu_\psi^t$ of ψ is a learnable parameter forming a multivariate Gaussian distribution. Samples are drawn using the reparameterization trick, with a diagonal covariance matrix $\Sigma_\psi$
- **Temporal drift function**: Since animal state can change over the course of a recording session (simulating progressive hearing loss), a piecewise drift function modifies the ψ mean:

$$T_\psi(\psi, t) = \begin{cases} \psi, & \text{if } t \leq o_t \\ \psi + \vec{s}_t \odot \sigma(\vec{W}_t(t-\vec{o}_t)-6) - I, & \text{otherwise} \end{cases}$$

where $\vec{s}_t$ controls the maximum degradation magnitude, $\vec{W}_t$ controls the degradation slope, and $\vec{o}_t$ controls the onset time. This design satisfies three observations: monotonicity, no degradation in the initial hours, and two degradation modes (fast/slow).

- **ψ expansion network**: 3 convolutional layers ($N_\psi$, $2N_\psi$, $3N_\psi$ kernels) expand ψ from a low-dimensional vector to a 320×$3N_\psi$ feature map

#### 2. **ABRNet Baseline**: Audiological Measurements vs. Data-Driven Conditioning

To validate that learning conditioning parameters directly from data is superior to using audiological measurements, an ABRNet variant is implemented in which ABR thresholds (auditory brainstem response thresholds at 1, 2, 4, 8, and 16 kHz) replace the learned ψ means. ABR thresholds are analogous to human audiograms; however, prior work has shown that audiograms do not fully characterize hearing loss (e.g., the phenomenon of "hidden hearing loss").

#### 3. **Cross-Brain Alignment**: Channel Alignment Preserving Hearing Loss Information

Even two normal-hearing animals will exhibit different responses due to differences in IC spatial organization. The critical requirement is that the alignment method must preserve differences caused by hearing loss (including scale differences) while removing spatial organization differences.

**Optimal transport** is employed to find channel permutations that minimize the Wasserstein distance from each brain to a reference normal-hearing animal. This permits only permutations of channel ordering, preserving all scale information.

### Loss & Training

The total loss function is:
$$\mathcal{L} = \mathcal{L}_{CE}(R, p(\hat{R})) + \alpha_{KL} D_{KL} + \alpha_{ABR} \mathcal{L}_{ABR}$$

- **Cross-entropy loss**: Between model output and the one-hot representation of real MUA
- **KL divergence regularization**: $\alpha_{KL} = 10^{-3}$, constraining the ψ distribution toward the target distribution (covariance $= \frac{1}{N_A} \cdot \mathbb{I}$)
- **ABR-guided loss** (first 10 epochs): $\alpha_{ABR} = 0.02$, penalizing animal pairs with large ABR threshold differences but similar ψ encodings, encouraging cluster structure in the ψ space to align with hearing status

Training configuration: Adam optimizer, learning rate 4e-4, up to 120 epochs with early stopping (patience 10), encoder initialized from ICNet weights trained on 12 normal-hearing animals.

## Key Experimental Results

### Main Results

Trained on 9 gerbils (3 normal-hearing, NH + 6 noise-exposed hearing-impaired, HI); evaluated using FEVE (fraction of explainable variance explained) and KL divergence.

| Model | NH FEVE | HI FEVE | NH KL/10⁻³ | HI KL/10⁻³ |
|-------|---------|---------|------------|------------|
| ICNet (single-branch) | 67±14 | 71±11 | 8.4±2 | 10.4±4 |
| ABRNet | -5.6±4 | -3.7±10 | -0.36±3 | -0.35±6 |
| ψNet-3 | -7.2±5 | -3.5±10 | -0.98±3 | -1.0±4 |
| ψNet-6 | **-5.3±5** | **-2.1±7** | -1.1±3 | -1.3±4 |

(Values are median differences ± median absolute deviations relative to ICNet. ψNet-6 lags behind ICNet by only 2.1% FEVE on HI animals, and matches or slightly surpasses ICNet on KL divergence.)

### Ablation Study: Generalization to Unseen Animals

Bayesian optimization is used to search for ψ parameters to fit new animals; results are compared between a 9-animal model and a 20-animal model.

| Condition | 9-animal model | 20-animal model |
|-----------|---------------|-----------------|
| In-sample FEVE | 55.6% | 50.7% |
| Unseen animal FEVE | 7.8% | **26.9%** |
| Fitting iterations | 15–30 | 15–35 |

- Increasing training data improves unseen animal FEVE from 7.8% to 26.9% (+245%), with a slight reduction in in-sample performance
- The loss landscape in ψ space is smooth; NH and HI animals exhibit clearly distinct landscape shapes
- Bayesian optimization converges to within 2% of the target loss in 15–30 iterations

### Key Findings

1. **Data-driven conditioning outperforms audiological measurements**: ψ-ICNet surpasses ABRNet on all metrics, demonstrating that a conditioning space learned directly from data captures hearing loss more effectively than ABR thresholds
2. **ψ space exhibits smooth structure**: Normal-hearing and hearing-impaired animals are well-separated in ψ space, with a continuous and smooth loss landscape
3. **More animals improve generalization**: The 20-animal model substantially outperforms the 9-animal model on unseen animals, indicating that the model continues to benefit from additional training data
4. **6 parameters suffice**: Increasing ψ dimensionality does not improve performance but increases fitting time
5. **Temporal drift function is effective**: It successfully captures progressive degradation of animal state during recordings

## Highlights & Insights

- **Paradigm shift from "one model per brain" to "universal parametric model"**: Encoding the entire hearing loss spectrum with 6 parameters enables rapid adaptation to new subjects
- **Elegant use of variational conditioning**: Randomness is introduced not only to improve generalization, but the KL divergence regularization also imposes structural constraints on the ψ space
- **Learning the hearing loss space directly from data**: No reliance on prior assumptions (e.g., audiograms), enabling the capture of complex phenomena such as "hidden hearing loss"
- **Practical relevance for hearing aids**: A parametric model can be used to train compensation strategies and rapidly personalized for new users through human-in-the-loop optimization

## Limitations & Future Work

1. In-sample performance remains slightly below animal-specific ICNet, particularly for normal-hearing animals
2. A substantial gap persists between unseen-animal FEVE (26.9%) and in-sample FEVE (50.7%), possibly because channel alignment residuals do not generalize
3. Training data covers only 9–20 gerbils; validation at larger scale is needed
4. The temporal drift function design is heuristic and may not generalize to all degradation patterns
5. Validation is currently limited to gerbil IC data; additional work is required to transfer findings to human hearing loss models

## Related Work & Insights

- **ICNet** (Drakopoulos 2025): The base model directly extended in this work, already shown to accurately simulate auditory midbrain activity under normal hearing
- **Neural Data Transformer** (Ye 2023): Conditional embeddings for cross-subject/session neural activity reconstruction; this paper applies a similar idea
- **Bysted 2022**: Conditions cochlear DNN model weights on audiograms, but is limited to the peripheral auditory system
- Insight: The combination of a **low-dimensional conditioning space and Bayesian optimization** provides an elegant solution for "universal model + rapid personalization," with potential generalization to other neural interface domains

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First universal parametric auditory midbrain model that encodes hearing loss; conceptually novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-model comparisons, ψ space visualization, and generalization tests are thorough, though the number of animals is limited
- **Writing Quality**: ⭐⭐⭐⭐ — Technically clear, well-motivated, and visually well-presented
- **Value**: ⭐⭐⭐⭐⭐ — Direct potential for future hearing aid/compensation technology, with clear application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] How Language Models Conflate Logical Validity with Plausibility: A Representational Analysis of Content Effects](../../ACL2026/interpretability/how_language_models_conflate_logical_validity_with_plausibility_a_representation.md)
- [\[AAAI 2026\] ElementaryNet: A Non-Strategic Neural Network for Predicting Human Behavior in Normal-Form Games](elementarynet_a_non-strategic_neural_network_for_predicting_human_behavior_in_no.md)
- [\[NeurIPS 2025\] What Happens During the Loss Plateau? Understanding Abrupt Learning in Transformers](../../NeurIPS2025/interpretability/what_happens_during_the_loss_plateau_understanding_abrupt_learning_in_transforme.md)
- [\[ICLR 2026\] Provably Explaining Neural Additive Models](../../ICLR2026/interpretability/provably_explaining_neural_additive_models.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](../../ICLR2026/interpretability/modal_logical_neural_networks_for_financial_ai.md)

</div>

<!-- RELATED:END -->
