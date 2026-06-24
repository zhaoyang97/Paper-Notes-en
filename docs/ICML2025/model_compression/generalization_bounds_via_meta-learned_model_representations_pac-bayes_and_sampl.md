---
title: >-
  [Paper Note] Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks
description: >-
  [ICML 2025][Model Compression][Generalization Bounds] This paper proposes a hypernetwork-based meta-learning framework to obtain tight generalization bounds for neural networks. Three encoder-decoder architectures are designed (PAC-Bayes encoder, sample compression encoder, and a hybrid encoder). The hybrid approach is based on a new PAC-Bayes Sample Compression theorem supporting continuous messages, explicitly measuring model complexity via an information bottleneck…
tags:
  - "ICML 2025"
  - "Model Compression"
  - "Generalization Bounds"
  - "PAC-Bayes"
  - "Sample Compression"
  - "Meta-Learning"
  - "Hypernetwork"
date: 2026-05-08
content_hash: 366089fcc62e21c8
---

# Generalization Bounds via Meta-Learned Model Representations: PAC-Bayes and Sample Compression Hypernetworks

**Conference**: ICML 2025  
**arXiv**: [2410.13577](https://arxiv.org/abs/2410.13577)  
**Code**: [GRAAL-Research/DeepRM](https://github.com/GRAAL-Research/DeepRM)  
**Area**: Model Compression  
**Keywords**: Generalization Bounds, PAC-Bayes, Sample Compression, Meta-Learning, Hypernetwork

## TL;DR

This paper proposes a hypernetwork-based meta-learning framework to obtain tight generalization bounds for neural networks. Three encoder-decoder architectures are designed (PAC-Bayes encoder, sample compression encoder, and a hybrid encoder). The hybrid approach is based on a new PAC-Bayes Sample Compression theorem supporting continuous messages, explicitly measuring model complexity via an information bottleneck, and obtaining non-vacuous generalization guarantees on both synthetic and real datasets.

## Background & Motivation

### Background
Ensuring the reliability of machine learning models fundamentally relies on understanding their generalization capabilities. For deep neural networks, classical approaches (which rely on naive complexity measures like parameter count) often yield vacuous generalization bounds. Recently, PAC-Bayes theory and sample compression theory have been shown to provide non-vacuous generalization bounds for neural networks.

### Limitations of Prior Work

**Failure of complexity measures**: Naive measures like parameter count cannot reflect the true effective complexity of deep networks.

**PAC-Bayes dependence on priors**: These methods require specifying a prior distribution and usually only guarantee the expected loss of a randomized predictor.

**Sample compression constraints**: Prior works traditionally support only discrete messages, limiting expressiveness.

**Lack of practical frameworks**: No systematic framework exists to simultaneously train models and compute generalization bounds.

### Key Challenge
How to obtain computationally feasible and numerically non-vacuous generalization bounds via learned compact representations while preserving model accuracy?

### Key Insight
Design hypernetwork architectures with an explicit information bottleneck, allowing the complexity of the bottleneck to be directly used to compute generalization guarantees. Core metaphor: "The hypernetwork acts as a learning algorithm that explicitly exposes the complexity of the generated models."

## Method

### Overall Architecture

**Meta-Learning Setup**:

1. Train a hypernetwork $\mathscr{H}_\theta$, which takes a training dataset $S$ as input and outputs the parameters $\gamma$ of a downstream predictor.
2. The hypernetwork adopts an encoder-decoder architecture with an information bottleneck in between.
3. The "size" of the bottleneck directly corresponds to the complexity terms in the generalization bounds.
4. After meta-training, for any new task dataset $S'$, the output predictor $h_{\gamma'}$ is equipped with a generalization guarantee.

### Key Designs

#### 1. PAC-Bayes Hypernetwork (PBH)

- **Encoder** $\mathscr{E}_\phi$: Maps the dataset to a mean vector $\boldsymbol{\mu} \in \mathbb{R}^{|\boldsymbol{\mu}|}$
- **Bottleneck**: Posterior distribution $Q_{\boldsymbol{\mu}} = \mathcal{N}(\boldsymbol{\mu}, \mathbf{I})$, prior $P_0 = \mathcal{N}(\mathbf{0}, \mathbf{I})$
- **Decoder** $\mathscr{D}_\psi$: Generates predictor parameters $\gamma$ from latent vectors sampled from the posterior

**Generalization Bound**:
$$\text{kl}\left(\mathbb{E}\hat{\mathcal{L}}_{S'}(h_{\gamma'}), \tau\right) \leq \frac{\frac{1}{2}\|\boldsymbol{\mu}\|^2 + \ln\frac{2\sqrt{m'}}{\delta}}{m'}$$

where $\frac{1}{2}\|\boldsymbol{\mu}\|^2 = \text{KL}(Q_{\boldsymbol{\mu}} \| P_0)$, meaning the L2 norm of the representation directly controls the generalization bound.

#### 2. Sample Compression Hypernetwork (SCH)

- **Sample Compressor** $\mathscr{C}_{\phi_1}$: Selects $c$ key samples from the training set to form a compression set
- **Message Compressor** $\mathscr{M}_{\phi_2}$: Generates a binary message $\boldsymbol{\omega} \in \{-1, 1\}^b$
- **Reconstructor** $\mathscr{R}_\psi$: Generates predictor parameters from the compression set and the message

**Attention-based Sample Selection**: Uses $c$ independent attention mechanisms, where queries come from a DeepSet, keys from an FC network, and values are the samples themselves. Samples with the highest probabilities are selected for the compression set.

**Generalization Bound** (Theorem 2.2/2.3): Complexity is controlled by the compression set size $c$ and the message length $b$.

#### 3. PAC-Bayes Sample Compression Hypernetwork (PB SCH) —— Core Innovation

**New Theorem 2.4**: Replaces discrete messages in the sample compression framework with continuous messages, utilizing PAC-Bayes to handle the posterior distribution of the messages.

- **Sample Compressor**: Unchanged, selects $c$ samples
- **PAC-Bayes Encoder**: Replaces the message compressor and outputs a continuous mean $\boldsymbol{\mu} \in \mathbb{R}^b$
- **Posterior**: $Q_{\Omega,\boldsymbol{\mu}} = \mathcal{N}(\boldsymbol{\mu}, \mathbf{I})$

**Hybrid Generalization Bound**:
$$\text{kl}\left(\mathbb{E}\hat{\mathcal{L}}_{S'_{\bar{\mathbf{j}}}}(h_{\gamma'}), \tau\right) \leq \frac{\frac{1}{2}\|\boldsymbol{\mu}'\|^2 + \ln\frac{2\sqrt{m'-c}}{p \cdot \delta}}{m' - c}$$

**Disintegrated Version (Theorem 2.5)**: Further derives a generalization bound for a single deterministic predictor (rather than expected value) using Rényi divergence $D_\alpha$.

### Loss & Training

**Meta-training Objective**:
$$\min_{\psi, \phi_1, \phi_2} \frac{1}{n} \sum_{i=1}^n \mathbb{E}\hat{\mathcal{L}}_{\hat{T}_i}(h_{\gamma_i})$$

where $\gamma_i = \mathscr{R}_\psi(\mathscr{C}_{\phi_1}(\hat{S}_i), \mathscr{E}_{\phi_2}(\hat{S}_i) + \boldsymbol{\epsilon})$, with $\boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$.

**Key Point**: The reconstructor is learned on meta-training data, but the generalization bounds are computed on new task data, which satisfies statistical validity.

## Key Experimental Results

### Main Results: MNIST-pixels-swap Task

| Method | 100 Pixel Swap | | 200 Pixel Swap | | 300 Pixel Swap | |
|------|------|------|------|------|------|------|
| | Bound ↓ | Error ↓ | Bound ↓ | Error ↓ | Bound ↓ | Error ↓ |
| Pentina & Lampert 2014 | 0.190 | 0.019 | 0.240 | 0.026 | 0.334 | 0.038 |
| PBH (Ours) | ~0.15 | ~0.015 | ~0.20 | ~0.022 | ~0.28 | ~0.032 |
| SCH- (Ours) | — | ~0.02 | — | ~0.025 | — | ~0.035 |
| PB SCH (Ours) | **~0.12** | **~0.013** | **~0.17** | ~0.020 | **~0.25** | ~0.030 |

The hybrid method (PB SCH) performs best in both bound tightness and test error.

### Ablation Study: Synthetic Moons Task

| Method | Compression Set Size $c$ | Message Size $b/|\mu|$ | Test Error |
|------|-------------|-----------------|---------|
| PBH | — | 2 | ~1% |
| SCH- | 3 | 0 | ~3% |
| SCH+ | 3 | 4 | ~2% |
| PB SCH | 3 | 2 | **~1%** |

### Key Findings

1. **Information bottleneck is an effective complexity proxy**: A smaller $\|\boldsymbol{\mu}\|^2$ leads to a tighter bound and positively correlates with actual generalization capability.
2. **Interpretable latent space**: Figure 5 demonstrates the independent control of each dimension over the decision boundary in the 2D latent space.
3. **Sample compressor learns meaningful compression**: Figure 6 shows that the selected 3 samples successfully represent key structures of the dataset.
4. **Hybrid approach is optimal**: PB SCH combines discrete compression via sample selection with the flexibility of continuous messages.
5. **Non-vacuous bounds**: Achieved generalization bounds of < 0.25 on real datasets, which is far superior to vacuous trivial bounds of 1.0.

## Highlights & Insights

1. **Theoretical Innovation**: Theorem 2.4 is the first PAC-Bayes sample compression theorem to support continuous messages, unifying two previously independent learning theory frameworks.
2. **A "Learning Reconstructor" Perspective**: Traditional sample compression relies on predefined reconstruction functions. This paper learns the reconstruction function via meta-learning.
3. **Information Bottleneck = Generalization Guarantee**: The architectural design directly links the bottleneck size with the generalization bound, elegantly unifying model design with theoretical analysis.
4. **Disintegrated Bound** (Theorem 2.5): Extends expected value bounds to single predictor bounds, offering greater practical relevance.
5. **DeepSet Encoding**: Smartly uses $\mathbf{z} = \frac{1}{m}\mathbf{M}^T\mathbf{y}$ to achieve permutation invariance, which is simple yet effective.

## Limitations & Future Work

1. **Scalability**: Currently verified on small-scale datasets and simple networks (few-shot), and not yet scaled to massive deep networks.
2. **Fixed Compression Set Size**: The hyperparameter $c$ must be preset. Adaptive selection of compression set size could further improve performance.
3. **Prior Selection**: Using a standard normal prior; data-dependent priors might yield tighter bounds.
4. **Computational Cost**: PB SCH requires simultaneously training the sample compressor and the PAC-Bayes encoder, which increases training complexity.
5. **Meta-learning Assumptions**: The assumption that all tasks are sampled i.i.d. from a meta-distribution may not hold in practice.
6. **Lack of comparison with latest meta-learning methods**: Such as MAML, ProtoNet, etc.

## Related Work & Insights

- **Classical PAC-Bayes**: McAllester 1998, Germain et al. 2015 → General PAC-Bayes Theorem
- **Non-vacuous Bounds**: Dziugaite & Roy 2017 → First non-vacuous PAC-Bayes bounds for deep networks
- **LLM Generalization Bounds**: Lotfi et al. 2024 → Meaningful generalization bounds might be obtainable even for large models
- **Sample Compression**: Littlestone & Warmuth 1986, Laviolette et al. 2005, Bazinet et al. 2025 → Theoretical foundations
- **Meta-learning PAC-Bayes**: Pentina & Lampert 2014, Amit & Meir 2018 → Hierarchical priors and posteriors
- **Insight**: Measuring complexity via learned representations reflects the true effective dimension of models much better than hand-crafted complexity metrics.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ A new theorem unifying PAC-Bayes and sample compression, alongside the novel perspective of a meta-learned reconstructor.
- Experimental Thoroughness: ⭐⭐⭐ Thorough verification on synthetic and small-scale real data, but lacks large-scale experiments.
- Writing Quality: ⭐⭐⭐⭐ Rigorous and clear theoretical parts and elegant illustrations, but high theorem density makes for a steep learning curve.
- Value: ⭐⭐⭐⭐ Significant theoretical contribution, though the path to practical applications remains to be expanded.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Tighter CMI-Based Generalization Bounds via Stochastic Projection and Quantization](../../NeurIPS2025/model_compression/tighter_cmi-based_generalization_bounds_via_stochastic_projection_and_quantizati.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](../../CVPR2025/model_compression/learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[ICLR 2026\] Beyond Uniformity: Sample and Frequency Meta Weighting for Post-Training Quantization of Diffusion Models](../../ICLR2026/model_compression/beyond_uniformity_sample_and_frequency_meta_weighting_for_post-training_quantiza.md)
- [\[ICML 2025\] any4: Learned 4-bit Numeric Representation for LLMs](any4_learned_4-bit_numeric_representation_for_llms.md)
- [\[ICML 2025\] RADIO: Rate-Distortion Optimization for Large Language Model Compression](radio_rate-distortion_optimization_for_large_language_model_compression.md)

</div>

<!-- RELATED:END -->
