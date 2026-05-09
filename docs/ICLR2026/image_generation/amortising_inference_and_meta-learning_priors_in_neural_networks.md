---
title: >-
  [Paper Note] Amortising Inference and Meta-Learning Priors in Neural Networks (BNNP)
description: >-
  [ICLR 2026][Image Generation][Bayesian neural network] This paper proposes BNNP (Bayesian Neural Network Process), a neural process that treats BNN weights as latent variables and the BNN itself as the decoder. Through layer-wise amortised variational inference, BNNP jointly learns BNN priors and inference networks across multiple datasets. It is the first work to empirically answer "Does a good prior eliminate the need for a good approximate inference method?"—the answer is no; there is no free lunch.
tags:
  - ICLR 2026
  - Image Generation
  - Bayesian neural network
  - neural process
  - meta-learning
  - amortised inference
  - prior learning
date: 2026-05-08
content_hash: aa9b713ed0624d9e
---

# Amortising Inference and Meta-Learning Priors in Neural Networks (BNNP)

**Conference**: ICLR 2026
**arXiv**: [2602.08782](https://arxiv.org/abs/2602.08782)
**Code**: Available
**Area**: Bayesian Deep Learning
**Keywords**: Bayesian neural network, neural process, meta-learning, amortised inference, prior learning

## TL;DR
This paper proposes BNNP (Bayesian Neural Network Process), a neural process that treats BNN weights as latent variables and the BNN itself as the decoder. Through layer-wise amortised variational inference, BNNP jointly learns BNN priors and inference networks across multiple datasets. It is the first work to empirically answer "Does a good prior eliminate the need for a good approximate inference method?"—the answer is no; there is no free lunch.

## Background & Motivation
**Background**: BNNs are theoretically elegant but prior selection remains a central challenge—weights lack interpretability, and convenient priors (isotropic Gaussians) degrade BNNs to Gaussian processes, sacrificing hierarchical representation learning. Neural processes learn implicit priors via meta-learning but cannot evaluate or sample from them.

**Limitations of Prior Work**: (1) It is unclear what constitutes a "good prior" for BNNs; (2) Even given a good prior, whether existing approximate inference methods (MFVI, HMC, etc.) are sufficient remains unknown; (3) Neural processes cannot perform within-task minibatching, leading to memory explosion on large datasets.

**Key Challenge**: Studying BNN behavior under a reasonable prior requires a good prior → a good prior must be learned from data → learning the prior requires good inference → good inference requires a good prior, forming a circular dependency.

**Goal**: Simultaneously address prior learning and amortised inference—meta-learn BNN priors from multiple related datasets while obtaining high-quality per-dataset posteriors.

**Key Insight**: BNN inference is reformulated as a neural process, where the latent variables are the weights and the decoder is the network parameterised by those weights. Per-layer conditional posteriors are solved in closed form via Bayesian linear regression.

**Core Idea**: BNN weights serve as the latent variables of a neural process, and the BNN itself serves as the decoder; priors and amortised inference are jointly learned from multi-task data.

## Method

### Overall Architecture
For each layer of the BNN, BNNP uses an inference network to map data points to pseudo-likelihood parameters (pseudo-observations and noise levels), then computes a closed-form layer-conditional posterior via Bayesian linear regression combined with a Gaussian prior. Samples are drawn from the first layer, propagated to the next, and the process proceeds layer by layer. Prior parameters and inference network parameters are jointly optimised across multi-task datasets.

### Key Designs

1. **Amortised Linear Layer (Core Component)**:

    - **Function**: Performs amortised inference for an arbitrary linear layer within the network.
    - **Mechanism**: An inference network $g_{\theta_l}$ maps each data point $(x_n, y_n)$ to pseudo-observations $y_n^l$ and noise levels $\sigma_{n,d}^l$. Combined with a Gaussian prior $p_{\psi_l}(W^l)$, a closed-form Bayesian linear regression yields the conditional posterior $q(W^l | W^{1:l-1}, \mathcal{D})$.
    - **Design Motivation**: The closed-form posterior corresponds to exact inference within each layer, approximating the full BNN posterior. The inference network amortises the inference process—a single forward pass suffices.

2. **PP-AVI Training Objective**:

    - **Function**: Jointly optimises prior parameters $\Psi$ and inference network parameters $\Theta$.
    - **Formula**: $\mathcal{L}_{PP-AVI} = \log q(Y_t | \mathcal{D}_c, X_t) + \mathcal{L}_{ELBO}(\mathcal{D}_c)$
    - The first term (posterior predictive density) drives predictive quality and prior learning; the second term (ELBO) drives approximate inference quality.
    - Proposition 1 proves that all three desiderata are simultaneously satisfied in the infinite-dataset limit.

3. **Within-Task Minibatching (Unique Capability)**:

    - **Function**: Enables minibatch inference over large datasets via sequential Bayesian updating.
    - **Mechanism**: The dataset is split into minibatches; the per-layer posterior is updated sequentially, with each posterior serving as the prior for the next update. Predictions are identical to full-batch processing.
    - **Design Motivation**: This capability is largely absent from existing neural processes, which require processing the entire context set at once.

4. **Adjustable Prior Flexibility**:

    - **Function**: Fixes the prior for a subset of weights while learning the prior only for the remaining weights.
    - **Design Motivation**: Prevents prior overfitting when the meta-dataset is small—inference network and prior parameters can be controlled independently.

### Loss & Training
PP-AVI objective (posterior predictive + ELBO). LoRA-style inference networks. Meta-learning paradigm with multi-task training.

## Key Experimental Results

### Approximate Posterior Quality (KL Divergence ↓)

| Method | KL(q \|\| p(W\|D)) |
|--------|-------------------|
| MFVI | High (especially at low noise) |
| FCVI | Very high |
| GIVI | Moderate |
| **BNNP (amortised)** | **Lowest** |
| **BNNP (per-task)** | **Lowest** |

### Prior Learning Capability

| Data Generating Process | Prior Sample Quality |
|------------------------|---------------------|
| Sawtooth function | Learned prior nearly indistinguishable from ground truth |
| Heaviside function | Multimodal prior successfully recovered |
| MNIST pixel regression | Recognisable digits; supports super-resolution |
| ERA5 precipitation | Effective prior learned on real-world data |

### Key Findings
- **A good prior ≠ free lunch**: Even under a learned prior, performance differences across approximate inference methods remain substantial. HMC continues to perform best under a good prior, confirming that inference quality always matters.
- Gaussian priors are sufficient to induce highly complex functional priors (including multimodal priors for the Heaviside function)—prior design is less difficult than commonly assumed.
- BNNP with partially trainable priors outperforms fully trainable neural processes in small meta-dataset settings, preventing prior overfitting.
- Within-task minibatching enables BNNP to scale to large datasets—a capability not available to other neural processes.

## Highlights & Insights
- **Unified view of "BNN weights as neural process latent variables"**: This perspective elegantly connects two distinct fields (Bayesian deep learning and probabilistic meta-learning).
- **First empirical answer to a foundational BDL question**: Does a good prior eliminate the need for good inference? The answer is no—this carries important guidance for the BDL community.
- **Surprising flexibility of Gaussian priors**: Simple Gaussian weight priors, once appropriately learned, can induce extraordinarily rich functional priors—challenging the assumption that complex prior structures are necessary.

## Limitations & Future Work
- Inference complexity scales unfavourably with network width—validation is currently limited to small-scale BNNs.
- Learning the prior requires multiple related datasets; the single-dataset setting remains unresolved.
- BNAM (the attention-based variant) breaks consistency (it is no longer a valid stochastic process)—theoretical completeness is lacking.
- Experiments are primarily conducted on 1D/2D regression tasks; validation on higher-dimensional, more complex tasks is needed.

## Related Work & Insights
- **vs. Neural Process family**: BNNP's latent variables are the decoder weights themselves (rather than abstract representations), enabling explicit evaluation and sampling of the prior.
- **vs. MFVI / HMC**: BNNP's layer-wise amortised inference quality approaches that of HMC and substantially surpasses MFVI.
- **vs. Function-space priors (Cinquin et al.)**: Function-space priors typically degenerate to GPs, whereas BNNP's weight-space prior preserves the hierarchical representational capacity of BNNs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The unified framework recasting BNNs as neural processes is highly original and conceptually profound.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four research questions, synthetic and real-world data, and comparisons against multiple VI baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations and compelling narrative centred on the "no free lunch" message.
- Value: ⭐⭐⭐⭐⭐ A foundational contribution to Bayesian deep learning—providing a new tool for studying BNN behaviour.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] NeuralOS: Towards Simulating Operating Systems via Neural Generative Models](neuralos_towards_simulating_operating_systems_via_neural_generative_models.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)
- [\[ICLR 2026\] Sample-Efficient Evidence Estimation of Score-Based Priors for Model Selection](sample-efficient_evidence_estimation_of_score_based_priors_for_model_selection.md)
- [\[ICLR 2026\] DragFlow: Unleashing DiT Priors with Region Based Supervision for Drag Editing](dragflow_unleashing_dit_priors_with_region_based_supervision_for_drag_editing.md)
- [\[ICLR 2026\] TAVAE: A VAE with Adaptable Priors Explains Contextual Modulation in the Visual Cortex](tavae_a_vae_with_adaptable_priors_explains_contextual_modulation_in_the_visual_c.md)

<!-- RELATED:END -->
