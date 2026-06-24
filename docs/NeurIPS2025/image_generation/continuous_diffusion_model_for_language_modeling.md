---
title: >-
  [Paper Note] Continuous Diffusion Model for Language Modeling
description: >-
  [NeurIPS 2025][Image Generation][Riemannian diffusion] This paper proposes RDLM (Riemannian Diffusion Language Model), which constructs a continuous diffusion process on a statistical manifold (hypersphere) to model discrete distributions. It establishes a theoretical connection between discrete diffusion and continuous flows, and leverages radial symmetry to enable simulation-free training and a dimension-splitting technique for handling large vocabularies. RDLM achieves 1.3…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Riemannian diffusion"
  - "statistical manifold"
  - "discrete data"
  - "sphere"
  - "language model"
date: 2026-05-08
content_hash: 186b616995839cef
---

# Continuous Diffusion Model for Language Modeling

**Conference**: NeurIPS 2025
**arXiv**: [2502.11564](https://arxiv.org/abs/2502.11564)  
**Code**: [GitHub](https://github.com/harryjo97/RDLM)  
**Area**: Generative Models / Language Modeling
**Keywords**: [Riemannian diffusion, statistical manifold, discrete data, sphere, language model]

## TL;DR

This paper proposes RDLM (Riemannian Diffusion Language Model), which constructs a continuous diffusion process on a statistical manifold (hypersphere) to model discrete distributions. It establishes a theoretical connection between discrete diffusion and continuous flows, and leverages radial symmetry to enable simulation-free training and a dimension-splitting technique for handling large vocabularies. RDLM achieves 1.32 BPC on Text8, surpassing all discrete and continuous diffusion models.

## Background & Motivation

**Background**: Discrete diffusion models (D3PM, SEDD, MDLM) model discrete state spaces via Markov chains and have demonstrated competitive performance in language modeling. However, the jump-based transitions between discrete states lead to information loss and prevent these models from fully exploiting iterative refinement, a core advantage of continuous diffusion.

**Limitations of Prior Work**: (1) Discrete diffusion transitions between states are irreversible, limiting generation quality and controllability. (2) Existing continuous diffusion methods (e.g., relaxation in Euclidean space) ignore the geometric structure of categorical distributions and fall significantly behind discrete methods in performance. (3) Flow matching methods on statistical manifolds (Fisher-Flow, CatFlow) are restricted to short sequences and small vocabularies.

**Key Challenge**: How to preserve the geometric structure of categorical distributions while enabling discrete data to benefit from continuous diffusion's iterative refinement, and scale to large vocabularies and long sequences.

**Goal**: Establish a unified theoretical connection between discrete diffusion and continuous flows, and design a practical continuous diffusion framework as an alternative to discrete jumps.

**Key Insight**: The statistical manifold of categorical distributions (probability simplex) is diffeomorphic to the positive orthant of the hypersphere via $\pi: p_i \mapsto \sqrt{p_i}$.

**Core Idea**: The transition distributions of discrete diffusion can be modeled as continuous flows on the statistical manifold, and a mixture of bridge processes on the hypersphere can generalize and improve upon discrete diffusion.

## Method

### Overall Architecture

The core mechanism of RDLM is to map discrete tokens via one-hot encoding to standard basis vectors $\mathbf{e}_k$ on the hypersphere $\mathbb{S}^{d-1}$, construct bridge processes from an initial point (a mask token $\mathbf{e}_m$ or a uniform point $\sum \mathbf{e}_i/\sqrt{d}$) to a target token $\mathbf{e}_k$, and then combine all bridge processes using a diffusion mixture representation to form the complete generative process. The model predicts arrival probabilities $p_{T|t}(\mathbf{e}_k|\mathbf{X}_t)$, is trained with a cross-entropy loss, and approximates the transition distribution via a Riemannian normal distribution to enable simulation-free training.

### Key Designs

1. **Unification of Discrete Diffusion and Continuous Flow (Proposition 3.1)**:

    - **Function**: Proves that the transition distributions of discrete diffusion processes can be modeled by continuous flows on the hypersphere.
    - **Mechanism**: The parameter space of categorical distributions (probability simplex $\Delta^{d-1}$) forms a statistical manifold $\mathcal{P}(\mathcal{X})$ under the Fisher-Rao metric, which is diffeomorphic to $\mathbb{S}^{d-1}_+$. Under this mapping, the categorical distribution $\text{Cat}(x_t; \bar{Q}_t x)$ induced by the discrete diffusion transition matrix $\bar{Q}_t$ can be exactly reproduced by the flow of the geodesic ODE $\frac{d\mathbf{Y}_t}{dt} = -\frac{d\log\kappa_t}{dt}\exp^{-1}_{\mathbf{Y}_t}(\mathbf{y}_1)$ on the hypersphere. In particular, setting $\mathbf{y}_1 = \mathbf{e}_m$ recovers masked diffusion, and $\mathbf{y}_1 = \sum \mathbf{e}_i/\sqrt{d}$ recovers uniform diffusion.
    - **Design Motivation**: Once this theoretical connection is established, discrete jumps can be "smoothed" into continuous trajectories, with intermediate states providing ongoing opportunities for correction.

2. **Simulation-Free Training via Radial Symmetry**:

    - **Function**: Exploits the radial symmetry of the hypersphere to derive a tractable approximation of the transition distribution, avoiding the need to simulate expensive SDEs during training.
    - **Mechanism**: The transition distribution of the $d$-dimensional bridge process is approximated by a Riemannian normal distribution $\mathcal{N}_{\mathbb{S}^{d-1}}(\boldsymbol{\mu}_t, \rho_t^2 \mathbf{I})$. The parameters $\alpha_t, \rho_t$ are derived from one-dimensional projection processes $z_t^T = \langle \mathbf{X}_t, \mathbf{e}_k \rangle$ and $z_t^0 = \langle \mathbf{X}_t, \mathbf{X}_0 \rangle$, requiring only the precomputation of moments of a one-dimensional SDE. The training objective uses a cross-entropy loss $\mathcal{L}^{CE}(\theta) = \mathbb{E}[-\log\langle p_\theta(\mathbf{X}_t, t), \mathbf{e}_k\rangle]$, consistent in form with the training objective of discrete diffusion.
    - **Design Motivation**: Directly simulating SDEs on high-dimensional hyperspheres is computationally prohibitive. Radial symmetry ensures that statistics are identical in all directions, so high-dimensional distribution parameters can be recovered from one-dimensional projections, achieving approximately a 50× speedup.

3. **Dimension Splitting**:

    - **Function**: Represents large-vocabulary tokens in base $b$, mapping from $\mathbb{S}^{d-1}$ to $(\mathbb{S}^b)^m$ (where $m = \lceil\log_b d\rceil$), reducing the dimensionality of each hypersphere.
    - **Mechanism**: Bridge processes on high-dimensional hyperspheres exhibit a sharp transition near the terminal time, which is difficult for neural networks to learn. Splitting the $d$-dimensional sphere into $m$ spheres of dimension $b$ yields smoother processes on each sphere. This works best in conjunction with a mixed path of masked and uniform diffusion (Eq. 9): $\lambda_t \mathbb{Q}_t^{mask} + (1-\lambda_t)\mathbb{Q}_t^{unif}$.
    - **Design Motivation**: Language model vocabularies are typically on the order of tens of thousands; training directly on $\mathbb{S}^{30000}$ is infeasible. Dimension splitting combined with mixed paths is the key technique enabling the framework to scale to practical vocabularies.

### Loss & Training

The cross-entropy loss is $\mathcal{L}^{CE}(\theta) = \mathbb{E}[\int_0^T -\log\langle p_\theta(\mathbf{X}_t, t), \mathbf{e}_k\rangle dt]$, combined with importance sampling $q(t)$ to focus training on difficult time steps. A geometric noise schedule $\sigma_t = \sigma_0^{T-t}\sigma_T^t$ (with $\sigma_0 < \sigma_T$) ensures asymptotic convergence. Sampling is performed via geodesic random walks: $\mathbf{X} \leftarrow \exp_{\mathbf{X}}(\eta_\theta \delta t + \sigma_t\sqrt{\delta t}\mathbf{w})$.

## Key Experimental Results

### Main Results

Text8 (character-level language modeling, BPC↓):

| Method | Type | BPC |
|--------|------|-----|
| Transformer AR | Autoregressive | 1.23 |
| ARDM | Order-agnostic AR | ≤1.43 |
| D3PM Absorb | Discrete Diffusion | ≤1.45 |
| SEDD Absorb | Discrete Diffusion | ≤1.39 |
| MDLM | Discrete Diffusion | ≤1.40 |
| MD4 | Discrete Diffusion | ≤1.37 |
| BFN | Continuous Diffusion | ≤1.41 |
| **RDLM (Ours)** | **Continuous Diffusion** | **≤1.32** |

LM1B (PPL↓):

| Method | Parameters | PPL |
|--------|------------|-----|
| MDLM | 110M | ≤27.04 |
| Diffusion-LM | 80M | ≤118.62 |
| **RDLM (Ours)** | 110M | **≤28.44** |

CIFAR-10 (pixel-level image modeling, BPD↓):

| Method | BPD |
|--------|-----|
| MD4 | ≤2.78 |
| Sparse Transformer | 2.80 |
| **RDLM (Ours)** | **≤2.73** |

### Ablation Study

| Configuration | Text8 BPC | Note |
|---------------|-----------|------|
| MSE loss | Higher | Slow convergence |
| Cross-entropy loss | Lower | Fast convergence, better performance |
| Without importance sampling | Higher | Difficult time steps undertrained |
| With importance sampling | Lowest | Concentrated training on hard intervals |
| Without dimension splitting (LM1B) | Failed | High-dimensional sphere training infeasible |
| Dimension splitting + mixed path | Optimal | Essential for large vocabularies |

### Key Findings

- RDLM achieves BPC=1.32 on Text8, the best result among all diffusion models (discrete and continuous).
- It also surpasses continuous-domain autoregressive models on CIFAR-10 pixel modeling, demonstrating cross-modal potential.
- The MMD distance of the Riemannian normal approximation approaches zero in high dimensions, confirming that approximation quality improves with dimensionality.
- RDLM achieves state-of-the-art performance on DNA sequence design (MSE=0.027), validating the generality of the framework.

## Highlights & Insights

- **Theoretical Elegance**: Proposition 3.1 establishes an exact correspondence between discrete diffusion and continuous flows on the statistical manifold, providing a mathematical foundation for unifying both classes of methods.
- **Advantages of Continuity**: Discrete jumps are irreversible, whereas continuous trajectories allow for gradual correction — this is precisely the core reason diffusion models succeed in continuous domains.
- **Simulation-Free Training**: Radial symmetry reduces the $d$-dimensional problem to one-dimensional precomputation, yielding a 50× training speedup.
- **Mixed-Path Innovation**: A time-varying mixture of masked and uniform diffusion paths generalizes discrete flow matching and state-dependent schedules.

## Limitations & Future Work

- On LM1B, PPL=28.44 still lags behind MDLM (27.04) and autoregressive models (22.32); the gap on large-scale language tasks remains to be closed.
- Dimension splitting introduces a hyperparameter for base selection, and base encoding may disrupt the semantic adjacency relations between tokens.
- Sampling requires multi-step SDE simulation (geodesic random walk), which is slower than parallel decoding in discrete diffusion.
- Conditional and controllable generation have not been explored.

## Related Work & Insights

- **D3PM / SEDD / MDLM**: Discrete diffusion baselines; RDLM demonstrates that they are special cases of the continuous framework.
- **Fisher-Flow / CatFlow**: Flow matching methods on statistical manifolds, but restricted to short sequences and small vocabularies; RDLM overcomes this limitation via dimension splitting.
- **Dirichlet Diffusion (DDSM)**: Uses the Dirichlet distribution as a prior on the probability simplex but does not exploit Fisher-Rao geometry.
- **Insights**: The statistical manifold provides a natural unified framework for discrete and continuous data; the dimension-splitting idea may inspire other high-dimensional structured generation problems.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Outstanding theoretical contributions — discrete-continuous unification, simulation-free training via radial symmetry, dimension splitting.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated across three modalities (text/image/DNA) with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are clear, though notation is occasionally heavy.
- Value: ⭐⭐⭐⭐ Opens a unified geometric perspective for diffusion-based modeling of discrete data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Generative Audio Language Modeling with Continuous-Valued Tokens and Masked Next-Token Prediction](../../ICML2025/image_generation/generative_audio_language_modeling_with_continuous-valued_tokens_and_masked_next.md)
- [\[NeurIPS 2025\] Continuous Uniqueness and Novelty Metrics for Generative Modeling of Inorganic Crystals](continuous_uniqueness_and_novelty_metrics_for_generative_modeling_of_inorganic_c.md)
- [\[ECCV 2024\] NL2Contact: Natural Language Guided 3D Hand-Object Contact Modeling with Diffusion Model](../../ECCV2024/image_generation/nl2contact_natural_language_guided_3d_hand-object_contact_modeling_with_diffusio.md)
- [\[NeurIPS 2025\] FineGRAIN: Evaluating Failure Modes of Text-to-Image Models with Vision Language Model Judges](finegrain_evaluating_failure_modes_of_text-to-image_models_with_vision_language_.md)
- [\[NeurIPS 2025\] Non-Markovian Discrete Diffusion with Causal Language Models](non-markovian_discrete_diffusion_with_causal_language_models.md)

</div>

<!-- RELATED:END -->
