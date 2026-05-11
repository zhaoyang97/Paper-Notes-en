---
title: >-
  [Paper Note] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data
description: >-
  [ICLR 2026][Interpretability][Interpretable ML] Inspired by the utility maximization paradigm in behavioral science, this paper proposes the Behavior Learning (BL) framework…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "Interpretable ML"
  - "Utility Maximization"
  - "Hierarchical Optimization Structure"
  - "Identifiability"
  - "Gibbs Distribution"
date: 2026-05-08
content_hash: 9534f1a8244e242b
---

# Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data

**Conference**: ICLR 2026
**arXiv**: [2602.20152](https://arxiv.org/abs/2602.20152)
**Code**: [https://github.com/MoonYLiang/Behavior-Learning](https://github.com/MoonYLiang/Behavior-Learning) (pip install blnetwork)
**Area**: Interpretability
**Keywords**: Interpretable ML, Utility Maximization, Hierarchical Optimization Structure, Identifiability, Gibbs Distribution

## TL;DR
Inspired by the utility maximization paradigm in behavioral science, this paper proposes the Behavior Learning (BL) framework, which models data as a Gibbs distribution induced by a hierarchical composition of interpretable, modular utility maximization problems (UMPs), achieving a unified balance among predictive performance, intrinsic interpretability, and parameter identifiability.

## Background & Motivation

**Background**: Interpretable machine learning has long faced a performance–interpretability trade-off — deep neural networks offer strong predictive power but lack transparency, while linear models and decision trees are transparent but fail to capture complex nonlinear patterns. Existing methods such as EBMs and NAMs partially alleviate this tension but remain insufficient for scientific modeling scenarios.

**Limitations of Prior Work**:
- **Lack of alignment with scientific theory**: Most interpretable methods are extensions of existing ML approaches rather than designs grounded in scientific principles (e.g., optimization problems, differential equations), making it difficult to extract scientific knowledge from the learned models.
- **Non-unique explanations**: The vast majority of models lack identifiability — the same predictive behavior may correspond to multiple distinct sets of parameters or explanations, providing no guarantee that the recovered "scientific knowledge" is genuine.

**Key Challenge**: How can one design a machine learning framework that simultaneously achieves strong predictive capability, intrinsic interpretability, and parameter identifiability?

**Key Insight**: The paper departs from the core paradigm of behavioral science — utility maximization problems (UMPs). UMPs assume that human behavior arises from optimization processes that maximize utility under constraints, and since any optimization problem can be equivalently rewritten as a UMP (Theorem 2.2), a framework built on UMPs is general-purpose.

**Core Idea**: The model is parameterized as a "composite utility function" formed by hierarchically composing multiple interpretable UMP modules. The conditional distribution of data is modeled via a Gibbs distribution, such that each module can be interpreted as a symbolic optimization problem.

## Method

### Overall Architecture
BL takes features $\mathbf{x}$ and responses $\mathbf{y}$ as input and outputs a conditional probability $p(\mathbf{y}|\mathbf{x})$. The core mechanism is to construct a "composite utility function" $\mathrm{BL}(\mathbf{x}, \mathbf{y})$, and then induce the data distribution via the Gibbs distribution $p_\tau(\mathbf{y}|\mathbf{x}) \propto \exp(\mathrm{BL}(\mathbf{x},\mathbf{y})/\tau)$. As the temperature $\tau \to 0$, the model degenerates into a deterministic optimal response.

### Key Designs

1. **Base Module $\mathcal{B}(\mathbf{x}, \mathbf{y})$ (UMP Block)**:

    - **Function**: Each block represents a penalty-form reformulation of a utility maximization problem.
    - **Mechanism**: Based on Theorem 2.1 (locally exact penalty reconstruction), the constrained UMP is equivalently transformed into an unconstrained form: $\mathcal{B} = \lambda_0^\top \tanh(\mathbf{p}_u) - \lambda_1^\top \mathrm{ReLU}(\mathbf{p}_c) - \lambda_2^\top |\mathbf{p}_t|$, where $\mathbf{p}_u, \mathbf{p}_c, \mathbf{p}_t$ are polynomial feature mappings. The $\tanh$ term represents diminishing marginal utility preferences; $\mathrm{ReLU}$ penalizes inequality constraint violations (budgets); $|\cdot|$ captures equality constraint deviations (beliefs).
    - **Design Motivation**: Each block can be written as a symbolic UMP, achieving a level of transparency comparable to linear regression.

2. **Three Architectural Variants**:

    - **BL(Single)**: A single $\mathcal{B}$ block with the strongest interpretability, directly corresponding to one UMP.
    - **BL(Shallow)**: A 1–2 layer shallow composition where outputs of multiple parallel blocks are fed into a subsequent layer and transformed via an affine mapping.
    - **BL(Deep)**: More than two layers of hierarchical composition, modeling layered optimization structures (e.g., micro-preferences → macro trade-offs → global decisions). Unified formulation: $\mathrm{BL}(\mathbf{x},\mathbf{y}) = \mathbf{W}_L \cdot \mathbb{B}_L(\cdots\mathbb{B}_2(\mathbb{B}_1(\mathbf{x},\mathbf{y}))\cdots)$.

3. **Identifiable Variant IBL**:

    - **Function**: Imposes stricter structural constraints on top of BL (strictly increasing $\phi^{\rm id}$; symmetric and strictly increasing $\psi^{\rm id}$) to ensure parameter identifiability.
    - **Mechanism**: Instantiated as $\mathcal{B}^{\rm id} = \lambda_0^\top \tanh(\mathbf{p}_u) - \lambda_1^\top \mathrm{softplus}(\mathbf{p}_c) - \lambda_2^\top (\mathbf{p}_t)^{\odot 2}$, replacing ReLU with softplus (smooth and monotone) and absolute value with the square function (symmetric and smooth).
    - **Design Motivation**: Guarantees unique parameter determination on the quotient space, lending scientific credibility to the model's explanations.

### Loss & Training
The training objective handles mixed discrete and continuous responses:
- **Discrete component**: Standard cross-entropy loss.
- **Continuous component**: Denoising score matching, treating BL as an energy function.
- **Joint objective**: $\mathcal{L}(\theta) = \gamma_d \mathbb{E}[-\log p_\tau(\mathbf{y}^{\rm disc}|\mathbf{x})] + \gamma_c \mathbb{E}\|\nabla_{\tilde{\mathbf{y}}} \log p_\tau - \sigma^{-2}(\tilde{\mathbf{y}} - \mathbf{y})\|^2$

### Theoretical Guarantees
- **Universal Approximation** (Theorem 2.3): BL/IBL can approximate any continuous conditional distribution (in the KL divergence sense) given sufficient capacity.
- **Identifiability** (Theorems 2.4–2.5): IBL parameters are uniquely determined on the quotient space.
- **Consistency / Universal Consistency** (Theorems 2.6–2.7): The IBL estimator converges to the true parameters as the sample size grows (under correct specification) or to the true distribution (even under model misspecification).
- **Asymptotic Normality and Efficiency**: IBL parameter estimates are asymptotically normal and achieve the efficient information bound.

## Key Experimental Results

### Main Results: Predictive Performance across 10 Datasets
Comparisons against 10 baselines (MLP, XGBoost, LightGBM, decision trees, Bayesian methods, etc.) on 10 datasets spanning varying sample sizes, feature dimensions, and scientific domains.

| Model | F1-Macro Rank | AUC Rank | Interpretability |
|-------|--------------|----------|-----------------|
| BL(Shallow) | 2nd–3rd | Top tier | ✓ Intrinsically interpretable |
| BL(Single) | 3rd–4th | Top tier | ✓ Strongest interpretability |
| MLP | Below BL(Shallow) | Top tier | ✗ |
| XGBoost | 1st–2nd | Top tier | ✗ |
| Decision Tree | Last | Last | ✓ |

### Ablation Study: BL vs. E-MLP on High-Dimensional Inputs

| Dataset | Model | Accuracy (%) | OOD AUROC (%) |
|---------|-------|-------------|--------------|
| MNIST | BL(d=3) | 97.93 | **92.92** |
| MNIST | E-MLP(d=3) | 98.14 | 87.76 |
| Fashion-MNIST | BL(d=2) | 88.96 | **89.87** |
| Fashion-MNIST | E-MLP(d=2) | 88.88 | 84.61 |
| AG News | BL(d=1) | **89.52** | **66.18** |
| AG News | E-MLP(d=1) | 88.74 | 59.24 |

Under comparable parameter counts, BL matches E-MLP on in-distribution accuracy while substantially outperforming it on OOD detection, and achieves better calibration on ECE/NLL metrics (e.g., Fashion-MNIST NLL: BL=0.36 vs. E-MLP=0.74).

### Key Findings
- BL(Shallow) surpasses MLP, demonstrating that interpretability need not come at the cost of performance.
- BL exhibits a clear advantage in OOD detection, indicating that modeling optimization structure aids uncertainty estimation.
- Preference patterns and trade-off structures recovered by BL on the Boston Housing dataset are consistent with findings in the economics literature (Table 11), suggesting that BL can reconstruct scientific knowledge from data.
- The BL(Deep) [5,3,1] architecture recovers a hierarchical structure of "5 micro-preferences → 3 macro trade-offs → 1 representative buyer."

## Highlights & Insights
- **Elegant integration of scientific theory and ML**: By grounding the framework in the UMP paradigm from behavioral science, every network module carries a clear economic/optimization interpretation (utility, constraints, beliefs), rather than being explained post hoc. This "theory-first" design paradigm for interpretable ML is highly original.
- **Automatic discovery of hierarchical optimization structures**: BL(Deep) not only achieves accurate predictions but also automatically discovers hierarchical optimization structures embedded in data. On Boston Housing, the coarse-graining process from raw features to micro-preferences to macro trade-offs resonates with the renormalization group idea in statistical physics.
- **Identifiability guarantees**: IBL achieves parameter identifiability by imposing smooth monotonicity constraints on the penalty functions, without sacrificing universal approximation capability — a rare property in interpretable ML. This means the "explanations" learned by the model are mathematically unique.
- **Shifting the Pareto frontier**: BL achieves comparable predictive performance, intrinsic interpretability, and superior OOD detection relative to E-MLP with similar parameter counts and runtime, pushing forward the performance–interpretability Pareto frontier.

## Limitations & Future Work
- **Scalability of polynomial feature mappings**: BL(Single) uses quadratic polynomial bases, causing feature dimensionality to grow explosively in high-dimensional spaces. Deeper variants mitigate this via affine mappings but sacrifice the granularity of symbolic interpretability.
- **Scope of the UMP assumption**: Not all data-generating processes can be characterized by utility maximization (e.g., certain physical processes may be better described by differential equations). Although any optimization problem is theoretically equivalent to a UMP, such equivalences may be far from intuitive in practice.
- **Computational efficiency**: Training the continuous response component via denoising score matching incurs slightly higher training time than standard MLP training.
- **Interpretive complexity of deep BL**: As depth increases, although a hierarchical optimization structure interpretation exists in theory, understanding multi-layer nested UMP compositions in practice is non-trivial and requires domain expertise.

## Related Work & Insights
- **vs. EBM (Nori et al., 2019)**: EBMs are based on generalized additive models and are interpretable per feature but do not model feature interactions. BL naturally captures interactions via polynomial mappings and hierarchical composition, while offering scientific interpretations grounded in optimization structures.
- **vs. Energy-Based Models (LeCun et al., 2006)**: BL can be viewed as a structured energy-based model, but each BL module carries explicit optimization semantics (utility/constraints/beliefs), whereas traditional energy-based models are black-box.
- **vs. Inverse Optimization (Ahuja & Orlin, 2001)**: Classical inverse optimization typically assumes the form of the optimization problem is known and only the parameters are unknown. BL simultaneously learns both the structure and the parameters of the optimization from data, and supports hierarchical composition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The idea of constructing an interpretable ML framework grounded in the UMP paradigm from behavioral science is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Coverage is comprehensive across 10 benchmark datasets, case studies, and high-dimensional experiments, though validation on large-scale real-world scientific discovery tasks is lacking.
- Writing Quality: ⭐⭐⭐⭐⭐ — Theoretically rigorous, clearly structured, and well-illustrated.
- Value: ⭐⭐⭐⭐ — Offers a new paradigm for interpretable ML; its impact on practical scientific applications remains to be seen.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LORE: Jointly Learning the Intrinsic Dimensionality and Relative Similarity Structure from Ordinal Data](lore_jointly_learning_the_intrinsic_dimensionality_and_relative_similarity_struc.md)
- [\[AAAI 2026\] Data Whitening Improves Sparse Autoencoder Learning](../../AAAI2026/interpretability/data_whitening_improves_sparse_autoencoder_learning.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ICLR 2026\] Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer](towards_understanding_subliminal_learning_when_and_how_hidden_biases_transfer.md)

</div>

<!-- RELATED:END -->
