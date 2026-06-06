---
title: >-
  [Paper Note] Identifiable Equivariant Networks are Layerwise Equivariant
description: >-
  [ICML2026][Equivariance] This paper proves within an architecture-agnostic abstract framework that as long as parameters satisfy "weak identifiability…
tags:
  - "ICML2026"
  - "Equivariance"
  - "Identifiability"
  - "Submodel"
  - "Latent Space Symmetry"
  - "MLP"
  - "Multi-head Attention"
date: 2026-05-08
content_hash: db96356a36a34928
---

# Identifiable Equivariant Networks are Layerwise Equivariant

**Conference**: ICML2026  
**arXiv**: [2601.21645](https://arxiv.org/abs/2601.21645)  
**Code**: None  
**Area**: Equivariant Neural Networks / Geometric Deep Learning Theory  
**Keywords**: Equivariance, Identifiability, Submodel, Latent Space Symmetry, MLP, Multi-head Attention

## TL;DR
This paper proves within an architecture-agnostic abstract framework that as long as parameters satisfy "weak identifiability," an end-to-end $G$-equivariant deep network must possess an equivalent parameterization where every layer is equivariant to some latent group action. This provides a theoretical explanation for the long-observed phenomenon in experiments where "end-to-end equivariance spontaneously collapses into layerwise equivariance."

## Background & Motivation

**Background**: The dominant paradigm in geometric deep learning is "layerwise equivariance"—given a symmetry group $G$ acting on inputs/outputs, linear operators are constructed such that each layer is $G$-equivariant (e.g., translation equivariance in CNNs, permutation equivariance in GNNs, rotation equivariance in equivariant transformers). This approach has been repeatedly validated in vision, graphs, and molecular modeling.

**Limitations of Prior Work**: However, is "layerwise equivariance" the *only* way to construct end-to-end equivariant networks? An intuitive phenomenon has been observed in experiments: even when fitting symmetric data using a standard MLP **without any equivariance constraints** (or using symmetry-based data augmentation), the weights of intermediate layers spontaneously exhibit equivariant structures after training (Lenc & Vedaldi 2015, Gruver et al. 2023, Bökman & Kahl 2023). A unified theoretical explanation for when and why this "spontaneous equivariance" occurs has been lacking.

**Key Challenge**: There exists a gap between the "global constraint" of end-to-end equivariance and the "local structure" of layerwise equivariance. While the latter obviously implies the former, whether the converse holds and under what conditions has previously only been addressed by scattered results for shallow ReLU MLPs (Agrawal & Ostrowski 2022, Marchetti et al. 2024).

**Goal**: To answer the question "Does end-to-end equivariance $\Rightarrow$ layerwise equivariance?" at a level as abstract as possible and clearly outline the necessary sufficient conditions.

**Key Insight**: The authors adopt the language of "parameterized functions" from category theory, abstracting each layer as $f_i: V_{i-1} \times \Theta_i \to V_i$ and the symmetry as an arbitrary group action $K_i$ on the latent space $V_i$. They use "parameter identifiability" (where parameters corresponding to the same function can only differ by a symmetry transformation) as the master key to unlock the problem.

**Core Idea**: Using "weak identifiability" as the sole assumption, the paper proves that $G$-equivariance automatically propagates inward layer by layer. The group action at the input is transmitted through the identifiability symmetry groups $K_i$ of the latent spaces all the way to the output, forcing each layer to become equivariant.

## Method

This paper is a theoretical work rather than an algorithmic one; the "Method" refers to its proof framework and abstract formalization.

### Overall Architecture

The authors redefine a deep model as a quadruple $(V_i, \Theta_i, f_i, K_i)$: where $V_i$ are sets of latent spaces, $\Theta_i$ are parameter spaces, $f_i: V_{i-1}\times\Theta_i \to V_i$ are layer mappings, and $K_i$ are symmetry groups acting on $V_i$. The end-to-end function is defined as $f(\bullet; \theta) = f_L(\bullet; \theta_L)\circ\cdots\circ f_1(\bullet; \theta_1)$. At this abstract level, MLPs, attention networks, and convolutional networks are merely different concrete instances of the same definition. The proof strategy involves formally defining "submodels," characterizing "degenerate parameters" using them, and completing the inductive derivation of layerwise equivariance through "weak identifiability + adjunction property."

### Key Designs

1. **Unified Formalization of Submodels and Redundant/Inactive Neurons**:

    - **Function**: Transitions degenerate cases in MLPs—such as "certain neurons not participating in forward computation" or "two neurons having identical functions"—to a category-theoretic level independent of specific architectures.
    - **Mechanism**: A submodel consists of $(\widetilde V_i, \widetilde\Theta_i, \widetilde f_i)$ plus a set of linking maps $\alpha_i: \widetilde V_i \to V_i$, $\alpha_i^*: V_i\to\widetilde V_i$ (satisfying $\alpha_i^*\circ\alpha_i = \mathrm{Id}$), and $\beta_i: \widetilde\Theta_i\to\Theta_i$, requiring a commutative diagram to hold. For MLPs taking linear $\alpha_i$, a submodel precisely corresponds to a smaller network obtained by "removing inactive neurons / merging redundant neurons."
    - **Design Motivation**: Direct discussion of "parameter uniqueness" is undermined by degenerate cases (where one function corresponds to infinite equivalent parameters). By explaining all degeneracies as "originating from a smaller submodel," identifiability can be required specifically for the "non-degenerate" portion, bypassing issues caused by pathological activations like ReLU.

2. **Weak Identifiability**:

    - **Function**: Standardizes prior identifiability results for specific activations (e.g., Sussmann 1992, Fefferman 1994, Vlačić & Bölcskei 2022) into a minimal assumption usable by this framework.
    - **Mechanism**: A parameter $\theta$ is weakly identifiable if there exists an identifiable parameter $\widetilde\theta$ on some submodel such that $\theta_i = \beta_i(\widetilde\theta_i)$. Here, "identifiable" means: if $f(\bullet;\theta) = f(\bullet;\theta')$, then there exists a unique sequence $k_i\in K_i$ such that $f_i(x;\theta_i') = k_i\cdot f_i(k_{i-1}^{-1}\cdot x;\theta_i)$.
    - **Design Motivation**: Many activations (Tanh, polynomial, sigmoidal families) have been proven to have identifiable parameters after removing degeneracies; this framework uses these results as "inputs." While ReLU remains an open problem, the authors explicitly state that once weak identifiability for ReLU is proven, the conclusions of this paper will immediately apply.

3. **Adjunction Property and Layerwise Symmetrization**:

    - **Function**: Translates the "global constraint" of end-to-end equivariance into local conditions that can be derived layer by layer.
    - **Mechanism**: Requires $G$ to also act on the parameters of the first and last layers, satisfying $f_1(g\cdot x_0; \theta_1) = f_1(x_0; g^{-1}\cdot\theta_1)$ and $g\cdot f_L(x_{L-1}; \theta_L) = f_L(x_{L-1}; g\cdot\theta_L)$. This allows "moving" the $g$ action from the input side to the first-layer parameters, then using weak identifiability to transform it into a $k_1\in K_1$ action on $V_1$, and proceeding by induction.
    - **Design Motivation**: This adjunction condition is not arbitrary—it is naturally satisfied by MLPs (when $G$ acts linearly on input/output spaces) and attention networks with positional encodings. For attention without positional encodings or CNNs (where layers "absorb" specific group actions), the authors provide a generalized adjunction $g^{-1}\cdot f_i(g\cdot x, \theta) = f_i(x, g^{-1}\cdot\theta)$, and the proof still holds. Cases like Deep Sets and equivariant GNNs, where layers are themselves equivariant linear operators, break the adjunction condition, and the authors honestly state the theorem does not apply here.

### Main Theorem and Proof Sketch

**Theorem 4.1**: If $\theta$ is weakly identifiable and the model is end-to-end $G$-equivariant at $\theta$, then there exist actions of $G$ on each $V_i$ ($i=1,\dots,L-1$) such that each $f_i$ is $G$-equivariant at $\theta_i$.

The inductive skeleton of the proof is: for any $g\in G$, use the adjunction property to transform the "input-side $g$ action" of the first layer into a "parameter-side $g^{-1}$ action," resulting in two parameters that are end-to-end equivalent. Weak identifiability then yields a unique latent space symmetry $k_i(g)\in K_i$. Finally, verify that $g\mapsto k_i(g)$ is a group homomorphsm and define it as the $G$-action on $V_i$. Remark 4.2 notes that this latent space action is actually given by a homomorphism $G\to K_i$, meaning the latent group action must "factor" through the latent space’s native identifiability symmetry group.

## Key Experimental Results

Theoretical results are qualitatively and quantitatively validated using a small MLP on CIFAR-10 (depth 4, Tanh or GELU for the first layer, GELU for subsequent layers) and a single-layer multi-head attention network. A mirror-equivariance soft loss (rather than a hard constraint) is added during the second half of training to observe whether latent spaces spontaneously form equivariant structures.

### Main Results: Relative Equivariance Error of the First/Second Layers

Linear transformations $A_i$ are estimated via least squares such that $A_i f_i(x)\approx f_i(\mathrm{mirror}[x])$. The median of $|f_i(\mathrm{mirror}[y]) - A_i f_i(y)|/|f_i(y)|$ is calculated over 100,000 independent noise samples:

| Task | Layer | Mirror Direction | Tanh | GELU |
|------|-----|---------|------|------|
| Autoencoder | $f_1$ | Left-Right | **0.029** | 0.40 |
| Autoencoder | $f_2$ | Left-Right | **0.022** | 0.11 |
| Autoencoder | $f_1$ | Up-Down (Ref) | 0.15 | 0.49 |
| Classifier | $f_1$ | Left-Right | **0.077** | 0.19 |
| Classifier | $f_2$ | Left-Right | **0.054** | 0.064 |
| Classifier | $f_1$ | Up-Down (Ref) | 0.48 | 0.56 |

The conclusion is clear: even though the training objective only requires *end-to-end* approximate mirror equivariance, the first layer of the Tanh network spontaneously approximates linear equivariance (error 0.029-0.077, corresponding to a signed permutation matrix, which is exactly the intertwiner group $\{\pm 1\}^{d_i}\rtimes S_{d_i}$ of Tanh). This is not the case for the GELU network because GELU satisfies $\sigma(x)-\sigma(-x)=x$, allowing it to "bypass" the nonlinearity; a combination of two layers $f_2$ is needed to recover linear equivariance.

### Ablation Study: Activation Function vs. Latent Space Action Structure

| Activation | Latent Action $K_i$ | Weak Identifiability | $f_1$ Linearly Equivariant? |
|---------|---------------------|------------|-------------------|
| Tanh | $\{\pm 1\}^{d_i}\rtimes S_{d_i}$ (Signed Permutation) | Proven | Yes (Error ≈ 0.03) |
| Large Power $t^m$ | $(\mathbb R^\times)^{d_i}\rtimes S_{d_i}$ (Monomial) | Proven | Supported by Theory |
| GELU | Permutation + satisfies $\sigma(x)-\sigma(-x)=x$ | Difficult (ReLU-like pathology) | No (Error 0.4) |
| ReLU | Positive Scaling + Permutation | Open Problem | Only for shallow + reparameterized cases |

### Key Findings
- **Correspondence between Theory and Phenomena**: The 64 filters learned by the first layer of the Tanh network visually exhibit three types of paired structures—self-symmetric, mirrored, and negated (Pink/Light Blue/Gold)—which is exactly the geometric manifestation of the signed permutation group. The emergence of many "negated copies" in GELU stems from the $\sigma(x)-\sigma(-x)=x$ bypass effect, which indicates that the action on $V_1$ is not a simple permutation.
- **Equivariant Structure of Attention is Head Permutation**: In a single-layer multi-head attention autoencoder for CIFAR-10, when the input image is flipped left-right, the attention maps of most heads flip accordingly, but heads 1 and 5 switch positions—this is the visual manifestation of the $S_{h_i}$ part of $K_i = \mathrm{GL}(d_i)^{h_i}\rtimes S_{h_i}$.
- **Attention with Positional Encodings satisfies Standard Adjunction**: Therefore, even for $G\subseteq S_n$ (token permutation) symmetries, identifiability implies the equivariance of the positional encodings themselves, providing an explanation for "why positional encodings spontaneously learn structured geometric patterns."

## Highlights & Insights
- **First architecture-agnostic "End-to-End $\Rightarrow$ Layerwise" theorem**: Previous results only covered shallow ReLU MLPs or layers with specific scaling symmetries. This paper uses category theory + identifiability to complete the entire proof "once and for all," making MLPs/attention/polynomial CNNs mere corollaries.
- **Introduction of "Weak Identifiability" is a key technique**: Directly requiring identifiability is broken by degenerate parameters. By using "submodels" to absorb all degenerate cases, the problem is reduced to checking existing identifiability results for each known architecture, cleanly decoupling the abstract proof from specific architectures.
- **Explanation for why symmetry emerges without equivariant constraints**: Previously, one could only say it was "observed empirically." Now, one can say "as long as data/augmentation forces the network to be end-to-end equivariant and parameter identifiability holds, layerwise equivariance is a mathematical necessity."
- **Identification of ReLU as an open problem**: The theorem's structure makes the "study of layerwise equivariance in ReLU networks" completely equivalent to the "study of identifiability," turning a hard problem in deep learning theory into a problem of pure algebraic geometry (due to the parallel structure between ReLU MLPs and polynomial models).

## Limitations & Future Work
- **Theoretical assumption is "exact" equivariance**, whereas real-world networks are only approximately equivariant. The authors acknowledge that the remedy relies on qualitative experimental demonstrations (CIFAR-10 mirror experiments) and lacks a quantitative "approximate end-to-end $\Rightarrow$ approximate layerwise" stability bound.
- **Skip connections are not covered**: Residual blocks increase inter-layer dependencies and reduce identifiability (as setting residual channels to zero can equivalently change effective depth). Strictly speaking, the theorem cannot be directly applied to modern mainstream architectures like ResNet/Transformer.
- **Does not specify what the latent action is**: The theorem only guarantees "an action on some $K_i$ exists such that each layer is equivariant," but it does not specify how the optimal action should be chosen or which one the training process will actually converge to.
- **Excludes training dynamics**: It is an existence result rather than a constructive one—it tells you an "equivalent parameterization exists" but not whether a symmetry-free MLP initialized randomly will necessarily converge to such a parameterization under SGD.
- **Does not apply to Deep Sets / Equivariant GNNs**: In these architectures, the layers themselves "absorb" specific group actions (rather than acting on the parameter space), breaking the adjunction property. The theorem fails here, and the authors explicitly list this as a future direction.

## Related Work & Insights
- **vs. Agrawal & Ostrowski (2022)**: They focused on shallow ReLU MLPs using specific reparameterizations to recover identifiability. This paper generalizes the "identifiability $\Rightarrow$ layerwise equivariance" reduction to arbitrary architectures and groups, making the shallow ReLU result a corollary.
- **vs. Marchetti et al. (2024)**: They only proved equivariant structures for the first layer of models with neuron-wise scaling symmetry. This paper proves it for any layer at any depth.
- **vs. Xie & Smidt (2025) / Brehmer et al. (2024)**: They empirically debated whether "hard equivariance constraints vs. data augmentation" is superior. This paper provides a theoretical conclusion: provided identifiability holds, the *function classes* provided by both routes are identical (a network trained with augmentation has an equivalent layerwise equivariant parameterization), shifting the debate to "training dynamics / optimization efficiency."
- **vs. Categorical Deep Learning (Gavranović et al. 2024)**: This paper partially adopts categorical formalisms (Remark 3.3) and represents a substantive application of this route to equivariance, serving as a template for future uses of category theory in deep learning theory proofs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First unified theorem for "end-to-end $\Rightarrow$ layerwise" equivariance across architectures.
- Experimental Thoroughness: ⭐⭐⭐ Theory-focused; qualitative + one quantitative table. Sufficient for validation on CIFAR-10 small networks but not exhaustive.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear rhythm between abstract formalization and concrete examples (MLP/attention).
- Value: ⭐⭐⭐⭐⭐ Provides a theoretical foundation for "spontaneous equivariance" and reduces ReLU layerwise equivariance to a known open problem.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks](how_the_optimizer_shapes_learned_solutions_in_equivariant_neural_networks.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)
- [\[ICLR 2026\] Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges](../../ICLR2026/others/latent_equivariant_operators_for_robust_object_recognition_promises_and_challeng.md)
- [\[ICML 2026\] Singular Bayesian Neural Networks](singular_bayesian_neural_networks.md)

</div>

<!-- RELATED:END -->
