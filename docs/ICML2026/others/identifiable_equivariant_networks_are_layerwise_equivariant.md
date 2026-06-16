---
title: >-
  [Paper Note] Identifiable Equivariant Networks are Layerwise Equivariant
description: >-
  [ICML 2026][Others][MLP] This paper proves within an architecture-agnostic abstract framework that as long as parameters satisfy "weak identifiability," an end-to-end $G$-equivariant deep network must possess an equivalent parameterization where each layer is equivariant to some latent group action. This provides a theoretical explanation for
tags:
  - ICML 2026
  - Others
  - MLP
date: 2026-05-08
content_hash: 6f2e2632d3a13423
---
# Identifiable Equivariant Networks are Layerwise Equivariant

**Conference**: ICML 2026  
**arXiv**: [2601.21645](https://arxiv.org/abs/2601.21645)  
**Code**: None  
**Area**: Equivariant Neural Networks / Geometric Deep Learning Theory  
**Keywords**: Equivariance, Identifiability, Submodel, Latent space symmetry, MLP, Multi-head attention

## TL;DR
This paper proves within an architecture-agnostic abstract framework that as long as parameters satisfy "weak identifiability," an end-to-end $G$-equivariant deep network must possess an equivalent parameterization where each layer is equivariant to some latent group action. This provides a theoretical explanation for the long-observed experimental phenomenon where "end-to-end equivariance spontaneously collapses into layerwise equivariance."

## Background & Motivation

**Background**: The mainstream paradigm of geometric deep learning is "layerwise equivariance"—given a symmetry group $G$ acting on inputs/outputs, one constructs linear operators for each layer that are $G$-equivariant (e.g., translation equivariance in CNNs, permutation equivariance in GNNs, rotation equivariance in equivariant transformers). This approach has been repeatedly validated in vision, graphs, and molecular modeling.

**Limitations of Prior Work**: Is "layerwise equivariance" the *only* way to construct end-to-end equivariant networks? Experiments have observed a counter-intuitive phenomenon: even when using a vanilla MLP **without any equivariance constraints** to fit symmetric data (or with symmetry-based data augmentation), the weights of middle layers spontaneously exhibit equivariant structures after training (Lenc & Vedaldi 2015, Gruver et al. 2023, Bökman & Kahl 2023). A unified theoretical explanation for when and why this "spontaneous equivariance" appears is lacking.

**Key Challenge**: A gap exists between the "global constraint" of end-to-end equivariance and the "local structure" of layerwise equivariance—the latter clearly implies the former, but whether the converse holds, and under what conditions, has previously only been addressed by scattered results for shallow ReLU MLPs (Agrawal & Ostrowski 2022, Marchetti et al. 2024).

**Goal**: To answer the question "Does end-to-end equivariance $\Rightarrow$ layerwise equivariance?" at the most abstract level possible and clarify the sufficient conditions required.

**Key Insight**: The authors adopt the language of "parameterized functions" from category theory, abstracting each layer as $f_i: V_{i-1} \times \Theta_i \to V_i$, and symmetries as actions of an arbitrary group $K_i$ on the latent space $V_i$. They then use "parameter identifiability" (where parameters corresponding to the same function only differ by a symmetry transformation) as the key to unlock the problem.

**Core Idea**: Using "weak identifiability" as the sole assumption, the paper proves that $G$-equivariance automatically propagates inward through the layers—the group action at the input passes through the identification symmetry groups $K_i$ of the latent spaces all the way to the output, forcing each layer to be equivariant.

## Method

This is a theoretical paper rather than an algorithmic one; "Method" refers to its proof framework and abstract formalization.

### Overall Architecture

The authors redefine a deep model as a quadruple $(V_i, \Theta_i, f_i, K_i)$: where $V_i$ is the set of latent spaces, $\Theta_i$ is the parameter space, $f_i: V_{i-1}\times\Theta_i \to V_i$ is the layer mapping, and $K_i$ is the symmetry group acting on $V_i$. The end-to-end function is defined as $f(\bullet; \theta) = f_L(\bullet; \theta_L)\circ\cdots\circ f_1(\bullet; \theta_1)$. At this abstract level, MLPs, attention networks, and convolutional networks are merely specific instances of the same definition. The proof strategy involves formally defining "submodels" to characterize "degenerate parameters," and then using "weak identifiability + adjunction property" to complete an inductive derivation of layerwise equivariance.

### Key Designs

**1. Submodel: Lifting degeneracies like redundant/inactive neurons to an architecture-agnostic unified form**  
Discussing "parameter uniqueness" inevitably encounters a problem: in an MLP, some neurons might not participate in the forward pass, or two neurons might be functionally identical, leading to infinitely many equivalent parameters and breaking identifiability. The paper’s solution is to interpret all degeneracies as "coming from a smaller submodel." A submodel consists of $(\widetilde V_i,\widetilde\Theta_i,\widetilde f_i)$ plus a set of linking maps $\alpha_i:\widetilde V_i\to V_i$, $\alpha_i^*:V_i\to\widetilde V_i$ (satisfying $\alpha_i^*\circ\alpha_i=\mathrm{Id}$), and $\beta_i:\widetilde\Theta_i\to\Theta_i$, requiring a commutative diagram. For MLPs with linear $\alpha_i$, the submodel is exactly the smaller network obtained after "deleting inactive neurons and merging redundant ones." Consequently, degeneracies are handled by the submodel, and the remaining non-degenerate parts can be required to satisfy identifiability, bypassing issues caused by pathological activations like ReLU.

**2. Weak Identifiability: Unifying identifiability results for specific activations into a minimal assumption**  
Previous works (Sussmann, Fefferman, Vlačić & Bölcskei) proved identifiability for specific activations like Tanh, polynomial, or sigmoidal. This paper seeks a universal key to reuse these conclusions. Weak identifiability is defined as: a parameter $\theta$ is weakly identifiable if there exists an identifiable parameter $\widetilde\theta$ on a submodel such that $\theta_i=\beta_i(\widetilde\theta_i)$; while "identifiable" means if $f(\bullet;\theta)=f(\bullet;\theta')$, there exists a unique sequence $k_i\in K_i$ such that $f_i(x;\theta_i')=k_i\cdot f_i(k_{i-1}^{-1}\cdot x;\theta_i)$. This assumption decouples the "proof" from "specific activations."

**3. Adjunction Property: Translating global constraints into layerwise derivable local conditions**  
The main theorem propagates end-to-end $G$-equivariance into layerwise equivariance. The key pivot is the adjunction property: requiring $G$ to also act on the parameters of the first and last layers, satisfying $f_1(g\cdot x_0;\theta_1)=f_1(x_0;g^{-1}\cdot\theta_1)$ and $g\cdot f_L(x_{L-1};\theta_L)=f_L(x_{L-1};g\cdot\theta_L)$. This allows "moving" the $g$ action from the input side to the first-layer parameters, then using weak identifiability to transform it into the action of some $k_1\in K_1$ on $V_1$, and proceeding by induction. MLPs and attention networks with positional encodings naturally satisfy this.

### Main Theorem

**Theorem 4.1**: If $\theta$ is weakly identifiable and the model is end-to-end $G$-equivariant at $\theta$, then there exists an action of $G$ on each $V_i$ ($i=1,\dots,L-1$) such that each $f_i$ is $G$-equivariant at $\theta_i$.

The inductive skeleton of the proof follows: for any $g\in G$, use the adjunction property to transform the "input-side $g$ action" into a "parameter-side $g^{-1}$ action," yielding two end-to-end equivalent parameters. Weak identifiability then provides a unique latent symmetry $k_i(g)\in K_i$. Finally, it is verified that $g\mapsto k_i(g)$ is a group homomorphism, defining the $G$-action on $V_i$.

## Key Experimental Results

Theoretical results are qualitatively and quantitatively validated using small MLPs (depth 4, Tanh or GELU) and single-layer multi-head attention networks on CIFAR-10. A mirror-equivariance soft loss is added during the late stages of training to observe whether latent spaces spontaneously form equivariant structures.

### Main Results: Relative Equivariance Error of Layers 1 and 2

Linear transformations $A_i$ are estimated via least squares such that $A_i f_i(x)\approx f_i(\mathrm{mirror}[x])$. The median of $|f_i(\mathrm{mirror}[y]) - A_i f_i(y)|/|f_i(y)|$ over 100k noise samples is calculated:

| Task | Layer | Mirror Direction | Tanh | GELU |
|------|-----|---------|------|------|
| Autoencoder | $f_1$ | Horizontal | **0.029** | 0.40 |
| Autoencoder | $f_2$ | Horizontal | **0.022** | 0.11 |
| Autoencoder | $f_1$ | Vertical (Ref) | 0.15 | 0.49 |
| Classifier | $f_1$ | Horizontal | **0.077** | 0.19 |
| Classifier | $f_2$ | Horizontal | **0.054** | 0.064 |
| Classifier | $f_1$ | Vertical (Ref) | 0.48 | 0.56 |

Conclusion: Even though the objective only requires *end-to-end* mirror equivariance, the first layer of the Tanh network spontaneously approximates linear equivariance (error 0.029-0.077), corresponding to signed permutation matrices. The GELU network does not exhibit this in the first layer because GELU satisfies $\sigma(x)-\sigma(-x)=x$, allowing it to "bypass" the non-linearity; linear equivariance is only recovered at $f_2$.

### Ablation Study: Activation vs. Latent Action Structure

| Activation | Latent Action $K_i$ | Weak Identifiability | $f_1$ Linearly Equivariant? |
|---------|---------------------|------------|-------------------|
| Tanh | $\{\pm 1\}^{d_i}\rtimes S_{d_i}$ (Signed Perm) | Proved | Yes (Error ≈ 0.03) |
| High power $t^m$ | $(\mathbb R^\times)^{d_i}\rtimes S_{d_i}$ (Monomial) | Proved | Theoretically Supported |
| GELU | Permutation only + $\sigma(x)-\sigma(-x)=x$ | Difficult | No (Error 0.4) |
| ReLU | Positive Scaling + Permutation | Open Problem | Only for specific cases |

### Key Findings
- **Filters mirror theoretical groups**: The 64 filters in the first layer of the Tanh network show paired structures (Identical/Mirror/Negative), matching the geometric representation of the signed permutation group.
- **Attention equivariance is head permutation**: In the CIFAR-10 autoencoder, flipping the input image horizontally causes certain attention maps to flip, while specific heads (e.g., Head 1 and Head 5) swap with each other, reflecting the $S_{h_i}$ component of $K_i$.
- **Positional encodings follow the adjunction property**: This explains why positional encodings spontaneously learn structured geometric patterns when exposed to symmetric data augmentation.

## Highlights & Insights
- **First architecture-agnostic "End-to-end $\Rightarrow$ Layerwise" theorem**: Generalizes previous results for shallow ReLU MLPs to arbitrary architectures including MLPs, attention, and polynomial CNNs.
- **Introducing "Weak Identifiability" as a key technique**: By using submodels to absorb degenerate parameters, the proof cleanly decouples the abstract theorem from specific activation types.
- **Explaining the "Spontaneous Equivaiance" phenomenon**: Mathematically proves that if data/augmentation forces end-to-end equivariance and parameter identifiability holds, layerwise equivariance is inevitable.
- **Identifying ReLU as an open problem**: The framework equates the study of layerwise equivariance in ReLU networks with the algebraic geometric problem of ReLU identifiability.

## Limitations & Future Work
- **Exact vs. Approximate Equivariance**: The theory assumes exact equivariance, while real networks are only approximate; stability bounds are currently lacking.
- **Skip Connections**: Residual blocks introduce inter-layer dependencies that reduce identifiability, meaning the theorem cannot be directly applied to standard ResNets/Transformers.
- **Selection of Latent Actions**: The theorem guarantees the existence of some $K_i$ action but does not specify how the optimal action is chosen or which one SGD converges to.
- **Training Dynamics**: This is an existence result, not a constructive one regarding the optimization process.
- **Deep Sets / Equivariant GNNs**: These architectures fail the adjunction property as layers "absorb" group actions, making the theorem inapplicable.

## Related Work & Insights
- **vs. Agrawal & Ostrowski (2022)**: They used specific reparameterizations for shallow ReLU MLPs; this paper generalizes the derivation to arbitrary architectures and groups.
- **vs. Marchetti et al. (2024)**: They proved equivariant structures only for the first layer with neuron-wise scaling; this paper covers any depth and layer.
- **vs. Xie & Smidt (2025)**: While others debate whether "hard constraints" or "data augmentation" is better, this paper suggests that the resulting function classes are equivalent given identifiability.
- **vs. Categorical Deep Learning**: This work represents a substantial application of category theory formalization to equivariance problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First unified "End-to-end $\Rightarrow$ Layerwise" theorem)
- Experimental Thoroughness: ⭐⭐⭐ (Sufficient for a theory paper, but qualitative)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear rhythm between abstract formalization and concrete examples)
- Value: ⭐⭐⭐⭐⭐ (Provides a theoretical foundation for spontaneous equivariance)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] How the Optimizer Shapes Learned Solutions in Equivariant Neural Networks](how_the_optimizer_shapes_learned_solutions_in_equivariant_neural_networks.md)
- [\[NeurIPS 2025\] Equivariance by Contrast: Identifiable Equivariant Embeddings from Unlabeled Finite Group Actions](../../NeurIPS2025/others/equivariance_by_contrast_identifiable_equivariant_embeddings_from_unlabeled_fini.md)
- [\[NeurIPS 2025\] Learning (Approximately) Equivariant Networks via Constrained Optimization](../../NeurIPS2025/others/learning_approximately_equivariant_networks_via_constrained_optimization.md)
- [\[NeurIPS 2025\] On Universality Classes of Equivariant Networks](../../NeurIPS2025/others/on_universality_classes_of_equivariant_networks.md)
- [\[ICML 2025\] Permutation Equivariant Neural Networks for Symmetric Tensors](../../ICML2025/others/permutation_equivariant_neural_networks_for_symmetric_tensors.md)

</div>

<!-- RELATED:END -->
