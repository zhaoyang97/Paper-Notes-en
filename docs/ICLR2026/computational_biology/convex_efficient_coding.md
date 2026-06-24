---
title: >-
  [Paper Note] Convex Efficient Coding
description: >-
  [ICLR 2026][Computational Biology][Efficient Coding] This paper reformulates a broad class of "neural representation optimization" problems (efficient coding, semi-non-negative matrix factorization, non-negative sparse coding, etc.) as **convex optimization** over the **representation similarity matrix $Q$** (the matrix of pairwise dot products of neural responses). This approach maintains the flexibility of deep networks while regaining the analyzability of linear models. It…
tags:
  - "ICLR 2026"
  - "Computational Biology"
  - "Efficient Coding"
  - "Representational Similarity Matrix"
  - "Convex Optimization"
  - "Identifiability"
  - "Semi-Non-negative Matrix Factorization"
date: 2026-05-08
content_hash: 16677dfee7ea5446
---

# Convex Efficient Coding

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=Se3YaqtjqE](https://openreview.net/forum?id=Se3YaqtjqE)  
**Code**: https://github.com/WilburDoz/Convex_Efficient_Coding_ICLR_2026  
**Area**: Computational Neuroscience / Neural Coding Theory / Convex Optimization  
**Keywords**: Efficient Coding, Representational Similarity Matrix, Convex Optimization, Identifiability, Semi-Non-negative Matrix Factorization

## TL;DR
This paper reformulates a broad class of "neural representation optimization" problems (efficient coding, semi-non-negative matrix factorization, non-negative sparse coding, etc.) as **convex optimization** over the **representation similarity matrix $Q$** (the matrix of pairwise dot products of neural responses). This approach maintains the flexibility of deep networks while regaining the analyzability of linear models. It provides the first necessary and sufficient condition for the identifiability of semi-NMF, offers a theoretical justification for single-neuron tuning analysis, and explains the sparsity threshold for retinal ON-OFF coding.

## Background & Motivation

**Background**: "Normative" theories in neuroscience view neural activity as a solution to an optimization problem. The most classic is the **efficient coding hypothesis**, which posits that neurons respond in a way that "optimally encodes required information under efficiency constraints." These theories span a wide range of complexity: from simple, analytically solvable linear models like Atick & Redlich to task-optimized deep neural networks like Lindsay.

**Limitations of Prior Work**: Complex models (deep networks, sparse coding, NMF) are flexible but almost "analytically intractable." After training a network, it remains unclear **why** it developed a certain structure, under what conditions Gabor filters emerge, or whether single-neuron tuning curves truly reflect population function. Simple models are analytical but have limited expressive power. A significant gap exists between these two paradigms.

**Key Challenge**: Direct optimization over neural activities $\{z^{[i]}\}$ is typically **non-convex**. For instance, swapping two neurons and their corresponding readout weights yields two valid solutions, but their convex combination does not satisfy the constraints (this paper provides an explicit $2\times2$ example). Non-convexity implies no guarantee of global optimality and precludes clean identifiability analysis.

**Key Insight**: The authors build upon a key observation by Sengupta et al. (2018): instead of optimizing over neural activities, one can optimize over the **representation dot-product similarity matrix** $Q_{ij} = (z^{[i]})^\top z^{[j]}$. Previously, Sengupta only proved convexity for a bespoke "similarity matching" objective, which had limited applicability.

**Core Idea**: To systematize the "variable substitution + classic convex relaxation" approach. The authors prove that a **large family** of objectives and constraints are convex in the $Q$ space. These can be combined like building blocks to form various optimization problems of interest while maintaining overall convexity. This convexity guarantees a unique global optimum, opening the door to analytical investigations of three specific neuroscience problems.

## Method

### Overall Architecture

The logical structure of the paper consists of two layers: first, **building the machine**, and then **using the machine**.

The "machine" works as follows: consider a representation of $N$ data points, each represented by a $d_z$-dimensional neural activity $z^{[i]} \in \mathbb{R}^{d_z}$. A typical efficient coding problem requires certain targets $y^{[i]}$ to be **linearly decodable** from the representation ($y^{[i]} = W_{\text{out}}z^{[i]}$), while constraining the activity to be **non-negative** ($z^{[i]} \ge 0$) for biological realism and penalizing energy consumption ($L_2$ norm of firing rates and synaptic weights). This problem, formulated in terms of activity $z$, is non-convex. By switching to the similarity matrix $Q_{ij} = (z^{[i]})^\top z^{[j]}$, and assuming the number of neurons is larger than the number of data points ($d_z > N$, such that the rank of $Q$ is not restricted), every constraint and objective can be proven to be a convex function or convex set of $Q$. The benefit of convexity is that **all locally optimal $Q$ are globally optimal**, which serves as the foundation for the subsequent identifiability conclusions.

Once the machine is built, the authors apply it to three problems:
- **Application 1 (Section 3)**: Reformulating "non-negative affine autoencoding" (a form of semi-NMF) within this framework to derive its **first necessary and sufficient identifiability condition** ("tight scattering").
- **Application 2 (Section 4)**: Using the same theory to precisely link **population-level representational similarity** with **single-neuron tuning curves**, explaining under what conditions analyzing "what a single neuron encodes" is meaningful.
- **Application 3 (Section 5)**: Leveraging the fact that **nonlinear problems remain solvable** in this framework to analytically explain why the retina splits a single variable into ON and OFF channels and when it does not.

### Key Designs

**1. Convexification over the representation similarity matrix $Q$**

The pain point is that optimizing efficient coding problems directly over $z$ is non-convex. This paper substitutes the variable with $Q_{ij} = (z^{[i]})^\top z^{[j]}$ and verifies convexity term by term. For example, in a "linearly decodable + minimum energy" problem:

$$\min_{W_{\text{out}},\,\{z^{[i]}\}}\ \big(\langle\|z^{[i]}\|^2\rangle_i + \lambda\|W_{\text{out}}\|_F^2\big)\quad \text{s.t. } z^{[i]}\ge 0,\ W_{\text{out}}z^{[i]}=y^{[i]}.$$

In terms of $Q$: the firing rate cost $\langle\|z^{[i]}\|^2\rangle_i = \tfrac{1}{N}\operatorname{Tr}[Q]$ is linear (hence convex) in $Q$. The weight cost, assuming a minimum-norm readout (pseudoinverse $W_{\text{out}} = YZ^\dagger$), becomes $\|W_{\text{out}}\|_F^2 = \operatorname{Tr}[Y^\top Y\,Q^\dagger]$, which is provably convex in $Q$. The non-negativity constraint corresponds to the set of "completely positive" matrices $\{Q = Z^\top Z, Z \ge 0\}$, which is a convex set. The linear decodability constraint restricts $Q$ to a convex set where a specific subspace encodes the labels. Because the composition of convex functions and sets remains convex, the overall problem is convex. This step is the pivot of the paper, as global optimality allows for exact characterizations of the "optimal representation."

**2. Tight Scattering: The first necessary and sufficient identifiability for semi-NMF**

Matrix factorization $Y = AS$ is inherently non-unique. The authors study **non-negative affine autoencoding** (where data $X = AS$ is fed into a two-layer affine autoencoder with non-negative hidden layers, minimizing reconstruction error plus $L_2$ weight and activity norms). They ask: under what conditions does the optimal hidden activity $Z$ recover the sources $S$ ($Z = \Pi S + b\mathbf{1}$, where $\Pi$ is a "rectangular permutation" matrix). Prior work only provided conditions for **orthogonal** mixing matrices $A$ and was typically either sufficient **or** necessary. By leveraging convexity, this paper generalizes the condition to **arbitrary linear mixing $A$** and gives a "tight scattering" characterization (Definition 1). Using data covariance $\Sigma$, source minima, and the mixing Gram matrix $A^\top A$, they construct a symmetric matrix $F = \lambda D(A^\top A)D - \lambda(A^\top A)^{-1} - \Sigma$, defining an ellipsoid $E = \{x \mid x^\top F^{-1} x = 1\}$. Identifiability occurs if and only if the convex hull $\operatorname{Conv}(\bar S)$ **swallows** this ellipsoid (Theorem 1).

**3. Non-negativity Breaks Rotational Symmetry: From population similarity to unique tuning**

Neuroscience often infers function by correlating neural activity with task variables. However, the same function can be implemented by many networks whose internal representations are rotationally related. If a whole population is **rotated**, tuning curves change entirely while representational similarity remains unchanged. Does the observed "neural basis" have meaning? The authors argue that while regularization makes $Q$ unique, **non-negative constraints break rotational symmetry**. Theorem 2 shows that if tuning curves satisfy the tight scattering condition relative to an ellipsoid $E$, any non-negative-preserving orthogonal transformation $O$ must be a permutation matrix. This explains the modularity of grid cells: discrete modules emerge because only certain frequency relationships satisfy the identifiability condition.

**4. Sparsity Threshold for ON-OFF Coding**

The retina splits single variables into ON and OFF channels (rectified in opposite directions). Sterling & Laughlin hypothesized that **sparsity** determines this split. Using the framework's **non-negative, affine-decodable** single-variable representation:

$$\min\big(\langle\|z(I)\|^2\rangle_{p(I)}+\lambda\|w\|_F^2\big)\quad\text{s.t. } w^\top z(I)+b=I,\ z(I)\ge 0,$$

The authors solve for the unique optimum via KKT conditions. When the variable is dense, splitting into ON/OFF channels reduces firing rates. When the variable is sufficiently sparse (e.g., $I \ge 0$ and frequently $I = 0$), a single channel is more efficient. The sparsity threshold for this phase transition is analytically derived as:

$$\Pr(I=0) > \frac{\langle I\rangle_{p(I)}^2}{\langle I^2\rangle_{p(I)}}.$$

## Key Experimental Results

### Identifiability (Section 3, Fig. 1)

| Setting | Tight Scattering Condition | Numerical Optimal Behavior | Consistent with Theorem 1? |
|------|-----------|-----------------|----------------------|
| Orthonormal Encoding Sources | Satisfied | Modular (each neuron encodes one source) | Yes |
| Anti-aligned Mixed Sources | Not Satisfied | Mixed (sources are entangled) | Yes |
| Moderately Aligned Mixed Sources | Satisfied (after distortion) | Recovered Sources (identifiability restored) | Yes |

**Key Findings**: The alignment of the mixing matrix $A$ **distorts** the identifiability boundary. Alignment can "rescue" sources that would be unidentifiable under orthogonal mixing, while anti-alignment can "destroy" them.

### Single-Neuron Tuning Uniqueness & Grid Cells (Section 4, Fig. 2)

| Tuning Scenario | Tight Scattering Satisfied? | Non-trivial Non-negative Rotation Found? |
|----------|---------------|------------------------------|
| Separated Place Cells | Yes | No (Tuning is unique) |
| Overlapping Place Cells | No | Yes (Same $Q$, different tunings) |
| Non-integer Grid Cell Modules | Yes | No (Modules identifiable) |
| Integer-ratio Grid Cell Modules | No | Yes (Degrades to unidentifiable) |

**Key Findings**: **Non-negativity** is the key that locks "population similarity" to "single-neuron identity." Without it, rotational symmetry renders single-neuron analysis groundless.

### ON-OFF Phase Transition (Section 5, Fig. 3)

The numerical firing rates follow the sparsity: at low sparsity, the population contains both ON and OFF neurons. As sparsity increases, the ON encoding range expands while OFF contracts until passing the threshold $\Pr(I=0) > \langle I\rangle^2 / \langle I^2\rangle$, where OFF neurons disappear. The numerical phase transition aligns perfectly with the analytical formula.

## Highlights & Insights
- **Transformation as an elegant simplification**: Moving non-convex problems to the $Q$ space makes convexity "emerge naturally." This "modular convex building block" approach can be applied to any new normative modeling that fits the criteria.
- **Necessary and sufficient identifiability is a major breakthrough**: Most identifiability results in matrix factorization are only sufficient. This paper provides the **first necessary and sufficient** condition for semi-NMF using convexity.
- **Justification for single-neuron tuning analysis**: By clarifying how "non-negativity breaks rotational symmetry $\Rightarrow$ unique single-neuron identity," the paper addresses long-standing debates about the level at which the brain should be studied.
- **Theories yield falsifiable predictions**: Explanations for why grid cells are modular and the exact sparsity threshold for ON-OFF coding are analytical and testable, rather than just abstract theorems.

## Limitations & Future Work
- **Dependency on $d_z > N$**: Convexity only holds if the rank of $Q$ is not further restricted. If the number of neurons is limited, the problem becomes non-convex. Future work suggests using the nuclear norm as a convex relaxation for rank.
- **Computational Intractability**: While convex, membership in the "completely positive" set and verifying scattering conditions are NP-hard. Thus, the framework is currently an **analytical tool** rather than a practical large-scale solver.
- **Neuroscience Focus**: Unlike the "convex ReLU network" line of work (e.g., Pilanci & Ergen), this paper optimizes **representations** rather than weights, which makes it highly relevant for neuroscience but less directly applicable to standard ML training.

## Related Work & Insights
- **vs. Sengupta et al. (2018)**: This paper generalizes their work from a single "similarity matching" objective to a broad family of combinable convex objectives and constraints.
- **vs. Whittington et al. (2023) / Dorrell et al. (2025)**: These works study semi-NMF but only for **orthogonal** mixing matrices. Ours generalizes to arbitrary mixing $A$ with a necessary and sufficient condition.
- **vs. Donoho & Stodden (2003)**: Prior scattering conditions were independent of $A$ and only sufficient. Ours is the first to adapt the criterion to both $A$ and the data covariance.
- **vs. Braun et al. (2025)**: Ours extends their argument for unique population-level representation similarity from linear networks to a much wider class of convex problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Providing the first necessary and sufficient identifiability condition for semi-NMF via a unified convex framework is a significant methodological contribution.
- Experimental Thoroughness: ⭐⭐⭐ A theory paper where simulations confirm theorems. No benchmarks, but all analytical predictions are numerically verified.
- Writing Quality: ⭐⭐⭐⭐ The "building vs. using" logic is clear, though the density of theorems and appendices makes for a challenging read.
- Value: ⭐⭐⭐⭐⭐ Provides reusable tools and conclusions for both neuroscience (tuning analysis, grid cells) and ML (identifiability, convex analysis).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] DCFold: Efficient Protein Structure Generation with Single Forward Pass](dcfold_efficient_protein_structure_generation_with_single_forward_pass.md)
- [\[ICLR 2026\] Efficient Prediction of Large Protein Complexes via Subunit-Guided Hierarchical Refinement](efficient_prediction_of_large_protein_complexes_via_subunit-guided_hierarchical_.md)
- [\[ICLR 2026\] GAGA: Gaussianity-Aware Gaussian Approximation for Efficient 3D Molecular Generation](gaga_gaussianity-aware_gaussian_approximation_for_efficient_3d_molecular_generat.md)
- [\[ICLR 2026\] FragFM: Hierarchical Framework for Efficient Molecule Generation via Fragment-Level Discrete Flow Matching](fragfm_hierarchical_framework_for_efficient_molecule_generation_via_fragment-lev.md)
- [\[AAAI 2026\] Efficient Chromosome Parallelization for Precision Medicine Genomic Workflows](../../AAAI2026/computational_biology/efficient_chromosome_parallelization_for_precision_medicine_genomic_workflows.md)

</div>

<!-- RELATED:END -->
