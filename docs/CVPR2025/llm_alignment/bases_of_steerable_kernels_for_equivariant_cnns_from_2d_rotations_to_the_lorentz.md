---
title: >-
  [Paper Note] Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group
description: >-
  [CVPR 2025][LLM Alignment][Equivariant CNNs] Proposes an alternative method to solve the constraint equations of steerable equivariant CNN kernels. By solving simpler invariance conditions at a fixed point and then "steering" to arbitrary points, this approach bypasses the need for computing Clebsch-Gordan coefficients, providing explicit kernel basis formulas for SO(2), O(2), SO(3), O(3), and the Lorentz group.
tags:
  - "CVPR 2025"
  - "LLM Alignment"
  - "Equivariant CNNs"
  - "steerable kernels"
  - "representation theory of groups"
  - "Clebsch-Gordan coefficients"
  - "SO(2)"
  - "SO(3)"
  - "Lorentz group"
date: 2026-05-08
content_hash: 4227357e03710c23
---

# Bases of Steerable Kernels for Equivariant CNNs: From 2D Rotations to the Lorentz Group

**Conference**: CVPR 2025  
**arXiv**: [2603.12459](https://arxiv.org/abs/2603.12459)  
**Code**: None  
**Area**: 3D Vision (Note: This paper belongs to equivariant networks/geometric deep learning, but was misclassified into llm_alignment)  
**Keywords**: Equivariant CNNs, steerable kernels, representation theory of groups, Clebsch-Gordan coefficients, SO(2), SO(3), Lorentz group

## TL;DR
Proposes an alternative method to solve the constraint equations of steerable equivariant CNN kernels. By solving simpler invariance conditions at a fixed point and then "steering" to arbitrary points, this approach bypasses the need for computing Clebsch-Gordan coefficients, providing explicit kernel basis formulas for SO(2), O(2), SO(3), O(3), and the Lorentz group.

## Background & Motivation
**Background**: Steerable equivariant CNNs improve performance by hard-coding symmetry priors into network designs—if the input undergoes a symmetric transformation, the output automatically transforms accordingly, eliminating the need for data augmentation. The core constraint is that the convolution kernel $K$ must satisfy the steerability equation: $K(g \cdot x) = \rho_{\text{out}}(g) K(x) \rho_{\text{in}}(g)^{-1}$.

**Limitations of Prior Work**: Existing general solving methods (e.g., Lang & Weiler 2021) rely on Clebsch-Gordan (CG) coefficients—requiring the tensor product representation to be decomposed into irreducible representations (coupled basis) first before transforming it back. The computation of CG coefficients is complex and difficult for certain groups, and this indirect path is unfriendly to non-specialist readers.

**Key Challenge**: Theoretically, equivariant CNNs are applicable to any group. However, in practice, computing CG coefficients during the construction of steerable kernels becomes a bottleneck, which restricts the extension of equivariant CNNs to more complex symmetry groups (such as the Lorentz group).

**Key Insight**: It is observed that the steerability constraint simplifies to a representation homomorphism condition at the stabilizer subgroup of a fixed point, while the homomorphisms of completely reducible representations can be directly written out using Schur's Lemma. The steering equation is then used to "steer" the solution at the fixed point to any other point.

**Core Idea**: Select a fixed point $x_0$ on the orbit, solve the simplified invariance conditions (finding representation homomorphisms) at its stabilizer subgroup $H$, and then generalize it to the entire orbit using the steering formula.

## Method

### Overall Architecture
The method consists of three steps:
1. Select a fixed point $x_0$ on the $G$-orbit and determine its stabilizer subgroup $H = \text{Stab}_{x_0}$.
2. Decompose the restricted representations $\rho_j^H, \rho_l^H$ into the direct sum of irreducible representations of $H$, and use Schur's Lemma to solve for the bases of the homomorphism space $\text{Hom}_H(V_l, V_j)$, yielding all possible values for $K(x_0)$.
3. For each basis element, generalize to any point $x = g \cdot x_0$ on the orbit using the steering formula $K(g \cdot x_0) = \rho_j(g) K(x_0) \rho_l(g)^{-1}$.

Input: Symmetry group $G$, input/output representations $\rho_{\text{in}}, \rho_{\text{out}}$. Output: Explicit basis functions of the steerable kernel.

### Key Designs

1. **Simplified Constraints at the Stabilizer Subgroup**:

    - **Function**: Simplifies the global steering constraint (Eq.11) to a local constraint at the fixed point (Eq.14).
    - **Mechanism**: At $x_0$, the equation degenerates to $K(x_0) = \rho_j(h) K(x_0) \rho_l(h)^{-1}$ for all $h \in H$, meaning $K(x_0)$ must be an intertwiner (homomorphism) between $H$-representations. For trivial stabilizer subgroups (such as points on a circle under SO(2)), any matrix is a solution.
    - **Design Motivation**: Replaces solving constraint equations directly over the entire space with a much simpler algebraic problem.
    - **Mathematical Insight**: The orbit-stabilizer theorem guarantees that the $G$-space can be decomposed into orbits $G/H$, with the kernel on each orbit fully determined by its value at the fixed point. Consequently, global solving is reduced to single-point solving.

2. **Systematic Application of Schur's Lemma**:

    - **Function**: Leverages Schur's Lemma to transform the task of solving representation homomorphisms into a combinatorial counting problem.
    - **Mechanism**: Decomposes $\rho_j^H, \rho_l^H$ into a direct sum of irreducible subrepresentations of $H$. After partitioning the homomorphism matrix into blocks, non-isomorphic blocks are zero, and isomorphic blocks are scalar multiples of the identity matrix (for complex representations) or specific known isomorphisms (for real representations).
    - **Design Motivation**: This step completely bypasses CG coefficients—whereas traditional methods require CG coefficients to decompose tensor products into irreducible direct sums, this method only needs to know the decomposition of the restricted representation.
    - **Real vs. Complex Representation Differences**: For complex irreducible representations, Schur's Lemma gives $\text{Hom}_H(\rho_a, \rho_b) = \mathbb{C} \cdot \delta_{a,b}$ (1D or 0D). For real irreducible representations, the homomorphism space can be $\mathbb{R}$, $\mathbb{C}$, or $\mathbb{H}$ (quaternions), depending on the representation type (real/complex/quaternionic type), with basis dimensions being 1, 2, or 4 respectively.

3. **Explicit Bases Covering Multiple Symmetry Groups**:

    - **Function**: Derives explicit bases individually for SO(2), O(2), SO(3), O(3), and the Lorentz group.
    - **Mechanism**: For each group, an appropriate fixed point and stabilizer subgroup are selected, followed by systematic application of the aforementioned framework. For instance, in SO(3), $x_0 = (0,0,R)$ is selected with the stabilizer subgroup being SO(2); in the Lorentz group, $x_0$ is selected on the light-cone or time-cone with the stabilizer subgroup being $\mathbb{R}$ or $\mathbb{C}$/SO(2), respectively.
    - **Design Motivation**: Provides "ready-to-use" kernel basis formulas for equivariant CNN practitioners.
    - **Specific Example of SO(2)**: Irreducible representations are $\rho_m(R_\alpha) = e^{im\alpha}$ (complex) or 2x2 rotation matrices (real). On the circle $S^1$, the stabilizer subgroup is trivial, so $K(x_0)$ can be any matrix, which yields the Fourier expansion form of $K(\theta) = e^{im\theta}$ after steering. For real representations, each frequency corresponds to 4 basis matrices (combinations of $\cos/\sin$ and rows/columns).
    - **Specific Example of Lorentz Group**: In $\text{SO}^+(1,3)$, select the light-like vector $x_0 = (1,0,0,1)$ with the stabilizer subgroup being the Abelian subgroup of the Euclidean group $E(2)$. Decomposing the restricted representations requires handling continuous spectra for non-compact groups, but discrete decompositions can still be applied to finite-dimensional spacetime tensor representations.

## Key Experimental Results

### Main Results
This paper is a **purely theoretical/methodological paper** and does not contain numerical experiments. The core contribution is mathematical derivation:

| Symmetry Group | Representation | Stabilizer Subgroup | Basis Dimension | Comparison with Traditional Methods |
|--------|------|----------|---------|------------|
| SO(2) | Complex Irreducible | Trivial Group | 1 | Consistent with Weiler 2019 Table 8 |
| SO(2) | Real Irreducible | Trivial Group | 4 (j,l≥1) | Consistent with Weiler 2019 Table 8 |
| O(2) | Real Irreducible | {I, σ₃} | 2 or 1 | Consistent with Weiler 2019 Table 9 |
| SO(3) | Complex Irreducible | SO(2) | (2min(j,l)+1) | Consistent with Lang 2021 |
| O(3) | — | O(2) | Case-by-case analysis | Consistent with e3nn |
| Lorentz SO⁺(1,3) | Spacetime Tensor | SO(2) | Explicit formula | Bypasses the CG method of Bogatskiy 2020 |

### Key Findings
- All explicit bases are fully consistent with the solution spaces obtained by existing methods (CG coefficient methods), validating correctness.
- For spacetime tensor representations of the Lorentz group, explicit steerable kernel bases independent of CG coefficients are presented for the first time.
- The method is conceptually more direct: it eliminates the need for coupled/decoupled basis transformations, reducing implementation complexity.
- The results for SO(3) can be directly applied to kernel parameterization under spherical harmonics bases, maintaining full compatibility with popular implementations (such as e3nn).
- The number of kernel bases (number of free parameters) is determined by the representation decomposition: for $\rho_j \otimes \rho_l$ in SO(3), the basis dimension is $2\min(j,l)+1$, which is consistent with the predictions of the CG decomposition (integer number of irreducible representations from $|j-l|$ to $j+l$).
- Parity handling: O(3) introduces an additional spatial inversion operation compared to SO(3), leading to representations being categorized into even/odd types and extending the stabilizer subgroup from SO(2) to O(2), which requires further case-by-case analysis.

## Highlights & Insights
- **Clear Methodological Contribution**: Simplifies equivariant CNN kernel design into a three-step workflow ("select fixed point $\rightarrow$ solve homomorphisms $\rightarrow$ steer"), which is easy to grasp even for non-group-theory experts. This is a rare endeavor in this field that aims to "make methods simpler" rather than "performing more complex tasks."
- **Practical Relevance of Lorentz Group Extension**: Highly valuable for geometric deep learning over particle collision data in high-energy physics, circumventing the mathematical difficulties of computing CG coefficients for the Lorentz group.
- **Outstanding Pedagogical Value**: This work can serve as an introductory tutorial for constructing steerable kernels in equivariant CNNs—progressing step-by-step from the simplest SO(2) to the complex Lorentz group, with explicit calculations presented for each stage, making it highly friendly to researchers new to the field.

## Limitations & Future Work
- **Lack of Experimental Validation**: The authors acknowledge that "validation across different experiments is needed" to determine if their independent truncation strategy yields superior expressiveness, but no numerical results are provided in this paper.
- **No Coverage of Discrete Groups**: The focus is primarily on continuous Lie groups; applications to finite groups (such as crystallographic symmetry groups) are not discussed.
- **Missing Implementation Details**: How to handle anti-aliasing (aliasing) issues is only mentioned briefly without providing a solution.
- **Function Space Issues in Non-Compact Groups**: The Lorentz group is non-compact, and steerable kernels on it may need to be defined in a distributional sense, analytical details of which are not discussed in the text.
- **No Efficiency Comparison with Numerical Methods**: Fails to compare computational efficiency and precision with methods that solve kernel constraints via numerical optimization (e.g., parameterized kernel networks).

## Related Work & Insights
- **vs Lang & Weiler (2021)**: They solve constraints using a general method with CG coefficients and coupled-basis representations. This work bypasses CG coefficients to be more direct but yields an equivalent solution space.
- **vs Weiler (2019, E2CNN)**: Designed for 2D methods; our SO(2)/O(2) results are in full agreement with theirs.
- **vs Bogatskiy (2020, Lorentz Net)**: Also handles the Lorentz group but relies on CG coefficient methods. Our approach is more direct for spacetime tensor representations.
- **vs e3nn (Geiger & Smidt 2022)**: e3nn is the standard library for SO(3)/O(3) equivariant networks implemented via CG coefficients. Our SO(3) formula is consistent with e3nn outputs but uses a different derivation path—making it more suitable for theoretical understanding than a replacement implementation.
- **vs Pontrjagin Classes Methods**: Topological methods construct equivariant kernels via sections of fiber bundles, which is more abstract but general. Our approach is more concrete and computable.

## Rating
- Novelty: ⭐⭐⭐⭐ The method has been mentioned in literature but was not fully developed; this paper systematizes it.
- Experimental Thoroughness: ⭐⭐⭐ Lacks any numerical experiments.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear mathematical derivations and rich examples.
- Value: ⭐⭐⭐⭐ Possesses utility value for equivariant CNN practitioners; the extension to the Lorentz group is highly innovative.

## Supplementary Notes
- This paper being classified into llm_alignment is a misclassification; it belongs to the geometric deep learning / equivariant networks direction.
- Suitable as a reference for equivariant CNN kernel construction, especially when custom symmetry groups are required.
- Extremely helpful for understanding the underlying mathematics of equivariant network libraries like e3nn and escnn.
- The Lorentz group section is of most value to the high-energy physics ML (jet tagging, particle tracking) community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Steerable Cultural Preference Optimization of Reward Models](../../ICML2026/llm_alignment/steerable_cultural_preference_optimization_of_reward_models.md)
- [\[ICML 2026\] VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models](../../ICML2026/llm_alignment/valueflow_toward_pluralistic_and_steerable_value-based_alignment_in_large_langua.md)
- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](../../NeurIPS2025/llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [\[ICLR 2026\] Group-Normalized Implicit Value Optimization for Language Models](../../ICLR2026/llm_alignment/group-normalized_implicit_value_optimization_for_language_models.md)
- [\[ACL 2026\] MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following](../../ACL2026/llm_alignment/mdp-grpo_stabilized_group_relative_policy_optimization_for_multi-constraint_inst.md)

</div>

<!-- RELATED:END -->
