---
title: >-
  [Paper Note] Reducing Symmetry Increase in Equivariant Neural Networks
description: >-
  [ICLR 2026][Learning Theory][Equivariant networks] This paper systematically characterizes the degenerative phenomenon where the "output symmetry is amplified and directional information is lost" when equivariant neural networks process symmetric inputs. It proves that the amplified symmetry has a symmetry infimum uniquely determined by the feature space structure and provides a computable algorithm and feature design guidelines to predict and avoid harmful symmetry increase.
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Equivariant Neural Networks"
  - "Geometric Deep Learning"
  - "Equivariant networks"
  - "symmetry increase"
  - "orbit types"
  - "symmetry infimum"
  - "equivariant maps"
date: 2026-05-08
content_hash: 091c7bd9611a66eb
---

# Reducing Symmetry Increase in Equivariant Neural Networks

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=dUgq4bLY4X](https://openreview.net/forum?id=dUgq4bLY4X)  
**Code**: To be confirmed  
**Area**: Learning Theory / Equivariant Neural Networks / Geometric Deep Learning  
**Keywords**: Equivariant networks, symmetry increase, orbit types, symmetry infimum, equivariant maps

## TL;DR
This paper systematically characterizes the degenerative phenomenon where the "output symmetry is amplified and directional information is lost" when equivariant neural networks process symmetric inputs. It proves that the amplified symmetry has a symmetry infimum uniquely determined by the feature space structure and provides a computable algorithm and feature design guidelines to predict and avoid harmful symmetry increase.

## Background & Motivation
**Background**: Equivariant Neural Networks (ENNs) have become a cornerstone of geometric deep learning by embedding physical symmetries into model architectures, achieving exceptional data efficiency and generalization in scientific scenarios such as molecular dynamics and materials design. Their core constraint is equivariance: for any group element $g$ and input $x$, the mapping satisfies $f(\rho_X(g)x) = \rho_Y(g)f(x)$.

**Limitations of Prior Work**: When the input itself is symmetric, the expressivity of ENNs degenerates—the output representation becomes invariant to transformations "beyond the input's own symmetry group," thereby losing information. A typical example is the $k$-fold structure ($k$-fold symmetric polygon point cloud): theoretically, it only possesses specific dihedral symmetry, but ENNs map its different rotated versions to the same feature, erasing orientation information. The authors name this phenomenon **symmetry increase**.

**Key Challenge**: The mathematical essence behind this is that for a symmetric input passing through an equivariant map, its equivariance (intrinsic symmetry, characterized by the isotropy subgroup $G_x$) can only increase and never decrease (Curie’s Principle, $G_x \subseteq G_{f(x)}$). When the feature space $Y$ cannot "contain" the input's symmetry type, the increase becomes inevitable, leading to degeneration. Existing works have either performed empirical observations (Joshi et al.), covered only the extreme "collapse-to-zero" case (Cen et al.), or bypassed the problem by relaxing equivariance constraints (Kaba & Ravanbakhsh). A **rigorous, predictable, and unified theory** solvable within the equivariant framework is missing.

**Goal**: (1) Clearly characterize the lower bound of symmetry increase; (2) Provide a computable decision algorithm; (3) Prove that under reasonable regularity conditions, features designed according to the guidelines can truly eliminate harmful symmetry increase.

**Core Idea**: The amplified symmetry has a **unique** symmetry infimum $I_G(Y, G_x)$ determined by the algebraic structure of the feature space. By calculating the orbit types of the feature space, one can predict which types of degeneration will occur in advance and avoid them by selecting appropriate feature components.

## Method

### Overall Architecture
This is a pure theory + algorithm work aiming to answer three progressive questions: How low can the symmetry increase be pushed? How is this lower bound calculated? And is this lower bound attainable in a real trainable ENN? The logic chain is: Given input symmetry group $H=G_x$ and feature space $Y$ → Use algebraic structures to define and prove the existence of a unique **symmetry infimum** $I_G(Y, H)$ (§3) → Use Michel's criterion + orbit type algorithm to **calculate** this infimum and predict three categories of $k$-fold degeneration (§4) → Translate computable results into **feature design guidelines** (§4.2) → Prove that for sufficiently expressive ENNs under the manifold hypothesis, this infimum is **generically attainable** (§5). Finally, validate with synthetic data and QM9 experiments (§6).

The following diagram connects the reasoning pipeline from "Input → Infimum Calculation → Degeneration Prediction → Guided Design → Reachability Guarantee":

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input Symmetry H=Gx<br/>+ Feature Space Y"] --> B["Symmetry Infimum<br/>Unique Minimal Orbit Type I(Y,H)"]
    B --> C["Orbit Type Calculation Algorithm<br/>Michel Criterion + Chain Recursion"]
    C -->|Compare H and I(Y,H)| D["Predict Degeneration Type<br/>Full / Axial / Half"]
    D --> E["Feature Design Guidelines<br/>Select Components Matching Task Symmetry"]
    E --> F["Density Theorem<br/>Infimum Generically Attainable with Sufficient Expressivity"]
    F --> G["Design ENNs that Retain Directional Information"]
```

### Key Designs

**1. Symmetry Infimum: Compressing "How Much Symmetry is Amplified" into a Unique Lower Bound**

To address the fundamental pain point that "equivariant maps only increase symmetry and the extent of increase is unpredictable," the authors introduce the concept of the symmetry infimum. First, a partial order is established on orbit types: $(H_1) \geq (H_2)$ if and only if $H_1$ contains a conjugate subgroup of $H_2$—a larger orbit type represents higher symmetry. The key observation is that the fixed-point subspace $X^H$ contains all "higher symmetry" points, and within it exists a unique minimal orbit type. The **Uniqueness of Minimal Type Theorem (Thm 3.1)** guarantees: for a representation $X$ of a compact Lie group $G$ and any closed subgroup $H$, there exists a unique minimal orbit type in $X^H$, denoted as $I_G(X, H)$.

With this infimum, an ideal equivariant map can be precisely defined as an **isovariant map**—a map that strictly preserves symmetry $G_x = G_{f(x)}$. When the actual increase exceeds the infimum, i.e., $(G_{f(x)}) > I_G(Y, G_x)$, unexpected symmetry increase occurs. The authors further provide a **necessary condition (Thm 3.2)** for the existence of an isovariant map: $O_G(X) \subseteq O_G(Y)$, which is equivalent to $I_G(Y, H) = (H)$ for all $(H) \in O_G(X)$. This transforms the vague empirical question of "whether directional information is lost" into a decidable algebraic inclusion relationship.

Feature spaces often have **non-trivial kernels** (e.g., point cloud tasks require invariance to permutation $S_n$, naturally including $S_{k+1}$ in the kernel). In such cases, absolute isovariance is too strict. The authors use the operator $p_Y = \pi_Y^{-1} \circ \pi_Y$ (where $\pi_Y: G \to G/\ker\rho_Y$ is the natural projection) to isolate the "inevitable increase": $G_x \subseteq p_Y(G_x) \subseteq G_{f(x)}$. The goal then relaxes to **relative isovariance** $\rho_Y(G_x) = \rho_Y(G_{f(x)})$, meaning no additional increase occurs beyond that forced by the kernel. The necessary condition becomes $I_G(Y, H) = (p_Y(H))$ (Thm 3.3).

**2. Orbit Type Calculation Algorithm: Using Michel's Criterion to Compute the Abstract Infimum**

Although the infimum is unique, it must be computable in high-multiplicity feature spaces to be useful in representation learning—this is exactly where existing bifurcation theories (multi-focus irreducible representations) fall short. The authors' entry point is **Michel's Criterion (Thm 4.1)**: a necessary condition for a closed subgroup $H$ to be an isotropy subgroup is that for any adjacent closed supergroup $H' \supsetneq H$, the dimension of the fixed-point subspace strictly decreases $\dim V^{H'} < \dim V^H$. While generally only necessary, the authors prove (**Prop 4.2**) that when $V$ is a high-multiplicity representation (each non-zero isotypic component multiplicity $m(V, V_i) > \dim G$), this condition is also sufficient.

This sufficiency brings two direct benefits: first, it turns decision-making into a **chain recursion** that only checks adjacent subgroups, where fixed-point dimensions can be computed via trace formulas; second, this condition is frequently satisfied in practice (for all finite groups and high-channel feature spaces). Based on this, the authors designed the **Orbit Type Test (Algo 1)** and **Symmetry Infimum Calculation (Algo 2)**. The former checks if $(H)$ is an orbit type by verifying strict dimension decreases of adjacent supergroups; the latter filters supergroups of $H$ belonging to $O_G(V)$ and takes the minimal one as $I_G(V, H)$.

Using this algorithm to analyze $k$-fold structures (geometric symmetry $D_{kh}$ in $Y = V_{l=l_0}^{\oplus r}$), the authors obtained a complete classification of $k$-fold degeneration—a key advancement over previous work that "could only explain full degeneration":

| Degeneration Type | Symmetry Infimum | Physical Meaning |
|-------------------|------------------|------------------|
| Half | $(D_{2kh} \times S_{k+1})$ | Cannot distinguish $k$-fold from itself rotated by $\pi/k$ around $z$-axis |
| Axial | $(D_{\infty h} \times S_{k+1})$ | Cannot distinguish any rotation around $z$-axis |
| Full | $(O(3) \times S_{k+1})$ | Cannot distinguish any rotation around any axis |

Which category the degeneration falls into is determined by the relationship between feature order $l_0$ and $k$, as well as their parity (Thm/Table 1), allowing for prior prediction.

**3. Feature Design Guidelines: Translating Computable Conclusions into "Which Feature Components to Choose"**

Calculability is not enough; the authors translate it into actionable design rules. Utilizing the orbit type properties of direct sums—$O_G(V_1) \cup O_G(V_2) \subseteq O_G(V_1 \oplus V_2)$ and $I_G(V_1 \oplus V_2, H) \leq I_G(V_i, H)$—adjusting feature components becomes a direct lever to control symmetry increase. One simply selects components whose infimum aligns with the required symmetry behavior of the task.

Tasks are categorized into two types: For **direction-dependent tasks** (e.g., geometric graph discrimination), non-trivial symmetry increase must be avoided (i.e., ensure relative isovariance) by including feature components that contain the orbit type $(p_Y(H))$ for a given input symmetry $(H)$; otherwise, orientation information is accidentally lost. For **general tasks** (e.g., molecular property prediction), the output symmetry reflects the dimension of the fixed-point subspace where equivariant features reside. One should avoid components where the infimum severely collapses the fixed-point subspace—specifically, be wary of components that "increase symmetry to the full group $O(3)$ despite being non-trivial representations," as this causes the component to be annihilated and lose all discriminative power.

**4. Density of (Approximate) Isovariant Maps: Proving the Infimum is Generically Attainable with Sufficient Expressivity**

The previous sections provide "necessary conditions for existence," but necessity does not equal sufficiency (Cex D.3 shows that even if orbit type inclusion holds, a perfect isovariant map might not exist due to insufficient irreducible representation multiplicity). To bridge theory to real trainable models, the authors model data distribution under the **manifold hypothesis** (supported on a union of finite smooth compact submanifolds $M = \bigcup_j M_j$) and relax the definition to **almost isovariant**: the isovariance condition holds on each orbit type $M_{(H)}$ except for a subset of measure zero.

The authors then prove a strong approximation property: taking TFN as an example, the family of TFNs with smooth activations is $C^\infty$-dense in the space of smooth equivariant maps $C_G^\infty(X, Y)$ (Thm 5.1, including uniform approximation of all derivatives). Based on this, the core conclusion (**Thm 5.2**) states: for any equivariant parameterization $F$ with $C^\infty$ approximation capability, as long as $(p_Y(H)) \in O_G(Y)$ for every $(H) \in O_G(M)$, the necessary condition becomes sufficient—there exists $g \in F$ that is relatively isovariant almost everywhere. Furthermore, if the feature space contains $\tilde Y^{\oplus r}$ with multiplicity $r > \max_j\{\dim M_j\}$, an approximation map that is relatively isovariant everywhere can be obtained. This strictly docks the "calculated infimum" with the "trainable network": when expressivity and multiplicity are sufficient, the output symmetry is exactly that predictable infimum, and directional information is preserved.

## Key Experimental Results

### Feature Space Visualization (§6.1)
Using a single-layer randomly initialized TFN to encode 3-fold structures (placed in $m=6$ planes, each plane rotated discretely into $res=49/98$ parts). Dimensionality reduction visualization for different feature orders $l_0$ exactly matches the theoretical predictions of Ex 4.3:

| Feature Order $l_0$ | Observed Degeneration | Corresponding Infimum |
|---------------------|-----------------------|-----------------------|
| $l_0 = 0, 1$        | Full                  | $(O(3) \times S_{k+1})$ |
| $l_0 = 2, 4$        | Axial                 | $(D_{\infty h} \times S_{k+1})$ |
| $l_0 = 10$           | Half                  | $(D_{2kh} \times S_{k+1})$ |

For Half degeneration under $res=98$ vs. $res=49$, the overall shapes are identical, but data points from the second half of the rotation completely overlap the first half, visually validating the "inability to distinguish self from self-rotated by $\pi/k$."

### Symmetric Graph Discrimination (§6.2)
Constructing $k$-fold structures $G_0$ for $k \in \{2,3,4,6\}$, with $G_1$ as a randomly rotated version. The norm of the embedding difference measures if the ENN can distinguish them (tested for 2D/3D rotations using 12 channel/layer configurations each for TFN and HEGNN):

| Phenomenon | Value | Meaning |
|------------|-------|---------|
| Distinguishable | Difference norm $> 10^{-3}$ | Structures are separated |
| Indistinguishable | Difference norm $< 10^{-6}$ | Degeneration collapse |
| Gap | $> 10^3$ force difference | Clear binary pattern |

Results show a clean binary distribution. Since the maximum value across all configurations remains $< 10^{-6}$ for certain cases, it indicates that degeneration is **independent** of model choice, channels, or layers—it is a structural rather than training issue, which precisely matches theoretical predictions.

### QM9 Molecule Property Prediction (§6.3)
Predicting isotropic polarizability $\alpha$ using HEGNN as the backbone. A shared encoder is pre-trained on $l \leq 11$, followed by two fine-tuning strategies for 12 prediction heads each: (a) only $l=l_0$, (b) all $l \leq l_0$.

### Key Findings
- In most samples, different feature components contribute similarly; however, for non-trivial components where molecular symmetry increases to $O(3)$, the prediction MAE is significantly higher—confirming "increase to full group $\Rightarrow$ component annihilation $\Rightarrow$ loss of discriminative power."
- For symmetries that undergo full degeneration on 1st-order features, adding more 1st-order features brings almost no improvement, directly validating the design guidelines of §4.2.
- $k$-fold degeneration is a structural issue: changing models, adding channels, or adding layers cannot save it; it must be avoided beforehand through the orbit types of the feature space.

## Highlights & Insights
- It unifies the scattered empirical phenomenon of "equivariant networks losing directional information" into a uniquely existing **symmetry infimum** and proves it is determined solely by the algebraic structure of the feature space—a paradigm shift from "post-hoc observation" to "prior prediction."
- The elevation of Michel's Criterion from "necessary" to "sufficient" under high-multiplicity representations (Prop 4.2) is the key technical bridge that turns abstract group theory conclusions into computable algorithms (chain recursion + trace formulas), reusable for other geometric deep learning analyses requiring isotropy subgroup identification.
- The treatment of density + almost isovariance is elegant: using the manifold hypothesis and measure-zero sets to bypass counterexamples where "perfect isovariance does not exist," proving that "sufficiently expressive networks generically attain the infimum," allowing pure theory to apply to practical architectures like TFN/HEGNN.
- The classification of the three types of $k$-fold degeneration (Full/Axial/Half), along with their precise switching tables based on $l_0$ and $k$ parity, provides ENN designers with a "look-up table" guide for selecting irreps that do not lose information.

## Limitations & Future Work
- The sufficiency conclusion depends on the **high-multiplicity hypothesis** ($m(V, V_i) > \dim G$) and the **manifold hypothesis**; conclusions might degrade for low-multiplicity or non-manifold data (Cex D.3 suggests perfect isovariance might not exist when multiplicity is insufficient).
- The complete orbit type calculation and the three-tier degeneration classification mainly focus on $SO(3)/O(3)$ and $k$-fold structures; extending to more complex groups (e.g., product groups with permutations in more general tasks) still requires case-by-case analysis.
- Experiments focused on synthetic data and a single QM9 property (polarizability). Whether "following design guidelines to select features" brings substantial gains in larger-scale, higher-dimensional scientific tasks requires broader validation.
- The guidelines provide qualitative directions for "avoiding harmful increase"; an automated search for the optimal combination of feature components under fixed computational/channel budgets is a natural engineering extension.

## Related Work & Insights
- **vs. Cen et al. (collapse-to-zero)**: Their theory only covers the extreme case of full degeneration; this paper's infimum framework unifies Full/Axial/Half degenerations, where full degeneration is a special case where $I_G = (O(3) \times S_{k+1})$.
- **vs. Joshi et al. (empirical observations)**: They empirically found that degeneration depends on the feature space but provided no theoretical explanation; this paper provides computable orbit type decisions that precisely predict which type of degeneration occurs at which $l_0$, replicating their experimental settings.
- **vs. Kaba & Ravanbakhsh (relaxing equivariance)**: They use orbit types to describe the phenomenon, but their solutions often involve relaxing equivariance itself; this paper remains within the equivariant framework, avoiding degeneration through feature design rather than breaking equivariance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifying symmetry increase into a unique infimum and providing a computable algorithm is the first rigorous and predictive theoretical framework for this problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic visualization + graph discrimination + QM9 tri-layer validation match theoretical predictions, though real-world task scale and diversity could be broader.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are progressive and definitions are clear, though the high density of group theory poses a threshold for non-geometric deep learning readers.
- Value: ⭐⭐⭐⭐⭐ Provides an actionable guide for "which irreps to choose to keep directional information," directly impacting the feature design of equivariant networks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] On Universality of Deep Equivariant Networks](on_universality_of_deep_equivariant_networks.md)
- [\[ICLR 2026\] Quasi-Equivariant Metanetworks](quasi-equivariant_metanetworks.md)
- [\[ICLR 2026\] Proper Velocity Neural Networks](proper_velocity_neural_networks.md)
- [\[ICLR 2026\] The Logical Expressiveness of Topological Neural Networks](the_logical_expressiveness_of_topological_neural_networks.md)
- [\[ICLR 2026\] To Augment or Not to Augment? Diagnosing Distributional Symmetry Breaking](to_augment_or_not_to_augment_diagnosing_distributional_symmetry_breaking.md)

</div>

<!-- RELATED:END -->
