---
title: >-
  [Paper Note] Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model
description: >-
  [NeurIPS 2025][Image Generation][Optimal Transport] This paper proposes A2A-FM, a method that simultaneously learns optimal transport mappings across all pairs of conditional distributions within the Flow Matching framework via a novel cost function. It is theoretically shown to converge to pairwise optimal transport in the infinite-sample limit, and is particularly suited for non-grouped data with continuous conditional variables.
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Optimal Transport"
  - "Flow Matching"
  - "Condition Transfer"
  - "Pairwise Optimal Transport"
  - "Molecular Optimization"
date: 2026-05-08
content_hash: 06f6d924f4ba093f
---

# Pairwise Optimal Transports for Training All-to-All Flow-Based Condition Transfer Model

**Conference**: NeurIPS 2025
**arXiv**: [2504.03188](https://arxiv.org/abs/2504.03188)  
**Code**: [GitHub](https://github.com/kotatumuri-room/A2A-FM)  
**Area**: Diffusion Models / Flow Matching
**Keywords**: Optimal Transport, Flow Matching, Condition Transfer, Pairwise Optimal Transport, Molecular Optimization

## TL;DR

This paper proposes A2A-FM, a method that simultaneously learns optimal transport mappings across all pairs of conditional distributions within the Flow Matching framework via a novel cost function. It is theoretically shown to converge to pairwise optimal transport in the infinite-sample limit, and is particularly suited for non-grouped data with continuous conditional variables.

## Background & Motivation

Condition transfer is a central task in conditional generative modeling: given a sample from a conditional distribution $P_{c_1}$, generate a sample $x \sim P_{c_2}$ satisfying a target condition $c_2$. Typical applications include image style transfer and molecular property modification.

Existing methods face two key challenges:

**Continuous condition problem**: When the conditioning variable $c$ is continuous, each condition $c$ may correspond to only a single observed sample $x$, making it impossible to estimate $P_c$ for individual conditions. Most existing methods (e.g., Multimarginal SI, EFM) require *grouped data*, i.e., sufficiently many i.i.d. samples per condition.

**All-to-all scalability**: The number of condition pairs $(c_1, c_2)$ can be infinite, making it computationally intractable to learn transport maps pairwise.

**Key Challenge**: How to learn optimal transport mappings between arbitrary condition pairs from non-grouped data?

**Key Insight**: The paper extends conditional optimal transport (COT) techniques to the pairwise setting, designing a novel cost function such that mini-batch-level couplings converge to optimal transport across all condition pairs.

## Method

### Overall Architecture

A2A-FM builds upon the Flow Matching framework, learning a velocity field $v(x, t | c_1, c_2)$ parameterized by $(c_1, c_2)$ to transport $P_{c_1}$ to $P_{c_2}$ via an ODE:

$$\dot{x}_{c_1, c_2}(t) = v(x_{c_1, c_2}(t), t | c_1, c_2)$$

where $x_{c_1,c_2}(0) \sim P_{c_1}$ and $x_{c_1,c_2}(1) \sim P_{c_2}$.

### Key Designs

1. **Pairwise Optimal Transport Cost Function**: The core innovation of A2A-FM lies in its coupling strategy. Two batches $B_1$ and $B_2$ are independently drawn from the dataset $D = \{(x^{(i)}, c^{(i)})\}$, and the optimal coupling $\pi_\beta^*$ is obtained by minimizing the following cost function:

$$\sum_{i=1}^N \|x_1^{(i)} - x_2^{\pi(i)}\|^2 + \beta \left(\|c_1^{(i)} - c_1^{\pi(i)}\|^2 + \|c_2^{(i)} - c_2^{\pi(i)}\|^2\right)$$

The key insight is that the $\beta$ term jointly constrains matching of **both source and target conditions**, rather than a single condition alone (distinguishing it from the COT cost function in Eq. (6)). A large $\beta$ encourages pairing samples with similar $(c_1, c_2)$; a small $\beta$ allows transport information to be shared across different condition pairs.

2. **Theoretical Guarantee (Proposition 3.1)**: It is proven that as $\beta \to \infty$ and sample size $N$ grows correspondingly, the optimal coupling under the above cost converges to the true pairwise optimal transport. That is, for almost all $(c_1, c_2)$:

$$\int \|x_1 - x_2\|^2 d\Pi^*(x_1, x_2 | c_1, c_2) = W_2^2(P_{c_1}, P_{c_2})$$

This implies that, given sufficient data, mini-batch approximations can capture the true optimal transport between each pair of conditional distributions.

3. **Applicability to Non-Grouped Data**: A key advantage of A2A-FM is that it requires neither grouping nor discretization of conditions. Through the balancing role of $\beta$, the method can approximate pairwise optimal transport by sharing information across samples with nearby conditions, even when each condition has only one sample. In practice, $\beta = N^{1/(2d_c)}$ (where $d_c$ is the condition dimensionality) serves as an effective heuristic.

### Loss & Training

The training procedure follows standard CFM:
1. Sample batches $B_1, B_2$ from the dataset.
2. Obtain the optimal coupling $\pi_\beta^*$ by minimizing the above cost via the OPTC algorithm.
3. Construct linear paths $\psi_i(t) = (1-t)x_1^{(i)} + tx_2^{\pi_\beta^*(i)}$.
4. Update velocity field parameters $\theta$ by minimizing the FM loss $L(\theta) = \sum_i \|v_\theta(\psi_i(t_i), t_i | c_1^{(i)}, c_2^{\pi_\beta^*(i)}) - \dot{\psi}_i(t_i)\|^2$.

At inference, condition transfer is performed by solving the ODE from $t=0$ to $t=1$.

## Key Experimental Results

### Synthetic Data Validation

| Data Setting | Method | MSE w.r.t. Pairwise OT |
|---|---|---|
| Grouped | A2A-FM | **(5.81±2.22)×10⁻²** |
| Grouped | Generalized geodesic | (1.03±0.04)×10⁰ |
| Non-grouped | A2A-FM | **(1.51±0.17)×10⁻²** |
| Non-grouped | Partial diffusion | (6.77±0.14)×10⁻² |
| Non-grouped | Multimarginal SI | (4.90±0.28)×10⁻² |

### Molecular Optimization (QED Neighbor Sampling)

| Method | Success Rate (%) |
|---|---|
| A2A-FM | **97.5** |
| COATI-LDM | 95.6 |
| MolMIM | 94.6 |
| QMO | 92.8 |
| DESMILES | 76.9 |

### LogP-TPSA Multi-Attribute Transfer (AUC)

| Method | AUC |
|---|---|
| A2A-FM | **0.990** |
| OT-CFM | 0.819 |
| SI (K=10) | 0.583 |
| PD+CFG (T=300) | 0.450 |

### Key Findings

- On both grouped and non-grouped data, the couplings and learned velocity fields of A2A-FM are closer to the true pairwise optimal transport.
- Multimarginal SI degrades significantly on non-grouped data due to discretization; Partial Diffusion produces near-random couplings.
- In molecular optimization, A2A-FM achieves higher success rates with fewer oracle calls, substantially outperforming baselines in sampling efficiency.
- The antisymmetry constraint $v_{c_1,c_2} = -v_{c_2,c_1}$ improves the success rate from 94.6% to 97.5% in the QED experiment.

## Highlights & Insights

- The elegance of the cost function design lies in jointly constraining both source and target conditions, enabling transfer across different condition pairs beyond what the single-condition COT cost supports.
- The theoretical result is clean: the limiting behavior as $\beta \to \infty$ aligns with the intuitive interpretation for finite $|\mathcal{C}|$ and generalizes naturally to continuous conditions.
- The connection to the functional representation theorem provides a deeper understanding of why pairwise OT is effective for condition transfer.
- Computational cost scales with $|D|$ rather than the number of condition pairs $K^2$, making the method more scalable than approaches that require grouped data.

## Limitations & Future Work

- The choice of $\beta$ still involves a trade-off between approximation accuracy and OT quality; while $\beta = N^{1/(2d_c)}$ is an effective heuristic, its optimality is not rigorously guaranteed.
- Cycle consistency ($T_{c_2 \to c_3} \circ T_{c_1 \to c_2} = T_{c_1 \to c_3}$) is not ensured, as OT itself does not satisfy this property.
- Convergence rates may slow when the condition dimensionality $d_c$ is large.
- Experiments are conducted at relatively limited scales; performance on large-scale image datasets remains unvalidated.

## Related Work & Insights

- **Relation to OT-CFM**: A2A-FM can be viewed as generalizing OT-CFM from unidirectional transport (source to target) to all-to-all transport between arbitrary condition pairs.
- **COT methods** (Chemseddine et al.) provide technical inspiration for the proof strategy, but their cost function supports conditional generation rather than condition transfer.
- The method has significant application potential in drug design, materials science, and other domains where conditions are continuous physical quantities.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Introduces a novel pairwise OT cost function with rigorous theoretical proof, filling the gap in condition transfer for non-grouped data.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic data validates the theory and molecular optimization demonstrates practical value, though broader domain experiments are lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are rigorous and clear, intuitions are well-explained, and distinctions from related work are thoroughly discussed.
- **Value**: ⭐⭐⭐⭐⭐ Addresses a fundamental problem in condition transfer with broad applicability; the chemistry application demonstrates real-world impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Counterfactual Identifiability via Dynamic Optimal Transport](counterfactual_identifiability_via_dynamic_optimal_transport.md)
- [\[NeurIPS 2025\] On the Relation between Rectified Flows and Optimal Transport](on_the_relation_between_rectified_flows_and_optimal_transport.md)
- [\[NeurIPS 2025\] GuideFlow3D: Optimization-Guided Rectified Flow For Appearance Transfer](guideflow3d_optimization-guided_rectified_flow_for_appearance_transfer.md)
- [\[ICCV 2025\] The Curse of Conditions: Analyzing and Improving Optimal Transport for Conditional Flow-Based Generation](../../ICCV2025/image_generation/the_curse_of_conditions_analyzing_and_improving_optimal_transport_for_conditiona.md)
- [\[ECCV 2024\] AFreeCA: Annotation-Free Counting for All](../../ECCV2024/image_generation/afreeca_annotation-free_counting_for_all.md)

</div>

<!-- RELATED:END -->
