---
title: >-
  [Paper Note] Addressing Divergent Representations from Causal Interventions on Neural Networks
description: >-
  [ICLR 2026 Oral][causal intervention] This paper systematically demonstrates that causal interventions (activation patching, DAS, SAEs…
tags:
  - "ICLR 2026 Oral"
  - "causal intervention"
  - "mechanistic interpretability"
  - "representational divergence"
  - "Counterfactual Latent loss"
  - "DAS"
date: 2026-05-08
content_hash: a33d8fb4d6e21f07
---

# Addressing Divergent Representations from Causal Interventions on Neural Networks

**Conference**: ICLR 2026 Oral
**arXiv**: [2511.04638](https://arxiv.org/abs/2511.04638)  
**Code**: [GitHub](https://github.com/grantsrb/rep_divergence)  
**Area**: Other
**Keywords**: causal intervention, mechanistic interpretability, representational divergence, Counterfactual Latent loss, DAS

## TL;DR

This paper systematically demonstrates that causal interventions (activation patching, DAS, SAEs, etc.) push model internal representations off their natural distribution. It theoretically distinguishes "benign shifts" from "harmful shifts," proposes the Counterfactual Latent (CL) loss to constrain intervened representations to remain near the natural manifold, and validates on a 7B LLM that this approach reduces divergence while preserving intervention accuracy.

## Background & Motivation

**Background**: Causal intervention is the core methodology of mechanistic interpretability—manipulating internal representations via activation patching, DAS, SAEs, and related techniques to infer what those representations encode. Even correlation-based methods such as SAEs and PCA typically rely on causal intervention as the ultimate arbiter of whether a feature is genuinely meaningful. Causal intervention thus occupies a central role in claims about functional mechanisms.

**Limitations of Prior Work**: These causal intervention methods implicitly assume—without verification—that the counterfactual model states produced by intervention are "realistic" for the target model. For instance, some activation patching experiments amplify feature values by a factor of 15, under which conditions the intervened representations are likely to deviate severely from the model's natural distribution.

**Key Challenge**: If intervened representations are out-of-distribution, downstream layers may respond to these OOD inputs by activating hidden pathways never encountered during training, causing the observed causal effects to be spurious. What appears to be a discovery of the model's natural mechanism may in fact be an artifact of the intervention.

**Key Insight**: The authors proceed along both theoretical and empirical dimensions simultaneously: (1) establishing that divergence is pervasive; (2) distinguishing when divergence is benign versus harmful; (3) proposing a mitigation strategy. This constitutes a meta-level examination of the interpretability methodology itself.

**Core Idea**: Not all shifts are harmful—shifts within the behavioral null space are benign, whereas shifts that activate hidden pathways or trigger dormant behavioral changes are harmful. By constraining intervened representations to remain near the natural manifold via CL loss, harmful divergence can be systematically mitigated.

## Method

### Overall Architecture

The paper's logical structure proceeds in four steps:

1. **Establishing the pervasiveness of divergence** (Section 3): theoretical proofs + empirical demonstration across three mainstream intervention methods.
2. **Distinguishing benign vs. harmful divergence** (Section 4): behavioral null space theory + hidden pathways + dormant behavioral changes.
3. **Proposing CL loss as a mitigation** (Section 5.1): applied to Boundless DAS on a 7B LLM.
4. **Improving CL loss for OOD generalization** (Section 5.2): a modified CL loss that constrains only the causal subspace.

### Key Design 1: Theoretical Guarantees of Divergence

For coordinate-level patching, the authors prove that divergence is inevitable unless the manifold is an axis-aligned hyperrectangle. Consider the circular manifold $\mathcal{M}_K = \{c_K + u : \|u\|_2 \leq r_K\}$. Concatenating the first coordinate of $h^{\text{src}}$ with the second coordinate of $h^{\text{trg}}$ gives:

$$\hat{h} = \begin{bmatrix} h_1^{\text{src}} \\ h_2^{\text{trg}} \end{bmatrix}, \quad \|\hat{h} - c_K\|_2^2 = u_1^2 + v_2^2$$

Taking boundary points $u = (r_K, 0)$ and $v = (0, r_K)$ yields $\|\hat{h} - c_K\| = r_K\sqrt{2} > r_K$, placing the intervened representation outside the manifold boundary.

**Theorem A.2** further establishes that a non-empty convex set is patch-closed if and only if it is the Cartesian product of its coordinate projections (i.e., an axis-aligned hyperrectangle). Consequently, common manifold geometries—spheres, ellipsoids, general polytopes—all produce divergence under coordinate patching. This is a strong negative result.

### Key Design 2: Behavioral Null Space and Benign Shifts

The behavioral null space of a function $\psi: \mathbb{R}^d \to \mathbb{R}^{d'}$ with respect to a set $X$ is defined as:

$$\mathcal{N}(\psi, X) = \{v \in \mathbb{R}^d \mid \forall x \in X,\ \psi(x+v) = \psi(x)\}$$

If a shift $v \in \mathcal{N}(\psi, X)$, i.e., $\psi(x+v) = \psi(x)$, then the shift is harmless with respect to $\psi$'s overall computation—equivalent to adding a zero vector. However, the authors emphasize that **harmlessness is granularity-dependent**: a shift benign at the level of the full function may be harmful to sub-computations, since intermediate-layer representations may already differ.

The paper also introduces the notion of a "behaviorally binary subspace": if a subspace influences the output only through its sign, then value changes within that subspace are benign as long as $\text{sign}(D_{\text{var}} \mathcal{A}(h))$ remains unchanged, even if the resulting value combinations have never appeared in the natural distribution.

### Key Design 3: Hidden Pathways and Harmful Shifts

Two forms of harmful divergence are demonstrated via constructive proofs:

**(a) Hidden pathway activation**: A two-layer ReLU network is constructed, $s = \mathbf{1}^\top \text{ReLU}(W_\ell h^\ell + b_\ell)$, with weight matrix $W_\ell \in \mathbb{R}^{3 \times 4}$. Under natural representations, the third hidden unit is never activated (its pre-activation is always negative). Mean-difference patching ($\delta_{B \to A} = \mu_A - \mu_B$) produces an intervened representation that activates this unit, flipping the classification decision through a pathway never used under natural inputs. Projecting the intervened representation back onto $\text{conv}(S_A)$ eliminates the effect, confirming that it is driven by the shift rather than a genuine causal mechanism.

**(b) Dormant behavioral changes**: The above network is extended with a context vector $v$ and a second layer. The intervention behaves normally when $v_4 < 0.75$ (predicting class A), but triggers an anomalous class C prediction when $0.75 < v_4 < 1.0$—whereas natural representations only produce class C when $v_4 > 1$. Dormant behavioral changes make intervention safety context-dependent, and exhaustive enumeration of contexts is infeasible. These are formally defined as $\mathcal{V}(\psi, X, \mathcal{C}_1, \mathcal{C}) = \mathcal{N}(\psi, X, \mathcal{C}_1) \setminus \mathcal{N}(\psi, X, \mathcal{C})$.

### Loss & Training: Counterfactual Latent (CL) Loss

**Original CL loss** (from Grant 2025), combining L2 and cosine distance:

$$\mathcal{L}_{\text{CL}}(\hat{h}, h_{\text{CL}}) = \frac{1}{2}\|\hat{h} - h_{\text{CL}}\|_2^2 - \frac{1}{2}\frac{\hat{h} \cdot h_{\text{CL}}}{\|\hat{h}\|_2 \|h_{\text{CL}}\|_2}$$

where $h_{\text{CL}}$ is a counterfactual latent vector obtained by averaging natural representations that share the same causal variable values: $h_{\text{CL}} = \frac{1}{m} \sum_{i=1}^{m} h_{\text{CL}}^{(x_i)}$. The total loss is $\mathcal{L}_{\text{total}} = \epsilon \mathcal{L}_{\text{CL}} + \mathcal{L}_{\text{DAS}}$, where $\epsilon$ is a tunable hyperparameter.

**Improved CL loss**, constraining only the causal subspace dimensions and usable independently of the behavioral loss:

$$\mathcal{L}'_{\text{CL}} = \sum_{i=1}^{n} \left(\frac{1}{2}\|\hat{h}^{\text{var}_i} - h_{\text{CL}}^{\text{var}_i}\|_2^2 - \frac{1}{2}\frac{\hat{h}^{\text{var}_i} \cdot h_{\text{CL}}^{\text{var}_i}}{\|\hat{h}^{\text{var}_i}\|_2 \|h_{\text{CL}}^{\text{var}_i}\|_2}\right)$$

where $\hat{h}^{\text{var}_i} = \mathcal{A}^{-1}(D_{\text{var}_i} \mathcal{A}(\hat{h}))$ denotes the component of the intervened representation in causal subspace $i$, and $h_{\text{CL}}^{\text{var}_i}$ is treated with stop-gradient to prevent gradient flow.

## Key Experimental Results

### Main Results: Pervasiveness of Divergence (Section 3.2)

| Intervention Method | Model | Layer | EMD | Significant Divergence |
|---|---|---|---|---|
| Mean Diff Vector Patching | Llama-3-8B-Instruct | L10 (lowest-EMD layer) | Significantly above natural baseline | ✓ |
| SAE Reconstruction | Llama-3-8B-Instruct | L25 | Significantly above natural baseline | ✓ |
| Boundless DAS | wu2024 setting | Designated layer | Significantly above natural baseline | ✓ |

All three mainstream methods show clear divergence of intervened representations from the natural distribution, as confirmed by PCA visualization and Earth Mover's Distance quantification. Additional metrics—nearest-neighbor cosine distance, L2 pairwise distance, Local PCA Distance, KDE Density Score, and Local Linear Reconstruction Error—consistently corroborate these findings.

### CL Loss on Boundless DAS (7B LLM) (Section 5.1)

| CL weight $\epsilon$ | IIA (intervention accuracy) | EMD (divergence) | Notes |
|---|---|---|---|
| 0 (no CL) | Baseline IIA | Higher | Original DAS |
| Small $\epsilon$ | Maintained or slightly improved | Markedly reduced | **Optimal range** |
| Large $\epsilon$ | IIA degrades | Lowest | CL overpowers behavioral signal |

Key finding: a sweet spot exists at small $\epsilon$ where divergence is substantially reduced without sacrificing IIA.

### Improved CL Loss on Synthetic Tasks (Section 5.2)

| Method | EMD (feature dimensions) | IIA | OOD Generalization |
|---|---|---|---|
| DAS behavioral loss | 0.032 ± 0.003 | 0.997 ± 0.001 | Lower |
| Improved CL loss | **0.007 ± 0.001** | **0.9988 ± 0.0005** | **Higher** |

CL loss reduces EMD by approximately 4.5×, with a slight improvement in IIA. In the OOD setting (transferring alignment matrices between dense/sparse subtasks), alignments trained with CL loss substantially outperform those trained with behavioral loss alone. Regression analysis confirms that EMD and OOD IIA are negatively correlated (coefficient −0.34, $R^2 = 0.73$, $p < 0.001$), establishing that reducing divergence has genuine practical value.

### Key Findings

- Divergence is not specific to individual methods but is a systemic property of causal interventions.
- Hidden pathways can produce behaviorally "correct" outcomes while relying entirely on non-natural mechanisms—the most dangerous failure mode.
- Dormant behavioral changes make intervention safety context-dependent, and the context space cannot be exhaustively enumerated.
- CL loss provides a simple and effective preliminary mitigation with the added benefit of OOD generalization.

## Highlights & Insights

1. **Meta-methodological contribution**: Rather than applying interpretability tools to analyze models, this work audits the reliability of those tools themselves—with far-reaching implications for the methodological foundations of the field.

2. **The "hidden pathway" concept**: Interventions may activate computational paths that are never used under natural inputs, leading to conclusions that are behaviorally correct but mechanistically wrong. This directly challenges the common assumption that high IIA implies correct mechanism discovery.

3. **A principled benign/harmful framework**: The behavioral null space theory provides a principled criterion for judging whether a shift is harmful, rather than treating all divergence as problematic.

4. **Elegance of Theorem A.2**: Only axis-aligned hyperrectangles are patch-closed—for virtually all realistic manifold geometries, coordinate patching inevitably produces divergence.

5. **Practicality**: CL loss is straightforward to implement, can be inserted into existing DAS pipelines, and has been validated on a 7B LLM.

## Limitations & Future Work

1. **No automatic classification of harmful divergence**: The paper provides no method for automatically distinguishing benign from harmful shifts, limiting practical applicability.
2. **CL loss is a broad-spectrum strategy**: It reduces all divergence indiscriminately (including benign shifts) rather than precisely targeting harmful ones.
3. **Improved CL loss validated only on simple synthetic tasks**: The 10-class synthetic dataset is far removed from real LLM scenarios.
4. **Restricted to linear alignment functions**: Sutter et al. identify more fundamental problems with nonlinear alignment functions, which this paper does not address.
5. **CL vector acquisition requires annotation**: Identifying which natural representations share the same causal variable values is difficult to obtain in complex settings.
6. **Directions for future work**: (a) online divergence detection via ReLU activation pattern auditing; (b) combining manifold projection with CL loss; (c) self-supervised discovery of harmful divergence.

## Related Work & Insights

- **Makelov et al. (2023)**: Previously identified interactions between null spaces and dormant subspaces in DAS; this paper generalizes their concerns to a broader class of causal intervention methods.
- **Zhang et al. (2024) / Heimersheim (2024)**: Noted that patching results are easily misinterpreted; this paper offers a new theoretical explanation from the perspective of representational divergence.
- **Sutter et al. (2025)**: Questioned the meaning of causal interventions under nonlinear alignment functions; their findings are complementary to this work.
- **Grant (2025)**: Original source of the CL loss, extended here to the causal subspace level.
- **Implications for SAE research**: SAE reconstruction is itself an intervention that produces divergence, casting doubt on the "causal validation" step commonly applied to SAE features.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Challenges foundational methodological assumptions of interpretability research; a significant meta-level contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Theoretical proofs are rigorous and LLM experiments are meaningful, but the improved method is validated only on synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Problem formulation is precise, reasoning is clear, and theory and experiments are tightly integrated.
- **Value**: ⭐⭐⭐⭐⭐ — Broad implications for causal intervention experiments in mechanistic interpretability; Oral recognition is well deserved.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/others/learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[NeurIPS 2025\] SAD Neural Networks: Divergent Gradient Flows and Asymptotic Optimality via o-minimal Structures](../../NeurIPS2025/others/sad_neural_networks_divergent_gradient_flows_and_asymptotic_optimality_via_o-min.md)
- [\[ICLR 2026\] Exchangeability of GNN Representations with Applications to Graph Retrieval](exchangeability_gnn_representations.md)
- [\[ICLR 2026\] Entropic Confinement and Mode Connectivity in Overparameterized Neural Networks](entropic_confinement_and_mode_connectivity_in_overparameterized_neural_networks.md)
- [\[ICLR 2026\] On the Lipschitz Continuity of Set Aggregation Functions and Neural Networks for Sets](on_the_lipschitz_continuity_of_set_aggregation_functions_and_neural_networks_for.md)

</div>

<!-- RELATED:END -->
