---
title: >-
  [Paper Note] Addressing Divergent Representations from Causal Interventions on Neural Networks
description: >-
  [ICLR 2026][Interpretability][causal intervention] This work systematically reveals that causal interventions (such as activation patching, DAS, and SAE) push internal model representations away from their natural distributions. It theoretically distinguishes between "harmless" and "harmful" shifts and proposes the Counterfactual Latent (CL) loss to constrain intervene
tags:
  - ICLR 2026
  - Interpretability
  - causal intervention
  - mechanistic interpretability
  - representational divergence
  - Counterfactual Latent loss
  - DAS
date: 2026-05-08
content_hash: 9ecae14827b59b4c
---
# Addressing Divergent Representations from Causal Interventions on Neural Networks

**Conference**: ICLR 2026 Oral  
**arXiv**: [2511.04638](https://arxiv.org/abs/2511.04638)  
**Code**: [GitHub](https://github.com/grantsrb/rep_divergence)  
**Area**: Others  
**Keywords**: causal intervention, mechanistic interpretability, representational divergence, Counterfactual Latent loss, DAS  

## TL;DR

This work systematically reveals that causal interventions (such as activation patching, DAS, and SAE) push internal model representations away from their natural distributions. It theoretically distinguishes between "harmless" and "harmful" shifts and proposes the Counterfactual Latent (CL) loss to constrain intervened representations within the natural manifold. Evaluations on 7B LLMs demonstrate that this approach reduces divergence while maintaining intervention accuracy.

## Background & Motivation

**Background**: The core methodology of mechanistic interpretability is causal intervention—manipulating internal representations via activation patching, DAS, or SAE to observe behavioral changes and infer what features are encoded. Even correlational methods like SAE or PCA often use causal interventions as the final arbiter to verify if features are truly meaningful. Causal interventions occupy a central position in making functional mechanistic claims.

**Limitations of Prior Work**: These intervention methods implicitly assume that the counterfactual model states generated are "realistic" for the target model. For instance, some activation patching experiments scale feature values by 15x, in which case the intervened representations likely deviate significantly from the model's natural distribution.

**Key Challenge**: If intervened representations are out-of-distribution (OOD), the response of subsequent layers to these OOD inputs may activate "hidden pathways" never seen during training. This leads to observed causal effects being spurious—what is perceived as the model's natural mechanism may actually be an artifact of the intervention.

**Key Insight**: The authors approach this from both theoretical and experimental dimensions: (1) first proving that divergence is a universal phenomenon; (2) then distinguishing when divergence is harmless versus harmful; (3) and finally proposing a mitigation strategy. This represents a meta-level scrutiny of the entire interpretability methodology.

**Core Idea**: Not all shifts are harmful—shifts within the behavioral null space are harmless, but shifts that activate hidden pathways or trigger dormant behavioral changes are harmful. By using CL loss to constrain intervened representations to the natural manifold, harmful shifts can be systematically mitigated.

## Method

### Overall Architecture

The paper follows a logical chain of "diagnosis, triage, and prescription": it first proves through theory and empirical evidence that causal interventions generally push representations off the natural manifold, then utilizes behavioral null space theory to categorize shifts into harmless and harmful types, and finally introduces the Counterfactual Latent (CL) loss to pull intervened representations back toward the manifold. The core of the method is not a new model, but a set of criteria to determine if intervention results are trustworthy, supplemented by a lightweight regularizer.

### Key Designs

**1. Theoretical Guarantee of Divergence: Proving Coordinate-wise Patching Almost Certainly Leaves the Manifold**

The issue lies in the default assumption that the counterfactual state obtained by patching two real representations remains "realistic." The authors prove that as long as the manifold is not an axis-aligned hyperrectangle, this assumption fails. Consider a cyclic manifold $\mathcal{M}_K = \{c_K + u : \|u\|_2 \leq r_K\}$. Patching the first coordinate of $h^{\text{src}}$ and the second coordinate of $h^{\text{trg}}$ to form $\hat{h} = [h_1^{\text{src}};\, h_2^{\text{trg}}]$ yields $\|\hat{h} - c_K\|_2^2 = u_1^2 + v_2^2$. Taking boundary points $u=(r_K,0)$ and $v=(0,r_K)$ results in $\|\hat{h}-c_K\| = r_K\sqrt{2} > r_K$; thus, the intervened representation directly crosses the manifold boundary. More generally, Theorem A.2 proves that a non-empty convex set is patch-closed if and only if it is the Cartesian product of its coordinate projections (an axis-aligned hyperrectangle). Consequently, for geometries like spheres, ellipsoids, or general polyhedra—where real representations almost always reside—coordinate patching inevitably causes divergence.

**2. Behavioral Null Space: Defining Which Shifts are Harmless**

Since divergence is unavoidable, the key is whether it changes the computation. The authors define the behavioral null space $\mathcal{N}(\psi, X) = \{v \in \mathbb{R}^d \mid \forall x \in X,\ \psi(x+v) = \psi(x)\}$ for a function $\psi: \mathbb{R}^d \to \mathbb{R}^{d'}$ relative to a set $X$. If a shift $v$ falls within this null space, it is equivalent to adding a zero vector to the overall computation of $\psi$ and is thus harmless. However, harmlessness is highly dependent on the granularity of the claim—a shift harmless to the overall function might have already altered sub-computational representations in intermediate layers, making it harmful to finer mechanistic claims. This can be extended to "behavioral binary subspaces": if a subspace only affects output via signs, any value within the subspace is harmless as long as $\text{sign}(D_{\text{var}} \mathcal{A}(h))$ remains unchanged, even if that value combination never appeared in the natural distribution.

**3. Hidden Pathways and Dormant Behavioral Changes: Identifying Dangerous Shifts**

Harmful divergence is perilous because the behavior may appear correct while the mechanism is faked. The authors demonstrate this with two constructive counter-examples. The first is hidden pathway activation: constructing a two-layer ReLU network $s = \mathbf{1}^\top \text{ReLU}(W_\ell h^\ell + b_\ell)$ with $W_\ell \in \mathbb{R}^{3\times4}$, where the pre-activation of the third hidden unit is always negative (dormant) under natural representations. Mean difference patching ($\delta_{B \to A} = \mu_A - \mu_B$) could activate this unit, using a pathway never used by natural inputs to flip the classification decision. Projecting the intervened representation back to $\text{conv}(S_A)$ makes the effect disappear, proving the flip was driven by divergence rather than a true causal mechanism. The second is dormant behavioral changes: in the same network, interventions behave normally for context $v_4 < 0.75$, but trigger an abnormal class C for $0.75 < v_4 < 1.0$ that should only appear when $v_4 > 1$. This is formalized as $\mathcal{V}(\psi, X, \mathcal{C}_1, \mathcal{C}) = \mathcal{N}(\psi, X, \mathcal{C}_1) \setminus \mathcal{N}(\psi, X, \mathcal{C})$, implying intervention safety drifts with context. Since the context space cannot be exhausted, this is the most difficult type to defend against.

### Loss & Training

The mitigation strategy adds a Counterfactual Latent (CL) loss to alignment training, pulling the intervened representation $\hat{h}$ toward a counterfactual latent vector $h_{\text{CL}}$. The original version (from Grant 2025) penalizes both L2 distance and cosine angle:

$$\mathcal{L}_{\text{CL}}(\hat{h}, h_{\text{CL}}) = \frac{1}{2}\|\hat{h} - h_{\text{CL}}\|_2^2 - \frac{1}{2}\frac{\hat{h} \cdot h_{\text{CL}}}{\|\hat{h}\|_2 \|h_{\text{CL}}\|_2}$$

where $h_{\text{CL}}$ is the mean of vectors in the natural representation sharing the same causal variable values $h_{\text{CL}} = \frac{1}{m} \sum_{i=1}^{m} h_{\text{CL}}^{(x_i)}$, serving as a manifold anchor. The total loss $\mathcal{L}_{\text{total}} = \epsilon \mathcal{L}_{\text{CL}} + \mathcal{L}_{\text{DAS}}$ uses a weight $\epsilon$ to balance the regularizer with the original behavioral goal. To allow use independent of behavioral loss, the authors provide an improved version focusing only on causal subspace dimensions:

$$\mathcal{L}'_{\text{CL}} = \sum_{i=1}^{n} \left(\frac{1}{2}\|\hat{h}^{\text{var}_i} - h_{\text{CL}}^{\text{var}_i}\|_2^2 - \frac{1}{2}\frac{\hat{h}^{\text{var}_i} \cdot h_{\text{CL}}^{\text{var}_i}}{\|\hat{h}^{\text{var}_i}\|_2 \|h_{\text{CL}}^{\text{var}_i}\|_2}\right)$$

Here $\hat{h}^{\text{var}_i} = \mathcal{A}^{-1}(D_{\text{var}_i} \mathcal{A}(\hat{h}))$ is the component of the intervened representation on the $i$-th causal subspace, and $h_{\text{CL}}^{\text{var}_i}$ is treated with stopgrad to prevent gradient feedback—tightening only on causal dimensions to decouple divergence control from OOD generalization optimization.

## Key Experimental Results

### Main Results: Generality of Divergence (Section 3.2)

| Intervention Method | Model | Layer | EMD | Significant Divergence |
|---------|------|---|-----|---------|
| Mean Diff Vector Patching | Llama-3-8B-Instruct | L10 (Lowest EMD layer) | Sig. higher than natural baseline | ✓ |
| SAE Reconstruction | Llama-3-8B-Instruct | L25 | Sig. higher than natural baseline | ✓ |
| Boundless DAS | wu2024 setup | Specified layer | Sig. higher than natural baseline | ✓ |

Across PCA visualizations and Earth Mover's Distance (EMD) quantification, these three mainstream methods show intervened representations clearly deviating from the natural distribution. The conclusion is consistently confirmed using multiple metrics, including nearest neighbor cosine distance, L2 pairwise distance, Local PCA distance, KDE Density Score, and Local Linear Reconstruction Error.

### CL Loss Performance in Boundless DAS (7B LLM) (Section 5.1)

| CL Weight $\epsilon$ | IIA (Intervention Accuracy) | EMD (Divergence Level) | Description |
|---------------------|-----------------|---------------|------|
| 0 (No CL) | Baseline IIA | High | Original DAS |
| Small $\epsilon$ | Maintained/Slight Gain | Significantly Reduced | **Optimal Range** |
| Large $\epsilon$ | IIA Drops | Lowest | CL too strong, hurts behavior |

Key Finding: A sweet spot exists where a small $\epsilon$ significantly reduces divergence without sacrificing IIA.

### Improved CL Loss on Synthetic Tasks (Section 5.2)

| Method | EMD (Feature Dim) | IIA | OOD Generalization |
|------|---------------|-----|---------|
| DAS Behavioral Loss | 0.032 ± 0.003 | 0.997 ± 0.001 | Low |
| Improved CL Loss | **0.007 ± 0.001** | **0.9988 ± 0.0005** | **High** |

CL loss reduces EMD by approximately 4.5x with a slight IIA improvement. In OOD settings (transferring alignment matrices between different subtasks), alignments trained with CL loss significantly outperform behavioral loss. Regression analysis confirms that EMD is anti-correlated with OOD IIA (coeff -0.34, $R^2 = 0.73$, $p < 0.001$), proving the practical value of reducing divergence.

### Key Findings

- Divergence is not an issue with specific methods but a systemic problem with causal intervention.
- Hidden pathways can make behavior appear "correct" while using completely unnatural mechanisms—the most dangerous scenario.
- Dormant behavioral changes make intervention safety dependent on context, which cannot be exhaustively searched.
- CL loss offers a simple and effective preliminary mitigation with OOD generalization advantages.

## Highlights & Insights

1. **Meta-Methodological Contribution**: Rather than using interpretability tools to analyze models, this work scrutinizes the reliability of the tools themselves, significantly impacting the field's methodological foundation.

2. **"Hidden Pathways" Concept**: Interventions may activate computational paths never used in natural states, leading to correct behaviors from false mechanisms. This directly challenges the common assumption that "High IIA = Correct Mechanism Found."

3. **Harmless vs. Harmful Framework**: Provides a principled way to judge the harmfulness of shifts using behavioral null space theory, rather than treating all shifts as inherently problematic.

4. **Elegance of Theorem A.2**: Only axis-aligned hyperrectangles are patch-closed—for almost any practical manifold, coordinate patching will inevitably produce divergence.

5. **Practicality**: CL loss is simple to implement, can be integrated into existing DAS workflows, and is validated on 7B LLMs.

## Limitations & Future Work

1. **Lack of Automatic Classification for Harmful Shifts**: No method to automatically distinguish harmless from harmful shifts currently exists, limiting practical utility.
2. **CL Loss as a "Broad" Strategy**: It reduces all shifts (including harmless ones) rather than precisely eliminating harmful ones.
3. **Improved CL Loss Validated Only on Synthetic Tasks**: Synthetic datasets with 10-class classification are far from real LLM scenarios.
4. **Limited to Linear Alignment Functions**: Sutter et al. suggest non-linear AFs have more fundamental issues not covered here.
5. **CL Vector Acquisition Depends on Labels**: Requires knowing which natural representations share the same causal variable values, which is difficult in complex scenarios.
6. **Future Directions**: (a) Online divergence detection via ReLU activation pattern audits; (b) combining manifold projection with CL loss; (c) self-supervised discovery of harmful shifts.

## Related Work & Insights

- **Makelov et al. (2023)**: Previously noted interactions between the null space and dormant subspaces in DAS; this work generalizes that to broader intervention methods.
- **Zhang et al. (2024) / Heimersheim (2024)**: Noted that patching results are easily misinterpreted; this work provides a theoretical explanation via representation divergence.
- **Sutter et al. (2025)**: Questioned the meaning of causal intervention under non-linear AFs; these findings are complementary.
- **Grant (2025)**: Source of the original CL loss, which this work extends to the causal subspace level.
- **Inspiration for SAE Research**: SAE reconstruction is itself an intervention that produces divergence, raising questions about the "causal validation" step for SAE features.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Questions fundamental methodological assumptions; significant meta-level contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Solid theoretical proofs and meaningful LLM experiments, though improved methods are only tested on synthetic data.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Precise problem definition, clear logic, and tight integration of theory and experiments.
- **Value**: ⭐⭐⭐⭐⭐ — Broad impact on causal intervention experiments in mechanistic interpretability; well-deserved Oral.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Bayesian Neural Networks for Functional ANOVA Model](bayesian_neural_networks_for_functional_anova_model.md)
- [\[ICLR 2026\] Explainable K-means Neural Networks for Multi-view Clustering](explainable_k_-means_neural_networks_for_multi-view_clustering.md)
- [\[ICLR 2026\] FAME: Formal Abstract Minimal Explanation for Neural Networks](fame_formal_abstract_minimal_explanation_for_neural_networks.md)
- [\[ICLR 2026\] Causal Interpretation of Neural Network Computations with Contribution Decomposition](causal_interpretation_of_neural_network_computations_with_contribution_decomposi.md)
- [\[ICLR 2026\] Estimating Dimensionality of Neural Representations from Finite Samples](estimating_dimensionality_of_neural_representations_from_finite_samples.md)

</div>

<!-- RELATED:END -->
