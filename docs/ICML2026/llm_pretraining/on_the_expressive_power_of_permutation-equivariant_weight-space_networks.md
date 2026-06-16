---
title: >-
  [Paper Note] On the Expressive Power of Permutation-Equivariant Weight-Space Networks
description: >-
  [ICML 2026][Pretraining][weight-space learning] This paper establishes the first systematic expressivity theory for permutation-equivariant weight-space networks (DWS / NFN / GMN / NG-GNN, etc.) operating on MLP weights. It proves these architectures are almost entirely equivalent in expressivity and provides universality characterizations for four approximation sce
tags:
  - ICML 2026
  - Pretraining
  - weight-space learning
  - permutation equivariance
  - universality
  - INR editing
  - OCE
date: 2026-05-08
content_hash: 98f6a7bdeacd82f9
---
# On the Expressive Power of Permutation-Equivariant Weight-Space Networks

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2602.01083](https://arxiv.org/abs/2602.01083)  
**Code**: https://github.com/dayanadir/capacity_increase_inr_editing_experiment  
**Area**: Weight-space learning / Expressivity theory / Equivariant neural networks  
**Keywords**: weight-space learning, permutation equivariance, universality, INR editing, OCE

## TL;DR
This paper establishes the first systematic expressivity theory for permutation-equivariant weight-space networks (DWS / NFN / GMN / NG-GNN, etc.) operating on MLP weights. It proves these architectures are almost entirely equivalent in expressivity and provides universality characterizations for four approximation scenarios (function-space functionals/operators, permutation-invariant functionals, and permutation-equivariant operators) under the "general position" assumption. Based on theoretical insights, a simple modification called OCE (Output Capacity Expansion via MLP ensembles) achieves a 34% improvement over SOTA on INR editing benchmarks.

## Background & Motivation

**Background**: Weight-space learning treats trained neural networks as "structured data," using a meta-network to directly process the parameters $v=(W_1,b_1,\dots,W_L,b_L)$ of another MLP for downstream tasks such as accuracy prediction, INR editing, and meta-optimization. Since hidden neuron permutations $\tau$ of an MLP satisfy $f_{\rho(\tau)v}=f_v$ (function invariance), mainstream SOTA architectures (DWS / NP-NFN / HNP-NFN / GMN / NG-GNN / NFT) are explicitly constructed to be equivariant to $G_A=S_{d_1}\times\cdots\times S_{d_{L-1}}$.

**Limitations of Prior Work**: (1) While these architectures appear distinct (DWS uses manually aligned equivariant linear layers, GMN/NG-GNN treat networks as graphs, and NFT uses attention), the community lacks clarity on their relative strengths. (2) Symmetry constraints inherently weaken expressivity, but existing theories only provide fragmented forward-pass simulation results (Navon et al., 2023; Lim et al., 2023; Kalogeropoulos et al., 2024) rather than a unified characterization of universality. (3) For certain natural targets, such as "zoom-out" operators for INRs, the complexity of the output function might exceed the capacity limit of the input MLP architecture. Theoretically, it is impossible to approximate such operators using the same output architecture—a point not clearly identified previously.

**Key Challenge**: Weight-space approximation problems naturally span two semantic levels—the parameter space $\mathcal V$ and the function space $\mathcal F$ they realize. "Equivariant weight-to-weight mappings" and "function-space operators" are different types of targets and must be discussed separately. Symmetry constraints are sufficient in some settings but insufficient (i.e., non-universal) in others, requiring a precise characterization of the "boundary."

**Goal**: (a) Place all mainstream permutation-equivariant weight-space architectures into the same equivalence class in terms of expressivity; (b) systematically categorize "approximation targets" into four types and provide a complete map of universality vs. non-universality; (c) translate theoretical findings into actionable architectural improvements.

**Key Insight**: The authors observe that the *general position* (GP) paradigm—which guarantees universality outside a degenerate subset $\mathcal E$—is commonly used in other symmetric domains in Geometric Deep Learning (e.g., graphs, point clouds) (Maron et al., 2020; Finkelshtein et al., 2025). In weight space, the natural degenerate set consists of cases where "two identical biases $b_i=b_j$ exist in a hidden layer"—a subset of Lebesgue measure zero. Almost all trained MLPs fall within the GP. By utilizing GP as a theoretical leverage, the degeneracies caused by equivariance can be isolated.

**Core Idea**: First prove that all mainstream architectures are equivalent in expressivity → unify the analysis around a "universal permutation-equivariant weight-space network" → determine universality for each of the four approximation scenarios. Simultaneously, from the impossibility conclusion that function-space operators are non-universal under fixed architectures, derive a simple solution: **make the output larger than the input** (OCE outputs the average of $k$ MLPs).

## Method

This paper consists of pure theory and a simple theory-driven architectural modification; "Method" refers to the theoretical framework and proof sketches rather than a standalone model.

### Overall Architecture

The entire paper revolves around a 2D table: the horizontal axis represents the "approximation target type," and the vertical axis indicates whether the input lies in the *general position* (GP). Targets are classified into four types per Definition 4.1: function-space functionals $\Psi:\mathcal C(X, \mathbb R^{d_L})\to \mathbb R^n$ (e.g., accuracy prediction, INR classification), permutation-invariant functionals $\Psi:\mathcal V_A\to \mathbb R^n$ (e.g., weight $\ell_2$ norm, loss landscape curvature), function-space operators $\Psi:\mathcal C\to \mathcal C$ (e.g., INR editing, domain adaptation), and permutation-equivariant operators $\Psi:\mathcal V_A\to \mathcal V_A$ (e.g., pruning mask prediction, gradient prediction for meta-optimization). The input domain is divided into the full parameter space $\mathcal V$ and the GP subset $\mathcal V\setminus\mathcal E$, where $\mathcal E_A=\{v\mid\exists\ell\in[L-1],\,i\ne j,\,(b_\ell)_i=(b_\ell)_j\}$ is the measure-zero set of MLPs with duplicate biases. Approximation error is measured by $L^2$ for functionals and equivariant operators, and by $L^\infty$ for function-space operators, where the latter requires that the weight-to-weight mapping $\Phi$ satisfies $\sup_v\|\Psi(f_v)-f_{\Phi(v)}\|_\infty<\epsilon$ when pulled back via the realization map $R(v)=f_v$.

### Key Designs

**1. Architecture Equivalence Theorem: Folding diverse architectures into one equivalence class (Theorem 5.2 + Proposition 5.3)**

The community has long struggled to compare the strengths of DWS, GMN/NG-GNN, and NFT. This paper proves their expressivity is essentially identical: for any compact set $K\subseteq\mathcal V$ and any $\pi,\pi'\in\Pi\setminus\{\text{NFT}\}$ (where $\Pi=\{\text{DWS, NP-NFN, HNP-NFN, GMN, NG-GNN, NFT}\}$), it holds that $\mathcal N^\pi_{\text{inv}}(K)=\mathcal N^{\pi'}_{\text{inv}}(K)$ and $\mathcal N^\pi_{\text{equi}}(K)=\mathcal N^{\pi'}_{\text{equi}}(K)$. This is shown through mutual approximation of their basic layers. NFT falls behind on the full space due to its non-standard attention mechanism (Proposition 5.3 provides a counterexample $K$), but it joins the equivalence class when restricted to the GP subset $K\subset\mathcal V\setminus\mathcal E$. This reduces the "which architecture to use" question from a theoretical choice to an engineering preference.

**2. Four-Quadrant Universality Map under GP: Precisely identifying sufficient vs. insufficient scenarios (Theorems 6.1 / 6.3 / 7.2 / 7.4)**

Each of the four target types is analyzed, with GP acting as the key to distinguishing theoretical non-universality from practical approximability. **Function-space functionals** (Thm 6.1) are universal over the entire space $K\subseteq\mathcal V$. The proof uses DWS to simulate MLP forward passes (Navon et al., 2023) to establish separation—if DWS cannot distinguish $v$ and $v'$, then $f_v=f_{v'}$—and applies the separation-to-approximation theorem (Pacini et al., 2025b). **Permutation-invariant functionals** (Prop 6.2 + Thm 6.3) are actually non-universal over the full space: one can construct $v, v'$ with different $W_2, W_2'$ ranks that cannot be distinguished by 1-WL (Figure 3), but universality is restored on the GP. The core construction for Thm 6.3 is a **continuous canonization mapping** $\operatorname{canon}:K\to\mathcal V$. Since $K\cap\mathcal E=\varnothing$, biases $b_\ell$ in each layer are distinct; using $\operatorname{argsort}(b_\ell)$ as the permutation for each layer yields a unique and continuous orbit representative. DWS inherently supports DeepSets primitives which are universal for ranking (Segol & Lipman, 2019), allowing the construction via an MLP head. **Function-space operators** (Prop 7.1 + Thm 7.2) are non-universal for fixed ReLU architectures: the number of linear regions in a ReLU MLP is bounded (Montúfar et al., 2014), so operators that increase geometric complexity (e.g., zoom-out) cannot be approximated by an output of the same capacity. However, universality is achieved if the output architecture $A$ is allowed to be "large enough." **Permutation-equivariant operators** (Prop 7.3 + Thm 7.4) are dual to the invariant case: through broadcasting, invariant universality implies equivariant universality.

**3. OCE (Output Capacity Expansion): Translating "larger output" requirements into a plug-and-play modification (Section 8)**

The impossibility root in Prop 7.1 is the constraint that "output MLP capacity $=$ input MLP capacity." OCE resolves this simply: an extra dimension $k>1$ is added to the final feature dimension of any weight-space network. The output tensor is interpreted as parameters for $k$ parallel MLPs, and the final prediction is the average of their outputs. By sharing the backbone and only expanding the output head by a factor of $k$, parameter count remains nearly constant while the effective number of ReLU regions increases $k$-fold, bypassing the capacity bottleneck while maintaining equivariance.

### Loss & Training
The theoretical sections do not specify training objectives; OCE experiments follow standard MSE supervision on the INR dilation benchmark (Zhou et al., 2023a).

## Key Experimental Results

The experiment focuses on the MNIST INR dilation benchmark to verify the gains from OCE and validate the practical value of Theorem 7.2.

### Main Results

| Method | Reference | MSE ($\times 10^{-2}$, ↓) |
|------|------|---------------------------|
| NFT | Zhou et al. 2023b | 5.10 ± 0.04 |
| NP-NFN | Kofinas et al. 2024 | 2.55 ± 0.00 |
| NG-GNN-64 | Kofinas et al. 2024 | 2.06 ± 0.01 |
| ScaleGMN-B | Kalogeropoulos et al. 2024 | 1.89 ± 0.00 |
| NG-T-64 | Kofinas et al. 2024 | 1.75 ± 0.01 |
| ScaleGMN + GradMetaNet++ | Gelberg et al. 2026 | 1.60 ± 0.01 |
| DWS (k=1, baseline) | Gelberg et al. 2026 | 2.29 ± 0.01 |
| GMN (k=1, baseline) | Gelberg et al. 2026 | 1.96 ± 0.02 |
| **DWS + OCE (k=8)** | This paper | **1.36 ± 0.03** |
| **GMN + OCE (k=8)** | This paper | **1.06 ± 0.13** |

GMN+OCE reduces MSE by 34% compared to the previous SOTA (ScaleGMN+GradMetaNet++ at 1.60). Compared to their $k=1$ baselines, DWS and GMN MSEs drop by 41% and 46%, respectively.

### Ablation Study

Trends from Appendix Table 2:

| Configuration | Key Observation | Note |
|------|---------|------|
| DWS, $k=1\to 8$ | MSE decreases ~41% | No additional parameters (shared backbone) |
| GMN, $k=1\to 8$ | MSE decreases ~46% | Validates Thm 7.2 "expanded output" guidance |
| Comparison vs Baselines | Baselines use heavy gradients/probes | OCE outperforms without extra signals |

### Key Findings

- **Theory-to-Experiment Loop**: The performance bottleneck is identified as "insufficient output representation capacity" rather than a "weak backbone," as predicted by Prop 7.1 and verified by OCE.
- **OCE as a Free Lunch for Weight-Space Learning**: With nearly no increase in parameters, it is compatible with DWS/GMN without requiring extra supervision, significantly lowering MSE.
- **NFT underperforms significantly** (5.10 vs. DWS+OCE 1.36), suggesting that the advantages of attention in sequence modeling do not translate well to weight space, echoing the theoretical observation in Prop 5.3.

## Highlights & Insights

- **Folding diverse architectures into an equivalence class** is the highest-density insight: choosing between DWS, GMN, or NG-GNN is now largely an engineering preference.
- **Dual use of the GP assumption**: It isolates "counterexamples" from "universality" (Prop 6.2 vs. Thm 6.3) and integrates NFT back into the equivalence class (Prop 5.3). This methodology can be extended to other domains like transformers with parameter sharing.
- **Continuous canonization as a universal key**: Because $\operatorname{argsort}(b_\ell)$ is unique and locally constant under GP, continuous canonization naturally exists, reducing "equivariant universality" to "DeepSets universality."
- **Engineering significance of Prop 7.1 and OCE**: Previously, the difficulty of INR editing was attributed to model weakness. This paper points to the locked output MLP capacity, leading to the "nearly zero-cost" OCE fix.
- **Metaphor with over-parameterization**: The authors explicitly link "expanding output architecture" to the idea that over-parameterization eases optimization and improves generalization, suggesting that future weight-space designs might favor input-output asymmetry.

## Limitations & Future Work

- **Limited to MLP weight space**, excluding transformers/CNNs (though Appendix H provides a transformer sketch). Symmetry groups for convolution and pooling are not yet handled.
- **Excludes scale-equivariant architectures** (e.g., ScaleGMN), despite ScaleGMN's strong empirical performance, leaving a gap between theory and the best available practice.
- **Focuses on expressivity, not optimization or generalization**: Approximability does not guarantee that gradient descent will find the solution, especially in irregular weight-space loss landscapes.
- **ReLU dependence**: The impossibility result (Prop 7.1) relies on linear region counting; generalizations to other activations are believed possible but not yet proven.
- **OCE hyperparameter $k$**: Only validated on the INR dilation benchmark; performance on other function-space operator tasks (e.g., NeRF editing) remains untested.

## Related Work & Insights

- **vs. Navon et al. 2023, Lim et al. 2023, Kalogeropoulos et al. 2024**: These works provide "forward-pass simulation" or partial expressivity. This paper unifies them, proves equivalence, and completes the universality map for four quadrants.
- **vs. Maron et al. 2020 / Finkelshtein et al. 2025 (GP methodology)**: This work systematically brings the "universality outside GP" paradigm to weight space, defining GP as distinct hidden biases.
- **vs. Pacini et al. 2025b (separation-to-approximation)**: Provides a non-trivial application of Stone–Weierstrass-style theorems in weight space, using DWS forward-pass simulation to prove separation.
- **vs. Bronze et al. GDL Surveys**: This work treats weight space as a fourth type of structured symmetric data (alongside graphs, point clouds, and sets), building the corresponding expressivity toolbox for the meta-learning era.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First unified expressivity characterization for the weight-space family; links equivalence, universality, and engineering fixes.
- Experimental Thoroughness: ⭐⭐⭐ Primarily theoretical; validated on one main benchmark, though results are significantly strong (34% SOTA gain).
- Writing Quality: ⭐⭐⭐⭐⭐ The "Expressivity Map" is highly effective; clear rhythm between theorems and counterexamples.
- Value: ⭐⭐⭐⭐⭐ Simplifies architecture selection for future research and introduces OCE as a plug-and-play trick.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](../../ICLR2026/llm_pretraining/steering_language_models_with_weight_arithmetic.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](../../NeurIPS2025/llm_pretraining/predict_training_data_quality_via_its_geometry_in_metric_space.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/llm_pretraining/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)

</div>

<!-- RELATED:END -->
