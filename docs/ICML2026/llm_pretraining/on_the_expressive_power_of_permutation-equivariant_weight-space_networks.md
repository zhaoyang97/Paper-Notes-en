---
title: >-
  [Paper Note] On the Expressive Power of Permutation-Equivariant Weight-Space Networks
description: >-
  [ICML 2026][LLM Pretraining][weight-space learning] This paper establishes the first systematic expressivity theory for permutation-equivariant weight-space networks (DWS / NFN / GMN / NG-GNN…
tags:
  - "ICML 2026"
  - "LLM Pretraining"
  - "weight-space learning"
  - "permutation equivariance"
  - "universality"
  - "INR editing"
  - "OCE"
date: 2026-05-08
content_hash: babc043798eb3b4f
---

# On the Expressive Power of Permutation-Equivariant Weight-Space Networks

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2602.01083](https://arxiv.org/abs/2602.01083)  
**Code**: https://github.com/dayanadir/capacity_increase_inr_editing_experiment  
**Area**: Weight-space learning / Expressive power theory / Equivariant neural networks  
**Keywords**: weight-space learning, permutation equivariance, universality, INR editing, OCE

## TL;DR
This paper establishes the first systematic expressivity theory for permutation-equivariant weight-space networks (DWS / NFN / GMN / NG-GNN, etc.) operating on MLP weights. It proves that these architectures are almost entirely equivalent in terms of expressivity and provides universality characterizations for four approximation scenarios (function-space functionals/operators, permutation-invariant functionals, permutation-equivariant operators) under the "general position" assumption. A simple theoretical modification, OCE (ensembling multiple MLPs at the output), achieves a 34% improvement over SOTA on the INR editing benchmark.

## Background & Motivation

**Background**: Weight-space learning treats trained neural networks as "structured data," using a meta-network to directly consume the parameters $v=(W_1,b_1,\dots,W_L,b_L)$ of another MLP for downstream tasks like accuracy prediction, INR editing, and meta-optimization. Since hidden neuron permutations $\tau$ in an MLP satisfy $f_{\rho(\tau)v}=f_v$ (function invariance), mainstream SOTA architectures (DWS / NP-NFN / HNP-NFN / GMN / NG-GNN / NFT) are explicitly constructed to be equivariant to the permutation group $G_A=S_{d_1}\times\cdots\times S_{d_{L-1}}$.

**Limitations of Prior Work**: (1) These architectures appear significantly different (DWS uses manually aligned equivariant linear layers, GMN/NG-GNN treat the network as a graph for GNNs, and NFT uses attention), but the community lacks clarity on their relative strengths. (2) Symmetry constraints inherently weaken expressivity, yet existing theories only provide fragmented forward-pass simulation results (Navon et al., 2023; Lim et al., 2023; Kalogeropoulos et al., 2024) without a unified universality characterization. (3) Some natural targets, such as the "zoom-out" operator for INRs, may produce output functions with complexity exceeding the capacity of the input MLP; theoretically, it is impossible to approximate such targets using the same output architecture—a fact not previously articulated.

**Key Challenge**: The approximation problem in weight-space naturally spans two semantic levels—the parameter space $\mathcal V$ and the function space $\mathcal F$ they realize. "Equivariant weight-to-weight mappings" and "function-space operators" are distinct types of targets and must be discussed separately. Symmetry constraints are sufficient in some settings but insufficient (i.e., non-universal) in others, necessitating a precise characterization of the "boundaries."

**Goal**: (a) Place all mainstream permutation-equivariant weight-space architectures into the same equivalence class in terms of expressivity; (b) systematically categorize "approximation targets" into 4 types and provide a complete map of universality/non-universality; (c) translate theoretical findings into actionable architectural modifications.

**Key Insight**: The authors observe that in other symmetrical domains of Geometric Deep Learning (graphs, point clouds), the *general position* (GP) paradigm—where universality holds except on a degenerate subset $\mathcal E$—is commonly used (Maron et al., 2020; Finkelshtein et al., 2025). In weight-space, the natural degenerate set is where "a hidden layer contains two identical biases $b_i=b_j$," which is a subset of Lebesgue measure zero. Almost all trained MLPs fall within the GP. By leveraging GP, the degeneracies introduced by equivariance can be isolated.

**Core Idea**: First prove that all mainstream architectures have equivalent expressivity → unify the analysis object into a "Universal Permutation-Equivariant Weight-Space Network" → determine universality for each of the 4 approximation scenarios. Simultaneously, from the impossibility conclusion that "function-space operators are not universal under fixed architectures," derive a simple workaround: **make the output larger than the input** (OCE directly outputs and averages $k$ MLPs).

## Method

This paper consists of pure theory plus a theory-driven simple experimental architectural modification. The "Method" should be understood as the theoretical framework and proof skeleton rather than a new model per se.

### Overall Architecture

The weight-space problem is organized into a 2D grid: "Target Type × Input General Position":

- **Target Type** (Definition 4.1):
    1. Function-space functional $\Psi:\mathcal C(X,\mathbb R^{d_L})\to\mathbb R^n$, e.g., accuracy prediction, INR classification.
    2. Permutation-invariant functional $\Psi:\mathcal V_A\to\mathbb R^n$, e.g., weight $\ell_2$ norm, loss landscape curvature.
    3. Function-space operator $\Psi:\mathcal C\to\mathcal C$, e.g., INR editing, domain adaptation.
    4. Permutation-equivariant operator $\Psi:\mathcal V_A\to\mathcal V_A$, e.g., pruning mask prediction, gradient prediction for meta-optimization.

- **Input Domain**: The entire $\mathcal V$ vs. the GP subset $\mathcal V\setminus\mathcal E$, where
$\mathcal E_A=\{v\mid\exists\ell\in[L-1],\,i\ne j,\,(b_\ell)_i=(b_\ell)_j\}$.

Approximation is measured by $L^2$ (functionals/equivariant operators) or $L^\infty$ (function-space operators). For function-space operators, it is measured as $\sup_v\|\Psi(f_v)-f_{\Phi(v)}\|_\infty<\epsilon$, meaning the "equivariant weight-to-weight mapping" approximates the target operator after being mapped to the function space by the realization map $R(v)=f_v$.

### Key Designs

1. **Architecture Equivalence Theorem (Theorem 5.2 + Proposition 5.3)**:

    - **Function**: Collapses 6 mainstream architectures $\Pi=\{\text{DWS, NP-NFN, HNP-NFN, GMN, NG-GNN, NFT}\}$ into a single expressivity equivalence class.
    - **Mechanism**: For any compact set $K\subseteq\mathcal V$ and any $\pi,\pi'\in\Pi\setminus\{\text{NFT}\}$, $\mathcal N^\pi_{\text{inv}}(K)=\mathcal N^{\pi'}_{\text{inv}}(K)$ and $\mathcal N^\pi_{\text{equi}}(K)=\mathcal N^{\pi'}_{\text{equi}}(K)$. The proof uses the basic layers of one architecture to "simulate" those of another (mutual approximation). NFT is not equivalent on the full space due to its non-standard attention mechanism (Proposition 5.3 provides a counterexample $K$), but remains equivalent on the GP subset $K\subset\mathcal V\setminus\mathcal E$.
    - **Design Motivation**: Downgrades the choice of architecture from a selection problem to an engineering one and allows all subsequent theorems to be stated once for a "universal permutation-equivariant weight-space network."

2. **Unified Universality under GP (Theorems 6.1 / 6.3 / 7.2 / 7.4)**:

    - **Function**: Provides universality determinations (including upper and lower bounds) for each of the 4 target categories.
    - **Mechanism**:
        - **Function-space functional** (Thm 6.1): Universal over the *entire space* $K\subseteq\mathcal V$. The proof shows DWS can simulate an MLP forward pass (Navon et al., 2023) → derives a separation property where "DWS cannot distinguish $v,v'$ ⇒ $f_v=f_{v'}$" → concludes using the separation-to-approximation theorem (Pacini et al., 2025b).
        - **Permutation-invariant functional** (Prop 6.2 + Thm 6.3): *Not universal* over the full space (constructs $v,v'$ where $W_2,W_2'$ have different ranks but are indistinguishable by 1-WL, see Figure 3), but universal on the GP. The key construction for Thm 6.3 is a **continuous canonization mapping** $\operatorname{canon}:K\to\mathcal V$: since $K\cap\mathcal E=\varnothing$, elements of the bias $b_\ell$ are unique per layer, allowing $\operatorname{argsort}(b_\ell)$ to serve as a layer-wise permutation. The resulting orbit representative is unique and continuous. DWS contains DeepSets primitives, which are universal for ranking (Segol & Lipman, 2019), followed by an MLP head.
        - **Function-space operator** (Prop 7.1 + Thm 7.2): *Not universal* on a *fixed ReLU architecture*—the number of linear regions in a ReLU MLP is bounded (Montúfar et al., 2014); hence, operators like "zoom-out" that increase geometric complexity cannot be approximated by a same-architecture equivariant mapping. However, it is universal if the "output architecture $A$ is sufficiently large." The proof uses a partition of unity to write $\Psi(f_v)$ as a continuous convex combination of $M$ reference MLPs $\Rightarrow$ uses canonization to make it permutation-equivariant $\Rightarrow$ concludes via Thm 7.4.
        - **Permutation-equivariant operator** (Prop 7.3 + Thm 7.4): Dual to the invariant case—through broadcasting, invariant universality implies equivariant universality. Thm 7.4 uses "broadcasting canonization" $\widetilde{\operatorname{canon}}(v)$ (appending $\operatorname{Flat}(\operatorname{canon}(v))$ to each weight/bias entry) + pointwise MLPs.
    - **Design Motivation**: After determining all four cells, the resulting "Expressivity Map" in Figure 1 informs practitioners where existing architectures are sufficient and pinpoints where new architectures are needed. The GP assumption clarifies both "theoretical non-universality" and "practical approximability," preventing the misuse of pessimistic conclusions.

3. **OCE: Output Capacity Expansion (Section 8)**:

    - **Function**: Implements the "larger output architecture" requirement from Thm 7.2 as a minimal, plug-and-play modification.
    - **Mechanism**: Adds a dimension $k>1$ to the final feature dimension of any weight-space network. The output tensor is interpreted as parameters for $k$ parallel MLPs, and the final prediction is the average of these $k$ MLP outputs. This requires changing only one line of code; model parameters remain largely unchanged as the backbone is shared, only the output head's channels are expanded by $k$.
    - **Design Motivation**: The root of the impossibility in Prop 7.1 is the capacity bottleneck when "output MLP capacity $=$ input MLP capacity." Ensembling increases the effective number of ReLU regions by a factor of $k$, bypassing the bottleneck while naturally maintaining permutation equivariance (each branch is independently equivariant). This is an engineering improvement directly derived from theory.

## Key Experimental Results

There is one primary experiment: the MNIST INR dilation benchmark, used to validate the SOTA gains brought by OCE and indirectly verify the practical value of Thm 7.2.

### Main Results

| Method | Ref | MSE ($\times 10^{-2}$, ↓) |
|------|------|---------------------------|
| NFT | Zhou et al. 2023b | 5.10 ± 0.04 |
| NP-NFN | Kofinas et al. 2024 | 2.55 ± 0.00 |
| NG-GNN-64 | Kofinas et al. 2024 | 2.06 ± 0.01 |
| ScaleGMN-B | Kalogeropoulos et al. 2024 | 1.89 ± 0.00 |
| NG-T-64 | Kofinas et al. 2024 | 1.75 ± 0.01 |
| ScaleGMN + GradMetaNet++ | Gelberg et al. 2026 | 1.60 ± 0.01 |
| DWS (k=1, baseline) | Gelberg et al. 2026 | 2.29 ± 0.01 |
| GMN (k=1, baseline) | Gelberg et al. 2026 | 1.96 ± 0.02 |
| **DWS + OCE (k=8)** | Ours | **1.36 ± 0.03** |
| **GMN + OCE (k=8)** | Ours | **1.06 ± 0.13** |

GMN+OCE reduces MSE by 34% relative to the previous SOTA (ScaleGMN+GradMetaNet++ at 1.60). DWS and GMN themeselves show improvements of 41% and 46% respectively over their $k=1$ baselines.

### Ablation Study

Trends from Appendix Table 2 (summarized):

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| DWS, $k=1\to 8$ | MSE decreases ~41% | No additional parameters (shared backbone) |
| GMN, $k=1\to 8$ | MSE decreases ~46% | Validates the "expand output architecture" guidance of Thm 7.2 |
| Control Baselines | Extensive use of gradients/probes | OCE outperforms without needing extra signals |

### Key Findings

- **Theory-to-Experiment Loop**: The performance bottleneck is identified as "insufficient output representation capacity" rather than a "weak backbone," a prediction made by Prop 7.1 and verified by OCE.
- **OCE as a Free Lunch for Weight-Space Learning**: It requires virtually no extra parameters, is compatible with both DWS and GMN, and does not need additional supervision, yet it significantly widens the MSE gap, serving as a strong exemplar for future baseline settings.
- **NFT Ranks Surprisingly Low** (5.10 vs. DWS+OCE 1.36), suggesting that the advantages of attention in weight-space are far less significant than in sequence modeling—consistent with the theoretical observation in Prop 5.3 that NFT is not equivalent to other architectures on the full space.

## Highlights & Insights

- **Collapsing seemingly diverse architectures into a single equivalence class** is the highest-density insight: choosing between DWS, GMN, or NG-GNN is now largely an engineering preference; researchers need only select the most analytically convenient one (DWS here) to prove universal properties.
- **Dual Use of GP Assumption**: It is used both to separate "counterexamples" from "universality" (Prop 6.2 vs. Thm 6.3) and to bring NFT back into the equivalence class (Prop 5.3). This methodology of discussing universality outside measure-zero degenerate sets can be extended to other symmetric domains (e.g., transformers with shared parameters, scale-equivariant networks).
- **Continuous Canonization as a Universal Key**: Since $\operatorname{argsort}(b_\ell)$ is unique and locally constant under GP, continuous canonization exists naturally. This reduces the "equivariant universality" problem to "DeepSets universality," resulting in a clean proof skeleton shared across almost all scenarios.
- **Engineering Significance of Prop 7.1 & OCE**: Previously, the community attributed poor INR editing performance to weak models; this paper points out the root cause is the locked capacity of the output MLP. This observation directly led to OCE—a nearly zero-cost modification applicable to any meta-learning setting involving a "small input network → small output network."
- **Metaphor of Overparameterization**: The authors explicitly compare "expanding the output architecture" to the idea that "overparameterization eases optimization and improves generalization" (Du et al., 2019; Belkin et al., 2019), suggesting that the next breakthrough in weight-space learning may lie in asymmetric input-smaller-than-output designs.

## Limitations & Future Work

- **Restricted to MLP Weight Spaces**: Does not cover transformer or convolutional weights (though Appendix H provides a sketch for transformers); symmetry groups for convolution and pooling in CNNs are not yet addressed.
- **Excludes Scale-Equivariant Architectures**: Architectures like ScaleGMN (Kalogeropoulos et al. 2024; Tran et al. 2024) are not covered, despite ScaleGMN's strong performance in baselines, indicating a gap between theory and practice.
- **Expressivity vs. Optimization/Generalization**: The authors emphasize the theory only addresses expressivity; the fact that a mapping "can be approximated" does not mean gradient descent will find it, which is a significant gap in fields with highly irregular loss landscapes like weight-space.
- **Reliance on ReLU for Impossibility**: Prop 7.1 relies on counts of linear regions; other activations would require re-constructing degenerate function families (the authors believe this generalizes but have not provided a full proof).
- **Hyperparameter $k$**: The ensemble number $k$ in OCE is a hyperparameter, and its practical value has only been validated on the INR dilation benchmark, with other function-space operator tasks (domain adaptation, NeRF editing) yet to be empirically tested.

## Related Work & Insights

- **vs. Navon et al. 2023 (DWS), Lim et al. 2023 (GMN), Kalogeropoulos et al. 2024 (ScaleGMN)**: These works provide "forward-pass simulations" or partial expressivity for specific target classes. This paper integrates them into a unified framework, proves their mutual equivalence, and completes the four-quadrant universality map.
- **vs. Maron et al. 2020 / Finkelshtein et al. 2025 / Gelberg et al. 2026 (GP Methodology in GDL)**: This paper systematically applies the "universality outside GP" paradigm to weight-space for the first time, defining GP naturally as unique hidden biases.
- **vs. Pacini et al. 2025b (separation-to-approximation)**: Provides a non-trivial application of these Stone–Weierstrass-style theorems in weight-space, with the separation property proven via DWS forward-pass simulation.
- **vs. Bronstein et al. (GDL Overview)**: This work treats weight-space as a fourth type of symmetric structured data (alongside graphs, point clouds, and sets), building a corresponding expressivity toolbox as an extension of GDL in the meta-learning era.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Provides the first unified expressivity characterization for the weight-space network family, closing the loop on architecture equivalence, universality, impossibility, and engineering fixes.
- **Experimental Thoroughness**: ⭐⭐⭐ Primarily theoretical, with only one benchmark; however, the result is significant (34% SOTA gain) and directly supports the theory.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The "Expressivity Map" in Figure 1 is highly effective, and the rhythm of theorem-counterexample-theorem is clear with readable proof sketches.
- **Value**: ⭐⭐⭐⭐⭐ Simplifies architecture selection for future research and suggests a new design direction for input-smaller-than-output models; OCE is a practical, plug-and-play trick.

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
