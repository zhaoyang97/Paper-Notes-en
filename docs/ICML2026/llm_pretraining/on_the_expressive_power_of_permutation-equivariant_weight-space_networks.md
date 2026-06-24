---
title: >-
  [Paper Note] On the Expressive Power of Permutation-Equivariant Weight-Space Networks
description: >-
  [ICML 2026 Spotlight][LLM Pretraining][weight-space learning] This paper establishes the first systematic theory of expressive power for permutation-equivariant weight-space networks (e.g., DWS, NFN, GMN, NG-GNN) operating on MLP weights. It proves these architectures are nearly equivalent in expressivity and characterizes their universality across four approximation scenarios (function-space functionals/operators, permutation-invariant functionals…
tags:
  - "ICML 2026 Spotlight"
  - "LLM Pretraining"
  - "weight-space learning"
  - "permutation equivariance"
  - "universality"
  - "INR editing"
  - "OCE"
date: 2026-05-08
content_hash: 278345fd2a187f3a
---

# On the Expressive Power of Permutation-Equivariant Weight-Space Networks

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2602.01083](https://arxiv.org/abs/2602.01083)  
**Code**: https://github.com/dayanadir/capacity_increase_inr_editing_experiment  
**Area**: Weight-space learning / Expressive power theory / Equivariant neural networks  
**Keywords**: weight-space learning, permutation equivariance, universality, INR editing, OCE

## TL;DR
This paper establishes the first systematic theory of expressive power for permutation-equivariant weight-space networks (e.g., DWS, NFN, GMN, NG-GNN) operating on MLP weights. It proves these architectures are nearly equivalent in expressivity and characterizes their universality across four approximation scenarios (function-space functionals/operators, permutation-invariant functionals, and permutation-equivariant operators) under the "general position" assumption. A theoretically derived modification, OCE (Output Capacity Expansion via ensembling multiple MLPs), achieves a 34% improvement over SOTA on INR editing benchmarks.

## Background & Motivation

**Background**: Weight-space learning treats trained neural networks as "structured data," using a meta-network to directly process the parameters $v=(W_1,b_1,\dots,W_L,b_L)$ of another MLP for downstream tasks like accuracy prediction, INR editing, or meta-optimization. Since hidden neuron permutations $\tau$ in an MLP satisfy $f_{\rho(\tau)v}=f_v$ (functional invariance), mainstream SOTA architectures (DWS, NP-NFN, HNP-NFN, GMN, NG-GNN, NFT) are explicitly constructed to be equivariant to the permutation group $G_A=S_{d_1}\times\cdots\times S_{d_{L-1}}$.

**Limitations of Prior Work**: (1) While many architectures exist (DWS uses manual alignment, GMN/NG-GNN treat networks as graphs, and NFT uses attention), it remains unclear which is more expressive. (2) Symmetry constraints inherently weaken expressive power, but existing theories only provide sporadic forward-pass simulation results (Navon et al., 2023; Lim et al., 2023; Kalogeropoulos et al., 2024), lacking a unified characterization of universality. (3) For certain natural targets, such as the "zoom-out" operator for INRs, the complexity of the output function may exceed the capacity limits of the input MLP architecture—a theoretical impossibility that has not been explicitly addressed previously.

**Key Challenge**: Approximation in weight-space naturally spans two semantic layers: the parameter space $\mathcal V$ and the function space $\mathcal F$ they realize. "Equivariant weight-to-weight maps" and "function-space operators" represent different target types and must be analyzed separately. Furthermore, symmetry constraints are sufficient in some settings but insufficient (non-universal) in others, necessitating a precise characterization of these boundaries.

**Goal**: (a) Map all mainstream permutation-equivariant weight-space architectures into a single equivalence class in terms of expressivity; (b) Systematically categorize "approximation targets" into four types and provide a complete landscape of universality vs. non-universality; (c) Translate theoretical insights into actionable architectural improvements.

**Key Insight**: The authors observe that in other symmetric domains (graphs, point clouds) within geometric deep learning, it is common to use the *general position* (GP) paradigm, where universality holds outside a degenerate subset $\mathcal E$ (Maron et al., 2020; Finkelshtein et al., 2025). In weight-space, the natural degenerate set consists of cases where a hidden layer has two identical biases ($b_i=b_j$)—a subset of Lebesgue measure zero. By focusing on GP, the degeneracies introduced by equivariance can be isolated.

**Core Idea**: First, prove that all mainstream architectures are equivalent in expressivity → Unify the analysis under a "universal permutation-equivariant weight-space network" → Determine universality in four approximation scenarios. Finally, from the impossibility result that function-space operators are non-universal under fixed architectures, derive a simple solution: **make the output larger than the input** (OCE outputs $k$ MLPs and averages them).

## Method

This paper is purely theoretical with one theory-driven architectural change; "Method" refers to the theorem framework and proof skeleton rather than a specific new model.

### Overall Architecture

The argument centers on a 2D map: the horizontal axis represents "approximation targets," and the vertical axis represents "whether the input lies in general position (GP)." Targets are divided into four categories according to Definition 4.1: Function-space functionals $\Psi:\mathcal C(X, \mathbb R^{d_L}) \to \mathbb R^n$ (e.g., accuracy prediction, INR classification), Permutation-invariant functionals $\Psi:\mathcal V_A \to \mathbb R^n$ (e.g., weight $\ell_2$ norm, loss landscape curvature), Function-space operators $\Psi:\mathcal C \to \mathcal C$ (e.g., INR editing, domain adaptation), and Permutation-equivariant operators $\Psi:\mathcal V_A \to \mathcal V_A$ (e.g., pruning mask prediction, gradient prediction for meta-optimization). The input domain is split between the full parameter space $\mathcal V$ and the GP subset $\mathcal V\setminus\mathcal E$, where $\mathcal E_A=\{v \mid \exists \ell \in [L-1], i \neq j, (b_\ell)_i=(b_\ell)_j\}$ is the measure-zero set where neurons share biases. $L^2$ error measures functionals and equivariant operators, while $L^\infty$ measures function-space operators via the realization map $R(v)=f_v$, requiring $\sup_v\|\Psi(f_v)-f_{\Phi(v)}\|_\infty < \epsilon$ for a weight-to-weight map $\Phi$.

### Key Designs

**1. Architecture Equivalence Theorem: Folding six different architectures into one class (Theorem 5.2 + Proposition 5.3)**

The community lacked clarity on whether DWS, GMN/NG-GNN, or NFT was superior. This paper proves their expressivity is nearly identical: for any compact set $K \subseteq \mathcal V$ and any $\pi, \pi' \in \Pi \setminus \{\text{NFT}\}$ (where $\Pi = \{\text{DWS, NP-NFN, HNP-NFN, GMN, NG-GNN, NFT}\}$), it holds that $\mathcal N^\pi_{\text{inv}}(K) = \mathcal N^{\pi'}_{\text{inv}}(K)$ and $\mathcal N^\pi_{\text{equi}}(K) = \mathcal N^{\pi'}_{\text{equi}}(K)$. The technique involves mutual approximation between layers. While NFT (attention) lags in the full space (Prop 5.3), it rejoins the class when inputs are restricted to the GP subset $K \subset \mathcal V \setminus \mathcal E$. This reduces the choice of architecture from a theoretical problem to one of engineering preference.

**2. Universality Map under GP: Pinpointing sufficient and insufficient scenarios (Theorems 6.1 / 6.3 / 7.2 / 7.4)**

Each of the four targets is analyzed, using the GP assumption to bridge "theoretical non-universality" and "practical approximability."
- **Function-space functionals** (Thm 6.1) are universal on the full space $K \subseteq \mathcal V$, proven by showing DWS can simulate MLP forward passes to achieve separation (if DWS cannot distinguish $v, v'$, then $f_v = f_{v'}$) and applying the separation-to-approximation theorem (Pacini et al., 2025b).
- **Permutation-invariant functionals** (Prop 6.2 + Thm 6.3) are non-universal in the full space (counterexamples exist where 1-WL cannot distinguish weights, Figure 3) but universal on GP. The core construction is a **continuous canonization map** $\operatorname{canon}: K \to \mathcal V$: since $b_\ell$ has unique elements on GP, $\operatorname{argsort}(b_\ell)$ provides unique and continuous orbital representatives.
- **Function-space operators** (Prop 7.1 + Thm 7.2) are non-universal for fixed ReLU architectures because the number of linear regions is bounded. Tasks like "zoom-out" that increase geometric complexity cannot be approximated by a same-capacity output. However, they become universal if the **output architecture is sufficiently large**.
- **Permutation-equivariant operators** (Prop 7.4) are universal on GP via "broadcasting canonization," where the canonized flat vector is concatenated to every weight/bias entry followed by a pointwise MLP.

**3. OCE (Output Capacity Expansion): A near-zero-cost theoretical fix (Section 8)**

The impossibility in Prop 7.1 stems from "output capacity = input capacity." OCE's solution is simple: expand the final feature dimension of any weight-space network by a factor $k > 1$. The output tensor is interpreted as the parameters for $k$ parallel MLPs, and the final prediction is the average of their outputs. By sharing the backbone and only expanding the head, parameters remain similar while the effective number of ReLU regions increases $k$-fold, bypassing the capacity bottleneck while maintaining equivariance.

## Key Experimental Results

The experiment focuses on the MNIST INR dilation benchmark to verify the practical value of Thm 7.2 and OCE.

### Main Results

| Method | Reference | MSE ($\times 10^{-2}$, ↓) |
| :--- | :--- | :--- |
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

GMN+OCE reduces MSE by 34% relative to the previous SOTA. DWS and GMN individually improve by 41% and 46% compared to their $k=1$ baselines.

### Ablation Study

Trends from Appendix Table 2:
- **DWS/GMN, $k=1 \to 8$**: MSE drops significantly (~41-46%) without adding extra backbone parameters, confirming the Thm 7.2 insight regarding output architecture size.
- **Comparison**: OCE outperforms baselines even though those baselines use additional signals like gradients or probes.

### Key Findings
- **Theory-to-Practice Loop**: Performance bottlenecks are identified as "representation capacity of the output" rather than "backbone weakness," as predicted by Prop 7.1 and verified by OCE.
- **OCE as a Free Lunch**: It requires no extra supervision, nearly zero extra parameters, and only one line of code changes, yet it significantly outperforms complex architectural additions.
- **NFT Performance**: NFT ranks lower than expected (5.10 vs. 1.36), suggesting attention mechanisms in weight-space do not share the same dominance as in sequence modeling, aligning with the theoretical observation of its non-equivalence in full space.

## Highlights & Insights

- **Collapsing complex architectures into one equivalence class** is the highest-density insight: for future research, the choice between DWS, GMN, or NG-GNN is largely a matter of engineering implementation rather than fundamental expressive capability.
- **Dual Use of GP Hypothesis**: It is used both to separate counterexamples from universality (Prop 6.2 vs. Thm 6.3) and to reconcile NFT within the equivalence class.
- **Continuous Canonization as a Master Key**: Since $\operatorname{argsort}(b_\ell)$ is unique and locally constant on GP, continuous canonization naturally exists, reducing equivariant universality to the known universality of DeepSets.
- **Implications of Prop 7.1**: Previously, the community attributed poor INR editing results to weak models. This paper proves the root cause is fixed output capacity, leading to the simple but effective OCE fix.
- **Metaphor for Over-parameterization**: The authors link "expanding output architecture" to the theory that over-parameterization eases optimization and generalization, suggesting that asymmetric designs (input smaller than output) are the next frontier for weight-space learning.

## Limitations & Future Work

- **Scope limited to MLPs**: Does not cover transformer or conv weights (though a transformer sketch is provided in Appendix H).
- **Scale-equivariant architectures excluded**: Architectures like ScaleGMN are not covered by the theory, despite showing strong empirical performance.
- **Theory focuses on expressivity**: It does not address optimization or generalization—the gap between "approximable" and "learnable via SGD" remains large in weight-space.
- **ReLU dependence**: Impossibility results rely on ReLU region counting (Prop 7.1); other activations require new proofs.
- **Hyperparameter $k$**: The ensemble number $k$ is an additional hyperparameter and has primarily been validated on the INR dilation benchmark.

## Related Work & Insights

- **vs. Navon et al. 2023, Lim et al. 2023**: These works provided scattered simulations; this paper unifies them and completes the universality map for all four target types.
- **vs. Maron et al. 2020, Finkelshtein et al. 2025**: This paper introduces the "universality outside GP" paradigm to weight-space for the first time with a definition tailored to biases.
- **vs. Bronstein et al. (GDL Survey)**: This work positions weight-space as a fourth structural data type alongside graphs/point clouds/grids, building out its respective expressivity toolbox for the meta-learning era.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First unified expressive power characterization for the weight-space network family.
- **Experimental Thoroughness**: ⭐⭐⭐ Theoretical focus; only one benchmark, though the SOTA gain (34%) is significant.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The "Expressivity Map" is highly effective; the theorem-counterexample-theorem rhythm is clear.
- **Value**: ⭐⭐⭐⭐⭐ Simplifies architecture selection and suggests a new design direction (large output vs. small input).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Different Layers, Different Manifolds: Module-Wise Weight-Space Geometry in Transformer Optimization](different_layers_different_manifolds_module-wise_weight-space_geometry_in_transf.md)
- [\[NeurIPS 2025\] Power Lines: Scaling Laws for Weight Decay and Batch Size in LLM Pre-training](../../NeurIPS2025/llm_pretraining/power_lines_scaling_laws_for_weight_decay_and_batch_size_in_llm_pre-training.md)
- [\[ICLR 2026\] Steering Language Models with Weight Arithmetic](../../ICLR2026/llm_pretraining/steering_language_models_with_weight_arithmetic.md)
- [\[ICLR 2026\] Intrinsic Training Dynamics of Deep Neural Networks](../../ICLR2026/llm_pretraining/intrinsic_training_dynamics_of_deep_neural_networks.md)
- [\[ACL 2026\] Data Mixing Agent: Learning to Re-weight Domains for Continual Pre-training](../../ACL2026/llm_pretraining/data_mixing_agent_learning_to_re-weight_domains_for_continual_pre-training.md)

</div>

<!-- RELATED:END -->
