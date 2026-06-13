---
title: >-
  [Paper Note] On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks
description: >-
  [NeurIPS 2025][LLM Efficiency][Mixture-of-Experts] This paper presents the first systematic analysis of MoE expressive power on structured complex tasks. It proves that shallow MoE can overcome the curse of dimensionalit…
tags:
  - "NeurIPS 2025"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "expressive power"
  - "approximation theory"
  - "compositional sparsity"
  - "curse of dimensionality"
  - "manifold"
  - "piecewise function"
  - "gating mechanism"
date: 2026-05-08
content_hash: ba13d90cad2cec2b
---

# On the Expressive Power of Mixture-of-Experts for Structured Complex Tasks

**Conference**: NeurIPS 2025 Spotlight  
**arXiv**: [2505.24205](https://arxiv.org/abs/2505.24205)  
**Code**: None  
**Area**: LLM Efficiency / MoE Theory / Approximation Theory
**Keywords**: Mixture-of-Experts, expressive power, approximation theory, compositional sparsity, curse of dimensionality, manifold, piecewise function, gating mechanism

## TL;DR

This paper presents the first systematic analysis of MoE expressive power on structured complex tasks. It proves that shallow MoE can overcome the curse of dimensionality on low-dimensional manifolds (approximation rate governed by intrinsic dimension $d$ rather than ambient dimension $D$), and that deep MoE with $E$ experts × $L$ layers can efficiently approximate piecewise functions with $E^L$ pieces through hierarchical composition—far exceeding the naive upper bound of $LE$.

## Background & Motivation

**Background**: MoE has become a core architectural component of modern LLMs (Mixtral, Switch Transformer, DeepSeek, Phi-3), achieving strong performance on diverse complex tasks including mathematical reasoning, code generation, and language understanding. However, a theoretical explanation for why MoE efficiently models complex tasks remains lacking.

**Limitations of Prior Work**:
   - Prior theoretical work has focused on training dynamics (Chen et al. 2022), the benefits of routing mechanisms (Dikkala et al. 2023), or sparse activation equivalence (Baykal et al. 2022), without studying the expressive power of MoE for structured complex functions.
   - While approximation theory for dense networks (functions on manifolds, compositionally sparse functions) is well-developed, the theoretical advantages of the MoE architecture have not been formally characterized.

**Key Challenge**: MoE activates only $K$ experts at a time (typically $K=1$), with far lower parameter utilization than dense networks, yet matches or surpasses dense network performance. What is the theoretical origin of this efficiency advantage?

**Goal**: How strong is the approximation capacity of MoE under two common structural priors (low-dimensionality and sparsity)? What are the respective roles of depth and the number of experts?

**Key Insight**: Starting from classical approximation theory, the paper analyzes the $L_\infty$ approximation error of the MoE network class $\mathcal{H}_{l,m}^{L,E}$ for target functions.

**Core Idea**: MoE naturally decomposes a complex approximation problem into "multiple localized simple subproblems + assignment of inputs to experts"—experts solve local subproblems, and the gating network performs the assignment.

## Method

### Overall Architecture

The expressive power of MoE is studied at two levels:
- **Shallow MoE** (depth-2): Handles functions on low-dimensional manifolds, revealing the complementary roles of gating and experts.
- **Deep MoE** (depth-$2L$): Handles piecewise functions with compositional sparsity, revealing the complementary roles of depth and the number of experts.

### Key Designs

1. **Shallow MoE Overcomes the Curse of Dimensionality (Theorem 4.8)**

    - **Function**: Proves that a depth-2 MoE (with $E$ experts, each a 3-layer width-$m$ ReLU network) achieves approximation error $\max_i \tilde{O}(m^{-\kappa(f|_{U_i})/d \wedge 1/2})$ for a function $f$ on a $d$-dimensional manifold $\mathcal{M} \subset \mathbb{R}^D$.
    - **Mechanism**: The global approximation is decomposed into $E$ local subproblems. The error consists of two terms: (a) approximating the low-dimensional target function $f|_{U_i} \circ \phi_i^{-1}$ (in $d$ dimensions); and (b) approximating the smooth coordinate map $\phi_i$ (whose high-order smoothness yields small error). The rate is governed by intrinsic dimension $d$ rather than ambient dimension $D$.
    - **Design Motivation**: Dense networks achieve approximation rate $O(m^{-\kappa(f)/D})$; when $d \ll D$, MoE achieves an exponential advantage. The key insight is that the MoE structure naturally matches the atlas decomposition of a manifold.

2. **Complementary Roles of Experts and Gating**

    - **Function**: Reveals the division of labor between the two core components of MoE—Layer-2 expert networks approximate the local low-dimensional subfunctions $f|_{U_i} \circ \phi_i^{-1}$ and coordinate maps $\phi_i$; Layer-1 and Layer-2 gating jointly perform precise assignment of inputs to the correct experts.
    - **Mechanism**: Layer-1 behaves analogously to a dense network, approximating smooth partition-of-unity functions $\rho_i$ (partitioning the manifold into $E$ regions); Layer-2 gating selects the expert corresponding to the relevant region based on $\rho_i$.
    - **Design Motivation**: Standard linear gating lacks the capacity to model nonlinear $\rho_i$, necessitating an additional layer for their approximation. This explains why depth-2 rather than depth-1 is required.

3. **Exponential Task Capacity of Deep MoE (Theorem 5.2)**

    - **Function**: Proves that a depth-$2L$ MoE (with $E$ experts per layer) can approximate piecewise functions with $E^L$ pieces (possessing compositional sparsity on the product manifold $\mathcal{M}_1 \times \cdots \times \mathcal{M}_L$).
    - **Mechanism**: Each pair of MoE layers handles $E$ subfunctions $f_{l,i}$ on a sub-manifold $\mathcal{M}_l$. The composition of $L$ layer-pairs yields $E^L$ task combinations. The naive upper bound is $LE$ tasks (each expert handles one task independently), but the compositionally sparse structure enables an exponential breakthrough.
    - **Design Motivation**: Real-world LLMs handle diverse complex tasks (mathematics, logic, language, code), each corresponding to a piecewise function region. Tasks typically share sub-structures (compositional sparsity), enabling parameters to be reused across tasks.

4. **Compositional Sparsity: Mathematical Formalization**

    - **Function**: Defines the target function as $f(\mathbf{x}) = f_{\text{out}}(f_{1,i_1}(\mathbf{x}_1), \ldots, f_{L,i_L}(\mathbf{x}_L))$, where each subfunction $f_{l,i}$ depends only on a subspace $\mathbf{x}_l \in \mathcal{M}_l$ of the input.
    - **Mechanism**: Although there are only $LE$ subfunctions, their compositions yield $E^L$ distinct functions (e.g., in Example 5.1: 3 mathematical subtypes × 3 language subtypes = 9 tasks).
    - **Design Motivation**: Analogous to the modular compositional nature of real-world tasks—"solving a geometry problem in French" = "French comprehension" ∘ "geometry solving."

### Loss & Training

This paper is a purely approximation-theoretic work. Constructive proofs directly yield MoE networks satisfying the stated error bounds; no training procedure is involved.

## Key Experimental Results

### Main Results — Shallow MoE vs. Curse of Dimensionality (Experiment I)

| Input dimension $D$ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 1-4-MoE test error | 3.40e-4 | 3.38e-4 | 3.17e-4 | 3.42e-4 |

The target function is defined on a 1-dimensional manifold (unit circle) in $\mathbb{R}^D$, with $f(\mathbf{x}) = \sin(5x_1) + \cos(3x_2)$. MoE error does not increase with $D$, validating the theoretical prediction of overcoming the curse of dimensionality.

### Ablation Study — Deep MoE vs. Shallow MoE (Experiment II)

| Expert width $m$ | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| 2-3-MoE (2 layers, 3 experts/layer) | 8.32e-5 | 1.41e-5 | 4.73e-6 | **2.59e-6** |
| 1-6-MoE (1 layer, 6 experts/layer) | 7.96e-5 | 2.17e-5 | 2.65e-5 | 4.60e-5 |

The target is a piecewise function with $3^2=9$ pieces. The 2-3-MoE error decreases consistently with width; the 1-6-MoE plateaus for $m \geq 32$, validating the critical role of depth for compositional structures.

### Key Findings

1. **Dimensionality does not affect MoE accuracy**: When the target function is supported on a low-dimensional manifold, MoE approximation accuracy remains constant regardless of the ambient dimension $D$. This is the core theoretical advantage of MoE over dense networks.
2. **Depth is irreplaceable**: A shallow MoE (1-6-MoE) with comparable parameter count cannot match the accuracy of a deep MoE (2-3-MoE) on piecewise functions. Depth is a necessary condition for achieving hierarchical composition.
3. **The gap between $E^L$ and $LE$**: 2 layers × 3 experts = 6 experts yet capable of handling 9 tasks, precisely due to the compositionally sparse structure.

## Highlights & Insights

- **The theoretical division of labor between experts and gating is formalized for the first time**: Experts are responsible for local function approximation (the substantive work), while gating is responsible for input assignment (the routing work). Both are indispensable, and their roles are not interchangeable. This provides a solid theoretical foundation for understanding the role of the router in MoE.
- **The roles of depth and number of experts are distinct and complementary**: Depth controls hierarchical composition, while the number of experts controls subtask parallelism (specialization). More experts cannot simply substitute for greater depth, and vice versa.
- **Concrete recommendations for MoE architecture design**:
    - Nonlinear gating can reduce the required depth (eliminating the extra layer used to approximate $\rho_i$), consistent with recent empirical findings.
    - MoE–Dense interleaved architectures (GShard, GLAM) are equivalent in expressive power to consecutive MoE layers.
    - The shared + routed expert design (Qwen2, DeepSeek) is theoretically supported.
    - Low-dimensional expert networks (Encoder + low-dimensional FFN) can substantially reduce parameter counts.
- **Practical implication of $E^L$ exponential capacity**: A 32-layer × 8-expert MoE (realistic scale) can theoretically handle $8^{32} \approx 10^{28}$ combinatorial tasks—while this is an upper bound, it explains why MoE can cover a vast range of tasks with a limited number of parameters.

## Limitations & Future Work

1. **Realism of the compositional sparsity assumption**: The theory requires the target function to exhibit compositionally sparse structure on a product manifold; whether real-world LLM tasks strictly satisfy this assumption remains unclear.
2. **Approximation theory only, no training analysis**: Constructive proofs assume weights can be set exactly; whether SGD/Adam can find such solutions is not addressed. The training dynamics of the router (load balancing, non-differentiable TopK) further complicate learnability analysis.
3. **ReLU activation assumption**: All theoretical results are derived for ReLU activations; generalization to modern activations such as GELU and SwiGLU remains to be explored.
4. **Simplification to $K=1$**: Only top-1 routing is analyzed, whereas $K=2$ is common in practice. Extension to $K > 1$ is theoretically straightforward but may affect the specific form of approximation rates.
5. **Limited scale of validation experiments**: Experiments I and II use very small MoE models (4–6 experts, width 10–128), far smaller than the MoE models deployed in actual LLMs.

## Related Work & Insights

- **Shaham et al. (2018); Chen et al. (2019)**: Show that dense networks can also efficiently approximate functions on low-dimensional manifolds, but require stronger regularity assumptions on the manifold and activate all parameters (MoE activates only 1 expert, yielding $E$-fold parameter efficiency).
- **Mhaskar & Poggio (2016); Poggio (2023)**: Analyze the approximation capacity of dense neural networks under compositionally sparse structures; this paper extends their results to MoE.
- **Baykal et al. (2022)**: Show that sparsely activated networks can match the approximation performance of dense networks; this paper further quantifies the additional advantage conferred by the MoE structure.
- **Chen et al. (2022)**: Analyze training dynamics of softmax-gating MoE; this paper provides a complementary perspective from expressive power rather than training dynamics.
- **Insights**: The theoretically motivated Encoder + low-dimensional FFN expert architecture warrants empirical validation in practical MoE systems; the theoretical necessity of nonlinear gating can guide next-generation MoE design.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First systematic approximation-theoretic analysis of MoE expressive power; both the shallow and deep MoE results make important contributions. The $E^L$ exponential capacity result is particularly illuminating.
- **Experimental Thoroughness**: ⭐⭐⭐ — Primarily a theoretical work; validation experiments are small-scale but directionally sound. Experiments at realistic MoE scale would substantially strengthen the empirical case.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are rigorous yet clearly presented; Example 5.1 (multilingual + mathematical tasks) provides an intuitive illustration.
- **Value**: ⭐⭐⭐⭐⭐ — Provides a solid theoretical foundation for MoE architecture design; several recommendations (nonlinear gating, low-dimensional experts, interleaved architectures) are already corroborated by practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Let the Experts Speak: Improving Survival Prediction & Calibration via Mixture-of-Experts Heads](let_the_experts_speak_improving_survival_prediction_calibration_via_mixture-of-e.md)
- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](../../ICML2026/llm_efficiency/probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](../../ICML2026/llm_efficiency/hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ACL 2026\] The Hallucination of Specialization: Revealing the "Standing Committee" in Mixture-of-Experts Models](../../ACL2026/llm_efficiency/the_illusion_of_specialization_unveiling_the_domain-invariant_34standing_committ.md)

</div>

<!-- RELATED:END -->
