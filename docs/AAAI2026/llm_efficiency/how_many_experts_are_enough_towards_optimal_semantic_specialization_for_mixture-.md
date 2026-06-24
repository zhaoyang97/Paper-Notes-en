---
title: >-
  [Paper Note] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts
description: >-
  [AAAI 2026][LLM Efficiency][Mixture-of-Experts] This paper proposes MASS, a framework that adaptively expands the MoE expert pool via gradient-based semantic drift detection, combined with a Top-p confidence routing strategy, to automatically discover the optimal number of experts without hyperparameter search while enhancing semantic differentiation across experts.
tags:
  - "AAAI 2026"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "expert scaling"
  - "semantic specialization"
  - "gradient drift detection"
  - "dynamic routing"
  - "Top-p routing"
date: 2026-05-08
content_hash: 2f4dce7efe923d19
---

# How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts

**Conference**: AAAI 2026  
**arXiv**: [2512.19765](https://arxiv.org/abs/2512.19765)  
**Authors**: Sumin Park, Noseong Park  
**Code**: Not released  
**Area**: LLM Efficiency  
**Keywords**: Mixture-of-Experts, expert scaling, semantic specialization, gradient drift detection, dynamic routing, Top-p routing  

## TL;DR

This paper proposes MASS, a framework that adaptively expands the MoE expert pool via gradient-based semantic drift detection, combined with a Top-p confidence routing strategy, to automatically discover the optimal number of experts without hyperparameter search while enhancing semantic differentiation across experts.

## Background & Motivation

### State of the Field
Sparse Mixture-of-Experts (SMoE) is an effective approach for scaling large model capacity by selectively routing tokens to a small subset of expert sub-networks, trading sparse computation for increased model scale. SMoE has been widely adopted in LLMs (e.g., Mixtral, DeepSeekMoE, Qwen2) and vision Transformer architectures.

### Limitations of Prior Work
- **Expert count K is highly sensitive to hyperparameter search**: Existing methods typically pre-define a fixed number of experts K and top-k activations, requiring extensive grid search resources.
- **Limitations of DynMoE**: Although DynMoE attempts to adaptively adjust the expert pool size, its expansion criterion is based solely on statistics of tokens not activated by any expert, without explicitly assessing whether the expert pool has reached semantic saturation.
- **Neglect of semantic specialization**: Existing methods overlook the core challenge of MoE modeling—fine-grained semantic division of labor among experts—leading to functional redundancy and insufficient complementarity.

### Core Problem
The paper aims to design a semantics-aware adaptive MoE expansion framework that determines when to add new experts by detecting gradient-level semantic drift, thereby automatically finding the optimal expert count and maximizing semantic differentiation across experts.

## Method

### Overall Architecture: MASS
MASS (Mixture-of-Experts for Adaptive Semantic Specialization) consists of two training phases:
1. **Adaptive expansion phase** (first 10% of training steps): Dynamically expands the expert pool via gradient monitoring.
2. **Standard training phase** (remaining 90% of training steps): Fixes the expert set and performs standard gradient training.

### MoE Layer Structure
Given an input token representation $\mathbf{x} \in \mathbb{R}^d$, the router selects a subset of experts $\mathcal{N}(\mathbf{x})$, and the output is:

$$\mathbf{y} = \sum_{k \in \mathcal{N}(\mathbf{x})} r_k(\mathbf{x}) \cdot e_k(\mathbf{x})$$

where the routing weights are $\mathbf{r}(\mathbf{x}) = \text{Softmax}(\mathbf{x}^\top \mathbf{W}_r)$.

### Key Design 1: Top-p Confidence Routing
Unlike conventional top-k routing with a fixed number of activated experts, MASS adopts a Top-p strategy: given sorted routing scores, the minimal subset of experts whose cumulative probability exceeds threshold $p$ is selected:

$$\sum_{j=1}^{k^*} r^{(j)} \geq p, \quad \mathcal{N}(\mathbf{x}) = \{\mathcal{I}_1, \dots, \mathcal{I}_{k^*}\}$$

**Effect**: Tokens with high routing confidence activate only a few experts (saving computation), while uncertain tokens are routed to more experts for richer processing.

### Key Design 2: Gradient Monitoring via Probabilistic Change-Point Detection
For each expert $e_k$, the gradient L2 norm at training step $t$ is tracked: $g_t^{(k)} = \|\nabla_{\theta_k} \mathcal{L}_t\|_2$.

After warmup, a sliding window $\omega$ of gradient norms is maintained, and a cumulative z-score is computed and normalized:

$$\tilde{s}_t^{(k)} = s_t^{(k)} / \sqrt{\omega}$$

Under the null hypothesis of normality, the right-tail p-value is $p_t^{(k)} = 1 - \Phi(\tilde{s}_t^{(k)})$. If $p_t^{(k)} \leq \alpha$, a significant upward trend in gradient magnitude is detected, indicating that the expert may require substantial adjustment of its representations.

### Key Design 3: Semantic Alignment Test
For an expert $e_k$ flagged by gradient monitoring, the cosine similarity between the gradient matrix and the weight matrix is computed:

$$\cos(\nabla^{(k)}, \mathbf{W_e}^{(k)}) = \frac{\langle \nabla^{(k)}, \mathbf{W_e}^{(k)} \rangle}{\|\nabla^{(k)}\|_2 \cdot \|\mathbf{W_e}^{(k)}\|_2}$$

If $\|\cos(\nabla^{(k)}, \mathbf{W_e}^{(k)})\| < \delta$ (with $\delta = 0.001$), semantic drift is declared and expert expansion is triggered.

### Key Design 4: Expert Duplication and Gradient Decomposition
Upon confirming semantic drift, expert $e_k$ is duplicated to produce a new expert $e_k'$:
- **New expert**: Receives the full gradient update to explore a divergent semantic role.
- **Original expert**: Receives only the gradient component aligned with its current weights, preserving its existing semantics.

Gating vectors are duplicated synchronously, and a redundancy regularization loss is applied to prevent functional collapse:

$$\mathcal{L}_{\text{red}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j) \in \mathcal{P}} (\cos(\mathbf{w}_i, \mathbf{w}_j))^2$$

### Expansion Stopping Criteria
Expansion stops when either of the following conditions is met:
1. The number of experts reaches the upper bound $K_{\max}$.
2. Adding a new expert fails to improve the loss (verified via NLL comparison, with $\gamma$ patience counts allowed).

## Key Experimental Results

### Experiment 1: Synthetic Data — Optimal MoE Configuration Discovery

A polynomial HMM-based GINC synthetic dataset is used, with 5 latent concepts and structured semantic labels (entity and property). The architecture is a single-layer Transformer + MoE.

| Method | Avg. Expert Count K | Avg. Active Experts k | Avg. Test Loss | Requires Hyperparameter Search |
|--------|--------------------|-----------------------|----------------|-------------------------------|
| Naive MoE (K=5, best top-k) | 5 | Fixed | ~2.6 | Yes |
| Naive MoE (K=10, best top-k) | 10 | Fixed | ~2.25 | Yes |
| Naive MoE (K=15, best top-k) | 15 | Fixed | ~2.2 | Yes |
| Naive MoE (K=20, best top-k) | 20 | Fixed | ~2.3 | Yes |
| **MASS (avg. over 5 runs)** | **12.4** | **3.9** | **2.15** | **No** |

MASS automatically converges to an expert count close to the empirical "elbow" (K=15), achieving lower test loss with fewer experts and validating its optimality in cost-performance trade-off. JSD analysis shows that MASS exhibits significantly higher routing differentiation along both the entity and property semantic dimensions.

### Experiment 2: Real-World Tasks — GLUE Language Understanding and DomainBed Visual Generalization

**GLUE Benchmark (BERT-large fine-tuning)**: MASS matches or outperforms other MoE variants on CoLA, QNLI, RTE, MNLI, and MRPC, achieving the highest average top-1 accuracy. The MASS expert pool adaptively expands to K between 9.5 and 12.7, while DynMoE converges to a fixed K=9.0. A key distinction is that MASS activates only 25%–30% of experts on average (k=2.6–3.2), compared to DynMoE's 70%–90% (k=6.5–8.0).

| Method | PACS | VLCS | OfficeHome | TerraIncognita | Avg. |
|--------|------|------|-----------|---------------|------|
| GMoE (K=4) | 88.2 | 79.8 | 73.5 | 47.8 | 72.3 |
| GMoE (K=6) | 88.1 | 80.2 | 74.2 | 48.5 | 72.8 |
| GMoE (K=8) | 88.2 | 80.0 | 74.2 | 47.2 | 72.4 |
| DynMoE (Gshard Loss) | 88.4 | 79.4 | 73.6 | 46.6 | 72.0 |
| **MASS** | **88.7** | **81.1** | **73.8** | 47.5 | **72.8** |

MASS outperforms DynMoE across all four DomainBed visual domain generalization datasets, achieving the best result of 81.1% on VLCS. Its average accuracy matches the best fixed GMoE and surpasses DynMoE.

## Highlights & Insights

- **Semantics-aware expansion mechanism**: Unlike DynMoE's heuristic based on token coverage, MASS directly identifies semantic insufficiency via a two-stage detection pipeline (gradient change-point detection + semantic alignment test), enabling more principled expert expansion.
- **Automatic discovery of optimal expert count**: Synthetic experiments validate that MASS automatically converges to the elbow of the cost-performance trade-off curve, eliminating the need for hyperparameter search over K.
- **Sparse and efficient routing**: Top-p routing causes MASS to activate only 25%–30% of the expert pool, far fewer than DynMoE's 70%–90% activation rate, yielding significant gains in computational efficiency.
- **Cross-domain robustness**: MASS performs consistently well across synthetic data, NLU (GLUE), and visual generalization (DomainBed) tasks.
- **Elegant gradient decomposition**: During expert duplication, gradient decomposition into aligned and orthogonal components preserves the original expert's semantics while allowing the new expert to explore divergent directions.

## Limitations & Future Work

- **Limited experimental scale**: Validation is restricted to BERT-large and ViT-S/16; no experiments involve truly large-scale LLMs (e.g., 7B+ parameters), leaving applicability to very large models uncertain.
- **Expansion confined to the first 10% of steps**: The expansion window is hard-coded to the first 10% of training, which lacks flexibility for scenarios requiring longer warmup or exhibiting semantic drift at later stages.
- **New hyperparameter sensitivity**: Although K search is eliminated, new hyperparameters are introduced, including the semantic alignment threshold $\delta=0.001$, CPD significance level $\alpha$, and the Top-p threshold $p$.
- **Absence of large-scale pretraining experiments**: All experiments are conducted in fine-tuning settings; the effectiveness of MASS for pretraining large models from scratch remains unverified.
- **Limited gains on vision tasks**: On DomainBed, MASS matches the best fixed GMoE (K=6), with its advantage primarily manifesting as adaptivity rather than absolute performance improvement.
- **No expert pruning mechanism**: Only an expansion mechanism is provided; a complementary operation for pruning redundant experts is absent.

## Related Work & Insights

- **DynMoE (Guo et al. 2025)**: Expands experts based on token-expert coverage rate without assessing semantic saturation; MASS achieves more refined expansion decisions through gradient-based semantic drift detection.
- **GMoE (Li et al. 2023)**: A fixed MoE initialized from pretrained ViT for domain generalization; MASS achieves comparable performance under the same setting with an adaptive expert count.
- **MoEfication (Zhang et al. 2022)**: Restructures pretrained FFN layers into sparse expert modules; MASS validates language task performance within the same fine-tuning framework.
- **Switch Transformer (Fedus et al. 2022)**: A classic SMoE design with top-1 routing; MASS replaces fixed top-k with Top-p routing to enable adaptive routing.
- **DeepSeekMoE (Dai et al. 2024)**: Improves LLM efficiency through finer-grained expert partitioning; this is complementary to MASS (MASS addresses when to add experts, DeepSeekMoE addresses how to partition experts).
- **Top-p Routing**: MASS draws on the Top-p routing concept and integrates it organically with the adaptive expansion mechanism.

## Rating

- Novelty: ⭐⭐⭐⭐ — The two-stage expansion scheme combining gradient change-point detection and semantic alignment testing is novel and theoretically grounded.
- Experimental Thoroughness: ⭐⭐⭐ — The synthetic experiment is elegantly designed, but real-world task scale is limited and large-model validation is absent.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, with well-articulated method intuitions and informative visual comparisons.
- Value: ⭐⭐⭐⭐ — Addresses the practical challenge of expert count selection in MoE, though larger-scale experiments are needed to substantiate the claims.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Illusion of Specialization: Revealing the "Standing Committee" in Mixture-of-Experts Models](../../ACL2026/llm_efficiency/the_illusion_of_specialization_unveiling_the_domain-invariant_34standing_committ.md)
- [\[ICLR 2026\] DirMoE: Dirichlet-Routed Mixture of Experts](../../ICLR2026/llm_efficiency/dirmoe_dirichlet-routed_mixture_of_experts.md)
- [\[ICML 2025\] Mixture of Lookup Experts](../../ICML2025/llm_efficiency/mixture_of_lookup_experts.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](../../ICML2026/llm_efficiency/probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[ICLR 2026\] Understanding the Mixture-of-Experts with Nadaraya-Watson Kernel](../../ICLR2026/llm_efficiency/understanding_the_mixture-of-experts_with_nadaraya-watson_kernel.md)

</div>

<!-- RELATED:END -->
