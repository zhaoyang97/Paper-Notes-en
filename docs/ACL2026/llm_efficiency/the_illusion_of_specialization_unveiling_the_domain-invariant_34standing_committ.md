---
title: >-
  [Paper Note] The Hallucination of Specialization: Revealing the "Standing Committee" in Mixture-of-Experts Models
description: >-
  [ACL2026][LLM Efficiency][Mixture-of-Experts] By introducing the CommitteeAudit framework, the authors discover a "Standing Committee" in MoE models—a compact…
tags:
  - "ACL2026"
  - "LLM Efficiency"
  - "Mixture-of-Experts"
  - "Routing Analysis"
  - "Expert Specialization"
  - "Model Interpretability"
  - "Sparse Computation"
date: 2026-05-08
content_hash: 0defb15740b3ece1
---

# The Hallucination of Specialization: Revealing the "Standing Committee" in Mixture-of-Experts Models

**Conference**: ACL2026  
**arXiv**: [2601.03425](https://arxiv.org/abs/2601.03425)  
**Code**: https://github.com/The-FinAI/CommitteeAudit  
**Area**: Model Compression / LLM Efficiency  
**Keywords**: Mixture-of-Experts, Routing Analysis, Expert Specialization, Model Interpretability, Sparse Computation

## TL;DR
By introducing the CommitteeAudit framework, the authors discover a "Standing Committee" in MoE models—a compact, persistent ensemble of experts that are consistently activated across different domains and occupy the majority of routing weights. This stands in sharp contrast to the widely assumed domain-specific specialization, revealing an inherent centralized structure in sparse computation.

## Background & Motivation

**Background**: Mixture-of-Experts (MoE) models have become the primary direction for scaling Large Language Models. Through a "divide and conquer" strategy—routing inputs from different domains to specialized experts—conditional computation can theoretically be achieved while avoiding a linear increase in inference latency. Recent architectures like DeepSeek even introduce shared expert layers to attempt to force the specialization of non-shared experts through architectural isolation.

**Limitations of Prior Work**: However, previous research on representation collapse warns that the optimization dynamics of routing networks frequently conflict with the ideal of "expert specialization." More critically, even in architectures with shared experts, routed experts exhibit significant cross-domain overlap—this is not an optimization failure, but a prerequisite for these experts to remain active and functionally effective while still refusing specialization.

**Key Challenge**: Load-balancing auxiliary loss functions widely adopted in MoE training aim to encourage uniform expert utilization and prevent expert "death." But if the natural optimization path of the model actually tends toward centralized computation—where a "Standing Committee" dominates inference across all domains—then these loss functions are essentially working against the model's inherent tendencies.

**Goal**: To analyze the routing organization of MoE at the group level (rather than individual experts) to confirm the existence of domain-invariant expert connectomes and how they evolve across depth and architecture.

**Core Idea**: Shift from individual expert statistics to "committee-level" structural analysis—using Pareto optimality and stability diagnostics to quantify how expert groups are organized, rather than analyzing activation frequencies in isolation.

## Method

### Overall Architecture
CommitteeAudit is a three-stage post-hoc analysis framework. It first extracts task-level routing features from pre-trained MoE models (quantified by Jaccard similarity and Gini coefficients), then assesses the degree of task-specificity in routing, and finally identifies stable "Standing Committees" via Pareto optimization—expert groups that consistently occupy Top-k routing positions across multiple domains with low ranking variance.

### Key Designs

1. **Expert Contribution Index (ECI) and Task-Specificity Score**:

    - **Function**: Aggregates from full routing vectors to domain-task level measures of expert importance.
    - **Mechanism**: ECI is defined as $c_{i,\tau}^{(\ell)} = \mathbf{E}_{x \in \mathcal{D}_\tau}[G^{(\ell)}(x)_i]$—the average routing weight of expert $i$ in layer $\ell$ for domain $\tau$. Unlike simple activation frequency, ECI preserves information about the magnitude of router preferences. Task-specificity is measured using a silhouette coefficient-based method: $S_\ell(\tau) = \frac{1}{|\mathcal{D}_\tau|}\sum_{x_i} \frac{b_i - a_i}{\max(a_i, b_i)}$, where $a_i$ is the intra-domain distance and $b_i$ is the nearest inter-domain distance.
    - **Design Motivation**: These two metrics build a bridge from individual routing behavior to domain-level patterns. Subsequent "Standing Committee" analysis is performed only in domains with sufficiently high task-specificity to filter out tasks with mixed routing.

2. **Pareto Optimal Committee Identification**:

    - **Function**: Filters "Standing Committees" from candidate experts that are both top-ranked and stable across domains.
    - **Mechanism**: First, experts are filtered using a consistency threshold $\gamma > 0.8$ for those appearing in the Top-k in $\geq 80\%$ of domains. Then, the optimal trade-off is selected using the Pareto curve $\{(\mu_i, \sigma_i): \mu_i = \mathbf{E}_\tau[R(i,\tau)], \sigma_i = \mathrm{Var}_\tau[R(i,\tau)]\}$—identifying experts with high mean rank and low cross-domain rank variance.
    - **Design Motivation**: Effective "committee members" should be frequently used across multiple domains with consistent usage patterns. Pareto optimization avoids hard thresholds and automatically finds this trade-off, solving issues where frequency-only filtering includes unstable experts and variance-only filtering selects consistently low-ranked experts.

3. **Qualitative Functional Analysis and Masking Intervention**:

    - **Function**: Verifies whether the Standing Committee truly carries critical semantic functions through token-level activation matrices and masking experiments.
    - **Mechanism**: Activation matrices record which tokens consistently activate the same committee expert across at least three different domains. Masking experiments involve zeroing out the routing weights of committee experts (followed by re-normalization) and observing performance degradation on MMLU. Qualitative analysis reveals that abstract reasoning words ("Which", "What", "Suppose") and high-frequency structural words ("the", "a", "in") are routed to the same committee experts, while domain-specific terms are dispersed to peripheral experts. Masking experiments show that disabling the committee drops accuracy from 0.39 to 0.03-0.12, with "No Answer" rates jumping from 3% to 36%-38%.
    - **Design Motivation**: Qualitative and quantitative evidence corroborate each other, refuting the suspicion that the "Standing Committee" is merely a high-frequency statistical coincidence and proving they carry a genuine computational role.

## Key Experimental Results

### Main Results: Existence and Stability of Standing Committees

Experiments were conducted on three MoE models of different scales and architectures (OLMoE-1B-7B, DeepSeek-V2-Lite, Qwen3-30B-A3B) using nine semantic domains from MMLU.

| Metric | Statistic | OLMoE | DeepSeek-V2-Lite | Qwen3-30B-A3B |
|------|--------|-------|----------------|--------------|
| Jaccard Similarity | Max | 1.0 | 1.0 | 1.0 |
|  | Min | 0.7963 | 0.7103 | 0.5300 |
|  | Global Avg | 0.8735 | 0.8670 | 0.8670 |
| Gini Coefficient | Max | 0.9082 | 0.9360 | 0.9605 |
|  | Min | 0.8814 | 0.9092 | 0.9405 |
|  | Global Avg | 0.8957 | 0.9207 | 0.9465 |

**Interpretation**: High Jaccard values indicate that the models activate nearly identical sets of Top-k experts even across different domains. Gini coefficients >0.88 indicate that routing weights are monopolized by a few experts. A larger expert pool (128 in Qwen3) does not alleviate this centralization but rather intensifies the Gini values.

### Scale and Contribution of Standing Committees

| Model | Stage | Layer | Committee Size | Mean Rank $\mu$ | Rank Var $\sigma^2$ | ECI Coverage | Relative Impact Density |
|------|------|-----|-----------|--------------|----------------|-----------|----------|
| DeepSeek-V2-Lite | Shallow | 3 | 4 | 3.36 | 1.81 | 66.3% | 29.5× |
|  | Middle | 11 | 3 | 3.15 | 1.98 | 60.7% | 31.4× |
|  | Deep | 19 | 4 | 3.11 | 0.76 | 70.5% | 35.8× |
| OLMoE | Shallow | 2 | 3 | 3.41 | 2.15 | 43.9% | 15.9× |
|  | Middle | 8 | 2 | 3.28 | 0.49 | 29.7% | 13.1× |
|  | Deep | 16 | 3 | 3.19 | 1.52 | 44.0% | 16.0× |

**Interpretation**: Committee size remains constant at 2-5 members yet accounts for 60%-70% of routing weights. Even in Qwen3 with up to 128 experts, the committee still consists of only 3-5 individuals. Relative impact density (factor of each committee member relative to a uniform distribution baseline) reached 35.8× in DeepSeek’s deep layers, indicating extremely high computational density.

### Masking Intervention Experiments

| Masked Layer Stage | Layer No. | Accuracy | Error Rate | No Answer Rate |
|-----------|------|--------|--------|---------|
| Baseline (No Mask) | — | 0.39 | 0.58 | 0.03 |
| Shallow | 2 | 0.12 | 0.52 | 0.36 |
| Middle | 10 | 0.09 | 0.55 | 0.36 |
| Deep | 26 | 0.03 | 0.59 | 0.38 |

**Interpretation**: Disabling the committee in any layer leads to a sharp performance drop. Masking in deep layers is most fatal (accuracy drops to 3%), indicating that the underlying reasoning skeleton relies heavily on the stable support of the committee.

## Highlights & Insights

- **Breaking the "Specialization" Assumption**: Directly challenges a core assumption of MoE design philosophy. Even in architectures that explicitly isolate shared experts (e.g., DeepSeek), the routed "specialized" experts still form hidden Standing Committees. Centralized computation is not an architectural choice but an **inherent necessity of sparse routing**.
- **The Paradox of Load Balancing**: Standard load-balancing losses attempt to force uniform usage of all experts. However, if the model's natural optimal path is centralized, these loss functions are actually penalizing the model's intrinsic tendencies, potentially acting as a **source of restricted training efficiency and performance**.
- **Core-Periphery Division of Labor**: Qualitative analysis reveals a sophisticated division of labor—committee members act as reasoning controllers and syntactic skeletons, while peripheral experts handle domain knowledge on demand. This pattern is consistent across DeepSeek and Qwen3, suggesting it may be a **universal property of sparse computation**.
- **Cross-Architecture Universality**: From OLMoE with E=64 to Qwen3 with E=128, and from full routing to hybrid shared designs, the Standing Committee phenomenon consistently emerges, indicating this is a system-level phenomenon rather than a bug in a specific architecture.

## Limitations & Future Work

**Limitations**:

- Limited Experimental Coverage: Only three MoE models were evaluated, excluding more complex designs such as hierarchical routing or dynamic routing.
- Incomplete Causality: Although masking trials provide evidence of intervention, systematic controls (e.g., comparing random masks or frequency-matched masks) have not yet been performed.
- Finite Evaluation Scope: Analysis was primarily at the MMLU domain level; generalization to scenarios like multi-step reasoning, programming, or tool use remains unknown.
- Neglect of Dynamic Learning: CommitteeAudit is a post-hoc analysis and does not track **when and how** Standing Committees emerge during training.

**Future Work**:

- Design Awareness-based Routing Objectives: Instead of forcing uniform usage, **explicitly encourage core-periphery division of labor**—for example, by setting different target utilization rates for different experts.
- Extension to the Training Process: Monitor the formation of Standing Committees throughout the process from random initialization to convergence.
- Validation Across More Architectures and Datasets: Particularly datasets involving challenging scenarios such as long context, multi-language, and code.

## Related Work & Insights

- **vs Individual Expert Specialization Analysis** (Lo et al. 2025; Olson et al. 2025): These works focus on the semantic routing or activation patterns of single experts, making it difficult to capture the **synergistic structure** between experts. This paper's group perspective fills this gap.
- **vs Super-expert Discovery** (Su et al. 2025): While both focus on high-frequency experts, work on super-experts emphasizes the Pareto distribution of individual experts, whereas this paper reveals the **stable connectomes** formed by these high-frequency experts and their cross-domain invariance.
- **vs Representation Collapse Research** (Chi et al. 2022): Representation collapse emphasizes redundancy caused by optimization failure. The centralization found in this paper is a product of **successful optimization**—experts are not dead but are actively participating in computation.
- **Insight**: Future MoE design should shift from "how to force diversification" to "how to align with the model's natural structure," designing architectures that both respect this core-periphery division of labor and assign meaningful tasks to peripheral experts.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Shifting from individual experts to group-level analysis is a major paradigm shift in MoE interpretability, challenging the core assumption of "specialization."
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three models, multiple evaluation metrics (Jaccard, Gini, masking), and qualitative case studies provide strong support for the main findings, though coverage of more complex architectures and training dynamics is lacking.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logically clear, precisely articulated, with a well-defined problem setting and powerful argumentation.
- **Value**: ⭐⭐⭐⭐⭐ Directly inspires the design and optimization of MoE models, particularly regarding the reconsideration of load-balancing objective functions and revealing the inherent characteristics of sparse computation, with substantial impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] How Many Experts Are Enough? Towards Optimal Semantic Specialization for Mixture-of-Experts](../../AAAI2026/llm_efficiency/how_many_experts_are_enough_towards_optimal_semantic_specialization_for_mixture-.md)
- [\[ACL 2026\] Alloc-MoE: Budget-Aware Expert Activation Allocation for Efficient Mixture-of-Experts Inference](alloc-moe_budget-aware_expert_activation_allocation_for_efficient_mixture-of-exp.md)
- [\[ICML 2026\] ProbMoE: Differentiable Probabilistic Routing for Mixture-of-Experts](../../ICML2026/llm_efficiency/probmoe_differentiable_probabilistic_routing_for_mixture-of-experts.md)
- [\[NeurIPS 2025\] Advancing Expert Specialization for Better MoE](../../NeurIPS2025/llm_efficiency/advancing_expert_specialization_for_better_moe.md)
- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](../../ICML2026/llm_efficiency/hyperparameter_transfer_with_mixture-of-expert_layers.md)

</div>

<!-- RELATED:END -->
