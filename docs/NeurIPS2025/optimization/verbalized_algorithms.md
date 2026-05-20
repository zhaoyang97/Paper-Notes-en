---
title: >-
  [Paper Note] Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees
description: >-
  [NeurIPS 2025][Optimization][Verbalized Algorithms] This paper proposes the Verbalized Algorithms (VAs) framework, which preserves the control flow of classical algorithms while replacing their atomic operations (e.g.…
tags:
  - "NeurIPS 2025"
  - "Optimization"
  - "Verbalized Algorithms"
  - "LLM Reasoning"
  - "Classical Algorithms"
  - "Sorting"
  - "Submodular Maximization"
date: 2026-05-08
content_hash: 5058507bd0fb1d04
---

# Verbalized Algorithms: Zero-shot Classical Algorithmic Reasoning for Correctness and Runtime Guarantees

**Conference**: NeurIPS 2025
**arXiv**: [2509.08150](https://arxiv.org/abs/2509.08150)  
**Code**: None  
**Area**: Optimization / LLM Reasoning
**Keywords**: Verbalized Algorithms, LLM Reasoning, Classical Algorithms, Sorting, Submodular Maximization

## TL;DR
This paper proposes the Verbalized Algorithms (VAs) framework, which preserves the control flow of classical algorithms while replacing their atomic operations (e.g., binary comparisons) with LLM queries, thereby inheriting the correctness and complexity guarantees of classical algorithms for natural language reasoning tasks. The framework is validated across four case studies: sorting, maximum finding, clustering, and submodular maximization.

## Background & Motivation
LLM outputs on reasoning tasks lack correctness guarantees. For computational reasoning tasks such as sorting and search, prompting an LLM to produce answers in a single pass is not only unreliable but also constrained by context length. Classical algorithms (e.g., merge sort, greedy submodular maximization) offer well-established theoretical guarantees but cannot directly handle natural language inputs. Existing formalization-based methods use LLMs to translate natural language into formal languages (e.g., PDDL) for downstream solvers, but suffer from three issues: (1) they require the ability to generate complex formal syntax; (2) translations are often inaccurate; and (3) the expressiveness of formal languages limits the range of tractable tasks.

The core insight of this paper is that most reasoning tasks can be decomposed into simple atomic operations (e.g., pairwise comparisons), which LLMs can perform reliably. By preserving the control flow of classical algorithms and replacing only the atomic operations with LLM calls, one can simultaneously obtain the theoretical guarantees of classical algorithms and the flexibility of LLMs in processing natural language.

## Method

### Overall Architecture
Verbalized Algorithms are formally defined as follows: given a computational task $T[\tau]$ over type $\tau$ and an algorithm $A[\tau]$ using an $n$-ary Boolean oracle $f[\tau]$, a verbalized algorithm instantiates $\tau$ as $\text{str}$ (strings), implements the oracle $f[\text{str}]$ via LLM queries with outputs constrained to $\{\text{yes}, \text{no}\} \mapsto \{\top, \bot\}$, and fully preserves the high-level control flow.

VAs are categorized into three types:
- **Naive VA**: treats LLM responses directly as ground truth.
- **Robust VA**: treats the LLM as a noisy oracle, using majority voting over multiple samples to quantify and control error rates.
- **Probabilistic VA**: leverages LLM output probabilities or logits to improve accuracy or quantify uncertainty.

### Key Designs

1. **Verbalized Maximum**: Overrides Python's `__gt__` operator with an LLM call and invokes the built-in `max` function, performing $O(n)$ sequential comparisons. Each comparison queries "Is X larger than Y?" with constrained decoding to restrict outputs to yes/no. Even on a 1.7B model, verbalized maximum yields substantially lower error than prompting the LLM to produce the answer directly.

2. **Verbalized Sorting**: Two implementations are provided — (a) **VA Powersort**: overrides `__gt__` and calls Python's built-in `sorted()`, performing $O(n \log n)$ comparisons; (b) **VA Bitonic**: uses a bitonic sorting network from the parallel sorting literature, requiring $O(n(\log n)^2)$ comparisons but only $O((\log n)^2)$ parallel time. A symmetrization strategy is also introduced to mitigate LLM positional bias and sycophancy bias.

3. **Verbalized Clustering**: Combines an $O(n \log n)$ approximate Delaunay graph construction (based on ternary comparisons of the form "Is Z more similar to X than to Y?") with $O(n \log^2 n)$ greedy modularity maximization. After constructing an approximate similarity graph, community detection algorithms are applied for clustering.

4. **Verbalized Greedy Submodular Maximization (VGSM)**: Applied to document retrieval in multi-hop QA. The evaluation function in greedy submodular maximization is replaced by implicit LLM scoring. The LLM provides relative preference rankings via a Plackett–Luce reward model without requiring absolute scores, while preserving the $1 - 1/e$ optimality guarantee.

### Efficiency Optimizations
- Bitonic sorting networks support parallelization, achieving 2.1–2.3× speedup over sequential Powersort in practice.
- KV-cache prefix sharing in inference engines (e.g., SGLang's RadixAttention) accelerates the numerous shared-prefix queries in VAs.
- Constrained decoding ensures well-formed outputs (yes/no or list elements).

## Key Experimental Results

### Maximum Finding (Random Integer Lists, MAE)

| Model | $n$=10 Baseline | $n$=10 VA | $n$=100 Baseline | $n$=100 VA |
|-------|----------------|-----------|-----------------|-----------|
| Qwen3-1.7B | 500.14 | **0** | 1418.9 | 9.45 |
| Qwen3-4B | 0 | **0** | 34.25 | **0** |
| Qwen3-32B | 0 | **0** | 48.6 | **0** |

### Sorting (Amazon Review Sentiment Ranking, Kendall-Tau, higher is better)

| Method | Qwen3-1.7B | Qwen3-32B |
|--------|-----------|-----------|
| Constraint Decoding Baseline | 0.00 | 0.30 |
| I.I.D Scoring | 0.18 | 0.57 |
| Naive VA Bitonic | 0.30 | 0.39 |
| Naive VA Powersort | 0.33 | 0.56 |
| Robust VA Powersort ($K$=3) | 0.44 | 0.58 |
| Symmetrized VA Powersort | **0.48** | **0.60** |

### Clustering (ARI)

| Method | Medarxiv | xkcdcolors-h | xkcdcolors-l (OOD) |
|--------|---------|-------------|-------------------|
| Qwen3-Embedding-8B (KMeans) | 0.11 | 0.43 | 0.01 |
| VA Clustering (Qwen3-32B, $k$=20) | **0.17** | **0.46** | **0.11** |

### Submodular Maximization / RAG (HotpotQA, logp, $k$=12)

| Method | $\phi$=0.6B | $\phi$=32B |
|--------|------------|-----------|
| Embedding (top-$k$) | baseline | baseline |
| Verbalized Sorting | below VGSM | below VGSM |
| VGSM (2k pre-selection) | significantly above Embedding | significantly above Embedding |

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|-----------|-------|
| Robust VA ($K$=3 majority vote) | Kendall-Tau | 10–20% gain over Naive VA; improves robustness |
| Symmetrized vs. non-symmetrized | Kendall-Tau | Symmetrization eliminates positional and sycophancy bias |
| Bitonic vs. Powersort runtime | Wall time | Bitonic is 1.2–2.3× faster due to parallelization |
| KV-cache (SGLang) | Wall time | RadixAttention yields 1.3–1.4× speedup |
| VGSM 4k vs. 2k pre-selection | logp | Smaller models degrade at 4k due to limited long-context understanding |

### Key Findings
- A 1.7B VA outperforms a 32B baseline on sorting, demonstrating that algorithmic structure matters more than model scale.
- Constraint decoding baselines generate duplicate or missing elements, violating sorting invariants; VAs are immune to this failure mode.
- VA Clustering substantially outperforms embedding-based methods on out-of-distribution tasks (xkcdcolors-l), as LLM queries can flexibly interpret diverse notions of similarity.
- In submodular maximization, the implicit submodular objective of VGSM outperforms simple similarity-based ranking, particularly benefiting multi-hop reasoning.

## Highlights & Insights
- **Paradigm-level innovation**: Rather than improving LLM reasoning capabilities directly, this work leverages the structure of classical algorithms to regulate how LLMs are used, restricting them to simple judgments they can reliably perform.
- **Small models can outperform large ones**: With the support of algorithmic structure, a 1.7B model can surpass brute-force reasoning by a 32B model.
- **Inheritable theoretical guarantees**: If the LLM oracle is accurate, all theoretical guarantees of the underlying classical algorithm hold naturally; when the oracle is noisy, Robust VAs provide explicit error upper bounds.
- **Deep integration with inference engines**: The query patterns of VAs are particularly well-suited for prefix caching optimizations.

## Limitations & Future Work
- The accuracy of the LLM oracle is a core assumption; VAs degrade when task semantics are ambiguous or small models have insufficient capability.
- Absolute Kendall-Tau scores in sorting experiments remain modest (best: 0.60), indicating that LLM pairwise comparison ability still has room for improvement.
- The pre-selection stage in submodular maximization still relies on embedding models, making the pipeline not a purely VA-based solution.
- Clustering experiments use relatively small datasets (100 elements); efficiency at larger scales remains to be validated.
- More complex classical algorithms (e.g., graph algorithms, dynamic programming) have not been explored for verbalization.

## Related Work & Insights
- The sorting task corresponds to pairwise ranking in the RAG literature, but this work emphasizes the theoretical properties of classical sorting algorithms (e.g., the parallelism of bitonic networks).
- VAs complement formalization-based methods (LLM → PDDL/SMT solver): VAs are more flexible but depend on oracle accuracy, whereas formalization-based methods are more precise but limited by the expressiveness of formal languages.
- The framework offers direct insights for RAG system design: replacing simple top-$k$ similarity retrieval with VGSM can improve multi-hop reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Faster Algorithms for Structured John Ellipsoid Computation](faster_algorithm_for_structured_john_ellipsoid_computation.md)
- [\[NeurIPS 2025\] Adaptive Algorithms with Sharp Convergence Rates for Stochastic Hierarchical Optimization](adaptive_algorithms_with_sharp_convergence_rates_for_stochas.md)
- [\[NeurIPS 2025\] Preference Learning with Response Time: Robust Losses and Guarantees](preference_learning_with_response_time_robust_losses_and_guarantees.md)
- [\[NeurIPS 2025\] Natural Gradient VI: Guarantees for Non-Conjugate Models](natural_gradient_vi_guarantees_for_non-conjugate_models.md)
- [\[NeurIPS 2025\] DynaAct: Large Language Model Reasoning with Dynamic Action Spaces](dynaact_large_language_model_reasoning_with_dynamic_action_spaces.md)

</div>

<!-- RELATED:END -->
