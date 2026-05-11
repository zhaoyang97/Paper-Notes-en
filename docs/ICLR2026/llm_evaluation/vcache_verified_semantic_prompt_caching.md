---
title: >-
  [Paper Note] vCache: Verified Semantic Prompt Caching
description: >-
  [ICLR2026][LLM Evaluation][Semantic Caching] This paper proposes vCache — the first semantic caching system with **user-defined error-rate guarantees** — which employs online learning to independently estimate the optimal similarity threshold for each cached embedding. Without any pre-training, vCache achieves up to a 12.5× improvement in cache hit rate and a 26× reduction in error rate while satisfying correctness constraints.
tags:
  - ICLR2026
  - LLM Evaluation
  - Semantic Caching
  - LLM Inference Optimization
  - Error-Rate Guarantee
  - online learning
  - Per-Embedding Threshold
date: 2026-05-08
content_hash: 4d71a736b64db3d4
---

# vCache: Verified Semantic Prompt Caching

**Conference**: ICLR2026
**arXiv**: [2502.03771](https://arxiv.org/abs/2502.03771)
**Code**: [GitHub](https://github.com/vcache-project/vCache) | [Benchmarks](https://huggingface.co/vCache)
**Area**: LLM Evaluation
**Keywords**: Semantic Caching, LLM Inference Optimization, Error-Rate Guarantee, online learning, Per-Embedding Threshold
**Authors**: Luis Gaspar Schroeder, Aditya Desai, Alejandro Cuadron, Kyle Chu, Shu Liu, Mark Zhao, Stephan Krusche, Alfons Kemper, Matei Zaharia, Joseph E. Gonzalez (UC Berkeley, TU Munich, ETH Zurich, Stanford)

## TL;DR

This paper proposes vCache — the first semantic caching system with **user-defined error-rate guarantees** — which employs online learning to independently estimate the optimal similarity threshold for each cached embedding. Without any pre-training, vCache achieves up to a 12.5× improvement in cache hit rate and a 26× reduction in error rate while satisfying correctness constraints.

## Background & Motivation

**High LLM inference cost**: Each prompt requires multiple forward passes, making deployment expensive and latency-intensive.

**Value of semantic caching**: When a response for "Which city is Canada's capital?" is already cached, a semantically equivalent query such as "加拿大首都是哪里？" should reuse the same response, potentially reducing latency by up to 100×.

**Fundamental limitations of static thresholds**:
   - Existing systems (GPTCache, Azure, AWS, etc.) employ a **globally fixed threshold** $t$ applied uniformly to all prompts.
   - However, experiments (Figure 3) show that the similarity distributions of correct and incorrect cache hits **overlap substantially**.
   - **The optimal threshold varies dramatically across embeddings**, making a single threshold incapable of simultaneously achieving low error rates and high hit rates.

**Absence of error-rate guarantees**: Existing systems cannot guarantee the correctness of returned cached responses, making them difficult to trust in production environments.

**Limitations of embedding fine-tuning**: Fine-tuning requires supervised training, is restricted to open-source models, and generalizes poorly to out-of-distribution data.

## Core Contributions

1. The first verifiable semantic cache providing **user-defined error-rate guarantees**.
2. An **online threshold learning algorithm**: no pre-training required; independently learns a threshold per cached embedding and is agnostic to the choice of embedding model.
3. Empirical and theoretical demonstration that **dynamic per-embedding thresholds** outperform static thresholds and fine-tuned embeddings.
4. Release of four open-source semantic caching benchmarks (LMArena / Classification / SearchQueries / Combo).

## Method

### System Overview

The basic pipeline of semantic caching:
1. Embed prompt $x$ as $\mathcal{E}(x) \in \mathbb{R}^d$.
2. Retrieve the nearest neighbor from the vector store: $\text{nn}(x) = \arg\max_{y \in C} \text{sim}(\mathcal{E}(x), \mathcal{E}(y))$.
3. Compute similarity $s(x) = \text{sim}(\mathcal{E}(x), \mathcal{E}(\text{nn}(x)))$.
4. Decision: if $s(x) \ge t$, **exploit** (return the cached response); otherwise, **explore** (invoke the LLM).

### Data Model

The cache stores triples:

$$\mathcal{D} = \{(\mathcal{E}(x_i), r(x_i), \mathcal{O}(x_i))\}_{i=0}^{n}$$

where the per-embedding metadata $\mathcal{O}(x_i)$ records the similarity scores and match indicators for all subsequent prompts whose nearest neighbor is $x_i$:

$$\mathcal{O}(x_i) = \{(s(x_j), c(x_j)) \mid \text{nn}(x_j) = x_i\}_{j=i+1}^{n}$$

$$c(x) = \begin{cases} 1 & \text{if } r(\text{nn}(x)) = r(x) \\ 0 & \text{otherwise} \end{cases}$$

### User Guarantee

The user specifies a maximum error rate $\delta$, and vCache guarantees:

$$\Pr(\mathbf{vCache}(x) = r(x)) \ge 1 - \delta$$

Decomposing the correctness probability into two mutually exclusive events:

$$\Pr(\mathbf{vCache}(x) = r(x)) = \Pr(\text{explore} \mid x, \mathcal{D}) + (1 - \Pr(\text{explore} \mid x, \mathcal{D})) \cdot \Pr(c(x) = 1 \mid x, \mathcal{D})$$

This yields a **lower bound on the exploration probability**:

$$\Pr(\text{explore} \mid x, \mathcal{D}) \ge \frac{(1-\delta) - \Pr(c(x)=1 \mid x, \mathcal{D})}{1 - \Pr(c(x)=1 \mid x, \mathcal{D})} = \tau_{\text{nn}(x)}(s(x))$$

### Sigmoid Parameterization

For each embedding, a sigmoid function models the relationship between similarity and correctness:

$$\Pr(c(x) = 1 \mid x, \mathcal{D}) = \mathcal{L}(s(x), t, \gamma) = \frac{1}{1 + e^{-\gamma(s(x) - t)}}$$

where $t$ is the embedding-specific decision boundary and $\gamma$ controls the steepness. Parameters are estimated via **maximum likelihood** (binary cross-entropy loss):

$$\hat{t}, \hat{\gamma} = \arg\min_{t, \gamma} \mathbb{E}_{(s,c) \in \mathcal{O}_{\text{nn}(x)}} \left[ c \cdot \log(\mathcal{L}(s, t, \gamma)) + (1-c) \cdot \log(1 - \mathcal{L}(s, t, \gamma)) \right]$$

### Confidence Interval Calibration

Due to finite samples, directly using $\hat{t}, \hat{\gamma}$ cannot guarantee exactness. vCache adopts a **pessimistic estimate** — a conservative value $t'(\epsilon), \gamma'(\epsilon)$ from the $(1-\epsilon)$ confidence band — to compute an upper bound on the exploration probability:

$$\hat{\tau} = \min_{\epsilon \in (0,1)} \frac{(1-\delta) - (1-\epsilon)\mathcal{L}(s(x), t'(\epsilon), \gamma'(\epsilon))}{1 - (1-\epsilon)\mathcal{L}(s(x), t'(\epsilon), \gamma'(\epsilon))} \ge \tau_{\text{nn}(x)}(s(x))$$

### Decision Rule (Algorithm 2)

1. Compute the embedding and retrieve the nearest neighbor $y = \text{nn}(x)$.
2. Fit a sigmoid to $\mathcal{O}(y)$ to obtain $\hat{t}, \hat{\gamma}$.
3. Sweep over $\epsilon \in (0,1)$ to compute $\hat{\tau}$.
4. Sample $u \sim \text{Uniform}(0,1)$.
5. If $u \le \hat{\tau}$: **explore** (invoke the LLM and update $\mathcal{O}(y)$).
6. Otherwise: **exploit** (return $r(y)$).

**Theoretical guarantee** (Theorem 4.1): Under the i.i.d. assumption and correct sigmoid specification, for any prompt $x$ and any time step $n$:

$$\Pr(\mathbf{vCache}(x) = r(x) \mid \mathcal{D}) \ge 1 - \delta$$

## Key Experimental Results

### Experimental Setup
- **Embedding models**: GteLargeENv1-5, E5-large-v2, OpenAI text-embedding-3-small
- **LLMs**: Llama-3-8B-Instruct, GPT-4o-mini
- **Vector store**: HNSW with cosine similarity
- **Hardware**: Intel Xeon Platinum 8570 + NVIDIA Blackwell 192 GB

### Benchmark Datasets

| Benchmark | Size | Characteristics |
|-----------|------|-----------------|
| SemCacheLMArena | 60K | Open user prompts from LM-Arena |
| SemCacheClassification | 45K | 3 classification datasets |
| SemCacheSearchQueries | 150K | Web search queries |
| SemCacheCombo | 27.5K | Mixed, including no-hit prompts |

### Main Results

**vs. static threshold (GPTCache)**:
- On SemCacheLMArena: up to **26× lower error rate** and **12.5× higher cache hit rate**.
- On SemCacheClassification: comprehensively outperforms the static baseline when $\delta > 1.5\%$.
- At $\delta < 1.5\%$, vCache is more conservative (prioritizing correctness by increasing exploration).

**Error-rate guarantee validation** (Figure 4):
- vCache **consistently satisfies the error-rate constraint** across all values of $\delta$.
- GPTCache's error rate **continuously increases** with more samples, exposing the weakness of static thresholds.
- vCache's cache hit rate grows steadily over time, reflecting the online learning effect.

**Pareto frontier comparison** (Figure 5):
- On the error rate vs. cache hit rate Pareto plot, vCache **dominates** static threshold approaches across all datasets.
- vCache also outperforms GPTCache with fine-tuned embeddings.

### vs. Fine-tuned Embedding Baseline
- vCache **matches or exceeds** the fine-tuned embedding method of Zhu et al. (2024) **without any training**.
- Fine-tuned embeddings generalize poorly to out-of-distribution data, while vCache's online learning naturally adapts to distribution shifts.

## Highlights & Insights

1. **Formal correctness guarantee**: vCache is the first semantic cache to provide a user-controllable error-rate upper bound $\delta$, addressing the trust barrier for production-level deployment.
2. **Per-embedding adaptive thresholds**: Figure 3 demonstrates that the optimal threshold varies substantially across embeddings, rendering a single global threshold infeasible; vCache naturally captures this variability.
3. **Zero-pretraining online learning**: Model-agnostic, embedding-model-agnostic, and requiring no labeled data — vCache learns continuously during inference.
4. **Elegant probabilistic framework**: The explore/exploit decision is formalized as an optimization problem under a correctness-probability constraint.
5. **Unified theory and practice**: Combines the theoretical guarantee of Theorem 4.1 with large-scale benchmark validation.

## Limitations & Future Work

1. **Long responses require LLM-as-a-judge for equivalence checking** (Algorithm 1, L8), introducing additional LLM calls (though these can be executed asynchronously without impacting latency).
2. **Reliance on the i.i.d. assumption**: If the prompt distribution shifts abruptly, the guarantee may be temporarily violated.
3. **Sigmoid modeling assumption**: If the true correctness probability does not follow a sigmoid relationship with similarity, the theoretical analysis may not hold.
4. **Cold-start problem**: For newly added embeddings with no historical observations, vCache defaults to a conservative strategy (full exploration).
5. **Applicable only to semantically similar scenarios**: Semantic caching is inherently unsuitable for prompts requiring precise reasoning (e.g., mathematical computation).
6. **Multi-turn dialogue caching not evaluated** (the paper focuses on single-turn prompts).

## Related Work & Insights

| Method | Threshold Type | Error-Rate Guarantee | Training Required | Model-Agnostic | Online Learning |
|--------|---------------|:--------------------:|:-----------------:|:--------------:|:---------------:|
| GPTCache | Global static | ✗ | ✗ | ✓ | ✗ |
| Azure/AWS Cache | Global static | ✗ | ✗ | ✓ | ✗ |
| Zhu et al. 2024 | Global static | ✗ | ✓ | ✗ | ✗ |
| SCalM | Global static | ✗ | ✗ | ✓ | ✗ |
| EVM (Rudd et al.) | Per-class | ✗ | ✓ | — | ✗ |
| **vCache** | **Per-embedding dynamic** | **✓** | **✗** | **✓** | **✓** |

**Inspirations and connections**:
- **Semantic caching + RAG**: vCache can be directly integrated into RAG systems to cache retrieval results for high-frequency queries.
- **Connection to conformal prediction**: vCache's probabilistic guarantee framework is conceptually analogous to conformal prediction, but requires no labeled calibration set.
- **Online learning paradigm**: Combining the explore/exploit framework with correctness guarantees is a generalizable design pattern applicable to recommendation systems and A/B testing.
- **Multimodal extension**: The semantic caching principle extends naturally to image/video prompts, and vCache's per-embedding threshold mechanism is directly applicable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The first semantic cache with correctness guarantees; per-embedding online threshold learning is an entirely new direction.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across 3 embedding models × 2 LLMs × 4 benchmarks, though real-deployment latency comparisons are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Formally rigorous, clearly motivated, with outstanding figure design (Figures 1–3 are particularly intuitive).
- Value: ⭐⭐⭐⭐⭐ — Addresses the core trust problem in semantic caching with an elegant theoretical framework and strong practical utility; fully open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](spectral_attention_steering_for_prompt_highlighting.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Models](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[ACL 2026\] PIArena: A Platform for Prompt Injection Evaluation](../../ACL2026/llm_evaluation/piarena_a_platform_for_prompt_injection_evaluation.md)
- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](prompt_and_parameter_co-optimization_for_large_language_models.md)
- [\[ICCV 2025\] Supercharging Floorplan Localization with Semantic Rays](../../ICCV2025/llm_evaluation/supercharging_floorplan_localization_with_semantic_rays.md)

</div>

<!-- RELATED:END -->
