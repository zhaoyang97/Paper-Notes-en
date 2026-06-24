---
title: >-
  [Paper Note] Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective
description: >-
  [ICLR 2026][LLM Efficiency][KV Cache Eviction Policy] The authors propose the first unified mathematical model for KV cache-aware load balancing, designing the Randomized Leaf-node Token eviction algorithm (RLT) with an $O(\log n)$ competitive ratio and the Learning-Based Greedy Routing (LBGR). This approach reduces latency by up to 11.96× and TTFT by 14.06× in multi-LLM serving scenarios.
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "KV Cache Eviction Policy"
  - "Randomized Algorithms"
  - "Load Balancing Routing"
  - "Multi-LLM Serving"
  - "Competitive Analysis"
date: 2026-05-08
content_hash: 1e5ee3b629e29a54
---

# Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective

**Conference**: ICLR 2026  
**arXiv**: [2601.18999](https://arxiv.org/abs/2601.18999)  
**Code**: [GitHub](https://github.com/fzwark/KVRouting)  
**Area**: LLM Serving / KV Cache / Load Balancing  
**Keywords**: KV Cache Eviction Policy, Randomized Algorithms, Load Balancing Routing, Multi-LLM Serving, Competitive Analysis  

## TL;DR
The authors propose the first unified mathematical model for KV cache-aware load balancing, designing the Randomized Leaf-node Token eviction algorithm (RLT) with an $O(\log n)$ competitive ratio and the Learning-Based Greedy Routing (LBGR). This approach reduces latency by up to 11.96× and TTFT by 14.06× in multi-LLM serving scenarios.

## Background & Motivation
**Background**: KV caching is a core technology for LLM inference acceleration, avoiding redundant computation by reusing key-value pairs. However, its effectiveness under memory constraints depends heavily on eviction policies.

**Limitations of Prior Work**: 
- **LRU Flaws**: Traditional Least Recently Used (LRU) policies are fragile under dynamic query patterns. In the worst case, they evict tokens required for the next query, causing cache hit rates to plummet.
- **Key Challenge**: Maximizing individual LLM cache hit rates (routing similar queries to the same worker) conflicts with global load balancing (avoiding queue delays).
- **Heuristic Reliance**: Existing systems like SGLang use fixed thresholds, and NVIDIA/llm-d uses static linear scoring, both lacking formal modeling to adapt to dynamic patterns.
- **Theoretical Gap**: There is a significant disconnect between system design and theoretical understanding, with no unified model capturing the coupling between cache eviction and load balancing.
- **Coarse Modeling**: Current methods often use the number of pending queries to measure congestion, ignoring actual service time variations between queries.

## Method

### Overall Architecture
**Mechanism**: The paper integrates "cache eviction" and "query routing" into a single mathematical model: $M$ workers (LLMs) each possess a KV cache of size $B_i$, processing an online query sequence $Q=\{q_j\}$. Each query is assigned to a worker, subsequently altering that worker's cache state and queue load. The objective is to minimize the makespan (maximum cumulative load). This unified model reveals that the Leaf-node LRU (L-LRU) used in SGLang has an $O(n)$ worst-case competitive ratio. Consequently, two coupled components are designed: RLT for intra-worker eviction to achieve an $O(\log n)$ ratio, and LBGR for inter-worker routing using greedy load estimation.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    Q["Online Query Sequence q_j"] --> MODEL["Unified Cost Model<br/>Service Time Cost_ij + Queue Load P_i<br/>Goal: Minimize Makespan"]
    MODEL -->|Reveals L-LRU competitive ratio degrades to O(n)| RLT["RLT Randomized Eviction<br/>Marking set size reaches B_i+1 -> Clear<br/>Unmarked leaf nodes: Uniform Random Eviction -> O(log n)"]
    RLT -->|Robust Worker Cache Hits| LBGR["LBGR Learning Greedy Routing<br/>Estimate Ê_ij = Cost + Decayed Load + Residual<br/>Select Worker with Min Predicted Latency"]
    LBGR --> OUT["Assign Query → Service → Update Radix Tree<br/>Approximate Min Makespan / E2E Latency"]
    OUT -.Observe Real Latency E_ij.-> UPD["Update Regression θ_i Online<br/>+ Exponential Load Decay ρ"]
    UPD -.Continuous Calibration.-> LBGR
```

### Key Designs

**1. Unified Formal Model: Coupling Eviction and Routing**
**Design Motivation**: Prior systems struggle with the trade-off between grouping similar queries for hits and spreading them for balancing. The paper formalizes service time as a weighted sum of hit and miss portions: $\text{Cost}_{ij} = \alpha_{\text{CACHED}}\cdot h_{ij} + \alpha_{\text{MISS}}\cdot(|q_j| - h_{ij}) + O_{ij}$, where $h_{ij}$ is the number of hit tokens. Routing decisions $x_{ij} \in \{0, 1\}$ update the cache state and queue load $P_i$. This formalization quantifies L-LRU's fragility ($O(n)$ competitive ratio) under unbalanced query lengths.

**2. RLT (Randomized Leaf-node Token Eviction)**
**Novelty**: Deterministic LRU is vulnerable to adversarial arrival orders. Inspired by marking algorithms in online theory, RLT introduces randomness. It maintains a marking set $T$ for accessed tokens, clearing it when it reaches $B_i+1$. When the cache is full, a token is selected for eviction **uniformly at random** from the **unmarked leaf nodes**. This randomness prevents adversarial exploitation, reducing the competitive ratio to $\Theta(\log(B_i - L))$, which is an exponential improvement over $O(n)$ and theoretically optimal.

**3. LBGR (Learning-Based Greedy Routing)**
**Function**: LBGR estimates the expected latency for worker $i$ as $\hat{E}_{ij} = \text{Cost}_{ij} + \tilde{P}_i + \theta_i^{\top}\varphi_{ij}$ and routes to the worker with the minimum value. 
- $\text{Cost}_{ij}$: Tracks cache hits via a global Radix Tree.
- $\tilde{P}_i$: Simulates load decay using $\tilde{P}_i \leftarrow \rho\cdot\tilde{P}_i$ to account for service completion.
- $\theta_i^{\top}\varphi_{ij}$: An online linear regression residual term that captures environmental fluctuations. Parameters are updated online using $(E_{ij} - \hat{E}_{ij})^2$ to adapt to workload shifts.

## Key Experimental Results

### Main Results

| Method | Median Latency Gain | Median TTFT Gain | Cache Hit Rate Gain | Throughput Gain |
|---------|------------|------------|-------------|---------|
| vs Random+LRU | 30.9× | 44.49× | - | - |
| vs Cache-Aware+LRU (SOTA) | 11.96× | 14.06× | 36.45% | 36.51% |
| vs Cache-Aware+LRU (P95) | 2.03× | 2.62× | - | - |

### Ablation Study

| Method | P50 Latency (ms) | P50 TTFT (ms) | Hit Rate | Overhead |
|------|-----------|------------|-------|---------|
| Cache-Aware+LRU (Baseline) | 26680 | 25022 | 23.89% | - |
| Cache-Aware+RLT | 19191 | 14332 | 26.36% | +0.58ms (Eviction) |
| LBGR+LRU | 6025 | 2958 | 33.33% | +0.56ms (Routing) |
| LBGR+RLT | 2263 | 1088 | 37.31% | +2ms (Total) |

### Key Findings
- **Generalization**: Effective across Llama-3.1-8B/70B and Mixtral-8×7B (MoE) architectures and various GPU hardware (L40, H200).
- **Robustness**: In worst-case round-robin arrival sequences, LBGR+RLT maintains a massive lead (22.8× latency reduction).
- **Scalability**: Performance gains over baselines **increase** as cache capacity grows.
- **Single-Worker Efficiency**: RLT alone improves hit rates by up to 6.92× and throughput by 77.4% compared to L-LRU.
- **Efficiency**: Total runtime overhead is ~2ms per query, making it highly practical for production.

## Highlights & Insights
- Sets a theoretical foundation for KV cache load balancing, closing the gap between theory and system practice.
- RLT's $O(\log n)$ competitive ratio is information-theoretically optimal.
- The "Analytical Cost + Decaying Load + Online Regression" combination in LBGR avoids heavy RL or prediction models while remaining adaptive.
- Demonstrates that theoretical insights (randomization) can solve practical system bottlenecks (LRU fragility).

## Limitations & Future Work
- **Modality**: Evaluated only on text; multi-modal KV caches (e.g., image tokens) may exhibit different characteristics.
- **Scale**: Focused on single-domain deployment; wide-area network latency in geo-distributed scenarios is not explored.
- **Theoretical Bounds**: While $O(\log n)$ is optimal for adversarial input, specific workload distributions (e.g., Zipf) might allow for tighter bounds.
- **Generation Length**: Generation tests were limited to 128 tokens; behavior in extremely long-context scenarios (4K+) needs verification.

## Related Work & Insights
- **KV Compression**: Works like H2O and StreamingLLM compress at the model level; this work optimizes the system-level eviction policy.
- **Serving Systems**: Builds upon SGLang's RadixAttention, replacing the heuristic routing and eviction modules.
- **Online Algorithms**: RLT is a non-trivial application of the classic marking algorithm (Fiat et al., 1991) to Radix tree structures.

## Rating ⭐⭐⭐⭐⭐
- **Novelty**: 5/5
- **Experimental Thoroughness**: 5/5
- **Writing Quality**: 5/5
- **Value**: 5/5

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] QuoKA: Query-Oriented KV Selection for Efficient LLM Prefill](quoka_query-oriented_kv_selection_for_efficient_llm_prefill.md)
- [\[ICLR 2026\] FlashDLM: Accelerating Diffusion Language Model Inference via Efficient KV Caching and Guided Diffusion](flashdlm_accelerating_diffusion_language_model_inference_via_efficient_kv_cachin.md)
- [\[ICML 2026\] CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](../../ICML2026/llm_efficiency/criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)
- [\[ICLR 2026\] Frayed RoPE and Long Inputs: A Geometric Perspective](frayed_rope_and_long_inputs_a_geometric_perspective.md)
- [\[ICLR 2026\] Revisiting Long-context Modeling from Context Denoising Perspective](revisiting_long-context_modeling_from_context_denoising_perspective.md)

</div>

<!-- RELATED:END -->
