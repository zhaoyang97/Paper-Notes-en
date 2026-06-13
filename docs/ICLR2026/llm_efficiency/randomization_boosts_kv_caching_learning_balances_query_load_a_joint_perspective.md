---
title: >-
  [Paper Note] Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective
description: >-
  [ICLR 2026][LLM Efficiency][KV cache eviction policy] This paper proposes the first unified mathematical model for KV cache-aware load balancing…
tags:
  - "ICLR 2026"
  - "LLM Efficiency"
  - "KV cache eviction policy"
  - "randomized algorithms"
  - "load-balanced routing"
  - "multi-LLM serving"
  - "competitive ratio analysis"
date: 2026-05-08
content_hash: a1ead00f439af961
---

# Randomization Boosts KV Caching, Learning Balances Query Load: A Joint Perspective

**Conference**: ICLR 2026
**arXiv**: [2601.18999](https://arxiv.org/abs/2601.18999)  
**Code**: [GitHub](https://github.com/fzwark/KVRouting)  
**Area**: LLM Serving / KV Cache / Load Balancing
**Keywords**: KV cache eviction policy, randomized algorithms, load-balanced routing, multi-LLM serving, competitive ratio analysis

## TL;DR
This paper proposes the first unified mathematical model for KV cache-aware load balancing, introducing a randomized leaf-node eviction algorithm RLT (with $O(\log n)$ competitive ratio) and a learning-based greedy router LBGR, achieving up to 11.96× latency reduction and 14.06× TTFT reduction in multi-LLM serving scenarios.

## Background & Motivation
**KV caching is central to LLM inference acceleration**: by reusing key-value pairs from prior queries to avoid redundant computation, yet its effectiveness under memory constraints is highly sensitive to the choice of eviction policy.

**LRU eviction has fundamental flaws**: the traditional Least Recently Used (LRU) policy is fragile under dynamic query arrival patterns — in the worst case, it evicts precisely the tokens needed by the next query, causing a sharp drop in cache hit rate.

**Inherent tension in multi-LLM serving**: maximizing per-LLM cache hit rate (routing similar queries to the same LLM) and achieving global load balance (avoiding queue delays) are conflicting objectives.

**Existing methods rely on heuristics**: SGLang uses a fixed threshold to switch routing strategies; NVIDIA/llm-d uses a static linear scoring function. Neither approach offers formal modeling or adaptability to dynamic query patterns.

**Lack of theoretical analysis**: a significant gap exists between practical system design and theoretical understanding, with no unified model capturing the coupling between cache eviction and load balancing.

**Coarse-grained queue load modeling**: existing methods measure congestion solely by the number of pending queries, ignoring variation in actual service time across queries.

## Method

### Unified Formal Model
- $M$ workers (LLMs), each equipped with a KV cache of size $B_i$, process $N$ queries $Q = \{q_j\}$ online.
- Service time: $\text{Cost}_{ij} = \alpha_{\text{CACHED}} \cdot h_{ij} + \alpha_{\text{MISS}} \cdot (|q_j| - h_{ij}) + O_{ij}$, where $h_{ij}$ is the number of cache-hit tokens.
- Routing decision $x_{ij} \in \{0, 1\}$ determines query assignment; the objective is to minimize makespan (maximum cumulative load across all workers).
- Cache state is updated via an UpdateCache function; queue load $P_i$ accumulates by service time.
- The model formally exposes the theoretical limitation of LRU: L-LRU (SGLang's leaf-node LRU) has a competitive ratio of $O(n)$, degenerating severely when query lengths are highly imbalanced.

### RLT: Randomized Leaf-node evicTion
- A marking set $T$ tracks visited tokens; when the mark count reaches $B_i + 1$, marks are cleared (retaining only the most recent token).
- When the cache is full and new tokens must be loaded, a leaf node is chosen **uniformly at random** from **unmarked leaf nodes** for eviction.
- **Theoretical guarantee**: competitive ratio $\Theta(\log(B_i - L))$, an exponential improvement over L-LRU's $O(n)$.
- This is proven to match the optimal lower bound for all randomized eviction algorithms (information-theoretically optimal).

### LBGR: Learning-Based Greedy Routing
- **Service time estimation**: a global Radix Tree tracks per-worker cache state to estimate cache-hit token count $\tilde{h}_{ij}$.
- **Queue load estimation**: $\tilde{P}_i$ tracks per-worker load using exponential decay (factor $\rho$) to simulate natural load reduction over time; residual load is released upon query completion.
- **Residual correction**: an online linear regression model $\theta_i$ captures latency deviations caused by environmental fluctuations, with feature vector $\phi$ containing hit count, miss count, and current load.
- **Greedy decision**: $\hat{E}_{ij} = \text{Cost}_{ij} + \tilde{P}_i + \theta_i^T \cdot \phi_{ij}$; queries are routed to the worker with the lowest predicted latency.
- **Online update**: the regression model is continuously updated by minimizing $(E_{ij} - \hat{E}_{ij})^2$ upon observing actual latency.
- **Background decay thread**: every $\Delta t$ time units, all workers execute $\tilde{P}_i \leftarrow \rho \cdot \tilde{P}_i$ to prevent accumulated distortion in load estimates.

## Experiments

### Main Results

| Baseline | Median Latency Reduction | Median TTFT Reduction | Cache Hit Rate Gain | Throughput Gain |
|---|---|---|---|---|
| vs Random+LRU | 30.9× | 44.49× | — | — |
| vs Cache-Aware+LRU (SOTA) | 11.96× | 14.06× | 36.45% | 36.51% |
| vs Cache-Aware+LRU (P95) | 2.03× | 2.62× | — | — |

### Ablation Study

| Method | P50 Latency (ms) | P50 TTFT (ms) | Hit Rate | Overhead |
|---|---|---|---|---|
| Cache-Aware+LRU | 26680 | 25022 | 23.89% | baseline |
| Cache-Aware+RLT | 19191 | 14332 | 26.36% | +0.58ms eviction |
| LBGR+LRU | 6025 | 2958 | 33.33% | +0.56ms routing |
| LBGR+RLT | 2263 | 1088 | 37.31% | +2ms total |

### Generalization across Model Scales and Architectures

| Model | Hardware | Latency Reduction vs Cache-Aware+LRU | TTFT Reduction vs Cache-Aware+LRU |
|---|---|---|---|
| Llama-3.1-8B | 10×L40 | 11.96× | 14.06× |
| Llama-3.1-70B | 4×H200 | 5.46× | 7.19× |
| Mixtral-8×7B (MoE) | 4×H200 | significantly outperforms all baselines | significantly outperforms all baselines |

### Key Findings
- Consistent gains across Llama-3.1-8B (L40), 70B (H200), and Mixtral-8×7B, generalizing across dense and sparse architectures.
- Under worst-case round-robin arrival order, LBGR+RLT still leads substantially (22.8× latency reduction, 15.5× TTFT reduction).
- Robust across cache sizes from 50% to 90%, worker counts from 2 to 10, and request rates of 4–20 req/s.
- Single-worker experiments also confirm RLT's superiority over L-LRU: up to 6.92× higher cache hit rate and 77.4% throughput gain.
- The performance gap between LBGR+RLT and baselines **widens further** as cache capacity increases.
- Throughput saturates across methods for worker counts $\geq 6$ under fixed request rate (12 req/s), indicating sufficient system capacity.

## Highlights & Insights
- The paper establishes the first unified mathematical model for KV cache-aware load balancing, bridging the gap between theory and practice.
- RLT's $O(\log n)$ competitive ratio is proven to be information-theoretically optimal (not improvable), representing an elegant application of classical online algorithm theory to LLM systems.
- Both algorithms introduce negligible runtime overhead (~2ms per query), making them highly practical and directly plug-and-play in existing systems.
- LBGR's exponential decay combined with online regression is simple yet effective, avoiding the training cost of complex RL or predictive models.
- Experiments cover 4 benchmarks (GSP/ShareGPT/UltraChat/Loogle), 3 prefix-sharing scenarios, and multiple model scales and architectures, providing comprehensive empirical coverage.

## Limitations & Future Work
- Evaluation is limited to text inference; multi-modal KV caching (where image token cache behavior may differ) is not addressed.
- Deployment is restricted to single-domain setups of up to 10 workers; the impact of network latency in large-scale or geo-distributed deployments remains unvalidated.
- Theoretical analysis assumes online adversarial arrivals; distributional properties of real workloads (e.g., Zipf distributions) could be exploited for tighter bounds.
- The linear regression residual model in LBGR may lack expressiveness in highly non-linear latency scenarios.
- Output token length experiments are limited to 128; longer generation settings (e.g., 4K+) are unexplored.

## Related Work & Insights
- **KV cache compression**: H2O (retaining high-attention tokens), StreamingLLM (attention sink tokens) — model-level compression approaches, complementary to the system-level eviction strategy proposed here.
- **KV cache systems**: vLLM (paged virtual memory to reduce fragmentation), SGLang RadixAttention (Radix tree prefix sharing) — this paper replaces the eviction and routing modules on top of SGLang.
- **Load-balanced routing**: SGLang (threshold-based switching between hit-rate and least-load routing), NVIDIA TensorRT-LLM/llm-d (static linear scoring) — all heuristic-based with no theoretical guarantees.
- **Online algorithm theory**: the classical marking algorithm (Fiat et al., 1991) provides the theoretical foundation for RLT; this paper extends it to leaf-node eviction on Radix tree structures.
- **Competitive ratio analysis**: this work connects online algorithm competitive analysis with LLM system optimization, exemplifying a theory-guided systems research paradigm.
- This is the first work to jointly model cache eviction and routing in a unified framework, serving as a canonical example of theoretically driven system optimization.

## Rating ⭐⭐⭐⭐⭐
- **Novelty**: 5/5 — First unified formalization with optimal competitive ratio proof; solid theoretical contribution.
- **Experimental Thoroughness**: 5/5 — 4 benchmarks × 3 settings × multiple models × extensive parameter sweeps; thorough ablation.
- **Writing Quality**: 5/5 — Theory and experiments are well-integrated; problem motivation is clearly articulated.
- **Value**: 5/5 — Only ~2ms overhead per query; directly integrable into systems such as SGLang.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] CriticalKV: Optimizing KV Cache Eviction from an Output Perturbation Perspective](../../ICML2026/llm_efficiency/criticalkv_optimizing_kv_cache_eviction_from_an_output_perturbation_perspective.md)
- [\[ICLR 2026\] Expert Divergence Learning for MoE-based Language Models](expert_divergence_learning_for_moe-based_language_models.md)
- [\[ICLR 2026\] Deep Hierarchical Learning with Nested Subspace Networks for Large Language Models](deep_hierarchical_learning_with_nested_subspace_networks_for_large_language_mode.md)
- [\[ICLR 2026\] One-Prompt Strikes Back: Sparse Mixture of Experts for Prompt-based Continual Learning](one-prompt_strikes_back_sparse_mixture_of_experts_for_prompt-based_continual_lea.md)
- [\[ACL 2026\] MTRouter: Cost-Aware Multi-Turn LLM Routing with History-Model Joint Embeddings](../../ACL2026/llm_efficiency/mtrouter_cost-aware_multi-turn_llm_routing_with_history-model_joint_embeddings.md)

</div>

<!-- RELATED:END -->
