---
title: >-
  [Paper Note] MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference
description: >-
  [NeurIPS 2025][LLM Safety][Private inference] This paper proposes MPCache, a KV cache eviction framework designed for secure multi-party computation (MPC), combining one-time static eviction with query-aware dynamic selection. Through hierarchical clustering, linearized similarity approximation, and cross-layer index sharing, MPCache achieves up to 2.01× latency reduction and 8.37× communication volume reduction without sacrificing LLM performance.
tags:
  - NeurIPS 2025
  - LLM Safety
  - Private inference
  - secure multi-party computation
  - KV cache eviction
  - LLM efficiency
  - sparse attention
date: 2026-05-08
content_hash: 31f34a9ded4dd32c
---

# MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference

**Conference**: NeurIPS 2025
**arXiv**: [2501.06807](https://arxiv.org/abs/2501.06807)
**Code**: [GitHub](https://github.com/zengwenxuan/MPCache)
**Area**: AI Security
**Keywords**: Private inference, secure multi-party computation, KV cache eviction, LLM efficiency, sparse attention

## TL;DR

This paper proposes MPCache, a KV cache eviction framework designed for secure multi-party computation (MPC), combining one-time static eviction with query-aware dynamic selection. Through hierarchical clustering, linearized similarity approximation, and cross-layer index sharing, MPCache achieves up to 2.01× latency reduction and 8.37× communication volume reduction without sacrificing LLM performance.

## Background & Motivation

In LLM-as-a-Service (MLaaS), users must upload prompts containing sensitive information to the cloud, while service providers are unwilling to disclose model weights. MPC enables inference while protecting both parties' privacy, but faces severe efficiency challenges, **particularly in attention computation over long sequences**.

The authors profile MPC inference of GPT-2 on the Secretflow framework and find that:
- Attention computation dominates both latency and communication overhead in 2PC and 3PC protocols
- Softmax accounts for the majority of attention cost due to complex operations including max, exponentiation, and division
- Softmax cost grows sharply as sequence length increases

Although KV cache eviction and sparse attention have been extensively studied in plaintext inference, **directly applying them to MPC is counterproductive**. As shown in Figure 1(d), directly applying dynamic KV eviction algorithms actually increases communication and latency, as they introduce operations that are costly in MPC: top-k sorting (frequent comparison protocols), cosine similarity (numerous multiplications), and token gathering (requiring indices to be converted to one-hot vectors before multiplying with the KV cache).

The core challenge is therefore: **how to design MPC-friendly KV cache eviction algorithms?**

## Method

### Overall Architecture

MPCache is designed around three key observations: (1) attention maps over long sequences are globally sparse, enabling static eviction of unimportant tokens; (2) attention exhibits token-level locality, supporting efficient dynamic selection via clustering; (3) top-k rankings are similar across adjacent layers, enabling cross-layer index sharing. The overall pipeline comprises one-time static eviction during the prefill phase and query-aware dynamic selection during the decoding phase.

### Key Designs

1. **One-time static KV cache eviction**: During the prefill phase, tokens are partitioned into three categories—IA (important to all tokens, always retained), UIA (unimportant to all tokens, directly discarded), and IC (important to some tokens, dynamically selected). Token importance is measured by accumulating attention scores from the last 20% of prompt tokens; UIA tokens are then evicted after ranking. Experiments show that approximately 60% of tokens in LLaMA-2-7B can be statically evicted without performance degradation. This operation is performed only once, and its cost is amortized over the entire generation process.

2. **MPC-friendly dynamic KV cache selection**: Neighboring tokens are grouped into clusters, elevating selection granularity from the token level to the cluster level. The core challenge is measuring cluster similarity accurately and efficiently.

   **Similarity approximation**: The maximum dot-product upper bound is used instead of mean or cosine similarity. Given query $\mathbf{q}$ and cluster key cache $\mathbf{K}_c$, the upper bound is derived as:
   $$\mathrm{Sim}(\mathbf{q}, \mathbf{K}_c) \leq \sum_{i=0}^{d-1} \max(\mathbf{q}_i \mathbf{r}_i^{\max}, \mathbf{q}_i \mathbf{r}_i^{\min})$$
   where $\mathbf{r}_i^{\max}$ and $\mathbf{r}_i^{\min}$ are the per-dimension extrema of the key cache within the cluster (computed only once).

   **Linearization and reordering**: To eliminate the costly max operation in MPC, the approximation is further linearized as:
   $$\mathrm{Sim}(\mathbf{q}, \mathbf{K}_c) \approx \sum_{i=0}^{d-1} \mathbf{q}_i \cdot (\alpha \mathbf{r}_i^{\max} + (1-\alpha) \mathbf{r}_i^{\min})$$
   where $\alpha=0.6$. After reordering, summation precedes multiplication, reducing the number of multiplications by 2× and eliminating all max operations.

3. **Hierarchical KV cache clustering**: The KV cache is organized into a hierarchical structure, selecting from coarse-grained (large clusters) to fine-grained (small clusters). The coarse-grained level substantially narrows the search space, while the fine-grained level refines selection within already-chosen clusters, balancing accuracy and efficiency.

4. **Cross-layer index sharing strategy**: Exploiting the similarity of attention patterns across adjacent layers (commonality scores exceeding 80%), adjacent layer pairs share the selected token indices, directly halving the overhead of dynamic selection. The first two layers do not participate in sharing due to lower commonality.

### Token Gathering Protocol Optimization

The clustering design directly reduces the overhead of the most expensive MPC operation—the token gathering protocol:
- The number of equivalent protocol rounds decreases from $T$ to $C$ (the number of clusters)
- The bit width of one-hot vectors decreases from $\log T$ to $\log C$
- For example, with $T=1024$ and $C=64$, latency is reduced by 73.5× and communication by 369.8×

## Key Experimental Results

### Comparison with Dynamic Methods (LongChat-7B, LongBench)

| Dataset | Cache Budget | InfLLM Perf./Latency | LongCache Perf./Latency | MPCache Perf./Latency |
|---------|-------------|----------------------|------------------------|-----------------------|
| HotpotQA | 5% | 28.20/51.64s | 24.31/89.46s | **30.27/39.85s** |
| TriviaQA | 5% | 75.65/51.64s | 59.85/89.46s | **75.61/37.37s** |
| NarrQA | 5% | 12.80/47.74s | 14.65/86.42s | **17.23/36.13s** |
| PassageR | 10% | 8.87/68.04s | 24.92/123.1s | 27.75/58.47s |

### Latency and Communication Improvements

| Sequence Length | Decoding Latency Reduction | Communication Reduction |
|----------------|--------------------------|------------------------|
| Short sequences | 1.8× | 3.39× |
| Long sequences | 2.01× | **8.37×** |

### Extension to LLaMA-3.1-8B-Instruct

| Method | NarrQA | Qasper | HotpotQA | TriviaQA | Avg. |
|--------|--------|--------|----------|----------|------|
| Full Cache | 30.21 | 45.52 | 55.53 | 91.74 | — |
| StreamingLLM (2048) | 27.40 | 30.77 | 49.23 | 90.98 | Large gap |
| SnapKV (2048) | 28.53 | 39.13 | 54.32 | 92.04 | Moderate |
| MPCache (2048) | **30.17** | **46.11** | **55.21** | **91.53** | Near Full |

### XSUM Summarization Task (ROUGE-L)

| Budget | Model | Full Cache | H2O | MPCache |
|--------|-------|-----------|-----|---------|
| 10% | 7B | 11.90 | 10.50 | **11.10** |
| 5% | 7B | 11.90 | 4.886 | **10.08** |
| 10% | 13B | 13.60 | 13.24 | **13.44** |
| 5% | 13B | 13.60 | 9.081 | **13.08** |

### Key Findings

- MPCache consistently outperforms existing KV eviction methods across all benchmark tasks, with particularly pronounced advantages at aggressive compression rates (5% budget)
- The linearized approximation ($\alpha=0.6$) eliminates max operations with negligible performance loss
- The clustering strategy reduces token gathering protocol latency and communication by 73.5× and 369.8×, respectively
- Cross-layer index sharing has minimal impact on performance (commonality > 80% when sharing across two layers)

## Highlights & Insights

- **Precise problem formulation**: Rather than naively porting plaintext KV eviction to MPC, the authors rigorously analyze the cost of each operation in MPC, identify three MPC-unfriendly operations, and address them individually
- **From "cryptographic protocols" to "algorithmic optimization"**: Without modifying the underlying MPC protocols, the work achieves efficiency improvements at the algorithm level, ensuring broad compatibility
- **Hierarchical design philosophy**: Clustering, hierarchical selection, and cross-layer sharing work in concert to form a systematic efficiency optimization scheme

## Limitations & Future Work

- The 60% threshold for static eviction may require tuning across different tasks and models
- The choice of $\alpha=0.6$ in the linearized approximation is empirically determined, lacking an adaptive mechanism
- Evaluation is limited to GPT-2 and the LLaMA family; Transformer variants (e.g., MoE models) remain untested
- The security analysis relies on the security of existing protocols and does not consider malicious adversary models

## Related Work & Insights

- Compared to MPC protocol optimization works (PUMA, Bolt), MPCache operates at the algorithm level; the two approaches are complementary
- Compared to static eviction methods such as H2O and StreamingLLM, MPCache's hybrid static-plus-dynamic strategy achieves a better accuracy–efficiency trade-off in the MPC setting
- Insight: MPC-friendly algorithm design should be established as an independent research direction within privacy-preserving AI

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The idea of adapting KV cache eviction specifically for MPC is novel, and the clustering-based approximation design is elegant
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multiple models (7B/8B/13B), multiple tasks (QA/summarization/Needle-in-a-Haystack), and comprehensive efficiency measurements
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear motivation, observation-driven design rationale, and information-rich figures and tables
- **Value**: ⭐⭐⭐⭐ — Significant practical implications for private LLM inference deployment, though scope is constrained by the applicability of MPC

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[NeurIPS 2025\] FedSVD: Adaptive Orthogonalization for Private Federated Learning with LoRA](fedsvd_adaptive_orthogonalization_for_private_federated_learning_with_lora.md)

<!-- RELATED:END -->
