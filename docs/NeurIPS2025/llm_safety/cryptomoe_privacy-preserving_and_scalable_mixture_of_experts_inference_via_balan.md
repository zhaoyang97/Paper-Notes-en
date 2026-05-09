---
title: >-
  [Paper Note] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing
description: >-
  [NeurIPS 2025][LLM Safety][Privacy-preserving inference] CryptoMoE is the first framework supporting privacy-preserving inference for MoE-based LLMs. By combining balanced expert routing to conceal routing information, a confidence-aware dispatch protocol, and a batch ciphertext matrix multiplication protocol, it achieves 2.8–3.5× latency reduction and 2.9–4.3× communication reduction compared to a dense baseline, with only 0.8% accuracy loss.
tags:
  - NeurIPS 2025
  - LLM Safety
  - Privacy-preserving inference
  - MoE
  - Homomorphic Encryption
  - Secure Multi-Party Computation
  - Expert Routing
date: 2026-05-08
content_hash: 449ad53fccefcf2a
---

# CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing

**Conference**: NeurIPS 2025
**arXiv**: [2511.01197](https://arxiv.org/abs/2511.01197)
**Code**: [https://github.com/PKU-SEC-Lab/CryptoMoE](https://github.com/PKU-SEC-Lab/CryptoMoE)
**Area**: AI Security
**Keywords**: Privacy-preserving inference, MoE, Homomorphic Encryption, Secure Multi-Party Computation, Expert Routing

## TL;DR
CryptoMoE is the first framework supporting privacy-preserving inference for MoE-based LLMs. By combining balanced expert routing to conceal routing information, a confidence-aware dispatch protocol, and a batch ciphertext matrix multiplication protocol, it achieves 2.8–3.5× latency reduction and 2.9–4.3× communication reduction compared to a dense baseline, with only 0.8% accuracy loss.

## Background & Motivation

**Background**: MoE architectures have been widely adopted by mainstream LLMs such as LLaMA-4, DeepSeek-V3, and Qwen-3, enabling large-capacity models through sparse activation. Meanwhile, HE/MPC-based privacy-preserving inference frameworks already support dense models such as GPT-2 and LLaMA-1.

**Limitations of Prior Work**: Existing privacy-preserving inference frameworks (e.g., BOLT, Bumblebee) only support dense architectures and cannot handle the dynamic routing mechanism of MoE layers. Expert selection in MoE is highly input-dependent—mathematical reasoning tasks and language understanding tasks activate noticeably different experts—so exposing routing information is equivalent to leaking the input type.

**Key Challenge**: The most straightforward approach to protecting routing privacy is to route all tokens through all experts (dense baseline), but this amplifies computation by 8–15×, completely negating the efficiency advantage of MoE.

**Goal**: While preserving routing privacy, the paper aims to (a) avoid prohibitive computational overhead; (b) design secure token dispatch and aggregation protocols; (c) maintain accuracy close to the original model.

**Key Insight**: Each expert processes a fixed number $t$ of tokens (balanced routing), making expert load input-independent and thereby concealing routing information.

**Core Idea**: Protect privacy via inference-time balanced expert routing with confidence-prioritized selection, and eliminate the HE rotation bottleneck via batch ciphertext matrix multiplication.

## Method

### Overall Architecture
Privacy-preserving inference over MoE layers proceeds in four steps: ❶ Gate Routing (secure softmax + top-k) → ❷ Secure Dispatch (assign tokens to experts) → ❸ Expert Compute (ciphertext linear layers) → ❹ Secure Combine (aggregate expert outputs). CryptoMoE's core contributions lie in steps ❷❸❹.

### Key Designs

1. **Inference-Time Balanced Expert Routing**:

    - Function: Each expert processes a fixed $t$ tokens; slots are zero-padded if insufficient and tokens are dropped if exceeding capacity.
    - Mechanism: Set $t = mk/n$ (where $m$ is the number of tokens, $k$ is the number of activated experts per token, and $n$ is the total number of experts), yielding the same FLOPs as the original MoE. Since routing is naturally imbalanced, $t = 2mk/n$ is used in practice for a better accuracy–efficiency trade-off.
    - Design Motivation: Fixed expert load prevents the server from inferring input type from computation patterns, thereby achieving routing privacy.

2. **Confidence-Aware Secure Dispatch Protocol**:

    - Function: Under a fixed capacity constraint, each expert preferentially retains the $t$ tokens with the highest routing confidence.
    - Mechanism: Three steps—❶ Compute a boolean mask $[[M_i]]$ for each token–expert pair using $\Pi_{\text{equal}}$; ❷ Combine with routing weights via $\Pi_{\text{mux}}$ to obtain priority scores $[[S_i]]$, then select top-$t$ using $\Pi_{\text{topk}}$; ❸ Retrieve corresponding token embeddings using $\Pi_{\text{onehot}}$ + $\Pi_{\text{matmul}}$.
    - Communication complexity is reduced from $O(kmtd)$ (as in CipherPrune) to $O(km\log(km))$ by decoupling scoring from embedding retrieval.

3. **Secure Combine Protocol**:

    - Function: Reorder and weighted-aggregate expert outputs according to the original token sequence.
    - Mechanism: Reuse the one-hot matrix from the dispatch phase transposed for $\Pi_{\text{matmul}}$, simultaneously multiplied by routing weights. Only one $\Pi_{\text{mul}}$ and one $\Pi_{\text{matmul}}$ are required.
    - The total communication overhead of dispatch + combine accounts for only ~18% of the MoE layer.

4. **Batch Ciphertext Matrix Multiplication (Batch MatMul)**:

    - Function: Pack token embeddings from $n$ experts into a single ciphertext.
    - Mechanism: The naive approach independently packs each expert's $t \times d_1$ embeddings, requiring many rotations when the hidden dimension is large. Batch MatMul packs all experts' local embeddings as $(nt \times d_1/n)$, reducing HE rotations from $O(nd_1)$ to $O(d_1)$.
    - Effect: Expert linear layer computation is accelerated by 3–6×; end-to-end latency is reduced by 2–3×.

### Loss & Training
CryptoMoE is an inference framework and involves no additional training. It directly uses pretrained DeepSeekMoE, OLMoE, and QWenMoE models.

## Key Experimental Results

### Main Results (End-to-end performance, LAN setting, $t=2.0$)

| Model | Method | Avg Acc (%) | LAN Latency (min/tok) | Communication (GB) |
|---|---|---|---|---|
| DeepSeekMoE-16.4B | Dense baseline | 62.2 | 2.33 | 9.16 |
| | CipherPrune | 59.8 | 1.14 | 5.55 |
| | **CryptoMoE** | **61.8** (−0.4) | **0.76** (3.1×) | **2.46** (3.7×) |
| OLMoE-6.9B | Dense baseline | 63.0 | 0.99 | 3.82 |
| | **CryptoMoE** | **62.5** (−0.5) | **0.36** (2.8×) | **1.31** (2.9×) |
| QWenMoE-14.3B | Dense baseline | 62.0 | 1.98 | 7.61 |
| | **CryptoMoE** | **62.0** (−0.0) | **0.56** (3.5×) | **1.76** (4.3×) |

### Ablation Study (LAN latency, DeepSeekMoE)

| Configuration | Acc (%) | Latency (min/tok) |
|---|---|---|
| Dense Baseline | 62.2 | 2.33 |
| + Balanced Expert Routing | 57.9 | 1.20 |
| + Confidence-aware selection | 61.8 | 1.20 |
| + Batch MatMul | 61.8 | **0.76** |

### Key Findings
- Confidence-aware selection is critical for preserving accuracy: without it, accuracy drops sharply from 62.2% to 57.9% (−4.3%), and recovers to 61.8% upon its inclusion.
- Batch MatMul does not affect accuracy but halves latency (1.20→0.76 min/tok).
- Under certain configurations, CryptoMoE is even faster than the insecure baseline, as the insecure baseline cannot exploit the batch packing optimization.
- Scaling to Mixtral-47B and LLaMA4-Scout-109B still maintains 98.8%–100% accuracy.

## Highlights & Insights
- The **privacy protection mechanism via balanced routing** is remarkably elegant: by fixing the number of tokens each expert processes, routing information cannot be leaked through computational side channels. This idea generalizes to any privacy-preserving inference scenario involving dynamic branching.
- The **dispatch protocol that decouples token scoring from embedding retrieval** is a sophisticated design that reduces communication complexity from $O(kmtd)$ to $O(km\log(km))$, representing a general "select index first, then fetch data" paradigm.
- Batch MatMul exploits the structural parallelism of MoE experts for ciphertext packing, achieving an $n$-fold reduction in rotation operations.

## Limitations & Future Work
- Under WAN settings, $\Pi_{\text{topk}}$ becomes a severe bottleneck due to excessive communication rounds, necessitating more round-efficient protocols.
- Performance degrades for short sequences (<64 tokens), as balanced routing cannot effectively balance load with very few tokens.
- Memory overhead is substantial; privacy-preserving inference for Mixtral-47B and LLaMA4-109B cannot be completed on a single machine.
- Evaluation is limited to zero-shot inference tasks; the autoregressive decoding scenario has not been validated.

## Related Work & Insights
- **vs. CipherPrune (ICLR'25)**: Directly applying its token pruning protocol to MoE leaks the token count per expert, and its communication overhead is 4.3× larger.
- **vs. BOLT/Bumblebee**: These support only dense transformers; CryptoMoE is the first to extend privacy-preserving inference to MoE architectures.
- **vs. Insecure baseline**: The fully exposed routing information serves as an upper bound; CryptoMoE is in some cases more efficient than this baseline.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First privacy-preserving inference framework for MoE; both the problem formulation and solution are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three models, eight tasks, two network environments, detailed ablation studies, and scalability analysis.
- Writing Quality: ⭐⭐⭐⭐ Protocol descriptions are clear and toy example illustrations are intuitive, though the dense notation requires frequent cross-referencing.
- Value: ⭐⭐⭐⭐⭐ Directly addresses practical privacy concerns in MoE model deployment; code is open-sourced.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] FedRW: Efficient Privacy-Preserving Data Reweighting for Enhancing Federated Learning of Language Models](fedrw_efficient_privacy-preserving_data_reweighting_for_enhancing_federated_lear.md)
- [\[AAAI 2026\] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](../../AAAI2026/llm_safety/prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)
- [\[NeurIPS 2025\] MPCache: MPC-Friendly KV Cache Eviction for Efficient Private LLM Inference](mpcache_mpc-friendly_kv_cache_eviction_for_efficient_private_llm_inference.md)
- [\[NeurIPS 2025\] Adaptive LoRA Experts Allocation and Selection for Federated Fine-Tuning](adaptive_lora_experts_allocation_and_selection_for_federated_fine-tuning.md)
- [\[NeurIPS 2025\] InvisibleInk: High-Utility and Low-Cost Text Generation with Differential Privacy](invisibleink_high-utility_and_low-cost_text_generation_with_differential_privacy.md)

<!-- RELATED:END -->
