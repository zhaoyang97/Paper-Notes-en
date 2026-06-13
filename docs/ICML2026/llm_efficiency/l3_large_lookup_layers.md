---
title: >-
  [Paper Note] L$^3$: Large Lookup Layers
description: >-
  [ICML 2026][LLM Efficiency][Sparse Models] This paper proposes L$^3$ (Large Lookup Layer), which generalizes the tokenizer embedding table into a "large lookup layer" that can be inserted into the decoder. It utilizes **…
tags:
  - "ICML 2026"
  - "LLM Efficiency"
  - "Sparse Models"
  - "Static Routing"
  - "Embedding Lookup"
  - "LZW Allocation"
  - "CPU Offload"
date: 2026-05-08
content_hash: e1a92182083dc0c1
---

# L$^3$: Large Lookup Layers

**Conference**: ICML 2026  
**arXiv**: [2601.21461](https://arxiv.org/abs/2601.21461)  
**Code**: TBD  
**Area**: LLM Efficiency / Sparse Architectures  
**Keywords**: Sparse Models, Static Routing, Embedding Lookup, LZW Allocation, CPU Offload

## TL;DR
This paper proposes L$^3$ (Large Lookup Layer), which generalizes the tokenizer embedding table into a "large lookup layer" that can be inserted into the decoder. It utilizes **static routing** based on token IDs to retrieve a set of learned key/value embeddings, which are then aggregated by current hidden states via attention. This increases model sparsity by an additional order of magnitude without the typical MoE issues of dynamic routing, auxiliary losses, and offloading difficulties. It outperforms dense models of comparable compute and MoEs of comparable sparsity across 800M–2.6B active parameters.

## Background & Motivation

**Background**: The current mainstream approach for "parameter sparsity" is Mixture-of-Experts (MoE). In each decoder layer, the MLP is replaced with a router and multiple dense experts. The router assigns tokens to top-k experts based on their current hidden states. This family of methods (GShard, Switch, DeepSeek-MoE, OLMoE, etc.) significantly improves quality under constant compute.

**Limitations of Prior Work**: The dynamic routing of MoE introduces a series of system-level challenges. Load-balancing loss and router z-loss are required to prevent router collapse. Furthermore, since the expert destination for a token is only known at the moment it reaches the router, expert parameters **cannot be effectively prefetched or offloaded** and must reside entirely in GPU memory. Under large batch sizes, all experts are typically hit, rendering offloading ineffective. Extremely large MoEs also require sophisticated sharding to execute.

**Key Challenge**: Researchers seek "parameter sparsity + context-aware aggregation." However, existing methods only provide context-awareness through dynamic routing, which is inherently system-unfriendly. The paper observes that the tokenizer embedding table is an extremely sparse layer (activating one row per token) that is highly system-friendly (static lookup, prefetchable), but it lacks **contextual information**.

**Goal**: To generalize the system-friendly static sparse structure of embedding tables into the middle of the decoder, allowing it to retain the system advantages of static routing while performing "context-aware aggregation" based on hidden states. It also aims to answer: (a) Can this structure outperform dense and MoE models under iso-FLOP conditions? (b) What is the optimal way to allocate embeddings per token?

**Key Insight**: The authors view "static routing by token ID + attention aggregation by hidden state" as a "soft lookup." Since the router remains static (token ID → set of embeddings), the routing is known as soon as a token is generated, allowing L$^3$ parameters to be prefetched from the CPU.

**Core Idea**: Replace MoE's hidden-state routing and dense experts with **token ID static routing and hidden-state attention aggregation**. This architecturally shifts the routing dependency from hidden states back to token IDs, utilizing an LZW-style information-theoretic allocation algorithm to determine embedding distribution per token.

## Method

### Overall Architecture
L$^3$ is a new decoder sublayer inserted between existing dense Llama decoder layers (it does not replace the MLP and is **orthogonal to MoE**). For a single token:

1. Input: Hidden state $x \in \mathbb{R}^{d_\text{in}}$ and token ID $t \in \{1, \dots, |\tau|\}$.
2. Static Routing: Use $t$ to look up and retrieve token-specific $K_t \in \mathbb{R}^{d_t \times d_\text{in}}$ and $V_t \in \mathbb{R}^{d_t \times d_\text{emb}}$ from global tables $W_K \in \mathbb{R}^{v \times d_\text{in}}$ and $W_V \in \mathbb{R}^{v \times d_\text{emb}}$.
3. Contextual Aggregation: Use $x$ as a query to compute softmax scores against $K_t$, then perform a weighted sum of $V_t$ to obtain a "context-aware lookup result."
4. Output: After upward projection $W_\text{up}$ and LayerNorm, the result is concatenated with the residual stream $x$ and passed through a mixing matrix $W_\text{mix}$:  
   $$L^3(x,t) = W_\text{mix}\big[\text{LN}(W_\text{up}(V_t^\top \text{Softmax}(K_t x)))\,;\,x\big]$$

The entire layer performs mixing only in the channel dimension with **no cross-token communication**, which is the basis for subsequent system optimizations. The dimension $d_t$ for each token is determined by the embedding allocation algorithm.

### Key Designs

1. **Static Token Routing + Hidden-state Soft Lookup**:
    - **Function**: Uses token ID to determine which group of embeddings to retrieve (static) and the current hidden state to determine the weighting for aggregation (context-aware), completely decoupling routing from aggregation.
    - **Mechanism**: While MoE routers take the form $r(x,e)$ (dependent on $x$), the L$^3$ "router" is simply $t \mapsto \{K_t, V_t\}$ (dependent only on token ID). Contextual relevance is handled by attention: $\text{Softmax}(K_t x)$ yields $d_t$ scores to weight $V_t$. Because routing is static, the addresses of $\{K_t, V_t\}$ are known the moment a token is generated, allowing for asynchronous CPU→GPU prefetching during the computation of preceding decoder layers (see Figure 4).
    - **Design Motivation**: MoE's system pain points (auxiliary loss, offloading inability, all-expert hits in large batches) stem from the "routing depends on hidden state" bottleneck. Shifting this dependency to the token ID eliminates these issues, while "hidden-state weighted aggregation" prevents the model from degrading into a simple tokenizer embedding.

2. **LZW Information-theoretic Embedding Allocation Algorithm**:
    - **Function**: Given a fixed total budget $v = \sum_i d_i$, it determines the number of embeddings allocated to each token, providing higher capacity to tokens that appear frequently and require more contextual distinction.
    - **Mechanism**: The authors frame "modelling a context router with a static router" as "finding a set of codewords covering common sequence suffixes," which is dual to LZW lossless compression. Algorithm 1 uses LZW to scan a corpus and construct a (codeword, frequency) dictionary. Codewords are traversed in descending frequency, and each assigns an embedding to its final token. Every token is guaranteed at least 1 and at most $k$ (e.g., $k=512$) embeddings, resulting in a near-Zipfian distribution (e.g., "then" receives 512, while "orm" receives 1).
    - **Design Motivation**: Uniform allocation performed significantly worse in ablations (Figure 7C). LZW is used because "longest-suffix routing" and "longest prefix matching" are information-theoretic duals; high-frequency suffixes correspond to contexts requiring the most disambiguation. The $k$ limit provides a hard guarantee on "worst-case active parameters"—at $k=512$, a token triggers at most $O(1\text{M})$ parameters, keeping CPU→GPU data transfer at the $O(1\text{MB})$ scale, ensuring prefetching can hide the latency.

3. **Block-diagonal Sorted Training + CPU Offload Inference**:
    - **Function**: Converts irregular row access into hardware-friendly batch-level attention and allows L$^3$ parameters to reside in CPU memory during inference.
    - **Mechanism**: During training, since L$^3$ mixes only in the channel dimension, tokens in a batch can be sorted by ID to group identical tokens together. This transforms the batch "attention mask" into a block-diagonal matrix (Figure 3), which can utilize kernels like FlexAttention or MegaBlocks. During inference (Figure 4), once a token is sampled, its $\{K_t, V_t\}$ is immediately prefetched from the CPU. For a 2.6B model on a B200, offloading L$^3$ results in only a minimal percentage drop in throughput for BS=1/8/300 (Table 2), as PCIe latency is masked by placing the first L$^3$ layer after the 4th decoder layer.
    - **Design Motivation**: Static routing pre-determines "which parameters to fetch" at the instant of token sampling. L$^3$ exploits this window, whereas MoE cannot. This allows a 7B total parameter model to run at inference speeds close to a 2.6B dense model.

### Loss & Training
The training objective is standard cross-entropy for language modeling with **no auxiliary losses**, an advantage over MoE. Based on the Llama architecture, models were pre-trained on FineWeb-Edu at 800M (400M decoder), 1.5B (1B), and 2.6B (1.9B) active parameters for 10B, 20B, and 30B tokens respectively, with a sequence length of 2048 and a 180K BPE vocab. Each L$^3$ layer defaults to $v = 710\text{K}$ and $k = 512$, targeting 2–4× sparsity.

## Key Experimental Results

### Main Results

| Active Params | L$^3$ Layers | Total Params | Wiki2 PPL ↓ | 0-shot Avg ↑ |
|---|---|---|---|---|
| 809M | 0 (dense) | 809M | 22.02 | 48.28 |
| 803M | 2 | 3.1B | 20.23 | 49.45 |
| 818M | 3 (wider) | 5.2B | **19.59** | **50.25** |
| 1.5B | 0 (dense) | 1.5B | 18.83 | 51.93 |
| 1.5B | 2 | 4.6B | **16.72** | **53.84** |
| 2.6B | 0 (dense) | 2.6B | 15.43 | 55.59 |
| 2.6B | 2 | 7B | **14.51** | **56.98** |

Adding L$^3$ reduces perplexity and improves downstream scores (ARC-c/e, HellaSwag, PIQA, Winogrande) at all scales. Gains appear from the start of training. Under iso-FLOP and iso-sparsity, L$^3$ consistently outperforms MoE baselines (Figure 8).

### Ablation Study

| Configuration | Observation | Explanation |
|---|---|---|
| 2 layers × 710K vs 4 × 355K vs 1 × 1420K | Similar quality | Single large layers compress the prefetch window; multiple small layers limit placement flexibility. |
| LZW vs Uniform Allocation | LZW significantly leads | Uniform allocation loses nearly all L$^3$ gains, proving allocation is the core knob. |
| LZW $k=\infty$ vs 512 vs 256 | $\infty$ is slightly better | $k=512$ nails worst-case activation to $O(1\text{M})$ with negligible quality drop. |
| $W_K$ and $W_V$ Weight Tying | Quality mostly unchanged | Directly halves the data transfer and sparsity ratio. |
| L$^3$ Placement (after layer 2/4/.../16) | Middle layers optimal | Early layers lack context; late layers have less time to influence output. |

### Key Findings
- **L$^3$ serves as an "information cache"**: Tuned lens analysis shows sharp KL divergence "steps" at layers containing L$^3$ (e.g., layers 4 and 16 in the 2.6B model), whereas dense models show smooth declines. This suggests L$^3$ caches information that would otherwise require re-computation by multiple decoder layers (Figure 10).
- **Early layers resemble lookup, deep layers resemble aggregation**: The KL divergence between the softmax distribution of the 1st L$^3$ layer and a uniform distribution is higher than that of the 2nd layer, implying early layers favor selecting 1-2 embeddings (lookup-like), while deeper layers aggregate broadly.
- **CPU offload is nearly cost-free**: For the 2.6B active / 7B total param model, offloading both L$^3$ layers to the CPU only dropped BS=1 throughput from 776 to 692 toks/s. PCIe latency is completely masked if the first L$^3$ layer is not in the very first position.
- **Training Throughput at 87%**: On 8×A100, the 800M dense model achieves 155K toks/s, which drops to 135K toks/s with L$^3$.

## Highlights & Insights
- **Reverting Router Form from Hidden-State to Token ID**: A seemingly "backward" design choice that solves almost all MoE system issues simultaneously while delegating context-awareness to attention. This is an elegant "separation of concerns" that could be applied to other dynamic routing scenarios.
- **Unlossy Compression for Capacity Allocation**: Treating the search for "suffixes covering a corpus" as dual to "codewords covering a corpus" to utilize LZW is a brilliant perspective.
- **$k$-Limit as a System-Quality Bridge**: A single hyperparameter controls both "worst-case active parameters" and "PCIe data transfer," ensuring predictable system latency.
- **Orthogonal to MoE**: L$^3$ is positioned as an additional sparse dimension alongside MoE, leaving room for "MoE + L$^3$" hybrids.

## Limitations & Future Work
- **Scale**: Experiments reached 2.6B active / 7B total parameters and 30B tokens. Verification of scaling laws at the scale of frontier MoEs (hundreds of billions of parameters) is pending.
- **Integration**: Explicit MoE + L$^3$ combination experiments were not included and are left for future work.
- **Fixed Vocabulary Dependency**: Allocation is determined once before training based on the BPE vocabulary; changing the vocab requires re-running LZW and re-training L$^3$.
- **Comparison with Engrams**: The authors acknowledge Engrams as concurrent work but lack a side-by-side comparison under identical settings.

## Related Work & Insights
- **vs MoE (Shazeer / DeepSeek-MoE / OLMoE)**: Both seek "Parameters ≫ Flops," but MoE uses hidden-state routing. L$^3$ is more system-friendly and avoids auxiliary losses. They are fundamentally complementary.
- **vs Product Key Networks (Lample 2019)**: PKN uses hidden-state queries for large embedding lookups, losing the offloading advantage of static token-ID routing.
- **vs SCONE (Yu et al. 2025)**: SCONE extends tokenizers at the **start** of the model; L$^3$ moves this to the **middle** with attention aggregation to cache intermediate representations.
- **vs Engrams (Cheng et al. 2026)**: Both use lookup tables and aggregation, but L$^3$ demonstrates that a minimal skeleton of "large table + context aggregation" is sufficient for scaling.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Decoupling static routing from contextual aggregation and using LZW for allocation is a clear and unique approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Sufficient ablations and comparisons across three scales, though lacks frontier-scale verification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The logical progression from motivation to system implementation is excellent; Figures 4, 5, and 10 are highly intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Provides a system-friendly axis for sparsity-based scaling that is immediately useful for CPU-offloaded inference.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ICML 2026\] ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](proactivellm_learning_active_interaction_for_streaming_large_language_models.md)
- [\[ACL 2026\] Lizard: An Efficient Linearization Framework for Large Language Models](../../ACL2026/llm_efficiency/lizard_an_efficient_linearization_framework_for_large_language_models.md)
- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](../../ACL2026/llm_efficiency/are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[ICLR 2026\] DND: Boosting Large Language Models with Dynamic Nested Depth](../../ICLR2026/llm_efficiency/dnd_boosting_large_language_models_with_dynamic_nested_depth.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Hyperparameter Transfer with Mixture-of-Experts Layers](hyperparameter_transfer_with_mixture-of-expert_layers.md)
- [\[ICML 2025\] Mixture of Lookup Experts](../../ICML2025/llm_efficiency/mixture_of_lookup_experts.md)
- [\[ICML 2026\] ProactiveLLM: Learning Active Interaction for Streaming Large Language Models](proactivellm_learning_active_interaction_for_streaming_large_language_models.md)
- [\[ACL 2025\] SpindleKV: A Novel KV Cache Reduction Method Balancing Both Shallow and Deep Layers](../../ACL2025/llm_efficiency/spindlekv_layered_kv_cache.md)
- [\[ACL 2026\] Lizard: An Efficient Linearization Framework for Large Language Models](../../ACL2026/llm_efficiency/lizard_an_efficient_linearization_framework_for_large_language_models.md)

</div>

<!-- RELATED:END -->
