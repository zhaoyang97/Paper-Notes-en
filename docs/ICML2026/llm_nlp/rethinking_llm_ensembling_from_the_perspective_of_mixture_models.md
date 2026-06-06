---
title: >-
  [Paper Note] Rethinking LLM Ensembling from the Perspective of Mixture Models
description: >-
  [ICML 2026][LLM/NLP][LLM Ensembling] This paper demonstrates that token-level ensembling for $n$ LLMs does not require running all models at every step. By randomly selecting one model per step according to weights to sa…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "LLM Ensembling"
  - "Mixture Models"
  - "Sampling Equivalence"
  - "KV Cache"
  - "Token-level Routing"
date: 2026-05-08
content_hash: aa374f5345239594
---

# Rethinking LLM Ensembling from the Perspective of Mixture Models

**Conference**: ICML 2026  
**arXiv**: [2605.00419](https://arxiv.org/abs/2605.00419)  
**Code**: https://github.com/jialefu/Mixture-model-like-Ensemble (Available)  
**Area**: LLM Efficiency / Decoding and Ensembling  
**Keywords**: LLM Ensembling, Mixture Models, Sampling Equivalence, KV Cache, Token-level Routing

## TL;DR
This paper demonstrates that token-level ensembling for $n$ LLMs does not require running all models at every step. By randomly selecting one model per step according to weights to sample the next token, the output distribution is strictly equivalent to "averaging before sampling." This reduces $n\times$ forward passes back to $1\times$ and, combined with "Lazy KV Cache Synchronization," achieves actual speedups of 1.78×–2.68×.

## Background & Motivation
**Background**: Traditional machine learning ensembling averages the probability distributions of multiple models and takes the argmax. This paradigm, when directly applied to LLMs, becomes "averaging the next-token distributions of $n$ models at each token step and then sampling from the averaged distribution," which improves generation quality but requires $n$ forward passes.

**Limitations of Prior Work**: Parallelizing $n$ models across $n$ GPUs still fails to approach $1\times$ speed because every token requires cross-device synchronization communication, creating heavy overhead. Existing methods like "reducing ensembling frequency" or "vocabulary truncation" only offer marginal optimizations; the bottleneck—the "necessity of explicitly constructing the ensemble distribution"—remains.

**Key Challenge**: The "argmax selection" behavior in traditional ensembling makes "explicitly averaged distributions" mandatory. However, LLM decoding itself is "sampling from a distribution," where the "shape" of the distribution only matters in the sense of sampling—an implicit assumption widely followed but never directly challenged.

**Goal**: To reduce the asymptotic inference cost of LLM ensembling from $O(n)$ to $O(1)$ with minimal algorithmic changes, while maintaining an output distribution perfectly identical to traditional ensembling.

**Key Insight**: The authors pose a simple yet critical question: Does LLM ensembling truly require invoking all models? They observe that "sampling from a weighted distribution" is equivalent to "selecting one component according to weights and then sampling from that component," which is precisely the definition of a mixture model.

**Core Idea**: View LLM ensembling as a mixture model $\sum_i \lambda_i M_i$. At each step, randomly sample an index $i\sim \mathrm{Mult}(\lambda)$, run only one forward pass for model $M_i$, and sample the token. The paper proves the resulting token distribution is identical to traditional ensembling and establishes an equivalence bridge between LLM ensembling and token-level routing.

## Method

### Overall Architecture
Given $n$ LLMs $M_1,\dots,M_n$ and weights $\lambda_i\ge 0$ where $\sum_i \lambda_i = 1$. Conventional Ensembling (CE) calculates $\bar P(y|x_{\le t}) = \sum_i \lambda_i M_i(y|x_{\le t})$ per step and samples the token. The proposed Mixture-model-like Ensembling (ME) first samples $i\sim\mathrm{Mult}(\lambda)$ at each step and uses only $M_i$ to calculate the distribution and sample the token. Equivalence proof: $P(x_{t+1}=y) = \sum_i P(\text{Select } i)\,M_i(y|\cdot) = \sum_i \lambda_i M_i(y|\cdot)$, which is perfectly consistent with CE. Combined with "Lazy KV Synchronization" and heterogeneous vocabulary alignment, this workflow can seamlessly replace CE.

### Key Designs

1.  **Replacing Explicit Averaging with Mixture-model-like Sampling**:
    - **Function**: Reduces the number of forward passes per step from $n$ to 1 without changing the token distribution.
    - **Mechanism**: For each generation step, an index $i$ is sampled independently from the multinomial distribution $\mathrm{Mult}(\lambda_1,\dots,\lambda_n)$. Only $M_i$ is executed for one forward pass to obtain $P_i = M_i(y|x_{\le t})$, followed by sampling $x_{t+1}$ from $P_i$. This algorithm only micro-modifies line 5 of the CE algorithm by "sampling the index first." Theoretically, the resulting token sequence follows the same distribution as CE.
    - **Design Motivation**: Traditional ML ensembling needs explicit averaging for the final argmax. Since LLM decoding inherently uses sampling, moving "sampling" to the model selection stage loses no information and saves $n-1$ forward passes.

2.  **Lazy KV Cache Synchronization Strategy**:
    - **Function**: Resolves the issue where $M_j$ lacks historical KV data when the previous step used $M_i$, preventing the "saved $n-1$ forward passes" from being wasted on KV synchronization.
    - **Mechanism**: Each model independently maintains its own KV cache. Only when a model is selected does it perform a "prefill-style completion" for the $k$ tokens it missed. This forward extension is memory-bandwidth bound; the latency of processing $k$ tokens is nearly identical to processing 1 token, making the amortized cost negligible.
    - **Design Motivation**: Naive synchronization of all model KV caches every step would force all model weights to be loaded, falling back to $O(n)$ memory bandwidth. Lazy synchronization exploits the hardware characteristic that LLM decoding is memory-bandwidth bound, compressing synchronization costs to near zero, similar to the verification phase in speculative decoding.

3.  **Unified Perspective on Heterogeneous Vocabularies + Token-level Routing**:
    - **Function**: Adapts ME to models with different vocabularies or architectures and places it within the same trade-off framework as token-level routing and MoE.
    - **Mechanism**: A mapping $F_i: P_i\mapsto \tilde P_i$ is defined for each model to project its distribution onto a unified vocabulary $U$. Replacing $M_i(y|x_{\le t})$ in the ME algorithm with $F_i[M_i(y|x_{\le t})]$ ensures seamless compatibility with vocabulary alignment schemes like UniTe. Theoretically, "training a router to select a model" vs "randomly selecting a model with fixed $\lambda$" is merely a distinction between input-dependent and input-independent routers. Thus, LLM ensembling can be viewed as the simplest special case of token-level routing.
    - **Design Motivation**: This perspective aligns ensembling, routing, and MoE on a single set of axes (training cost vs performance), turning the choice of scheme into a system design problem rather than a conceptual conflict.

### Loss & Training
ME requires no additional training and serves as a "plug-and-play" inference-time algorithm. It can be paired with vocabulary alignment (e.g., UniTe), with a slight one-time cost to inference speed from KV-prefill.

## Key Experimental Results

### Main Results

| Setting | Model Combinations | Task | CE Performance | ME Performance | Gain (Speedup) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Homogeneous/Same Vocab | Qwen-3B + Qwen-Math-1.5B | GSM8K/MMLU/BBH/ARC | Nearly identical to ME | Matches CE | 1.78×–2.68× vs CE (Seq/Par) |
| Heterogeneous/Diff Vocab | Openchat + DeepSeek-7B + Mistral-7B | 4 Datasets | Higher than single | Matches CE | Near single-model speed |
| Different Scales | Llama-3-8B + Llama-3-1B/3B | General | — | $\lambda$ speed vs accuracy | Significantly faster than CE |

### Ablation Study

| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| Single Model | Fastest speed, lowest accuracy | Upper bound comparison |
| CE (Sequential) | High accuracy, speed $\approx 1/n$ | Explicit averaging |
| CE (Parallel, GaC) | Slightly faster than Sequential | High cross-GPU communication overhead |
| ME | Accuracy equivalent to CE, speed near single-model | Key evidence |
| Model count 2→3 | No further improvement for most tasks | "More models is not necessarily better" |

### Key Findings
- ME and CE achieve equivalent accuracy across GSM8K, MMLU, BBH, and ARC tasks, strongly supporting the "distribution equivalence" proof.
- Parallel CE shows almost no speedup due to per-step cross-device communication, confirming that the LLM ensembling bottleneck is the "necessity of explicit distribution construction" rather than pure computation.
- Increasing the number of ensembled models does not monotonically improve performance; the optimal $n$ is related to the task/model combination, suggesting ensembling is more about "complementarity mining" than "brute-force averaging."

## Highlights & Insights
- Reducing "ensembling" from the conditional probability level to the mixture model level (selecting the source before sampling) is a rare example of a "one-line algorithm change + strict equivalence + massive efficiency gain" contribution with high pedagogical value.
- Lazy KV synchronization leverages the overlooked fact that "LLM decoding is memory-bandwidth bound," sharing conceptual roots with speculative decoding. This "amortized prefill" trick can be applied to optimize other multi-model collaboration scenarios.
- Treating ensembling as a degenerate case of token-level routing unifies "no training vs router training vs expert training" into a continuous spectrum, providing a clear coordinate system for future MoE/routing designs.

## Limitations & Future Work
- The output distribution of ME is equivalent to CE, so its benefits are primarily in efficiency; if CE itself yields only marginal gains (e.g., non-complementary models), ME cannot spontaneously create performance.
- The equivalence proof holds for "sampling-based decoding"; non-sampling scenarios like greedy search or beam search require separate analysis.
- Model selection is still determined by a fixed $\lambda$ and does not utilize contextual signals; a natural next step is to upgrade ME to input-dependent token-level routing using a lightweight router.

## Related Work & Insights
- **vs Traditional Ensembling (Rokach et al.)**: Traditional ensembling requires explicit averaging due to argmax; this work reveals that LLMs do not, because of sampling.
- **vs GaC/UniTe (Yu 2024 / Yao 2024)**: These improve vocabulary alignment or reduce ensembling frequency but are still limited by $n$ forward passes; this work bypasses that step fundamentally.
- **vs MoE / Token-level Routing**: The authors clearly compare the three in a "training cost-performance-inference speed" triangle, positioning ME as the cheapest ensembling option currently available (zero training + zero inference overhead + marginal performance gain).

## Rating
- Novelty: ⭐⭐⭐⭐ Simple idea but strictly equivalent and bridges ensembling/routing; a rare "missed truth" type of work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Verified across multiple model families × multiple tasks × homogeneous/heterogeneous / different sizes, with detailed speed testing.
- Writing Quality: ⭐⭐⭐⭐⭐ Strong narrative, clear motivation, and a short yet beautiful proof.
- Value: ⭐⭐⭐⭐ 1.78×–2.68× inference speedup with simple implementation, ready for immediate deployment in LLM applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](../../ICLR2026/llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](../../AAAI2026/llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)
- [\[ICML 2026\] SPA-Cache: Singular Proxies for Adaptive Caching in Diffusion Language Models](spa-cache_singular_proxies_for_adaptive_caching_in_diffusion_language_models.md)
- [\[ICML 2026\] Token-Efficient Change Detection in LLM APIs](token-efficient_change_detection_in_llm_apis.md)

</div>

<!-- RELATED:END -->
