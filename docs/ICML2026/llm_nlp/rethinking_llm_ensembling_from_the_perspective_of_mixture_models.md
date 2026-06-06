---
title: >-
  [Paper Note] Rethinking LLM Ensembling from the Perspective of Mixture Models
description: >-
  [ICML 2026][LLM/NLP][LLM Ensembling] This paper proves that when performing token-level ensembling over $n$ LLMs, it is unnecessary to run all models at each step—randomly sample one model according to the weights to gen…
tags:
  - "ICML 2026"
  - "LLM/NLP"
  - "LLM Ensembling"
  - "Mixture Models"
  - "Sampling Equivalence"
  - "KV Cache"
  - "Token-level Routing"
date: 2026-05-08
content_hash: c6577b5eed8ee00d
---

# Rethinking LLM Ensembling from the Perspective of Mixture Models

**Conference**: ICML 2026  
**arXiv**: [2605.00419](https://arxiv.org/abs/2605.00419)  
**Code**: https://github.com/jialefu/Mixture-model-like-Ensemble (available)  
**Area**: LLM Efficiency / Decoding & Ensembling  
**Keywords**: LLM Ensembling, Mixture Models, Sampling Equivalence, KV Cache, Token-level Routing

## TL;DR
This paper proves that when performing token-level ensembling over $n$ LLMs, it is unnecessary to run all models at each step—randomly sample one model according to the weights to generate the next token, and the output distribution is strictly equivalent to "average then sample." This reduces the $n$-fold forward pass back to a single forward pass, and, combined with "lazy KV cache synchronization," achieves a practical speedup of 1.78×–2.68×.

## Background & Motivation
**Background**: Traditional machine learning ensembling averages the probability distributions of multiple models and then takes the argmax; this paradigm has been directly applied to LLMs as "average the next-token distributions of $n$ models at each token, then sample from the averaged distribution," which improves generation quality but requires $n$ forward passes.

**Limitations of Prior Work**: Even if $n$ models are placed on $n$ GPUs in parallel, true $1\times$ speed is unattainable because each token requires cross-device synchronization, incurring heavy overhead. Existing methods that "reduce ensembling frequency" or "truncate the vocabulary" only optimize the margins; the bottleneck of "explicitly constructing the ensemble distribution" remains.

**Key Challenge**: The "argmax selection" behavior in traditional ensembling makes explicit averaging necessary, but LLM decoding itself is "sampling from the distribution," where the "shape" of the distribution only matters in the sense of sampling—this is an implicit assumption inherited by practice but not directly challenged.

**Goal**: With minimal algorithmic changes, reduce the asymptotic inference cost of LLM ensembling from $O(n)$ back to $O(1)$, while maintaining output distributions identical to traditional ensembling.

**Key Insight**: The authors pose a simple yet crucial question—does LLM ensembling really require querying all models? They observe that "sampling from a weighted distribution" is equivalent to "selecting a component according to the weights and then sampling from that component," which is precisely the definition of a mixture model.

**Core Idea**: Treat LLM ensembling as a mixture model $\sum_i \lambda_i M_i$, where at each step, a random $i\sim \mathrm{Mult}(\lambda)$ is drawn, only $M_i$ is run for a forward pass and sampling, and it is proven that the resulting token distribution matches traditional ensembling. This also establishes an equivalence between LLM ensembling and token-level routing.

## Method

### Overall Architecture
Given $n$ LLMs $M_1,\dots,M_n$ and weights $\lambda_i\ge 0$, $\sum_i \lambda_i = 1$. Traditional ensembling (CE) computes $\bar P(y|x_{\le t}) = \sum_i \lambda_i M_i(y|x_{\le t})$ at each step and then samples a token; the proposed mixture-model-like ensembling (ME) samples $i\sim\mathrm{Mult}(\lambda)$ at each step, runs only $M_i$ to compute the distribution and sample a token. The equivalence proof: $P(x_{t+1}=y) = \sum_i P(\text{select }i)\,M_i(y|\cdot) = \sum_i \lambda_i M_i(y|\cdot)$, which is identical to CE. With "lazy KV synchronization" and heterogeneous vocabulary alignment, the entire process can seamlessly replace CE.

### Key Designs

1. **Mixture-model Sampling Replaces Explicit Averaging**:

    - **Function**: Reduces the number of forward passes per step from $n$ to 1 without changing the token distribution.
    - **Mechanism**: At each generation step, independently sample an index $i$ from the multinomial $\mathrm{Mult}(\lambda_1,\dots,\lambda_n)$, run only $M_i$ for a forward pass to obtain $P_i = M_i(y|x_{\le t})$, and sample $x_{t+1}$ from $P_i$. The entire algorithm only introduces a "pre-sample index" micro-modification at line 5 of the CE algorithm, and theoretically, the resulting token sequence is identically distributed to CE.
    - **Design Motivation**: Traditional ML ensembling requires explicit averaging due to the final argmax; LLM decoding is inherently sampling, so moving "sampling" ahead to the model selection stage loses no information and saves $n-1$ forward passes.

2. **Lazy Synchronization KV Cache Strategy**:

    - **Function**: Addresses the issue where "the previous step used $M_i$, and the next step switches to $M_j$ which lacks historical KV," avoiding spending the "saved $n-1$ forward passes" on KV synchronization.
    - **Mechanism**: Each model maintains its own KV cache independently, and only when it is selected does it perform a "prefill-style completion" for the $k$ lagging tokens—this forward extend is memory-bandwidth bound, and the latency for $k$ tokens is nearly the same as for 1 token, so the amortized cost is negligible.
    - **Design Motivation**: Naively "synchronizing all model KVs at each step" would load all model weights at every step, reverting to $O(n)$ memory bandwidth; lazy synchronization leverages the hardware characteristic that "LLM decoding is memory-bandwidth bound," pushing synchronization cost close to zero, akin to the verify phase in speculative decoding.

3. **Unified Perspective of Heterogeneous Vocabulary + Token-level Routing**:

    - **Function**: Enables ME to support models with different vocabularies/architectures and places it in the same framework as token-level routing/MoE for comparative analysis.
    - **Mechanism**: Define a mapping $F_i: P_i\mapsto \tilde P_i$ for each model to project its distribution onto a unified vocabulary $U$, then replace $M_i(y|x_{\le t})$ in line 5 of the ME algorithm with $F_i[M_i(y|x_{\le t})]$, seamlessly compatible with vocabulary alignment schemes like UniTe. Theoretically, "training a router to select models" and "randomly selecting models according to fixed $\lambda$" differ only in whether the router is input-dependent or input-independent, so LLM ensembling can be seen as the simplest special case of token-level routing.
    - **Design Motivation**: This perspective places "ensembling, routing, MoE" on the same axis (training cost vs. performance), making the choice of approach a system design issue rather than a conceptual conflict.

### Loss & Training
ME requires no additional training and is a "plug-and-play" inference-time algorithm; it can be combined with vocabulary alignment methods like UniTe, with only a minor one-time KV-prefill cost for inference speed.

## Key Experimental Results

### Main Results

| Setting | Model Combination | Task | CE Performance | ME Performance | Speedup |
|---------|------------------|------|---------------|---------------|---------|
| Homogeneous, Same Vocabulary | Qwen-3B + Qwen-Math-1.5B | GSM8K/MMLU/BBH/ARC | Nearly identical to ME | On par with CE | 1.78×–2.68× vs CE (sequential/parallel) |
| Heterogeneous, Different Vocabularies | Openchat + DeepSeek-7B + Mistral-7B | Four datasets | Higher than single model | On par with CE | Close to single model speed |
| Different Scales | Llama-3-8B + Llama-3-1B/3B | Overall | — | $\lambda$ controls speed vs. accuracy tradeoff | Significantly faster than CE |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Single Model | Fastest, lowest accuracy | Upper bound comparison |
| CE (Sequential) | High accuracy, speed $\approx 1/n$ | Explicit averaging |
| CE (Parallel, GaC) | Slightly faster than Sequential | High multi-GPU communication overhead |
| ME | Accuracy equivalent to CE, speed close to single model | Key evidence |
| Number of Models 2→3 (❸+❹+❺) | No further improvement on most tasks | "More models not always better" |

### Key Findings
- ME matches CE in accuracy on GSM8K, MMLU, BBH, and ARC tasks, strongly supporting the "distribution equivalence" proof.
- Parallel CE achieves almost no speedup due to per-step cross-device communication, confirming that the bottleneck in LLM ensembling is "explicitly constructing the ensemble distribution" rather than pure computation.
- Increasing the number of ensemble models does not monotonically improve performance; the optimal $n$ depends on the task/model combination, suggesting ensembling is more about "mining complementarity" than "brute-force averaging."

## Highlights & Insights
- Reducing "ensembling" from the conditional probability level to the mixture model level of "select source then sample" is a rare case of "one-line algorithm change + strict equivalence + huge efficiency gain," with high pedagogical value.
- Lazy KV synchronization leverages the often-overlooked hardware fact that "LLM decoding is memory-bandwidth bound," sharing the same origin as speculative decoding; this "amortized prefill" trick can be used to explain and optimize other multi-model collaboration scenarios.
- Viewing ensembling as a degenerate case of token-level routing unifies "no training vs. training a router vs. training experts" into a continuous spectrum, providing a concise coordinate system for future MoE/routing design.

## Limitations & Future Work
- ME's output distribution is equivalent to CE, but the main benefit is efficiency; when CE itself brings only marginal gains (e.g., non-complementary, homogeneous models), ME cannot create performance out of thin air.
- The equivalence proof holds for "sampling decoding"; for greedy/beam search and other non-sampling scenarios, further analysis is needed.
- Model selection is still determined by fixed $\lambda$ and does not utilize contextual signals; a natural next step is to use a lightweight router to upgrade ME to input-dependent token-level routing.

## Related Work & Insights
- **vs Traditional Ensembling Paradigm (Rokach et al.)**: Traditional methods require explicit averaging due to argmax; this work reveals that LLMs, due to sampling, do not.
- **vs GaC/UniTe (Yu 2024 / Yao 2024)**: These improve vocabulary alignment or reduce ensembling frequency but are still limited by $n$ forward passes; this work fundamentally bypasses this step.
- **vs MoE / Token-level Routing**: The authors explicitly compare the three in the "training cost–performance–inference speed" triangle, proposing ME as "training-free + zero inference overhead + slight performance gain," making it the most cost-effective ensembling option currently.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple idea but strictly equivalent and unifies ensembling with routing—a rare "overlooked truth" type of work
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple model families × multiple tasks × homogeneous/heterogeneous / different sizes all validated, with detailed speed tests
- Writing Quality: ⭐⭐⭐⭐⭐ Strong narrative, clear motivation, concise and elegant proofs
- Value: ⭐⭐⭐⭐ 1.78×–2.68× inference speedup with simple implementation, immediately applicable to LLM applications

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Are Language Models Efficient Reasoners? A Perspective from Logic Programming](../../NeurIPS2025/llm_nlp/are_language_models_efficient_reasoners_a_perspective_from_logic_programming.md)
- [\[ICLR 2026\] Rethinking Code Similarity for Automated Algorithm Design with LLMs](../../ICLR2026/llm_nlp/rethinking_code_similarity_for_automated_algorithm_design_with_llms.md)
- [\[AAAI 2026\] An Invariant Latent Space Perspective on Language Model Inversion](../../AAAI2026/llm_nlp/an_invariant_latent_space_perspective_on_language_model_inve.md)
- [\[ICLR 2026\] Predicting LLM Reasoning Performance with Small Proxy Models](../../ICLR2026/llm_nlp/predicting_llm_reasoning_performance_with_small_proxy_models.md)
- [\[ICML 2026\] Escaping Mode Collapse in LLM Generation via Geometric Regulation](escaping_mode_collapse_in_llm_generation_via_geometric_regulation.md)

</div>

<!-- RELATED:END -->
