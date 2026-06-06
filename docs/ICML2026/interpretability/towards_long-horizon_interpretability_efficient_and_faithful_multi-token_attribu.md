---
title: >-
  [Paper Note] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs
description: >-
  [ICML 2026][Interpretability][token attribution] Focusing on the challenge where token-wise attribution in reasoning LLMs with long chain-of-thought is slow ($\mathcal{O}(M\cdot N)$) and quality is degraded by internal r…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "token attribution"
  - "reasoning LLM"
  - "span-wise aggregation"
  - "recursive attribution"
  - "long-context interpretability"
date: 2026-05-08
content_hash: 8e49ad68267db697
---

# Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs

**Conference**: ICML 2026  
**arXiv**: [2602.01914](https://arxiv.org/abs/2602.01914)  
**Code**: https://github.com/wbopan/flashtrace  
**Area**: Interpretability / LLM Reasoning / Token Attribution  
**Keywords**: token attribution, reasoning LLM, span-wise aggregation, recursive attribution, long-context interpretability

## TL;DR
Focusing on the challenge where token-wise attribution in reasoning LLMs with long chain-of-thought is slow ($\mathcal{O}(M\cdot N)$) and quality is degraded by internal reasoning tokens ("information absorption"), this paper proposes FlashTrace. It utilizes span-wise aggregation to compute the attribution of an entire target span in a single pass and employs recursive attribution to trace importance from the output back through the reasoning chain to the original input. FlashTrace is over 130x faster than the strongest baseline IFR on 5k target spans while consistently outperforming in faithfulness across RULER, MATH, and MoreHopQA.

## Background & Motivation

**Background**: Token attribution is a primary interpretability technique for explaining LLM outputs. Dominant approaches include perturbation-based (REAGENT/CLP), gradient-based (Integrated Gradients), and attention+relevance propagation (IFR, AttnLRP). These methods typically assume the target for explanation is a single token, calculating the causal contribution of each context token as a distribution for that target.

**Limitations of Prior Work**: Modern reasoning LLMs (o1, DeepSeek-R1, Qwen-3) generate thousands of chain-of-thought tokens before providing an answer, leading to two specific problems for token attribution:
- Efficiency Bottleneck: To explain an output span of length $M$, attribution must be run for each token individually, increasing complexity from $\mathcal{O}(N)$ to $\mathcal{O}(M\cdot N)$. Using IG for 5k outputs takes over 10 hours, and even the fastest baseline IFR takes 38 minutes, making it unusable in agent workflows.
- Faithfulness Drop (Information Absorption): Since autoregressive models trigger the next token directly from the preceding ones, reasoning tokens $\mathbf{T}$ absorb the vast majority of attribution mass. Figure 1 quantifies this: when CoT is enabled, the mass allocated to $\mathbf{T}$ increases from ~80% to >90%, while the recovery rate of ground-truth input tokens drops sharply from 26% to <10%. Ultimately, explanations merely state that "the answer was determined by the previous reasoning sentence," failing to trace back to the actual evidence in the prompt.

**Key Challenge**: Existing methods only characterize direct input $\to$ output dependencies, whereas the causal chain in reasoning LLMs is tripartite: $\mathbf{I}\to\mathbf{T}\to\mathbf{O}$. The goal is to bypass the intermediate bridge $\mathbf{T}$ to propagate importance back to $\mathbf{I}$, while avoiding brute-force attribution for every token in $\mathbf{T}$. In other words, "multi-token targets" and "multi-hop propagation" must be addressed simultaneously.

**Goal**: Define the multi-token attribution problem and decompose it into two sub-problems: (i) given a span $S$, calculate the contributions of all source tokens to $S$ in one pass; (ii) trace the mass absorbed by reasoning tokens back to the original input along the causal chain.

**Key Insight**: The authors observe that within the ALTI/IFR framework, the contribution of an attention head to a single target position $i$ is expressed as $\mathbf{f}_{j\to i}(\mathbf{x}_j)=\alpha_{i,j}^h \cdot (\mathbf{x}_j W_V^h W_O^h)$, where $\mathbf{v}_j = \mathbf{x}_j W_V^h W_O^h$ depends only on the source token and is decoupled from the target position $i$. By extending this observation to an entire target span, the computation for source token contributions to the whole span can be algebraically factorized.

**Core Idea**: Use span-wise aggregation to compress "attribution for the entire span" into a single forward pass, then use recursive attribution to treat the scores assigned to reasoning tokens in the previous hop as a "weighted target" for the next hop, effectively flowing importance back along $\mathbf{O}\to\mathbf{T}\to\mathbf{I}$ without significant added cost.

## Method

### Overall Architecture
Input: A complete context $\mathbf{S}=\mathbf{I}\circ\mathbf{T}\circ\mathbf{O}$ (user input + model-generated reasoning + final output), where the target for explanation is the output span $\mathbf{O}$.

Output: Importance scores $\mathbf{w}_{final}$ for each context token, expected to accurately locate the tokens in the original input $\mathbf{I}$ that truly determine the answer.

Mechanism:
1. **Hop 0**: Treat $\mathbf{O}$ as the target span and use span-wise aggregation to calculate the attribution distribution $\mathbf{w}^{(0)}$ over the context, identifying parts falling on the input $\mathbf{w}_{\mathbf{I}}^{(0)}$ and reasoning tokens $\mathbf{w}_{\mathbf{T}}^{(0)}$.
2. **Hop $k\ge 1$**: Use $\mathbf{w}_{\mathbf{T}}^{(k-1)}$ as weights for reasoning tokens to construct a weighted target span, then perform span-wise aggregation again to obtain a new $\mathbf{w}^{(k)}$.
3. **Aggregation**: Sum the input components across all hops by converting them based on "remaining mass" according to information flow semantics to obtain the final $\mathbf{w}_{final}$. In experiments, $K=1$ is largely sufficient.

The underlying metric utilizes the L1 proximity from ALTI: $\text{Proximity}(\mathbf{z},\mathbf{y}) = \max(0, -\|\mathbf{y}-\mathbf{z}\|_1 + \|\mathbf{y}\|_1)$, which measures "how much the norm of target vector $\mathbf{y}$ shrinks when contribution $\mathbf{z}$ is removed." This metric is more stable than cosine similarity in high-dimensional anisotropic Transformer spaces.

### Key Designs

1. **Span-wise Aggregation (One-pass attribution for entire span)**:
    - **Function**: Compresses "$M$ individual token attributions" into "one attribution for the entire span," reducing complexity from $\mathcal{O}(M\cdot N)\to\mathcal{O}(N)$.
    - **Mechanism**: The hierarchical representation of the target span is summed as $\mathbf{Y}_S=\sum_{i\in S}\mathbf{y}_i$. The contribution of source token $j$ to the span is defined as $\mathbf{Z}_S=\sum_{i\in S}\mathbf{z}_{j\to i}$, applying the same L1 proximity. By leveraging the linearity of attention, $\mathbf{v}_j = \mathbf{x}_j W_V^h W_O^h$ is extracted from the attention head contribution $\alpha_{i,j}^h \cdot \mathbf{v}_j$, resulting in $\mathbf{F}_{j\to S}=\mathbf{v}_j \cdot (\sum_{i\in S}\alpha_{i,j}^h)$. Expensive V/O projections are computed once, and each target position requires only one additional scalar multiplication-addition. The residual stream is also summed by span, ensuring memory consumption does not scale with $M$.
    - **Design Motivation**: Directly addresses the efficiency bottleneck of processing 5k-token outputs. Since this is an algebraic rearrangement without approximation, it theoretically retains all faithfulness properties of ALTI/IFR while freeing up computational budget for multi-hop propagation.

2. **Recursive Attribution (Tracing back along the reasoning chain)**:
    - **Function**: Converts the importance $\mathbf{w}_{\mathbf{T}}^{(k-1)}$ assigned to reasoning tokens in the previous hop into a "weighted target" for the next hop, preventing mass from stalling at $\mathbf{T}$.
    - **Mechanism**: Generalizes span-wise aggregation from a "binary mask" to a weighted span. The new target is $\mathbf{Y}^{(k)}=\sum_{j\in \mathbf{T}} w_j^{(k-1)} \cdot \mathbf{y}_j$, with corresponding contribution $\mathbf{Z}^{(k)}=\sum_{j\in \mathbf{T}} w_j^{(k-1)} \cdot \mathbf{z}_{k\to j}$. Factorization remains valid—$\mathbf{v}_k$ is computed once, followed by a dot product with the scalar $\sum_j w_j^{(k-1)}\alpha_{j,k}^h$. Thus, the cost per hop is roughly equal to one forward pass. Importance is treated as "information flow probability," where remaining mass $\rho_k=\sum_{t\in\mathbf{T}}w_t^{(k)}$ propagates along the chain.
    - **Design Motivation**: Specifically targets information absorption. Single-hop attribution only explains that the answer depends on the reasoning; multi-hop is required to answer which prompt segment the reasoning itself depends on. Designing this as the same "span-wise op on weighted spans" avoids sentence-level segmentation and preserves $\mathcal{O}(N)$ complexity.

3. **Probability-mass Aggregation across hops**:
    - **Function**: Combines the input components $\mathbf{w}_{\mathbf{I}}^{(k)}$ from $K$ hops into a single final distribution $\mathbf{w}_{final}$, making multi-hop results comparable and visualizable.
    - **Mechanism**: Treats the recursive process as a step-by-step diversion of mass—at each hop, mass either "precipitates" to the input or "remains on the reasoning chain" awaiting the next hop. The aggregation formula is $\mathbf{w}_{final}=\mathbf{w}_{\mathbf{I}}^{(0)}+\sum_{k=1}^{K}(\prod_{j=0}^{k-1}\rho_j)\cdot \mathbf{w}_{\mathbf{I}}^{(k)}$, where $\rho_j$ is the remaining mass on $\mathbf{T}$ at hop $j$.
    - **Design Motivation**: Uses "remaining mass accounting" to ensure distributions from different hops are merged on the same probability scale. This prevents the inflation of input contributions from shorter reasoning chains and provides a natural early-stopping condition when $\rho_k$ becomes negligible.

### Loss & Training
This method is a training-free post-hoc interpretability algorithm. There is no training loss. The only hyperparameter is the number of recursive hops $K$ (default $K=1$ in experiments). It does not modify model weights and makes no intrusive assumptions about the underlying Transformer—using only forward attention weights and value/output projections.

## Key Experimental Results

### Main Results

Evaluated on the RULER series (multi-demand Needle-in-a-Haystack mq, Variable Tracking mv, long-context HotpotQA) using Qwen-3 8B Instruct. Metrics: Recovery Rate ↑ / RISE ↓ / MAS ↓.

| Dataset (Task) | Metric | FlashTrace | Best Baseline | Gain |
|---|---|---|---|---|
| mq q4 (NIAH) | Recovery Rate ↑ | 0.413 | 0.328 (IFR) | +8.5 pp |
| mv v4 (Variable Tracking) | Recovery Rate ↑ | 0.516 | 0.452 (IFR) | +6.4 pp |
| HotpotQA h4 c1 | Recovery Rate ↑ | 0.755 | 0.253 (IFR) / 0.229 (AttnLRP) | +50 pp |
| HotpotQA(1024) | RISE ↓ | 0.033 | 0.074 (IFR) | −55% |
| MATH | MAS ↓ | 0.446 | 0.490 (IFR) | −9% |
| MoreHopQA | MAS ↓ | 0.205 | 0.228 (IFR) | −10% |
| Aider Code Gen | MAS ↓ | 0.173 | 0.773 (IFR per-token avg) | −78% |

Efficiency (5k token target span, RULER): FlashTrace takes < 20 s, IFR takes > 38 min, representing a **130x+ speedup**. IG / IG-Attn / Perturbation methods suffer from OOM in long contexts (Figure 4 dashed lines).

### Ablation Study

| Configuration | Complexity | Time (s) | RISE ↓ | MAS ↓ | Description |
|---|---|---|---|---|---|
| Exhaustive Token-Level Rollout | $\mathcal{O}(M\cdot N)$ | 11.2 | 0.116 | 0.193 | Theoretical upper bound on MoreHopQA |
| FlashTrace (span-wise + recursive) | $\mathcal{O}(N)$ | 0.72 | 0.128 | 0.205 | Time ↓93.6%, faithfulness drops only ~10% |
| FlashTrace, K=0 (no recursion) | — | — | — | — | Fig 1: mass stuck on $\mathbf{T}$, recovery rate <10% |
| FlashTrace on LLaMA-3.1-8B-It | — | — | 0.171 | 0.231 | Still outperforms IFR (0.206/0.298) and AttnLRP |

### Key Findings
- **Recursion is a qualitative leap**: With just $K=1$, the HotpotQA Recovery Rate jumps from 0.13–0.25 (IFR/AttnLRP) to 0.51–0.76. This proves that information absorption cannot be fixed by minor adjustments; explicit modeling of $\mathbf{O}\to\mathbf{T}\to\mathbf{I}$ flow is essential. Figure 3 shows scores generally decreasing for reasoning tokens (red) and increasing for input tokens (green) in the second hop.
- **Span-wise aggregation is a nearly cost-free approximation**: Compared to Exhaustive Rollout, FlashTrace degrades RISE/MAS by only 6–10% on MoreHopQA while reducing runtime from 11.2 s to 0.72 s (93.6% speedup). Aggregating multi-token targets into a single span provides a high-quality approximation without expensive sentence-level segmentation.
- **Stability across tasks and models**: In Aider Code Gen, MAS drops from ~0.78 (IFR) to 0.17, a nearly 5x improvement. This indicates that span-wise attribution handles structured intermediate products (code diff blocks) as well as natural language reasoning. Consistency across LLaMA-3.1 suggests the mechanism is not model-specific.
- **The efficiency curve determines usability**: Figure 4 demonstrates that IG/Perturbation methods OOM at ~2–4k inputs or a few hundred output tokens. While IFR avoids OOM, its time scales linearly with target span length. FlashTrace limits memory and time growth relative to target span length, making it viable for agent workflows.

## Highlights & Insights
- **Free lunch via algebraic identity**: The extraction of $\mathbf{F}_{j\to S}=\mathbf{v}_j \cdot (\sum_{i\in S}\alpha_{i,j}^h)$ is the lever of the paper. It mathematically transforms "attribution for a span" into a scalar weighting of single-token attribution, eliminating the $M$-dimensional loop without introducing approximations. This is a synergy of attention linearity and the additive property of the L1 proximity metric.
- **Framing multi-hop attribution as probability flow**: Using $\rho_k$ as "remaining mass" and $(\prod \rho_j)\cdot \mathbf{w}_{\mathbf{I}}^{(k)}$ as the "probability of precipitating to the input at hop $k$" provides a semantically clear aggregation framework that supports natural early stopping. This perspective can be extended to any scenario requiring back-propagation of importance along a generation chain (e.g., tool-calling, ReAct traces).
- **Correct diagnosis = Half the solution**: The authors first quantify "reasoning tokens absorbing mass $\to$ recovery rate drop" in Figure 1. This converts the vague notion that "CoT degrades explanation" into measurable metrics. Every methodological design thereafter explicitly maps to a diagnosed pain point.

## Limitations & Future Work
- **Proximity is correlation, not counterfactual**: The authors emphasize that L1 proximity measures "how much source representation flows into the target span," representing informational contribution. While validated with perturbation metrics like RISE/MAS, rigorous causal attribution (counterfactual/mediation analysis) was not performed.
- **Cost of fixing recursion at $K=1$**: While $K=1$ solves most reasoning chain dependencies, the impact of $K>1$ is only relegated to Appendix H. Whether ultra-long reasoning (dozens of reflections, multi-round tool-calls) requires adaptive $K$ or threshold-based early stopping is not fully explored.
- **Target spans must be contiguous**: The method naturally aggregates contiguous spans, but critical steps in reasoning traces are often non-contiguous. While the formula holds for any weighted subset, automatically selecting non-contiguous targets beyond simple top-k selection remains an open problem.
- **Validation limited to 8B scale**: Experiments only cover Qwen-3 8B and LLaMA-3.1-8B. MoE models or 30B+ parameters may exhibit different information flow patterns due to shared attention or expert routing.

## Related Work & Insights
- **Comparison with AttnLRP / IFR**: These also use attention/relevance propagation but are limited to single token targets, requiring "per-token execution + averaging" which is slow and prone to information absorption. FlashTrace upgrades the target to weighted spans and explicitly handles multiple hops.
- **Comparison with CAGE (concurrent work)**: CAGE also uses a recursive approach but segments at the sentence level, running full attributions multiple times. Its cost scales linearly with reasoning steps. FlashTrace operates at a finer token-span grain and uses algebraic span-wise reduction to reach $\mathcal{O}(N)$ efficiency.
- **Comparison with Integrated Gradients / Perturbation**: IG methods have strong theoretical guarantees but OOM in long contexts due to gradients and integration steps. FlashTrace sacrifices some "perfect causal solvability" for engine-ready latency in the range of seconds.
- **Comparison with Circuit-tracer / Transcoder**: Those explore internal feature circuits (which components are responsible for a behavior). FlashTrace answers which external input tokens triggered an output. They are complementary: one is mechanistic interpretability, while the other is data interpretability.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Leverages algebraic properties of ALTI/IFR to their fullest and constructs a precision recursive attribution for reasoning LLMs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers six task types (NIAH / VT / HotpotQA / MATH / MoreHopQA / Aider) and includes cross-model validation and efficiency/faithfulness Pareto analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear logical chain from diagnosis (Sec 3.2) to method (Sec 4) to experiments. Derivations are concise.
- **Value**: ⭐⭐⭐⭐⭐ Addresses the norm of long context and CoT in current agent deployments. Providing believable attribution in seconds is directly productive for debugging and prompt optimization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Memory as a Markov Matrix: Sample Efficient Knowledge Expansion via Token-to-Dictionary Mapping](memory_as_a_markov_matrix_sample_efficient_knowledge_expansion_via_token-to-dict.md)
- [\[NeurIPS 2025\] Efficient Vision-Language Reasoning via Adaptive Token Pruning](../../NeurIPS2025/interpretability/efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [\[ACL 2026\] Jacobian Scopes: Token-Level Causal Attributions in LLMs](../../ACL2026/interpretability/jacobian_scopes_token-level_causal_attributions_in_llms.md)
- [\[ICML 2026\] IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring](iqa-spider_unifying_multi-granularity_image_quality_assessment_with_reasoning_gr.md)
- [\[ICML 2026\] Barriers to Counterfactual Credit Attribution for Autoregressive Models](barriers_to_counterfactual_credit_attribution_for_autoregressive_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] IQA-Spider: Unifying Multi-Granularity Image Quality Assessment with Reasoning, Grounding and Referring](iqa-spider_unifying_multi-granularity_image_quality_assessment_with_reasoning_gr.md)
- [\[ICML 2026\] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)
- [\[ICML 2026\] From Rashomon Theory to PRAXIS: Efficient Decision Tree Rashomon Sets](from_rashomon_theory_to_praxis_efficient_decision_tree_rashomon_sets.md)
- [\[ICML 2026\] Learning Coherent Representations: A Topological Approach to Interpretability](learning_coherent_representations_a_topological_approach_to_interpretability.md)
- [\[ICML 2026\] LLMs Lean on Priors, Not Programming Language Semantics](llms_lean_on_priors_not_programming_language_semantics.md)

</div>

<!-- RELATED:END -->
