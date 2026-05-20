---
title: >-
  [Paper Note] Segment-Level Attribution for Selective Learning of Long Reasoning Traces
description: >-
  [ICLR2026][LLM Reasoning][reasoning trace] This paper applies Integrated Gradients to compute the attribution strength and direction consistency of each segment in long reasoning traces with respect to the final answer…
tags:
  - "ICLR2026"
  - "LLM Reasoning"
  - "reasoning trace"
  - "integrated gradients"
  - "selective SFT"
  - "segment attribution"
  - "CoT compression"
date: 2026-05-08
content_hash: a4fe8478dc8c23b9
---

# Segment-Level Attribution for Selective Learning of Long Reasoning Traces

**Conference**: ICLR2026
**arXiv**: [2602.00425](https://arxiv.org/abs/2602.00425)  
**Code**: [GitHub](https://github.com/SiyuanWangw/SegmentSelectiveSFT)  
**Area**: LLM Reasoning
**Keywords**: reasoning trace, integrated gradients, selective SFT, segment attribution, CoT compression

## TL;DR
This paper applies Integrated Gradients to compute the attribution strength and direction consistency of each segment in long reasoning traces with respect to the final answer, identifies important segments for selective SFT, and achieves up to 4.7% accuracy improvement over full-CoT training while reducing output length by 18%.

## Background & Motivation
1. Large reasoning models (LRMs) generate thousands of tokens in their CoT, yet only a small fraction genuinely contributes to answer prediction; much of the content is redundant, repetitive, or truncated.
2. Full SFT on redundant CoT causes models to learn verbose, uninformative patterns, wasting learning capacity and potentially degrading performance.
3. Existing compression methods based on token-level analysis disregard semantic coherence, while segment-level perplexity/entropy metrics do not fully correlate with actual importance.
4. Perplexity-based methods suffer from false positives (overestimating transitional text) and false negatives (underestimating verification steps and intermediate conclusions).
5. A direct measure of the causal contribution of each segment to correct answer prediction is needed.

## Method

### Overall Architecture
Segment-Level Selective SFT = Segment Splitting → IG Attribution → Dual-Metric Important Segment Selection → Selective Loss Training

### Segment Splitting
Long CoT traces are split into semantic units using transition keywords (e.g., "\n\nWait", "\n\nAlternatively", "\n\nLet me"). Each segment corresponds to an independent reasoning unit (e.g., problem understanding, intermediate exploration, verification).

### Integrated Gradients Attribution
For each token $o_n$, IG values are computed by integrating gradients along the straight-line path from a padding baseline to the actual embedding, measuring the direction and magnitude of that token's contribution to the probability of the correct answer. The integral is approximated with $J$ interpolation steps:
$$\text{IG}_i(x) \approx (x_i - x_i') \times \frac{1}{J}\sum_{j=1}^{J}\frac{\partial F(x'+j/J \cdot (x-x'))}{\partial x_i}$$

### Two Segment-Level Metrics
- **Attribution Strength**: $\text{Strength}(S) = \sum_{o_n \in S}|IG(o_n)| / \sqrt{N}$, measuring the magnitude of influence. The $\sqrt{N}$ normalization prevents longer segments from dominating due to token count alone. Cross-segment normalization enables comparison of relative importance within the same CoT.
- **Direction Consistency**: $\text{Consistency}(S) = |\sum IG(o_n)| / \sum|IG(o_n)|$, measuring the coherence of positive and negative contributions. A value near 1 indicates that tokens within the segment contribute uniformly in one direction (all positive or all negative), reflecting shallow confirmation or severely erroneous exploration; intermediate values indicate mixed positive and negative contributions — a hallmark of reflective reasoning where a segment contains both exploration and self-correction.

### Important Segment Selection (Two-Stage Filtering)
1. **Strength Threshold**: Segments are ranked by attribution strength in descending order; the top-$k^*$ segments whose cumulative strength reaches $\tau = 70\%$ are retained (approximately 30–40% of segments account for 80%+ of total attribution).
2. **Consistency Filtering**: Segments with direction consistency $> \beta = 0.8$ are removed from the top-$k^*$ set; only segments with consistency $\leq 0.8$ are designated as important. This yields approximately 33% of segments labeled as important, accounting for 45% of tokens — as important segments tend to be longer.

### Selective SFT
The full CoT is fed to the model as input (preserving autoregressive context), but the cross-entropy loss is computed only on tokens belonging to important segments; tokens in unimportant segments have their loss masked to zero:
$$L_{\text{Selective-SFT}}(\theta) = -\frac{1}{\sum_t I(o_t)}\sum_{t=1}^{T}I(o_t)\log P(o_t|o_{<t}, q; \theta)$$
This acts as an implicit regularizer — preventing the model from overfitting to redundant or repetitive content while maintaining coherence over the full context.

## Experiments

| Model | Method | Overall Acc | Output Length |
|-------|--------|:-----------:|:-------------:|
| R1-Distill-Qwen-1.5B | Full SFT | 44.8 | 16520 |
| R1-Distill-Qwen-1.5B | **Segment Selective** | **46.9**(+4.7%) | 13506(-18%) |
| R1-Distill-Qwen-7B | Full SFT | 62.1 | 9693 |
| R1-Distill-Qwen-7B | **Segment Selective** | **64.5**(+3.9%) | 8499(-12%) |
| Qwen2.5-7B-Instruct | Full SFT | 44.2 | 10317 |
| Qwen2.5-7B-Instruct | **Segment Selective** | **45.6**(+3.2%) | 9852(-5%) |

### Ablation Study

| Setting | Overall Acc | Overall Length |
|---------|:-----------:|:--------------:|
| R1-Distill-Qwen-7B (base) | 57.7 | 12518 |
| + Full CoT SFT | 62.1 | 9693 |
| + Token-level pruning SFT | 60.5 | 8112 |
| + **Segment Selective SFT** | **64.5** | **8499** |
| Strength only (w/o Consistency filtering) | 63.2 | 8856 |
| Consistency only (w/o Strength ranking) | 61.8 | 9234 |

**Key Findings**:
1. 30–40% of segments contribute over 80% of total attribution (verified by CDF curves), confirming substantial redundancy.
2. Important segments exhibit lower perplexity/entropy; unimportant segments contain more repetition (high BLEU > 0.8) and truncation (49% vs. 26%).
3. Selective SFT consistently outperforms both full SFT and token-level pruning — pruning disrupts contextual coherence.
4. The largest gains appear on OOD hard problems (AIME24, +13.3 pp), suggesting that selective learning improves generalization.
5. Direction consistency filtering ($\beta = 0.8$) contributes an additional ~1.3% accuracy improvement, validating the value of mixed-direction reasoning within segments.
6. The approach generalizes to RL settings by upweighting policy gradient contributions on important segments.
7. Advantages of Selective SFT are more pronounced under temperature sampling (pass@6), indicating that the model learns better reasoning patterns rather than merely memorizing specific outputs.

## Highlights & Insights
- IG attribution provides a direct measure of causal contribution from segments to the answer, which is more reliable than indirect metrics such as perplexity or entropy.
- The direction consistency metric is elegantly designed: it distinguishes shallow confirmation from reflective reasoning.
- Selective SFT simultaneously improves accuracy and efficiency (shorter outputs), yielding a win-win outcome.
- The analysis is thorough, empirically confirming that unimportant segments correspond to repetitive, truncated, or uninformative content.

## Loss & Training
Standard SFT computes loss uniformly over all tokens. The proposed Selective SFT applies an indicator function $I(o_t)$ to mask tokens: only tokens belonging to important segments contribute to the loss. This is equivalent to constructing an implicit curriculum in the loss landscape that focuses parameter updates on critical reasoning patterns rather than redundant filler content.

## Limitations & Future Work
- IG computation requires multi-step interpolated forward passes, incurring non-trivial computational overhead (though it is a one-time cost).
- The keyword-based segmentation strategy is relatively simple and may not generalize to all reasoning styles.
- Validation is limited to mathematical reasoning datasets; effectiveness on code generation and natural language reasoning remains unknown.
- The thresholds $\tau$ and $\beta$ require validation-set search, increasing tuning cost.

## Related Work & Insights
- CoT compression: Xia et al. 2025b (token-level analysis); Cui et al. 2025b (segment-level PPL); Li et al. 2025b (entropy-based)
- Selective SFT: Lin et al. 2024 (selective learning framework)
- Attribution methods: Sundararajan et al. 2017 (Integrated Gradients); this paper is the first to apply IG to reasoning-trace segments
- Long reasoning redundancy: Wang et al. 2025d (analysis of truncated thinking); Wu et al. 2025 (verbosity degrades reasoning performance)

## Rating
- Novelty: ⭐⭐⭐⭐ (novel combination of IG + segment attribution + selective SFT)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multiple models, ID/OOD benchmarks, comprehensive ablations)
- Writing Quality: ⭐⭐⭐⭐ (thorough analysis, good visualizations)
- Value: ⭐⭐⭐⭐ (direct engineering value for training on long reasoning traces)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models](../../NeurIPS2025/llm_reasoning/segment_policy_optimization_effective_segment-level_credit_assignment_in_rl_for_.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [\[ICLR 2026\] Is In-Context Learning Learning?](is_in-context_learning_learning.md)
- [\[ACL 2026\] Generating Effective CoT Traces for Mitigating Causal Hallucination](../../ACL2026/llm_reasoning/generating_effective_cot_traces_for_mitigating_causal_hallucination.md)
- [\[ICLR 2026\] The Illusion of Diminishing Returns: Measuring Long Horizon Execution in LLMs](the_illusion_of_diminishing_returns_measuring_long_horizon_execution_in_llms.md)

</div>

<!-- RELATED:END -->
