---
title: >-
  [Paper Note] Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling
description: >-
  [ICLR 2026][LLM Efficiency][N-gram] This paper constructs the first benchmark specifically for evaluating "speculative decoding for accelerating LLM test-time scaling." By comparing 9 speculative decoding methods under a unified protocol across Best-of-N (BoN) and multi-round thinking paradigms, the study finds that reasoning trajectories in test-time sc
tags:
  - ICLR 2026
  - LLM Efficiency
  - N-gram
  - Best-of-N
date: 2026-05-08
content_hash: 5abff4080cfa46ce
---
# Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=DjOmnwX4wJ](https://openreview.net/forum?id=DjOmnwX4wJ)  
**Code**: https://github.com/sunshy-1/SpecTTS-Bench (Available)  
**Area**: LLM Efficiency / Speculative Decoding / Test-Time Scaling  
**Keywords**: Speculative Decoding, Test-Time Scaling, N-gram, Best-of-N, Multi-round Thinking  

## TL;DR
This paper constructs the first benchmark specifically for evaluating "speculative decoding for accelerating LLM test-time scaling." By comparing 9 speculative decoding methods under a unified protocol across Best-of-N (BoN) and multi-round thinking paradigms, the study finds that reasoning trajectories in test-time scaling are highly redundant. Consequently, simple N-gram-based methods (particularly SAM) can approach or even outperform the training-based EAGLE-3, while hybrid methods combining both achieve the highest overall speedup.

## Background & Motivation

**Background**: Test-time scaling has become a mainstream approach for enhancing LLM reasoning capabilities—improving performance without modifying parameters by allowing the model to "think longer" during inference. Two representative paradigms are Best-of-N (BoN, sampling $N$ trajectories and selecting the best via a verifier) and multi-round thinking (multi-round self-correction by feeding previous answers back into the prompt).

**Limitations of Prior Work**: This "thinking longer" strategy trades compute for performance. Generating multiple full responses or long reasoning chains incurs significant latency, making it nearly unusable for real-time interactions. While speculative decoding (using a fast draft mechanism to generate candidates verified in parallel by a large target model) is a key tool for mitigating latency, its effectiveness in the "highly structured and redundant" scenarios of test-time scaling has not been systematically studied.

**Key Challenge**: Reasoning trajectories in test-time scaling possess an overlooked property: **redundancy**. When repeatedly sampling or re-answering the same problem, models frequently replicate identical logical phrases, template code, and transitional sentences. The paper categorizes this as intra-turn and inter-turn redundancy (e.g., repeating the calculation $8^3=512$ within one turn, or rewriting the opening `<think>` tag verbatim across turns). This redundancy naturally suits retrieval/cache-based N-gram speculative decoding. However, existing benchmarks only test general tasks and fail to answer: **Can a simple adaptive N-gram mechanism outperform complex pre-trained draft models in these redundancy-rich scenarios?** What are the trade-offs regarding flexibility, training costs, and real-time adaptability?

**Goal**: Establish the first benchmark to systematically compare speculative decoding methods in accelerating test-time scaling, quantify the trade-offs between "flexibility vs. training cost vs. real-time adaptation," and verify the hypothesis that N-gram patterns are particularly suited for this context.

**Key Insight**: Instead of designing a new method, the authors build a fair experimental arena to determine which speculative decoding category is most cost-effective in redundant scenarios—the answer unexpectedly points toward the simplest N-gram approach.

**Core Idea**: Partition speculative decoding into three families (model-based, training-based, and N-gram-based) plus a hybrid approach, comparing them head-to-head across BoN and multi-round thinking paradigms on four reasoning datasets under two sampling temperatures, using unified metrics: MAT and Walltime Speedup.

## Method

### Overall Architecture
This is a **benchmark/empirical study** paper. Rather than proposing a new model, it integrates "test-time scaling paradigms to be accelerated," "evaluation datasets," and "speculative decoding algorithms" into a unified evaluation framework.

The benchmark spans three axes: **Axis 1 (Acceleration Target)** includes BoN (generating $N$ candidates) and multi-round thinking ($M$ iterations); **Axis 2 (Methods)** covers 9 methods across three families: model-based (SpS), training-based (EAGLE-3), and N-gram-based (PLD, REST, Lookahead, PIA, SAM, Recycling), plus a hybrid SAM[EAGLE-3]; **Axis 3 (Conditions)** involves 4 reasoning datasets (AIME24/25, MATH500, GPQA; 120 problems total) $\times$ 2 target models (DeepSeek-R1-Distill-Llama-8B, Qwen3-8B) $\times$ 2 temperatures ($T=0$ greedy, $T=0.6$ sampling). All methods are evaluated on **MAT (Mean Accepted Tokens)** and **Speed (Walltime Speedup Ratio)**.

### Key Designs

**1. Dual Paradigms + Redundancy Modeling: Exploiting Repetition**
The benchmark targets paradigms that intentionally create redundancy. BoN samples $N$ trajectories ($N=4$) which often share identical openings and logic templates. Multi-round thinking ($M=2$) repeats the reasoning skeleton from previous rounds. By demonstrating intra-turn and inter-turn redundancy, the paper validates why N-gram methods succeed—structured repetition turns "caching and reusing recent token sequences" into a high-hit-rate strategy.

**2. Three Families + Hybrid Comparison: Decoupling Acceptance from Speed**
Methods are categorized by their drafting mechanism. **Model-based SpS** uses a small sibling model (e.g., Qwen3-0.6B), yielding high MAT due to aligned distributions. **Training-based EAGLE-3** uses trainable heads on the target model. **N-gram methods** use data structures like Tries or Suffix Automata (SAM) to retrieve sequences from generation history. The comparison reveals that while SpS has high MAT, the 0.6B drafter is too heavy relative to the 8B target, resulting in a **Speed < 1× (actual slowdown)**. This disproves the intuition that higher MAT always equals higher speed.

**3. Categorizing N-grams: Token-level vs. Probability-level**
Within the N-gram family, the paper distinguishes two routes. **Token N-gram** (SAM, PLD, PIA, Lookahead, REST) finds repeating suffixes. SAM, using a Suffix Automaton, is highly efficient, outperforming EAGLE-3 in greedy settings (DSL-8B speedup $2.66\times$ vs $1.93\times$). However, they are sensitive to temperature; as $T$ increases, output diverges and speed drops. **Probability N-gram** (Recycling) caches top-K token probabilities. Since it preserves distribution information, it is **insensitive to temperature** ($<5\%$ drop at $T=0.6$), though its high computational overhead for tree-based verification results in high MAT but lower Speed translation.

**4. SAM[EAGLE-3] Hybrid: Merging Semantic Alignment and Repetition**
The hybrid method dynamically switches between EAGLE-3's semantic modeling and SAM's repetition reuse. When SAM fails to find a sufficiently long matching suffix, it falls back to EAGLE-3. SAM[EAGLE-3] achieves the **highest overall speedup** across all scenarios. However, it inherits SAM's temperature sensitivity, showing reduced gains at $T=0.6$.

## Key Experimental Results

Experiments used DeepSeek-R1-Distill-Llama-8B (DSL-8B) and Qwen3-8B (QW3-8B) for multi-round thinking (2 rounds) and BoN (4 trajectories), using float16, batch=1.

### Main Results (Multi-round Thinking, Overall Speedup)

| Method | Category | DSL-8B (T=0) Speed | DSL-8B (T=0) MAT | QW3-8B (T=0) Speed | QW3-8B (T=0) MAT |
| :--- | :--- | :--- | :--- | :--- | :--- |
| EAGLE-3 | Training-based | 1.93× | 2.35 | 2.91× | 4.38 |
| SAM | Token N-gram | **2.66×** | 2.93 | 2.28× | 2.37 |
| Recycling | Prob. N-gram | 2.10× | 2.99 | 2.15× | 3.01 |
| PLD | Token N-gram | 1.84× | 2.33 | 1.74× | 2.05 |
| SpS | Model-based | — | — | 0.87× | **7.07** |
| **SAM[EAGLE-3]** | Hybrid | **3.97×** | 4.72 | **3.49×** | 4.76 |

Key observation: SpS has the highest MAT (7.07) but actual Speed is 0.87× (slower than no acceleration). SAM outperforms EAGLE-3 on DSL-8B in greedy mode. The hybrid method is the clear winner.

### Temperature Sensitivity / Multi-turn Analysis

| Phenomenon | T=0 | T=0.6 | Change | Note |
| :--- | :--- | :--- | :--- | :--- |
| SAM (QW3-8B) speed | 2.28× | 1.78× | −22% | Token N-gram is temperature sensitive |
| Recycling (QW3-8B) speed | 2.15× | 2.06× | <5% | Prob. N-gram is robust |
| EAGLE-3 (QW3-8B) speed | — | — | −6% | Training-based is robust |
| SAM cross-turn speed | 2.13× | 2.83× | +33% | Progressive acceleration (Turn 1 $\to$ 2) |
| PIA cross-turn speed | 1.45× | 2.10× | +45% | Reuses previous calculations |

### Key Findings
- **High MAT $\neq$ Real Speedup**: SpS and Recycling suffer from computational overhead; wall-clock speed is the only meaningful metric under tight latency constraints.
- **Simple N-grams Outperform**: SAM's efficient suffix matching + natural redundancy allows it to rival or beat training-intensive EAGLE-3 in greedy settings with zero training cost.
- **Temperature is the Achilles' heel for Token N-grams**: Higher temperature leads to divergent outputs and lower hit rates for token-level reuse.
- **Progressive Acceleration is unique to Token N-grams**: SAM/PIA can reuse intermediate results from previous turns, increasing speedup as turns progress.
- **Hybrid methods are SOTA but heuristic**: SAM[EAGLE-3] currently uses a simple fallback strategy, leaving room for optimized switching logic.

## Highlights & Insights
- **Redundancy as a First-class Citizen**: The insight that test-time scaling is "repetition-rich generation" transforms N-gram methods from baselines to scenario-specific winners.
- **MAT vs. Speed Decoupling**: This exposes the myth that higher acceptance rates are always better, highlighting that the draft model's own overhead is the bottleneck for smaller target models.
- **Progressive Acceleration**: The benefit of caching grows with the number of rounds, providing direct inspiration for multi-agent or self-refine systems.
- **Path to Improved Hybrids**: The paper identifies the current heuristic switching as a weakness, suggesting "learned switching" as a future direction.

## Limitations & Future Work
- **Limited Scale**: Each dataset uses only 30 problems; target models are limited to 8B scale. The behavior with much larger models remains unexplored.
- **Heuristic Hybrid Strategy**: The fallback based on suffix length is crude; it does not learn when to trust the N-gram vs. the draft model.
- **Temperature Sensitivity**: The paper identifies but does not solve the performance degradation of token N-grams at high temperatures.
- **Future Directions**: Exploring "learned hybrid controllers" and porting the temperature robustness of probability N-grams into low-latency token N-gram frameworks.

## Related Work & Insights
- **vs. General Benchmarks (Xia et al., 2024)**: While they test general tasks, this paper focuses on "high-redundancy" scenarios, changing the conclusion on N-gram effectiveness.
- **vs. Individual Methods (EAGLE-3, SAM, etc.)**: By using a unified protocol, this paper reveals the anti-intuitive "MAT vs. Speed" gap that individual method papers often obscure.
- **vs. Test-time Scaling**: While those works focus on accuracy ("how to be smarter"), this work focuses on efficiency ("how to be faster") as an orthogonal plugin.

## Rating
- Novelty: ⭐⭐⭐⭐ (First benchmark for this specific high-value scenario; identifies unique redundancy patterns)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Extensive methods and conditions, though dataset size per task is small)
- Writing Quality: ⭐⭐⭐⭐ (Clear takeaways; well-structured dual-metric analysis)
- Value: ⭐⭐⭐⭐⭐ (Directly actionable selection guide for accelerating Reasoning Models)

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fast Catch-Up, Late Switching: Optimal Batch Size Scheduling via Functional Scaling Laws](fast_catch-up_late_switching_optimal_batch_size_scheduling_via_functional_scalin.md)
- [\[ICLR 2026\] Test-Time Training Done Right](test-time_training_done_right.md)
- [\[ICLR 2026\] Scaling Attention via Feature Sparsity](scaling_attention_via_feature_sparsity.md)
- [\[ICLR 2026\] Scaling Laws Meet Model Architecture: Toward Inference-Efficient LLMs](scaling_laws_meet_model_architecture_toward_inference-efficient_llms.md)
- [\[ICLR 2026\] xLSTM Scaling Laws: Competitive Performance with Linear Time-Complexity](xlstm_scaling_laws_competitive_performance_with_linear_time-complexity.md)

</div>

<!-- RELATED:END -->
