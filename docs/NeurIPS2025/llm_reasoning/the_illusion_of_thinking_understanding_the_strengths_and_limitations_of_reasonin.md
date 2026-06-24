---
title: >-
  [Paper Note] The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity
description: >-
  [NeurIPS 2025][Reasoning][Large Reasoning Models] Using controlled puzzle environments, this paper systematically reveals a three-regime behavioral pattern in Large Reasoning Models (LRMs): performance falls below standard LLMs at low complexity (overthinking), substantially surpasses them at moderate complexity, and collapses completely (0%) at high complexity. Counterintuitively, models reduce thinking token usage at the point of collapse, demonstrating that current LRMs ha…
tags:
  - "NeurIPS 2025"
  - "Reasoning"
  - "Large Reasoning Models"
  - "Problem Complexity"
  - "Thinking Tokens"
  - "Chain-of-Thought"
  - "Reasoning Collapse"
date: 2026-05-08
content_hash: 1a3998e2e493e1c0
---

# The Illusion of Thinking: Understanding the Strengths and Limitations of Reasoning Models via the Lens of Problem Complexity

**Conference**: NeurIPS 2025
**arXiv**: [2506.06941](https://arxiv.org/abs/2506.06941)  
**Code**: Not yet open-sourced  
**Area**: LLM Reasoning
**Keywords**: Large Reasoning Models, Problem Complexity, Thinking Tokens, Chain-of-Thought, Reasoning Collapse

## TL;DR
Using controlled puzzle environments, this paper systematically reveals a three-regime behavioral pattern in Large Reasoning Models (LRMs): performance falls below standard LLMs at low complexity (overthinking), substantially surpasses them at moderate complexity, and collapses completely (0%) at high complexity. Counterintuitively, models reduce thinking token usage at the point of collapse, demonstrating that current LRMs have not developed genuinely generalizable reasoning capabilities.

## Background & Motivation

**Background**: LRMs such as o3, DeepSeek-R1, and Claude-3.7-Thinking achieve impressive results on benchmarks like MATH and AIME, yet these benchmarks are susceptible to data contamination and do not permit controlled manipulation of task complexity.

**Limitations of Prior Work**: Existing evaluations cannot answer the central question of whether LRM reasoning reflects genuine generalization or sophisticated pattern matching, nor how performance varies precisely with complexity.

**Key Challenge**: The fact that AIME25 performance is lower than AIME24—despite the problems being considered easier by humans—suggests data contamination. A controlled, contamination-free evaluation environment is therefore necessary.

**Goal**: To systematically measure the reasoning capability boundaries of LRMs and the effectiveness of their thinking mechanisms using puzzle tasks with precisely controllable complexity.

**Key Insight**: Four classical logic puzzles are designed (Tower of Hanoi, Checker Swapping, River Crossing, Blocksworld), each supporting continuous complexity scaling from $2^1-1$ to $2^{15}-1$ steps via a parameter $N$.

**Core Idea**: Controlled-complexity puzzles combined with reasoning trajectory analysis expose the "illusion of thinking" in LRMs—hard capability ceilings exist and performance is driven by training distribution rather than true reasoning.

## Method

### Overall Architecture
Four puzzle environments (Tower of Hanoi / Checker Swapping / River Crossing / Blocksworld) are designed, each allowing precise complexity control through a parameter $N$. As $N$ is progressively increased, accuracy and thinking token usage patterns of LRMs and standard LLMs are systematically compared.

### Key Designs

1. **Three-Regime Complexity Behavior**:

    - **Function**: Partitions model behavior into low, moderate, and high complexity regimes.
    - **Mechanism**: At low complexity, LRM $\leq$ standard LLM (overthinking); at moderate complexity, LRM $\gg$ standard LLM (discovery after exploration); at high complexity, both collapse entirely (0%).
    - **Design Motivation**: Challenges the assumption that LRMs are always superior and precisely identifies the complexity range in which "thinking" genuinely helps.

2. **Counterintuitive Thinking Token Pattern**:

    - **Function**: Tracks the relationship between thinking token count and accuracy.
    - **Mechanism**: Thinking token usage increases alongside accuracy at moderate complexity, but decreases approaching the collapse point—indicating that models are "giving up on thinking."
    - **Design Motivation**: Establishes that the reasoning limit of LRMs is a hard ceiling rather than a soft one (the bottleneck is not insufficient token budget but fundamental incapability).

3. **Fine-Grained Reasoning Trajectory Analysis**:

    - **Function**: Uses puzzle simulators to extract all intermediate solution candidates from the reasoning process.
    - **Mechanism**: For easy problems, correct solutions appear in the first third of the trajectory ("overthinking"); for moderate problems, solutions emerge late ("discovery after exploration"); for hard problems, erroneous solutions densely populate the entire trajectory ("complete fixation").
    - **Design Motivation**: Explains the mechanism behind all three failure modes, particularly why overthinking causes LRMs to underperform standard LLMs at low complexity.

### Experimental Setup
25 samples per difficulty level per model; Claude-3.7 uses a maximum budget of 64K tokens. Models evaluated: o3-mini (high/medium), Claude-3.7-Thinking vs. no-thinking, DeepSeek-R1 vs. V3, QwQ-32B vs. Qwen2.5-32B.

## Key Experimental Results

### Main Results (Collapse Thresholds per Model)

| Model | Tower of Hanoi ($N$) | Checkers ($n$) | River Crossing ($n$) | Blocksworld ($n$) |
|------|---------|---------|---------|---------|
| o3-mini (high) | ~9–10 | ~7–8 | ~4–5 | ~5–6 |
| DeepSeek-R1 | ~10–11 | ~8–9 | ~5–6 | ~6–7 |
| **Claude-3.7-Thinking** | **~11–12** | **~9–10** | **~6–7** | **~7–8** |
| Standard LLM (same scale) | ~7–8 | ~5–6 | ~3–4 | ~4–5 |

LRMs delay the collapse threshold by 2–3 levels, yet ultimately collapse completely.

### Ablation Study (Ineffectiveness of Algorithmic Guidance)

| Condition | Hanoi Collapse Point | Improvement |
|------|-----------|------|
| No algorithmic hint | $N=10$–$11$ | Baseline |
| Full algorithm pseudocode provided | $N=10$–$11$ | **No improvement** |
| Step-by-step instructions provided | $N=10$–$11$ | **No improvement** |

### Key Findings
- **Hard Capability Ceiling**: All LRMs exhibit a deterministic collapse threshold that cannot be overcome by increasing thinking tokens.
- **Thinking Paradox**: Models reduce rather than increase thinking tokens before collapse—they "know they cannot succeed yet still attempt."
- **Overthinking**: At low complexity, LRMs find the correct solution early in the trajectory but continue exploring erroneous paths, ultimately producing incorrect answers.
- **Algorithmic Guidance Inefficacy**: Providing the complete solution algorithm does not shift the collapse threshold—the bottleneck lies in symbolic manipulation and step execution, not strategy discovery.
- **Asymmetric Failure Pattern**: The asymmetry in collapse thresholds between Tower of Hanoi and River Crossing (Hanoi requires more steps yet collapses later) suggests that capability is driven by training distribution.

## Highlights & Insights
- **Disruptive Finding**: LRMs collapse completely (0%) at high complexity while simultaneously reducing thinking tokens, demonstrating that the assumption "more thinking = better reasoning" is fundamentally incorrect.
- **Evaluation Paradigm Innovation**: The controlled puzzle environment with continuous complexity gradients and reasoning trajectory analysis establishes a new standard for evaluating reasoning models.
- **High Practical Value**: For model deployment—LRMs cannot be blindly trusted for high-complexity tasks; for researchers—overcoming reasoning limitations requires addressing fundamental deficiencies in symbolic manipulation and self-verification.
- **Fine-Grained Failure Taxonomy**: The three failure modes—overthinking, discovery after exploration, and complete fixation—are accompanied by the first quantitative analysis of reasoning trajectory position distributions.

## Limitations & Future Work
- Only four puzzle tasks are used, covering combinatorial search and constraint satisfaction; findings may not generalize to knowledge-intensive reasoning.
- Black-box API access precludes observation of internal mechanisms (attention, activations, etc.).
- The absence of human baseline data makes it impossible to determine whether collapse thresholds reflect task difficulty for humans.
- Other reasoning architectures (tree search, explicit planners) are not evaluated.
- Sample size of $N=25$ per difficulty level per model may yield insufficient statistical power.

## Related Work & Insights
- **vs. MATH/AIME Benchmarks**: Subject to data contamination risk and do not permit complexity control. This work uses puzzle environments to eliminate contamination.
- **vs. Faith & Fate (Dziri et al.)**: Demonstrates LLM failure on compositional generalization; this paper extends the analysis to LRMs and identifies the same collapse behavior.
- **vs. Ruoss / Valmeekam et al.**: Evaluates o1 with similar puzzles; this paper provides a deeper three-regime analysis and qualitative reasoning trajectory analysis.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic exposure of the three-regime behavior in LRMs and the thinking token paradox, overturning the intuition that "more thinking = better reasoning."
- Experimental Thoroughness: ⭐⭐⭐⭐ Four puzzles × 5+ models × continuous complexity gradients, though human baselines and deeper analysis of open-source models are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear structure, information-dense figures, and candid discussion of limitations.
- Value: ⭐⭐⭐⭐⭐ Provides fundamental insights for both the research and deployment of reasoning models, reshaping our understanding of the "thinking" mechanism.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Topology of Reasoning: Understanding Large Reasoning Models through Reasoning Graph Properties](topology_of_reasoning_understanding_large_reasoning_models_through_reasoning_gra.md)
- [\[NeurIPS 2025\] Controlling Thinking Speed in Reasoning Models](controlling_thinking_speed_in_reasoning_models.md)
- [\[NeurIPS 2025\] Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning](towards_thinking-optimal_scaling_of_test-time_compute_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Does Thinking More Always Help? Mirage of Test-Time Scaling in Reasoning Models](does_thinking_more_always_help_mirage_of_test-time_scaling_in_reasoning_models.md)
- [\[ICML 2026\] Modeling Hierarchical Thinking in Large Reasoning Models](../../ICML2026/llm_reasoning/modeling_hierarchical_thinking_in_large_reasoning_models.md)

</div>

<!-- RELATED:END -->
