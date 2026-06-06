---
title: >-
  [Paper Note] InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models
description: >-
  [ICLR 2026][Model Compression][long-context reasoning] InftyThink is proposed as a new paradigm that transforms monolithic long-form reasoning into iterative short-form reasoning with intermediate summarization. Without…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "long-context reasoning"
  - "iterative reasoning"
  - "summarization compression"
  - "computational efficiency"
  - "reasoning paradigm"
date: 2026-05-08
content_hash: d0e6ece4349678a4
---

# InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2503.06692](https://arxiv.org/abs/2503.06692)  
**Code**: [Project Page](https://zju-real.github.io/InftyThink)  
**Area**: Model Compression
**Keywords**: long-context reasoning, iterative reasoning, summarization compression, computational efficiency, reasoning paradigm

## TL;DR
InftyThink is proposed as a new paradigm that transforms monolithic long-form reasoning into iterative short-form reasoning with intermediate summarization. Without modifying model architecture, it achieves theoretically unbounded reasoning depth and significantly reduced computational cost, yielding an 11% improvement for Qwen2.5-Math-7B on AIME24.

## Background & Motivation
Reasoning models such as DeepSeek-R1 and o1 achieve strong performance through extended chain-of-thought, yet long-context reasoning faces three fundamental challenges:

**Quadratic computational scaling**: The computational complexity of decoder-based LLMs grows quadratically with sequence length, resulting in prohibitive resource consumption at inference time.

**Context length ceiling**: Reasoning processes are constrained by max_length and are frequently truncated before reaching a conclusion.

**Performance degradation beyond training window**: Most models have pretraining windows of only 4k–8k tokens, and performance degrades noticeably when inference exceeds this range.

Existing solutions (e.g., CoT-Valve for chain compression, TokenSkip for redundant token removal, LightThinker for dynamic compression via special tokens) optimize within the paradigm of single continuous reasoning and do not address the fundamental computational scaling problem.

**Core Idea**: Drawing inspiration from human cognition—decomposing complex problems into manageable segments and summarizing intermediate progress—InftyThink divides monolithic reasoning into multiple bounded segments, generates a summary after each segment, and uses the summary as the context for the next segment, forming a "sawtooth" memory pattern.

## Method

### Overall Architecture
InftyThink decomposes reasoning into multiple iterative rounds: Round 1 generates reasoning passage $RP_1$ and summary $S_1$; subsequent rounds use the previous summary as historical context to generate new reasoning passage $RP_i$ and summary $S_i$; the final round generates reasoning passage $RP_n$ and conclusion $C$.

### Key Designs

1. **Iterative Reasoning with Summarization**:

    - **Function**: Replaces single monolithic reasoning with multiple bounded reasoning rounds.
    - **Mechanism**:
        - First round: `<|U|>Q<|A|><think>RP₁</think><summary>S₁</summary>`
        - Intermediate rounds: `<|U|>Q<|A|><history>Sᵢ₋₁</history><think>RPᵢ</think><summary>Sᵢ</summary>`
        - Final round: `<|U|>Q<|A|><history>Sₙ₋₁</history><think>RPₙ</think>C`
    - **Design Motivation**: Each round maintains a bounded context length (sawtooth memory pattern), theoretically supporting unlimited reasoning depth. Simple problems can reach a conclusion in the first round, naturally degenerating to the conventional paradigm.

2. **Data Reconstruction Pipeline**:

    - **Function**: Converts existing long-reasoning datasets into InftyThink format.
    - **Mechanism**: A three-step pipeline—
        - Step I *Reasoning Segmentation*: Segments are split at semantic boundaries (sentences/paragraphs) based on hyperparameter $\eta$ (maximum segment length).
        - Step II *Summary Generation*: Meta-Llama-3.3-70B-Instruct generates a summary for each segment, conditioned on all preceding segments to maintain reasoning continuity.
        - Step III *Training Instance Construction*: Instances are assembled as multi-turn examples: the first segment yields $(Q, RP_1, S_1)$, intermediate segments yield $(Q, S_{i-1}, RP_i, S_i)$, and the final segment yields $(Q, S_{n-1}, RP_n, C)$.
    - **Design Motivation**: OpenR1-Math (220K samples) is reconstructed into 333K InftyThink-format samples ($\eta$=4k), leveraging existing high-quality reasoning data to avoid generation from scratch.

3. **Inference-Time Execution Mechanism**:

    - **Function**: At inference time, the model iteratively generates reasoning segments and summaries until a conclusion is produced.
    - **Mechanism**: The output of each round is parsed, and the summary becomes the context for the next round. A max_iters=10 limit prevents infinite loops; experiments show that well-trained models naturally converge within a reasonable number of iterations.
    - **Design Motivation**: No architectural modifications are required; any decoder-only model can adopt this framework.

### Loss & Training
Instruction fine-tuning is applied on OpenR1-Math-Inf (InftyThink format) to train multiple base models, with $\eta$ = 4k and max_iters = 10.

## Key Experimental Results

### Main Results (base models, pass@16, temperature=0.7)

| Model | Format | MATH500 ACC | AIME24 ACC | GPQA ACC | Avg ACC |
|------|------|------------|-----------|---------|---------|
| Qwen2.5-Math-1.5B | Vanilla | 75.24 | 16.04 | 26.48 | 59.54 |
| Qwen2.5-Math-1.5B | InftyThink | **79.57** | **26.04** | **35.89** | **65.48** |
| Qwen2.5-Math-7B | Vanilla | 89.51 | 32.92 | 43.94 | 74.78 |
| Qwen2.5-Math-7B | InftyThink | **91.29** | **43.96** | **52.97** | **78.92** |
| Llama-3.1-8B | Vanilla | 82.10 | 20.83 | 41.35 | 68.49 |
| Llama-3.1-8B | InftyThink | **82.28** | **34.17** | **47.51** | **70.84** |

### Latency Comparison (inference time)

| Model | MATH500 Latency Vanilla→InftyThink | AIME24 Latency |
|------|------|------|
| Qwen2.5-Math-7B | 1.26s→0.76s | 4.15s→4.66s |
| Qwen2.5-14B | 1.49s→1.43s | 11.30s→7.11s |

### Key Findings
- Qwen2.5-Math-7B achieves an 11% improvement on AIME24 (32.92→43.96) and a 9% improvement on GPQA (43.94→52.97).
- Smaller models (1.5B) benefit more: +10% on AIME24 and +9.4% on GPQA.
- MATH500 latency decreases from 1.26s to 0.76s, demonstrating a significantly smaller area under the compute curve.
- For larger models (14B/32B), accuracy gains from InftyThink become more marginal, though latency benefits remain substantial.
- The scale of the summarization model has limited impact on final performance (70B vs. smaller models show minimal differences).

## Highlights & Insights
- The "sawtooth memory pattern" concept is intuitive yet powerful—periodic compression keeps computational complexity tractable.
- No architectural modifications or specialized training infrastructure are required; significant gains are achievable through data reconstruction and SFT alone.
- The work challenges the assumption that reasoning depth and computational efficiency must be traded off against each other—both can be improved simultaneously.

## Limitations & Future Work
- The impact of summary quality on reasoning correctness lacks systematic analysis; information loss may accumulate over long reasoning chains.
- The segment length $\eta$ is fixed at 4K; dynamic adjustment could be more effective (simple segments may not need 4K, while difficult ones may require more).
- The method relies on SFT; combining it with RL (e.g., GRPO) may unlock greater potential.
- The reliability of multi-round summarization may differ between numerical and linguistic reasoning tasks.

## Related Work & Insights
- **vs CoT-Valve**: CoT-Valve requires a predefined compression ratio, whereas InftyThink adaptively determines when to terminate.
- **vs LightThinker**: LightThinker compresses reasoning into implicit representations, while InftyThink preserves textual interpretability.
- **vs TokenSkip**: Token deletion in TokenSkip incurs reasoning performance loss, whereas InftyThink retains critical information through summarization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The iterative reasoning paradigm is simple yet effective, with a clear conceptual contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 5 base models, multiple benchmarks, latency analysis, and rich ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Excellent figures; the sawtooth comparison diagram is intuitive and easy to understand.
- **Value**: ⭐⭐⭐⭐⭐ Highly practical and directly applicable to existing models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Is Finer Better? The Limits of Microscaling Formats in Large Language Models](is_finer_better_the_limits_of_microscaling_formats_in_large_language_models.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[ICLR 2026\] Scaling Reasoning Hop Exposes Weaknesses: Demystifying and Improving Hop Generalization in Large Language Models](scaling_reasoning_hop_exposes_weaknesses_demystifying_and_improving_hop_generali.md)
- [\[ICLR 2026\] BeyondBench: Contamination-Resistant Evaluation of Reasoning in Language Models](beyondbench_contamination-resistant_evaluation_of_reasoning_in_language_models.md)
- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](../../ACL2026/model_compression/lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)

</div>

<!-- RELATED:END -->
