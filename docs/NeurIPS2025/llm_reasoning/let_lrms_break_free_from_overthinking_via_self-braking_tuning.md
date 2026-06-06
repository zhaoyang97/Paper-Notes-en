---
title: >-
  [Paper Note] Let LRMs Break Free from Overthinking via Self-Braking Tuning
description: >-
  [NeurIPS 2025][LLM Reasoning][Efficient Reasoning] This paper proposes the Self-Braking Tuning (SBT) framework, which identifies overthinking patterns in reasoning traces and constructs adaptive-length training data to t…
tags:
  - "NeurIPS 2025"
  - "LLM Reasoning"
  - "Efficient Reasoning"
  - "Overthinking"
  - "Self-Braking"
  - "Chain-of-Thought"
  - "Reasoning Efficiency"
date: 2026-05-08
content_hash: dab5f8f60bb14cc2
---

# Let LRMs Break Free from Overthinking via Self-Braking Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2505.14604](https://arxiv.org/abs/2505.14604)  
**Code**: [https://github.com/ZJU-REAL/Self-Braking-Tuning](https://github.com/ZJU-REAL/Self-Braking-Tuning)  
**Area**: LLM Reasoning
**Keywords**: Efficient Reasoning, Overthinking, Self-Braking, Chain-of-Thought, Reasoning Efficiency

## TL;DR

This paper proposes the Self-Braking Tuning (SBT) framework, which identifies overthinking patterns in reasoning traces and constructs adaptive-length training data to teach large reasoning models (LRMs) to autonomously determine when to stop reasoning. SBT reduces token consumption by 30%–60% on mathematical reasoning tasks while maintaining accuracy.

## Background & Motivation

**Background**: Large reasoning models (e.g., OpenAI o1, DeepSeek-R1) improve accuracy by generating detailed multi-step reasoning chains, achieving strong performance on mathematical and logical tasks.

**Limitations of Prior Work**: These models commonly exhibit "overthinking"—continuing to generate extensive redundant reasoning steps even after a correct answer has been reached. This leads to enormous computational overhead (thousands of tokens per problem), increased latency, and potential interference with the final answer from redundant reasoning.

**Key Challenge**: Most existing approaches to mitigating overthinking rely on external intervention mechanisms—RL reward constraints, token budget limits, inference-time truncation—without fully leveraging the model's own capacity to identify redundant reasoning. This external control paradigm increases system complexity and lacks flexibility.

**Goal**: Can LRMs autonomously identify redundant reasoning and stop at the appropriate moment? That is, can a "braking mechanism" be internalized so that the model naturally terminates reasoning upon reaching sufficient confidence, analogous to human reasoning?

**Key Insight**: The authors observe that LRM reasoning traces exhibit a clear structure: a Foundation Solution (the first complete solution attempt) followed by multiple Evolution Solutions (reflections, verifications, and alternative approaches). Overthinking primarily occurs during the Evolution Solution phase. The paper quantifies the turning point of overthinking by combining structural efficiency metrics with linguistic marker metrics.

**Core Idea**: Quantitative metrics are used to locate the onset of redundant reasoning; adaptive-length training data is then constructed with braking prompts, enabling models to learn to autonomously terminate overthinking via SFT.

## Method

### Overall Architecture

The input consists of a batch of mathematical QA data with long reasoning traces (based on OpenR1-Math). The system first analyzes the trace structure, quantifies the redundancy of each reasoning step using an overthink score, and identifies the optimal truncation point. Two strategies (SBT-E / SBT-D) are then used to construct truncated training data. Braking prompts in natural language are inserted at the truncation point, and loss masking is applied to redundant segments. The model is trained via SFT. At inference time, the model spontaneously generates braking statements and terminates its reasoning.

### Key Designs

1. **Structured Analysis of Reasoning Traces**

    - **Function**: Decomposes LRM reasoning traces into a Foundation Solution (the first systematic solution) and multiple Evolution Solutions (subsequent reflections, verifications, and alternative approaches).
    - **Mechanism**: Analysis of reasoning outputs from models such as DeepSeek-R1 reveals that Evolution Solutions typically begin with words such as "Wait," "Alternatively," or "However." On simple problems (e.g., GSM8K), the Foundation Solution already achieves high accuracy (~85%), and subsequent Evolution Solutions consist largely of redundant repeated verification.
    - **Design Motivation**: Understanding the reasoning structure is a prerequisite for precise truncation. Truncating without distinguishing effective reasoning from redundant reasoning risks compromising core problem-solving ability.

2. **Overthinking Quantification Metrics**

    - **Function**: Two complementary metrics quantitatively assess the degree of redundancy in each reasoning trace.
    - **Mechanism**:
        - Reasoning efficiency ratio $\eta_s = FS / TS$ (steps to first correct answer / total steps); values closer to 1 indicate more efficient reasoning.
        - Overthinking marker ratio $\kappa_t$ (density of reflection/verification keywords in the reasoning text); higher values indicate greater redundancy.
        - Combined Overthink Score $= \beta \cdot \kappa_t + (1-\beta) \cdot (1-\eta_s)$, where $\beta = 0.1$.
    - **Design Motivation**: Structural metrics alone cannot capture linguistic redundancy patterns, while linguistic metrics alone are style-sensitive. The choice of $\beta = 0.1$ reflects a design philosophy that weights structural efficiency at 90% and linguistic signals at 10%.

3. **SBT-E (Exact Truncation Strategy)**

    - **Function**: Uniformly retains the Foundation Solution plus one Evolution Solution for each trace, then truncates.
    - **Mechanism**: Retaining two complete solutions teaches the model the pattern of "stop when the same answer is obtained twice." A small amount of masked redundant content (the beginning of the next Evolution Solution) is retained after the truncation point, allowing the model to observe but not learn from redundant patterns.
    - **Design Motivation**: Retaining one Evolution Solution preserves the model's self-correction ability, while obtaining the same answer twice constitutes a natural termination signal.

4. **SBT-D (Dynamic Truncation Strategy)**

    - **Function**: Incrementally analyzes the reasoning trace and dynamically determines the truncation point based on the overthink score at each step.
    - **Mechanism**: Starting from the Foundation Solution, subsequent steps are added incrementally, with the overthink score recomputed after each addition. Truncation occurs when the score exceeds threshold $\tau_1 = 0.2$. Steps with scores between $\tau_1$ and $\tau_2 = \tau_1 + 5\%$ constitute the masked segment.
    - **Design Motivation**: Unlike the fixed truncation of SBT-E, SBT-D adapts to problem difficulty—retaining more reasoning steps for hard problems and truncating earlier for simple ones, more closely mirroring human reasoning behavior.

5. **Braking Prompt Mechanism**

    - **Function**: Inserts a natural-language braking statement at the truncation point, e.g., "Wait, I've gotten the same answer multiple times, time to end the thinking."
    - **Mechanism**: Compared to special tokens (e.g., `<stop_overthinking>`) or no prompt at all, natural-language braking prompts leverage the model's existing semantic understanding without requiring it to learn new control conventions.
    - **Design Motivation**: Experiments demonstrate that natural-language prompts reduce token consumption by 6.4% compared to special tokens. Natural-language prompts provide explicit metacognitive signals, helping the model understand *why* to stop rather than merely *where* to stop.

### Loss & Training

- Standard SFT training using Megatron-LM, 3 epochs, learning rate 1e-5 with cosine decay.
- **Key Design**: Masked Redundant Thinking (MRT)—redundant content retained after the truncation point is excluded from loss computation. This allows the model to observe overthinking patterns without reinforcing them, analogous to negative sample exposure without backpropagation.
- Removing MRT increases token consumption by 37.8%, demonstrating that this design is critical for teaching the model to recognize redundancy.

## Key Experimental Results

### Main Results

| Model | Method | GSM8K Acc | MATH500 Acc | AIME Acc | AMC23 Acc | Avg. Acc | Avg. #Tok |
|-------|--------|-----------|-------------|----------|-----------|----------|-----------|
| Qwen2.5-Math-1.5B | Baseline | 85.00 | 80.25 | 16.25 | 55.94 | 59.36 | 3277 |
| | SBT-E | 84.85 | 77.10 | 13.75 | 55.63 | 57.83 | 1673 (**-49%**) |
| | SBT-D | 84.87 | 77.30 | 14.17 | 50.31 | 56.66 | 1682 (**-49%**) |
| Qwen2.5-Math-7B | Baseline | 96.11 | 92.67 | 40.83 | 83.13 | 78.19 | 6029 |
| | SBT-E | 95.45 | 90.77 | 38.75 | 77.19 | 75.54 | 4178 (**-31%**) |
| | SBT-D | 95.37 | 91.15 | 38.38 | 80.06 | 76.24 | 4643 (**-23%**) |
| Llama-3.1-8B | Baseline | 88.03 | 59.98 | 9.58 | 36.75 | 48.59 | 8576 |
| | SBT-E | 85.03 | 57.60 | 6.84 | 33.44 | 45.73 | 3193 (**-63%**) |
| | SBT-D | 88.27 | 62.60 | 7.70 | 38.12 | 49.17 | 4291 (**-50%**) |

### Ablation Study

| Configuration | Avg. Acc | Avg. #Tok | Notes |
|---------------|----------|-----------|-------|
| Baseline | 59.36 | 3277 | Original SFT |
| SBT-E w/ MRT | 57.83 | 1673 | Full model |
| SBT-E w/o MRT | 58.02 | 2306 | Without masked redundant thinking; token +37.8% |
| Natural language braking | 56.66 | 1682 | Optimal braking approach |
| Special token braking | 56.61 | 1797 | Token +6.4% |
| No braking prompt | 56.39 | 1801 | Acc -0.27%, token +7.1% |
| Step-level detection | 56.66 | 1682 | Superior to token-level |
| Token-level detection | 56.24 | 1753 | Disrupts logical units |

### Key Findings

- **Masked Redundant Thinking is the core design**: Removing MRT increases token consumption by 37.8%, demonstrating that "observing but not learning from redundant patterns" is critical for the model's self-braking ability.
- **SBT-D outperforms on larger models**: On Llama-3.1-8B, SBT-D actually improves MATH500 accuracy (+2.62%), indicating that dynamic truncation can also eliminate harmful overthinking.
- **General-purpose models benefit more than domain-specific models**: Llama-8B achieves 63% token reduction while Qwen-Math-7B achieves only 31%, as specialized models inherently produce more focused reasoning.
- **Threshold 0.2 (identifying ~60% of samples as overthinking) is the optimal operating point.**

## Highlights & Insights

- **The "expose but do not reinforce" training paradigm is elegant**: Loss masking allows the model to observe redundant patterns without gradient updates on them, conceptually similar to negative samples in contrastive learning but more lightweight. This approach is transferable to any scenario requiring models to learn "what not to do" (e.g., hallucination suppression).
- **Structured analysis of reasoning traces**: Automating the Foundation Solution / Evolution Solution distinction provides a standardized tool for subsequent reasoning compression research.
- **Natural language outperforms special tokens as a control signal**: Leveraging the model's existing semantic understanding is more efficient than introducing new symbols—a finding with broad implications for any task requiring model self-regulation.
- **"Repeated answers" as a natural termination signal**: Training the model on two solutions that reach the same answer is a highly natural and interpretable stopping condition.

## Limitations & Future Work

- Validation is limited to mathematical reasoning tasks; overthinking in code generation, logical reasoning, creative writing, and other domains remains unexamined.
- The approach depends on the specific format of OpenR1-Math (`<think>...</think>` tags); generalization to other reasoning formats has not been verified.
- The hyperparameters $\beta = 0.1$ and $\tau_1 = 0.2$ in the Overthink Score may require retuning for different domains or model families.
- Only SFT is employed; combining SBT with RL (which may further enhance self-braking capability) is not explored.
- Training data construction relies on ground-truth answers to determine "first correct," making the approach not directly applicable to open-ended tasks without reference answers.

## Related Work & Insights

- **vs. DEER (inference-time truncation)**: DEER truncates at inference time based on confidence, constituting external control; SBT internalizes the braking capability during training, requiring no additional inference-time logic.
- **vs. Token-Budget / CoD (budget control)**: These methods require a pre-specified token budget, making them difficult to adapt to problems of varying difficulty; SBT enables the model to adaptively determine reasoning length based on the problem.
- **vs. LightThinker (intermediate step compression)**: LightThinker compresses intermediate reasoning content, while SBT directly reduces unnecessary reasoning steps; the two approaches are complementary.

## Rating

- Novelty: ⭐⭐⭐⭐ The self-braking concept is innovative, though the core contribution remains training data engineering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Ablation studies cover all key design choices; 4 model variants × 4 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, in-depth analysis, and rich figures and tables.
- Value: ⭐⭐⭐⭐ Highly practical; the method is simple yet effective, with direct value for LRM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mind the Gap: Bridging Thought Leap for Improved Chain-of-Thought Tuning](mind_the_gap_bridging_thought_leap_for_improved_chain-of-thought_tuning.md)
- [\[NeurIPS 2025\] Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones](let_me_think_a_long_chainofthought_can_be_worth_exponentiall.md)
- [\[NeurIPS 2025\] The Virtues of Brevity: Avoid Overthinking in Parallel Test-Time Reasoning](the_virtues_of_brevity_avoid_overthinking_in_parallel_test-time_reasoning.md)
- [\[NeurIPS 2025\] Scalable Best-of-N Selection for Large Language Models via Self-Certainty](scalable_best-of-n_selection_for_large_language_models_via_self-certainty.md)
- [\[AAAI 2026\] ExtendAttack: Attacking Servers of LRMs via Extending Reasoning](../../AAAI2026/llm_reasoning/extendattack_attacking_servers_of_lrms_via_extending_reasoning.md)

</div>

<!-- RELATED:END -->
