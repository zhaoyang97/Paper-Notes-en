---
title: >-
  [Paper Note] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning
description: >-
  [NeurIPS 2025][Multimodal VLM][Reinforcement Fine-Tuning] This paper systematically investigates whether explicit thinking is necessary in rule-based reinforcement fine-tuning (RFT). It finds that on visual perception tasks, No-Thinking-RFT consistently outperforms the conventional think-then-answer paradigm, and proposes an Adaptive-Thinking approach that allows models to autonomously determine whether to reason based on their own capability and task complexity.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - Reinforcement Fine-Tuning
  - Chain-of-Thought
  - Multimodal Large Language Models
  - GRPO
  - Visual Reasoning
date: 2026-05-08
content_hash: 6ddf79ad229a3604
---

# To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2503.16188](https://arxiv.org/abs/2503.16188)
**Code**: [https://github.com/minglllli/CLS-RL](https://github.com/minglllli/CLS-RL)
**Area**: Multimodal VLM
**Keywords**: Reinforcement Fine-Tuning, Chain-of-Thought, Multimodal Large Language Models, GRPO, Visual Reasoning

## TL;DR

This paper systematically investigates whether explicit thinking is necessary in rule-based reinforcement fine-tuning (RFT). It finds that on visual perception tasks, No-Thinking-RFT consistently outperforms the conventional think-then-answer paradigm, and proposes an Adaptive-Thinking approach that allows models to autonomously determine whether to reason based on their own capability and task complexity.

## Background & Motivation

Since DeepSeek-R1, rule-based RFT has achieved remarkable success in LLMs, with its core mechanism being the encouragement of explicit chain-of-thought (CoT) reasoning prior to answering—widely regarded as the key to RFT's effectiveness. Many multimodal RFT works have attempted to replicate the "gradually increasing thinking length" and "aha moment" phenomena observed in R1.

However, a critical and overlooked question remains: **Is explicit thinking always necessary and beneficial in RFT?** Prior work has shown that reasoning yields limited gains on commonsense tasks and that overthinking can even degrade performance. Yet these findings focus exclusively on the inference stage; the effect of explicit thinking during training has not been systematically explored. Furthermore, since RFT requires generating multiple lengthy responses, its fine-tuning time and GPU memory consumption far exceed those of SFT, making the efficiency of the thinking process an increasingly pressing concern.

The paper's **Starting Point** is the observation that, in classification tasks, response length drops sharply at a specific training step (rather than gradually increasing) while accuracy improves significantly—suggesting that thinking is unnecessary for certain visual tasks. Motivated by this insight, the authors propose No-Thinking-RFT and systematically compare four thinking strategies.

## Method

### Overall Architecture

The paper systematically compares four RFT thinking strategies—Thinking-RFT, No-Thinking-RFT, Think-After-Answer, and Adaptive-Thinking—all optimized via the GRPO (Group Relative Policy Optimization) algorithm. All methods follow the R1-zero training paradigm, applying RL directly to the base model without SFT.

### Key Designs

1. **Thinking-RFT (Baseline)**: Adopts the standard R1 thinking paradigm. The prompt instructs the model to produce its reasoning within `<think>` tags and the final answer within `<answer>` tags. The reward function comprises a format reward ($R_{\text{format}}$) and an accuracy reward ($R_{\text{accuracy}}$), both binary (0/1).

2. **No-Thinking-RFT**: The core contribution. The prompt instructs the model to output the answer directly, completely suppressing any thinking process. The format reward is **removed**; only a strict exact-match accuracy reward is used—the model receives 1 only when its output exactly matches the label. This design compels the model to bypass any intermediate reasoning and produce answers directly, substantially reducing both training and inference time.

3. **Think-After-Answer**: Designed to test the hypothesis that explicit thinking placed before a verifiable answer impedes learning. The model is required to answer first and then produce a brief reasoning trace. This conditions the thinking on the answer rather than the answer on the thinking, thereby mitigating the negative effect of explicit pre-answer reasoning on verifiable-answer tasks.

4. **Adaptive-Thinking**: Allows the model to autonomously decide whether to think. The prompt asks the model to first judge whether the question requires reasoning; if so, it outputs a thinking process before answering; otherwise, it answers directly. The format reward is 1 regardless of which format is chosen. Experiments show that the model ultimately converges to a single strategy (either always thinking or never thinking), and this strategy corresponds to the optimal choice.

### Loss & Training

All methods are optimized with the GRPO algorithm using binary reward functions (format + accuracy, or accuracy only). The learning rate is uniformly set to $1 \times 10^{-6}$, with $\beta = 0.04$ and temperature 1.0. The number of rollouts (4–8 for 2B models, 4 for 7B models) and training epochs vary across tasks.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | No-Thinking-RFT | Thinking-RFT | Gain |
|---|---|---|---|---|
| CVBench (2B) | Overall Acc | **76.76** | 70.36 | +6.40 |
| CVBench (7B) | Overall Acc | 80.67 | **80.36** | +0.31 |
| PuzzleVQA (2B) | Acc | **70.85** | 52.50 | +18.35 |
| PuzzleVQA (7B) | Acc | 80.65 | 66.60 | +14.05 |
| MathVista (2B) | Overall Acc | **48.80** | 44.90 | +3.90 |
| MathVista (7B) | Overall Acc | 59.10 | **64.60** | −5.50 |

Training time comparison (CVBench, 2B): No-Thinking-RFT requires only 139 minutes versus 599 minutes for Thinking-RFT—a **4.3× speedup**.

### Ablation Study

| Configuration | CVBench (2B) | PuzzleVQA (2B) | Note |
|---|---|---|---|
| Thinking-RFT | 70.36 | 52.50 | Full thinking |
| Think-After-Answer | 73.65 | 64.70 | Post-answer thinking; faster convergence |
| No-Thinking-RFT | **76.76** | **70.85** | No thinking; best overall |
| Adaptive-Thinking | 77.03 | 75.45 | Adaptive; converges to no-thinking |
| Thinking-RFT + empty think | — | — | Partial improvement; still behind No-Thinking |

### Key Findings

- **Finding 1**: Weaker small models (2B) under Thinking-RFT tend to generate trivial reasoning traces, resulting in lower performance than No-Thinking-RFT.
- **Finding 2**: Visual perception and puzzle tasks do not benefit from explicit thinking—No-Thinking-RFT matches or exceeds Thinking-RFT across all model sizes.
- **Finding 3**: In Thinking-RFT, inconsistencies arise between the content in `<think>` and `<answer>` tags; inconsistent responses exhibit lower accuracy than the overall average.
- **Finding 4**: The gains of No-Thinking-RFT stem from two sources: better learning during training and avoidance of overthinking at inference time.
- **Finding 5**: Placing explicit thinking before the answer leads to slower reward convergence and lower final performance.
- **Finding 6**: On mathematical tasks, the 2B model converges to a no-thinking strategy while the 7B model converges to a thinking strategy, demonstrating that models can adaptively determine the optimal approach based on their capacity.

## Highlights & Insights

- The paper challenges the prevailing assumption that explicit thinking is indispensable in RFT, providing solid empirical evidence to the contrary.
- No-Thinking-RFT is remarkably simple (relying solely on exact-match accuracy rewards) yet achieves superior performance on multiple tasks while delivering several-fold improvements in training efficiency.
- Adaptive-Thinking demonstrates the potential for models to autonomously learn when to think, with the emergent strategy consistently aligning with the optimal choice.
- Analysis of parameter changes reveals differential effects of distinct thinking strategies on the distribution of model weights.

## Limitations & Future Work

- Adaptive-Thinking currently converges to a uniform strategy at the task level and has not yet achieved question-level adaptivity.
- Experiments are primarily conducted on 2B–7B models; the behavior of larger-scale models remains to be investigated.
- A thorough theoretical explanation for why trivial reasoning emerges under Thinking-RFT is still lacking.

## Related Work & Insights

- **vs. DeepSeek-R1**: R1 demonstrates gradually increasing thinking length on mathematical tasks. This paper observes an abrupt length reduction in visual classification tasks, indicating that different tasks have fundamentally different demands for explicit reasoning.
- **vs. SFT**: Thinking-RFT substantially outperforms SFT on classification tasks and exhibits cross-dataset transfer—RFT on one dataset improves performance on others.
- **vs. inference-stage CoT studies**: This paper extends the question of "whether thinking is beneficial" from the inference stage to the training stage, offering a more comprehensive perspective.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically questions the necessity of thinking in RFT; the angle is novel, though the methodology itself is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive comparisons across six tasks, multiple model scales, and four thinking strategies, yielding rich findings.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with progressively deepening findings and persuasive experimental design.
- Value: ⭐⭐⭐⭐⭐ Highly instructive for the multimodal RFT community; effectively challenges the trend of blindly adopting chain-of-thought reasoning.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [\[ICCV 2025\] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding](../../ICCV2025/multimodal_vlm/docthinker_explainable_multimodal_large_language_models_with.md)
- [\[NeurIPS 2025\] Visual Instruction Bottleneck Tuning](visual_instruction_bottleneck_tuning.md)
- [\[CVPR 2026\] When to Think and When to Look: Uncertainty-Guided Lookback](../../CVPR2026/multimodal_vlm/when_to_think_and_when_to_look_uncertainty-guided_lookback.md)
- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)

<!-- RELATED:END -->
