---
title: >-
  [Paper Note] VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice
description: >-
  [CVPR 2026][Video Understanding][Video Reasoning] This paper proposes VideoAuto-R1, an on-demand reasoning framework for video understanding. During training, it adopts a "think once…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Video Reasoning"
  - "Adaptive Thinking"
  - "Chain-of-Thought"
  - "Reinforcement Learning"
  - "Inference Efficiency"
date: 2026-05-08
content_hash: c16e3303241fde77
---

# VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice

**Conference**: CVPR 2026
**arXiv**: [2601.05175](https://arxiv.org/abs/2601.05175)
**Code**: [https://ivul-kaust.github.io/projects/videoauto-r1](https://ivul-kaust.github.io/projects/videoauto-r1)
**Area**: Video Understanding / LLM Reasoning
**Keywords**: Video Reasoning, Adaptive Thinking, Chain-of-Thought, Reinforcement Learning, Inference Efficiency

## TL;DR
This paper proposes VideoAuto-R1, an on-demand reasoning framework for video understanding. During training, it adopts a "think once, answer twice" (answer→think→answer) paradigm; during inference, it uses the confidence of the first answer to determine whether to invoke CoT reasoning. The approach maintains SOTA accuracy while reducing average response length from 149 to 44 tokens (approximately 3.3× compression).

## Background & Motivation

1. **Background**: Chain-of-Thought (CoT) reasoning has become the dominant approach for enhancing video understanding in multimodal large language models. Models such as Video-R1, Time-R1, and VideoChat-R1 are trained via GRPO-based reinforcement learning to perform step-by-step reasoning before answering. These methods have proven highly effective on symbolic tasks such as mathematics and programming.

2. **Limitations of Prior Work**: (a) Video understanding is inherently more dependent on visual perception than step-by-step reasoning; once perception is accurate, subsequent symbolic reasoning tends to be shallow. (b) Forcing all samples to undergo CoT reasoning results in substantial redundant tokens (Video-R1 averages 386 tokens), significantly increasing latency and inference cost. (c) Surprisingly, for RL-trained video reasoning models, direct answering matches or even outperforms CoT on multiple benchmarks.

3. **Key Challenge**: CoT reasoning incurs computational overhead with limited benefit in video understanding. It is redundant or even detrimental on perception-intensive tasks (e.g., object/action recognition), and provides meaningful gains only on a small subset of tasks requiring multi-step derivation (e.g., physics/math reasoning in VideoMMMU).

4. **Goal**: To design a video understanding model capable of adaptively deciding whether reasoning is needed—answering simple questions directly and invoking CoT only for complex ones.

5. **Key Insight**: The authors systematically demonstrate the performance gap between direct answering and CoT mode in existing video reasoning models (Video-R1, Time-R1, VideoChat-R1) via Table 1, finding that CoT even degrades accuracy on VideoMME and LongVideoBench. This observation provides strong motivation for on-demand reasoning.

6. **Core Idea**: During training, the model is trained to generate both a direct answer and a reasoning-augmented answer (dual-answer GRPO). During inference, the token-level confidence of the first answer determines whether to continue generating a reasoning chain, enabling adaptive auto-thinking.

## Method

### Overall Architecture
**Training phase**: Given a question, the model generates a response in the format `\boxed{a1}<think>r</think>\boxed{a2}`, where $a_1$ is the initial answer, $r$ is the reasoning process, and $a_2$ is the reviewed answer. Both answers are supervised by verifiable rewards. **Inference phase**: The model first decodes up to $a_1$ and computes its token-level confidence; if the confidence exceeds threshold $\tau$, decoding terminates early; otherwise, the reasoning chain and $a_2$ are generated.

### Key Designs

1. **"Think Once, Answer Twice" Training Paradigm**:
    - **Function**: Enables the model to simultaneously learn direct answering and reasoning-augmented answering within a single generation pass.
    - **Mechanism**: Unlike conventional auto-thinking methods that require per-sample labels indicating whether thinking is needed, this approach always generates two answers. The system prompt instructs the model to first output an initial answer (without analysis), then reason inside `<think>`, and finally output a reviewed answer. If the model cannot answer without reasoning, it is allowed to output a fallback string "Let's analyze the problem step by step" in the first box.
    - **Design Motivation**: This eliminates the need for per-sample think/no-think annotations and avoids training mode collapse (always think or never think). The model only needs to learn to make both answers correct.

2. **Dual-Answer Reward GRPO Training**:
    - **Function**: Simultaneously incentivizes the correctness of both the direct answer and the reasoning-augmented answer.
    - **Mechanism**: The total reward is $R = w_1 R_{\text{task}}^{(1)}(a_1) + w_2 R_{\text{task}}^{(2)}(a_2) + \lambda R_{\text{fmt}} + \alpha R_{\text{fallback}}$, where $w_2 > w_1$ (specifically $w_1=0.9, w_2=1.1$), assigning higher weight to the reviewed answer to encourage improvement through reasoning. $R_{\text{fallback}}$ provides additional reward when $a_1$ uses the fallback string but $a_2$ is correct, preventing low-confidence guessing on hard problems. GRPO uses 16 rollouts at temperature 1.0.
    - **Design Motivation**: The higher $w_2$ ensures that the behavior of improving answers through reasoning is rewarded; meanwhile $w_1 > 0$ ensures the initial answer is also trained, making early exit effective. The fallback mechanism handles cases in math/symbol-intensive problems where intuitive answering is infeasible.

3. **Confidence-Based Early Exit Inference Strategy**:
    - **Function**: Adaptively decides at inference time whether to continue generating CoT.
    - **Mechanism**: After decoding the first `\boxed{a1}`, the length-normalized average log probability of the answer tokens is computed as the confidence score: $s(a_1) = \frac{1}{L}\sum_{\ell=1}^L \log p_\theta(t_\ell | t_{<\ell}, q)$. If $s(a_1) \geq \log \tau$ ($\tau=0.97$), $a_1$ is accepted and decoding terminates; otherwise, the reasoning chain and $a_2$ are generated. The confidence of the fallback string is set to $-\infty$, forcing continuation.
    - **Design Motivation**: Token-level confidence is strongly correlated with answer correctness (validated in Table 9), enabling precise identification of samples requiring reasoning. Since $a_1$ typically contains no more than 10 tokens, confidence computation incurs near-zero overhead. This approach fully decouples the training objective (learning dual answers) from the inference strategy (when to think).

### Loss & Training
- **Base models**: Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct
- Direct RL without cold-start SFT (experiments show that SFT on Video-R1-CoT data degrades baseline performance)
- **Training data**: 83K samples comprising text/image math-science problems, video QA, and temporal grounding
- Visual encoder frozen; only the projector and LLM are trained
- Training on 32 H100 GPUs for approximately 35 hours
- Inference: greedy decoding, maximum response length 4096 tokens, $\tau=0.97$

## Key Experimental Results

### Main Results (Video QA)

| Model | Inference Mode | Response Length | VideoMME | MVBench | VideoMMMU | MVP |
|-------|---------------|----------------|----------|---------|-----------|-----|
| Qwen2.5-VL-7B | Direct | 3.0 | 66.0 | 67.1 | 54.7 | 36.5 |
| Video-R1 | Think-Only | 386 | 61.8 | 65.5 | 51.4 | 33.0 |
| VideoChat-R1.5 | Think-Only | 133 | 65.2 | 70.6 | 49.6 | 38.6 |
| **VideoAuto-R1 (2.5VL)** | **AutoThink** | **44** | **67.3** | **71.0** | **58.6** | **39.4** |
| **VideoAuto-R1 (Q3VL)** | **AutoThink** | **52** | **71.7** | **72.0** | **65.0** | **43.0** |

### Temporal Grounding Results

| Model | Charades-STA mIoU | ActivityNet mIoU | NExT-GQA Acc |
|-------|------------------|-----------------|-------------|
| Qwen2.5-VL-7B | 52.9 | 26.9 | 53.3 |
| Time-R1 | 58.8 | 52.1 | - |
| VideoChat-R1.5 | 60.6 | 35.3 | - |
| **VideoAuto-R1 (2.5VL)** | **60.0** | **47.6** | **80.6** |
| **VideoAuto-R1 (Q3VL)** | **63.7** | **56.1** | **82.6** |

### Key Findings
- **Adaptive think ratio**: The think ratio is only 25% on the perception-oriented MVBench benchmark, rising to 51% on the reasoning-intensive VideoMMMU, demonstrating that the model genuinely learns on-demand reasoning. The Qwen3-VL variant reaches a think ratio of 53% on VideoMMMU.
- **Counter-intuitive finding on direct answering vs. CoT**: Existing video reasoning models (Video-R1, Time-R1, VideoChat-R1) exhibit accuracy drops of 1–2 points under CoT reasoning on VideoMME and LongVideoBench; CoT consistently outperforms direct answering only on VideoMMMU.
- **Temporal grounding does not require CoT**: On Charades-STA and ActivityNet, the initial boxed answer is already sufficiently precise; subsequent CoT primarily serves as explanation, resulting in default early exit.
- **3.3× efficiency gain**: Average response length is reduced from 386 tokens (Video-R1) to 44 tokens (Qwen2.5-VL variant), significantly reducing inference latency.
- **Direct RL without cold-start SFT is superior**: Early experiments show that SFT on Video-R1-CoT data degrades baseline performance; direct RL is more stable.

## Highlights & Insights
- **The "answer→think→answer" template is the core innovation**: By having the model produce two answers within a single generation, this approach elegantly resolves the challenge of "how to annotate per-sample think/no-think labels during training" in auto-thinking. No additional switch tokens, mode heads, or cold-start SFT are required, making training extremely straightforward. This design is transferable to any scenario requiring adaptive reasoning.
- **Confidence-based early exit is simple yet effective**: No additional classifier needs to be trained to decide whether to reason; the model's own token log probabilities are directly leveraged at near-zero cost. This idea is broadly applicable to inference efficiency optimization in any LLM.
- **Counter-intuitive findings on CoT in video understanding**: The paper systematically demonstrates that CoT is unhelpful or even harmful for most perception-oriented video tasks, an insight that warrants attention across the broader community—not all tasks require System 2 thinking.

## Limitations & Future Work
- **Fixed threshold $\tau$**: A single fixed threshold $\tau=0.97$ is applied universally across all benchmarks, which may not be optimal for all task types. Dynamic threshold adjustment could potentially yield further improvements.
- **Dual-answer training increases token consumption**: Each GRPO rollout during training requires generating a full answer-think-answer sequence, resulting in higher training token consumption than training on direct answers alone.
- **The fallback mechanism is relatively simple**: The current fallback string is fixed text; more flexible fallback strategies (e.g., progressive reasoning depth) may offer additional gains.
- **Validation limited to Qwen2.5-VL/Qwen3-VL**: Generalizability to other video LLM architectures remains unclear.

## Related Work & Insights
- **vs. Video-R1**: Video-R1 forces CoT on all samples, averaging 386 tokens, achieving 61.8% on VideoMME; VideoAuto-R1 achieves 67.3% with only 44 tokens, winning on both efficiency and accuracy.
- **vs. AdaptThink**: AdaptThink trains a binary mode-switching strategy for text-based math tasks, requiring balanced think/no-think data. VideoAuto-R1 circumvents this difficulty through the dual-answer paradigm and is more stable.
- **vs. R-4B (image-domain auto-thinking)**: R-4B employs a dual-mode training strategy (SFT initialization + RL fine-tuning); this paper requires no SFT initialization whatsoever, making it considerably simpler.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The "think once, answer twice" paradigm represents a genuinely novel auto-thinking design that eliminates the need for mode annotation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers video QA, temporal grounding, and image reasoning with detailed ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is rigorously argued (counter-intuitive findings in Table 1); method descriptions are clear.
- **Value**: ⭐⭐⭐⭐⭐ Achieves simultaneous breakthroughs in efficiency and accuracy; the auto-thinking paradigm is broadly reusable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning](../../NeurIPS2025/video_understanding/when_thinking_drifts_evidential_grounding_for_robust_video_reasoning.md)
- [\[CVPR 2026\] LongVideo-R1: Smart Navigation for Low-cost Long Video Understanding](longvideo-r1_smart_navigation_for_low-cost_long_video_understanding.md)
- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICLR 2026\] A.I.R.: Adaptive, Iterative, and Reasoning-based Frame Selection For Video Question Answering](../../ICLR2026/video_understanding/air_enabling_adaptive_iterative_and_reasoning-based_frame_selection_for_video_qu.md)
- [\[CVPR 2026\] MovieRecapsQA: A Multimodal Open-Ended Video Question-Answering Benchmark](movierecapsqa_a_multimodal_open-ended_video_question-answering_benchmark.md)

</div>

<!-- RELATED:END -->
