---
title: >-
  [Paper Note] When Thinking Drifts: Evidential Grounding for Robust Video Reasoning
description: >-
  [NeurIPS 2025][Video Understanding][Video Reasoning] This paper systematically identifies the "visual thinking drift" phenomenon in which CoT reasoning frequently degrades performance in video understanding…
tags:
  - "NeurIPS 2025"
  - "Video Understanding"
  - "Video Reasoning"
  - "Chain-of-Thought"
  - "Visual Thinking Drift"
  - "Reinforcement Learning"
  - "Visual Evidence Reward"
date: 2026-05-08
content_hash: b5ef29011df4a6a6
---

# When Thinking Drifts: Evidential Grounding for Robust Video Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2510.06077](https://arxiv.org/abs/2510.06077)  
**Code**: Not available  
**Area**: Video Understanding
**Keywords**: Video Reasoning, Chain-of-Thought, Visual Thinking Drift, Reinforcement Learning, Visual Evidence Reward

## TL;DR

This paper systematically identifies the "visual thinking drift" phenomenon in which CoT reasoning frequently degrades performance in video understanding, and proposes the Visual Evidence Reward (VER) reinforcement learning framework that corrects this problem by explicitly rewarding reasoning chains grounded in visual evidence.

## Background & Motivation

### CoT Is Harmful for Video Reasoning

Chain-of-Thought (CoT) reasoning excels in text-based tasks, yet its direct application to video understanding often proves counterproductive. Through a systematic study spanning 10 video benchmarks and multiple MLLMs, the authors identify a counterintuitive phenomenon: **prompting models to "think before answering" actually reduces accuracy**, particularly for open-source models such as Qwen2.5-VL and Video-R1.

Further analysis across the 20 sub-tasks of MVBench reveals that CoT causes the greatest harm on tasks requiring rapid visual judgment (e.g., scene-transition detection), where additional reasoning tokens introduce over-rationalization and hallucinated details. CoT remains beneficial on tasks requiring multi-step logic (e.g., counting object motion).

### Visual Thinking Drift

The authors characterize this failure mode as **visual thinking drift**: the model's reasoning chain appears logically coherent yet has in fact detached from the video content, constructing arguments based on hallucinated visual details or temporally incomplete information. The longer the reasoning chain, the higher the probability of error.

### Bayesian Interpretation

From a Bayesian perspective, the generation of a reasoning chain can be decomposed as:

$$p(c_{1:T}, a \mid q, \mathbf{v}) = p(a \mid c_{1:T}, q, \mathbf{v}) \prod_{t=1}^{T} p(c_t \mid c_{<t}, q, \mathbf{v})$$

At each generation step, the visual signal and language prior jointly influence the softmax distribution:

$$p(c_t \mid c_{<t}, q, \mathbf{v}) \propto \exp(\underbrace{\mathbf{h}_{c_{<t}}^\top W_{\text{lang}}}_{\text{language prior}} + \underbrace{\mathbf{h}_{\mathbf{v}}^\top W_{\text{vis}}}_{\text{visual likelihood}})$$

The core issue is that $\|W_{\text{lang}}\| \gg \|W_{\text{vis}}\|$: as the reasoning chain grows, self-attention increasingly concentrates on already-generated text tokens, diluting the visual signal. If each step is correct with probability $1-\varepsilon$, the probability that the entire chain is correct is $(1-\varepsilon)^T \approx 1-T\varepsilon$, so the failure rate grows linearly with chain length. Once an early token introduces a hallucinated fact, all subsequent reasoning is built on a false foundation, and autoregressive decoding provides no mechanism for backtracking and verification.

## Method

### Overall Architecture

Video-VER adopts a two-stage training pipeline: SFT cold-start to establish reasoning capability, followed by GRPO reinforcement learning optimized with the Visual Evidence Reward.

### Key Designs

#### 1. Visual Evidence Reward (VER)

**Core Idea**: Rather than rewarding correct answers alone, VER explicitly rewards responses whose reasoning process cites genuine visual evidence.

For each question $q$, the policy model $\pi_\theta$ generates a group of $G$ responses $\{o_i\}_{i=1}^G$. An auxiliary LLM judge (Llama-3.1-70B-Instruct) evaluates whether each response cites visual evidence, producing a binary score $e_i \in \{0, 1\}$. The evidence-augmented reward is:

$$r_i^{\text{evid}} = \begin{cases} r_i + \alpha & \text{if } o_i \text{ is correct and } e_i = 1 \\ r_i & \text{otherwise} \end{cases}$$

where $\alpha = 0.3$ is the evidence reward weight. The normalized advantage is:

$$A_i = \frac{r_i^{\text{evid}} - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

#### 2. GRPO-Based Policy Optimization

The clipped GRPO objective is:

$$\mathcal{J}_{\text{evid-GRPO}}(\theta) = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{\text{old}}}(o_i|q)}A_i, \text{clip}(\cdot, 1-\epsilon, 1+\epsilon)A_i\right) - \beta \mathbb{D}_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)\right]$$

The reward system comprises four components: accuracy reward, visual evidence reward ($\alpha=0.3$), format reward, and length reward (target 320–512 tokens).

#### 3. Visual Evidence Generation (Inverted Prompting)

**Design Motivation**: Generic video captions lack the visual granularity required to answer specific questions.

An **inverted prompting strategy** is employed: Qwen2.5-VL-72B is provided with a (video, question, correct answer) triple and prompted to generate a list of visual observations that support the given answer. Compared to standard CoT, which simultaneously explores reasoning paths and final answers, inverted prompting samples from the lower-entropy distribution $p(e_{1:K} \mid q, a, \mathbf{v})$, naturally prioritizing visual grounding over language priors. The external VLM is used only for offline training data generation and is not required at inference time.

### Loss & Training

- **Stage 1 (SFT)**: Fine-tuning on the Video-R1-COT-165k dataset to establish basic reasoning capability.
- **Stage 2 (GRPO + VER)**: Reinforcement learning on a mixed dataset of Reversed-in-Time and Video-R1-260k, with GRPO group size $G=8$, trained for 2,000 iterations on 8×H200 GPUs.

## Key Experimental Results

### Main Results

| Benchmark | Qwen2.5-VL-7B (DA) | Qwen2.5-VL-7B (CoT) | Video-VER (CoT) | Gain (vs CoT) |
|---|---|---|---|---|
| MVBench | 63.6 | 59.8 | **64.1** | +4.3 |
| Video-MME | 59.2 | 54.7 | **59.3** | +4.6 |
| VideoMMMU | 47.3 | 47.8 | **52.7** | +4.9 |
| MMVU | 64.2 | 60.5 | **65.1** | +4.6 |
| VideoHallucer | 51.8 | 44.1 | **53.1** | +9.0 |
| EventHallusion | 64.5 | 67.3 | **70.0** | +2.7 |
| TempCompass | 73.7/52.2 | 71.3/49.9 | **74.0/52.8** | +2.7/+2.9 |

Video-VER ranks first on 9 of 10 benchmarks, achieving an average improvement of +4.0% over the base model's CoT.

### Ablation Study

| Visual Evidence Type | MVBench | Video-MME | VideoMMMU | MMVU | Avg. |
|---|---|---|---|---|---|
| Question-Dependent VE (QD-VE) | **64.1** | **59.3** | **52.7** | **65.1** | — |
| Generic Video Caption (VC) | 63.9 | 58.7 | 52.2 | 64.8 | — |

| Frames | MVBench | Video-MME | VideoMMMU | MMVU |
|---|---|---|---|---|
| 32 frames | **64.1** | **59.3** | **52.7** | **65.1** |
| 16 frames | 63.2 | 56.0 | 50.0 | 64.8 |
| 8 frames | 60.5 | 53.3 | 45.4 | 63.2 |

### Key Findings

1. Question-aligned visual evidence outperforms generic captions on 9 of 10 benchmarks, confirming the importance of question-conditioned alignment.
2. More frames (32) yield the best results on 8/10 benchmarks, demonstrating that the method can effectively exploit longer temporal contexts.
3. A non-trivial proportion of GPT-4o questions are also better answered directly than with CoT, indicating that visual thinking drift is a general problem.
4. Majority voting (20 samples) substantially improves CoT performance but incurs significant computational overhead.

## Highlights & Insights

1. **Valuable phenomenon definition**: This work is the first to systematically define and analyze "visual thinking drift," providing a Bayesian theoretical framework for understanding MLLM failure in video reasoning.
2. **Elegant inverted prompting**: By fixing the answer and generating evidence conditioned on it, the high-entropy CoT generation problem is recast as a low-entropy evidence retrieval problem.
3. **Lightweight and efficient**: The method requires only an auxiliary LLM judge and offline evidence generation, with no architectural modifications, making it directly applicable to existing MLLMs.
4. **Largest gain on hallucination detection**: The +9.0% absolute improvement on VideoHallucer directly validates the effectiveness of VER in suppressing hallucinations.

## Limitations & Future Work

- When frame sampling is incomplete (missing critical frames), VER cannot compensate; visual representation quality is a prerequisite.
- The capability of the LLM judge (Llama-3.1-70B) inherently limits reward signal quality.
- Visual evidence generation relies on Qwen2.5-VL-72B; although used only for offline training, the associated cost is non-trivial.
- Validation is primarily conducted on medium-length videos; scenarios with sparse key information in long videos remain underexplored.
- Evaluation is limited to closed-form tasks (MCQ); extension to open-ended QA is an important future direction.

## Related Work & Insights

- **Video Reasoning**: Video-R1 (GRPO + rule-based reward), VideoChat-R1, TinyLLaVA-Video-R1
- **Hallucination Mitigation**: V-DPO (preference optimization for hallucination reduction), Self-Introspective Decoding
- **Text Reasoning**: DeepSeek-R1, Open Reasoner Zero, Kimi k1.5
- **Insights**: The VER paradigm could be extended to image reasoning or combined with dynamic frame selection for further improvement.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic analysis of CoT failure in video reasoning with a theoretical explanation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 10 benchmarks, multiple ablations, and comprehensive qualitative analysis
- Writing Quality: ⭐⭐⭐⭐⭐ — Problem formulation is clear; Bayesian analysis is elegant
- Value: ⭐⭐⭐⭐☆ — VER is broadly generalizable but currently limited to closed-form tasks

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[CVPR 2026\] VideoAuto-R1: Video Auto Reasoning via Thinking Once, Answering Twice](../../CVPR2026/video_understanding/videoauto-r1_video_auto_reasoning_via_thinking_once_answering_twice.md)
- [\[ICCV 2025\] Towards Video Thinking Test: A Holistic Benchmark for Advanced Video Reasoning and Understanding](../../ICCV2025/video_understanding/towards_video_thinking_test_a_holistic_benchmark_for_advanced_video_reasoning_an.md)
- [\[NeurIPS 2025\] Grounding Foundational Vision Models with 3D Human Poses for Robust Action Recognition](grounding_foundational_vision_models_with_3d_human_poses_for_robust_action_recog.md)
- [\[NeurIPS 2025\] Video Finetuning Improves Reasoning Between Frames](video_finetuning_improves_reasoning_between_frames.md)

</div>

<!-- RELATED:END -->
