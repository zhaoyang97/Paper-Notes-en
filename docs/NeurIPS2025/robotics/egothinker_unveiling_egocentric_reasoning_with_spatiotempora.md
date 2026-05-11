---
title: >-
  [Paper Note] EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT
description: >-
  [NeurIPS 2025][Robotics][Egocentric video] EgoThinker constructs EgoRe-5M, a 5-million-sample egocentric video QA dataset with causal CoT annotations and fine-grained hand-object localization data. Through a two-stage tr…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Egocentric video"
  - "chain-of-thought reasoning"
  - "hand-object grounding"
  - "GRPO reinforcement fine-tuning"
  - "large-scale dataset"
date: 2026-05-08
content_hash: cbf77506389aa9bb
---

# EgoThinker: Unveiling Egocentric Reasoning with Spatio-Temporal CoT

**Conference**: NeurIPS 2025
**arXiv**: [2510.23569](https://arxiv.org/abs/2510.23569)
**Code**: [https://github.com/InternRobotics/EgoThinker](https://github.com/InternRobotics/EgoThinker)
**Area**: Embodied Intelligence / Egocentric Video Understanding
**Keywords**: Egocentric video, chain-of-thought reasoning, hand-object grounding, GRPO reinforcement fine-tuning, large-scale dataset

## TL;DR

EgoThinker constructs EgoRe-5M, a 5-million-sample egocentric video QA dataset with causal CoT annotations and fine-grained hand-object localization data. Through a two-stage training paradigm—SFT for reasoning followed by GRPO for grounding—the approach enables a 7B MLLM to simultaneously perform egocentric causal reasoning and spatio-temporal fine-grained localization for the first time, achieving state-of-the-art results on 8+ benchmarks, with the 7B model surpassing 72B models on temporal grounding.

## Background & Motivation

**Background**: MLLMs have made significant progress on third-person visual understanding tasks. Chain-of-thought prompting (CoT) and reinforcement fine-tuning (RFT, e.g., GRPO from DeepSeek-R1) have further enhanced reasoning capabilities. However, these methods are almost entirely designed for the observer perspective, handling directly observable events.

**Limitations of Prior Work**: Egocentric video reasoning presents three unique challenges fundamentally distinct from third-person reasoning: (1) **Intent inference of an invisible actor**—the camera wearer is absent from the frame, requiring the model to infer hidden intentions and next actions from hand movements and object state changes, which demands causal reasoning rather than event recognition; (2) **Fine-grained hand-object interaction grounding**—understanding "what is being done" requires knowing precisely where the hands are and what is being grasped, yet existing MLLMs perform poorly on this task; (3) **Ultra-long temporal integration**—egocentric videos span from seconds to minutes, requiring models to track contextual evolution and retain details across thousands of frames. Existing datasets (Ego4D, EgoExo4D) provide large volumes of video but lack explicit reasoning chains, cross-temporal annotations, and fine-grained localization data.

**Key Challenge**: Existing MLLMs excel at general visual understanding but lack embodied egocentric cognition—they can interpret visual content but fail to understand "what I am doing" and "why I am doing it." Moreover, high-level reasoning (intent understanding) and low-level grounding (hand localization) are tightly coupled: accurate grounding is a prerequisite for sound reasoning, yet directly applying SFT on grounding data degrades reasoning performance.

**Goal**: (1) Construct a large-scale egocentric QA dataset with causal reasoning chains and spatio-temporal localization annotations; (2) Design a training strategy that enables MLLMs to jointly acquire high-level reasoning and low-level grounding capabilities without mutual interference.

**Key Insight**: The key observation is that GRPO reinforcement fine-tuning can directly optimize grounding accuracy using IoU as a verifiable reward, without training a reward model. Crucially, the KL regularization in GRPO constrains the model from deviating too far from its post-SFT state, thereby enhancing grounding capability without degrading acquired reasoning ability.

**Core Idea**: Construct a large-scale egocentric reasoning-and-grounding dataset via an automated annotation pipeline, then adapt a general-purpose MLLM into an egocentric reasoning expert through a two-stage approach: SFT for reasoning followed by GRPO for grounding.

## Method

### Overall Architecture

EgoThinker consists of two components: data construction and model training. On the data side, 8.7 million egocentric clips are extracted from large-scale web videos (e.g., HowTo100M) via a three-stage filtering pipeline, which are then combined with existing datasets (e.g., Ego4D) to yield 13 million clips total, from which 5 million QA pairs (EgoRe-5M) are automatically generated. On the model side, Qwen2-VL-7B serves as the backbone: Stage 1 performs SFT on 1.5 million mixed samples to establish a reasoning foundation; Stage 2 applies GRPO reinforcement fine-tuning on 70K fine-grained grounding samples.

### Key Designs

1. **Multi-Stage Egocentric Video Filtering Pipeline**:

    - Function: Efficiently select high-quality egocentric video clips from large-scale web video collections.
    - Mechanism: A three-stage filtering pipeline — (a) **Web-scale mining**: Starting from HowTo100M's HTM-AA and Howto-Interlink7M, 30 million initial clips are collected; (b) **Ego/Exo classification**: A classifier trained with an InternVideo backbone and MLP (92% accuracy, 89% AUC) filters out 12 million egocentric clips; (c) **Dynamic interaction filtering**: A hand-object detector retains clips containing dynamic hand-object interactions (requiring both visible hands and active objects), yielding 8.7 million high-quality clips, which are merged with Ego4D, EPIC-Kitchens, EgoExoLearn, and EgoExo4D to reach 13 million total.
    - Design Motivation: Existing egocentric datasets are far smaller than web video collections, yet the proportion and quality of egocentric content in web videos is low and inconsistent, making automated multi-stage filtering essential for scalable acquisition.

2. **Four-Dimensional QA Data Construction (EgoRe-5M)**:

    - Function: Comprehensively cover the capabilities required for egocentric reasoning through four complementary QA types.
    - Mechanism: Four data splits — (a) **Short-term perception QA (2.4M pairs)**: 1–10 second clips with 7 perception question types (object existence/attributes/count/interaction/action description/action reasoning/background attributes), generated by DeepSeek-V3 from original annotations and VideoChat2-HD captions; (b) **Long-term causal reasoning QA (2.5M pairs)**: Consecutive clips concatenated into 15–120 second segments with 6 temporal question types (action sequence/temporal grounding/object counting/action prediction/summarization/reasoning); (c) **CoT QA (50K pairs)**: DeepSeek-R1 generates questions with step-by-step reasoning from concatenated descriptions, with the model autonomously deciding whether to generate CoT questions for a given segment; (d) **Fine-grained grounding QA (70K pairs)**: Spatial grounding uses pixel-level annotations from EK-Visor to generate hand/object bounding-box questions; temporal grounding uses time annotations from EgoExoLearn to generate temporal interval questions, both requiring the model to produce reasoning before outputting coordinates.
    - Design Motivation: Existing datasets either cover only short-term perception, lack causal reasoning chains, or omit fine-grained localization. The four dimensions each address a distinct capability gap, and joint training enables comprehensive egocentric understanding.

3. **Two-Stage SFT + GRPO Training**:

    - Function: Establish a reasoning foundation first, then refine grounding capability via reinforcement learning without compromising reasoning.
    - Mechanism: **Stage 1 (SFT)**: Training on 1.5M samples spanning general visual captioning (100K), VQA (70K), egocentric-related data (390K, including SSV2 and EgoTimeQA), and the short-term, long-term, and CoT splits of EgoRe-5M (860K). **Stage 2 (RFT)**: GRPO reinforcement fine-tuning on 70K fine-grained grounding data. The reward function consists of two components — (a) Format reward $R_{\text{format}}$: Regex matching checks whether the output follows the `<think>...</think><answer>...</answer>` format, returning 1 for a match and 0 otherwise; (b) IoU reward $R_{\text{IoU}}$: Spatial grounding uses bounding-box mIoU; temporal grounding uses temporal-window mIoU. GRPO generates $N$ candidates per input, computes group-normalized advantages $A_i = (r_i - \text{mean})/\text{std}$, and maximizes advantage-weighted likelihood with KL divergence regularization.
    - Design Motivation: Directly applying SFT on grounding data degrades reasoning task performance (ablations show that SFT on fine-grained grounding reduces EgoSchema from 71.9 to 71.4 and QAEgo4D from 67.2 to 62.1), whereas RFT preserves acquired capabilities through KL regularization while substantially outperforming SFT on grounding (mIoU: 53.7 vs. 38.9).

### Loss & Training

The SFT stage uses standard cross-entropy supervised loss. The RFT stage uses the GRPO objective: $\max_{\pi_\theta} \mathbb{E}[\sum_i \frac{\pi_\theta(o_i)}{\pi_{\theta_{old}}(o_i)} \cdot A_i - \beta D_{KL}(\pi_\theta \| \pi_{ref})]$, where $\beta$ controls the deviation from the reference model.

## Key Experimental Results

### Main Results

| Benchmark | Metric | EgoThinker | Qwen2-VL-7B | Prev. SOTA | Gain |
|-----------|--------|-----------|-------------|------------|------|
| EgoTaskQA | Acc | **64.4** | 57.9 | InternVL2: 61.0 | +3.4 |
| EgoPlan-Val | Acc | **47.1** | 38.3 | Exo2Ego: 42.7 | +4.4 |
| EgoSchema | Acc | **67.6** | 63.3 | InternVL2: 64.2 | +3.4 |
| VLN-QA | Acc | **54.0** | 42.0 | InternVL2: 46.0 | +8.0 |
| RES Cross-view | Acc | **39.5** | 26.3 | LLaVA-Video: 31.1 | +8.4 |
| EK-Visor Spatial Grounding | Loc-Acc | **80.3** | 64.5 | 72B: 71.7 | +8.6 |
| EgoExoLearn Temporal Grounding | R1@0.05 | **63.9** | 5.4 | 72B: 49.9 | +14.0 |

### Ablation Study

| Configuration | EgoTaskQA | QAEgo4D | EgoSchema | EK-Visor mIoU/Loc |
|---------------|-----------|---------|-----------|-------------------|
| Baseline | 57.7 | 60.3 | 68.2 | 28.6/64.5 |
| +SFT (Short) | 61.6 | 63.1 | 69.1 | 29.1/64.9 |
| +SFT (Short+Long) | 64.2 | 63.7 | 71.1 | 28.9/64.5 |
| +SFT (Short+Long+CoT) | 64.3 | **67.2** | **71.9** | 28.5/64.4 |
| +SFT (FG direct SFT) | — | 62.1 | 71.4 | 38.9/74.1 |
| **+RFT (GRPO)** | **64.4** | 66.1 | 71.8 | **53.7/80.3** |

### Key Findings

- **Large RFT vs. SFT gap on grounding**: EK-Visor mIoU 53.7 vs. 38.9; temporal grounding R1@0.05 63.9 vs. 24.9. Crucially, RFT does not degrade reasoning (EgoSchema 71.8 vs. 71.4 after SFT-FG), whereas SFT on grounding causes QAEgo4D to drop sharply from 67.2 to 62.1.
- **7B surpasses 72B**: EgoThinker-7B outperforms Qwen2.5-VL-72B on both temporal grounding (R1@0.05: 63.9 vs. 49.9) and spatial grounding Loc-Acc (80.3 vs. 71.7), demonstrating that targeted training is more effective than simply scaling model size.
- **CoT data benefits memory-intensive reasoning most**: Adding the CoT split raises QAEgo4D (which focuses on episodic memory QA) substantially from 63.7 to 67.2, while EgoTaskQA improves by only 0.1—indicating that CoT is most beneficial for tasks requiring multi-step causal chains.
- **Grounding capability reduces hallucination**: Performance on the POPE benchmark improves by 3.2% (83.6→86.8), with enhanced hand-object grounding enabling more accurate object-existence judgments.

## Highlights & Insights

- **7B surpassing 72B is the most compelling result**: It demonstrates that domain-focused data and training strategies are more efficient than brute-force parameter scaling—a finding with significant practical implications for resource-constrained researchers.
- **Elegant combination of GRPO and IoU reward**: Using IoU as a verifiable reward for reinforcement fine-tuning avoids the complexity of training a reward model. Meanwhile, KL regularization naturally preserves the reasoning capabilities acquired during SFT. This "SFT→RFT" paradigm is generalizable to any task requiring both reasoning and precise structured output (e.g., medical imaging combined with diagnostic reasoning).
- **Industrial-grade data construction methodology**: The three-stage filtering pipeline from 30 million web videos to 8.7 million egocentric clips, combined with the automated CoT annotation approach using DeepSeek-R1, provides a reusable large-scale vertical data construction paradigm.

## Limitations & Future Work

- **Dependency on large-scale annotation and offline fine-tuning**: Although automatic generation of 5 million QA pairs is feasible, it still requires substantial GPU resources and API calls, and cannot adapt to new scenarios in real time.
- **Systematic bias in automatic annotation**: QA pairs are automatically generated by DeepSeek-V3/R1 with 95% accuracy verified by sampling, but may propagate model biases (e.g., tendencies toward certain question types or culturally-specific action misinterpretation).
- **Validation limited to Qwen2-VL-7B**: Effects on larger backbone models or alternative architectures remain unexplored.
- **No real-time inference or online adaptation**: The paper itself acknowledges this as a critical limitation—wearable assistants require streaming inference capability.

## Related Work & Insights

- **vs. Exo2Ego (2024)**: Exo2Ego enhances egocentric understanding via cross-view contrastive learning; EgoThinker leads by 8.4% on the RES cross-view benchmark through an end-to-end approach combining large-scale data, CoT, and RFT.
- **vs. VideoChat-R1**: Also uses RFT to enhance temporal perception, but targets general video understanding. EgoThinker specifically designs hand-object IoU rewards, yielding superior performance in egocentric scenarios.
- **vs. LLaVA-Video**: The general-purpose video model exhibits uneven performance on egocentric tasks (high on QAEgo4D but low on EgoPlan), whereas EgoThinker consistently leads across all egocentric benchmarks.

## Rating

- Novelty: ⭐⭐⭐⭐ The two-stage paradigm and IoU reward design are not entirely novel, but their application to egocentric reasoning-and-grounding achieving 7B>72B represents a meaningful contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 8+ benchmarks with comprehensive ablations (data splits, training paradigms, frame counts, hallucination evaluation) and qualitative visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, thorough data construction details, and high-quality figures and tables.
- Value: ⭐⭐⭐⭐ Provides both a large-scale dataset (EgoRe-5M) and a training paradigm with direct reference value for embodied AI and wearable assistant research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NeSyPr: Neurosymbolic Proceduralization For Efficient Embodied Reasoning](nesypr_neurosymbolic_proceduralization_for_efficient_embodied_reasoning.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[NeurIPS 2025\] GUI-Rise: Structured Reasoning and History Summarization for GUI Navigation](gui-rise_structured_reasoning_and_history_summarization_for_gui_navigation.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[NeurIPS 2025\] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning](mesatask_towards_task-driven_tabletop_scene_generation_via_3d_spatial_reasoning.md)

</div>

<!-- RELATED:END -->
