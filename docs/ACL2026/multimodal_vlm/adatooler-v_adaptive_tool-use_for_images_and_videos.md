---
title: >-
  [Paper Note] AdaTooler-V: Adaptive Tool-Use for Images and Videos
description: >-
  [ACL 2026][Multimodal VLM][Multi-modal Reasoning] This paper identifies the prevalent problem of **blind tool-use** in existing "thinking with images" MLLMs—where visual tools (zoom-in/frame extraction) are forcibly appl…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multi-modal Reasoning"
  - "Adaptive Tool-use"
  - "AT-GRPO"
  - "Tool Benefit Score"
  - "V* bench"
date: 2026-05-08
content_hash: 95ad89be8ec2f637
---

# AdaTooler-V: Adaptive Tool-Use for Images and Videos

**Conference**: ACL 2026 Findings  
**arXiv**: [2512.16918](https://arxiv.org/abs/2512.16918)  
**Code**: https://github.com/CYWang735/AdaTooler-V  
**Area**: Multi-modal VLM / Tool-use / Reinforcement Learning  
**Keywords**: Multi-modal Reasoning, Adaptive Tool-use, AT-GRPO, Tool Benefit Score, V* bench

## TL;DR
This paper identifies the prevalent problem of **blind tool-use** in existing "thinking with images" MLLMs—where visual tools (zoom-in/frame extraction) are forcibly applied to all visual questions. This leads to reduced accuracy from overthinking and increased inference costs. To address this, AdaTooler-V is proposed, introducing the AT-GRPO reinforcement learning algorithm. It uses a sample-level Tool Benefit Score to dynamically adjust reward scales (encouraging tool use when effective and penalizing it when unnecessary), enabling a 7B model to achieve 89.8% on the V* high-resolution benchmark, surpassing GPT-4o and Gemini 1.5 Pro.

## Background & Motivation

**Background**: The "thinking with images" paradigm has recently become popular in multi-modal LLM reasoning. It involves inserting calls to visual tools (cropping, frame extraction, path tracing) within the Chain-of-Thought, allowing the model to repeatedly ground itself in detailed pixels. This significantly improves performance on complex visual tasks like high-resolution images and long videos (e.g., OpenThinkIMG, PixelReasoner, VITAL). Open-source representatives like Vision-R1, Video-R1, and OneThinker have extended R1-style RL to VLMs.

**Limitations of Prior Work**: The authors observe a neglected core problem—**blind tool-use**. Specific manifestations include: (a) existing training rewards often implicitly encourage tool usage, causing models to zoom-in or extract frames for all questions; (b) many visual questions can be solved with pure text-based CoT (e.g., "Which clock in these two images shows what time?"); forced tool usage triggers overthinking, leading the model astray from the correct reasoning path; (c) repetitive and meaningless tool calls gradually weaken the model's reliance on original visual inputs, making it harder to focus on key visual cues; (d) redundant calls increase inference costs for tasks that do not require tools. Figure 1 shows a distribution in their 300k dataset where approximately half the samples are tool-helpful ($\Delta S > 0$), while the other half are tool-unhelpful or even tool-harmful.

**Key Challenge**: Models lack an explicit mechanism to "judge whether this question requires a tool." The reward signals in existing RL frameworks are one-size-fits-all, failing to distinguish between "should use tool" and "should not use tool" at the sample level.

**Goal**: (1) Enable the VLM to **adaptively** decide whether to call visual tools for each question; (2) Introduce a sample-level tool benefit signal in RL training so that the reward can perceive "whether this tool call actually improved performance."

**Key Insight**: The authors define a Tool Benefit Score $\Delta S$ as the difference in average accuracy between using a tool and not using a tool for the same sample. This explicitly categorizes samples into tool-helpful ($\Delta S > 0$) and tool-unhelpful ($\Delta S \leq 0$). The reward scale of GRPO is then modified to be sample-type aware.

**Core Idea**: Utilizing AT-GRPO (Adaptive Tool-use GRPO)—scaling up rewards for tool usage in tool-helpful samples and penalizing unnecessary usage in tool-unhelpful samples. Combined with a two-stage (SFT cold start + RL) training strategy, the model autonomously learns when to invoke tools.

## Method

### Overall Architecture
AdaTooler-V models multi-modal reasoning as a thought-action-observation loop. Given a query and an image/video, the policy model first decides whether a tool is needed: if not, it outputs a thought $T$ and provides the answer directly; if needed, it iteratively generates $(T_i, C_i)$. Each action $C_i$ calls one of four visual tools (CropImg / FrameAt / VideoClip / PathTracer), returning an observation $E_i$ which is fed back into the context to continue reasoning until an answer is reached or context/turn limits are met. Training occurs in two stages: (1) **SFT Cold Start**—fine-tuning on AdaTooler-V-CoT-100k (multi-turn tool-interaction trajectories) to establish basic reasoning patterns and tool-calling priors; (2) **RL with verifiable rewards**—RL training with AT-GRPO on AdaTooler-V-300k to allow the model to autonomously explore "when to use tools."

### Key Designs

1.  **AT-GRPO: Tool Benefit Score Driven Adaptive Reward**:
    - **Function**: Introduces a Tool Benefit Score $\Delta S$ for each sample within the GRPO framework to dynamically adjust the reward scale based on "whether the tool is actually useful," teaching the model to call tools on demand rather than blindly.
    - **Mechanism**: For each training sample, $\Delta S = \text{Acc}(\text{with tool}) - \text{Acc}(\text{without tool})$ is **pre-computed offline** (using Qwen2.5-VL-72B to run N iterations with and without tools per sample to find the mean; the distribution of $\Delta S$ for 300k samples is shown in Figure 1). During RL, the reward is rewritten based on $\Delta S$: for $\Delta S > 0$ (tool-helpful) samples, trajectories using tools receive higher rewards; for $\Delta S \leq 0$ (tool-unhelpful) samples, trajectories using tools are penalized, encouraging pure text CoT. This allows the model's policy gradient to perceive the meta-signal of "tool necessity," achieving adaptivity.
    - **Design Motivation**: Standard GRPO rewards only look at final answer correctness and are insensitive to path redundancy. Since tool usage involves overthinking and inference overhead, models easily learn a "blindly use tools for high scores" strategy. Introducing $\Delta S$ as a sample-level prior explicitly injects "tool calling necessity" into the reward, which is more precise than a simple step penalty.

2.  **Unified Action Space for 4 Visual Tools**:
    - **Function**: Enables the model to operate flexibly on any intermediate observation, covering common local interaction needs for images and videos.
    - **Mechanism**: Defines four tools—**CropImg** (crops/scales images by bbox for "zoom in"), **FrameAt** (extracts a single frame from video by timestamp), **VideoClip** (extracts video segments by start/end times), and **PathTracer** (draws a trajectory/connection between two points to assist spatial reasoning). Each tool inputs and outputs image patches, which can be further manipulated by subsequent tools (e.g., using FrameAt followed by CropImg).
    - **Design Motivation**: Limiting the tool space to "returning visual observations" avoids distracting training signals from text tools (search, calculators). These four actions cover the core "thinking-with-image" patterns: image zooming, video temporal anchoring, video segment focus, and spatial path tracing.

3.  **Two-stage Training + Joint Multi-modal Data Construction**:
    - **Function**: Injects tool-use priors via SFT, overcomes pattern matching bottlenecks via AT-GRPO, and performs joint training across single-image, multi-image, and video modalities.
    - **Mechanism**: (a) **Data**—AdaTooler-V-300k covers math, visual counting, logic, spatial, and video temporal tasks; Qwen2.5-VL-72B generates multi-turn trajectories, followed by rule-based filtering to obtain 100k high-quality SFT samples. (b) **SFT Phase**—Direct fine-tuning to teach the model to produce coherent (T, C, E) cycles. (c) **RL Phase**—AT-GRPO reinforcement on tasks with verifiable rewards (multiple choice = exact match, numerical = exact match, OCR = WER, free-form = ROUGE average) to move beyond SFT's pattern matching.
    - **Design Motivation**: Starting RL from scratch on multi-modal long trajectories is extremely difficult to converge due to the massive exploration space. The "SFT first, RL refine" approach is a standard paradigm for agentic tasks. Joint multi-modal training allows detail-focusing skills learned in single-image tasks to transfer to video frame selection.

### Loss & Training
SFT phase: Standard next-token prediction loss on multi-turn trajectories from AdaTooler-V-CoT-100k (incorporating thoughts, actions, and observations). RL phase: AT-GRPO, based on GRPO’s group-relative advantage estimation, with the key modification of using $\Delta S$ as a scaling factor in reward calculation (logic: "add bonus for tools when $\Delta S > 0$, penalize when $\Delta S \leq 0$"). The model is based on Qwen2.5-VL-7B-Instruct.

## Key Experimental Results

### Main Results
Covering 12 benchmarks across single-image (V*, MME, InfoVQA, MMBench, MathVista), multi-image (MMSI-Bench, SPAR-Bench), and video.

| Model | Params | V* | MME | MathVista | MMSI-Bench |
|------|--------|------|------|-----------|------------|
| GPT-4o (Closed) | – | 65.2 | 2328 | 63.8 | 30.3 |
| Gemini 1.5 Pro (Closed) | – | 71.7 | – | 63.9 | 36.9 |
| InternVL3-8B | 8B | – | 2415 | 71.6 | 25.7 |
| Qwen2.5-VL-7B (base) | 7B | – | – | – | – |
| **AdaTooler-V-7B** | 7B | **89.8** | – | – | – |

(The V* score of 89.8% significantly outperforms GPT-4o's 65.2% and Gemini 1.5 Pro's 71.7%).

### Ablation Study

| Configuration | V* | Description |
|------|------|------|
| Qwen2.5-VL-7B base | ~– | Baseline without tools |
| + Multi-modal interleaved CoT (No AT-GRPO) | ~– | Tool-calling but blind, suffers from overthinking |
| + AT-GRPO (No $\Delta S$ distinction, standard GRPO) | ~– | Adaptive reward disabled |
| + Full AT-GRPO with $\Delta S$ | **89.8** | Full model |

(Exact ablation figures were not provided in the first 2000 lines and require reference to the full text).

### Key Findings
- **Significant lead over GPT-4o on V* (+24.6)**: Tasks requiring high-resolution visual details are where tool benefit varies most; AT-GRPO provides the highest gains here.
- **Avoiding blind usage significantly reduces inference costs**: The motivation highlights that unnecessary tool calls waste compute; AT-GRPO allows the model to use direct text CoT for simple problems.
- **Multi-modal joint training is beneficial**: Mixed training across single-image, multi-image, and video allows "tool decision" capabilities to transfer across modalities.
- **$\Delta S$ distribution is approximately symmetric** (Figure 1): Roughly half the samples benefit from tools while half do not, confirming that "blind usage" is a widespread data-level phenomenon—justifying why sample-level adaptive rewards are superior to global hyperparameters.

## Highlights & Insights
- **"$\Delta S$ as a sample-level meta-signal" is simple yet effective**: Defining tool benefit by the accuracy delta of the same model (with/without tools) bypasses the meta-question of "how to judge when to use a tool" by empirically measuring it and feeding it back into RL. This could extend to other agentic training (e.g., code generation "should we execute a sandbox?").
- **First to explicitly identify "blind tool-use" as a bottleneck**: While prior works assumed "more tools are better," this work uses motivational data to convince the reader that tools are detrimental for half the samples—a valuable paradigm critique.
- **Unified image and video tool space**: CropImg + FrameAt + VideoClip + PathTracer offer clear, combinable semantics (e.g., "FrameAt a keyframe then CropImg a region"), avoiding over-specialization.
- **7B model surpassing closed-source GPT-4o on V***: Proves that with designed "thinking-with-image" and adaptive tool decisions, open-source mid-sized models can reach SOTA on specific high-res visual tasks, which is meaningful for deployable agentic VLMs.

## Limitations & Future Work
- Offline $\Delta S$ measurement relies on a "judge model" (Qwen2.5-VL-72B), making generating 300k samples costly; switching domains requires re-running the comparison, lacking zero-shot adaptability.
- The precise mathematical formula for reward scaling via $\Delta S$ is not detailed in the abstract/intro; engineering details regarding the scaling function's impact on stability require code verification.
- Four visual tools are still limited—lacking text tools (OCR, search, calculator), 3D operations, or comparison tools for more complex agentic scenarios.
- Validated only on 7B parameters; lacks scaling law analysis. On larger models, the $\Delta S$ distribution might shift toward tool-unhelpful (as base visual capabilities improve), making the effectiveness of AT-GRPO an open question.
- "When to use a tool" is determined by benchmark data; for real-world applications with imbalanced distributions, the model might need online adaptation rather than a fixed strategy.

## Related Work & Insights
- **vs PixelReasoner / OpenThinkIMG / VITAL**: These proposed the "thinking-with-images" paradigm but used rewards that implicitly encouraged tool usage. AdaTooler-V is the first to explicitly optimize "whether to call."
- **vs Video-R1 / OneThinker**: These extended R1 to video/multimodal but maintained a single reward signal. AT-GRPO integrates sample-level priors into GRPO.
- **vs Vision-R1**: Early R1-style VLM focusing on text-only CoT. AdaTooler-V introduces multi-modal interleaved CoT and addresses the over-tool problem.

## Rating
- Novelty: ⭐⭐⭐⭐ "$\Delta S$-driven adaptive reward" is a clean design that systematically tackles the blind tool-use problem.
- Experimental Thoroughness: ⭐⭐⭐⭐ 12 benchmarks across image/video + tool-usage analysis + surpassing GPT-4o on V*.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation via Figure 1, well-diagnosed pain points, and intuitive case studies.
- Value: ⭐⭐⭐⭐ Open-source 7B beating closed-source SOTA on V* offers direct lessons for training "thinking-with-image" VLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] VTool-R1: VLMs Learn to Think with Images via Reinforcement Learning on Multimodal Tool Use](../../ICLR2026/multimodal_vlm/vtool-r1_vlms_learn_to_think_with_images_via_reinforcement_learning_on_multimoda.md)
- [\[CVPR 2026\] AVR: Adaptive VLM Routing for Computer Use Agents](../../CVPR2026/multimodal_vlm/adaptive_vision-language_model_routing_for_computer_use_agents.md)
- [\[AAAI 2026\] VipAct: Visual-Perception Enhancement via Specialized VLM Agent Collaboration and Tool-use](../../AAAI2026/multimodal_vlm/vipact_visual-perception_enhancement_via_specialized_vlm_age.md)
- [\[ACL 2026\] Leave My Images Alone: Preventing Multi-Modal Large Language Models from Analyzing Unauthorized Images](leave_my_images_alone_preventing_multi-modal_large_language_models_from_analyzin.md)
- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)

</div>

<!-- RELATED:END -->
