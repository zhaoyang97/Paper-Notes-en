---
title: >-
  [Paper Note] PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation
description: >-
  [CVPR 2025][Video Generation][Text-to-Video Generation] PhyT2V leverages the Chain-of-Thought (CoT) and step-back reasoning capabilities of LLMs to iteratively analyze discrepancies between generated videos and physical rules, thereby optimizing text prompts. This improves physical rule adherence in existing T2V models by up to 2.3 times without requiring retraining.
tags:
  - "CVPR 2025"
  - "Video Generation"
  - "Text-to-Video Generation"
  - "Physical Rules"
  - "LLM Inference"
  - "Iterative Self-Refinement"
  - "Prompt Enhancement"
date: 2026-05-08
content_hash: efe7d2d01b24b6e0
---

# PhyT2V: LLM-Guided Iterative Self-Refinement for Physics-Grounded Text-to-Video Generation

**Conference**: CVPR 2025  
**arXiv**: [2412.00596](https://arxiv.org/abs/2412.00596)  
**Code**: None  
**Area**: Diffusion Models / Video Generation  
**Keywords**: Text-to-Video Generation, Physical Rules, LLM Inference, Iterative Self-Refinement, Prompt Enhancement

## TL;DR

PhyT2V leverages the Chain-of-Thought (CoT) and step-back reasoning capabilities of LLMs to iteratively analyze discrepancies between generated videos and physical rules, thereby optimizing text prompts. This improves physical rule adherence in existing T2V models by up to 2.3 times without requiring retraining.

## Background & Motivation

**Background**: In recent years, Transformer-based diffusion models (such as Sora, CogVideoX, and OpenSora) have made breakthrough progress in text-to-video (T2V) generation, producing visually highly realistic video frames. However, these models still suffer from significant limitations regarding the physical realism of the generated videos.

**Limitations of Prior Work**: Current T2V models exhibit obvious deficiencies in adhering to physical rules, including incorrect object quantities, unreasonable material properties, violations of fluid dynamics, incorrect gravitational directions, and unnatural motion or collisions. Existing solutions generally fall into two categories: (1) data-driven methods, which rely on large-scale multimodal training data to cover more physical scenarios, but fail to generalize to out-of-distribution domains; and (2) external engine injection methods, which inject physical knowledge using 3D engines like Blender or Unity3D or depth maps, but are only applicable to predefined, fixed physical scenarios.

**Key Challenge**: The contradiction between the diversity and complexity of physical rules and the limited coverage of training data. Real-world physical scenes are infinite, and no finite dataset can completely cover them, while the models themselves lack explicit mechanisms to embed physical rules.

**Goal**: To improve the physical realism of video generation through pure text prompt optimization without modifying the T2V model architecture or retraining, making the method generalizable to any out-of-distribution domain.

**Key Insight**: The authors observe that T2V models are highly sensitive to contextual details in prompt descriptions. If sufficient and appropriate physical rule descriptions are injected into the prompt, physically unrealistic videos can be significantly improved. Furthermore, the strong natural language reasoning capabilities of LLMs can be utilized to automate this process.

**Core Idea**: Utilizing the CoT and step-back reasoning capabilities of LLMs to construct an iterative feedback loop—analyzing physical rules $\rightarrow$ detecting semantic mismatches between the video and the prompt $\rightarrow$ optimizing the prompt $\rightarrow$ regenerating—iterating for several rounds until the video quality is satisfactory.

## Method

### Overall Architecture

PhyT2V is an iterative, three-step self-refinement framework. Each iteration consists of three steps: Step 1 uses the LLM to analyze the user prompt to extract a list of objects that should appear in the video and the physical rules they should follow; Step 2 utilizes a video describer model (Tarsier) to translate the currently generated video into text descriptions, and then leverages the CoT reasoning of the LLM to evaluate semantic mismatches between the video descriptions and the prompt; Step 3 utilizes the step-back reasoning of the LLM, combining the physical rules from Step 1 and the semantic mismatches from Step 2, to generate an optimized prompt. The optimized prompt is then fed into the T2V model to regenerate the video, starting the next round of iteration. This iterative process continues until the video quality converges (usually 3-4 rounds are sufficient).

### Key Designs

1. **Local CoT Reasoning (Parallel Sub-problems)**:

    - **Function**: Decomposes the complex prompt optimization problem into two sub-problems that can be processed in parallel.
    - **Mechanism**: Step 1 and Step 2 handle physical rule identification and semantic mismatch detection, respectively. The prompt for each sub-problem consists of three components: task instructions $[I]$ (linking the current sub-problem to the overall optimization goal), in-context examples $[E]$ (few-shot QA examples to assist the LLM with in-context learning), and current task information $[T]$ (including the current prompt and the trigger phrase "Let's think step by step"). Through this structured prompting, the LLM can reason step-by-step to identify the specific physical rules the video should follow and the discrepancy between the current video and the prompt.
    - **Design Motivation**: A single complex reasoning path is prone to errors. Decomposing it into two focused sub-problems allows the LLM to perform deep analyses of physical rules and semantic mismatches independently, avoiding omissions.

2. **Global Step-back Reasoning (Final Optimization)**:

    - **Function**: Integrates the outputs of the two parallel sub-problems to generate the final optimized prompt.
    - **Mechanism**: Instead of simply chaining CoT sequentially across sub-problems (which easily leads to incorrect reasoning paths), step-back reasoning is employed. The physical rules and semantic mismatch analysis results are formulated as higher-level abstract knowledge, natively embedded in the instructions for final prompt generation. Simultaneously, quantitative feedback is introduced: VideoCon-Physics is utilized to score the previous video $[S]$. If $[S] < 0.5$, the LLM is prompted that the prior optimization was ineffective and should try an alternative reasoning path. The trigger phrase $[t]$ is removed to avoid introducing information irrelevant to the user's initial prompt in the final answer.
    - **Design Motivation**: Step-back reasoning, by integrating information at a higher level of abstraction, can correct intermediate reasoning errors that might occur in CoT, ensuring consistent prompt optimization direction.

3. **Video Captioning Feedback Mechanism**:

    - **Function**: Translates the visual content of videos into text to support reasoning in the pure text domain.
    - **Mechanism**: The video describer model Tarsier is used to translate the semantic content of the generated video into textual descriptions based on the object list extracted in Step 1. This allows the LLM to complete CoT and step-back reasoning entirely in the text domain without handling cross-modal alignment. Mathematically, the optimization process of PhyT2V is formulated as $p' = f_{\text{enhance}}(p, f_{\text{mismatch}}(C(V(p)), p), f_{\text{phy}}(p), \theta)$, where $C$ is the video describer model, $V(p)$ is the currently generated video, $f_{\text{phy}}$ analyzes physical rules, and $f_{\text{mismatch}}$ detects semantic mismatches.
    - **Design Motivation**: Standard CoT methods are designed for single-modality linear reasoning and show limited performance when directly applied to multimodal T2V tasks. By bridging modalities with video captioning, the multimodal problem is transformed into pure text reasoning, fully leveraging the linguistic reasoning advantages of the LLM.

### Loss & Training

PhyT2V does not involve any model training. It is a training-free inference-time optimization method that improves output quality by manipulating the input prompts of T2V models. There are two exit conditions for the iteration: (1) the video quality satisfies the requirements (determined by a T2V evaluator); (2) the iteration converges, meaning that the improvement in video quality between consecutive rounds is negligible.

## Key Experimental Results

### Main Results

Using ChatGPT-4 o1-preview as the LLM and Tarsier as the video describer model, the method was evaluated on two physical rule benchmarks: VideoPhy (688 prompts) and PhyGenBench (160 prompts).

| T2V Model | Dataset | Metric | Round 1 (Original) | Round 4 (PhyT2V) | Gain |
|----------|--------|------|---------------|-----------------|---------|
| CogVideoX-2B | VideoPhy | PC | 0.13 | 0.29 | 2.2x |
| CogVideoX-2B | VideoPhy | SA | 0.22 | 0.42 | 1.9x |
| CogVideoX-5B | VideoPhy | PC | 0.26 | 0.42 | 1.6x |
| CogVideoX-5B | VideoPhy | SA | 0.48 | 0.59 | 1.2x |
| OpenSora | VideoPhy | PC | 0.17 | 0.31 | 1.8x |
| VideoCrafter | VideoPhy | PC | 0.15 | 0.33 | 2.2x |

Comparison with prompt enhancer baselines (VideoPhy dataset):

| Method | CogVideoX-5B PC | CogVideoX-5B SA | OpenSora PC | OpenSora SA |
|------|-----------------|-----------------|-------------|-------------|
| ChatGPT 4 | 0.33 | 0.41 | 0.21 | 0.32 |
| Promptist | 0.25 | 0.39 | 0.19 | 0.33 |
| **PhyT2V** | **0.42** | **0.59** | **0.31** | **0.47** |

### Ablation Study

Performance improvement analyzed by physical rule categories (VideoPhy, CogVideoX-5B):

| Physical Category | PC (Round 1→4) | SA (Round 1→4) |
|---------|----------------|----------------|
| Solid-Solid | 0.21 → 0.32 | 0.24 → 0.47 |
| Solid-Fluid | 0.22 → 0.30 | 0.39 → 0.61 |
| Fluid-Fluid | 0.57 → 0.62 | 0.41 → 0.67 |

### Key Findings

- Iterative optimization converges rapidly: Most of the improvement is achieved in the first two rounds, with virtually no extra gains in the fourth round, suggesting 3-4 rounds are sufficient for practical applications.
- The improvement is most significant on weaker models (e.g., CogVideoX-2B, with up to a 2.2x gain in PC), demonstrating the method's effectiveness in compensating for limited model capacity.
- PhyT2V shows improvements across all physical categories, even achieving further gains in fluid-fluid interaction scenarios, which already have a relatively high baseline.
- Compared to directly using ChatGPT for prompt enhancement, PhyT2V performs at least 35% better, because the latter lacks a feedback loop based on the generated video.

## Highlights & Insights

- **Solving Multimodal Problems in the Pure Text Domain**: By using a video captioning model to translate the multimodal task into pure text reasoning, the method cleverly bypasses the bottleneck of LLMs processing visual information. This "modal bridging" concept can be transferred to other tasks where LLMs need to understand non-text modalities.
- **Iterative Closed-loop Feedback**: Instead of a one-time prompt enhancement, a closed loop of "generation $\rightarrow$ evaluation $\rightarrow$ optimization" is constructed. This feedback mechanism design can be generalized to other generative tasks as a post-processing optimization pipeline.
- **Completely Plug-and-Play**: It requires no model modification or training, making it compatible with any T2V model. This paradigm of "prompt-level intervention" holds high practical value.

## Limitations & Future Work

- Each iteration requires calling an LLM (ChatGPT-4 o1-preview) + a video describer model + a T2V model, which incurs high computational overhead and API costs.
- It relies on the LLM's "common-sense understanding" of physical rules, which may be insufficient for highly precise or quantitative physical simulation scenarios.
- The accuracy of the video describer model presents a bottleneck—if the video descriptions are inaccurate, subsequent semantic mismatch analyses will also be flawed.
- For extremely complex multi-object interaction scenarios, it remains challenging to achieve physical realism even after multiple iterations.
- Future work can explore incorporating lightweight outputs from physical simulation engines as additional constraints during prompt optimization.

## Related Work & Insights

- **vs. Direct LLM Prompt Enhancement**: PhyT2V incorporates a video feedback mechanism and iterative optimization, whereas direct LLM enhancement only rewrites the prompt once and cannot perceive concrete flaws in the generated video.
- **vs. Data-driven Methods (Large-scale Training)**: PhyT2V is training-free and performs inference-time optimization. It can be plugged into any existing model but is ultimately limited by the inherent capability ceiling of the underlying model.
- **vs. 3D Engine Injection Methods**: PhyT2V is far more generalizable and is not restricted to specific physical categories, but its precision remains inferior to physics engine-based approaches.

## Rating

- Novelty: ⭐⭐⭐⭐ Combining LLM reasoning capabilities with video generation is a relatively new direction, though CoT + step-back itself is not a new technique.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on multiple models and datasets with comprehensive category-wise analysis, though lacking human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Clearly structured with logical motivation, and formulated descriptions enhance the rigor of the methodology.
- Value: ⭐⭐⭐⭐ High practical value as a plug-and-play solution, though high computational costs limit real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Self-Correcting Text-to-Video Generation with Misalignment Detection and Localized Refinement](../../ACL2026/video_generation/self-correcting_text-to-video_generation_with_misalignment_detection_and_localiz.md)
- [\[CVPR 2025\] VIRES: Video Instance Repainting via Sketch and Text Guided Generation](vires_video_instance_repainting_via_sketch_and_text_guided_generation.md)
- [\[NeurIPS 2025\] PhysCtrl: Generative Physics for Controllable and Physics-Grounded Video Generation](../../NeurIPS2025/video_generation/physctrl_generative_physics_for_controllable_and_physicsgrou.md)
- [\[CVPR 2025\] Can Text-to-Video Generation Help Video-Language Alignment?](can_text-to-video_generation_help_video-language_alignment.md)
- [\[CVPR 2025\] Optical-Flow Guided Prompt Optimization for Coherent Video Generation](optical-flow_guided_prompt_optimization_for_coherent_video_generation.md)

</div>

<!-- RELATED:END -->
