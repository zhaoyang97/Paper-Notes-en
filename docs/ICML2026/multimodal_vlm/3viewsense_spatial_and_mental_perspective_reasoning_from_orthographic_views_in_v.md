---
title: >-
  [Paper Note] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models
description: >-
  [ICML 2026][Multimodal VLM][Orthographic Views] 3ViewSense argues that the bottleneck in VLM spatial reasoning is not a lack of visual features or weak linguistic reasoning…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Orthographic Views"
  - "Spatial Reasoning"
  - "Mental Simulation"
  - "VLM"
  - "GRPO"
date: 2026-05-08
content_hash: 331c4073f19e9e28
---

# 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2603.07751](https://arxiv.org/abs/2603.07751)  
**Code**: https://github.com/Jasaxion/3ViewSense  
**Area**: Multimodal VLM / Spatial Reasoning  
**Keywords**: Orthographic Views, Spatial Reasoning, Mental Simulation, VLM, GRPO

## TL;DR
3ViewSense argues that the bottleneck in VLM spatial reasoning is not a lack of visual features or weak linguistic reasoning, but the absence of stable 3D intermediate representations. Consequently, the model is tasked to first induce front, left, and top views from a single image before reasoning based on these orthographic views. This approach significantly outperforms VLMs of similar scale in occlusion counting and view-consistent spatial reasoning.

## Background & Motivation
**Background**: Large Vision-Language Models (VLMs) can already handle complex image-text QA, symbolic logic, and multi-step reasoning, yet remain fragile in spatial understanding. A typical example is block stacking and counting: while human children can count by imagining occluded parts, VLMs often resort to repetitive guessing due to occlusion, perspective, and depth ambiguity.

**Limitations of Prior Work**: The paper first conducts two diagnostic tests. First, after freezing VLM visual features and training a lightweight MLP probe, the block counting accuracy reaches 55.8%, indicating that image encoders already preserve substantial geometric information. Second, if textual descriptions of the front/left/top views are directly provided to the same model, the spatial reasoning of models like Gemini-3-pro improves significantly. This suggests the problem is not merely "not seeing" or "not knowing how to reason," but that visual features are not organized into a spatial structure that the reasoner can stably utilize.

**Key Challenge**: A single egocentric image compresses 3D structures into 2D projections, naturally entailing occlusion and depth ambiguity. When end-to-end VLMs directly model $P(a \mid I_{ego}, q)$, they must simultaneously perform 3D completion, viewpoint transformation, and answer reasoning within the latent space. Once the intermediate state becomes unstable, subsequent linguistic reasoning suffers from drift and hallucinations.

**Goal**: The authors aim to externalize the process of "recovering reason-able 3D mental representations from 2D images" by requiring the model to generate a set of canonical orthographic views first, and then answer counting, position, and attribute-related questions based on these views.

**Key Insight**: Engineering graphics uses orthographic projections to decompose 3D objects into front, left, and top views, thereby reducing perspective-induced ambiguity. The paper transfers this concept to VLMs, using triple views as the interface between egocentric perception and allocentric spatial reasoning.

**Core Idea**: Replace the single-stage black-box image QA with a "Simulate-and-Reason" pipeline that first simulates triple views and then reasons based on them.

## Method
The crux of 3ViewSense is not the introduction of additional 3D sensors, but rather decomposing the internal reasoning process of the VLM into two supervisable capabilities: Orthographic Mental Simulation (OMS), responsible for converting single-view images into structured triple views, and View-Grounded Reasoning (VGR), responsible for integrating the triple views with the original question into an answer. Formally, the paper formulates the answer distribution as a two-stage process via latent orthographic views $\mathcal{V}=\{v_{front}, v_{left}, v_{top}\}$: first estimating $\hat{\mathcal{V}}=\arg\max_{\mathcal{V}}P_{\theta_{sim}}(\mathcal{V}\mid I_{ego},q)$, and then outputting the answer using $P_{\theta_{reason}}(a\mid \hat{\mathcal{V}},I_{ego},q)$.

### Overall Architecture
The input consists of an egocentric image and a spatial question, and the output is a count or a discrete directional label. Training is divided into three steps: First, Stage I SFT is performed on procedural data from OrthoMind-3D to learn the generation of structured descriptions for the front/left/top views. Second, Stage II SFT is conducted to let the model generate natural language reasoning chains and final answers conditioned on the triple views. Finally, starting from the Stage II model, GRPO is used with mathematically verifiable rewards to enhance answer correctness and reasoning stability.

### Key Designs
1.  **Orthographic Mental Simulation**:
    *   **Function**: Translates a single egocentric image into three structured descriptions: front, left, and top views.
    *   **Mechanism**: For block counting tasks, view descriptions include visible blocks, stacking heights, and occlusion cues; for object reasoning, descriptions are object sequences ordered by scanning (e.g., left-to-right or far-to-near). Stage I uses 19.5k samples with triple-view annotations for maximum likelihood training.
    *   **Design Motivation**: Triple views convert implicit 3D completion problems into 2D planar problems closer to symbolic pattern recognition, reducing the space for the model to guess depth on the fly during the linguistic reasoning stage.

2.  **View-Grounded Reasoning**:
    *   **Function**: Enables the model to explicitly read the triple views and synthesize the original image and question to generate an answer.
    *   **Mechanism**: Stage II constructs reasoning trajectories following a "front → left → top" integration order. These are generated by Gemini-3-Flash and filtered by final answer correctness, resulting in 21k training samples. Instead of just learning final labels, the model learns how to assemble multiple projections into a consistent mental structure.
    *   **Design Motivation**: Supervizing only the final answer leads to overfitting dataset patterns; supervising the reasoning process conditioned on views preserves "multi-view integration" as a transferrable capability.

3.  **Refinement via GRPO and Verifiable Rewards**:
    *   **Function**: Further improves the accuracy of counting and directional prediction after SFT while mitigating generalization degradation caused by large-scale SFT.
    *   **Mechanism**: Initialized with the Stage II model, a group of responses is sampled and scored using verifiable rewards for integer counting or discrete directions. Optimization uses group-normalized advantages $\hat{A}_i=(R_i-\mu_R)/(\sigma_R+\delta)$ and the clipped GRPO objective. The paper compares "strict reward" and "slack reward," the latter of which provides some tolerance for counting errors.
    *   **Design Motivation**: Answers for spatial tasks can be automatically verified, making them suitable for RL. However, performing RL directly from OMS-SFT is unstable; a "view reasoning warm start" from VGR is essential.

### Loss & Training
Stage I and Stage II both employ standard sequence maximum likelihood training to learn triple-view generation and view-conditioned reasoning, respectively. The RL phase uses 30k instances, with 10k resampled from Stage II and 20k newly generated; the base model is Qwen3-VL-4B-Instruct. Reward configurations include strict and slack variants, both utilizing automatic verification for integer count or discrete directional labels.

## Key Experimental Results

### Main Results
Main experiments on OrthoMind-3D show that general VLMs are very weak at occluded block counting, while 3ViewSense nearly pushes in-domain tasks to high accuracy after GRPO.

| Model | Block Count | Block Count (Attr.) | Object Count | Object Position | Object Count (Attr.) | Object Position (Attr.) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-4o | 15.8 | 53.2 | 68.3 | 39.3 | 71.2 | 47.2 |
| Gemini-3-pro | 13.8 | 80.2 | 83.3 | 71.6 | 93.2 | 93.6 |
| SpaceOm-4B | 10.4 | 47.2 | 63.6 | 17.6 | 60.2 | 25.4 |
| Qwen3-VL-4B-Instruct | 6.2 | 43.4 | 59.0 | 41.0 | 74.8 | 45.6 |
| 3ViewSense-4B-SFT | 33.4 | 63.1 | 97.0 | 91.0 | 95.4 | 91.8 |
| 3ViewSense-4B-RL-strict | 95.0 | 88.2 | 98.7 | 93.3 | 97.4 | 93.2 |
| 3ViewSense-4B-RL-slack | 94.4 | 88.6 | 98.7 | 92.3 | 98.4 | 93.4 |

On OOD and external benchmarks, the improvements are less dramatic than in-domain, but transfer gains from triple-view reasoning are still visible. Using Qwen3-VL-4B-Instruct as a baseline, RL-slack improves from 21.2 to 38.7 on OrthoMind-3D OOD Block Count and from 27.2 to 38.9 on MindCube-Tiny.

| Model | OOD Block Count | OOD Object Position | MindCube-Tiny | CV-Bench 2D | SPBench-SI | ViewSpatial |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Qwen3-VL-4B-Instruct | 21.2 | 46.7 | 27.2 | 77.9 | 22.2 | 35.5 |
| 3ViewSense-4B-SFT | 31.1 | 72.5 | 34.9 | 74.3 | 20.6 | 34.4 |
| 3ViewSense-4B-RL-strict | 33.2 | 74.3 | 36.7 | 78.1 | 23.2 | 36.6 |
| 3ViewSense-4B-RL-slack | 38.7 | 76.1 | 38.9 | 79.9 | 25.4 | 37.1 |

### Ablation Study
The ablation focuses on proving that gains come from the "view-conditioned reasoning process" rather than just more supervision data.

| Configuration | OrthoMind-3D ID | OrthoMind-3D OOD | SPBench-SI | ViewSpatial | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Direct QA | 80.3 | 49.8 | 1.3 | 7.2 | Same input and hyperparams, but only final answer supervised |
| 3ViewSense Reasoning | 70.3 | 46.6 | 21.2 | 34.1 | Supervised view-conditioned reasoning chain |

Direct QA is higher on OrthoMind-3D, indicating it fits the target dataset better, but it nearly collapses on SPBench-SI and ViewSpatial. 3ViewSense Reasoning has slightly lower in-domain scores but retains transferable spatial reasoning capabilities.

| Stage | OrthoMind-3D ID | OrthoMind-3D OOD | MindCube-Tiny | ViewSpatial | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| OMS-SFT only | 48.7 | 41.3 | 29.6 | 33.4 | Only induces views; insufficient for complex queries |
| VGR-SFT only | 70.3 | 46.6 | 32.4 | 34.1 | Direct learning of view-conditioned reasoning |
| OMS→VGR two-stage | 78.6 | 49.5 | 34.9 | 34.4 | Learning view induction then reasoning is most stable |

### Key Findings
*   Attribute-conditioned counting is significantly easier than pure counting, as salient attributes like color or size simplify 3D enumeration into local retrieval.
*   ICL cannot stably teach triple-view reasoning; only a few strong closed-source models show limited improvement, while most open-source models degrade, indicating this is not a simple prompt trick.
*   Base models produce verbose reasoning exceeding 10k tokens for block counting. 3ViewSense, by forming view sketches first, produces shorter and more stable outputs.
*   GRPO reward curves rise steadily when initialized from Stage II VGR-SFT, whereas starting directly from Stage I OMS-SFT results in high-variance oscillations, highlighting that the "view reasoning warm start" is critical for RL.

## Highlights & Insights
*   The most valuable aspect of the paper is turning VLM spatial failures into a diagnostic problem: visual probes and explicit triple-view prompting rule out "not seeing" and "not thinking" explanations, respectively, locating the bottleneck at the lack of an intermediate spatial interface.
*   Orthographic projection is a simple yet effective inductive bias. It does not introduce heavy 3D modules but explicitly disentangles depth ambiguity—the hardest part of occlusion counting—making it compatible with existing VLM training pipelines.
*   The Direct QA ablation is insightful: higher in-domain answer accuracy does not equate to better spatial ability. What truly transfers is the supervision of intermediate representations and reasoning processes, not answer-pattern fitting.

## Limitations & Future Work
*   Three fixed orthographic views cannot cover all spatial problems. Tasks involving support relations, affordance, dynamics, or physical stability require intermediate representations richer than geometric projections.
*   OMS relies on procedural synthesis and training data with available triple-view ground truth; obtaining equivalent quality orthographic annotations in open-world images is difficult.
*   The improvements on OOD and external benchmarks are relatively modest, suggesting that the triple-view capabilities learned by the model are still domain-constrained. Future work should enable the model to estimate view induction uncertainty and adaptively select other spatial abstractions when necessary.
*   The current method increases the training phase and reasoning chain length; actual deployment will require balancing output length, latency, and spatial reasoning gains.

## Related Work & Insights
*   **vs Spatial-MLLM / External 3D Encoder Methods**: These often rely on extra visual modules, masks, or 3D features. 3ViewSense instead learns interpretable spatial intermediate representations within the language interface, which is cheaper but limited by the expressive power of triple views.
*   **vs SpatialLadder / RL Self-Correction Methods**: Curriculum learning or general RL strengthens training strategies, whereas 3ViewSense focuses on changing the reasoning object to view-consistent representations. The two could be combined in the future.
*   **vs MindCube / Mental Imagery Methods**: MindCube leans toward implicit 3D mental imagery. 3ViewSense constrains this imagery with canonical orthographic projections, making the intermediate process more inspectable and easier to supervise.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Bringing engineering-style orthographic views into VLM spatial reasoning and supporting the problem localization with diagnostic experiments is a concise but impactful idea.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Main experiments, OOD, external benchmarks, and multi-layer ablations are comprehensive, though real-world open-world tasks and deployment cost analyses are insufficient.
*   Writing Quality: ⭐⭐⭐⭐⭐ The logic from diagnosis to method to ablation is very smooth, making the purpose of each module clear.
*   Value: ⭐⭐⭐⭐⭐ Very inspiring for multimodal spatial reasoning, especially as a case study for "explicit intermediate representations being superior to answer fitting."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models](active_exploring_like_a_pigeon_reinforcing_spatial_reasoning_via_agentic_vision-.md)
- [\[ICCV 2025\] Perspective-Aware Reasoning in Vision-Language Models via Mental Imagery Simulation](../../ICCV2025/multimodal_vlm/perspective-aware_reasoning_in_vision-language_models_via_mental_imagery_simulat.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[ICML 2026\] Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)

</div>

<!-- RELATED:END -->
