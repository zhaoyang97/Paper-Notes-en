---
title: >-
  [Paper Note] Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models
description: >-
  [ICML 2026][Multimodal VLM][Active Visual Exploration] This paper transforms VLM spatial reasoning from a "passive observation" paradigm into an agentic workflow of "question-driven active framing, cognitive map updating…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Active Visual Exploration"
  - "Spatial Reasoning"
  - "Dynamic Cognitive Map"
  - "Spatial Assertion Code"
  - "GRPO"
date: 2026-05-08
content_hash: aa0a145c3d3bb91a
---

# Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models

**Conference**: ICML 2026  
**arXiv**: [2606.02459](https://arxiv.org/abs/2606.02459)  
**Code**: https://github.com/dw-dengwei/active-spatial-reasoning.git  
**Area**: Multimodal VLM / Spatial Reasoning  
**Keywords**: Active Visual Exploration, Spatial Reasoning, Dynamic Cognitive Map, Spatial Assertion Code, GRPO  

## TL;DR
This paper transforms VLM spatial reasoning from a "passive observation" paradigm into an agentic workflow of "question-driven active framing, cognitive map updating, and verification via executable spatial assertions." By fine-tuning Qwen2.5-VL-3B with dense rewards, the method achieves 80.5% overall accuracy on MindCube-Tiny, specifically improving the Rotation subset to 85.0%.

## Background & Motivation
**Background**: Multimodal VLMs are capable of document understanding and basic spatial relation processing. However, most methods feed all images into the context simultaneously, forcing the model to reason over static inputs. Spatial VLMs like MindCube and 3DThinker introduce cognitive maps or 3D reconstruction auxiliary tasks, but their perception remains passive.

**Limitations of Prior Work**: In real embodied scenarios, agents rarely observe the full environment at once. They must select views based on the problem and assemble fragmented observations into a continuous spatial memory. Passive input of all views is computationally expensive and prone to noise; furthermore, RLVR/GRPO training usually relies on sparse rewards (final answer correctness), failing to pinpoint specific spatial reasoning errors.

**Key Challenge**: Spatial reasoning requires verifiable intermediate steps, but VLM natural language reasoning is open-ended text, making it difficult to judge the correctness of each intermediate relationship. Relying solely on final rewards may lead to surface-level pattern matching, while using an LLM judge introduces hallucinations and overconfidence into the reward loop.

**Goal**: The authors aim to solve three sub-problems: enabling the VLM to actively select relevant views, forming an updatable spatial memory from multiple observations, and providing programmatic checks for intermediate spatial relations to offer denser feedback than 0/1 final rewards.

**Key Insight**: Drawing inspiration from biological cognitive maps, the paper integrates observed objects, camera positions, and orientations into a unified top-down coordinate system. It translates natural language relations (e.g., "A is to the left of B in view X") into Python expressions executed against the cognitive map.

**Core Idea**: Use a dynamic cognitive map as spatial memory and Spatial Assertion Code (SAC) to turn intermediate reasoning into executable assertions. Combine retrieval, map updates, and SAC correctness into dense rewards to reinforce active spatial reasoning.

## Method
The core contribution is treating spatial QA as a sequential decision-making process. At each step, the model evaluates the current cognitive map state, decides the next view to retrieve based on the question, updates the map, and eventually terminates to output an answer. During training, the model is required to generate SAC alongside natural language explanations to verify internal consistency.

### Overall Architecture
Inputs include the question $\mathcal{Q}$, candidate views $V=\{v_n\}$, and transformation descriptions $E$. The system state $s_t$ is the dynamic cognitive map, and the action $a_{t+1}$ is a VLM-generated retrieval code (e.g., `retrieve(3)`). After execution, the model observes $v_{t+1}$ and updates the map $s_{t+1}=\mathcal{P}_\theta(s_t,v_{t+1},a_{t+1})$. This loop continues until a termination token is generated.

The dynamic cognitive map maintains two sets in a unified top-down coordinate system: objects $\mathcal{O}_t=\{(o_i,\mathbf{p}_i,\mathbf{d}_i)\}$ (category, position, orientation) and views $\mathcal{V}_t=\{(v_j,\mathbf{c}_j,\mathbf{f}_j)\}$ (camera position and orientation). This ensures new observations are integrated geometrically rather than just appended as text.

Training comprises two stages: SFT for cold-starting (retrieval, map updating, SAC generation) and RFT using GRPO for reinforcement fine-tuning.

### Key Designs
1. **Active Exploratory View Retrieval**:
	- **Function**: Enables the VLM to iteratively select the most relevant views instead of consuming all views at once.
	- **Mechanism**: Retrieval actions are written as parsable Python-like actions (e.g., `retrieve(k)`). The model can perform multi-round exploration or stop early.
	- **Design Motivation**: In "Rotation" type problems, views might lack shared visual anchors. Sequential exploration allows the model to stitch spatial relations along an egocentric reference frame.

2. **Dynamic Cognitive Map as Structured Long-term Memory**:
	- **Function**: Preserves scene layout using object/camera poses instead of raw text context.
	- **Mechanism**: The map state $s_t=\{\mathcal{O}_t,\mathcal{V}_t\}$ is updated after each observation. Reasoning occurs by querying the unified coordinates rather than re-reading long multimodal histories.
	- **Design Motivation**: Pure text context loses precise geometric structure. A top-down map converts cross-view observations into accumulatable geometric states.

3. **Spatial Assertion Code (SAC) and Dense Rewards**:
	- **Function**: Converts intermediate spatial relations into executable Python expressions to verify reasoning steps.
	- **Mechanism**: A statement like "from view 4, object 1 is to the left of object 0" corresponds to `obj1 in obj0.left(view=v4)`. The reward $R_{SAC}$ is calculated based on the execution accuracy of these assertions against the map.
	- **Design Motivation**: SAC transforms uncontrollable natural language steps into computable signals, addressing the sparse reward problem in RL.

### Loss & Training
SFT follows standard cross-entropy: $\mathcal{L}_{SFT}=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{SFT}}[\log p_\theta(y|x)]$.

RFT utilizes GRPO. The reward function is gated by final answer correctness: if the answer is wrong, the total reward is 0. If correct, it includes $R_{retrieval}$, $R_{cogmap}$, and $R_{SAC}$:
$$R=\mathbb{1}_{correct}[\mathbb{1}_{correct}+w(R_{retrieval}+R_{cogmap}+R_{SAC})]$$
$R_{SAC}$ measures assertion accuracy, $R_{cogmap}$ penalizes distance from ground truth map states, and $R_{retrieval}$ checks view relevance.

## Key Experimental Results

### Main Results
On MindCube-Tiny, the method (based on Qwen2.5-VL-3B-Instruct) is compared against general and spatial-specialized VLMs. A significant leap is observed in the Rotation subset.

| Method | Perception/Memory | Overall ↑ | Rotation ↑ | Among ↑ | Around ↑ |
|:-------|:------------------|:----------|:-----------|:--------|:---------|
| Qwen2.5-VL-3B-Instruct | Passive | 33.21 | 37.37 | 33.26 | 30.34 |
| GPT-4o | Passive | 38.81 | 32.65 | 40.17 | 29.16 |
| MindCube-Qwen2.5-VL-3B | Passive + Static Map | 70.7 | 48.0 | 79.2 | 68.4 |
| 3DThinker-Qwen2.5-VL-3B | Passive + 3D Recon | 75.2 | 55.5 | 81.8 | 75.2 |
| **Ours** | **Active + Dynamic Map** | **80.5** | **85.0** | **81.0** | **75.6** |

### Ablation Study
The study analyzes the perception paradigm, reward components, and memory format.

| Configuration | Key Metric | Description |
|:--------------|:-----------|:------------|
| Passive RFT | Rotation Acc. 27.5 | No step-by-step active framing |
| Active RFT | Rotation Acc. 38.5 | Active perception alone provides +11.0 gain |
| Full reward | Overall Acc. 80.4 | Full combination of dense rewards |
| w/o $R_{SAC}$ | Overall Acc. 70.2 | SAC intermediate reward is the most critical (-10.2) |
| Context memory | Acc. 50.9 | Using raw history instead of cognitive map |
| Cognitive map | Acc. 54.2 | Structured map provides +3.3 over raw context |

### Key Findings
- **Rotation Subset Dominance**: Previous SOTA reached 55.5, while this method reaches 85.0, proving that active exploration and coordinate memory effectively solve view-stitching challenges.
- **SAC Value**: Ablation shows $R_{SAC}$ causes the largest drop, indicating that executable intermediate steps are more valuable than simple text explanations.
- **SFT-RL Synergy**: RL improves performance from 67.2 (SFT) to 80.5, suggesting RL refines agentic behavior rather than teaching formats from scratch.

## Highlights & Insights
- The decomposition of spatial reasoning into "Exploration-Memory-Verification" interfaces is highly modular and clear.
- **SAC Practicality**: It avoids requiring full code generation, focusing only on key spatial relations as boolean expressions, which is more robust for QA.
- **Correctness Gating**: Only rewarding intermediate steps when the final answer is correct prevents the model from "reward hacking" the map or SAC components.
- The dynamic cognitive map concept is transferable to robotics, indoor navigation, and multi-view video understanding.

## Limitations & Future Work
- **Reliance on Metadata**: The method depends on available view transformations and map supervision. Real-world robotics will face pose estimation and occlusion noise.
- **SAC Expressivity**: Human-written relational APIs might not cover continuous geometric quantities or complex object deformations.
- **Training Cost**: Requires significant resources (SFT ~22h, RL ~1 day on 8x A100).
- **Benchmark Diversity**: Primarily validated on MindCube-Tiny; testing on real-world embodied benchmarks or closed-loop robot systems is a necessary next step.

## Related Work & Insights
- **vs MindCube**: MindCube uses cognitive maps but in a static/passive manner; this work treats the map as part of a sequential decision loop.
- **vs 3DThinker**: 3DThinker uses reconstruction as an auxiliary task; this work emphasizes active retrieval and intermediate verification for information-gathering scenarios.
- **vs RLVR/GRPO**: Most existing work uses final answers as verifiable rewards; this work extends verifiability to intermediate steps via SAC.
- **Insight**: For VLM tasks requiring multi-step tool use, prioritizing structured intermediate memory and executable assertions is more effective than relying solely on final outcome supervision.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Distinctive design using SAC for verifiable intermediate RL signals.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Strong on MindCube, but lacks real embodied scenario validation.
- **Writing Quality**: ⭐⭐⭐⭐ Logical flow, though some tables are dense.
- **Value**: ⭐⭐⭐⭐⭐ Significant utility for agentic VLM and verifiable reward design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICML 2026\] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models](circle-rope_cone-like_decoupled_rotary_positional_embedding_for_large_vision-lan.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[ICML 2026\] The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)
- [\[ICLR 2026\] Spatial-DISE: A Unified Benchmark for Evaluating Spatial Reasoning in Vision-Language Models](../../ICLR2026/multimodal_vlm/spatial-dise_a_unified_benchmark_for_evaluating_spatial_reasoning_in_vision-lang.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] 3ViewSense: Spatial and Mental Perspective Reasoning from Orthographic Views in Vision-Language Models](3viewsense_spatial_and_mental_perspective_reasoning_from_orthographic_views_in_v.md)
- [\[ICML 2026\] Circle-RoPE: Cone-like Decoupled Rotary Positional Embedding for Vision-Language Models](circle-rope_cone-like_decoupled_rotary_positional_embedding_for_large_vision-lan.md)
- [\[ICML 2026\] Learning GUI Grounding with Spatial Reasoning from Visual Feedback](learning_gui_grounding_with_spatial_reasoning_from_visual_feedback.md)
- [\[ICML 2026\] The Perceptual Bandwidth Bottleneck in Vision-Language Models: Active Visual Reasoning via Sequential Experimental Design](the_perceptual_bandwidth_bottleneck_in_vision-language_models_active_visual_reas.md)
- [\[NeurIPS 2025\] SpatialThinker: Reinforcing 3D Reasoning in Multimodal LLMs via Spatial Rewards](../../NeurIPS2025/multimodal_vlm/spatialthinker_reinforcing_3d_reasoning_in_multimodal_llms_via_spatial_rewards.md)

</div>

<!-- RELATED:END -->
