---
title: >-
  [Paper Note] V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents
description: >-
  [CVPR 2025][Multimodal VLM][Video Stylization] This work introduces V-Stylist, a video stylization system based on multi-agent collaboration and reflection of MLLMs. By coordinating three agent roles—Video Parser (video shot segmentation), Style Parser (style tree search), and Style Artist (multi-round self-reflective rendering)—V-Stylist achieves state-of-the-art performance on complex transition videos and open-domain style descriptions, outperforming FRESCO by 6.05% on ove…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Video Stylization"
  - "Multi-Agent Collaboration"
  - "MLLM Self-Reflection"
  - "Style Tree Search"
  - "AnimateDiff+ControlNet"
date: 2026-05-08
content_hash: 3ad60ef6601d10ea
---

# V-Stylist: Video Stylization via Collaboration and Reflection of MLLM Agents

**Conference**: CVPR 2025  
**arXiv**: [2503.12077](https://arxiv.org/abs/2503.12077)  
**Code**: To be confirmed  
**Area**: Multimodal VLM  
**Keywords**: Video Stylization, Multi-Agent Collaboration, MLLM Self-Reflection, Style Tree Search, AnimateDiff+ControlNet

## TL;DR
This work introduces V-Stylist, a video stylization system based on multi-agent collaboration and reflection of MLLMs. By coordinating three agent roles—Video Parser (video shot segmentation), Style Parser (style tree search), and Style Artist (multi-round self-reflective rendering)—V-Stylist achieves state-of-the-art performance on complex transition videos and open-domain style descriptions, outperforming FRESCO by 6.05% on overall metrics.

## Background & Motivation
**Background**: Significant progress has been made in video stylization; however, existing methods struggle to handle videos featuring complex transitions and fail to stylize videos based on open-domain style descriptions.

**Limitations of Prior Work**: (1) Style consistency is difficult to maintain in videos with complex transitions. (2) User style descriptions are often vague, making precise matching challenging.

**Key Challenge**: Simultaneously addressing both video structure understanding (transition analysis) and open-domain style understanding (vague description matching).

**Goal**: To build a general video stylization system that automatically processes complex transitions and vague style descriptions.

**Key Insight**: Emulating the professional workflows of humans—segmenting shots first, choosing style next, and refining details at last—where each step is managed by a specialized MLLM agent.

**Core Idea**: A three-role MLLM agent system (shot parser / style selector / renderer) + style tree search + multi-round self-reflection to simulate professional workflows.

## Method

### Overall Architecture
The system takes a user video and style descriptions as inputs and outputs the stylized video. Three MLLM agents collaborate under specialized roles: the Video Parser handles video shot segmentation, the Style Parser matches style models, and the Style Artist executes style rendering.

### Key Designs

1. **Video Parser (Video Analysis Agent)**:

    - Function: Decomposes the video into structured shot sequences and generates diffusion-compatible prompts for each shot.
    - Mechanism: AutoShot detects transitions $\rightarrow$ Qwen2-VL generates shot captions $\rightarrow$ Mistral8x7B translates captions into diffusion prompts.
    - Three keyframes are extracted per shot.
    - Design Motivation: Processing shot-by-shot effectively tackles complex transitions, while the caption-to-prompt translation ensures that the diffusion model correctly understands the content.

2. **Style Parser (Style Analysis Agent)**:

    - Function: Precisely matches the optimal style model from vague user style descriptions.
    - Mechanism: **Style Tree + Tree-of-Thought Search**. Style models are organized into a 3-layer tree structure (root $\rightarrow$ 2 main categories [Artistic/Realistic] $\rightarrow$ 25 models covering 17 styles). Mistral8x7B first extracts style preferences $\rightarrow$ A voting-based descent search is conducted layer-by-layer by 5 style experts and 1 chairman: $\mathcal{D}_{l+1} = LLM(\mathcal{D}_l | \mathcal{S}, \mathcal{T})$.
    - Style Coverage: 17 styles including oil painting, Japanese anime, pixel art, Western realism, etc.
    - Design Motivation: Direct matching easily fails when users input vague descriptions like "cinematic," but tree search can first localize to "Realistic" and then refine further to specific style models.

3. **Style Artist (Style Rendering Agent, featuring self-reflection)**:

    - Function: Renders video shots using the matched style model, iteratively optimizing the outputs through multi-round MLLM self-evaluation.
    - Mechanism: Video rendering is formulated using SD v1.5 + AnimateDiff (for temporal consistency) + 4 ControlNets (tile/depth/softedge/lineart). After rendering, an MLLM rates the output (0-100). If the score is $\ge 60$, it is accepted; otherwise, the MLLM generates new ControlNet weights for re-rendering, with a maximum of 3 rounds.
    - Equation: $\mathcal{Y}_t = \mathcal{M}_L(\mathcal{X}_t, \mathcal{P}_t | \mathcal{C}_{1:N} \cdot \mathcal{W}_{1:N})$
    - Design Motivation: A single render is often suboptimal. The self-reflection mechanism adaptively adjusts ControlNet weights to balance content preservation and style expression.

## Key Experimental Results

### TVSBench (50 videos, average 30 seconds at 30FPS, 17 styles)

| Method | CLIP-T | Aesthetic-V | Distortion-V | **Overall** |
|------|:---:|:---:|:---:|:---:|
| ControlVideo | 0.263 | 0.587 | 0.587 | 0.541 |
| FRESCO | 0.239 | 0.527 | 0.556 | 0.541 |
| Rerender | 0.206 | 0.412 | 0.404 | 0.502 |
| FLATTEN | 0.243 | 0.528 | 0.557 | 0.487 |
| **V-Stylist** | **0.267** | **0.591** | **0.745** | **0.601** |

### Ablation Study

| Configuration | CLIP-T | Distortion-V | Description |
|------|:---:|:---:|------|
| Baseline (No agent) | 0.256 | 0.576 | Direct generation via SD+ControlNet |
| + Video Parser | 0.263 | 0.584 | With shot segmentation and prompt translation |
| + Style Parser | 0.266 | 0.575 | With style tree search |
| **Full V-Stylist** | **0.266** | **0.590** | With self-reflective rendering |

The self-reflection in Style Artist contributes the most—**boosting video quality and style alignment by 25.16%**.

### Key Findings
- V-Stylist outperforms FRESCO by 6.05% and ControlVideo by 4.51% on overall metrics.
- The advantage is most significant in video-level distortion (Distortion-V): 0.745 vs the second-best 0.587 (+27%), indicating that the self-reflection mechanism contributes significantly to temporal consistency.
- Style tree search is more robust than direct matching when handling vague descriptions.
- Hardware: 8×RTX A6000, parameters top-k=10, top-p=0.95, temp=0.7.

## Highlights & Insights
- **Mimicking professional human workflows** is an elegant design concept—segmentation, style selection, and detail adjustment are standard practices for professional video editors.
- The combination of **Style Tree + Tree-of-Thought Search** provides a fresh approach for precisely matching vague user requirements.
- The multi-round self-reflection mechanism can be effectively transferred to other visual generation tasks.

## Limitations & Future Work
- Dependent on pre-built style trees and style model libraries.
- Relatively high computational overhead due to multi-agent collaboration and multi-round reflection.
- Maintaining style consistency on exceptionally long videos may still be challenging.

## Rating
- Novelty: ⭐⭐⭐⭐ Highly novel three-role collaboration and style tree search.
- Experimental Thoroughness: ⭐⭐⭐⭐ New benchmark, quantitative comparisons, and user studies.
- Writing Quality: ⭐⭐⭐⭐ Very clear and well-structured writing.
- Value: ⭐⭐⭐⭐ Highly valuable reference for multi-agent visual generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Efficient Motion-Aware Video MLLM](efficient_motion-aware_video_mllm.md)
- [\[CVPR 2025\] Scalable Video-to-Dataset Generation for Cross-Platform Mobile Agents](scalable_video-to-dataset_generation_for_cross-platform_mobile_agents.md)
- [\[CVPR 2025\] Visual and Semantic Prompt Collaboration for Generalized Zero-Shot Learning](visual_and_semantic_prompt_collaboration_for_generalized_zero-shot_learning.md)
- [\[CVPR 2025\] DPC: Dual-Prompt Collaboration for Tuning Vision-Language Models](dpc_dual-prompt_collaboration_for_tuning_vision-language_models.md)
- [\[CVPR 2025\] From Multimodal LLMs to Generalist Embodied Agents: Methods and Lessons](from_multimodal_llms_to_generalist_embodied_agents_methods_and_lessons.md)

</div>

<!-- RELATED:END -->
