---
title: >-
  [Paper Note] See, Symbolize, Act: Grounding VLMs with Spatial Representations for Better Gameplay
description: >-
  [AAAI 2026][Multimodal VLM][Symbolic grounding] This paper systematically evaluates the effect of symbolic spatial representations (object coordinates) on VLM gameplay…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Symbolic grounding"
  - "VLM game agents"
  - "spatial reasoning"
  - "object detection"
  - "Atari"
date: 2026-05-08
content_hash: 243eb9ee91aef528
---

# See, Symbolize, Act: Grounding VLMs with Spatial Representations for Better Gameplay

**Conference**: AAAI 2026
**arXiv**: [2603.11601](https://arxiv.org/abs/2603.11601)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Symbolic grounding, VLM game agents, spatial reasoning, object detection, Atari

## TL;DR
This paper systematically evaluates the effect of symbolic spatial representations (object coordinates) on VLM gameplay, finding that symbolic information is beneficial only when detection is accurate; when VLMs self-extract symbols, effectiveness depends on model capability and scene complexity, while visual frames remain indispensable throughout.

## Background & Motivation

### State of the Field
VLMs are increasingly employed to build general-purpose AI agents that must not only understand visual scenes but also make decisions in interactive environments. As VLMs transition from passive perception to interactive decision-making (e.g., robotics, embodied AI, game environments), they face the challenge of precise spatial understanding—a capability that current VLMs do not yet reliably provide.

### Limitations of Prior Work

**Atari games** offer a controlled environment for studying spatial reasoning challenges. Games such as Pong, Breakout, and Space Invaders require precise tracking of paddle, ball, and enemy positions. Under these conditions, current VLMs frequently exhibit:
- **Object misidentification**
- **Repetitive ineffective actions**
- **Failure in precise control**

### Root Cause
- **Fine-tuning approaches** (Zhai et al. 2024): Fine-tune VLMs on thousands of game trajectories, but at the cost of zero-shot generalization
- **Vision-only approaches** (Atari-GPT): Use only visual frames, with spatial reasoning remaining a bottleneck
- **Critical gap**: No systematic study has evaluated VLM **accuracy in recognizing objects and coordinates**, nor how coordinate precision **affects decision-making**

### Starting Point
If VLMs are provided with both visual frames and **symbolic representations** of the scene (object names + precise coordinates), can their interactive decision-making be improved? The paper isolates the contribution of each factor through a systematic comparison of four pipelines: frames only, frames + self-extracted symbols, frames + ground-truth symbols, and symbols only.

## Method

### Overall Architecture
Four experimental pipelines are designed and evaluated across three environments (Atari, VizDoom, and AI2-THOR) using three state-of-the-art VLMs (Claude-4-Sonnet, GPT-4o, Gemini-2.5-Pro).

### Key Designs

1. **Four Experimental Pipelines**:

    - **F+S-GT (Frames + Ground-Truth Symbols)**: Object coordinates read directly from game RAM (OCAtari), serving as the performance upper bound
    - **F (Frames Only)**: VLM receives only raw game frames, testing pure visual spatial reasoning
    - **F+S-self (Frames + Self-Extracted Symbols)**: Two-stage pipeline — VLM first extracts object coordinates from frames, then makes decisions using both frames and coordinates
    - **S-GT (Symbols Only)**: Only RAM-derived object coordinates are provided without visual frames, testing the independent contribution of symbolic information
    - **Design Motivation**: Controlled variable methodology to isolate the respective contributions of visual frames, symbol accuracy, and self-extraction capability

2. **Evaluation Metrics**:

    - **Gameplay metrics**: Cumulative reward over 600 frames, normalized to the 0–100 range
    - **Detection metrics**: F1 score (object recognition accuracy) and IoU (coordinate overlap), evaluated over 100 frames
    - **Design Motivation**: Decouple gameplay performance from perceptual quality to analyze causal relationships

3. **Ablation Study Design**:

    - **Resolution ablation**: Tests the effect of four resolutions ranging from 160×210 to 1280×720 on detection quality
    - **Noise ablation**: Injects Gaussian noise into ground-truth coordinates as $x' = x + \mathcal{N}(0, \sigma \times W)$, with σ ranging from 0 to 0.4, across 10 seeds and 300 frames each
    - **Design Motivation**: Quantify the "tipping point at which symbols shift from helpful to harmful"

### Loss & Training
Fully **zero-shot** setting with no fine-tuning. General-purpose prompt templates are used, containing no task-specific instructions or strategy hints.

## Key Experimental Results

### Main Results (Atari Environment — Cumulative Reward over 600 Frames)

| Model | Pipeline | Pong | Breakout | Space Invaders |
|-------|----------|------|----------|----------------|
| Claude-4-Sonnet | F (Frames Only) | -16.0 | 0.0 | 80.0 |
| Claude-4-Sonnet | F+S-self | **-3.0** | **12.0** | **150.0** |
| Claude-4-Sonnet | F+S-GT (Upper Bound) | -1.0 | 12.0 | 175.0 |
| GPT-4o | F (Frames Only) | -5.0 | 7.5 | **130.0** |
| GPT-4o | F+S-self | -6.5 | 8.0 | 65.0 ↓ |
| GPT-4o | F+S-GT (Upper Bound) | -3.0 | 13.0 | 185.0 |
| Gemini-2.5-Pro | F (Frames Only) | -7.0 | 7.0 | **95.0** |
| Gemini-2.5-Pro | F+S-self | -3.0 | 10.0 | 80.0 ↓ |
| Gemini-2.5-Pro | F+S-GT (Upper Bound) | -1.0 | 12.0 | 170.0 |

Key finding: Claude achieves substantial improvements with self-extracted symbols, whereas GPT-4o and Gemini actually decline in the more complex scenario (Space Invaders).

### Ablation Study 1: Object Detection Quality

| Model | F1 Score | IoU |
|-------|----------|-----|
| Claude-4-Sonnet | **0.715** | **0.533** |
| Gemini-2.5-Pro | 0.189 | 0.202 |
| GPT-4o | 0.124 | 0.128 |

**Claude's detection F1 is 5.8× that of GPT-4o**, explaining why only Claude benefits from self-extracted symbols.

### Ablation Study 2: Effect of Coordinate Noise on Breakout Performance

| Noise Level σ | Claude Reward | GPT-4o Reward | Note |
|---------------|---------------|---------------|------|
| 0.0 (no noise) | 5.0 | 5.0 | Baseline |
| 0.1 (~16–20 px error) | 4.3 | 4.0 | Already 30–40% drop |
| 0.2 (~32–40 px error) | 3.4 | 3.0 | No advantage |
| 0.3 | 3.4 | 2.3 | Below frames-only |
| 0.4 (~64–80 px error) | 2.8 | 2.6 | Severe degradation |

### Extended Environments (VizDoom + AI2-THOR)

| Model | VizDoom F | VizDoom F+S-self | AI2-THOR F | AI2-THOR F+S-self |
|-------|-----------|-----------------|------------|-------------------|
| Claude-4-Sonnet | 5 | **9** | -1.0 | **2.0** |
| GPT-4o | **12** | 8↓ | 7.0 | **9.0** |
| Gemini-2.5-Pro | **11** | 4↓ | 5.0 | 1.0↓ |

Results in 3D environments confirm the core finding: symbolic information is beneficial if and only if detection is accurate.

### Key Findings

1. **"Symbols are helpful" is not unconditional**: Self-extracted symbols are beneficial only when detection accuracy is sufficiently high (Claude's F1 = 0.715)
2. **Visual frames are indispensable**: Even with perfect symbolic coordinates, removing visual frames causes performance to collapse (the symbols-only pipeline performs worst)
3. **Extremely low noise tolerance**: A coordinate error of merely 16–20 pixels causes a 30–40% performance drop
4. **Scene complexity is a critical variable**: Pong (2–4 objects) → success; Space Invaders (20–50 objects) → failure
5. **Resolution is a simple yet effective improvement**: F1 nearly doubles from 0.31 at 160×210 to 0.68 at 1280×720
6. **Perceptual quality is the bottleneck**: The issue lies not in the concept of symbolic grounding itself, but in the insufficient perceptual accuracy of current VLMs

## Highlights & Insights

- **A model study in controlled experimentation**: Four pipelines × three environments × three models constitute an exceptionally systematic design in which each factor is isolated and analyzed
- **Core conclusion is concise and powerful**: "Symbolic grounding helps only when symbols are accurate" — a single sentence that encapsulates the entire paper
- **Reveals perceptual disparities among VLMs**: Claude's object detection substantially outperforms GPT-4o and Gemini, a finding that carries independent value
- **Noise ablation** is the first to quantify the propagation relationship from coordinate error to decision degradation
- Extension to VizDoom and AI2-THOR demonstrates the generality of the findings

## Limitations & Future Work

- **Limited environments**: Despite coverage of three environments, all are game scenarios with a gap from real-world robot control
- **API cost**: Per-frame VLM API calls make real-time gameplay infeasible due to latency and expense
- **Robust symbol extraction methods unexplored**: E.g., hybrid detectors or lightweight fine-tuning
- **Fixed prompt templates**: No attempt to adaptively adjust prompting strategies for varying scene complexities
- **Only three VLMs tested**: Open-source models (e.g., LLaVA, Qwen-VL) are not included

## Related Work & Insights

- **Distinction from Atari-GPT**: This work systematically studies the effect of symbolic information rather than merely performing zero-shot inference with frames
- **Relationship with OCAtari**: OCAtari is utilized to provide ground-truth symbolic coordinates as an experimental upper bound
- **A new perspective on the symbol grounding problem (Harnad 1990)**: The challenge is not symbol–perception alignment but rather the precision of symbol extraction
- PoE-World assumes reliable symbolic input; this paper demonstrates that such an assumption frequently does not hold
- **Implication**: Improving VLMs' spatial perceptual accuracy may be more important than designing more sophisticated reasoning strategies

## Rating
- Novelty: ⭐⭐⭐ (experiment-driven rather than methodologically innovative, though the problem formulation is novel)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (highly systematic controlled experiments + multiple environments + ablation studies)
- Writing Quality: ⭐⭐⭐⭐⭐ (clear structure, well-defined conclusions, rich figures and tables)
- Value: ⭐⭐⭐⭐ (provides an important empirical foundation for VLM agent research)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles](../../CVPR2026/multimodal_vlm/see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[CVPR 2026\] See, Think, Act: Teaching Multimodal Agents to Effectively Interact with GUI by Identifying Toggles (StaR)](../../CVPR2026/multimodal_vlm/see_think_act_teaching_multimodal_agents_to_effectively_interact_with_gui_by_ide.md)
- [\[AAAI 2026\] The Triangle of Similarity: A Multi-Faceted Framework for Comparing Neural Network Representations](the_triangle_of_similarity_a_multi-faceted_framework_for_comparing_neural_networ.md)
- [\[AAAI 2026\] Yes FLoReNce, I Will Do Better Next Time! Agentic Feedback Reasoning for Humorous Meme Detection](yes_florence_i_will_do_better_next_time_agentic_feedback_reasoning_for_humorous_.md)

</div>

<!-- RELATED:END -->
