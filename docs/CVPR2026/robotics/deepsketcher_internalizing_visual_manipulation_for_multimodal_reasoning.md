---
title: >-
  [Paper Note] DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning
description: >-
  [CVPR 2026][Robotics][Visual Reasoning] This paper proposes the DeepSketcher suite — comprising a 31k high-quality interleaved image-text CoT dataset built via code rendering and a self-contained Embedding Editor model —…
tags:
  - "CVPR 2026"
  - "Robotics"
  - "Visual Reasoning"
  - "Interleaved Image-Text Reasoning"
  - "Visual Thinking"
  - "Embedding Editor"
  - "Code Rendering"
date: 2026-05-08
content_hash: 9afc09730875f8db
---

# DeepSketcher: Internalizing Visual Manipulation for Multimodal Reasoning

**Conference**: CVPR 2026
**arXiv**: [2509.25866](https://arxiv.org/abs/2509.25866)  
**Code**: [GitHub](https://github.com/MiliLab/DeepSketcher)  
**Area**: Robotics
**Keywords**: Visual Reasoning, Interleaved Image-Text Reasoning, Visual Thinking, Embedding Editor, Code Rendering

## TL;DR

This paper proposes the DeepSketcher suite — comprising a 31k high-quality interleaved image-text CoT dataset built via code rendering and a self-contained Embedding Editor model — enabling VLMs to generate "visual thoughts" directly in the visual embedding space for multimodal reasoning without relying on any external tools.

## Background & Motivation

"Thinking with images" represents an emerging paradigm for VLM reasoning, in which models manipulate visual inputs (cropping, zooming, drawing auxiliary lines, etc.) during inference to achieve deeper visual understanding. However, existing approaches face three core contradictions:

1. **Limited action space**: Methods such as VILASR support only predefined operation sets (zoom, crop), offering poor flexibility.
2. **Difficult spatial localization**: Methods such as DeepEyes learn operations via RL but rely on precise coordinate regression, which introduces substantial training-data noise.
3. **Extremely high training difficulty**: Methods such as Bagel attempt to unify generation and reasoning, but the "imagination" space is too large and their effectiveness has not been thoroughly validated.

DeepSketcher takes code-rendered VQA data as its starting point and proposes a complementary perspective: all images are produced through code rendering, and visual manipulations are realized by modifying the underlying code — making them precise, reproducible, and free of spatial localization noise.

## Method

### Overall Architecture

Code-rendered image + question → VLM generates reasoning text + editing instructions → Embedding Editor operates in the visual embedding space → updated visual embeddings are injected into the context → reasoning continues → final answer.

### Key Designs

1. **Data Construction in Code Space**:
    - **Function**: Generate high-quality interleaved image-text CoT training trajectories.
    - **Mechanism**: A dual-agent collaboration system in which a Solver LLM performs reasoning and issues manipulation requests, while a Code Editor LLM modifies the rendering code and regenerates the image, forming a closed loop of "reasoning → instruction → code editing → rendering → reasoning."
    - **Design Motivation**: Editing in code space is precise and controllable, avoiding the localization noise of pixel-level operations and the uncontrollability of generative models.

2. **Embedding Editor**:
    - **Function**: Execute visual operations directly in the visual embedding space without external tool calls.
    - **Mechanism**: A Q-Former-style cross-attention architecture in which visual tokens serve as queries, and the hidden states of editing instructions — after adaptive pooling — serve as keys and values; cross-attention followed by FFN updates the visual embeddings.
    - **Design Motivation**: Eliminate dependence on code execution, external tools, and repeated image encoding, enabling more flexible "thinking with images."

3. **Three-Stage Progressive Training**:
    - **Function**: Progressively decouple the model's dependence on ground-truth visual inputs.
    - **Mechanism**: Phase 1 (reasoning warm-up using GT image features) → Phase 2 (Editor training with L1 loss aligning predicted embeddings to GT edited image embeddings, other modules frozen) → Phase 3 (joint adaptation, unfreezing the LLM backbone to accommodate Editor outputs).
    - **Design Motivation**: Direct end-to-end training causes the Editor to produce noisy embeddings that interfere with reasoning; progressive training ensures each component stabilizes before the next stage.

### Loss & Training

- **Phase 1**: Standard autoregressive language modeling loss (text tokens only).
- **Phase 2**: L1 embedding reconstruction loss + conditional language modeling loss.
- **Phase 3**: Same objectives as Phase 2, but with the LLM backbone unfrozen.

## Key Experimental Results

### Main Results (Multimodal Reasoning Benchmarks)

| Model | MathVerse | MathVision | MathVista | LogicVista | WeMath | Avg. |
|-------|-----------|------------|-----------|------------|--------|------|
| Qwen2.5-VL-7B | 41.1 | 27.0 | 68.2 | 39.8 | 34.3 | 42.1 |
| DeepEyes-7B | 42.2 | 26.6 | 70.1 | 47.7 | 38.9 | 45.1 |
| Mirage-7B (Inner Visual) | 27.3 | 28.6 | 63.7 | 40.7 | 16.7 | 35.4 |
| DeepSketcher-7B | 43.2 | 32.3 | 69.1 | 48.1 | 37.1 | 46.0 |

### Ablation Study

| Stage | Setting | MathVerse | WeMath | Indicator-500 |
|-------|---------|-----------|--------|---------------|
| Phase 2 | Text-only baseline | 37.2 | 28.3 | 38.3 |
| Phase 2 | +Editor | 41.6 | 37.5 | 33.8 |
| Phase 3 | Text-only baseline | 38.1 | 31.2 | 37.5 |
| Phase 3 | +Editor | 43.2 | 37.1 | 40.5 |

### Key Findings

- Improvements are most pronounced on geometry and counting tasks (MathVision +5.3), while gains on tasks involving symbolic manipulation are more modest.
- The dual-agent collaboration (Solver + Code Editor) substantially outperforms standalone reasoning (GPT-4.1 pass@8: 0.72 → 0.80).
- Difference-map visualizations of the Embedding Editor show that edited regions are highly consistent with the corresponding instructions.

## Highlights & Insights

- The code-space data construction is an elegant solution: precise, reproducible, and verifiable, avoiding the noise inherent in coordinate regression and image generation.
- The Embedding Editor's design of operating in embedding space is distinctive — it modifies visual representations directly rather than generating pixel-level images.
- As the strongest method among "Inner Visual Thought VLMs," this work demonstrates the feasibility of internalizing visual manipulation.
- The 31k dataset spans multiple disciplines (mathematics, physics, chemistry, etc.), is high-quality, and is scalable.

## Limitations & Future Work

- Code-rendered data constrains the scope of application (primarily structured graphics); natural image scenarios are not covered.
- The editing quality of the Embedding Editor still falls short of GT code-rendered images (a gap remains on Indicator-500).
- The approach is slower than tool-calling methods due to the additional forward pass through the Editor.
- Unfreezing the LLM in Phase 3 sometimes leads to performance degradation on Indicator-500, indicating incomplete adaptation.

## Related Work & Insights

- **vs. VILASR / DeepEyes**: These rely on predefined operation sets and coordinate regression; DeepSketcher offers an open action space without requiring coordinates.
- **vs. Mirage / Bagel**: These edit images in a compressed latent space; DeepSketcher operates in the visual token space, preserving richer semantic information.
- **vs. Visual Sketchpad**: That work depends on external tools for execution; DeepSketcher internalizes the entire manipulation pipeline.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Dual innovation: code-space data construction + embedding-space visual editing.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-benchmark coverage with detailed ablations, though evaluation on natural image scenarios is absent.
- **Writing Quality**: ⭐⭐⭐⭐ Method pipeline is clearly presented; the three-stage training design is well motivated.
- **Value**: ⭐⭐⭐⭐ Provides a new data and modeling pathway for the "thinking with images" paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Probabilistic Concept Graph Reasoning for Multimodal Misinformation Detection](probabilistic_concept_graph_reasoning_for_multimodal_misinformation_detection.md)
- [\[CVPR 2026\] Diagnose, Correct, and Learn from Manipulation Failures via Visual Symbols](diagnose_correct_and_learn_from_manipulation_failures_via_visual_symbols.md)
- [\[CVPR 2026\] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation](maniparena_comprehensive_real-world_evaluation_of_reasoning-oriented_generalist_.md)
- [\[CVPR 2026\] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation](palm_progress-aware_policy_learning_via_affordance_reasoning_for_long-horizon_ro.md)
- [\[CVPR 2026\] FORCE: Transferable Visual Jailbreaking Attacks via Feature Over-Reliance CorrEction](force_transferable_visual_jailbreaking_attacks_via_feature_over_reliance_correct.md)

</div>

<!-- RELATED:END -->
