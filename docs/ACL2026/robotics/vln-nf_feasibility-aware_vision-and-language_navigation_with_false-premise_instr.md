---
title: >-
  [Paper Note] VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions
description: >-
  [ACL 2026][Robotics][Vision-Language Navigation] VLN-NF is the first benchmark requiring VLN agents to identify false-premise instructions and output NOT-FOUND in 3D partially observable environments. The paper also proposes REV-SPL evaluation metric and ROAM two-stage hybrid framework, achieving 6.1 REV-SPL (+45% over supervised baselines).
tags:
  - ACL 2026
  - Robotics
  - Vision-Language Navigation
  - False Premise
  - NOT-FOUND
  - Embodied Exploration
  - Feasibility Awareness
content_hash: 0c70fa9b2ad63116
---

# VLN-NF: Feasibility-Aware Vision-and-Language Navigation with False-Premise Instructions

**Conference**: ACL 2026
**arXiv**: [2604.10533](https://arxiv.org/abs/2604.10533)
**Code**: [https://vln-nf.github.io/](https://vln-nf.github.io/)
**Area**: Robotics & Embodied AI
**Keywords**: Vision-Language Navigation, False Premise, NOT-FOUND, Embodied Exploration, Feasibility Awareness

## TL;DR
VLN-NF is the first benchmark requiring VLN agents to identify false-premise instructions and output NOT-FOUND in 3D partially observable environments. The paper also proposes REV-SPL evaluation metric and ROAM two-stage hybrid framework, achieving 6.1 REV-SPL (+45% over supervised baselines).

## Method

### Key Designs

1. **Dataset Construction Pipeline (Rewrite + Verify)**: LLM Rewriter generates semantically fluent but factually incorrect instructions by replacing target objects with plausible alternatives absent from the target room. VLM Verifier confirms via open-vocabulary detection. Human audit error rate <2%.

2. **REV-SPL Evaluation Metric**: Jointly evaluates navigation efficiency, exploration coverage, and FOUND/NOT-FOUND decision correctness. Penalizes premature stopping and incorrect decisions.

3. **ROAM Two-Stage Hybrid Framework**: Stage 1 uses supervised DUET model for room-level navigation; Stage 2 uses LLM/VLM for in-room exploration with free-space clearance prior guidance.

## Key Experimental Results

| Method | Type | REV-SPL |
|--------|------|---------|
| DUET + VLN-NF | Supervised | 4.2 |
| NaviLLM | LLM-based | 1.0 |
| **ROAM** | Hybrid | **6.1** |

## Highlights & Insights
- Fills VLN reliability gap: first systematic study of false-premise navigation in 3D partially observable environments
- Two-stage decomposition strategy is transferable to other embodied tasks requiring decisions under uncertainty

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Towards Open Environments and Instructions: General Vision-Language Navigation via Fast-Slow Interactive Reasoning](../../CVPR2026/robotics/towards_open_environments_and_instructions_general_vision-language_navigation_vi.md)
- [\[ICLR 2026\] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](../../ICLR2026/robotics/janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](../../CVPR2026/robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](../../CVPR2026/robotics/decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[AAAI 2026\] Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation](../../AAAI2026/robotics/recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)

<!-- RELATED:END -->
