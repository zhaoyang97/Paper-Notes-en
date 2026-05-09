---
title: >-
  [Paper Note] HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation
description: >-
  [CVPR 2026][Image Generation][Vision-Language Navigation] This paper proposes HaltNav, a hierarchical navigation framework that combines lightweight text-based topological maps (osmAG) for global planning with a VLN model for local execution. A Reactive Visual Halting (RVH) mechanism is introduced to interrupt execution upon encountering unknown obstacles, update the topology, and trigger replanning for detour. The framework achieves significant improvements over baselines in both simulation and real-robot experiments.
tags:
  - CVPR 2026
  - Image Generation
  - Vision-Language Navigation
  - Topological Priors
  - Reactive Halting
  - Hierarchical Navigation Framework
  - MLLM
  - osmAG
date: 2026-05-08
content_hash: 4c8291d0c145d728
---

# HaltNav: Reactive Visual Halting over Lightweight Topological Priors for Robust Vision-Language Navigation

**Conference**: CVPR 2026
**arXiv**: [2603.12696](https://arxiv.org/abs/2603.12696)
**Code**: TBD
**Area**: Image Generation
**Keywords**: Vision-Language Navigation, Topological Priors, Reactive Halting, Hierarchical Navigation Framework, MLLM, osmAG

## TL;DR
This paper proposes HaltNav, a hierarchical navigation framework that combines lightweight text-based topological maps (osmAG) for global planning with a VLN model for local execution. A Reactive Visual Halting (RVH) mechanism is introduced to interrupt execution upon encountering unknown obstacles, update the topology, and trigger replanning for detour. The framework achieves significant improvements over baselines in both simulation and real-robot experiments.

## Background & Motivation
VLN is evolving from step-by-step instruction following toward open-vocabulary, goal-driven autonomous navigation. However: (1) existing methods rely on dense 2D/3D metric maps, which are costly to build and maintain and suffer from modality alignment issues; (2) purely static prior maps are brittle in real deployment—doors may be closed and corridors may be congested, causing execution failures; (3) users typically provide only brief goal descriptions (e.g., "take me to the restroom") rather than step-by-step route instructions. There is a need for a framework that leverages structural priors for long-horizon planning while reactively adapting to local anomalies.

## Core Problem
How to achieve long-horizon VLN using lightweight topological priors under resource-constrained conditions, while maintaining robust navigation in dynamically changing environments?

## Method

### Overall Architecture
A hierarchical semi-Markov decision process: (1) **Macro level**: an LLM-based GGTD performs room-level global planning on osmAG, decomposing the route into door-to-door local sub-instructions; (2) **Micro level**: an end-to-end VLN model (InternVLA-N1) executes local navigation; (3) **Monitor level**: a VLM-based RVH continuously monitors the visual stream, interrupts the execution loop upon detecting blockage, updates the topology, and triggers replanning.

### Key Designs
1. **osmAG Text Topological Prior**: An area graph based on the OpenStreetMap format that represents the environment as a hierarchical structure of area polygons (nodes) and passages/doors (edges). It can be automatically generated from floor plans or CAD files, and its plain-text format is natively compatible with LLM reasoning.
2. **GGTD (Graph-Grounded Task Dispatcher)**: Uses Gemini 3 Flash to directly read the osmAG text representation, combined with the target instruction and navigation history, to output the next macro waypoint as a VLN sub-instruction $m_i = \text{GGTD}(\mathcal{P}(\mathcal{G}_t), I_{target}, \mathcal{H}_{i-1})$.
3. **RVH (Reactive Visual Halting)**: A dual-signal termination function that fuses low-level cumulative collision signals ($\sum c_{t-j} \geq \tau_c$, serving as a physical safety net) with high-level MLLM semantic judgment ($s_{MLLM}(o_t, m_i)$, identifying closed doors, congestion, etc.).
4. **Dynamic Topology Update**: Upon detecting a blockage, the cost of the corresponding passage in the passage-level graph is set to $\infty$, enabling automatic detour—more reliable than prompt injection.
5. **Data Synthesis Pipeline**: A physics engine injects 3D obstacles, and a diffusion model performs inpainting to generate counterfactual blocked scenes. Paired (normal/blocked) data are constructed to fine-tune Qwen-2.5-VL-7B via LoRA SFT.

### Loss & Training
- RVH fine-tuning: standard SFT negative log-likelihood $\mathcal{L}_{SFT} = -\sum \log p_{MLLM}(y_j | y_{<j}, X; \Theta)$
- Low-rank adaptation (LoRA) fine-tuning of Qwen-2.5-VL-7B for obstacle judgment
- The VLN executor (InternVLA-N1) uses pretrained weights without additional training

## Key Experimental Results

| Method | L0 SR-B/O (%) | L0 Drop | L2 SR-B/O (%) | L2 Drop |
|------|---------|---------|---------|---------|
| Navid | 73.1/6.3 | 66.9 | 49.4/0.0 | 49.4 |
| OmniNav | 90.6/12.5 | 78.1 | 54.4/6.3 | 48.1 |
| StreamVLN | 72.5/37.5 | 35.0 | 36.3/18.8 | 17.5 |
| InternVLA-N1 | 58.8/12.5 | 46.3 | 33.1/0.0 | 33.1 |
| **HaltNav** | **79.4/50.0** | **29.4** | **55.6/31.3** | **24.4** |

- **Simulation**: HaltNav achieves the lowest Drop values across all instruction levels, indicating the smallest performance degradation after obstacle injection.
- **Real robot**: HaltNav achieves 56.66% SR on L0-O (vs. StreamVLN 0%, InternVLA-N1 0%); 46.66% SR on L2-O.
- OmniNav achieves the highest B-column SR but requires multi-camera panoramic observation (high hardware cost); HaltNav uses only a single camera.

### Ablation Study
- Without osmAG prior: all baselines collapse under L2 (goal-only instructions), with InternVLA-N1 achieving 0% SR in the real world.
- Without RVH: performance drops sharply after obstacle injection, with substantially higher Drop values.
- Real world vs. simulation: advantages are more pronounced in complex topological environments—university buildings with long corridors and multiple exits are more challenging than simulated home environments.
- L0→L2 instruction degradation: HaltNav's SR drops by ~30% relatively, compared to 40–44% for baselines, demonstrating that topological priors reduce reliance on detailed instructions.

## Highlights & Insights
- The plain-text osmAG topological prior is an elegant design—token-efficient, natively readable by LLMs, and automatically generated from floor plans—elegantly sidestepping the need for dense map reconstruction.
- The dual-signal halting strategy combining collision accumulation and MLLM semantic judgment balances physical safety with semantic understanding.
- Directly modifying edge weights in the graph (setting them to $\infty$) is more reliable than injecting obstacle descriptions into prompts, avoiding context overflow and spatial hallucinations.
- The idea of using diffusion model inpainting to generate obstacle training data is generalizable to other vision tasks requiring hard negatives.

## Limitations & Future Work
- osmAG requires pre-obtained building floor plans or CAD files, making it unsuitable for completely unknown environments.
- The simulation dataset is limited in scale, covering only 5 HM3D scenes and 176 tasks.
- RVH relies on VLM accuracy for blockage judgment, which may produce false positives in complex or ambiguous scenarios.
- No direct comparison is made with methods using 2D/3D metric maps (e.g., MapNav).
- Evaluation is restricted to indoor scenarios; large-scale outdoor navigation remains unvalidated.

## Related Work & Insights
- **OmniNav**: Uses panoramic multi-camera input with lookahead exploration, achieving the highest B-column SR but requiring multi-camera hardware. HaltNav uses only a single camera and substantially outperforms it in obstacle robustness.
- **osmAG-LLM**: Also uses osmAG for global planning but lacks reactive halting and dynamic update capabilities; HaltNav adds closed-loop conditioning.
- **ReCAPA**: A hierarchical prediction-correction framework, but with passive replanning; HaltNav performs active visual monitoring, immediate interruption, and topological pruning.

## Rating
- Novelty: ⭐⭐⭐⭐ (a complete combination of lightweight topological priors, reactive halting, and generative data synthesis)
- Experimental Thoroughness: ⭐⭐⭐⭐ (simulation + real robot, three levels of instruction granularity, obstacle injection)
- Writing Quality: ⭐⭐⭐⭐ (clear system design, complete formalization)
- Value: ⭐⭐⭐⭐ (practical framework design targeting real-world deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ViHOI: Human-Object Interaction Synthesis with Visual Priors](vihoi_human-object_interaction_synthesis_with_visual_priors.md)
- [\[CVPR 2026\] Language-Free Generative Editing from One Visual Example](language-free_generative_editing_from_one_visual_example.md)
- [\[CVPR 2026\] Tiny Inference-Time Scaling with Latent Verifiers](tiny_inference-time_scaling_with_latent_verifiers.md)
- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)
- [\[CVPR 2026\] Organizing Unstructured Image Collections using Natural Language](organizing_unstructured_image_collections_using_natural_language.md)

</div>

<!-- RELATED:END -->
