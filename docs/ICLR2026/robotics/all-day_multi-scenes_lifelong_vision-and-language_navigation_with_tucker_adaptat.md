---
title: >-
  [Paper Note] All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation
description: >-
  [ICLR 2026][Robotics][Lifelong Vision-and-Language Navigation] This paper proposes Tucker Adaptation (TuKA), which represents multi-level navigation knowledge across multiple scenes and environments as a high-order tensor, decomposed via Tucker decomposition into a shared subspace (core tensor + encoder/decoder) and scene/environment expert vectors. Combined with a Decoupled Knowledge Incremental Learning (DKIL) strategy, TuKA enables all-day multi-scene lifelong VLN, achieving superior SR and lower forgetting rates over LoRA variants across 24 navigation scenarios.
tags:
  - ICLR 2026
  - Robotics
  - Lifelong Vision-and-Language Navigation
  - Tucker Decomposition
  - Parameter-Efficient Fine-Tuning
  - Catastrophic Forgetting
  - Multi-level Knowledge Decoupling
date: 2026-05-08
content_hash: 939ed915ac398843
---

# All-day Multi-scenes Lifelong Vision-and-Language Navigation with Tucker Adaptation

**Conference**: ICLR 2026
**arXiv**: [2603.14276](https://arxiv.org/abs/2603.14276)
**Code**: [https://ganvin-li.github.io/AlldayWalker/](https://ganvin-li.github.io/AlldayWalker/)
**Area**: Robotics
**Keywords**: Lifelong Vision-and-Language Navigation, Tucker Decomposition, Parameter-Efficient Fine-Tuning, Catastrophic Forgetting, Multi-level Knowledge Decoupling

## TL;DR
This paper proposes Tucker Adaptation (TuKA), which represents multi-level navigation knowledge across multiple scenes and environments as a high-order tensor, decomposed via Tucker decomposition into a shared subspace (core tensor + encoder/decoder) and scene/environment expert vectors. Combined with a Decoupled Knowledge Incremental Learning (DKIL) strategy, TuKA enables all-day multi-scene lifelong VLN, achieving superior SR and lower forgetting rates over LoRA variants across 24 navigation scenarios.

## Background & Motivation

**Background**: VLN agents have evolved from discrete graph-based navigation to continuous low-level action navigation, yet real-world deployment requires agents to adapt to diverse scenes (bedroom, living room, etc.) and varying environmental conditions (normal, low-light, overexposed, hazy), necessitating continual learning.

**Limitations of Prior Work**: VLN agents fine-tuned on specific scenes suffer catastrophic forgetting when switched to new scenes. Existing LoRA/MoE-LoRA methods can only represent a two-level knowledge structure ("shared matrix + task-specific matrix"), and cannot decouple the orthogonal dimensions of "scene knowledge" and "environment knowledge."

**Key Challenge**: Navigation knowledge exhibits a multi-level structure — core navigation skills (shared across all scenes), scene-specific knowledge (e.g., indoor layouts), and environment-specific knowledge (e.g., visual adaptation under low light) — all three levels must be learned independently while being shared across tasks simultaneously.

**Goal**: Formalize the All-day Multi-scene Lifelong VLN (AML-VLN) problem and design a parameter-efficient adaptation method capable of decoupling multi-level knowledge.

**Key Insight**: Leverage the natural multi-modal factorization capability of Tucker tensor decomposition — the core tensor captures shared knowledge, while rows of the factor matrices encode scene/environment experts respectively.

**Core Idea**: A fourth-order Tucker tensor decomposition simultaneously encodes shared core navigation skills, scene experts, and environment experts; decoupled incremental learning achieves forgetting-free lifelong navigation.

## Method

### Overall Architecture
TuKA introduces a fourth-order tensor $\mathcal{X}^l \in \mathbb{R}^{a_l \times b_l \times M \times N}$ at each layer of an LLM backbone (Qwen2-7B), decomposed via Tucker decomposition into: a core tensor $\mathcal{G}$ (shared navigation skills), $U^1, U^2$ (shared encoder/decoder), $U^3 \in \mathbb{R}^{M \times r_3}$ ($M$ scene experts), and $U^4 \in \mathbb{R}^{N \times r_4}$ ($N$ environment experts). When learning the $t$-th scene, the corresponding scene expert row $U^3[s,:]$ and environment expert row $U^4[e,:]$ are selected and combined with the shared components to produce the layer-wise adaptation weight $\Delta W_t$.

### Key Designs

1. **Tucker Adaptation Architecture**

    - Function: Replace LoRA's low-rank matrix decomposition with a high-order Tucker tensor decomposition.
    - Mechanism: $\Delta W_t = U^1 \cdot (\mathcal{G} \times_3 U^3[s,:] \times_4 U^4[e,:]) \cdot (U^2)^T$. Scene experts are selected via mode-3 indexing (selecting the $s$-th row from $M$ candidates), and environment experts via mode-4 indexing (selecting the $e$-th row from $N$ candidates), naturally realizing a "scene × environment" two-dimensional compositional space.
    - Design Motivation: LoRA/MoE-LoRA compresses all knowledge into a two-dimensional matrix (one shared + multiple task-specific), which cannot independently model the orthogonal knowledge dimensions of scene and environment. The high-order nature of Tucker decomposition natively supports multi-dimensional knowledge decoupling — the multi-modal structure of core tensor + factor matrices precisely matches the hierarchical structure of navigation knowledge.

2. **Decoupled Knowledge Incremental Learning (DKIL)**

    - Function: Consolidate shared knowledge and constrain task-specific experts during continual learning of new scenes.
    - Mechanism: Three losses work in concert:
        - **Shared Knowledge EWC** ($\mathcal{L}_{ewc}$): Applies Fisher information-weighted quadratic constraints to the core tensor and encoder/decoder to prevent the shared components from drifting. Fisher weights are updated via exponential moving average.
        - **Expert Consistency** ($\mathcal{L}_{co}$): Applies L2 constraints to previously learned scene/environment experts to prevent forgetting.
        - **Expert Separability** ($\mathcal{L}_{es}$): Encourages new experts to be orthogonal to existing experts, ensuring new knowledge is learned in an independent subspace.
    - Design Motivation: Shared knowledge requires gradual consolidation (EWC), learned experts must be preserved (consistency constraint), and new experts require independent exploration (orthogonality constraint) — the three mechanisms address distinct challenges of continual learning.

3. **Task Expert Inference Search**

    - Function: Automatically match scene and environment experts at test time (without task-id).
    - Mechanism: During training, CLIP visual feature prototypes are stored for each scene/environment. At test time, visual features of current observations are extracted and matched to the nearest scene and environment experts via cosine similarity.
    - Design Motivation: Task-id is unavailable in real-world deployment; automatic routing to the correct expert combination based on visual features is required.

### Allday-Habitat Simulation Platform
Built on Habitat with three imaging models (atmospheric scattering model, low-light noise model, overexposure clipping model) to synthesize degraded environments from normal ones, constructing 24 navigation scenarios (5 simulated scenes × 4 environments + 2 real-world scenes × 2 environments).

## Key Experimental Results

### Main Results (Average SR% across 24 scenarios)

| Method | Avg SR↑ | Avg F-SR↓ | Notes |
|--------|---------|-----------|-------|
| Seq-FT | 11% | High | Sequential fine-tuning, severe forgetting |
| EWC-LoRA | 15% | — | LoRA + EWC |
| HydraLoRA | ~17% | — | MoE-LoRA |
| BranchLoRA | ~18% | — | Branched LoRA |
| **AlldayWalker (TuKA)** | **Best** | **Lowest** | Tucker adaptation |

TuKA consistently outperforms all LoRA-variant baselines on SR and SPL across all 24 scenarios, with significantly lower forgetting rates.

### Ablation Study

| Configuration | Avg SR | Notes |
|---------------|--------|-------|
| w/o core tensor sharing | Drops | Shared knowledge cannot transfer across tasks |
| w/o EWC constraint | Notable drop | Shared knowledge overwritten by new tasks |
| w/o orthogonality constraint | Drops | New experts interfere with old experts in the same subspace |
| w/o expert consistency | Drops | Learned experts are modified, causing forgetting |
| Full TuKA | **Best** | Complete framework |

### Key Findings
- Sequential fine-tuning (Seq-FT) reduces SR on earlier scenes to nearly 0% (T1–T6 all at 0%), demonstrating severe catastrophic forgetting.
- The mode-3/4 factor matrices of Tucker decomposition naturally support compositional generalization across "scene × environment" — scenes seen during training show some generalization to unseen environments.
- The orthogonality constraint, though simple, is critical for independent learning of new experts.
- Real-world deployment on two real scenes also validates the effectiveness of the approach.

## Highlights & Insights
- The idea of **modeling multi-level knowledge via tensor decomposition** is elegant — the "core tensor + factor matrices" structure of Tucker decomposition maps naturally onto the "shared skills + scene experts + environment experts" knowledge hierarchy.
- The solution to the **dimensionality alignment problem** is clever: selecting a single row vector from each factor matrix reduces the high-order tensor to a two-dimensional weight matrix, perfectly matching the matrix structure of the LLM backbone.
- The **three-level DKIL mechanism** (EWC consolidation + consistency constraint + orthogonal exploration) forms a complete toolkit for continual learning.
- The Allday-Habitat platform synthesizes degraded environments via physics-based imaging models (rather than simple filters), enhancing the realism of environmental variation.

## Limitations & Future Work
- The current setup covers only 5+2=7 scenes and 4 environments — scalability to large numbers of scenes (hundreds) remains unknown.
- The number of experts $M$ and $N$ must be predefined and cannot grow dynamically — truly open-ended lifelong learning should support unbounded expansion.
- Expert search at inference time relies on CLIP feature matching, which may fail when new environments differ substantially from seen ones.
- The four degradation types (normal/low-light/overexposed/hazy) have physical grounding but are simpler than real-world environmental variation (rain, motion blur, occlusion, etc.).
- The rank choices $r_1=r_2=8, r_3=r_4=64$ appear somewhat arbitrary; sensitivity analysis of rank selection is absent.

## Related Work & Insights
- **vs. LoRA**: LoRA's two-dimensional matrix factorization cannot decouple multi-dimensional knowledge; TuKA extends this to fourth-order tensor decomposition.
- **vs. HydraLoRA/BranchLoRA**: These MoE-LoRA methods only support a two-level "shared + specific" structure; TuKA introduces a three-level "shared + scene + environment" hierarchy.
- **vs. EWC/LwF and other continual learning methods**: Traditional continual learning approaches do not account for the hierarchical structure of knowledge; DKIL applies different strategies to different levels of knowledge.
- **vs. StreamVLN**: AlldayWalker builds on the StreamVLN agent architecture, with TuKA inserted as a parameter-efficient adaptation layer.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of Tucker decomposition and multi-level knowledge decoupling is pioneering in VLN and continual learning; the problem formulation (AML-VLN) is also novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of 24 scenarios, real-world deployment, and ablation studies is solid, though the scale of scenarios is limited.
- Writing Quality: ⭐⭐⭐⭐ Problem formalization is clear and method diagrams are intuitive.
- Value: ⭐⭐⭐⭐ Directly relevant to real-world VLN deployment; the Tucker adaptation paradigm is transferable to other multi-dimensional continual learning settings.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] JanusVLN: Decoupling Semantics and Spatiality with Dual Implicit Memory for Vision-Language Navigation](janusvln_decoupling_semantics_and_spatiality_with_dual_implicit_memory_for_visio.md)
- [\[CVPR 2026\] DecoVLN: Decoupling Observation, Reasoning, and Correction for Vision-and-Language Navigation](../../CVPR2026/robotics/decovln_decoupling_observation_reasoning_and_correction_for_vision-and-language_.md)
- [\[CVPR 2026\] ProFocus: Proactive Perception and Focused Reasoning in Vision-and-Language Navigation](../../CVPR2026/robotics/profocus_proactive_perception_and_focused_reasoning_in_vision-and-language_navig.md)
- [\[AAAI 2026\] Recursive Visual Imagination and Adaptive Linguistic Grounding for Vision Language Navigation](../../AAAI2026/robotics/recursive_visual_imagination_and_adaptive_linguistic_grounding_for_vision_langua.md)
- [\[ICLR 2026\] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)

<!-- RELATED:END -->
