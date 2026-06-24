---
title: >-
  [Paper Note] MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning
description: >-
  [NeurIPS 2025][Robotics][Tabletop scene generation] This paper proposes the MesaTask framework, which decomposes task descriptions into a Spatial Reasoning Chain — object reasoning → spatial relationship reasoning → scene graph construction → 3D layout — and combines a 10K+ manually annotated dataset with DPO optimization to generate physically plausible, task-aligned tabletop manipulation scenes.
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Tabletop scene generation"
  - "spatial reasoning chain"
  - "LLM scene generation"
  - "DPO"
  - "robotic manipulation"
date: 2026-05-08
content_hash: dc94ec37bca29dc1
---

# MesaTask: Towards Task-Driven Tabletop Scene Generation via 3D Spatial Reasoning

**Conference**: NeurIPS 2025
**arXiv**: [2509.22281](https://arxiv.org/abs/2509.22281)  
**Code**: Available  
**Area**: Robotics / 3D Scene Generation
**Keywords**: Tabletop scene generation, spatial reasoning chain, LLM scene generation, DPO, robotic manipulation

## TL;DR

This paper proposes the MesaTask framework, which decomposes task descriptions into a Spatial Reasoning Chain — object reasoning → spatial relationship reasoning → scene graph construction → 3D layout — and combines a 10K+ manually annotated dataset with DPO optimization to generate physically plausible, task-aligned tabletop manipulation scenes.

## Background & Motivation

**Background**: Robotic manipulation requires diverse tabletop scenes for policy training, yet traditional approaches rely on hand-designed or purely random layouts, making it difficult to achieve both diversity and physical plausibility.

**Limitations of Prior Work**: Existing LLM-based scene generation methods (e.g., LayoutGPT) exhibit limited zero-shot capability and struggle to model complex inter-object relationships such as stacking and containment. Image reconstruction methods are severely affected by occlusion.

**Key Challenge**: A substantial gap exists between high-level task descriptions and concrete 3D layouts — how does "prepare a dinner" translate into precise 3D positions and orientations of tableware and food?

**Key Insight**: The Spatial Reasoning Chain decomposes the problem into a CoT pipeline: object reasoning → attribute description → spatial relationships → scene graph → 3D coordinates.

**Core Idea**: SFT injects spatial reasoning capability, and DPO eliminates collisions and task misalignment.

## Method

### Overall Architecture

(1) MesaTask-10K dataset construction (T2I → depth estimation → 3D retrieval → manual refinement → physics simulation); (2) Spatial Reasoning Chain training data construction; (3) LLM SFT + DPO training.

### Key Designs

1. **MesaTask-10K Dataset**

    - Function: Constructs 10,700 manually annotated tabletop scenes.
    - Mechanism: LLM generates scene descriptions → FLUX generates reference images → depth estimation and detection yield coarse layouts → manual refinement in Blender (10–20 min/scene) → IsaacSim physics simulation eliminates collisions.
    - Design Motivation: Covers 6 tabletop categories (office, dining, kitchen, etc.), 12,000+ 3D assets (including articulated objects), and 200+ object categories.

2. **Spatial Reasoning Chain**

    - Function: Structures the task-to-scene process as a reasoning chain.
    - Mechanism: Task description → object list reasoning (what objects are needed) → spatial relationship reasoning (who is on/inside/beside whom) → scene graph construction (nodes + edges) → 3D coordinate generation.
    - Design Motivation: Direct prediction of 3D coordinates is overly challenging; step-by-step reasoning reduces complexity.

3. **SFT + DPO Training**

    - Function: SFT injects fundamental spatial reasoning capability; DPO eliminates collisions and misalignment.
    - Mechanism: The SFT stage trains on reasoning chain data; the DPO stage constructs preferred/rejected pairs (collision-free vs. collision, task-aligned vs. misaligned).
    - Design Motivation: Residual collisions and task drift persist after SFT; DPO effectively corrects these issues.

### Loss & Training

SFT stage: standard language modeling loss. DPO stage: $\mathcal{L}_{DPO} = -\log\sigma(\beta(\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}))$.

## Key Experimental Results

### Main Results

| Method | FID↓ | Task Alignment↑ | Physical Plausibility↑ | Layout Quality↑ |
|--------|------|----------------|----------------------|----------------|
| LayoutGPT (zero-shot) | 185.3 | 3.2/5 | 2.8/5 | 2.5/5 |
| LLPlace (SFT) | 142.7 | 3.8/5 | 3.5/5 | 3.2/5 |
| **MesaTask (SFT)** | **98.5** | **4.2/5** | **4.0/5** | **3.8/5** |
| **MesaTask (SFT+DPO)** | **87.3** | **4.5/5** | **4.3/5** | **4.1/5** |

### Ablation Study

| Configuration | Collision Rate↓ | Task Alignment↑ |
|---------------|----------------|----------------|
| SFT only | 12.3% | 4.2/5 |
| + DPO (collision pairs) | 4.1% | 4.2/5 |
| + DPO (task pairs) | 11.8% | 4.5/5 |
| **+ DPO (both)** | **3.8%** | **4.5/5** |

### Key Findings

- DPO reduces the collision rate from 12.3% to 3.8% while simultaneously improving task alignment.
- Generation quality for complex relationships (stacking, containment) is markedly superior to zero-shot methods.
- In user studies, MesaTask achieves the highest scores across all evaluation dimensions.

## Highlights & Insights

- **Spatial Reasoning Chain**: Decomposes the large gap between abstract task descriptions and 3D coordinates into learnable steps. This structured reasoning paradigm is transferable to other 3D generation tasks.
- **Dataset Contribution**: The 10K+ manually annotated scenes, encompassing complex relationships such as stacking and containment, fill a critical data gap in the field.
- **DPO for 3D**: This work represents the first application of DPO to physical collision elimination, yielding significant improvements.

## Limitations & Future Work

- Manual annotation is costly (10–20 min/scene), making large-scale expansion difficult.
- Despite the substantial size (12K+), the 3D asset library still has coverage gaps.
- Validation is conducted solely in simulation; real-robot deployment requires cross-domain transfer.

## Related Work & Insights

- **vs. LayoutGPT**: Limited zero-shot capability and difficulty modeling complex relationships; MesaTask addresses these via data-driven SFT.
- **vs. SetItUp**: Fixed object sets limit diversity; MesaTask covers 200+ categories with 12K+ assets.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formal treatment of task-to-scene generation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ FID + VLM evaluation + user study.
- Writing Quality: ⭐⭐⭐⭐⭐ Dataset, method, and evaluation are systematically complete.
- Value: ⭐⭐⭐⭐⭐ Foundational infrastructure for robotic manipulation scene generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[CVPR 2026\] DextER: Language-driven Dexterous Grasp Generation with Embodied Reasoning](../../CVPR2026/robotics/dexter_language-driven_dexterous_grasp_generation_with_embodied_reasoning.md)
- [\[NeurIPS 2025\] CogVLA: Cognition-Aligned Vision-Language-Action Model via Instruction-Driven Routing & Sparsification](cogvla_cognition-aligned_vision-language-action_model_via_instruction-driven_rou.md)
- [\[NeurIPS 2025\] UniDomain: Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Task Planning](pretraining_a_unified_pddl_domain_from_real-world_demonstrations_for_generalizab.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](learning_spatial-aware_manipulation_ordering.md)

</div>

<!-- RELATED:END -->
