---
title: >-
  [Paper Note] Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions
description: >-
  [ICCV 2025][LLM Agent][Embodied Perception] A three-stage self-supervised framework is proposed that significantly improves cross-view description consistency and accuracy for the same object in indoor environments, achieved through agent-driven multi-view observation collection, LLM consensus-based pseudo-label generation, and contrastive fine-tuning of the captioner.
tags:
  - "ICCV 2025"
  - "LLM Agent"
  - "Embodied Perception"
  - "Image Captioning"
  - "Self-supervised Learning"
  - "Pseudo-labeling"
  - "Contrastive Learning"
date: 2026-05-08
content_hash: ac250a2f46cce5a3
---

# Embodied Image Captioning: Self-supervised Learning Agents for Spatially Coherent Image Descriptions

**Conference**: ICCV 2025
**arXiv**: [2504.08531](https://arxiv.org/abs/2504.08531)  
**Code**: [https://hsp-iit.github.io/embodied-captioning/](https://hsp-iit.github.io/embodied-captioning/)  
**Area**: LLM Agent / Embodied Intelligence
**Keywords**: Embodied Perception, Image Captioning, Self-supervised Learning, Pseudo-labeling, Contrastive Learning

## TL;DR
A three-stage self-supervised framework is proposed that significantly improves cross-view description consistency and accuracy for the same object in indoor environments, achieved through agent-driven multi-view observation collection, LLM consensus-based pseudo-label generation, and contrastive fine-tuning of the captioner.

## Background & Motivation

**Background**: Image captioning models deployed on autonomous agents frequently produce inconsistent or erroneous descriptions of the same object across different viewpoints, particularly under occlusion or unfavorable viewing angles.

**Limitations of Prior Work**: Navigation-based methods (e.g., CaBOT) require prior knowledge of the optimal viewpoint and are limited to simple scenes; noisy-label methods (e.g., ECO) rely on CLIP alignment and may select incorrect descriptions; summarization methods (e.g., IC3) use sampling diversity to generate summaries but cannot filter out erroneous information.

**Key Challenge**: Automatically improving the cross-view description consistency of a captioner for the same object in complex indoor environments without manual annotation remains an open challenge.

**Key Insight**: The problem is decomposed into three decoupled stages — navigation-based data collection, pseudo-label generation, and model fine-tuning.

**Core Idea**: A 3D voxel map is used to aggregate multi-view descriptions of the same object; an LLM combined with frequency information and in-context learning then distills consistent pseudo-labels; a triplet loss subsequently enforces proximity between visual features of the same object.

## Method

### Overall Architecture
A three-stage pipeline: (1) the agent autonomously navigates a simulated environment, constructs a semantic voxel map, and aggregates detections and descriptions; (2) for each 3D object instance, an LLM distills all associated descriptions into a single pseudo-label; (3) the captioner is fine-tuned using the pseudo-labels and contrastive learning.

### Key Designs

1. **Navigation and 3D Clustering (Phase 1)**:

    - **Function**: The agent explores the environment according to a policy, detects objects with Mask2Former, projects detections into a voxel map, and clusters them via connected components to obtain unique object instances.
    - **Mechanism**: 2D detection logits, masks, and captions are projected into 3D voxel space via depth maps; a 26-connected 3D connected-component algorithm assigns a unique object ID to each voxel.
    - **Design Motivation**: Associating multi-timestep, multi-view observations with the same 3D object resolves the cross-view correspondence problem.
    - Exploration strategy CLA: A disagreement map constructed from inter-caption inconsistency (SBERT cosine distance) guides navigation.

2. **LD-CPS Pseudo-label Generation (Phase 2)**:

    - **Function**: Generates a consistent pseudo-label for each clustered object instance.
    - **Mechanism**: Captioner-induced bias phrases (e.g., "A picture of...") are removed in preprocessing; all descriptions and their occurrence frequencies are then provided to an LLM prompt, which leverages in-context learning to judge caption reliability and distill a concise, consistent pseudo-label.
    - **Design Motivation**: Frequency information ensures that the majority consensus is adopted while noise is suppressed; in-context examples improve LLM distillation quality.

3. **Contrastive Fine-tuning (Phase 3)**:

    - **Function**: Fine-tunes the captioner with pseudo-labels and enhances cross-view consistency.
    - **Mechanism**: Standard captioning loss combined with triplet loss; for each anchor, positives are different viewpoints of the same object instance, and negatives are other objects: $\mathcal{L} = \mathcal{L}_{cap} + \lambda_{tr}\mathcal{L}_{tr}$
    - **Design Motivation**: The triplet loss enforces proximity between visual representations of the same object across viewpoints, improving description consistency.

### Loss & Training
Total loss = cross-entropy captioning loss + $\lambda_{tr}$ × triplet loss ($\lambda_{tr}=0.1$, margin $\epsilon=2$). The built-in contrastive loss of CoCa is disabled to avoid penalizing the encoder; BLIP-2 is fine-tuned via LoRA on the Q-Former module.

## Key Experimental Results

### Main Results

| Method | Dataset | B4 | METEOR | CIDEr | SPICE | CS (Semantic Similarity) |
|--------|---------|-----|--------|-------|-------|--------------------------|
| CoCa off-the-shelf | Gibson | 7.30 | 20.16 | 0.45 | 22.22 | 66.01 |
| CoCa + LD-CPS | Gibson | 14.70 | 25.13 | 1.05 | 30.39 | 72.08 |
| CoCa + LD-CPS + triplet | Gibson | **15.47** | **26.22** | **1.10** | **31.75** | **72.91** |
| BLIP2 off-the-shelf | Gibson | 6.59 | 17.91 | 0.35 | 19.32 | 63.32 |
| BLIP2 + LD-CPS + triplet | Gibson | **14.05** | **23.89** | **1.19** | **28.25** | **71.46** |

### Ablation Study

| Pseudo-label Method | B4 | CS |
|--------------------|-----|-----|
| ECO (best caption selection) | 10.07–14.70 | 69.43 |
| IC3 (LLM summarization) | 1.25 | 56.68 |
| LD-CPS (Ours) | **14.70** | **72.08** |

### Key Findings
- The CLA strategy uncovers caption similarity below the threshold for 50% of the data compared to other strategies, more effectively identifying high-disagreement regions.
- LD-CPS substantially outperforms ECO and IC3 across all metrics, particularly achieving 6–16 points higher semantic similarity.
- The triplet loss consistently improves performance across all strategy–captioner combinations.
- After self-supervised fine-tuning, CoCa's description quality approaches that of ChatGPT o1.

## Highlights & Insights
- **The three-stage decoupled design is highly practical**: each stage can be independently replaced (exploration strategy, pseudo-label method, or captioner), making the framework broadly applicable.
- **The frequency + in-context learning pseudo-labeling strategy is elegant**: it operationalizes the intuition of "majority voting + noise filtering" through an LLM, achieving robust cross-view label consistency.
- **The learned exploration strategy CLA** is a novel approach that couples active perception with semantic understanding by using caption consistency to drive navigation.

## Limitations & Future Work
- Evaluation is limited to 6 categories of indoor objects, offering limited categorical diversity.
- 3D voxel projection introduces occlusion and projection noise, which may degrade object instance clustering quality.
- CLA training is based on CoCa's disagreement signal; switching to a different captioner requires retraining the strategy.
- The effect of open-vocabulary detectors on the framework has not been explored.

## Related Work & Insights
- **vs. CaBOT**: CaBOT requires prior knowledge of the optimal viewpoint and is limited to simple scenes; the proposed method requires no such prior and scales to complex indoor environments.
- **vs. ECO**: ECO relies on CLIP alignment for caption selection; the proposed method employs LLM-based distillation with frequency information, yielding greater robustness.
- **vs. IC3**: IC3 cannot handle large quantities of noisy captions; LD-CPS leverages frequency information and in-context learning to substantially outperform it.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework design is novel, though individual components are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive comparisons across multiple datasets, captioners, and strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with good modularity.
- Value: ⭐⭐⭐⭐ Practically valuable for visual understanding in embodied scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GUI-Shift: Enhancing VLM-Based GUI Agents through Self-supervised Reinforcement Learning](../../ICLR2026/llm_agent/gui-shift_enhancing_vlm-based_gui_agents_through_self-supervised_reinforcement_l.md)
- [\[ACL 2025\] Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](../../ACL2025/llm_agent/llm_agent_image_classification.md)
- [\[CVPR 2026\] RetouchIQ: MLLM Agents for Instruction-Based Image Retouching with Generalist Reward](../../CVPR2026/llm_agent/retouchiq_mllm_agents_for_instruction-based_image_retouching_with_generalist_rew.md)
- [\[CVPR 2026\] Universal Guideline-Driven Image Clustering via a Hybrid LLM Agent](../../CVPR2026/llm_agent/universal_guideline-driven_image_clustering_via_a_hybrid_llm_agent.md)
- [\[AAAI 2026\] PerTouch: VLM-Driven Agent for Personalized and Semantic Image Retouching](../../AAAI2026/llm_agent/pertouch_vlm-driven_agent_for_personalized_and_semantic_image_retouching.md)

</div>

<!-- RELATED:END -->
