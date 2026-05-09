---
title: >-
  [Paper Note] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation
description: >-
  [NeurIPS 2025 (Spotlight)][Graph Learning][Scene graph generation] This paper proposes the ESCA framework, which provides structured visual understanding context for MLLM-driven embodied agents via open-vocabulary scene graph generation (the SGClip model), substantially reducing perception error rates and improving task completion rates.
tags:
  - NeurIPS 2025 (Spotlight)
  - Graph Learning
  - Scene graph generation
  - embodied agents
  - CLIP
  - vision-language models
  - neuro-symbolic learning
date: 2026-05-08
content_hash: 0cf4acbbc7ddde73
---

# ESCA: Contextualizing Embodied Agents via Scene-Graph Generation

**Conference**: NeurIPS 2025 (Spotlight)  
**arXiv**: [2510.15963](https://arxiv.org/abs/2510.15963)  
**Code**: [SGCLIP](https://github.com/video-fm/LASER) / [ESCA](https://github.com/video-fm/ESCA)  
**Area**: Graph Learning / Embodied AI  
**Keywords**: Scene graph generation, embodied agents, CLIP, vision-language models, neuro-symbolic learning

## TL;DR

This paper proposes the ESCA framework, which provides structured visual understanding context for MLLM-driven embodied agents via open-vocabulary scene graph generation (the SGClip model), substantially reducing perception error rates and improving task completion rates.

## Background & Motivation

Multimodal large language models (MLLMs) have seen rapid adoption in embodied agents, yet existing MLLMs suffer from fundamental deficiencies in the following areas:

**Insufficient fine-grained visual-semantic grounding**: MLLMs struggle to reliably establish connections between low-level visual features and high-level textual semantics, resulting in weak spatial and temporal visual localization.

**Perception errors as the primary failure cause**: Empirical analysis shows that up to 69% of agent failures stem from perception errors, such as object hallucination, entity misidentification, and incorrect spatial relationships.

**Limitations of existing visual augmentation modules**: Object detection models such as Grounding DINO and YOLO focus primarily on object recognition while neglecting semantic attributes, inter-object relationships, and temporal consistency.

## Method

### Overall Architecture

ESCA (Embodied and Scene-Graph Contextualized Agent) provides context to MLLMs through four modular stages:

1. **Selective Concept Extraction**: The MLLM extracts structured concepts—including entity categories (e.g., car, knife), attributes (e.g., red, small), and relations (e.g., behind, cutting)—based on instructions and interaction history.
2. **Object Identification**: A Grounding DINO + SAM2 pipeline grounds extracted concepts to specific image regions, producing precise segmentation masks.
3. **Scene Graph Prediction**: The SGClip model generates a probabilistic scene graph containing unary facts (object attributes) and binary facts (inter-object relations).
4. **Visual Summarization and Verification**: The scene graph is converted into natural language descriptions, and consistency between visual feedback and the scene graph is verified.

### Key Designs

**SGClip model architecture**: A CLIP-based scene graph generation model supporting three inference modes:
- **Entity category inference**: Applies softmax normalization over candidate categories.
- **Attribute inference**: Constructs attribute–negation pairs (e.g., "red" vs. "not red") for binary contrastive probability estimation.
- **Binary relation inference**: Colors target regions to mark subject/object roles, and augments relation phrases with entity categories (e.g., "(robot, cutting, cabbage)").

**ESCA-Video-87K dataset**: Constructed from LLaVA-Video-178K, containing 87K video data points, each represented as a quintuple $(\\bar{I}, L_{cap}, \\Sigma, \\bar{c}, \\phi)$ comprising video, caption, object trajectories, concept set, and spatiotemporal procedural specification.

**Transfer Protocol**: Downstream adaptation to different embodied environments is achieved via two customized prompt templates—a concept extraction prompt and a visual summarization prompt—without retraining the core system.

### Loss & Training

SGClip is trained with a neuro-symbolic learning pipeline comprising three losses:
- **Contrastive loss**: Distinguishes matched from unmatched video–specification pairs, using a chunked event training strategy (at most 3 events per chunk).
- **Temporal loss**: Improves the precision of event-to-video-segment temporal alignment.
- **Semantic loss**: Leverages commonsense negation knowledge (e.g., a bed is unlikely in an outdoor scene) by sampling semantically distant words from the top-5,000 high-frequency keywords as negative examples.

Training configuration: learning rate $1 \times 10^{-6}$, batch size 2, 1 FPS sampling, trained for 3 epochs on 10 H100 GPUs (~10 days).

## Key Experimental Results

### Main Results

**EB-Navigation environment** (success rate %):

| Model | Base | + GD | + ESCA |
|-------|------|------|--------|
| InternVL-2.5 | 47.33 | 47.67 | **51.66** |
| Gemini-2.0 | 40.68 | 40.53 | **42.00** |
| Qwen2.5 | 44.99 | 48.27 | **49.33** |
| GPT-4o | 51.33 | 53.33 | **54.67** |

**EB-Manipulation environment** (success rate %):

| Model | Base | + YOLO | + ESCA |
|-------|------|--------|--------|
| InternVL-2.5 | 19.31 | 19.30 | **24.30** |
| GPT-4o | 23.47 | 28.48 | **34.44** |

Key finding: InternVL-2.5 + ESCA surpasses the performance of vanilla GPT-4o on EB-Navigation.

### Ablation Study

**SGClip zero-shot generalization** (Recall metrics):
- SGClip consistently outperforms vanilla CLIP on three out-of-domain datasets: OpenPVSG, Action Genome, and VidVRD.
- Performance improves steadily as training data scales from 1K to 10K to 87K samples.

**ActivityNet action recognition**:

| Method | Data | Accuracy |
|--------|------|----------|
| SGClip (zero-shot) | 0% | 76.34% |
| CLIP (zero-shot) | 0% | 74.37% |
| SGClip (few-shot) | 5% | **92.10%** |
| InternVL-6B (full) | 100% | 95.90% |

With only 5% of training data, SGClip approaches the performance of fully supervised InternVL-6B.

**VidVRD scene graph relation annotation** (after fine-tuning):

| Model | P@1 | R@1 | P@5 | R@5 | P@10 | R@10 |
|-------|-----|-----|-----|-----|------|------|
| SGClip-CLIP | 0.469 | 0.085 | 0.321 | 0.250 | 0.246 | 0.353 |
| SGClip | **0.495** | **0.087** | **0.350** | **0.270** | **0.278** | **0.385** |

### Key Findings

1. **Error decomposition analysis**: ESCA reduces InternVL's perception error rate on EB-Navigation from 69% to 30%.
2. **Cross-environment generalization**: ESCA yields consistent improvements on EB-Habitat and EB-Alfred as well.
3. **Comparison with GD/YOLO**: Although Grounding DINO and YOLO also improve over the baseline, ESCA provides additional and significant gains beyond them.

## Highlights & Insights

1. **Selective scene graphs**: Rather than injecting a complete scene graph (which may degrade performance), the MLLM first identifies the most instruction-relevant concept subset, then generates a targeted scene graph.
2. **Probabilistic prediction**: Each fact in the scene graph is associated with a confidence score, enabling uncertainty capture.
3. **Model-driven self-supervision**: Learning signals are derived from GPT-4-generated captions and spatiotemporal specifications, requiring no human annotation.
4. **Elegant Transfer Protocol design**: Adaptation to four distinct embodied environments is achieved through only two prompt templates.

## Limitations & Future Work

1. **Insufficient real-time capability**: High-level LLM planning introduces latency, making the framework unsuitable for low-level real-time control.
2. **2D-only input**: The absence of 3D representations (e.g., point clouds) limits depth reasoning and spatial precision.
3. **Lack of state verification**: No formal mechanism is provided to verify intermediate and final states during execution.

## Related Work & Insights

- ESCA shares a neuro-symbolic learning pipeline with LASER (ICLR 2025); SGClip can be viewed as an application of that pipeline to the embodied domain.
- The Scallop programming language enables differentiable symbolic alignment and serves as a key tool in the neuro-symbolic paradigm.
- The design philosophy of the Transfer Protocol generalizes to other visual understanding systems requiring cross-task adaptation.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of selective scene graphs and neuro-symbolic self-supervision is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four embodied environments × four MLLMs, plus independent scene graph evaluation.
- Practicality: ⭐⭐⭐⭐ — A plug-and-play framework applicable to diverse MLLMs.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure with rich illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Interaction-Centric Knowledge Infusion and Transfer for Open-Vocabulary Scene Graph Generation](interaction-centric_knowledge_infusion_and_transfer_for_open-vocabulary_scene_gr.md)
- [\[CVPR 2026\] WSGG: Towards Spatio-Temporal World Scene Graph Generation from Monocular Videos](../../CVPR2026/graph_learning/wsgg_spatiotemporal_world_scene_graph.md)
- [\[ICLR 2026\] Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](../../ICLR2026/graph_learning/embodied_agents_meet_personalization_investigating_challenges_and_solutions_thro.md)
- [\[NeurIPS 2025\] Agint: Agentic Graph Compilation for Software Engineering Agents](agint_agentic_graph_compilation_for_software_engineering_age.md)
- [\[CVPR 2026\] Graph2Eval: Automatic Multimodal Task Generation for Agents via Knowledge Graphs](../../CVPR2026/graph_learning/graph2eval_multimodal_task_generation_agents.md)

</div>

<!-- RELATED:END -->
