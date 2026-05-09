---
title: >-
  [Paper Note] Text-to-Scene with Large Reasoning Models
description: >-
  [AAAI 2026][LLM Reasoning][Text-to-3D Scene] This paper proposes Reason-3D, which leverages the multi-step spatial reasoning capabilities of large reasoning models (LRMs) to achieve zero-shot text-to-3D scene generation via semantic-voting-based object retrieval and a two-stage layout strategy (autoregressive placement + collision-aware refinement). The system achieves an Elo score of 2248 in human evaluation, substantially outperforming Holodeck (1500) and LayoutVLM (1650).
tags:
  - AAAI 2026
  - LLM Reasoning
  - Text-to-3D Scene
  - Large Reasoning Models
  - Structured Reasoning
  - Scene Generation
  - Reason-3D
date: 2026-05-08
content_hash: 85985032d94407ae
---

# Text-to-Scene with Large Reasoning Models

**Conference**: AAAI 2026
**arXiv**: [2509.26091](https://arxiv.org/abs/2509.26091)
**Code**: None
**Area**: LLM Reasoning
**Keywords**: Text-to-3D Scene, Large Reasoning Models, Structured Reasoning, Scene Generation, Reason-3D

## TL;DR

This paper proposes Reason-3D, which leverages the multi-step spatial reasoning capabilities of large reasoning models (LRMs) to achieve zero-shot text-to-3D scene generation via semantic-voting-based object retrieval and a two-stage layout strategy (autoregressive placement + collision-aware refinement). The system achieves an Elo score of 2248 in human evaluation, substantially outperforming Holodeck (1500) and LayoutVLM (1650).

## Background & Motivation

- **Background**: Demand for 3D scene generation is rapidly growing across applications such as interior design, game development, autonomous driving, and robotics. Traditional methods rely on scene priors learned from training data (e.g., DiffuScene trained on the 3D-FRONT dataset), which limits generalization to novel scene types beyond the training distribution.
- **Limitations of Prior Work**: Directly prompting standard LLMs to output object coordinates frequently yields physically implausible results—including object overlap and unrealistic placement—because general-purpose LLMs lack geometric, scale, and collision reasoning capabilities. Existing LLM-based approaches require additional layout engines or fine-tuning to compensate.
- **Key Challenge**: There is an inherent tension between the flexibility required to interpret open-ended natural language instructions and the geometric precision required to handle 3D spatial relationships—properties that existing methods struggle to satisfy simultaneously.
- **Key Insight**: Large reasoning models (LRMs, e.g., Gemini 2.5 Pro) can address complex spatial relationships through multi-step reasoning chains at test time. This paper explores leveraging such reasoning capabilities directly for scene generation without any domain-specific training.

## Method

### Overall Architecture

Reason-3D is a modular zero-shot scene generation pipeline consisting of two main stages: (1) **Object Retrieval**—selecting the most suitable assets from the Objaverse library via embedding similarity combined with LRM semantic voting; and (2) **Object Layout**—determining the 3D position and rotation of each object through a two-stage placement procedure comprising autoregressive initial placement followed by collision-aware refinement.

### Key Designs

1. **Object Retrieval: Three-Dimensional Semantic Voting**

    - Each object in Objaverse is rendered from two viewpoints using a VLM, which generates structured descriptions along three dimensions: physical properties, functional properties, and contextual properties.
    - Descriptions are encoded into embedding vectors and stored in a vector database.
    - The LRM extracts a required object list from the scene prompt and performs semantic retrieval using three-dimensional structured descriptions.
    - The top-5 candidates are ranked by cosine similarity, and the LRM votes to select the most appropriate instance.
    - Retrieval accuracy: Top-1 reaches 75% and Top-10 reaches 90% (compared to Holodeck's 7%/8%).

2. **Two-Stage Object Layout**

    - **Autoregressive Initial Placement**: The LRM first extracts implicit spatial constraints (e.g., "to the left of the sofa" → explicit coordinate constraints), generates a placement priority list (e.g., tables before objects placed on them), and sequentially places each object while receiving metadata of already-placed objects as context.
    - **Collision-Aware Refinement**: After initial placement, all bounding box overlaps are detected. The LRM receives collision information and corrects placements one by one, while also reasoning about which collisions are semantically acceptable (e.g., overlap between a trash can under a table and the table's bounding box is considered acceptable).
    - To reduce the geometric reasoning burden on the LRM, each object is annotated with a "rotated dimensions" attribute—the pre-computed axis-aligned bounding box size after rotation.

3. **Data Preprocessing**

    - Objaverse objects are normalized for consistent upward and forward orientation. A VLM analyzes four-view renderings to determine the canonical front-facing direction.
    - When preprocessing is imperfect, the LRM can dynamically adjust object rotation during the layout stage.

4. **Pure Language Reasoning Without Visual Feedback**

    - During layout, the LRM receives only textual metadata (object names, dimensions, and the current placement list) without rendered images.
    - This validates the LRM's ability to reason purely from spatial metadata.

### Loss & Training

- No training or fine-tuning is required—the system operates in a completely zero-shot manner.
- The pipeline relies entirely on the multi-step reasoning capabilities of LRMs, using Gemini 2.5 Pro as the default reasoning engine.
- GPT-4.1, Claude Sonnet 4, DeepSeek-R1, and GPT-o3 are also evaluated for comparison.

## Key Experimental Results

### Main Results (Full Scene Generation, 60-Participant Human Evaluation)

| Model | Win Rate vs. Holodeck | Win Rate vs. LayoutVLM | Elo Score |
|---|---|---|---|
| Holodeck | — | 26.9% | 1500 |
| LayoutVLM | 73.1% | — | 1650 |
| Reason-3D | **95.2%** | **98.4%** | **2248** |

### Ablation Study (Object Layout Human Scores, Scale 1–5)

| Instruction Complexity | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| LayoutVLM | 2.8 | 3.4 | 3.0 | 2.5 | 2.4 |
| Reason-3D | **4.4** | **3.9** | **4.4** | **4.1** | **4.3** |

### Key Findings

- **Advantage grows with instruction complexity**: At complexity levels 4–5, Reason-3D maintains scores of 4.1–4.3, while LayoutVLM drops to 2.4–2.5.
- **Large gap in retrieval quality**: Reason-3D achieves a Top-1 accuracy of 75% versus Holodeck's 7%—a gap of more than 10×.
- **LRM choice significantly impacts performance**: Gemini 2.5 Pro achieves an Elo score of 2248, substantially outperforming GPT-o3 (1938) and DeepSeek-R1 (1809); GPT-4.1 performs worst (1500).
- **Necessity of collision-aware refinement**: Dense scenes frequently exhibit collisions after initial placement, as the LRM considers only local object relationships at that stage.
- **Generalizes to outdoor scenes**: The pipeline generalizes to outdoor and mixed environments without modification, transcending the limitations of training distributions.

## Highlights & Insights

- **Zero-shot outperforms trained methods**: Reason-3D requires no domain-specific training yet substantially surpasses methods that rely on layout engines or fine-tuning in terms of scene plausibility.
- **First systematic benchmark of LRM spatial reasoning**: The paper compares five LRMs on spatial reasoning tasks and reveals substantial performance differences across models.
- **Ability to reason about "acceptable collisions"**: The LRM not only resolves collisions but also judges which collisions are semantically reasonable—a capability that is difficult to achieve with rule-based engines.

## Limitations & Future Work

- Full reliance on LRM reasoning capabilities incurs high API costs and latency (Gemini 2.5 Pro outputs an average of 114K+ tokens per scene).
- Pure text-based reasoning without visual feedback still leaves residual collisions in extremely dense scenes.
- The coverage of the Objaverse asset library is limited, and retrieval failure rates for certain specific objects remain relatively high.
- The system does not support fine-grained geometric deformation (e.g., bending, scaling) and handles only rigid placement and rotation.

## Related Work & Insights

- **vs. Holodeck (LLM multi-agent + layout engine)**: Holodeck relies on a domain-specific language (DSL) and a layout optimization engine, which limits flexibility. Reason-3D is purely language-driven and exhibits stronger generalization.
- **vs. LayoutVLM (VLM + visual tokens)**: LayoutVLM requires rendered images and self-consistency decoding, resulting in high computational overhead while achieving inferior retrieval performance. Reason-3D outperforms LayoutVLM in both retrieval and layout.

## Rating

| Dimension | Score | Rationale |
|---|---|---|
| Novelty | ⭐⭐⭐⭐ | First systematic application of LRM multi-step reasoning to zero-shot 3D scene generation |
| Technical Depth | ⭐⭐⭐ | Pipeline architecture is clear, but core innovations are concentrated in pipeline design and prompt engineering |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | Large-scale human evaluation with 60 participants, multi-LRM benchmarking, ablation studies, and outdoor generalization |
| Practical Value | ⭐⭐⭐⭐ | Broad applicability of automated 3D scene generation; zero-shot nature lowers the barrier to adoption |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Trade-offs in Large Reasoning Models: An Empirical Analysis of Deliberative and Adaptive Reasoning over Foundational Capabilities](trade-offs_in_large_reasoning_models_an_empirical_analysis_of_deliberative_and_a.md)
- [\[AAAI 2026\] Answering the Unanswerable Is to Err Knowingly: Analyzing and Mitigating Abstention Failures in Large Reasoning Models](answering_the_unanswerable_is_to_err_knowingly_analyzing_and.md)
- [\[ICLR 2026\] Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention](../../ICLR2026/llm_reasoning/towards_safe_reasoning_in_large_reasoning_models_via_corrective_intervention.md)
- [\[AAAI 2026\] BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models](badthink_triggered_overthinking_attacks_on_chain-of-thought_reasoning_in_large_l.md)
- [\[ICLR 2026\] Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](../../ICLR2026/llm_reasoning/uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)

</div>

<!-- RELATED:END -->
