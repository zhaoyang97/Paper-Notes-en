---
title: >-
  [Paper Note] Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models
description: >-
  [NeurIPS 2025][Multimodal VLM][Urban Design] This paper proposes a human-AI collaborative computer vision framework that employs Grounding DINO for urban object detection…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Urban Design"
  - "Human-AI Collaboration"
  - "Co-Occurrence Embeddings"
  - "VLM Recommendation"
  - "AR Interaction"
date: 2026-05-08
content_hash: 33e15d1a9d83c742
---

# Scene-Aware Urban Design: A Human-AI Recommendation Framework Using Co-Occurrence Embeddings and Vision-Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2511.06201](https://arxiv.org/abs/2511.06201)  
**Code**: None  
**Area**: Multimodal VLM
**Keywords**: Urban Design, Human-AI Collaboration, Co-Occurrence Embeddings, VLM Recommendation, AR Interaction

## TL;DR
This paper proposes a human-AI collaborative computer vision framework that employs Grounding DINO for urban object detection, constructs co-occurrence embeddings from the ADE20K dataset to capture real-world spatial configurations, leverages a VLM for scene-aware third-object recommendation, and generates 3D models for AR preview — all aimed at enabling residents to participate in micro-scale urban design.

## Background & Motivation

The quality of urban public spaces is shaped not only by large-scale master planning but also by small, everyday interventions: a bench beneath a tree, a bike rack beside a storefront, a shade structure in a plaza. These micro-scale decisions are critical to the lived urban experience, yet residents' capacity to participate in such design is constrained by regulations, resources, and a lack of design expertise.

The root cause lies in a fundamental tension: municipal design processes often lack the spatial knowledge accumulated through residents' everyday use, while residents themselves lack the tools to make informed spatial decisions. Existing tangible user interfaces (e.g., CityScope) excel at macro-level tasks — zoning, density, transport networks — but offer limited support for object-level everyday design.

The paper's starting point is to learn object co-occurrence patterns from existing urban scene data using computer vision, combine this with the scene reasoning capabilities of VLMs, and provide residents with evidence-grounded micro-scale design suggestions within a human-in-the-loop framework — positioning AI as a collaborator in spatial creation rather than an automated replacement.

## Method

### Overall Architecture

The pipeline consists of four stages: (1) scene filtering and object detection — analyzing the ADE20K dataset using Grounding DINO; (2) co-occurrence aggregation and embedding — statistically capturing spatial associations among objects; (3) VLM recommendation — generating scene-aware third-object suggestions via a vision-language model; and (4) 3D model generation — producing previewable three-dimensional models of recommended objects using text-to-3D technology. Users participate in decision-making at two levels: selecting anchor and co-occurring objects, and accepting, rejecting, or regenerating VLM suggestions.

### Key Designs

1. **Scene Filtering and Object Detection**: The pipeline uses the ADE20K dataset (20,000+ annotated images) and Grounding DINO (open-vocabulary, zero-shot detection). Scenes are first filtered by category, and only those in which Grounding DINO detects five or more high-confidence pedestrians are retained — using crowd density as a heuristic proxy for socially active spaces. This two-stage filtering yields approximately 900 images. Grounding DINO's open-vocabulary capability enables detection of micro-urban elements absent from conventional closed-label detection datasets.

2. **Co-Occurrence Aggregation and Embedding**: For each filtered image, all unordered pairs of urban objects are recorded and their co-occurrence counts accumulated in a symmetric co-occurrence matrix. For a scene containing a bench, a tree, and a trash can, the matrix entries for (bench, tree), (bench, trash can), and (tree, trash can) are each incremented by one. Each row of the matrix is normalized to a conditional probability vector: $P(o_j|o_i) = \frac{\text{count}(o_i \wedge o_j)}{\text{count}(o_i)}$. The embedding of each object is its conditional probability vector $o = [P(o_1|o), P(o_2|o), \ldots, P(o_n|o)]$. These embeddings are not learned via backpropagation but are constructed empirically from real-world spatial data.

3. **VLM Scene Recommendation**: After the user selects an anchor object and a system-suggested co-occurring object, the VLM receives three inputs: the full scene image, cropped and normalized bounding boxes of the anchor and co-occurring objects, and a compact scene summary (scene type, five-color palette, dominant materials, and a rough depth sketch). The VLM returns five candidate objects, each with an object type, material and surface finish, approximate dimensions, color hints, simple placement guidelines relative to the anchor, and a one-sentence rationale. The system also filters out infeasible suggestions (e.g., crosswalks when no street edge is present).

### Loss & Training

No model training is involved. The co-occurrence matrix is constructed through statistical aggregation, the VLM (GPT-4 Vision) is used via zero-shot prompting, and 3D model generation relies on the Meshy API. The entire system is modular, with each component independently replaceable.

## Key Experimental Results

### Main Results (Co-Occurrence Embedding Top-5)

| Anchor Object | Embed. 1 | Embed. 2 | Embed. 3 | Embed. 4 | Embed. 5 |
|--------------|----------|----------|----------|----------|----------|
| bench | window | tree | sign | traffic light | crosswalk |
| tree | traffic light | window | sidewalk | door | planter |
| planter | tree | sidewalk | window | balcony | traffic light |
| sign | traffic light | window | crosswalk | tree | sidewalk |
| trash can | window | tree | traffic light | sign | door |

### Ablation Study (VLM Recommendation Quality Analysis)

| Scene Feature | VLM Recommendation Examples | Evaluation |
|--------------|----------------------------|------------|
| Park scene + bench anchor | outdoor chess table, drinking fountain, bike rack | Functional complementarity, stylistic coherence |
| Street scene + traffic light anchor | bus stop, information kiosk, directional signage | Contextual reasoning beyond statistical co-occurrence |
| Residential area + balcony anchor | curb, lamp post, trash can | Infrastructure completion is reasonable |
| 3D generation failure cases | inaccurate chess table pattern, drinking fountain missing base basin | Meshy API limited in expressing fine-grained details |

### Key Findings

- Co-occurrence embeddings reveal intuitive spatial associations: high co-occurrence of benches with trees, trash cans, and signs aligns with common public space configurations.
- VLM recommendations go beyond purely statistical co-occurrence lists by integrating visual context, spatial cues, and object semantics to produce more functionally and site-specifically appropriate suggestions — such as bus stops and information kiosks that statistical lists would not surface.
- Grounding DINO exhibits false detections in visually cluttered or low-resolution images (e.g., confusing planters with trash cans), necessitating visual post-processing and vocabulary refinement.
- The 3D generation stage suffers from detail loss; Meshy AI sometimes fails to fully capture all details specified in the prompt.
- The two-tier human-AI interaction design — user selects anchor and co-occurring object, then reviews VLM suggestions — effectively preserves user agency throughout the pipeline.

## Highlights & Insights

- Positioning AI as "a collaborator in everyday spatial creation" rather than an automated design tool is well-motivated — it preserves residents' spatial knowledge and design intent.
- The two-layer architecture combining statistical co-occurrence embeddings with VLM semantic reasoning is pragmatic: the first layer provides evidence-grounded candidates, while the second transcends statistics for scene-aware recommendation.
- The completeness of the end-to-end pipeline (detection → co-occurrence → recommendation → 3D generation → AR preview) is impressive, demonstrating a viable path from research to prototype.
- Using "crowd density > 5" as a proxy for socially active spaces is simple but establishes a reasonable baseline for future work.

## Limitations & Future Work

- Co-occurrence estimation is based on pixel-level distances in 2D images and cannot capture true 3D spatial relationships, reducing precision for site-specific interventions.
- The ADE20K dataset is skewed toward particular geographic and cultural contexts (primarily North America/Europe), limiting generalization to other regions.
- The VLM may overlook location-specific social, cultural, or legal constraints; suggestions may be visually plausible but socially inappropriate or non-compliant.
- The VLM cannot assess installation feasibility (terrain, underground utilities, budget, and other non-visual factors).
- No participatory user evaluation has been conducted — it remains untested how AI suggestions influence or deviate from user intent.
- The 3D generation model (Meshy) requires improved fidelity in capturing fine-grained details.

## Related Work & Insights

- Tangible interfaces such as CityScope excel at macro-level planning but fall short at micro-scale design; this paper addresses that gap.
- Grounding DINO's open-vocabulary zero-shot detection enables the system to discover micro-urban elements absent from traditional closed-label datasets.
- Unlike fully automated design systems, the paper's human-in-the-loop design aligns with Kindberg et al.'s notion of "physical objects as queryable agents."
- A key insight is that combining statistical priors (co-occurrence) with semantic reasoning (VLM) allows recommendations to remain evidence-grounded while transcending the limitations of purely data-driven approaches.

## Rating

- Novelty: ⭐⭐⭐⭐ (Cross-domain fusion of urban design × CV × VLM × AR is novel; individual techniques are not original)
- Experimental Thoroughness: ⭐⭐⭐ (Prototype validation is solid but lacks quantitative metrics and user studies)
- Writing Quality: ⭐⭐⭐⭐ (Problem framing is clear; system design is described comprehensively)
- Value: ⭐⭐⭐ (Concept is meaningful but practical deployment remains distant; application scope is relatively narrow)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models](text_to_robotic_assembly_of_multi_component_objects_using_3d_generative_ai_and_v.md)
- [\[NeurIPS 2025\] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)
- [\[ICML 2026\] Benchmarks for Vision-Language Models in Urban Perception Should Be Reliability-Aware and Negotiated](../../ICML2026/multimodal_vlm/benchmarks_for_vision-language_models_in_urban_perception_should_be_reliability-.md)
- [\[NeurIPS 2025\] Test-Time Spectrum-Aware Latent Steering for Zero-Shot Generalization in Vision-Language Models](test-time_spectrum-aware_latent_steering_for_zero-shot_generalization_in_vision-.md)
- [\[CVPR 2026\] Scene-VLM: Multimodal Video Scene Segmentation via Vision-Language Models](../../CVPR2026/multimodal_vlm/scene-vlm_multimodal_video_scene_segmentation_via_vision-language_models.md)

</div>

<!-- RELATED:END -->
