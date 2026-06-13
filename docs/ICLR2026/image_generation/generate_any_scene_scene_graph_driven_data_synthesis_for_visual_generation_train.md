---
title: >-
  [Paper Note] Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training
description: >-
  [ICLR 2026][Image Generation][Scene Graph] This paper proposes the Generate Any Scene data engine, which systematically enumerates scene graphs from a visual element taxonomy comprising 28K objects × 1.5K attributes × 10…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "Scene Graph"
  - "Compositional Generation"
  - "Data Engine"
  - "Self-Improvement"
  - "Targeted Distillation"
  - "Reward Model"
date: 2026-05-08
content_hash: 8a7504a8ca89ff26
---

# Generate Any Scene: Scene Graph Driven Data Synthesis for Visual Generation Training

**Conference**: ICLR 2026
**arXiv**: [2412.08221](https://arxiv.org/abs/2412.08221)  
**Code**: [GitHub](https://github.com/RAIVNLab/GenerateAnyScene)  
**Area**: Image Generation / Data Synthesis
**Keywords**: Scene Graph, Compositional Generation, Data Engine, Self-Improvement, Targeted Distillation, Reward Model

## TL;DR
This paper proposes the Generate Any Scene data engine, which systematically enumerates scene graphs from a visual element taxonomy comprising 28K objects × 1.5K attributes × 10K relations, and converts them into caption–VQA pairs. The engine supports four applications: self-improvement (SD1.5 +4%), targeted distillation (<800 samples, TIFA +10%), a scene-graph reward model (DPG-Bench +5% vs. CLIP), and content moderation enhancement.

## Background & Motivation

**Background**: Text-to-image generation models (e.g., DALL-E 3, SD, Flux) have achieved high levels of visual fidelity, yet remain severely deficient in compositional generalization and semantic alignment. A typical failure case: given the prompt "a black dog chasing a rabbit in Van Gogh style," a model may generate the dog while omitting the rabbit or rendering the wrong style.

**Limitations of Prior Work**: The root of the problem lies in training data. Mainstream datasets such as LAION and CC3M consist of web-crawled image–caption pairs that are inherently noisy, compositionally impoverished, and biased toward coarse single-object descriptions. These datasets lack explicit annotations of object–attribute relations and multi-object interactions, limiting models' ability to generalize to complex scenes. Dense compositional annotation by humans is not scalable, while automated VLM-based annotation introduces hallucinations and semantic noise.

**Key Challenge**: Large-scale, high-quality, compositionally rich training data are required, yet no systematic method exists for producing data that covers the visual compositional space. Existing evaluation tools (DSG, DPG) already employ scene graphs to assess generation quality, but scene graphs have never been systematically applied on the data production side.

**Goal**: How to construct a scalable data engine that systematically generates compositionally rich training data (captions + evaluation signals + reward signals) to improve the compositional generalization and semantic alignment of generative models?

**Key Insight**: Scene graphs—grounded in cognitive science as a structured representation of visual space, with objects as nodes, attributes as node properties, and relations as edges—enable near-infinite compositional scene descriptions through systematic enumeration of graph topologies populated with metadata. VQA pairs for evaluation and reward modeling are automatically derived as a byproduct.

**Core Idea**: Scene graphs serve as an intermediate representation for systematically enumerating the visual compositional space. A data engine is constructed to simultaneously produce training captions and fine-grained evaluation signals, driving self-improvement, distillation, and RLHF training for generative models.

## Method

### Overall Architecture
Generate Any Scene is a five-stage data engine pipeline: (1) enumerate scene graph topologies under user-specified structural constraints; (2) sample objects, attributes, and relations from the visual element taxonomy to populate the scene graph; (3) sample scene-level attributes (style, viewpoint, etc.); (4) deterministically translate the scene graph into a natural language caption; (5) automatically enumerate exhaustive VQA pairs from the scene graph. Building on this engine, the paper proposes four downstream applications: self-improvement, targeted distillation, a scene-graph reward model, and content moderation enhancement.

### Key Designs

1. **Visual Element Taxonomy and Scene Graph Enumeration**

    - **Function**: Constructs a structured knowledge base of visual concepts to support systematic enumeration of scene graphs of arbitrary complexity.
    - **Mechanism**: Objects, attributes, relations, and scene-level attributes are collected from multiple sources—WordNet, Wikipedia, Visual Genome, Places365, etc.—yielding 28,787 objects, 1,494 attributes, 10,492 relations, and 2,193 scene attributes, organized into a hierarchical taxonomy (e.g., flower → daisy → white daisy). Scene graph enumeration begins by sampling the number of object nodes, then systematically enumerates edge sets and attribute assignments satisfying degree-sequence constraints. Three types of control are supported: degree upper-bound constraints, seed-graph preservation (embedding user-supplied subgraphs), and commonsense plausibility filtering. All enumerations are precomputed and cached as parameter tuples.
    - **Design Motivation**: The compositional diversity of existing datasets is determined by the distribution of naturally occurring data, which is inevitably biased toward common scenes. Systematic enumeration covers rare compositions and fills the long-tail gaps in training data. The hierarchical taxonomy allows granularity to be controlled from coarse to fine.

2. **Bidirectional Translation from Scene Graphs to Captions and VQA Pairs**

    - **Function**: Converts structured scene graphs into text captions consumable by generative models, while simultaneously producing VQA pairs for evaluation and reward modeling.
    - **Mechanism**: Caption translation uses a deterministic procedural algorithm—traversing the scene graph in topological order and converting objects, attributes, and relations into descriptive text, with grammatical disambiguation rules (e.g., "the first/second" to distinguish identical objects). VQA generation uses templates to query object attributes ("What color is the sphere?"), spatial relations ("What is to the left of the cube?"), and other properties; each answer maps directly to a node or edge in the scene graph, ensuring completeness.
    - **Design Motivation**: Procedural translation is fast and free from hallucination risk (empirically showing no significant difference from LLM-based paraphrasing). VQA pairs enable fine-grained semantic evaluation at zero additional cost—covering every element of the scene graph and being far more precise than coarse metrics such as CLIPScore.

3. **Unified Application Framework: Self-Improvement, Distillation, and RLHF**

    - **Function**: Leverages data engine outputs to drive three complementary model improvement strategies.
    - **Mechanism**: (a) *Self-improvement*: for each synthetic caption, 8 images are generated; the top-25% by VQAScore are selected as fine-tuning data for the next round, iterated for 3 epochs. (b) *Targeted distillation*: Generate Any Scene captions are used to systematically evaluate open-source versus closed-source models, identifying weaknesses in the open-source model (e.g., SD1.5's poor multi-object composition); fewer than 800 DALL-E 3 image–caption pairs targeting the identified capability gap are then used for LoRA fine-tuning. (c) *Scene-graph reward model*: VQA accuracy (answered by Qwen2.5-VL-3B) serves as a reward signal; GRPO is used to train SimpleAR-0.5B.
    - **Design Motivation**: The three strategies address distinct bottlenecks—self-improvement requires no external data but is bounded by the model's own capability; distillation requires a teacher model but demands very little data; RLHF provides the finest-grained semantic alignment signal. All three rely on the structured captions and evaluation capabilities provided by Generate Any Scene.

### Loss & Training
Both self-improvement and distillation employ LoRA for parameter-efficient fine-tuning. Each self-improvement round generates 10K captions × 8 images, selecting the top-25% (i.e., 2.5K pairs). Distillation uses only 778 captions. RLHF applies the GRPO algorithm, training for 1 epoch on 10K captions with VQA accuracy as the reward.

## Key Experimental Results

### Main Results

Self-improvement (SDv1.5 baseline, evaluated on GenAI-Bench + 1K Generate Any Scene evaluation set):

| Method | CLIPScore | ImageReward | LPIPS | VQAScore |
|--------|-----------|-------------|-------|----------|
| SDv1.5 (original) | 0.3167 | 0.2056 | 0.7297 | 0.5823 |
| CC3M real-data fine-tuning | 0.3196 | 0.3842 | 0.7356 | 0.6044 |
| **GAS self-improvement** | **0.3206** | **0.3927** | **0.7329** | **0.6109** |

RLHF reward model comparison (SimpleAR-0.5B):

| Method | DPG-Bench Global | DPG Relation | GenEval Overall | GenAI All |
|--------|-----------------|-------------|-----------------|-----------|
| SFT baseline | 85.02 | 86.59 | 0.53 | 0.66 |
| CLIP-RL | 86.64 | 88.51 | 0.59 | 0.67 |
| **GAS Reward (Ours)** | **88.46** | **90.13** | **0.61** | **0.68** |

### Ablation Study

Targeted distillation (SDv1.5, TIFA Score vs. caption complexity):

| Configuration | TIFA (multi-object captions) | Gain | Notes |
|---------------|------------------------------|------|-------|
| SDv1.5 (original) | ~0.50 | — | Poor compositional ability |
| Random-caption distillation | ~0.55 | +5% | Non-targeted data |
| **Targeted multi-object distillation** | **~0.60** | **+10%** | Only 778 captions |

Compositional generalization test (400 unseen compositional captions):

| Method | VQAScore | CLIPScore |
|--------|----------|-----------|
| SDv1.5 | 0.5823 | 0.2876 |
| CC3M-FT | 0.6044 | 0.2927 |
| **GAS-FT** | **0.6109** | **0.2938** |

### Key Findings
- Fine-tuning on synthetic data consistently outperforms fine-tuning on real CC3M data of the same scale, demonstrating that structured compositional data is more valuable than naturally occurring data.
- Targeted distillation achieves TIFA +10% with only 778 captions, substantially outperforming random distillation (+5%)—pinpointing weaknesses matters more than using large amounts of data.
- GRPO with the scene-graph reward surpasses the CLIP reward by +1.8% on DPG-Bench (88.46 vs. 86.64), with particularly pronounced gains on GenEval two-object tasks.
- Generation diversity (LPIPS) is preserved after fine-tuning, indicating that improved semantic alignment does not require sacrificing diversity.

## Highlights & Insights
- The paradigm of "data engine over model modification"—improving capability systematically through better training data rather than architectural changes—aligns with the data-centric AI trend.
- The dual role of scene graphs is particularly elegant: they serve simultaneously as a structured template for data generation (input side) and as a completeness guarantee for evaluation and reward modeling (output side). This "generation–evaluation closed loop" makes the entire pipeline self-consistent.
- The efficiency of targeted distillation with 778 samples is impressive. By using Generate Any Scene to systematically identify weaknesses and then selectively supplementing with strong-model data, the approach offers a broadly applicable "precision gap-filling" strategy.
- VQA pairs as reward signals are more fine-grained than the holistic image–text matching score of CLIP, enabling attribution of errors to specific objects, attributes, or relations.

## Limitations & Future Work
- Self-improvement relies on VQAScore as the selection signal, but VQAScore itself may have blind spots for certain compositions.
- Targeted distillation depends on a closed-source teacher model (DALL-E 3) and cannot compensate for domains where the teacher is also weak.
- Although the compositional enumeration is systematic, it remains bounded by the taxonomy's coverage—28K objects cannot encompass all visual concepts.
- The content moderation experiment is relatively small-scale (5K images); larger-scale validation is needed.
- Applying the scene graph engine to training data augmentation for video generation has not been explored.

## Related Work & Insights
- **vs. DreamSync**: DreamSync also performs self-improvement but without structured scene graphs; the systematic enumeration in Generate Any Scene provides broader compositional coverage.
- **vs. DSG/DPG**: These works use scene graphs for evaluation; Generate Any Scene extends scene graphs from the evaluation side to the data production side, closing the loop.
- **vs. DALL-E 3 recaptioning**: DALL-E 3 uses VLM-based recaptioning to improve caption quality; Generate Any Scene uses procedural generation to avoid VLM hallucinations.
- The scene graph data engine paradigm is generalizable to any task requiring structured coverage of a compositional space (e.g., video generation, 3D generation, robot instruction following).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The closed-loop design extending scene graphs from evaluation to data generation is novel; the unified framework across four applications is coherent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ All four application scenarios are quantitatively validated, including compositional generalization tests.
- **Writing Quality**: ⭐⭐⭐⭐ The overall structure is clear and the pipeline description is thorough.
- **Value**: ⭐⭐⭐⭐⭐ The data-centric paradigm offers important insights for T2I training; the targeted distillation strategy has strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Consistent Text-to-Image Generation via Scene De-Contextualization](consistent_text-to-image_generation_via_scene_de-contextualization.md)
- [\[NeurIPS 2025\] SceneDecorator: Towards Scene-Oriented Story Generation with Scene Planning and Scene Consistency](../../NeurIPS2025/image_generation/scenedecorator_towards_scene-oriented_story_generation_with_scene_planning_and_s.md)
- [\[ICCV 2025\] Lay-Your-Scene: Natural Scene Layout Generation with Diffusion Transformers](../../ICCV2025/image_generation/lay-your-scene_natural_scene_layout_generation_with_diffusion_transformers.md)
- [\[AAAI 2026\] STELLAR: Scene Text Editor for Low-Resource Languages and Real-World Data](../../AAAI2026/image_generation/stellar_scene_text_editor_for_low-resource_languages_and_real-world_data.md)
- [\[ICLR 2026\] PolyGraph Discrepancy: a classifier-based metric for graph generation](polygraph_discrepancy_a_classifier-based_metric_for_graph_generation.md)

</div>

<!-- RELATED:END -->
