---
title: >-
  [Paper Note] 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding
description: >-
  [ICCV 2025][3D Vision][3D scene understanding] This paper proposes 3DGraphLLM, which encodes semantic inter-object relationships in 3D scenes as learnable graph representations and feeds them into an LLM. The method sign…
tags:
  - "ICCV 2025"
  - "3D Vision"
  - "3D scene understanding"
  - "scene graph"
  - "large language models"
  - "vision-language tasks"
  - "semantic relations"
date: 2026-05-08
content_hash: 7f88667a32ee6e35
---

# 3DGraphLLM: Combining Semantic Graphs and Large Language Models for 3D Scene Understanding

**Conference**: ICCV 2025
**arXiv**: [2412.18450](https://arxiv.org/abs/2412.18450)
**Code**: [https://github.com/CognitiveAISystems/3DGraphLLM](https://github.com/CognitiveAISystems/3DGraphLLM)
**Area**: 3D Vision
**Keywords**: 3D scene understanding, scene graph, large language models, vision-language tasks, semantic relations

## TL;DR
This paper proposes 3DGraphLLM, which encodes semantic inter-object relationships in 3D scenes as learnable graph representations and feeds them into an LLM. The method significantly outperforms baselines that ignore semantic relations across multiple 3D vision-language tasks — including object grounding, scene captioning, and visual question answering — while achieving 5× faster inference than LVLM-based approaches.

## Background & Motivation
3D scene understanding is a foundational capability for embodied agents interacting with users, encompassing tasks such as object grounding, scene captioning, and visual question answering. LLMs have emerged as powerful tools for these tasks due to their natural language understanding and reasoning capabilities.

Existing methods for feeding 3D scenes into LLMs fall into two categories: text-based descriptions and learnable representations. Text descriptions are intuitive but may require hundreds of tokens per object, resulting in slow inference. Learnable representations are more compact and efficient; however, prior methods (e.g., Chat-Scene) rely solely on object geometry coordinates and overlook the rich semantic relationships between objects.

The root cause of the problem lies in the fact that object localization and description in 3D scenes often depend on spatial and semantic inter-object relations (e.g., "the cup on the table"), yet existing learnable representation methods cannot encode such relational information, thereby limiting LLM reasoning.

The paper's starting point is to explicitly encode semantic relations from 3D scene graphs as learnable embeddings in triplet form (object$_1$, relation, object$_2$) fed into the LLM, addressing the gap in semantic relation modeling in prior work. The core idea is to construct a flattened scene graph representation based on k-nearest-neighbor subgraphs and map semantic edge features into the LLM token embedding space.

## Method

### Overall Architecture
3DGraphLLM takes point clouds of scene objects as input, extracts 2D/3D object features and semantic relation features via pretrained encoders, and maps them into the LLM embedding space through projection layers. Each object is represented as a local subgraph (comprising itself and its $k$ nearest neighbors) and fed into the LLM as a sequence of triplets. Training proceeds in two stages: pretraining on ground-truth segmentation followed by fine-tuning on predicted segmentation.

### Key Designs
1. **Object Feature Encoding**:

    - Function: Extract 2D and 3D visual features for each object.
    - Mechanism: DINOv2 extracts 2D features $Z_i^{2d} \in \mathbb{R}^{1\times1024}$ (aggregated from multi-view masked images); Uni3D extracts 3D point cloud features $Z_i^{v_p} \in \mathbb{R}^{1\times1024}$ (aligned with text descriptions).
    - Design Motivation: DINOv2 provides rich visual semantics, while Uni3D — pretrained on large-scale data — offers strong cross-domain generalization.

2. **Semantic Relation Encoding**:

    - Function: Generate latent features encoding semantic relations for each object pair.
    - Mechanism: VL-SAT is used to produce relation features $Z_{ij}^e \in \mathbb{R}^{1\times512}$, taking latent features before the classification head rather than discrete categories to capture compositional relation semantics.
    - Design Motivation: VL-SAT requires only 3D coordinate inputs and leverages CLIP knowledge transfer for good cross-domain performance; using latent features instead of classification outputs captures non-mutually-exclusive relation combinations (e.g., "larger" and "near" simultaneously).

3. **Scene Graph Flattening Representation**:

    - Function: Convert the full scene graph into a token sequence amenable to LLM processing.
    - Mechanism: A subgraph is constructed for each object by selecting its $k$ nearest neighbors, then unrolled as triplets $(F_i^v, F_{ij}^e, F_j^v)$. The full graph requires $n(n-1)$ edges, whereas the k-NN subgraph needs only $2n + 3nk$ tokens — just 800 tokens when $k=2$, $n=100$.
    - Design Motivation: Relations with nearest neighbors are most relevant for answering user queries; the dramatic reduction in token count substantially accelerates inference.

### Loss & Training
- Standard autoregressive language modeling loss: $L(\theta) = -\sum_{i=1}^{\ell} \log P(s_i^{res} | s_{[1,...,i-1]}^{res}, s^{prefix})$
- Two-stage training: (1) pretrain projection layers and LLM on GT instance segmentation; (2) fine-tune on neural-network-predicted segmentation.
- Joint training across multiple tasks: visual grounding, scene captioning, and visual question answering.
- LLM fine-tuned with LoRA (rank=16); training takes approximately 24 hours on 4×A100 GPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric | 3DGraphLLM (LLAMA3) | Chat-Scene (Baseline) | Gain |
|--------|------|---------------------|-------------------|------|
| ScanRefer | Acc@0.5 | 56.6 | 50.2 | +6.4 |
| Multi3DRefer | F1@0.5 | 59.9 | 52.4 | +7.5 |
| Scan2Cap | CIDEr@0.5 | 81.0 | 77.1 | +3.9 |
| ScanQA | CIDEr | 88.8 | 87.7 | +1.1 |
| SQA3D | EM | 55.9 | 54.6 | +1.3 |

### Ablation Study

| Configuration | ScanRefer Acc@0.5 | Multi3DRef F1@0.5 | Note |
|------|-------------------|-------------------|------|
| 3DGraphLLM-2 (LLAMA3, pretrain+3RScan) | 56.6 | 59.9 | Full model |
| 3DGraphLLM-2 (LLAMA3, no pretrain) | 54.3 | 57.3 | Without pretraining |
| 3DGraphLLM-0 (LLAMA3, no semantic edges) | 52.0 | 55.1 | Without semantic relations |
| 3DGraphLLM-2 (Vicuna, pretrain) | 53.1 | 57.3 | Weaker LLM |
| 3DGraphLLM-0 (Vicuna, no semantic edges) | 50.2 | 52.4 | Baseline Chat-Scene |

### Key Findings
- Incorporating semantic relations yields +4.6% on ScanRefer and +4.8% on Multi3DRefer for LLAMA3, confirming the importance of semantic relation modeling.
- The two-stage pretraining strategy — first learning projections on GT segmentation, then fine-tuning — proves effective.
- 3DGraphLLM requires only 800 tokens to describe a scene, compared to 10,400 for GPT4Scene, achieving 5× faster inference.
- NMS filtering and minimum-distance filtering further improve performance under Mask3D segmentation.

## Highlights & Insights
- 3DGraphLLM is the first method to encode semantic relations from 3D scene graphs as learnable representations fed into an LLM.
- The k-NN subgraph flattening strategy elegantly balances information completeness with computational efficiency.
- Using VL-SAT's latent features rather than classification outputs is a principled design choice that avoids information loss due to discretization.
- The two-stage training pipeline (GT pretraining followed by predicted-segmentation fine-tuning) effectively mitigates the impact of segmentation noise on semantic relation encoding.
- High inference efficiency: only 800 tokens per scene, with ScanRefer inference taking just 0.4 seconds per query.

## Limitations & Future Work
- N-gram metrics (CIDEr/BLEU) may not accurately evaluate the rich descriptions generated by LLMs, potentially underestimating model performance.
- The VL-SAT relation encoder is trained on 3RScan; cross-domain transfer to ScanNet may introduce a performance gap.
- A fixed $k=2$ neighborhood size may lack flexibility for scenes of varying complexity.
- The method has a strong dependency on instance segmentation quality; segmentation noise directly affects graph construction and relation extraction.
- Validation is limited to indoor scenes (ScanNet/3RScan); applicability to large-scale outdoor scenes remains unexplored.

## Related Work & Insights
- **vs. Chat-Scene**: 3DGraphLLM extends it with semantic relation encoding, yielding substantial gains in object grounding (ScanRefer +6.4%).
- **vs. GPT4Scene**: Achieves comparable quality with 13× fewer tokens (800 vs. 10,400) and 5× faster inference, demonstrating the advantage of compact representations.
- **vs. Robin3D**: Reaches Robin3D-level performance (trained on 1M data) using only 370K data, evidencing the data efficiency of semantic graph representations.
- **vs. expert models (e.g., 3D-VisTA)**: Highlights the generality of LLM-based approaches — a single model handles multiple tasks.
- **Insight**: The paradigm of injecting structured knowledge (graphs) into LLMs is generalizable to other domains, such as knowledge-graph-augmented dialogue systems.

## Rating
- Novelty: ⭐⭐⭐⭐ First to integrate scene graph semantic relations into LLMs via learnable representations; the idea is clear and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 5 benchmarks with detailed ablations covering both GT and predicted segmentation settings.
- Writing Quality: ⭐⭐⭐⭐ Well-structured, with detailed method descriptions and intuitive prompt template examples.
- Value: ⭐⭐⭐⭐ Provides an efficient graph–LLM fusion paradigm for 3D scene understanding with practical relevance to embodied intelligence.
- Overall: Solid work that strikes a strong balance between compact representation and semantic modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Open-Vocabulary Octree-Graph for 3D Scene Understanding](open-vocabulary_octree-graph_for_3d_scene_understanding.md)
- [\[ICCV 2025\] Global-Aware Monocular Semantic Scene Completion with State Space Models](global-aware_monocular_semantic_scene_completion_with_state_space_models.md)
- [\[ICCV 2025\] Articulate3D: Holistic Understanding of 3D Scenes as Universal Scene Description](articulate3d_holistic_understanding_of_3d_scenes_as_universal_scene_description.md)
- [\[ICCV 2025\] ExCap3D: Expressive 3D Scene Understanding via Object Captioning with Varying Detail](excap3d_expressive_3d_scene_understanding_via_object_captioning_with_varying_det.md)
- [\[ICCV 2025\] Monocular Semantic Scene Completion via Masked Recurrent Networks](monocular_semantic_scene_completion_via_masked_recurrent_networks.md)

</div>

<!-- RELATED:END -->
