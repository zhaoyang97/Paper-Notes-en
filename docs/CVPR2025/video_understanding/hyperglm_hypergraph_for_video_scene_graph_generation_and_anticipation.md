---
title: >-
  [Paper Note] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation
description: >-
  [CVPR 2025][Video Understanding][Video Scene Graph] HyperGLM proposes representing entity scene graphs (capturing spatial relationships) and program graphs (modeling causal temporal transitions) dynamically into a unified HyperGraph, and injecting it into a multimodal LLM to perform video scene graph generation, anticipation, and reasoning. Additionally, it releases the VSGR dataset containing 1.9 million frames to support five tasks.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Scene Graph"
  - "Hypergraph"
  - "Large Language Model"
  - "Relation Reasoning"
  - "Scene Graph Anticipation"
date: 2026-05-08
content_hash: 43556aa20107ceef
---

# HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation

**Conference**: CVPR 2025  
**arXiv**: [2411.18042](https://arxiv.org/abs/2411.18042)  
**Code**: [https://uark-cviu.github.io/projects/HyperGLM](https://uark-cviu.github.io/projects/HyperGLM)  
**Area**: Video Understanding/Scene Graph Generation  
**Keywords**: Video Scene Graph, Hypergraph, Large Language Model, Relation Reasoning, Scene Graph Anticipation

## TL;DR

HyperGLM proposes representing entity scene graphs (capturing spatial relationships) and program graphs (modeling causal temporal transitions) dynamically into a unified HyperGraph, and injecting it into a multimodal LLM to perform video scene graph generation, anticipation, and reasoning. Additionally, it releases the VSGR dataset containing 1.9 million frames to support five tasks.

## Background & Motivation

### Background

**Background**: Video Scene Graph Generation (VidSGG) aims to model multi-object relationships across video frames, serving as a foundation for high-level tasks like autonomous driving, smart surveillance, and video question answering. Recent advances have utilized Transformers and spatio-temporal contexts.

**Limitations of Prior Work**: (1) Traditional scene graph methods only model pairwise object relationships, failing to represent higher-order multi-object interactions (e.g., "a person sitting on a sofa and holding a guitar to play" involves a chained relationship among person, sofa, and guitar); (2) Progressive feature fusion and batch Transformer methods fail to capture long-range temporal dependencies; (3) Existing datasets only support scene graph generation and anticipation, lacking evaluation for reasoning capabilities (such as VQA, video captioning, and relation reasoning).

**Key Challenge**: Interactions in real-world videos are inherently many-to-many, higher-order, and temporally evolving, but traditional graph structures can only express pairwise connections, leading to insufficient representational capacity.

**Goal**: (1) Design a unified graph structure capable of expressing higher-order relationships; (2) Inject structured graph knowledge into LLMs to enable reasoning; (3) Provide a comprehensive evaluation benchmark.

## Method

### Overall Architecture

HyperGLM consists of five components: an image encoder, an MLP projector, a temporal aggregator, a unified hypergraph, and a language model. The pipeline is as follows: (1) The image encoder extracts features frame-by-frame $\rightarrow$ MLP projects them to the language space; (2) The temporal aggregator compresses $T \times N$ embeddings; (3) Entities' scene graphs and program graphs are constructed based on detected objects $\rightarrow$ unified into a hypergraph via a random walk algorithm; (4) The hypergraph is injected into the LLM as tokens, and the LLM autoregressively generates answers.

### Key Designs

1. **Unified HyperGraph**:
    - Function: Fusion of spatial relations and temporal causal relations into a unified representation.
    - Mechanism: The hypergraph consists of two parts:
        - **Entity Scene Graph**: Objects and their pairwise relations in each frame ($subject \rightarrow relation \rightarrow object$).
        - **Program Graph**: Causal transition probabilities between relations (e.g., the transition frequency from "holding" to "playing"), computed statistically from relationship changes in adjacent frames of the training data.
    - Core Advantage of Hypergraph: Hyperedges can connect multiple nodes (rather than just two), naturally expressing higher-order relations, e.g., "a person sitting on a sofa, holding a guitar, and playing a guitar" can be represented by a single hyperedge.

2. **Random Walk Hypergraph Construction Algorithm**:
    - Function: Sample representative substructures from the unified hypergraph to generate new hyperedges.
    - Mechanism: Alternating "node $\rightarrow$ hyperedge $\rightarrow$ node" random walks are executed on the hypergraph, collecting all visited nodes at each walk to form a new hyperedge. Parameters $N_w$ (number of walks) and $N_l$ (walk length) control the number and depth of hyperedges. Experiments demonstrate that $N_w = 60$ and $N_l = 7$ yield optimal performance.
    - Design Motivation: Exact subgraph matching is NP-hard. Random walks provide an efficient approximation scheme while capturing higher-order connection patterns across frames.

3. **Causal Transition Probabilities in Program Graphs**:
    - Function: Model relation evolution over time to support Scene Graph Anticipation (SGA).
    - Mechanism: Transition frequencies of relationships between adjacent frames in the training set are statistics-collected and normalized into probability distributions. During anticipation, the most likely next-step relationship is selected. Self-loops (remaining in the same relationship) are removed to focus probabilities on relationship changes.

### Loss & Training

The scene graph generation and prediction tasks use cross-entropy loss to minimize the negative log-likelihood of predicted relationship categories against ground-truth labels. The LLM part uses standard autoregressive language modeling loss. Training employs LoRA (rank=128, scaling=256) to fine-tune Mistral-7B-Instruct, taking about 6 hours on 4$\times$GPUs.

## Key Experimental Results

### Main Results

| Task/Dataset | HyperGLM | Best Baseline | Metric |
|------|------|------|------|
| SGA@R50 (Action Genome, F=0.5) | **53.5** | 51.4 (SceneSayerSDE) | Recall@50 |
| SGA@mR50 (Action Genome, F=0.5) | **40.5** | 39.9 (SceneSayerSDE) | mRecall@50 |
| SGA@R50 (Action Genome, F=0.9) | **50.0** | 47.4 (SceneSayerSDE) | Recall@50 |
| SGA@mR50 (Action Genome, F=0.9) | **38.0** | 37.1 (SceneSayerSDE) | mRecall@50 |
| VSGR Dataset Scale | **1.9M frames** | ASPIRe: 1.6M frames | Frame Count |
| Supported Tasks in VSGR | **5** | Other datasets: $\le 3$ | SGG+SGA+VQA+VC+RR |
| Optimal Hyperedge Count | 60 | - | $N_w=60, N_l=7$ |
| Training Config | 4$\times$GPU, ~6 hours | - | LoRA rank=128 |
| VQA QA Pairs | 74,856 | - | ~20 questions/video |
| VC Description Pairs | 82,532 | - | ~22 captions/video |
| RR Inference Tasks | 61,120 | - | ~16 tasks/video |

## Highlights & Insights

1. **Hypergraphs are a natural choice for representing higher-order relationships in video**: Traditional pairwise graphs fail to represent chain-like interactions of "person-object1-object2". The hyperedges of a hypergraph naturally support multi-entity connections, an advantage consistently validated in experiments.
2. **The unification of entity graphs and program graphs is an elegant design**: Spatial relations ("who is interacting with whom") and temporal evolution ("how relationships change") are modeled separately and then unified, analogous to combining schema and instance in a knowledge graph.
3. **The statistical method for relation transition probabilities is simple yet effective**: It circumvents the need to learn complex temporal models; merely counting transition frequencies in the training set provides a reliable prior for anticipation and reduces bias against low-frequency relationship categories.
4. **Comprehensiveness of the VSGR dataset**: It is the first to support five tasks (SGG, SGA, VQA, VC, RR) and covers three perspectives: third-person, egocentric, and UAV.

## Limitations & Future Work

1. Hyperparameters for random walks ($N_w$, $N_l$) require manual tuning; different datasets may require different configurations.
2. The transition probabilities in the program graph are statistically global, which may be inaccurate for specific scenarios or rare interactions.
3. Hypergraph construction and random walks introduce additional computational overhead during inference.
4. LLM inference cost is high, which may be less efficient than lightweight methods in real-world deployments.

## Related Work

- **Scene Graph Generation**: STTran (Spatio-Temporal Transformer), DSGDetr (DETR-based Scene Graph Detection), SceneSayer (ODE/SDE modeling of temporal evolution), ASPIRe (Large-scale spatial-aware scene graphs).
- **Hypergraph Applications**: HyperGraph Convolution, HyperGraph Attention, hypergraph methods in accident anticipation and group activity recognition.
- **Multimodal LLMs**: LLaVA/Video-LLaVA (visual-language reasoning), Mistral-7B (foundational language model), CLIP-ViT (visual encoding).
- **Datasets**: Action Genome (234K frames, SGG+SGA), PVSG (153K frames, SGG+VQA+VC), ASPIRe (1.6M frames, SGG only), SportsHHI (11.4K frames, SGG only).
- **Open-Vocabulary Methods**: Leveraging vision-language models to handle unseen objects and relationship categories, enhancing generalization capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ (Combining hypergraph + LLM is a brand-new practice in the video scene graph field)
- Utility: ⭐⭐⭐⭐ (The VSGR dataset supporting five tasks carries widespread value)
- Technical Depth: ⭐⭐⭐⭐ (The unified hypergraph design is theoretically grounded, and the random walk algorithm features mathematical guarantees)
- Clarity of Writing: ⭐⭐⭐ (Rich content but complex structure; multi-task evaluations add to the cognitive load)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VGEnt: Graph-Based Retrieval-Reasoning-Augmented Generation for Long Video Understanding](../../NeurIPS2025/video_understanding/vgent_graph-based_retrieval-reasoning-augmented_generation_for_long_video_unders.md)
- [\[ICML 2025\] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation](../../ICML2025/video_understanding/fine-grained_captioning_of_long_videos_through_scene_graph_consolidation.md)
- [\[CVPR 2025\] EgoTextVQA: Towards Egocentric Scene-Text Aware Video Question Answering](egotextvqa_towards_egocentric_scene-text_aware_video_question_answering.md)
- [\[CVPR 2025\] GG-SSMs: Graph-Generating State Space Models](gg-ssms_graph-generating_state_space_models.md)
- [\[CVPR 2026\] Hypergraph-State Collaborative Reasoning for Multi-Object Tracking](../../CVPR2026/video_understanding/hypergraph-state_collaborative_reasoning_for_multi-object_tracking.md)

</div>

<!-- RELATED:END -->
