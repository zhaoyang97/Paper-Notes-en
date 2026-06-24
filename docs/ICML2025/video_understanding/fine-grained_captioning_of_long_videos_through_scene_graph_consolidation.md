---
title: >-
  [Paper Note] Fine-Grained Captioning of Long Videos through Scene Graph Consolidation
description: >-
  [ICML 2025][Video Understanding][Long Video Captioning] This paper proposes the SGVC framework, which achieves state-of-the-art zero-shot long video captioning performance while substantially reducing computational overhead compared to LLM-based methods. It parses segment-level video descriptions into scene graphs, iteratively consolidates them into a unified graph representation using the Hungarian algorithm, and generates video-level descriptions using a lightweight graph-t…
tags:
  - "ICML 2025"
  - "Video Understanding"
  - "Long Video Captioning"
  - "Scene Graphs"
  - "Graph Consolidation"
  - "Zero-Shot Video Captioning"
  - "Graph-to-Text Generation"
date: 2026-05-08
content_hash: e0f6acdc000d40a9
---

# Fine-Grained Captioning of Long Videos through Scene Graph Consolidation

**Conference**: ICML 2025  
**arXiv**: [2502.16427](https://arxiv.org/abs/2502.16427)  
**Code**: None  
**Area**: Video Understanding  
**Keywords**: Long Video Captioning, Scene Graphs, Graph Consolidation, Zero-Shot Video Captioning, Graph-to-Text Generation

## TL;DR

This paper proposes the SGVC framework, which achieves state-of-the-art zero-shot long video captioning performance while substantially reducing computational overhead compared to LLM-based methods. It parses segment-level video descriptions into scene graphs, iteratively consolidates them into a unified graph representation using the Hungarian algorithm, and generates video-level descriptions using a lightweight graph-to-text decoder.

## Background & Motivation

While Vision-Language Models (VLMs) have achieved excellent performance in image and short video captioning, generating coherent and comprehensive descriptions for long videos remains a major challenge. The **Key Challenge** lies in the limited temporal receptive field of existing models, which makes them unable to encode an entire long video at once.

To address this issue, existing solutions can be divided into three categories, but each has drawbacks:

**Memory-based / Recursive frameworks**: Require supervised fine-tuning on target datasets, resulting in poor generalization.

**LLM-based summarization methods** (e.g., VidIL, Video ChatCaptioner): Utilize LLMs to summarize information across multiple segments; they require no fine-tuning but incur **enormous inference overhead** (requiring proprietary GPT APIs or 7B+ parameter models), and LLMs occasionally ignore scene details or hallucinate.

**Zero-shot methods** (e.g., ZeroCap, MAGIC): Guide language models using CLIP, but perform poorly on videos with complex events.

The **Key Insight** of this paper is highly unique: **Instead of using massive language models for text-level summarization, information fusion is performed at the structured scene graph level.** Segment-level video captions are parsed into scene graphs (objects + attributes + relations), multiple scene graphs are consolidated into a unified representation via a graph consolidation algorithm, and a lightweight decoder with only 235M parameters generates the final description. This preserves fine-grained object-level details while avoiding the heavy computational cost of LLMs.

## Method

### Overall Architecture

The SGVC (Scene Graph-based Video Captioning) framework consists of four stages:
1. **Segment-Level Caption Generation**: The video is uniformly divided into multiple segments (frames or short video clips), and off-the-shelf VLMs (such as BLIP, BLIP2, InternVL2.5) are used to generate descriptions for each segment.
2. **Scene Graph Parsing**: A textual scene graph parser (FACTUAL-MR parser) is used to translate each segment-level caption into a scene graph.
3. **Scene Graph Consolidation**: Iteratively merges all segment-level scene graphs into a unified graph representation.
4. **Video Caption Generation**: A graph-to-text model decodes the consolidated scene graph into the final video-level description.

The entire framework is **training-free**—requiring no fine-tuning of any component on target long-video datasets, and is compatible with any off-the-shelf VLM.

### Key Designs

1. **Scene Graph Definition and Parsing**: A scene graph $G = (\mathcal{O}, \mathcal{E})$ consists of a set of objects and a set of edges. Each object $o_i = (c_i, \mathcal{A}_i)$ contains a category label and a set of attributes, and each directed edge $e_{i,j}$ carries a relationship label $r_{i,j}$. The FACTUAL-MR parser is used to map textual captions into intermediate semantic representations (objects, attributes, relationships), which are then deterministically converted into scene graphs. This structured representation is more suitable for information fusion than raw text, as occurrences of the same object across different frames can be precisely matched and merged.

2. **Scene Graph Consolidation Algorithm**: This is the core of the method. The consolidation process is iterative: in each round, the **most similar pair of graphs** in the graph pool is selected for merging until only one unified graph remains.

   Steps to consolidate two graphs:
    - Use a **graph encoder** $\phi(\cdot)$ to encode both graphs, obtaining embedding representations for each object.
    - Solve the optimal object matching problem using the **Hungarian algorithm**, where the objective function is based on the **cosine similarity** of object embeddings:
    $$\pi^* = \arg\max_{\pi \in \Pi} \sum_i \frac{\psi_i(\phi(G^s))}{\|\psi_i(\phi(G^s))\|} \cdot \frac{\psi_i(\phi(G_\pi^t))}{\|\psi_i(\phi(G_\pi^t))\|}$$
    - For matching pairs $(o_p^s, o_q^t)$ with similarity exceeding a threshold $\tau$, merge the two objects **into one**: $\hat{o}_m = (\hat{c}, \mathcal{A}_p^s \cup \mathcal{A}_q^t)$, where the category label $\hat{c}$ may differ from the original labels (inferred via the encoder).
    - Update the edge set of the consolidated graph: redirect edges originally connected to the merged objects to the newly formed merged object.

   Varying numbers of objects are aligned by introducing dummy objects before matching. The performance is stable when the threshold $\tau$ is within the range of 0.80-0.95; the experiments uniformly use 0.9.

3. **Priority Subgraph Extraction**: When concise video descriptions are required, the **merge count** of each node is tracked during consolidation as a measure of importance, and the top-k nodes with the highest merge counts along with their subgraphs are extracted. A high merge count indicates that the object repeatedly appears in multiple video frames and is highly likely to be a key entity. Setting $k=1$ produces the most concise subgraph (improving precision metrics), while larger $k$ retains more context (improving recall metrics).

4. **Graph-to-Text Model**:

    - **Graph Encoder**: Based on BERT-base, applying an **attention mask** to restrict attention propagation only along the edges defined by the scene graph (rather than global attention) to preserve graph structure. A learnable **global embedding token** is added to allow disconnected subgraphs to exchange information.
    - **Text Decoder**: Utilizes the decoder portion of T5-base.
    - Total parameters count is only **235M** (compared to 7.5B for Mistral-7B).

### Loss & Training

- The graph-to-text model is trained with a **next-token prediction** objective: $\mathcal{L}(\theta) = \sum_{i=1}^{N} \log P_\theta(t_i | t_{1:i-1}, G)$
- Training data: Approximately **2.5 million** graph-caption pairs from image captioning datasets such as MS-COCO, Flickr30k, TextCaps, Visual Genome, as well as video captions generated for Kinetics-400 using LLaVA-NeXT-7B.
- Trained for 1K iterations with a batch size of 512, using the AdamW optimizer with a learning rate of 0.0001.
- The video paragraph captioning task is further fine-tuned on Visual Genome Paragraph Captions for 400 iterations.
- Inference uses beam search (5 beams, max length 32, length penalty 0.6).

## Key Experimental Results

### Main Results

Zero-shot video captioning (MSR-VTT and MSVD):

| Method | Backbone | B@4 | METEOR | CIDEr | F_BERT |
|------|----------|-----|--------|-------|--------|
| VidIL (zero-shot) | BLIP+CLIP | 3.2 | 14.8 | 3.1 | 0.225 |
| Video ChatCaptioner | BLIP2 | 13.2 | 22.0 | 16.5 | 0.436 |
| **SGVC (Ours)** | **BLIP2** | **18.4** | **23.1** | **26.1** | **0.487** |
| VidIL† (few-shot) | BLIP+CLIP | 13.6 | 20.0 | 20.2 | 0.490 |

On MSR-VTT, the zero-shot CIDEr score of SGVC (26.1) even outperforms VidIL† (20.2), which utilizes reference descriptions for few-shot prompting.

Zero-shot video paragraph captioning (ActivityNet Captions):

| Method | Backbone | B@4 | METEOR | CIDEr | F_BERT |
|------|----------|-----|--------|-------|--------|
| Video ChatCaptioner | BLIP2 | 2.4 | 8.9 | 1.6 | 0.200 |
| Summarization w/ GPT-4o mini | InternVL2.5 | 5.8 | 11.4 | 15.3 | 0.336 |
| **SGVC (Ours)** | **InternVL2.5** | **8.0** | **13.2** | **24.1** | **0.338** |

On the long video paragraph captioning task, SGVC's CIDEr (24.1) significantly outperforms the GPT-4o mini summarization method (15.3), yielding a performance gain of over 57%.

### Ablation Study

Influence of the $k$-value in subgraph extraction (MSR-VTT, BLIP2 backbone):

| k | METEOR | CIDEr | P_BERT | R_BERT | F_BERT |
|-----|--------|-------|--------|--------|--------|
| 1 | 23.1 | **26.1** | **0.467** | 0.542 | **0.487** |
| 3 | **23.8** | 24.9 | 0.454 | **0.554** | 0.486 |

Influence of consolidating threshold $\tau$ (MSVD, stability analysis):

| τ | CIDEr | F_BERT |
|---|-------|--------|
| 0.95 | 50.0 | 0.589 |
| 0.90 | **50.2** | **0.589** |
| 0.85 | 49.9 | 0.589 |
| 0.80 | 49.9 | 0.589 |

### Key Findings

- **Significant Computational Efficiency**: SGVC (BLIP backbone) requires only 0.74B parameters, 5.07GB GPU memory, and 1.14s/video, whereas Mistral-7B summarization requires 7.5B parameters, 14.5GB GPU memory, and 1.27s/video. SGVC achieves superior performance with less than 1/10 of the parameters.
- **Scene Graph Consolidation vs. LLM Summarization**: Under identical segment-level caption inputs, scene graph consolidation significantly outperforms LLM summarization on CIDEr (24.0 vs 10.8 on MSR-VTT with BLIP), demonstrating that structured representations preserve information much better than raw text summarization.
- **Hallucination in LLM Methods**: Video ChatCaptioner aggregates information using multi-turn QA, which frequently leads to hallucinations (e.g., "no animals in the park scene"), whereas the scene-graph-based method effectively avoids this problem via structured representation.
- **Backbone Flexibility**: The framework is plug-and-play with different VLMs (BLIP, BLIP2, InternVL2.5), where stronger backbones yield consistent performance gains.

## Highlights & Insights

1. **Elegant "Structured Intermediate Representation" Approach**: Instead of having an LLM perform vague summarization in text space, this method converts text into structured scene graphs to perform precise object matching and consolidation in graph space. This strategy prevents the loss of detail and hallucinations inherent in LLMs.
2. **Hungarian Algorithm for Optimal Matching**: Unlike simple text similarity comparisons, the Hungarian algorithm matches objects across two graphs to achieve global optimality, which is crucial for correctly associating entities across frames.
3. **Lightweight Design Philosophy**: A graph-to-text model with only 235M parameters plus a CPU-executable graph consolidation algorithm truly achieves "more with less" compared to heavy 7B+ LLM schemes.
4. **Text-Only Training Requirement**: The training of the graph-to-text model relies solely on graph-caption pairs, completely removing the need for video-text paired data, which greatly expands the scale of usable training data.

## Limitations & Future Work

1. **Dependency on Textual Scene Graph Parser Quality**: The parsing accuracy of the FACTUAL-MR parser directly affects downstream performance. Improper parsing of entities and relations in captions degrades the consolidated graph quality.
2. **Graph Consolidation within Text Space**: Object matching relies on the semantic representations of a pre-trained graph encoder, potentially ignoring visual-level similarities (e.g., matching might fail if the same person wears different clothes across frames).
3. **Generation Complexity Limitations of Graph-to-Text Models**: The 235M parameter model might struggle to generate longer and more complex paragraph descriptions.
4. **Neglect of Temporal Information**: The scene graph consolidation algorithm selects pairs for merging based on similarity rather than chronological order, which may result in a loss of timeline information.
5. **CPU-Bound Graph Consolidation**: Although currently fast, a GPU implementation could achieve further speedup.

## Related Work & Insights

- **Novel Application of Scene Graphs in Video Captioning**: While scene graphs have been primarily used for visual relationship detection and visual question answering, this paper leverages them as an "intermediate representation" for cross-segment information fusion, presenting a novel and effective application.
- **Complementarity with Memory-based Methods**: The graph consolidation strategy proposed in this paper can serve as a plug-and-play alternative to memory-based methods for any scenario requiring the aggregation of multi-segment video information.
- **Inspiration**: This "structure first, consolidate, then generate" pipeline can be extended to other multi-document or multimodal information fusion tasks, such as multi-document summarization and multi-view scene understanding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[ICLR 2026\] CaReBench: A Fine-grained Benchmark for Video Captioning and Retrieval](../../ICLR2026/video_understanding/carebench_a_fine-grained_benchmark_for_video_captioning_and_retrieval.md)
- [\[CVPR 2025\] HyperGLM: HyperGraph for Video Scene Graph Generation and Anticipation](../../CVPR2025/video_understanding/hyperglm_hypergraph_for_video_scene_graph_generation_and_anticipation.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](../../CVPR2026/video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[ECCV 2024\] FinePseudo: Improving Pseudo-Labelling through Temporal-Alignability for Semi-Supervised Fine-Grained Action Recognition](../../ECCV2024/video_understanding/finepseudo_improving_pseudo-labelling_through_temporal-alignablity_for_semi-supe.md)

</div>

<!-- RELATED:END -->
