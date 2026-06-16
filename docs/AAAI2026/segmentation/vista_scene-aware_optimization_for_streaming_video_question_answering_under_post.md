---
title: >-
  [Paper Note] Vista: Scene-Aware Optimization for Streaming Video Question Answering Under Post-Hoc Queries
description: >-
  [AAAI 2026][Segmentation][Streaming video question answering] Vista proposes a scene-aware streaming video question answering framework that dynamically segments streaming video into semantically coherent scene units…
tags:
  - "AAAI 2026"
  - "Segmentation"
  - "Streaming video question answering"
  - "scene-aware compression"
  - "multimodal large language models"
  - "video memory retrieval"
  - "real-time inference"
date: 2026-05-08
content_hash: 13d76560e5bd4f35
---

# Vista: Scene-Aware Optimization for Streaming Video Question Answering Under Post-Hoc Queries

**Conference**: AAAI 2026
**arXiv**: [2602.08448](https://arxiv.org/abs/2602.08448)  
**Code**: None  
**Area**: Image Segmentation
**Keywords**: Streaming video question answering, scene-aware compression, multimodal large language models, video memory retrieval, real-time inference

## TL;DR

Vista proposes a scene-aware streaming video question answering framework that dynamically segments streaming video into semantically coherent scene units, applies spatiotemporal compression to each scene and offloads it to CPU memory, and selectively recalls the most relevant scenes upon user queries, achieving high-accuracy video QA under strict GPU memory and latency constraints.

## Background & Motivation

**Background**: Multimodal large language models (MLLMs) have achieved significant progress in video question answering, but most methods target offline settings where the full video and query are simultaneously available, enabling global analysis. In interactive real-time applications such as autonomous driving and video dialogue systems, frames arrive continuously and users may query at any moment, making traditional offline approaches inapplicable.

**Limitations of Prior Work**: Streaming video QA faces two core challenges. First, video streams are theoretically unbounded, and fixed-rate frame sampling rapidly exhausts memory and compute resources. Second, users demand low-latency responses, precluding full-sequence attention computation at inference time. Existing streaming methods such as Flash-VStream and VideoLLM-Online either employ fixed-size memory buffers that cause context loss, or apply simplistic compression strategies that incur information loss, resulting in performance far below offline models.

**Key Challenge**: Under the *post-hoc queries* setting, user queries arrive at arbitrary moments after the video stream has begun, so the model cannot leverage query semantics to select keyframes during encoding. The model must therefore retain sufficient visual information for unknown future queries while simultaneously controlling memory usage and response latency.

**Goal**: To achieve efficient video encoding, compressed storage, and query-driven information retrieval in the post-hoc streaming video QA setting, maintaining high accuracy under strict memory and latency constraints.

**Key Insight**: The authors observe that during inference, model attention typically concentrates on a small number of semantically salient segments, which temporally tend to belong to the same "scene." Videos can therefore be compressed and retrieved at scene granularity rather than frame by frame.

**Core Idea**: Dynamically segment streaming video at scene granularity, apply compact compression, and perform on-demand recall, enabling efficient real-time video QA without sacrificing semantic integrity.

## Method

### Overall Architecture

Vista operates in three stages: (1) **Scene-aware Segmentation**: as video frames arrive sequentially, the system automatically partitions them into semantically coherent scene units based on inter-frame visual similarity; (2) **Scene-aware Compression**: each completed scene is reduced to a compact scene token via spatiotemporal compression, while the original high-resolution frames are offloaded to CPU memory; (3) **Scene-aware Recall**: upon a user query, the system computes relevance scores between the query and each scene token, retrieves the full frames of the top-$k$ most relevant scenes, and feeds them together with the current sliding-window frames into the vision-language model to generate an answer.

### Key Designs

1. **Scene-aware Segmentation**:

    - **Function**: Online partitioning of continuously arriving video frames into temporally and semantically coherent scene units.
    - **Mechanism**: An anchor frame $F_a$ representing the start of the current scene is maintained. For each newly arrived frame $F_i$, two similarity scores are computed: the similarity to the anchor frame $\mathcal{S}_{\text{anchor}}(F_i)$ and the similarity to the previous frame $\mathcal{S}_{\text{adj}}(F_i)$. A scene boundary is declared only when both scores simultaneously fall below threshold $\tau$. This dual-condition criterion distinguishes gradual transitions from abrupt cuts, avoiding false detections. Temporal overlap is introduced between adjacent scenes by sharing a small number of frames, mitigating abrupt boundary effects and preserving temporal continuity.
    - **Design Motivation**: Since queries are unknown during the encoding phase, semantic relevance cannot guide keyframe selection. Unsupervised visual similarity is therefore used for online scene boundary detection, ensuring query-agnostic generality.

2. **Scene-aware Compression**:

    - **Function**: Compress each completed scene into a single compact scene token stored on GPU for subsequent retrieval; offload original frames to CPU memory.
    - **Mechanism**: A three-step Temporal-Spatial Compression strategy is proposed. **Temporal compression**: all frames within a scene are averaged patch-wise along the temporal axis, exploiting the high inter-frame correlation within a scene to remove temporal redundancy, yielding a temporally compressed feature map $F_{\text{temp}}$. **Spatial compression**: $F_{\text{temp}}$ is reshaped into a 2D spatial token grid, and sliding-window aggregation is applied with each patch's L2 norm as an importance weight, highlighting salient regions. **Final aggregation**: the spatially weighted tokens are further average-pooled to produce the final scene token. The entire process involves no learnable parameters and relies solely on pooling and weighting operations.
    - **Design Motivation**: Intra-scene inter-frame redundancy is extremely high; storing all frames would cause GPU memory explosion. Hierarchical spatiotemporal compression drastically reduces storage overhead (one token per scene) while preserving discriminative spatial information via L2 weighting.

3. **Scene-aware Recall**:

    - **Function**: At query time, retrieve the most relevant scenes from the compressed scene bank and restore their full frames for answer generation.
    - **Mechanism**: The query $Q$ is embedded via a language encoder into a query vector $\mathbf{q} = \psi(Q)$, which computes dot-product attention scores $\alpha_i = \mathbf{q} T_i^\top$ against each scene token $T_i$. The top-$k$ highest-scoring scenes are selected, their high-resolution frames are fetched from CPU memory, and together with the current sliding-window frames and the query they form the final model input: $\text{Input}_{\text{VLM}} = (\mathcal{V}_{\text{final}}, Q)$, where $\mathcal{V}_{\text{final}} = (\bigcup_{j \in \mathcal{I}_k} \mathcal{F}_j) \cup \mathcal{L}$.
    - **Design Motivation**: Although compression must be query-agnostic, query information can be exploited at the answer generation stage for targeted retrieval. Restoring full frames for only a small number of the most relevant scenes, rather than all historical frames, ensures answer quality while controlling GPU memory usage.

### Loss & Training

Vista is a **training-free** framework with no additional loss functions or training procedures. It serves as a plug-and-play module that can be integrated with various vision-language models (e.g., LLaVA-OneVision-7B, Video-LLaMA2-7B) and operates purely at inference time. All inference experiments use greedy decoding (temperature=0) to ensure deterministic generation.

## Key Experimental Results

### Main Results

Key results on the StreamingBench benchmark:

| Model | RT | ER | SCU | SD | MA | ACU | MCU | SQA | Overall |
|------|-----|-----|------|-----|-----|------|------|------|---------|
| Flash-VStream | 23.23 | 25.91 | 24.90 | 25.60 | 28.40 | 24.80 | 25.20 | 26.80 | - |
| Dispider | 67.63 | 35.46 | 25.26 | 38.57 | 43.34 | 39.62 | 27.65 | 34.80 | - |
| LLaVA-OV-7B | 70.92 | 40.00 | 24.80 | 31.20 | 44.40 | 32.40 | 35.60 | 30.80 | - |
| **+Vista** | **71.36** | **46.40** | **37.20** | **43.60** | **74.00** | **43.20** | **36.80** | **34.40** | - |

Offline video QA benchmarks:

| Dataset | Metric | Ours (Vista) | Prev. SOTA | Gain |
|--------|------|-------------|----------|------|
| MLVU | Accuracy | 63.8% | - | Significantly outperforms Dispider et al. |
| EgoSchema | Accuracy | 58.7% | - | Surpasses all streaming and most offline models |

### Ablation Study

| Configuration | ER Accuracy | Note |
|------|------------|------|
| Base (uniform sampling) | 40.00% | No modules |
| +Compression+Recall | 38.80% | Compression without segmentation is harmful |
| +Segmentation+Recall | 42.00% | Semantic segmentation provides structural prior |
| +Segmentation+Compression | 44.00% | Intra-scene compression effectively preserves information |
| **All three modules** | **46.40%** | Complementary combination achieves optimum |

### Key Findings

- The improvement on the **Multi-modal Alignment (MA)** task is most striking: Vista achieves 74.00%, outperforming the LLaVA-OV-7B baseline (44.40%) by 29.6 percentage points and even surpassing GPT-4o (56.00%).
- As frame count grows continuously, Vista's GPU memory usage and inference latency remain stable, demonstrating strong scalability.
- Applying compression without scene segmentation degrades performance (38.80% < 40.00%), confirming that semantically coherent segmentation is a prerequisite for effective compression.
- Hyperparameter analysis shows robust performance across different settings, with optimal results at segmentation threshold $\tau=0.8$, spatial window $a=2$, and scene capacity–recall pair $m=8, k=3$.

## Highlights & Insights

- **Training-free plug-and-play design**: As a model-agnostic inference-time framework, Vista can directly enhance the streaming video processing capability of any vision-language model, offering strong practical utility.
- **The scene-granularity intuition is highly natural**: Human video comprehension also operates at the scene level rather than frame by frame. Performing compression and retrieval at scene granularity aligns with the natural organization of video information.
- **GPU–CPU cooperative memory management** is elegantly designed: compressed tokens reside on GPU for fast indexing, while full frames are stored on CPU for on-demand restoration, analogous to hierarchical virtual memory management in operating systems.
- The large gains on the MA task suggest that scene-level retrieval is particularly well-suited for tasks requiring cross-modal alignment.

## Limitations & Future Work

- In high-dynamic scenes with rapid motion or abrupt changes, scene boundary detection is unreliable and degrades to single-frame recall.
- Excessively long static scenes require forced truncation to prevent memory overflow, potentially causing information loss.
- Scene tokens are single-vector representations, which may have limited expressiveness for scenes containing complex multi-event content.
- Validation is currently limited to benchmarks such as StreamingBench; end-to-end performance evaluation in real-time deployment scenarios is lacking.
- The compression strategy relies entirely on pooling without learnable compression modules, which may impose an upper bound on information retention.

## Related Work & Insights

- **Long-video QA** (LLaMA-VID, Chat-UniVi): address long videos via token merging and sparse sampling, but assume simultaneous availability of video and query, making them inapplicable to streaming settings.
- **Streaming video QA** (Flash-VStream, Dispider, ReKV, StreamMem): perform compression/retrieval at frame or KV cache granularity, but are sensitive to temporal noise and achieve limited retrieval effectiveness.
- Vista's scene-aware approach is complementary to KV cache compression methods and can be combined for multi-level memory management.
- The GPU–CPU offloading strategy is generalizable to broader long-sequence inference scenarios, such as long document understanding and multi-turn dialogue.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of scene-aware compression and recall is intuitively motivated and systematically designed, though the core techniques (pooling, top-$k$ retrieval) are relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-benchmark evaluation, ablation study, hyperparameter analysis, and visualization are provided, but validation across more backbone models is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is described clearly with intuitive diagrams and rigorous problem formulation.
- **Value**: ⭐⭐⭐⭐ The training-free plug-and-play design offers high practical value and provides a clean, effective baseline for streaming video QA.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SAM-DAQ: Segment Anything Model with Depth-guided Adaptive Queries for RGB-D Video Salient Object Detection](sam-daq_segment_anything_model_with_depth-guided_adaptive_queries_for_rgb-d_vide.md)
- [\[NeurIPS 2025\] TabRAG: Improving Tabular Document Question Answering for Retrieval Augmented Generation via Structured Representations](../../NeurIPS2025/segmentation/tabrag_improving_tabular_document_question_answering_for_retrieval_augmented_gen.md)
- [\[ICML 2026\] Beyond Detection: A Structure-Aware Framework for Scene Text Tracking](../../ICML2026/segmentation/beyond_detection_a_structure-aware_framework_for_scene_text_tracking.md)
- [\[CVPR 2026\] DSFlash: Comprehensive Panoptic Scene Graph Generation in Realtime](../../CVPR2026/segmentation/dsflash_panoptic_scene_graph_realtime.md)
- [\[AAAI 2026\] Breaking the Stealth-Potency Trade-off in Clean-Image Backdoors with Generative Trigger Optimization](breaking_the_stealth-potency_trade-off_in_clean-image_backdoors_with_generative_.md)

</div>

<!-- RELATED:END -->
