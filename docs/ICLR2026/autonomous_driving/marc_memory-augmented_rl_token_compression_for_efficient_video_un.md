---
title: >-
  [Paper Note] MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding
description: >-
  [ICLR 2026][Autonomous Driving][Video token compression] MARC is a framework that adopts a "retrieve-then-compress" strategy: a Visual Memory Retriever (VMR) selects the most query-relevant video segments, and Compression GRPO (C-GRPO) distills the reasoning capability of a 64-frame teacher model into a student model that operates on only 1-frame tokens. This achieves 95% visual token compression, 72% GPU memory reduction, 23.9% inference latency reduction, with virtually no performance loss (42.20 vs. 42.21).
tags:
  - ICLR 2026
  - Autonomous Driving
  - Video token compression
  - reinforcement learning distillation
  - visual memory retrieval
  - GRPO
  - efficient inference
date: 2026-05-08
content_hash: 4a000da9ceec38a1
---

# MARC: Memory-Augmented RL Token Compression for Efficient Video Understanding

**Conference**: ICLR 2026
**arXiv**: [2510.07915](https://arxiv.org/abs/2510.07915)
**Code**: Available (Project Web / Code / Model all provided)
**Area**: Autonomous Driving
**Keywords**: Video token compression, reinforcement learning distillation, visual memory retrieval, GRPO, efficient inference

## TL;DR

MARC is a framework that adopts a "retrieve-then-compress" strategy: a Visual Memory Retriever (VMR) selects the most query-relevant video segments, and Compression GRPO (C-GRPO) distills the reasoning capability of a 64-frame teacher model into a student model that operates on only 1-frame tokens. This achieves 95% visual token compression, 72% GPU memory reduction, 23.9% inference latency reduction, with virtually no performance loss (42.20 vs. 42.21).

## Background & Motivation

**Computational bottleneck in video understanding**: Scaling VLMs from images to video causes an explosion in token count due to high frame rates and long durations, dramatically increasing inference cost and severely limiting deployment in latency-sensitive scenarios such as autonomous driving and surveillance.

**Limitations of existing token compression methods**: Mainstream compression methods (e.g., MovieChat, VidCom, ByteVideoLLM) rely primarily on training-free token merging strategies that handle spatial or temporal redundancy independently, inevitably losing critical information during compression and suffering significant performance degradation.

**Independent treatment of spatiotemporal redundancy**: Existing methods overlook the temporally organized, context-aware nature of human visual memory. Cognitive science research indicates that humans segment continuous experience into discrete events and retrieve them via episodic memory.

**Difficulty of maintaining performance under extreme compression**: Reducing video to the token count equivalent of a single frame makes it difficult for naive geometric token-reduction heuristics to preserve teacher-level reasoning quality.

**Lack of learning-based compression solutions**: Most existing approaches are training-free inference-time techniques, lacking end-to-end solutions that optimize compression quality through learning.

**Disconnect between retrieval and compression**: Video retrieval-augmented generation (Video-RAG) and token compression have typically been pursued as separate technical directions. This paper is the first to tightly integrate structured retrieval with RL-based compression.

## Method

### Overall Architecture

MARC is a **"retrieve-then-compress"** framework consisting of two core modules:

- **Visual Memory Retriever (VMR)**: Segments video into event-level clips and retrieves the top-k clips most relevant to the query.
- **C-GRPO training strategy**: Uses a 64-frame teacher network as reference, and distills its reasoning capability into a student network operating on only 1-frame tokens via reinforcement learning.

The overall pipeline: raw video → event segmentation → top-k clip retrieval → Memory-Aware Temporal Compression → compressed tokens fed into LLM → C-GRPO alignment training.

### Key Designs

#### 1. Visual Memory Retriever (VMR)

**Function**: Retrieves event-level clips most relevant to the query from long videos, serving as input to downstream compression.

**Design Motivation**: Inspired by cognitive science—humans segment continuous visual experience into discrete events via episodic memory and retrieve them contextually. Compressing the entire video directly introduces substantial redundancy and degrades compression quality; retrieving first then compressing significantly narrows the search space.

**Mechanism**:
- **Event-level video segmentation**: A deep event detection network (Soucek & Lokoc, 2024) identifies temporal boundaries such as scene cuts and topic transitions, segmenting the video into semantically coherent short clips (rather than fixed-length windows).
- **Memory retrieval**: An embedding model (Bolya et al., 2025) maps the query and all clips into a shared high-dimensional latent space; nearest-neighbor search trained via contrastive learning selects the top-k most relevant clips.
- In experiments, top-k = 3.

#### 2. Memory-Aware Temporal Compression Layer

**Function**: Applies two-stage temporal compression to VMR-selected clips to reduce the number of visual tokens.

**Design Motivation**: Leverages the event boundary structure provided by VMR to preferentially merge highly similar adjacent frames within the same event (where redundancy is greatest), while preserving event evidence deemed important by VMR.

**Mechanism**:
- **Stage 1 (intra-segment merging)**: For each retrieved clip, within a short-term memory window $m$, iteratively merges the adjacent frame pair with the highest cosine similarity, representing them by their mean $\mathbf{H}_{merge} = \frac{1}{2}(\mathbf{H}_a + \mathbf{H}_b)$, until the frame budget corresponding to compression ratio $\rho$ is satisfied.
- **Stage 2 (cross-segment merging)**: If the total frame count after intra-segment merging still exceeds the target $N_{target}$, a lightweight global merging step is applied.
- Similarity is measured as the mean patch-aligned cosine score: $\text{sim}(\mathbf{H}_a, \mathbf{H}_b) = \frac{1}{P}\sum_{p=1}^{P} \frac{\mathbf{h}_a^{(p)} \cdot \mathbf{h}_b^{(p)}}{\|\mathbf{h}_a^{(p)}\| \|\mathbf{h}_b^{(p)}\|}$

#### 3. Compression GRPO (C-GRPO)

**Function**: Trains the student model via reinforcement learning in a teacher-student distillation paradigm to preserve teacher-level reasoning under extreme compression.

**Design Motivation**: Standard GRPO focuses solely on answer correctness and format, without explicitly coupling student and teacher performance. C-GRPO introduces a retention alignment reward, reframing compression as an alignment problem rather than geometric reduction.

**Mechanism**:
- Defines a **retention ratio** $\eta = a_{comp} / a_{full}$ to quantify how much of the teacher's performance the student retains.
- Introduces a **compression reward** $r_c = \alpha \cdot \max(0, \eta - \tau)$, where $\tau$ is the minimum acceptable retention threshold.
- **Correctness gating**: $R_i = r_i + \mathbb{1}[\text{correct}] \cdot r_c$—only semantically correct generations receive the retention reward, preventing reward hacking.
- In-group advantage normalization: $A_i = (R_i - \bar{R}) / \sigma_R$.
- Final optimization uses a clipped objective with KL anchoring.

### Loss & Training

$$\mathcal{L}_{\text{C-GRPO}} = \mathbb{E}\left[\frac{1}{G}\sum_{i=1}^{G}\left(\text{clip}\left(\frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}, 1-\epsilon, 1+\epsilon\right) A_i\right) - \beta \text{KL}(\pi_\theta \| \pi_{ref})\right]$$

- **Teacher network**: Qwen2.5-VL-3B with 64-frame input.
- **Student network**: Same architecture, compressed to 1-frame tokens (~122 tokens).
- **Training data**: Only 5K samples randomly drawn from Video-R1-260K (including video and image data).
- **Group size** $G=8$, threshold $\tau=0.6$.
- Image data does not participate in compression reward computation but assists in building general reasoning capability for static scenes.

## Key Experimental Results

### Main Results

| Model | Frames | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME | Mean |
|-------|--------|-----------|-----------|------|---------|-------------|----------|------|
| Qwen2.5-VL-3B (baseline) | 64 | 32.93 | 35.33 | 48.64 | 44.77 | 38.05 | 53.55 | 42.21 |
| Qwen2.5-VL-3B | 16 | 27.63 | 30.78 | 45.28 | 43.89 | 37.95 | 44.37 | 38.32 |
| InternVL3.5-4B | 64 | 28.96 | 33.33 | 47.51 | 44.71 | 58.34 | 39.15 | 42.00 |
| Gemma-3-4B | 64 | 26.83 | 26.78 | 41.76 | 36.82 | 55.04 | 46.00 | 38.87 |
| ByteVideoLLM-3B | 64 | 21.33 | 22.33 | 28.63 | 22.56 | 35.55 | 22.70 | 25.52 |
| MovieChat-3B | 1 | 25.14 | 25.78 | 39.35 | 37.10 | 38.79 | 26.41 | 32.10 |
| VidCom2-3B | 64 | 25.50 | 23.89 | 31.08 | 29.88 | 35.23 | 21.48 | 27.84 |
| **MARC-3B** | **1** | **27.55** | **33.11** | **51.99** | **45.82** | **55.34** | **39.44** | **42.20** |

**Key result**: MARC-3B uses only 4.71% of visual tokens (122.69 vs. original 2589.93), with a mean score of 42.20—virtually identical to the 64-frame baseline of 42.21.

### Ablation Study

**Ablation on threshold $\tau$**:

| $\tau$ | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME | Mean |
|--------|-----------|-----------|------|---------|-------------|----------|------|
| 0.4 | 28.27 | 31.66 | 49.12 | 45.21 | 54.72 | 39.07 | 41.34 |
| **0.6** | **27.55** | **33.11** | **51.99** | **45.82** | **55.34** | **39.44** | **42.20** |
| 0.8 | 28.23 | 31.78 | 49.34 | 45.89 | 54.12 | 39.03 | 41.40 |

**Ablation on VMR and training strategy**:

| Method | Frames | Mean |
|--------|--------|------|
| Baseline (w/o VMR) | 64 | 42.21 |
| Baseline + VMR | 64 | 45.56 |
| SFT | 1 | 38.50 |
| SFT + VMR | 1 | 40.16 |
| **MARC (C-GRPO + VMR)** | **1** | **42.20** |

### Key Findings

1. **Near-lossless performance under extreme compression**: 95% token compression (64 frames → 1-frame tokens) yields a mean score of 42.20 vs. 42.21.
2. **Significant efficiency gains**: GPU memory reduced by 72.4% (41.63 GB → 11.48 GB), LLM generation latency reduced by 23.9%, end-to-end latency reduced by 11.1%.
3. **VMR alone improves performance**: Without compression, VMR improves the baseline from 42.21 to 45.56 (+7.9%), with up to 27.85% improvement on MVBench.
4. **C-GRPO substantially outperforms SFT**: MARC achieves a mean of 42.20 vs. 38.50 for SFT (+9.6%).
5. **Exceeds baseline on several benchmarks**: MARC surpasses the 64-frame baseline on MMVU, MVBench, and TempCompass.
6. **Outperforms larger models**: MARC-3B exceeds InternVL3.5-4B (42.20 vs. 42.00) and Gemma-3-4B (42.20 vs. 38.87).
7. **$\tau=0.6$ is optimal**: $\tau=0.4$ imposes too weak a constraint and leads to insufficient retention; $\tau=0.8$ yields sparse reward signals that limit learning.

## Highlights & Insights

- **Cognitive science-inspired retrieval design**: VMR's event-level segmentation simulates the encoding and retrieval mechanisms of human episodic memory, better reflecting the natural structure of video content than fixed windows or uniform sampling.
- **Reframing compression as an alignment problem**: The core insight of C-GRPO is to redefine token compression from a geometric/heuristic operation to a teacher-student alignment problem, leveraging RL reward shaping to guide the compression direction.
- **Elegant correctness gating**: Only correct generations receive the compression retention reward, preventing reward hacking and amplification of spurious patterns.
- **Only 5K training samples**: The minimal training data (5K sampled from 260K) demonstrates the high data efficiency of C-GRPO.
- **Retrieve-then-compress paradigm**: The pipeline design ensures that the compression module does not blindly compress the entire video, but instead performs meaningful compression on pre-selected key segments.

## Limitations & Future Work

1. **Performance degradation on long videos**: On VideoMME, MARC retains only 74% of baseline performance (39.44 vs. 53.55), indicating that extreme compression still carries a meaningful cost for long-video understanding.
2. **Validated only on 3B models**: All training experiments are based on Qwen2.5-VL-3B; the generalizability of MARC to 7B+ models has not been verified.
3. **VMR depends on event segmentation quality**: If the event detection module misidentifies boundaries or query semantic matching is poor, downstream compression quality will suffer.
4. **Fixed top-k=3**: The number of retrieved clips is fixed; adaptive selection strategies have not been explored.
5. **Fixed compression ratio**: $\rho$ is not adaptively adjusted based on video complexity, which may be suboptimal for diverse video types.
6. **Separate training of components**: VMR and C-GRPO are trained separately; fully end-to-end joint optimization has not been achieved.

## Related Work & Insights

- **Video-RAG direction**: The VMR module is essentially a video corpus retrieval scheme and can be integrated with agent-based systems (e.g., VideoAgent).
- **Scalability of GRPO**: The compression reward design in C-GRPO can be generalized to token compression in other modalities (e.g., 3D point clouds, long documents).
- **Token merging methods**: MovieChat's short-term memory merging is the direct predecessor of MARC's temporal compression layer; MARC significantly improves merging quality through the event structure provided by VMR.
- **New paradigm for knowledge distillation**: Traditional KD aligns logits/features via KL divergence; C-GRPO opens a new path for "behavior-level" distillation using RL reward signals.

## Rating

| Dimension | Score | Notes |
|-----------|-------|-------|
| Novelty | ⭐⭐⭐⭐ | First application of RL (GRPO) to video token compression; retrieve-then-compress combination is novel |
| Experimental Thoroughness | ⭐⭐⭐⭐ | 6 benchmarks, extensive comparisons/ablations, complete efficiency evaluation; lacks large-model validation |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure, complete mathematical derivations, well-motivated |
| Value | ⭐⭐⭐⭐⭐ | 95% compression with near-lossless performance; extremely high practical deployment value |

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] StreamForest: Efficient Online Video Understanding with Persistent Event Memory](../../NeurIPS2025/autonomous_driving/streamforest_efficient_online_video_understanding_with_persistent_event_memory.md)
- [\[ICLR 2026\] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video](egodex_learning_dexterous_manipulation_from_large-scale_egocentric_video.md)
- [\[ICLR 2026\] DrivingGen: A Comprehensive Benchmark for Generative Video World Models in Autonomous Driving](drivinggen_a_comprehensive_benchmark_for_generative_video_world_models_in_autono.md)
- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](../../AAAI2026/autonomous_driving/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[AAAI 2026\] FastDriveVLA: Efficient End-to-End Driving via Plug-and-Play Reconstruction-based Token Pruning](../../AAAI2026/autonomous_driving/fastdrivevla_efficient_end-to-end_driving_via_plug-and-play_.md)

<!-- RELATED:END -->
