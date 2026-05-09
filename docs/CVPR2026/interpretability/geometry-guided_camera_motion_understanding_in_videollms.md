---
title: >-
  [Paper Note] Geometry-Guided Camera Motion Understanding in VideoLLMs
description: >-
  [CVPR 2026][VideoLLM] This paper reveals that VideoLLMs perform near random-chance on fine-grained camera motion primitives (pan/tilt/dolly, etc.), constructs CameraMotionDataset (12K clips × 15 atomic motions) and the CameraMotionVQA benchmark, and proposes a model-agnostic approach that injects geometric camera cues extracted by a frozen 3D foundation model (VGGT) via a lightweight temporal classifier and structured prompting — bridging this capability gap without any fine-tuning of the VideoLLM.
tags:
  - CVPR 2026
  - VideoLLM
  - camera motion recognition
  - 3D foundation model
  - structured prompting
  - VGGT
date: 2026-05-08
content_hash: 201bc386fa897744
---

# Geometry-Guided Camera Motion Understanding in VideoLLMs

**Conference**: CVPR 2026
**arXiv**: [2603.13119](https://arxiv.org/abs/2603.13119)
**Code**: To be confirmed
**Area**: Interpretability
**Keywords**: VideoLLM, camera motion recognition, 3D foundation model, structured prompting, VGGT

## TL;DR
This paper reveals that VideoLLMs perform near random-chance on fine-grained camera motion primitives (pan/tilt/dolly, etc.), constructs CameraMotionDataset (12K clips × 15 atomic motions) and the CameraMotionVQA benchmark, and proposes a model-agnostic approach that injects geometric camera cues extracted by a frozen 3D foundation model (VGGT) via a lightweight temporal classifier and structured prompting — bridging this capability gap without any fine-tuning of the VideoLLM.

## Background & Motivation
**Background**: VideoLLMs (Qwen2.5-VL, InternVL, VideoLLaMA, etc.) have made substantial progress on high-level video semantic understanding — object recognition, action understanding, narrative reasoning, and so forth. However, these models are primarily optimized for semantic alignment and temporal reasoning, leaving the cinematographic grammar of *how a shot is captured* largely unmodeled.

**Limitations of Prior Work**:
- **Camera motion is a spatiotemporal geometric signal**: it cannot be inferred from a single frame and is easily confounded by object motion, cuts, and motion blur. Models with strong frame-level perception still fail to model the camera as the *origin* of the visual stream.
- **Deep ViT token compression discards motion cues**: visual tokens in the VideoLLM pipeline are progressively compressed with network depth, attenuating subtle temporal motion cues.
- **Training data lacks camera motion supervision**: large-scale video captioning/VQA corpora contain almost no explicit camera motion annotations.

**Key Challenge**: The representation space of VideoLLMs is optimized for semantic alignment rather than precise 3D geometric change, causing camera motion information to be "squeezed out" of the representation.

**Goal**: (a) construct a reliable camera motion evaluation benchmark; (b) diagnose where motion cues are lost within VideoLLMs; (c) inject geometric camera cues without modifying VideoLLM weights.

**Key Insight**: The core hypothesis is that reliable camera motion cues can be extracted from a geometry-aware 3D foundation model (3DFM) and injected into VideoLLMs in a plug-and-play manner. Synthetic data (UE5 rendering with precise camera extrinsics) provides deterministic annotations.

**Core Idea**: A frozen 3DFM extracts camera geometry cues; a lightweight classifier predicts constraint-aware motion primitives; results are injected into a frozen VideoLLM via structured prompting — improving camera motion perception with zero fine-tuning.

## Method

### Overall Architecture
The pipeline consists of four steps (see Figure 1): (1) the input video is segmented into shots, each shot divided into non-overlapping 1-second clips; (2) the frozen VGGT extracts camera tokens $\{c_t\}_{t=1}^T$ from $T=8$ frames per clip, $c_t \in \mathbb{R}^{2048}$; (3) a lightweight Transformer temporal classifier predicts constraint-aware multi-label motion primitives; (4) the predicted sequence is serialized into a structured prompt prefix and prepended to the downstream frozen VideoLLM at inference. The entire process is model-agnostic and leaves all VideoLLM parameters untouched.

### Key Designs

1. **CameraMotionDataset Construction**

    - **Function**: Constructs a precisely annotated camera motion dataset from ReCamMaster's MultiCamVideo (UE5-rendered, 136K videos, 112K camera trajectories).
    - **Mechanism**: Each video is segmented into non-overlapping 1-second clips; each clip uniformly samples $T=8$ frames resized to $336 \times 336$. Per-frame camera extrinsic matrices are used to compute per-clip translation and rotation changes (yaw/pitch/roll deltas and forward/backward translation), which are mapped via threshold-based pattern matching to 15 atomic motion primitives (pan-left, tilt-down, dolly-in, etc.). Multiple primitives may co-occur (e.g., arc-clockwise + dolly-in), but mutually exclusive pairs are disallowed. Stratified sampling yields a balanced subset of 12,274 clips. Manual validation on 720 clips achieves 93% inter-annotator agreement.
    - **Design Motivation**: Unlike manually annotated benchmarks such as CameraBench, labels in this dataset are deterministically derived from precise camera parameters, yielding higher annotation quality and scalability.
    - **CameraMotionVQA**: Each 1-second clip is converted into a 4-choice MCQ; distractors are selected to match the label complexity of the correct answer and satisfy mutual exclusivity constraints, avoiding answer-length bias.

2. **Constraint-Aware Motion Classifier**

    - **Function**: Maps VGGT camera tokens to constraint-compliant multi-label motion predictions.
    - **Mechanism**: Each camera token $c_t$ is first projected via a linear layer $W_p$ to $c_t' \in \mathbb{R}^{512}$ (information bottleneck), sinusoidal positional encodings are added, a learnable [CLS] token is prepended, and the sequence is processed by an $L=4$-layer Transformer encoder (8-head attention). The final [CLS] embedding is projected to $K=15$ logits $s$, with $p_k = \text{sigmoid}(s_k)$.
    - The training loss consists of three terms:
    $$\mathcal{L} = \mathcal{L}_{bce} + \lambda_{inc} \cdot \mathcal{L}_{inc} + \lambda_{card} \cdot \mathcal{L}_{card}$$
        - $\mathcal{L}_{bce}$: standard binary cross-entropy
        - $\mathcal{L}_{inc} = \sum_{i<j} M_{ij} \cdot p_i \cdot p_j$: incompatibility regularization, where $M \in \{0,1\}^{K \times K}$ is the mutual exclusivity matrix penalizing simultaneous activation of incompatible primitives
        - $\mathcal{L}_{card}$: cardinality regularization constraining the number of activated primitives to $[1, 3]$
    - At inference, predictions are thresholded at $\tau=0.5$ and post-processed using the incompatibility matrix to eliminate conflicting combinations.

3. **Structured Prompting Injection**

    - **Function**: Injects classifier-predicted motion primitives as structured text into the frozen VideoLLM's prompt.
    - **Mechanism**: For a shot with $S$ one-second clips, each clip's motion labels are serialized into a string (e.g., "pan-left and tilt-up") and concatenated into a per-shot list: "Per-second camera motion: [$m_1, m_2, \ldots, m_S$]", prepended to the user instruction. The prompt template guides the model to describe the video in cinematic language with emphasis on camera usage.
    - **Design Motivation**: Entirely training-free and plug-and-play; no VideoLLM weights are modified. The approach leverages the LLM's in-context learning capability to inject geometric priors into inference at no cost.

4. **Q-Former Probing Diagnosis**

    - **Function**: Diagnoses at which depth camera motion information is lost within the VideoLLM visual encoder.
    - **Mechanism**: The Qwen2.5-VL visual encoder is frozen; intermediate features are extracted at full-attention blocks of varying depth (indices 7, 15, 23, 31), and a Q-Former-style probe (2-layer Transformer + 4 learnable query tokens + 1D temporal conv) is trained for multi-label prediction.
    - **Key Findings**: Performance peaks at the first full-attention block and degrades monotonically with depth, confirming that camera motion cues are progressively eroded by the semantic alignment objective.

5. **VGGT–Q-Former Distillation** (optional efficiency optimization)

    - **Function**: Distills the 1.2B-parameter VGGT into a lightweight Q-Former student that reuses frozen VideoLLM visual features.
    - **Mechanism**: The student adopts interleaved local-frame/global attention (mimicking the VGGT architecture), with 4 learnable queries and 2 local + 2 global blocks. Three-stage progressive training: (1) train the motion classifier for 50 epochs; (2) train the Q-Former to regress projected VGGT tokens for 100 epochs (MSE loss); (3) joint fine-tuning for 30 epochs.
    - **Outcome**: Instance accuracy drops by 8.13%, but throughput improves by 5.3× and GPU memory usage decreases to 39% of the original.

### Loss & Training
Classifier: $\mathcal{L}_{bce} + \mathcal{L}_{inc} + \mathcal{L}_{card}$, with $\lambda_{inc} = \lambda_{card} = 1.0$. Distillation uses an MSE regression loss. All experiments run on a single RTX A6000 with Adam optimizer at lr=1e-4.

## Key Experimental Results

### Main Results: Multi-label Camera Motion Recognition (CameraMotionDataset test split)

| Method | Instance Acc. | Macro-F1 | Weighted-F1 |
|------|:---:|:---:|:---:|
| VGGT w. constraints | **0.738** | **0.87** | **0.92** |
| VGGT w/o constraints | 0.572 | 0.79 | 0.84 |
| VGGT–Q-Former (distilled) | 0.638 | 0.83 | 0.87 |
| Q-Former probing | 0.450 | 0.69 | 0.74 |

### Efficiency Comparison

| Pipeline | Trainable Params (M) | Peak Memory (MB) | Throughput (samples/s) |
|---------|:---:|:---:|:---:|
| VGGT classifier | 9.47 | 23649 | 4.39 |
| VGGT–Q-Former | 9.15 | 9203 | 23.36 |
| Q-Former probing | 15.18 | 9232 | 25.12 |

### Key Findings
- **Existing VideoLLMs perform near random chance**: On CameraMotionVQA, most models achieve accuracy close to 25% (the 4-choice random baseline), including Qwen2.5-VL and InternVL. Notably, CameraBench fine-tuned variants perform even worse than their base counterparts.
- **Constraint modeling is critical**: Adding the incompatibility constraint raises instance accuracy from 0.572 to 0.738 (+16.6%), demonstrating the significant benefit of encoding physical mutual exclusivity in multi-label prediction.
- **Motion cues attenuate with depth**: Probing experiments confirm that motion recoverability is highest at layer 7 of Qwen2.5-VL and lowest at layer 31 (the final layer), supporting the hypothesis that deep token compression erases motion information.
- **Structured prompting is effective**: After injecting motion labels, VideoLLM outputs shift from vague descriptions such as "camera quickly pans with motion blur" to precise ones such as "pan-left followed by static medium close-up," enabling temporally structured cinematic descriptions.
- **Distillation is viable but involves a trade-off**: VGGT→Q-Former distillation incurs an 8.13% accuracy loss but yields a 5.3× throughput gain and a 61% reduction in memory usage.
- **Static scenes are a weakness of VGGT**: Static clips are out-of-distribution for VGGT, whose reconstruction prior assumes camera movement, and require dedicated handling.

## Highlights & Insights
- The **"benchmarking → diagnosis → injection" research paradigm** is highly instructive: first quantify the problem (CameraMotionVQA shows VideoLLMs are nearly guessing) → diagnose the root cause (probing confirms motion information attenuates with depth) → propose a solution (plug-and-play 3DFM injection). This diagnosis-driven methodology is more convincing than directly proposing a method.
- **Constraint-aware multi-label modeling**: Using the incompatibility matrix $M$ to enforce physical constraints at both training and inference is simple yet highly effective. This idea transfers naturally to any multi-label classification task with physical or logical mutual exclusivity constraints.
- **Model-agnostic, training-free, plug-and-play design**: No VideoLLM weights are modified; geometric priors are injected solely through structured prompts, making the approach highly practical and immediately applicable to any new VideoLLM.

## Limitations & Future Work
- **Synthetic-to-real domain gap**: CameraMotionDataset is built on UE5-rendered synthetic data; motion blur, compression artifacts, and non-ideal camera models in real-world videos may degrade performance.
- **Only extrinsic motion is considered; intrinsic changes (zoom) are ignored**: Zoom in/out is a widely used cinematographic technique that the current method cannot detect.
- **Only one 3DFM (VGGT) is explored**: No comparison is made against other geometric foundation models such as DUSt3R or MASt3R.
- **Structured prompting depends on the LLM's in-context learning quality**: Different VideoLLMs vary in prompt sensitivity, potentially leading to inconsistent gains.
- **The 1-second clip granularity may be too coarse**: Rapid camera motion changes shorter than 0.5 seconds (e.g., whip pans) may go undetected.

## Related Work & Insights
- **vs. CameraBench**: CameraBench defines a camera motion taxonomy with manual annotations but lacks precise geometric labels. CameraMotionDataset deterministically derives labels from precise camera extrinsics of synthetic data, yielding higher annotation quality at the cost of a domain gap.
- **vs. VLM-3R**: VLM-3R integrates 3D reconstruction features into VLMs via end-to-end training. This paper adopts a fully training-free structured prompting approach — the two are complementary rather than competing: VLM-3R pursues deep integration while this work provides plug-and-play injection.
- **vs. SpatialVID**: SpatialVID provides per-frame depth and pose-derived instructions but focuses on spatial description rather than discrete motion primitive classification.
- A natural follow-up question: could VGGT camera tokens be fed directly as additional visual inputs to the VideoLLM — bypassing discrete classification and text-based injection — to enable finer-grained geometric perception?

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Problem formulation and diagnostic methodology are novel; the technical solution (classifier + prompt injection) is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Benchmark construction is rigorous and ablations are comprehensive; evaluation on real-world videos is absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The "benchmark → diagnosis → injection" structure is clear and well-organized; figures and tables are of high quality.
- **Value**: ⭐⭐⭐⭐ — Exposes a serious capability gap in VideoLLMs; the proposed solution is practical and plug-and-play.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Missing No More: Dictionary-Guided Cross-Modal Image Fusion under Missing Infrared](missing_no_more_dictionary-guided_cross-modal_image_fusion_under_missing_infrare.md)
- [\[CVPR 2026\] Text-guided Fine-Grained Video Anomaly Understanding](text-guided_fine-grained_video_anomaly_understanding.md)
- [\[CVPR 2026\] Beyond Semantics: Disentangling Information Scope in Sparse Autoencoders for CLIP](beyond_semantics_disentangling_information_scope_in_sparse_autoencoders_for_clip.md)
- [\[CVPR 2026\] Measuring the (Un)Faithfulness of Concept-Based Explanations](measuring_the_unfaithfulness_of_concept-based_explanations.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)

<!-- RELATED:END -->
