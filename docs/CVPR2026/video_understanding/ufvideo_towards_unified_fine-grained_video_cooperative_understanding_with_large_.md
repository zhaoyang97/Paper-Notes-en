---
title: >-
  [Paper Note] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models
description: >-
  [CVPR 2026][Video Understanding][Unified video understanding] UFVideo is the first Video LLM to unify global, pixel-level, and temporal-level video understanding within a single model. Through a visual-language guided al…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Unified video understanding"
  - "multi-grained cooperation"
  - "pixel-level segmentation"
  - "temporal grounding"
  - "Video LLM"
date: 2026-05-08
content_hash: eda9af7d11d3bfef
---

# UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models

**Conference**: CVPR 2026
**arXiv**: [2512.11336](https://arxiv.org/abs/2512.11336)
**Code**: [https://github.com/Heven-Pan/UFVideo](https://github.com/Heven-Pan/UFVideo)
**Area**: Video Understanding / Multimodal VLM
**Keywords**: Unified video understanding, multi-grained cooperation, pixel-level segmentation, temporal grounding, Video LLM

## TL;DR
UFVideo is the first Video LLM to unify global, pixel-level, and temporal-level video understanding within a single model. Through a visual-language guided alignment strategy and the SAM2 mask decoder, it simultaneously supports video question answering, object referring, video segmentation, and temporal grounding, and introduces UFVideo-Bench, a multi-grained cooperative understanding benchmark.

## Background & Motivation

1. **Background**: Current Video LLMs have expanded from general video QA to diverse fine-grained understanding tasks, including video referring, video segmentation, and temporal grounding, corresponding to pixel-level and temporal-level video understanding respectively.

2. **Limitations of Prior Work**: Existing methods focus on single-grained tasks and are trained and evaluated independently, failing to integrate perceptual and reasoning capabilities across granularities for mutual enhancement. For example, models proficient in object referring cannot handle event temporal grounding, and models specialized in temporal grounding cannot perform pixel-level segmentation.

3. **Key Challenge**: Video knowledge at different granularities is inherently complementary—fine-grained temporal knowledge can enhance comprehension of referred objects, while global video semantics can support fine-grained tasks. However, existing models generate outputs at each granularity in isolation, without explicit cross-granularity association.

4. **Goal**: To unify global, pixel-level, and temporal-level video understanding within a single model and enable these granularities to work cooperatively.

5. **Key Insight**: Design a unified visual-language guided alignment strategy that distinguishes task inputs and outputs via special tokens, and enables multi-task joint training over a shared LLM backbone.

6. **Core Idea**: Employ a unified token design (`<Ref>` / `<Seg>` / `<Temp>`) to consolidate global QA, pixel-level segmentation, and temporal grounding into a single Video LLM, achieving multi-grained cooperative video understanding.

## Method

### Overall Architecture
UFVideo uses an LLM as the backbone, with a visual encoder that maps video frames into discrete tokens aligned with text tokens. The input consists of video $V$, text query $Q$, and optional target visual prompt $M$ (mask); the output includes text answer $A$, temporal localization $T$, and segmentation mask $S$. Through unified visual-language guided alignment training, the model flexibly generates the corresponding output type for each task.

### Key Designs

1. **Multi-Grained Video Tasks Alignment**:

    - **Function**: Distinguish and unify video understanding tasks across different granularities via special tokens.
    - **Mechanism**: Three categories of special tokens are designed: `<Temp-τ>` encodes relative timestamps (video time is normalized to a fixed length $N_t$, then encoded as $\tau = \frac{t}{T_n} \times N_t$); `<Ref>` serves as a placeholder for injecting visual prompts in object referring tasks; `<Seg>` extracts segmentation-related language embeddings from LLM outputs. Text instructions are tokenized into $\mathcal{T}_i$ and temporal tokens into $\mathcal{T}_t$, forming a unified input representation.
    - **Design Motivation**: Using special tokens rather than independent modules to distinguish tasks allows sharing a single LLM backbone, avoiding task-specific architectural designs and enabling cross-task knowledge transfer through shared parameters.

2. **Encode for Multi-Modal Input**:

    - **Function**: Uniformly encode video and target visual prompts into tokens.
    - **Mechanism**: A pretrained visual encoder $\Phi_v$ (SigLIP-so400m) encodes both the video $V$ and the target visual prompt $M$, yielding video features $F_V$ and target visual features $F_M$. Target spatial features $S_M$ are then extracted from $F_M$ using the VideoRefer approach and projected into target visual tokens $\mathcal{T}_r$, which are injected at the `<Ref>` position. For segmentation tasks, $K$ frames are randomly selected and encoded by SAM2's Hiera-L encoder as visual input to the mask decoder.
    - **Design Motivation**: Mapping heterogeneous modalities into a unified token space enables the LLM to jointly process video content and object-level information.

3. **Decode for LLM Generative**:

    - **Function**: Decode text, temporal, and segmentation results from the unified LLM output.
    - **Mechanism**: Given LLM output hidden state $H$, text answers and temporal localizations are both generated as text-form tokens (temporal outputs are converted back to actual timestamps via $\mathcal{Y}_m = p_\theta(H) \times \frac{T_n}{N_t}$). Pixel-level segmentation extracts segmentation-related embeddings using the position mask $\rho_s$ of the `<Seg>` token, applies element-wise multiplication with the projection layer $\theta$, and feeds the result into SAM2's mask decoder to generate segmentation outputs. Dynamic embedding training is required to handle varying numbers of segmentation targets across samples.
    - **Design Motivation**: Text and temporal outputs can be generated directly via LLM next-token prediction, whereas pixel-level segmentation cannot be expressed directly through tokens; SAM2's mask decoder therefore serves as a bridge from language embedding space to pixel-level masks.

### Loss & Training

The total loss is $\mathcal{L} = \gamma \cdot \mathcal{L}_{text} + \mathcal{L}_{mask}$, where $\mathcal{L}_{text}$ is the standard negative log-likelihood loss for next-token prediction, and $\mathcal{L}_{mask} = \alpha \cdot \text{BCE}(S_p, S_t) + \beta \cdot \text{DICE}(S_p, S_t)$ combines binary cross-entropy and DICE losses. Hyperparameters are set to $\alpha=2.0, \beta=0.5, \gamma=1.0$. Training proceeds in two stages: Stage 1 uses a global batch size of 512 for 2 epochs; Stage 2 uses a batch size of 256 for 1 epoch. Training is conducted on 32 A800 GPUs. The visual encoder is SigLIP-so400m-patch14-384, and the pretrained model is VideoRefer 7B.

## Key Experimental Results

### Main Results

**General Video Understanding (MVBench)**:

| Model | Params | Avg. Score |
|-------|--------|------------|
| GPT-4V | - | 43.5 |
| Qwen2-VL | 7B | 67.0 |
| LLaVA-ST | 7B | 64.2 |
| UniPixel | 3B | 62.5 |
| **UFVideo** | **7B** | **67.3** |

**Video Referring Description (VideoRefer-Bench-D)**:

| Model | Single-Frame Avg | Multi-Frame Avg |
|-------|-----------------|-----------------|
| GPT-4o | 2.95 | 3.25 |
| VideoRefer | 3.42 | 3.46 |
| UniPixel | 3.47 | 3.48 |
| **UFVideo** | **3.59** | **3.61** |

**Video Referring QA (VideoRefer-Bench-Q)**:

| Model | Avg. Score |
|-------|------------|
| GPT-4o | 71.3 |
| RGA3 | 74.0 |
| UniPixel | 73.8 |
| **UFVideo** | **77.9** (Multi-Frame) |

### Ablation Study

| Configuration | MVBench Avg | VideoRefer-D (MF) | VideoRefer-Q (MF) |
|---------------|------------|-------------------|-------------------|
| Full model (UFVideo) | 67.3 | 3.61 | 77.9 |
| w/o temporal-level tasks | degraded | - | - |
| w/o pixel-level tasks | - | degraded | degraded |

### Key Findings
- UFVideo achieves state-of-the-art on all 9 public benchmarks, surpassing Qwen2-VL (67.0%) with 67.3% on MVBench.
- Multi-grained joint training yields significant mutual enhancement—performance on video referring tasks substantially exceeds that of VideoRefer, which is trained solely for referring.
- UFVideo also outperforms dedicated segmentation models on video segmentation benchmarks (MeViS, Ref-YouTube-VOS, etc.).
- UFVideo-Bench's three cooperative task types (PixRQA / PixHQA / PixTRQA) demonstrate the model's integrated capability to simultaneously produce text, temporal, and segmentation outputs.

## Highlights & Insights
- **Unified special token design is the key trick**: Using `<Ref>`, `<Seg>`, and `<Temp>` tokens to distinguish tasks rather than deploying independent modules is elegant and efficient, enabling a single 7B model to cover 4+ video understanding tasks.
- **SAM2 decoder as a segmentation bridge**: Directly generating masks from LLM outputs is infeasible; by extracting embeddings at the `<Seg>` positions and passing them into the SAM2 decoder, the method elegantly establishes a mapping between language space and pixel space.
- **Relative temporal token design**: Normalizing video duration to a fixed length before encoding enables the model to handle temporal grounding across videos of varying lengths, while allowing temporal tokens to be generated jointly with text tokens.

## Limitations & Future Work
- Results on UFVideo-Bench indicate substantial room for improvement in multi-grained cooperative understanding, particularly for PixTRQA, which requires simultaneous temporal retrieval, segmentation, and question answering.
- The number of frames and resolution are constrained by GPU memory, limiting the model's capacity to process very long videos.
- Segmentation quality is bounded by the capability ceiling of the SAM2 decoder.
- Experiments are conducted only at the 7B scale; scaling law behavior remains unverified.

## Related Work & Insights
- **vs. RGA3 / UniPixel**: These works unify pixel-level referring and segmentation but lack temporal understanding. UFVideo extends this by incorporating the temporal granularity, achieving true three-level unification.
- **vs. LLaVA-ST**: LLaVA-ST addresses spatial-temporal understanding but uses bounding boxes rather than masks, resulting in coarser granularity. UFVideo employs pixel-level masks for finer-grained understanding.
- **vs. VideoRefer**: UFVideo is initialized from VideoRefer 7B and extends it with segmentation and temporal grounding capabilities.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First Video LLM to unify three granularities, though the technical components are largely assembled from existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 9 public benchmarks plus a self-constructed benchmark with comprehensive comparisons.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured, though the notation system is relatively complex.
- **Value**: ⭐⭐⭐⭐ Provides a clear direction for unified multi-grained video understanding; UFVideo-Bench offers meaningful value to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] Token Reduction via Local and Global Contexts Optimization for Efficient Video Large Language Models](token_reduction_via_local_and_global_contexts_optimization_for_efficient_video_l.md)
- [\[CVPR 2026\] Understanding Temporal Logic Consistency in Video-Language Models through Cross-Modal Attention Discriminability](understanding_temporal_logic_consistency_in_video-language_models_through_cross-.md)
- [\[ICCV 2025\] DisTime: Distribution-based Time Representation for Video Large Language Models](../../ICCV2025/video_understanding/distime_distribution-based_time_representation_for_video_large_language_models.md)

</div>

<!-- RELATED:END -->
