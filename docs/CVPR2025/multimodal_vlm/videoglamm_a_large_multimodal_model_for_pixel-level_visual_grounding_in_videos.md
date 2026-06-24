---
title: >-
  [Paper Note] VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos
description: >-
  [CVPR 2025][Multimodal VLM][Visual Grounding] VideoGLaMM is a video large multimodal model that achieves pixel-level fine-grained visual grounding in videos using a dual-visual encoder (spatial + temporal), tunable V→L and L→V adapters, and a spatiotemporal pixel decoder, while establishing the first 38K video-grounded QA dataset.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Visual Grounding"
  - "Pixel-Level Annotation"
  - "Video Segmentation"
  - "Large Multimodal Models"
  - "Spatiotemporal Alignment"
date: 2026-05-08
content_hash: 7dbf6336d9d1f448
---

# VideoGLaMM: A Large Multimodal Model for Pixel-Level Visual Grounding in Videos

**Conference**: CVPR 2025  
**arXiv**: [2411.04923](https://arxiv.org/abs/2411.04923)  
**Code**: [https://mbzuai-oryx.github.io/VideoGLaMM](https://mbzuai-oryx.github.io/VideoGLaMM)  
**Area**: Multimodal VLM  
**Keywords**: Visual Grounding, Pixel-Level Annotation, Video Segmentation, Large Multimodal Models, Spatiotemporal Alignment

## TL;DR

VideoGLaMM is a video large multimodal model that achieves pixel-level fine-grained visual grounding in videos using a dual-visual encoder (spatial + temporal), tunable V→L and L→V adapters, and a spatiotemporal pixel decoder, while establishing the first 38K video-grounded QA dataset.

## Background & Motivation

**Background**: Grounded LMMs in the image domain (e.g., GLaMM) can already associate textual responses with pixel-level masks. However, video-level LMMs (e.g., VideoChat, Video-ChatGPT) only support global understanding and dialogue, failing to ground mentioned objects into specific pixels.

**Limitations of Prior Work**: (1) Existing Video LMMs employ a single projection layer to align vision and language, which is sufficient for global comprehension but fails to capture local object details; (2) There is a lack of video instruction-tuning datasets with pixel-level mask annotations; (3) Although PG-Video-LLaVA attempts video grounding, it aggregates pre-trained modules without end-to-end training, lacking fine-grained spatiotemporal modeling capabilities.

**Key Challenge**: Visual grounding in videos requires simultaneous understanding of spatial (content within each frame) and temporal (dynamics across frames) information, yet existing architectures either use only image encoders (ignoring time) or only video encoders (losing spatial details).

**Goal**: To build an end-to-end trainable video LMM capable of generating textual responses while outputting spatiotemporally consistent pixel-level segmentation masks for each mentioned entity.

**Key Insight**: Image and video encoders provide complementary information: the image encoder provides local spatial details, while the video encoder provides global temporal semantics. Simultaneous bidirectional alignment is required: Vision $\to$ Language and Language $\to$ Vision.

**Core Idea**: A trinity architecture of a dual encoder, bidirectional adapters, and a spatiotemporal pixel decoder, complemented by the first grounded video QA dataset.

## Method

### Overall Architecture

The input video $V \in \mathbb{R}^{T \times H \times W \times C}$ is processed via two streams: an image encoder extracts spatial features $f_g$ frame-by-frame, and a video encoder extracts temporal features $f_h$ segment-by-segment. Both feature sets are projected into the LLM space via V $\to$ L adapters and concatenated with text tokens before being fed into the LLM. The generated text from the LLM contains a `<SEG>` token whose last-layer embedding is projected back to the visual space via an L $\to$ V adapter. This projection, along with multi-scale frame features processed by a frame encoder, is put into a spatiotemporal pixel decoder to output the final pixel-level mask.

### Key Designs

1. **Spatio-Temporal Dual Encoder**:

    - **Function**: Extraction of frame-level spatial features and video-level temporal features, respectively.
    - **Mechanism**: The image encoder utilizes pre-trained CLIP ViT-L/14 (336x336) to process frames individually and output local spatial features $f_g$. The video encoder employs InternVideo-v2 (224x224) with segmented sampling to partition the video into $K$ segments, each containing $s=T/K$ frames, to output global temporal features $f_h$.
    - **Design Motivation**: Ablation studies show that using only the image encoder yields mIoU=60.06 and CLAIR=18.9, while using only the video encoder achieves mIoU=64.62 but CLAIR=26.5. Utilizing a dual encoder captures both local and global features, reaching the best balance (mIoU=62.34, CLAIR=28.2).

2. **V→L and L→V Adapters**:

    - **Function**: Implementing bidirectional alignment between vision and language.
    - **Mechanism**: V $\to$ L adapters $\mathcal{W}_g$ and $\mathcal{W}_h$ project image/video features into the LLM space to obtain $Z_g$ and $Z_h$, which are then concatenated with text tokens $Z_{text}$ as $\mathcal{Z} = [Z_g, Z_h, Z_{text}]$ to feed into the LLM. The L $\to$ V adapter $\mathcal{W}_p$ projects the LLM-generated `<SEG>` token embedding into the pixel decoder space.
    - **Design Motivation**: Existing methods only implement unidirectional V $\to$ L alignment, which cannot pass the rich spatiotemporal semantics of language back to the vision side for accurate mask generation. Bidirectional adapters allow textual comprehension to enhance grounding accuracy.

3. **Spatiotemporal Pixel Decoder**:

    - **Function**: Generating fine masks based on the vision-language features from the LLM.
    - **Mechanism**: Initialization based on the SAM2 encoder-decoder. A grounded frame encoder $\mathcal{P}$ extracts multi-scale visual features from the input frames. The L $\to$ V adapter output $e_{seg}^p$ is encoded by the prompt encoder $\mathcal{H}$ to act as the prompt for the mask decoder $\mathcal{D}$. Combined with frame features $\mathcal{P}(V)$, it predicts the mask: $M = \mathcal{D}(\mathcal{P}(V), \mathcal{H}(e_{seg}^p))$. This supports spatiotemporally consistent segmentation.
    - **Design Motivation**: SAM2 natively possesses cross-frame propagation capabilities. Initializing the pixel decoder with SAM2 directly leverages its temporal modeling priors, which is much more efficient than training a spatiotemporal segmentation head from scratch.

### Loss & Training

- Total Loss: $\mathcal{L}_{total} = CE + \mathcal{L}_{masked}$, where CE is the cross-entropy loss for LLM text generation, and $\mathcal{L}_{masked}$ is the IoU loss for mask prediction.
- Training Strategy: Progressive training. The first 20 epochs are trained on image/video segmentation datasets, and epochs 20-30 introduce the GCG dataset. For referring segmentation, fine-tuning continues from epochs 30-40.
- Frozen components: dual encoders and pixel decoder/frame encoder. Trainable components: V $\to$ L adapters, L $\to$ V adapters, and the LoRA parameters of the LLM.
- LLM: Phi3-Mini-3.8B; Hardware: 4x A100 40GB + DeepSpeed.
- Extra training data: ADE20K, COCO-Stuff, RefCOCO series, LLaVA-Instruct-150k, ReasonSeg, GranDf, Ref-DAVIS17, etc.

## Key Experimental Results

### Main Results

| Task | Metric | VideoGLaMM | GLaMM+SAM2 | PG-Video-LLaVA |
|------|------|------------|------------|-----------------|
| GCG | mIoU | **62.34** | 28.60 | 24.03 |
| GCG | CIDEr | **0.59** | 0.15 | 0.01 |
| GCG | CLAIR | **28.2** | 22.9 | 15.0 |
| MeViS (Ref-VOS) | J&F | **45.15** | 38.66 | 18.87 |
| Ref-DAVIS-17 | J&F | **69.5** | - | - |
| VidSTG (VG) | mIoU | **39.66** | 38.63 | 34.20 |

| Ref-DAVIS-17 | Method | J | F | J&F |
|-------------|------|---|---|-----|
| | VideoGLaMM | **73.3** | **65.6** | **69.5** |
| | VideoLISA | 72.7 | 64.9 | 68.8 |
| | TrackGPT-13B | 70.4 | 62.7 | 66.5 |

### Ablation Study

| Encoder Configuration | mIoU | CLAIR | Description |
|-----------|------|-------|------|
| Spatial-only encoder | 60.06 | 18.9 | Good spatial, poor temporal |
| Temporal-only encoder | 64.62 | 26.5 | Highest mIoU |
| Dual encoder | 62.34 | **28.2** | Balanced spatial + temporal |

| Decoder Configuration | mIoU | CLAIR | Description |
|-----------|------|-------|------|
| Spatial-only decoder | 59.68 | 26.7 | Lacks temporal context |
| Spatiotemporal decoder | **62.34** | **28.2** | Better spatiotemporal consistency |

| Decoder Frames | mIoU | METEOR | CLAIR | Description |
|-----------|------|--------|-------|------|
| 4 frames | **63.82** | 0.094 | 27.2 | Good mask, weak dialogue |
| 8 frames | 62.34 | **0.103** | **28.2** | Better dialogue quality |

### Key Findings

- The mIoU on the GCG task jumps from 28.60 (GLaMM+SAM2) to 62.34, showing that end-to-end training significantly outperforms modular concatenation.
- Outperforming VideoLISA (which requires post-processing) on MeViS motion-guided segmentation verifies the effectiveness of spatiotemporal modeling.
- The complementarity of dual encoders is evident: the image encoder excels in spatial details but lacks temporal information, while the video encoder captures temporal dynamics but struggles with details.
- The spatiotemporal pixel decoder improves mIoU by approximately 3% compared to the spatial-only version.
- Using more supervised frames (8 vs 4) sacrifices a small amount of mIoU but significantly enhances dialogue quality.

## Highlights & Insights

- **Clear end-to-end architectural design**: Dual encoder $\to$ bidirectional adapters $\to$ spatiotemporal decoder, where each component targets a clear sub-problem. It is modular yet fully end-to-end trainable.
- **Practical semi-automatic dataset construction pipeline**: Utilizing a combination of Gemini-Pro, GPT-4o, and SAM to automatically generate grounded captions and masks from existing video datasets serves as a highly replicable annotation paradigm.
- **First to achieve pixel-level grounded dialogue in video LMMs**, bridging the gap from image-level GLaMM to video.
- **Smart choice of SAM2 for pixel decoder initialization**, seamlessly borrowing its temporal propagation priors.

## Limitations & Future Work

- Although the GCG dataset contains 38K samples, the video pairs mostly comprise short-to-medium-length clips; long-form video understanding remains unverified.
- The data labels contain noise (partly generated by the semi-automatic pipeline), and the video descriptions do not exhaust all entities in the scene.
- The segmentation performance across objects of varying granularities (large/small/fine-grained) is unbalanced, potentially due to the uneven distribution in the training data.
- The quality of grounding across different granularities has not been quantitatively evaluated in detail.
- Future directions: long-video support, higher-quality dense annotation, and balanced multi-granularity segmentation.

## Related Work & Insights

- **vs GLaMM**: An image-level grounded LMM. Direct extension to video yields poor performance (requiring SAM2 concatenation), proving that moving from images to videos is not a simple architectural reuse.
- **vs PG-Video-LLaVA**: A modular concatenation method that uses audio transcripts to assist comprehension but is not end-to-end, resulting in low grounding accuracy (mIoU 24).
- **vs VideoLISA**: Focuses solely on referring segmentation instead of GCG, needing post-processing to boost performance. VideoGLaMM unifies all three tasks.
- **vs Video-ChatGPT/Video-LLaMA**: Only performs global dialogue, lacking pixel-level grounding capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ The first pixel-level grounded LMM for video, with a highly systematic architectural design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across three tasks + multi-dimensional ablation, though missing long-form and fine-grained analyses.
- Writing Quality: ⭐⭐⭐⭐ The architecture is clearly described, and the pipeline diagram is highly informative.
- Value: ⭐⭐⭐⭐ The dataset and model provide practical advancements to the video grounding community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] MIMO: A Medical Vision Language Model with Visual Referring Multimodal Input and Pixel Grounding Multimodal Output](mimo_a_medical_vision_language_model_with_visual_referring_multimodal_input_and_.md)
- [\[CVPR 2025\] Your Large Vision-Language Model Only Needs a Few Attention Heads for Visual Grounding](your_large_vision-language_model_only_needs_a_few_attention_heads_for_visual_gro.md)
- [\[CVPR 2025\] ReVisionLLM: Recursive Vision-Language Model for Temporal Grounding in Hour-Long Videos](revisionllm_recursive_vision-language_model_for_temporal_grounding_in_hour-long_.md)
- [\[CVPR 2025\] LamRA: Large Multimodal Model as Your Advanced Retrieval Assistant](lamra_large_multimodal_model_as_your_advanced_retrieval_assistant.md)
- [\[CVPR 2025\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)

</div>

<!-- RELATED:END -->
