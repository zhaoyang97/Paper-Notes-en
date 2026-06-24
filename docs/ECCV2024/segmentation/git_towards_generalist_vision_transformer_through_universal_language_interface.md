---
title: >-
  [Paper Note] GiT: Towards Generalist Vision Transformer through Universal Language Interface
description: >-
  [ECCV 2024][Segmentation][generalist vision model] The GiT framework is proposed, which unifies five major vision tasks—image captioning, object detection, instance segmentation, semantic segmentation, and visual grounding—into autoregressive sequence generation through a universal language interface. Using only a pure ViT (without any task-specific modules), it achieves multi-task joint training where tasks mutually enhance each other.
tags:
  - "ECCV 2024"
  - "Segmentation"
  - "generalist vision model"
  - "universal language interface"
  - "multi-task learning"
  - "ViT"
  - "auto-regressive"
date: 2026-05-08
content_hash: e8f045ab10f6a7c8
---

# GiT: Towards Generalist Vision Transformer through Universal Language Interface

**Conference**: ECCV 2024  
**arXiv**: [2403.09394](https://arxiv.org/abs/2403.09394)  
**Code**: [https://github.com/Haiyang-W/GiT](https://github.com/Haiyang-W/GiT)  
**Area**: Semantic Segmentation / Multi-Task Vision Modeling  
**Keywords**: generalist vision model, universal language interface, multi-task learning, ViT, auto-regressive

## TL;DR
The GiT framework is proposed, which unifies five major vision tasks—image captioning, object detection, instance segmentation, semantic segmentation, and visual grounding—into autoregressive sequence generation through a universal language interface. Using only a pure ViT (without any task-specific modules), it achieves multi-task joint training where tasks mutually enhance each other.

## Background & Motivation
**Background**: LLMs have demonstrated that multi-layer Transformer architectures (such as GPT) can handle various tasks with simple stacking. The vision domain has attempted to replicate this success but remains constrained by task-specific modules (detection heads, pixel decoders, etc.).

**Limitations of Prior Work**: Unified models like LLaVA, Unified-IO, and OFA still retain task-specific components such as vision encoders, RPNs, and perception heads. These modules lead to complex multi-stage training pipelines, make model scaling difficult, and primarily focus on image-level vision-language tasks while neglecting classic perception tasks like detection and segmentation.

**Key Challenge**: The output formats of vision tasks differ drastically—detection outputs a variable number of bounding boxes, segmentation outputs binary masks, and captioning generates text—making it difficult to handle them with a single model. Furthermore, the sequence length of dense prediction tasks causes the computational cost of autoregressive decoding to explode.

**Goal**: To design a generalist vision model with an extremely minimalist architecture that handles all tasks from the image level to the pixel level using a pure multi-layer Transformer, without any vision-specific modules (except for patch projection).

**Key Insight**: All vision task targets are converted into language token sequences, utilizing a standard vocabulary instead of extra tokens. Dense prediction is decomposed into $N$ parallel subprocesses through grid sampling to solve the efficiency bottleneck of pixel-level autoregression.

**Core Idea**: Utilizing a universal language interface combined with parallel grid decoding, allowing a pure ViT to achieve unified modeling for all tasks, from captioning to segmentation.

## Method

### Overall Architecture
The architecture of GiT is extremely simple: a window-based ViT (the same structure as used in SAM) serves as the shared backbone, supplemented by text embeddings and an out-of-vocabulary compression module. There are no RPNs, no pixel decoders, and no FPNs. Shared parameters account for $>98\%$ of the entire model. The input consists of image patches, instruction text, local features of $N$ grid points, and task identifiers. The image part utilizes bidirectional self-attention (similar to an encoder), and the local prediction part uses unidirectional causal attention (similar to a decoder), while sharing the same set of Transformer layers.

### Key Designs
1. **Unified Representation**

    - **Function**: Maps all modalities (images, text, bounding boxes, masks) into a unified token space.
    - **Mechanism**: Text is tokenized using WordPiece (~30K vocabulary). Multi-segment concepts (e.g., "traffic light") are compressed into a single out-of-vocabulary token via a single-layer attention mechanism: $\mathcal{F}_{\text{traffic light}} = \text{Attention}(\text{TE}(\mathcal{I}_0)+\text{PE}(0), \text{TE}(\mathcal{I}_1)+\text{PE}(1))$. Sparse objects (boxes, polygons) are represented as $(C, P=\{x_i,y_i\}_{i=1}^N)$. Dense labels are flattened into 1D sequences in raster order.
    - **Design Motivation**: Avoids extending the vocabulary with extra tokens, thus simplifying post-processing. The unified representation allows all tasks to share the same autoregressive loss.

2. **Multi-Task Template with Parallel Decoding**

    - **Function**: Embeds vision tasks of different granularities into a unified instruction template and processes them in parallel through grid sampling.
    - **Mechanism**: The template is defined as $\langle\text{Image}\rangle\langle\text{Instruction}\rangle + N \times [\langle\text{LocalFeature}_i\rangle\langle\text{Task}_i\rangle:\langle\text{Response}_i\rangle]$. For image-level tasks, $N=1$; for object detection, $N=625$ ($25 \times 25$ grid); for semantic segmentation, $N=1764$ ($42 \times 42$ grid). The local feature of each grid point is obtained via bilinear interpolation, and each subprocess is decoded independently and in parallel.
    - **Design Motivation**: Directly applying pixel-level autoregression to the entire image would result in an excessively long sequence ($672 \times 672$ has 450,000 pixels). Decomposing this into $1764$ subprocesses where each requires only $16$ decoding steps drastically reduces computational complexity. Meanwhile, grid parallelism avoids the efficiency bottlenecks associated with non-parallel decoding.

3. **Out-of-Vocabulary Compression Module**

    - **Function**: Compresses multi-token concepts, such as category names (e.g., "traffic light") and coordinate values, into a single token.
    - **Mechanism**: Multi-segment text is first tokenized by a standard tokenizer to generate subword indices, which are then fused into a single feature vector using a single-layer attention mechanism, taking the first output as the compressed representation.
    - **Design Motivation**: Avoids sequence expansion caused by introducing delimiters, and prevents complex post-processing stemming from variable-length, multi-token outputs.

4. **Attention Mask Strategy**

    - **Function**: Implements the dual functionality of both coder and decoder inside the same set of Transformer layers.
    - **Mechanism**: Bidirectional attention (including image-to-text attention) is applied between image patches and instructions. Causal unidirectional attention is applied to local features and target predictions. In window attention, patch tokens representing different grid subprocesses only interact with grid points inside the same window.
    - **Design Motivation**: Maintains architectural purity (no separate encoder/decoder structures are required) while enabling image-to-text attention to enhance text-conditioning capability.

### Loss & Training
- **Loss Function**: All tasks uniformly use the standard CrossEntropy loss for next-token prediction, with dynamic control applied to the vocabulary—each task uses its corresponding task-specific vocabulary subset during training and inference.
- **Multi-Task Sampling**: The 5 tasks are sampled uniformly ($1/5$ each), independent of dataset size, to prevent smaller datasets from being overwhelmed. Within each task, sampling is balanced by domain (daily, indoor, outdoor), and samples within a domain are drawn proportionally to their dataset sizes.
- **Model Scale**: Base (131M) / Large (387M) / Huge (756M). The initial layers inherit pre-trained parameters from SAM.
- **Joint Training**: Trained on 27 public datasets (17M samples) without task-specific fine-tuning.

## Key Experimental Results

### Main Results (Multi-task Generalist)

| Task | Metric | GiT-Huge (multi) | Single-Task Training | Multi-Task Gain |
|------|------|-------------------|-----------|-----------|
| Object Detection (COCO) | AP | 45.1 | 43.5 | +1.6 |
| Instance Segmentation (COCO) | AP₅₀ | 54.2 | 52.6 | +1.6 |
| Semantic Segmentation (ADE20K) | mIoU | 47.7 | 47.6 | +0.1 |
| Image Captioning (COCO) | CIDEr | 133.0 | 128.3 | +4.7 |
| Visual Grounding (RefCOCO) | Acc@0.5 | 82.4 | 79.9 | +2.5 |

### Comparison with Other Generalist Models

| Method | No. of Task-Specific Modules | Det AP | InsSeg AP | SemSeg mIoU | Caption CIDEr | Grounding Acc |
|------|-----------|--------|-----------|-------------|---------------|---------------|
| Pix2Seq v2 | 2 | 46.5 | 38.2 | - | - | - |
| VisionLLM | 6 | 44.8 | - | - | - | - |
| Uni-Perceiver v2 | 5 | 42.0/52.0† | - | - | - | - |
| **GiT-Huge** | **0** | **45.1** | **42.6** | **47.7** | **133.0** | **82.4** |

### Ablation Study

| Configuration | Det AP | Caption CIDEr | Description |
|------|---------|-----------|------|
| Without image-to-text attention | 44.3 | 130.2 | Removes text conditioning |
| With image-to-text attention | **45.1** | **133.0** | Default setting |
| Single-task training | 43.5 | 128.3 | No cross-task enhancement |
| Multi-task joint training | **45.1** | **133.0** | Mutual enhancement across tasks |

### Key Findings
- The gains from multi-task joint training are positive across all tasks, confirming a multi-task mutual enhancement effect similar to that observed in LLMs.
- Different types of capabilities (image understanding, grounding, segmentation, language) can be learned collaboratively within shared parameters.
- Joint training on 27 datasets enables GiT to exhibit strong zero-shot and few-shot transfer capabilities on unseen data.
- The simplicity of the architecture ($>98\%$ shared parameters) makes model scaling highly straightforward.

## Highlights & Insights
- **Minimalist architecture covering all vision tasks for the first time**: GiT is the first generalist model that supports detection, segmentation (instance + semantic), captioning, and grounding simultaneously without any vision-specific modules, demonstrating the feasibility of pure multi-layer Transformers in computer vision.
- **Parallel grid decoding as a key bottleneck breakthrough**: Decomposing dense prediction into thousands of independent subprocesses for parallel decoding maintains the uniformity of the autoregressive language interface while resolving the efficiency issue of excessively long pixel-level sequences. This design is highly elegant from an engineering perspective.
- **Multi-task mutual enhancement** mode in vision is systematically validated for the first time: sharing grounding capabilities yields improvements of 1.6 AP for detection and 4.7 CIDEr for captioning, reflecting the underlying commonalities between vision and language capabilities.

## Limitations & Future Work
- The mIoU for semantic segmentation (47.7) is close to the specialized model Mask2Former (47.2) but shows no significant advantage; multi-task gains for dense prediction tasks remain relatively limited (+0.1).
- Currently, it only handles 2D vision tasks; directions such as 3D perception and video understanding are not yet covered.
- Polygon-based instance segmentation differs significantly from mainstream mask-based methods, which limits direct comparison.

## Related Work & Insights
- **vs VisionLLM**: VisionLLM requires 6 task-specific modules and multi-stage training; in contrast, GiT uses 0 task-specific modules and adopts one-stage joint training.
- **vs Pix2Seq v2**: Pix2Seq v2 does not support semantic segmentation and suffers from low efficiency due to non-parallel decoding; GiT provides broader task coverage and processes in parallel.
- **vs LLaVA**: LLaVA requires an external vision encoder + LLM with a two-stage pipeline and does not support detection or segmentation; GiT is end-to-end and significantly more minimalist.

## Rating
- Novelty: ⭐⭐⭐⭐ The minimalist design unifying all vision tasks with a pure ViT is impressive, and the parallel grid decoding is highly creative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 5 tasks and 27 datasets, thorough ablation studies, and complete zero-shot validation.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the tables are highly informative.
- Value: ⭐⭐⭐⭐ An important exploration toward closing the gap between vision and language architectures, providing valuable insights for generalist vision foundation models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Rotary Position Embedding for Vision Transformer](rotary_position_embedding_for_vision_transformer.md)
- [\[ECCV 2024\] SiLC: Improving Vision Language Pretraining with Self-Distillation](silc_improving_vision_language_pretraining_with_self-distillation.md)
- [\[ECCV 2024\] SCLIP: Rethinking Self-Attention for Dense Vision-Language Inference](sclip_rethinking_self-attention_for_dense_vision-language_inference.md)
- [\[ECCV 2024\] UniFS: Universal Few-Shot Instance Perception with Point Representations](unifs_universal_few-shot_instance_perception_with_point_representations.md)
- [\[ICLR 2026\] Locality-Attending Vision Transformer](../../ICLR2026/segmentation/locality-attending_vision_transformer.md)

</div>

<!-- RELATED:END -->
