---
title: >-
  [Paper Note] Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models
description: >-
  [ECCV 2024][Multimodal Efficiency][visual grounding] Groma proposes a new paradigm that embeds localization capabilities directly into the visual tokenization process. By discovering regions of interest (ROIs) via a region proposer and encoding them into region tokens, Groma enables MLLMs to perform high-accuracy referring and grounding without relying on LLM-generated coordinates or external modules. It also leverages GPT-4V with visual prompting to construct Groma Instruct…
tags:
  - "ECCV 2024"
  - "Multimodal Efficiency"
  - "visual grounding"
  - "Region Tokenization"
  - "Grounding"
  - "Referring"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 23f3b4fa3310c11e
---

# Groma: Localized Visual Tokenization for Grounding Multimodal Large Language Models

**Conference**: ECCV 2024  
**arXiv**: [2404.13013](https://arxiv.org/abs/2404.13013)  
**Code**: [https://groma-mllm.github.io/](https://groma-mllm.github.io/)  
**Area**: Multimodal VLM  
**Keywords**: visual grounding, Region Tokenization, Grounding, Referring, Multimodal Large Language Models

## TL;DR

Groma proposes a new paradigm that embeds localization capabilities directly into the visual tokenization process. By discovering regions of interest (ROIs) via a region proposer and encoding them into region tokens, Groma enables MLLMs to perform high-accuracy referring and grounding without relying on LLM-generated coordinates or external modules. It also leverages GPT-4V with visual prompting to construct Groma Instruct, the first grounded chat dataset featuring dual visual-textual prompts.

## Background & Motivation

**Background**: MLLMs perform exceptionally well in image-level understanding (such as captioning and VQA) but generally lack localization capabilities, making them unable to link their understanding to specific regions in the visual context.

**Limitations of Prior Work**:
   - **Approach A** (LLM-based coordinate outputs, e.g., Kosmos-2, Shikra): These impose high computational overhead on the LLM and struggle with high-resolution inputs. Sequential output is also ill-suited for dense prediction tasks (such as segmentation).
   - **Approach B** (External modules, e.g., LISA using SAM): Requires the image to be processed twice (once by the MLLM and once by the localization module), resulting in high inference latency.

**Key Challenge**: Localization itself primarily requires perceptual capabilities rather than high-level semantic reasoning. However, existing methods either burden the heavy LLM with localization tasks or introduce separate, external modules.

**Goal**: How to equip MLLMs with precise localization capabilities without relying on external modules or increasing the burden on the LLM.

**Key Insight**: Borrowing from open-vocabulary detection, grounding is decoupled into localization and recognition. The localization task is then offloaded to the visual tokenizer.

**Core Idea**: Perceive-then-understand—utilizing the spatial understanding of the visual tokenizer for localization, while the LLM focuses solely on semantic understanding and reasoning.

## Method

### Overall Architecture

Pipeline: Input image ($448 \times 448$) $\rightarrow$ DINOv2 Image Encoder $\rightarrow$ split into two paths: (1) Global image tokens (4-to-1 downsampling) $\rightarrow$ MLP $\rightarrow$ LLM, (2) Region Proposer (DDETR) $\rightarrow$ discover ROIs $\rightarrow$ Region Encoder (Multi-scale ROIAlign) $\rightarrow$ region tokens $\rightarrow$ MLP $\rightarrow$ LLM $\rightarrow$ output text (optionally containing grounded references).

### Key Designs

1. **Image Encoder (DINOv2)**:

    - **Function**: Encodes the input image into patch-level features, while providing feature pyramids for both the region proposer and region encoder.
    - **Mechanism**: Chooses DINOv2 instead of CLIP as the visual encoder because DINOv2 excels at high-resolution input and fine-grained localization features. On the COCO detection task, DINOv2 at $448 \times 448$ achieves 43.6 AP, whereas CLIP at $336 \times 336$ only reaches 32.4 AP.
    - **Design Motivation**: Localization tasks demand precise spatial features. Self-supervised pre-training enables DINOv2 to preserve better spatial structural information.
    - **Token Merging**: Every 4 adjacent 2D patch tokens are merged into 1, reducing the input length for the LLM (resulting in at most 356 visual tokens).

2. **Region Proposer (Deformable DETR)**:

    - **Function**: Acts as a class-agnostic detection head to discover potential regions of interest (ROIs) from the image.
    - **Mechanism**: Constructs a feature pyramid from the last 4 layers of DINOv2 $\rightarrow$ Deformable DETR transformer $\rightarrow$ binary classifier (based on localization quality score rather than object class) $\rightarrow$ generates 300 proposals $\rightarrow$ NMS (threshold 0.6) + confidence filtering ($>0.15$) $\rightarrow$ extracts the top 100.
    - **Design Motivation**: Localization is a low-level perceptual task that does not require LLM intervention. Delegating localization to a specialized detector head allows pre-training on large-scale detection datasets (COCO, Objects365, OpenImages, V3Det, SA1B), which would be computationally prohibitive if directly processed by the LLM.
    - **Training Data**: 5.7M detection annotations (including 2M filtered SA1B data used to expand to part- and stuff-level proposals).

3. **Region Encoder (Multi-scale ROIAlign)**:

    - **Function**: Encodes region proposals (from either the proposer or user input) into region tokens.
    - **Mechanism**: Extracts a feature pyramid from the last 3 layers of DINOv2 $\rightarrow$ multi-scale ROIAlign crops and merges region features into unified region tokens $\rightarrow$ MLP projects them into the LLM's feature space.
    - **Design Motivation**: Compared to numeric coordinate representations (e.g., Shikra) or discrete positional tokens (e.g., Kosmos-2), region tokens directly carry the underlying regional semantic features, making them more intuitive and interpretable for the LLM.

4. **Unified Referring & Grounding Format**:

    - **Function**: Registers region tokens into the LLM vocabulary using proxy tokens `<r1>, <r2>, ..., <rn>` to unify inputs (referring) and outputs (grounding).
    - **Mechanism**: During grounding output, the LLM refers to proxy tokens to link text with image regions; during referring input, user-specified regions are similarly encoded into region tokens and inserted into the instruction. A special `[grounding]` token prompts the model to generate grounded responses.
    - **Design Motivation**: A unified format avoids having to design separate encoding schemes for referring and grounding, simplifying the architecture.

5. **Groma Instruct Dataset (GPT-4V Assisted)**:

    - **Function**: Constructs 30K visual grounded dialogue samples for instruction fine-tuning.
    - **Mechanism**: Targets images from Visual Genome that contain dense region annotations $\rightarrow$ labels numeric markers on the images using Set-of-Mark (SoM) technology $\rightarrow$ feeds the marked images along with rich textual contexts (region descriptions, image descriptions, QA pairs) into GPT-4V $\rightarrow$ uses in-context learning to generate grounded conversations.
    - **Design Motivation**: Existing dialogue benchmarks lack grounded information, which impairs the ability of grounded MLLMs to maintain grounding capabilities during long-turn dialogues. Groma Instruct is the first grounded chat dataset constructed using both visual and textual prompting.

### Loss & Training

Three-stage training:
- **Stage 1 - Detection Pre-training** (12 epochs): Trains the region proposer only (image encoder is frozen) using 5.7M detection data, with $lr=2e-4$ and $batch\ size=64$.
- **Stage 2 - Alignment Pre-training** (2 epochs): Trains the region encoder and MLP projector (all other parameters are frozen) using 3.2M vision-language data, with $lr=1e-4$ and $batch\ size=128$.
- **Stage 3 - Instruction Fine-tuning** (1 epoch): Unfreezes the LLM, training with 857K high-quality dataset samples (including Groma Instruct), with $lr=1e-5$ and $batch\ size=128$.

Total training cost: roughly 8 days on 8 $\times$ A100.

## Key Experimental Results

### Main Results

| Dataset | Metric | Groma | Ferret | MiniGPT-v2 | Shikra | Gain |
|--------|------|-------|--------|------------|--------|------|
| RefCOCO val | Acc@0.5 | **89.53** | 87.49 | 88.69 | 87.01 | +1.04 |
| RefCOCO testA | Acc@0.5 | **92.09** | 91.35 | 91.65 | 90.61 | +0.44 |
| RefCOCO testB | Acc@0.5 | **86.26** | 82.45 | 85.33 | 80.24 | +0.93 |
| RefCOCO+ val | Acc@0.5 | **83.90** | 80.78 | 79.97 | 81.60 | +2.30 |
| RefCOCOg val | Acc@0.5 | **86.37** | 83.93 | 84.44 | 82.27 | +1.93 |
| REC Average | Acc@0.5 | **86.52** | 83.91 | 84.29 | 82.93 | +2.23 |
| LVIS-Ground | AR | **28.8** | 16.8 | 11.4 | 4.9 | +12.0 |
| LVIS-Ground | AR@0.75 | **30.3** | 16.3 | 11.2 | 2.0 | +14.0 |

Outperforms all grounded MLLMs of the same scale, leading the second-best Ferret by over 10 AR on LVIS-Ground.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| CLIP $336 \times 336$ | 32.4 AP (COCO det) | Weak localization capabilities of CLIP |
| DINOv2 $336 \times 336$ | 38.9 AP (+6.5) | DINOv2 significantly outperforms CLIP |
| DINOv2 $448 \times 448$ | 43.6 AP (+4.7) | Higher resolution provides further improvement |
| Frozen LLM (finetune) | Grounding 84.02%, Referring CIDEr 148.0 | Strong localization capabilities can be achieved even with a frozen LLM |
| Unfrozen LLM (finetune) | Grounding 86.52%, Referring CIDEr 158.4 | Stronger comprehension capabilities when unfrozen |
| 4$\times$ Token Merge | REC Average 86.47% | 4-to-1 downsampling is practically lossless |
| No Merging | REC Average 86.55% | Only +0.08% difference |

### Key Findings

- **Localization can be decoupled from comprehension**: When the LLM is frozen, Groma still maintains grounding capabilities comparable to Ferret (84.02% vs 83.91%), demonstrating that localization does not strictly require LLM capacity.
- **Token merging is nearly lossless**: 4$\times$ downsampling has only a 0.08% impact on grounding, validating the optimality of the decoupled design in balancing efficiency and accuracy.
- **LVIS-Ground exposes limitations of exist methods**: All methods perform poorly on small object localization ($AR@s$), and most tend to predict only a single box per query—reflecting the nature of the training datasets (such as RefCOCO), where each query is annotated with only one target.
- Performs comparably to or better than GLaMM (which requires separate referring and grounding designs) on region captioning.

## Highlights & Insights

- **The "Perceive-then-Understand" design philosophy**: Carrying out localization within the visual tokenizer instead of the LLM mimics human visual processing. This represents a highly generalizable paradigm.
- **Training efficiency from decoupled architecture**: Enables pre-training the localization ability on million-scale detection datasets without involving the LLM, a feat impossible for conventional MLLMs.
- **LVIS-Ground benchmark**: Points out evaluation gaps by introducing the AS-MANY-Protocol evaluation scheme and a 1203-class grounding benchmark.
- **Groma Instruct construction**: Combining SoM visual prompting, GPT-4V, and rich textual context is an effective recipe for constructing grounded conversation data.
- **Unified region token**: Elegant design where the same token representation handles both referring inputs and grounding outputs.

## Limitations & Future Work

- Does not support free-form region inputs (such as clicks or scribbles), being restricted only to bounding boxes.
- Does not support pixel-level grounding (such as segmentation masks); the authors suggest replacing the box proposer with Mask2Former.
- DINOv2 features are not naturally aligned with text, resulting in slightly weaker performance on conversation and reasoning tasks compared to CLIP-based methods.
- The recall of the region proposer sets the performance ceiling of the entire system—if a target region is not proposed, it cannot be understood.
- Insufficient small-object annotations in the training data limit the capability to locate small objects.

## Related Work & Insights

- **vs Shikra/Kosmos-2 (LLM-based coordinate outputs)**: Groma avoids the computational bottleneck of the LLM processing high-resolution inputs and achieves superior localization accuracy (LVIS-Ground AR: 28.8 vs 4.9).
- **vs LISA/GLaMM (External localization modules)**: Groma does not require an external SAM module, thereby providing more efficient inference while unifying the representation of referring and grounding.
- **vs Ferret (Spatial-aware visual sampling)**: Groma actively discovers ROIs through a region proposer rather than passively receiving inputs, outperforming Ferret by over 12 AR on LVIS-Ground.
- **vs GPT4RoI (Simple pooling for region features)**: Groma extracts more refined, hierarchical features using multi-scale ROIAlign.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Embedding localization into visual tokenization is a completely fresh paradigm, and the "perceive-then-understand" design philosophy is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers three tasks—grounding, referring, and VQA—proposes the new LVIS-Ground benchmark, and provides comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The logic is exceptionally clear; the comparison of the three paradigms (Figure 2) is highly intuitive and strong, and the motivation is naturally derived.
- Value: ⭐⭐⭐⭐☆ Introduces a highly influential new paradigm, though practical application scenarios are somewhat constrained by the lack of support for masks and free-form inputs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] GroundVTS: Visual Token Sampling in Multimodal Large Language Models for Video Temporal Grounding](../../CVPR2026/vlm_efficiency/groundvts_visual_token_sampling_in_multimodal_large_language_models_for_video_te.md)
- [\[ECCV 2024\] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)
- [\[ICCV 2025\] ShortV: Efficient Multimodal Large Language Models by Freezing Visual Tokens in Ineffective Layers](../../ICCV2025/vlm_efficiency/shortv_efficient_multimodal_large_language_models_by_freezing_visual_tokens_in_i.md)
- [\[ECCV 2024\] Bad Students Make Great Teachers: Active Learning Accelerates Large-Scale Visual Understanding](bad_students_make_great_teachers_active_learning_accelerates_large-scale_visual_.md)
- [\[ACL 2025\] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?](../../ACL2025/vlm_efficiency/token_pruning_in_multimodal_large_language_models_are_we_solving_the_right_probl.md)

</div>

<!-- RELATED:END -->
