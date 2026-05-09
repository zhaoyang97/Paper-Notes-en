---
title: >-
  [Paper Note] Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing
description: >-
  [ICLR 2026][Image Generation][Unified Multimodal Model] This paper identifies a role imbalance in existing unified multimodal models, where the understanding module merely acts as a translator while the generation module is forced to simultaneously serve as both "designer" and "painter." By constructing the DIM dataset (14M long-context text-image pairs + 233K CoT editing blueprints), design responsibilities are transferred to the understanding module. The resulting 4.6B-parameter model surpasses models five times its size.
tags:
  - ICLR 2026
  - Image Generation
  - Unified Multimodal Model
  - Chain-of-Thought
  - image editing
  - Designer-Painter
  - Data-Centric
date: 2026-05-08
content_hash: 1007c7164d08dc74
---

# Draw-In-Mind: Rebalancing Designer-Painter Roles in Unified Multimodal Models Benefits Image Editing

**Conference**: ICLR 2026  
**arXiv**: [2509.01986](https://arxiv.org/abs/2509.01986)  
**Code**: [GitHub](https://github.com/showlab/DIM)  
**Area**: Diffusion Models / Image Editing  
**Keywords**: Unified Multimodal Model, Chain-of-Thought, image editing, Designer-Painter, Data-Centric

## TL;DR
This paper identifies a role imbalance in existing unified multimodal models, where the understanding module merely acts as a translator while the generation module is forced to simultaneously serve as both "designer" and "painter." By constructing the DIM dataset (14M long-context text-image pairs + 233K CoT editing blueprints), design responsibilities are transferred to the understanding module. The resulting 4.6B-parameter model surpasses models five times its size.

## Background & Motivation
**Background**: Unified multimodal understanding and generation models (e.g., Show-o, BAGEL, UniWorld) achieve strong performance on text-to-image (T2I) generation, yet still lag significantly behind proprietary models such as GPT-4o-Image on instruction-guided image editing.

**Limitations of Prior Work**: In existing editing models, the understanding module encodes user instructions purely as semantic conditions (acting as a "translator"), while the generation module must simultaneously infer original layouts, localize edit regions, and render new content (acting as both "designer" and "painter"). This allocation of responsibilities is fundamentally unreasonable.

**Key Challenge**: The understanding module is typically trained on complex reasoning tasks with several times more data than the generation module, yet remains underutilized for design planning. Simply scaling parameters (e.g., Step1X-Edit's 12.5B generation parameters) is not an effective strategy.

**Goal**: How can the responsibilities of the understanding and generation modules be rebalanced to enable more effective editing?

**Key Insight**: A data-centric approach—constructing an editing dataset with CoT reasoning blueprints so that an external designer (MLLM) performs edit planning in text space, leaving the generation module only the task of "painting."

**Core Idea**: Transfer "design" responsibilities from the generation module to the understanding module, explicitly reducing the cognitive burden on the generation module via CoT editing blueprints.

## Method

### Overall Architecture
A connector-based architecture is employed: a frozen Qwen2.5-VL-3B (understanding module) is connected to a trainable SANA1.5-1.6B (generation module) via a two-layer MLP, with a total of only 4.6B parameters. At inference time, an external designer (e.g., GPT-4o) generates a CoT editing blueprint, upon which the model executes the edit.

### Key Designs
1. **DIM-T2I Dataset (14M)**: High-resolution (≥512²) images are collected from the web and annotated along 21 dimensions using an internal model, producing long-context captions. The average prompt length reaches 146.76 words (existing datasets typically have <40 words), providing a foundation for complex CoT comprehension.

2. **DIM-Edit Dataset (233K)**:

    - Data sources: 160K UltraEdit (filtered via joint SSIM/DINO/CLIP criteria) + 46K ShareGPT-4o-Image + 27K manually edited data.
    - **Quality Assessment**: GPT-4o classifies original prompts into three categories—Misaligned (discarded), Partially Aligned (supplemented with unmentioned modifications), and Aligned (disambiguated and refined).
    - **CoT Blueprint Generation**: GPT-4o generates a four-step CoT for each image pair: (1) global layout perception, (2) local object perception, (3) edit region localization, and (4) imagination of the edited image. The average prompt length reaches 252.64 words, far exceeding existing datasets.

3. **Two-Stage Training Strategy**:

    - T2I Stage: The connector and SANA1.5-1.6B are trained on DIM-T2I + 6.9M public data, with Qwen2.5-VL-3B frozen.
    - Editing Stage I: Fine-tuned on UltraEdit to acquire basic editing capabilities (source images concatenated with noise along the channel dimension).
    - Editing Stage II: Fine-tuned on DIM-Edit, yielding the final DIM-4.6B-Edit model.

### Loss & Training
- Vanilla flow matching is used as the sole training objective.
- Optimizer: AdamW; learning rate $2 \times 10^{-5}$ for T2I, $1 \times 10^{-4}$ for Editing Stage I, and $1 \times 10^{-5}$ for Stage II.
- T2I batch size: 256 (8 epochs); editing batch size: 32 (Stage I: 10 epochs, Stage II: 50 epochs).
- Distillation data such as BLIP3-o-60K are deliberately excluded to prevent data leakage and benchmark hacking.
- GPT-4o is used as the default designer at inference time; the effectiveness of multiple designers including GPT-5 and Claude is also validated.
- During editing inference, the designer does not access the target image (only the source image and instruction are provided), ensuring alignment with real-world usage scenarios.

## Key Experimental Results

### Main Results (ImgEdit Benchmark)

| Model | Params | Add | Replace | Remove | Background | Style | Action | Overall |
|-------|--------|-----|---------|--------|------------|-------|--------|---------|
| Step1X-Edit | 7B+12.5B | 3.88 | 3.40 | 2.41 | 3.16 | 4.63 | 2.52 | 3.06 |
| BAGEL | 14B | 3.56 | 3.30 | 2.62 | 3.24 | 4.49 | 4.17 | 3.20 |
| UniWorld-V1 | 7B+12B | 3.82 | 3.47 | 3.24 | 2.99 | 4.21 | 2.74 | 3.26 |
| GPT-4o-Image | — | 4.61 | 4.35 | 3.66 | 4.57 | 4.93 | 4.89 | 4.20 |
| **DIM-4.6B-Edit** | **3B+1.6B** | **4.09** | **4.00** | **3.43** | **3.87** | **4.92** | **4.08** | **3.67** |

*DIM significantly outperforms open-source models in the 14B–19B range with fewer than 5B parameters, narrowing the gap with GPT-4o-Image.*

### GEdit-Bench-EN (Text Change task excluded)

| Model | BC | CA | MA | MC | SC | SA | SRM | SRP | TT | AVG (w/o TC) |
|-------|----|----|----|----|----|----|-----|-----|----|-------------|
| Step1X-Edit | 7.03 | 6.26 | 6.46 | 3.66 | 7.24 | 7.17 | 6.42 | 7.39 | 6.62 | 6.35 |
| **DIM-4.6B-Edit** | **7.02** | **6.81** | **6.00** | **4.67** | **7.16** | **7.48** | **6.67** | **6.76** | **6.55** | **6.50** |

### Key Findings
- With only 1.6B generation parameters, DIM surpasses Step1X-Edit backed by 12B FLUX, validating that data quality outweighs parameter scale.
- Janus-4o (7B), trained on the same data (ShareGPT-4o-Image), performs substantially worse than DIM, demonstrating that the gains stem from the CoT blueprints themselves rather than the data source.
- Multiple external designers (GPT-4o, GPT-5, Claude, etc.) all effectively drive DIM, confirming the generalizability of the framework.
- T2I quality is also strong: GenEval 0.77, MJHQ-30K FID best of 5.50.

## Highlights & Insights
- **Insightful Diagnosis**: Attributing editing failures to "role imbalance" rather than insufficient model size is a highly original perspective.
- **Exemplary Data Engineering**: The four-step CoT blueprint design (perception → localization → imagination) closely mirrors the human editing thought process.
- **Extreme Efficiency**: Only a two-layer MLP is used as the connector (MetaQuery employs a 1.6B transformer), demonstrating that complex connectors are unnecessary.
- **Rigorous Data Curation**: Three-tier prompt quality assessment combined with multi-dimensional filtering effectively mitigates common noise in AI-generated data.
- **Design/Execution Separation Paradigm**: This paradigm is transferable to other generative tasks requiring complex reasoning.

## Limitations & Future Work
- Reliance on an external MLLM (GPT-4o) as the designer increases inference cost and API dependency.
- Performance on the Text Change task is relatively weak due to the absence of corresponding training data; text editing data should be supplemented in future work.
- Internalizing the designer into the model has not been explored; an end-to-end approach may be preferable.
- The two-stage editing training may introduce catastrophic forgetting; curriculum learning strategies warrant further investigation.
- The 14M-scale DIM-T2I dataset still imposes considerable computational demands.
- Although L1 and CLIP-I metrics on the MagicBrush test set are competitive, the DINO metric underperforms certain methods, indicating room for improvement in fine-grained semantic preservation.
- Only single-round editing is currently supported; multi-round iterative editing (e.g., modifying background followed by foreground) remains to be explored.

## Related Work & Insights
- **MetaQuery**: Also a connector-based unified model, but employs a large 1.6B transformer connector; DIM achieves comparable performance with a two-layer MLP, suggesting the key bottleneck lies in data rather than connector architecture.
- **BAGEL**: A 14B integrated unified model whose editing performance is inferior to DIM's 4.6B, confirming that "bigger is not always better."
- **Step1X-Edit / UniWorld-V1**: State-of-the-art large-scale editing models (7B+12B backends), both surpassed by DIM with fewer parameters.
- **Janus-4o**: Trained on the same ShareGPT-4o-Image data but achieves an Overall score of only 3.19 (vs. DIM's 3.67), demonstrating that the gains from CoT blueprints do not derive from the data source itself.
- **InstructPix2Pix**: An early editing model; DIM's two-stage training strategy is inspired by its channel-concatenation design.
- **UltraEdit**: A large-scale AI editing dataset; DIM selects a high-consistency subset for Stage I training.
- **Insight**: As gains from architectural innovations plateau, **high-quality data design**—particularly CoT reasoning chains—may represent a more efficient breakthrough path. The "plan first, then execute" paradigm in human editing workflows can be directly translated into data design principles.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The "designer-painter" role separation insight is highly original and validated through data design rather than architectural changes.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark evaluation and multi-designer generalization tests are conducted, though ablation studies could be more thorough.
- Writing Quality: ⭐⭐⭐⭐⭐ — Logically clear, intuitively analogized, and well-illustrated.
- Value: ⭐⭐⭐⭐⭐ — Offers a fundamentally new perspective on image editing in unified models; dataset is open-sourced.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ConsistCompose: Unified Multimodal Layout Control for Image Composition](../../CVPR2026/image_generation/consistcompose_unified_multimodal_layout_control_for_image_composition.md)
- [\[ICLR 2026\] Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models](uni-x_mitigating_modality_conflict_with_a_two-end-separated_architecture_for_uni.md)
- [\[ICLR 2026\] Visual Autoregressive Modeling for Instruction-Guided Image Editing](visual_autoregressive_modeling_for_instruction-guided_image_editing.md)
- [\[CVPR 2026\] MICON-Bench: Benchmarking and Enhancing Multi-Image Context Image Generation in Unified Multimodal Models](../../CVPR2026/image_generation/micon-bench_benchmarking_and_enhancing_multi-image_context_image_generation_in_u.md)
- [\[ICLR 2026\] EditReward: A Human-Aligned Reward Model for Instruction-Guided Image Editing](editreward_a_human-aligned_reward_model_for_instruction-guided_image_editing.md)

<!-- RELATED:END -->
