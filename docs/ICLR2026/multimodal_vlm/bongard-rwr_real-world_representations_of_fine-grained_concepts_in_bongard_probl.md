---
title: >-
  [Paper Note] Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems
description: >-
  [ICLR 2026][Multimodal VLM][Bongard problems] Bongard-RWR+ is a benchmark comprising 5,400 Bongard problems, constructed via a VLM-based pipeline (Pixtral-12B + Flux.1-dev) that automatically generates photorealistic images to represent abstract concepts. Systematic evaluation reveals that state-of-the-art VLMs struggle to discriminate fine-grained visual concepts such as contour, rotation, and angle, with accuracy dropping as low as 19%.
tags:
  - ICLR 2026
  - Multimodal VLM
  - Bongard problems
  - abstract visual reasoning
  - few-shot learning
  - VLM benchmark
  - fine-grained concepts
date: 2026-05-08
content_hash: d245022a5c5b705c
---

# Bongard-RWR+: Real-World Representations of Fine-Grained Concepts in Bongard Problems

**Conference**: ICLR 2026
**arXiv**: [2508.12026](https://arxiv.org/abs/2508.12026)
**Code**: Available
**Area**: Multimodal VLM
**Keywords**: Bongard problems, abstract visual reasoning, few-shot learning, VLM benchmark, fine-grained concepts

## TL;DR
Bongard-RWR+ is a benchmark comprising 5,400 Bongard problems, constructed via a VLM-based pipeline (Pixtral-12B + Flux.1-dev) that automatically generates photorealistic images to represent abstract concepts. Systematic evaluation reveals that state-of-the-art VLMs struggle to discriminate fine-grained visual concepts such as contour, rotation, and angle, with accuracy dropping as low as 19%.

## Background & Motivation
**Background**: Bongard Problems (BPs) are a classic test of abstract visual reasoning — given six images on each side, the task is to identify the abstract concept that distinguishes the two groups. Existing BP datasets are either synthetic black-and-white images (Bongard-LOGO) or use real images to represent coarse-grained concepts (e.g., "a person driving").

**Limitations of Prior Work**: Although Bongard-RWR uses real images to represent fine-grained abstract concepts, it is hand-constructed with only 60 instances — a scale too small for robust evaluation. Moreover, no systematic diagnosis of VLM capabilities across different reasoning dimensions exists.

**Key Challenge**: VLMs perform reasonably well on coarse-grained concept recognition, but their ability to handle fine-grained abstract concepts (e.g., "arrows pointing in the same vs. different directions") remains unknown, necessitating a sufficiently large benchmark for systematic testing.

**Goal**: How can Bongard problems involving fine-grained abstract concepts be constructed at scale with photorealistic imagery? And how can the reasoning boundaries of VLMs be systematically evaluated?

**Key Insight**: A semi-automatic pipeline of I2T (image captioning) → T2T (description augmentation) → T2I (image generation) → human verification is employed to scale 60 Bongard-RWR instances up to 5,400.

**Core Idea**: A VLM-based pipeline is used to automatically generate photorealistic images for Bongard problems, enabling large-scale evaluation of VLMs' fine-grained abstract reasoning limits.

## Method

### Overall Architecture
Bongard-RWR+ is a benchmark paper rather than a methodology paper. Its core contributions are a semi-automatic data construction pipeline and a multi-task, multi-dimensional evaluation framework. The benchmark consists of 49 abstract concepts × 100 matrix variants = 5,400 BPs.

### Key Designs

1. **Semi-Automatic Image Generation Pipeline**:

    - **Function**: Starting from hand-crafted BP images, generates large quantities of new photorealistic images representing the same abstract concepts.
    - **Mechanism**: (1) Pixtral-12B generates a positive description (capturing the target concept) and a negative description (suppressing the opposing concept) for each image; (2) a T2T model augments each positive description into 15 diverse variants; (3) Flux.1-dev generates candidate images from these descriptions; (4) human verification ensures concept fidelity. Diversity-maximizing selection is performed using pairwise cosine similarity of ViT-L/14 embeddings.
    - **Design Motivation**: Manual construction does not scale. Automated generation must ensure concept fidelity — positive/negative descriptions prevent T2I models from conflating opposing concepts.

2. **Multi-Task Evaluation Framework (6 Task Types)**:

    - **Function**: Systematically evaluates VLMs from easy to difficult settings.
    - **I1S/I2S**: Single/dual test-image binary classification (assigning images to the left or right group).
    - **D1S/D2S**: Classification after converting images to text descriptions via I2T (testing the effect of intermediate steps).
    - **CS**: Selecting the correct concept from $K$ candidates ($K = 2, 4, 8, 16$).
    - **CG**: Free-text generation of the correct concept description.

3. **Semantic Grouping Analysis of Concepts**:

    - **Function**: Groups the 49 concepts into 9 semantic categories (Size, Position, Count, Branching, Similarity, Contour, Shape, Rotation, Angle).
    - **Design Motivation**: Pinpoints specific weaknesses of VLMs — identifying which abstract concept categories are most challenging and which are more tractable.

### Loss & Training
N/A (benchmark paper; existing models are evaluated without training).

## Key Experimental Results

### Main Results (Concept Selection Task)

| Model | K=2 | K=4 | K=8 | K=16 |
|-------|-----|-----|-----|------|
| InternVL2.5-78B | **91%** | **78%** | **68%** | **57%** |
| Qwen2-VL-72B | 85% | 65% | 48% | 33% |
| LLaVA-Next-110B | 73% | 45% | 30% | 19% |
| MiniCPM-o-8B | 72% | 44% | 28% | 19% |

### Binary Classification Tasks (I1S/I2S)

| Model | I1S | I2S | D1S | D2S |
|-------|-----|-----|-----|-----|
| InternVL2.5-78B | 0.50 | 0.39 | 0.57 | 0.49 |
| Qwen2-VL-72B | 0.49 | 0.44 | 0.58 | 0.42 |
| Random Baseline | 0.50 | 0.50 | 0.50 | 0.50 |

### Key Findings
- **Binary classification is near chance**: All VLMs achieve approximately 50% accuracy on I1S/I2S, equivalent to random guessing, demonstrating that VLMs are nearly incapable of inferring fine-grained abstract concepts from few-shot images.
- **Concept selection degrades rapidly**: InternVL2.5 achieves 91% at $K=2$ (demonstrating some discriminative ability), but collapses to 57% at $K=16$ as the number of distractors increases.
- **Significant variation across semantic groups**: Shape, Size, and Branching are relatively easy (~75%), whereas Contour, Rotation, and Angle are difficult (<50%) — the latter require precise spatial relational reasoning.
- DeepSeek-R1 achieves 0.56 on the text-only D2S task, suggesting that textual reasoning is more effective than visual reasoning — the bottleneck for VLMs lies in visual perception rather than reasoning.
- Color vs. grayscale input shows no significant difference, confirming that the target concepts are structural and color-independent.
- Small models (MiniCPM-8B) and large models (LLaVA-110B) achieve comparable performance, indicating that model scale is not a determining factor.

## Highlights & Insights
- **Reveals a fundamental weakness of VLMs**: Even the strongest 78B-parameter VLMs perform near chance on few-shot abstract visual reasoning — a failure mode unlikely to be resolved through scaling alone.
- **Methodological value of the semi-automatic pipeline**: The I2T → T2T → T2I → human verification workflow is reusable for other scenarios requiring large-scale conceptual datasets.
- **Comprehensive multi-task evaluation design**: Progressing from binary classification to multi-way selection to free-form generation, the framework enables precise localization of capability boundaries.

## Limitations & Future Work
- Concept fidelity of generated images still requires human verification, preventing full automation.
- The 49 concepts covered represent a limited subset of the 394 concepts in the original Bongard problems.
- Evaluation is restricted to zero-shot/few-shot VLMs; whether fine-tuning could yield improvements remains untested.
- Generated images may contain T2I model artifacts that interfere with concept judgment.

## Related Work & Insights
- **vs. Bongard-LOGO**: LOGO contains 12K instances but relies entirely on synthetic black-and-white images; RWR+ provides 5.4K instances with photorealistic imagery, more closely aligned with VLM training distributions.
- **vs. Bongard-HOI/OpenWorld**: These benchmarks use coarse-grained concepts (e.g., "a person driving"), on which VLMs perform relatively well; RWR+ employs fine-grained abstract concepts that expose genuine VLM weaknesses.
- **vs. ARC (Chollet)**: ARC similarly tests abstract reasoning but in a grid domain; RWR+ operates in the real-image domain, making the two complementary.

## Rating
- Novelty: ⭐⭐⭐⭐ Semi-automatic generation pipeline + multi-dimensional evaluation framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four large models, six task types, nine semantic groups, and extensive ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure and comprehensive evaluation
- Value: ⭐⭐⭐⭐ Establishes the capability ceiling and bottlenecks of VLMs in fine-grained reasoning

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Can Vision-Language Models Answer Face to Face Questions in the Real-World?](can_vision-language_models_answer_face_to_face_questions_in_the_real-world.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](../../NeurIPS2025/multimodal_vlm/focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[CVPR 2026\] Towards Real-World Document Parsing via Realistic Scene Synthesis and Document-Aware Training](../../CVPR2026/multimodal_vlm/towards_real-world_document_parsing_via_realistic_scene_synthesis_and_document-a.md)
- [\[CVPR 2026\] MA-Bench: Towards Fine-grained Micro-Action Understanding](../../CVPR2026/multimodal_vlm/ma-bench_towards_fine-grained_micro-action_understanding.md)
- [\[CVPR 2026\] Zina: Multimodal Fine-grained Hallucination Detection and Editing](../../CVPR2026/multimodal_vlm/zina_multimodal_fine-grained_hallucination_detection_and_editing.md)

<!-- RELATED:END -->
