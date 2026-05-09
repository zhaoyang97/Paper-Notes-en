---
title: >-
  [Paper Note] LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations
description: >-
  [AAAI 2026][Image Generation][Text-to-Image] This paper proposes LongT2IBench, the first evaluation benchmark targeting long-text-to-image (T2I) alignment, comprising 14K long-text–image pairs with graph-structured human annotations. It further introduces LongT2IExpert, an evaluator built by fine-tuning an MLLM via Hierarchical Alignment Chain-of-Thought (HA-CoT) instruction tuning, which jointly produces alignment scores and structured explanations.
tags:
  - AAAI 2026
  - Image Generation
  - Text-to-Image
  - Long Text Alignment
  - Graph-structured Annotation
  - Evaluation Benchmark
  - Multimodal Large Language Models
date: 2026-05-08
content_hash: 29b7e178d55e71d0
---

# LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations

**Conference**: AAAI 2026  
**arXiv**: [2512.09271](https://arxiv.org/abs/2512.09271)  
**Code**: [https://welldky.github.io/LongT2IBench-Homepage/](https://welldky.github.io/LongT2IBench-Homepage/)  
**Area**: Image Generation  
**Keywords**: Text-to-Image, Long Text Alignment, Graph-structured Annotation, Evaluation Benchmark, Multimodal Large Language Models

## TL;DR

This paper proposes LongT2IBench, the first evaluation benchmark targeting long-text-to-image (T2I) alignment, comprising 14K long-text–image pairs with graph-structured human annotations. It further introduces LongT2IExpert, an evaluator built by fine-tuning an MLLM via Hierarchical Alignment Chain-of-Thought (HA-CoT) instruction tuning, which jointly produces alignment scores and structured explanations.

## Background & Motivation

As text-to-image generation models are increasingly adopted in artistic creation and advertising design, user demand for **long-text T2I generation** has grown substantially. However, existing T2I alignment benchmarks suffer from severe limitations:

**Short-text bias**: Existing benchmarks (e.g., Pick-a-pic, HPDv2, TIFA) predominantly focus on short-prompt scenarios.

**Coarse-grained annotations**: Only MOS (Mean Opinion Score) or Likert-scale labels are provided, lacking interpretability.

**Challenges unique to long texts**:
   - **Detail Overload**: Annotators struggle to assign holistic alignment scores directly to long prompts.
   - **Alignment Complexity**: Element-level annotations fail to capture complex alignment relationships in long texts (e.g., links between distant elements).

These issues significantly impede the development of long T2I evaluators. The central motivation of this paper is to construct a benchmark that provides both quantitative scores and fine-grained interpretable alignment, thereby advancing the field of long T2I alignment evaluation.

## Method

### Overall Architecture

The work consists of two major components:

1. **LongT2IBench Benchmark Construction**: Data preparation → Data annotation (graph-structured conversion + alignment annotation) → Label generation.
2. **LongT2IExpert Evaluator**: An MLLM-based long T2I evaluation model, instruction-tuned via Hierarchical Alignment Chain-of-Thought.

### Key Designs

#### 1. **LongT2IBench Benchmark Construction**

**Data Preparation**:
- **Long prompt sources** (3K entries): Human-generated content (DiffusionDB), AI-generated content (GPT-4), and long image descriptions (DOCCI), ensuring diversity.
- **Balanced word-count distribution**: Uniform sampling across five intervals: 30–50, 50–70, 70–90, 90–110, and 110+.
- **T2I generation**: Images generated using six model categories — SD v3.5, PixArt-α (open-source baselines), DALL-E 3, Midjourney v6 (commercial), LongCLIP-SD, and LongSD (long-text specialist models).

**Graph-structured Annotation (Core Contribution)**:

A **Generate-Refine-Qualify** three-stage protocol is adopted:

1. **Generate**: GPT-4 converts long prompts into textual graph structures (containing Entity, Attribute, and Relation nodes).
2. **Refine**: Trained annotators refine the graph structures via additions, deletions, and modifications to ensure accuracy.
3. **Qualify**: Dual verification; only unanimously agreed conversions are retained.

From 4.5K initial long prompts, 3K accurately converted graphs are retained.

**Image–Text Graph Alignment Annotation**:
- Annotators perform binary alignment judgments on entities, attributes, and relations separately (E-Align, A-Align, R-Align).
- A hierarchical annotation logic is applied: entities are assessed first, and attributes and relations linked to unaligned entities are automatically filtered.
- Three independent annotators review each sample; majority agreement is required for retention.
- 14K image–text pairs are retained from an initial 18K, excluding NSFW content and severely distorted images.

**Label Generation**:
- **Alignment score**: Number of aligned elements / total elements (with weighted contributions from entities, attributes, and relations).
- **Alignment explanation**: Structured lists of aligned/unaligned elements generated from graph annotations.

#### 2. **LongT2IExpert Evaluator**

**Hierarchical Alignment Chain-of-Thought (HA-CoT)**: A three-hop reasoning process is designed to guide the MLLM in simulating human evaluation:

1. **First hop**: Entity aligner → analyzes alignment of all entities with the image.
2. **Second hop**: Attribute & Relation aligner → checks attributes and relations associated with aligned entities.
3. **Third hop**: Holistic assessment → produces an overall alignment score.

**Model Architecture**:
- Backbone: Qwen2.5-VL-7B-Instruct
- A `<<Level>>` token is added for numerical score output.
- A `<<Json>>` token is added for structured explanation output.
- LoRA (r=32, α=64) is used for parameter-efficient fine-tuning.

### Loss & Training

Multi-task training objective:

$$\mathcal{L} = \mathcal{L}_I + \lambda \cdot \mathcal{L}_S$$

- **Interpretation loss** $\mathcal{L}_I$: Cross-entropy loss supervising graph-structured explanation generation.
- **Scoring loss** $\mathcal{L}_S$: MSE loss, $\mathcal{L}_S = (\hat{y}_s - y_s)^2$, where $\hat{y}_s = \mathcal{R}(\hat{\mathbf{h}}_{Level})$.
- Hyperparameter $\lambda = 10$; trained for 3 epochs on A800 GPUs.

The scoring head consists of three linear layers, with learning rates of $5e^{-5}$ for LoRA parameters and $2e^{-4}$ for the scoring head.

## Key Experimental Results

### Main Results

**Long T2I alignment scoring comparison** (SRCC/PLCC across word-count intervals):

| Model | 30-50 SRCC | 50-70 SRCC | 70-90 SRCC | 110+ SRCC | Overall SRCC | Overall Avg |
|-----|-----------|-----------|-----------|----------|-------------|-------------|
| CLIPScore | 0.224 | 0.349 | 0.271 | 0.209 | 0.269 | 0.267 |
| HPSv2 | 0.540 | 0.479 | 0.394 | 0.148 | 0.381 | 0.387 |
| Q-Eval-Score | 0.470 | 0.460 | 0.339 | 0.422 | 0.361 | 0.358 |
| ImageReward* (fine-tuned) | 0.538 | 0.546 | 0.384 | 0.167 | 0.438 | 0.439 |
| **LongT2IExpert** | **0.781** | **0.605** | **0.548** | **0.431** | **0.558** | **0.557** |

**Long T2I alignment explanation comparison** (accuracy):

| Model | Entity Overall | Attribute Overall | Relation Overall | All Overall |
|-----|---------------|-------------------|-----------------|-------------|
| GPT-4o | 41.6% | 25.0% | 10.2% | 27.0% |
| Gemini-1.5-pro | 46.7% | 28.8% | 13.4% | 31.1% |
| Grok-3 | 43.1% | 21.7% | 7.4% | 25.7% |
| **LongT2IExpert** | **71.9%** | **47.3%** | **35.2%** | **53.2%** |

### Ablation Study

| Configuration | Alignment Score (Avg) | Alignment Explanation (Overall Acc) |
|------|--------------|---------------------|
| LongT2IExpert (w/o Score) | — | 32.8% |
| LongT2IExpert (w/o Interpretation) | 0.474 | — |
| LongT2IExpert (w/o HA-CoT) | 0.516 | 39.9% |
| **LongT2IExpert (full)** | **0.557** | **53.2%** |

### Key Findings

1. **Longer prompts yield lower alignment**: Alignment scores decrease substantially as prompt length increases.
2. **Relation alignment is the greatest challenge**: Among entities, attributes, and relations, relations exhibit the highest misalignment rate.
3. **Detecting misalignment is harder**: Model accuracy for identifying unaligned elements consistently falls below that for aligned elements.
4. **Multi-task training is mutually beneficial**: Joint training on scoring and explanation outperforms training each task independently (scoring: 0.474→0.557; explanation: 32.8%→53.2%).
5. **Effectiveness of HA-CoT**: Structured reasoning significantly improves alignment evaluation performance.
6. **Commercial model advantage**: DALL-E 3 and Midjourney v6 outperform open-source models on long-text alignment.

## Highlights & Insights

- **Pioneering work**: The first benchmark specifically targeting long T2I alignment, filling an important gap in the field.
- **Graph-structured annotation as the core contribution**: Decomposing long prompts into entity–attribute–relation graphs enables a leap from coarse-grained scoring to fine-grained interpretable evaluation.
- **Three-stage annotation ensures quality**: The Generate-Refine-Qualify protocol guarantees high fidelity of graph structures.
- **Unified scoring and explanation**: LongT2IExpert achieves both quantitative scoring and structured explanation within a single model.
- **Transferability of HA-CoT**: The hierarchical reasoning chain is extensible to other fine-grained multimodal evaluation tasks.

## Limitations & Future Work

- Annotation cost is high (filtered from 4.5K to 3K), limiting scalability.
- Only six T2I models are used for image generation, restricting coverage.
- Long-text alignment remains extremely challenging (even for human annotators), and current performance still has substantial room for improvement.
- Accuracy for relation alignment (especially spatial relations) remains low, requiring stronger model support.
- End-to-end methods for automated graph structure generation have not been explored.

## Related Work & Insights

This paper is closely related to the following directions:
- **T2I alignment evaluation**: The evolution from CLIPScore → PickScore/ImageReward → VQAScore/Q-Eval-Score.
- **Fine-grained evaluation**: TIFA (QA decomposition), EvalMuse-40K (phrase decomposition) → the graph-structured decomposition proposed in this paper.
- **Long-text T2I generation**: LongCLIP, LongAlign, and similar methods addressing CLIP token length limitations.

Insight: Graph-structured representations provide a powerful tool for fine-grained semantic analysis of text, and similar frameworks could be explored for video, 3D, and other modalities in future work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first long T2I alignment benchmark + graph-structured annotation)
- Experimental Thoroughness: ⭐⭐⭐⭐ (extensive comparisons + ablations + visualizations)
- Writing Quality: ⭐⭐⭐⭐ (clear structure, well-presented pipeline diagrams and statistical analyses)
- Value: ⭐⭐⭐⭐⭐ (provides foundational infrastructure for the long T2I evaluation field)

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Evaluating the Evaluators: Metrics for Compositional Text-to-Image Generation](../../NeurIPS2025/image_generation/evaluating_the_evaluators_metrics_for_compositional_text-to-image_generation.md)
- [\[AAAI 2026\] T2I-RiskyPrompt: A Benchmark for Safety Evaluation, Attack, and Defense on Text-to-Image Model](t2i-riskyprompt_a_benchmark_for_safety_evaluation_attack_and_defense_on_text-to-.md)
- [\[AAAI 2026\] Right Looks, Wrong Reasons: Compositional Fidelity in Text-to-Image Generation](right_looks_wrong_reasons_compositional_fidelity_in_text-to-image_generation.md)
- [\[CVPR 2026\] MultiBanana: A Challenging Benchmark for Multi-Reference Text-to-Image Generation](../../CVPR2026/image_generation/multibanana_a_challenging_benchmark_for_multi_reference_text_to_image_generation.md)
- [\[CVPR 2026\] Erasure or Erosion? Evaluating Compositional Degradation in Unlearned Text-To-Image Diffusion Models](../../CVPR2026/image_generation/erasure_or_erosion_evaluating_compositional_degradation_in_unlearned_text-to-ima.md)

<!-- RELATED:END -->
