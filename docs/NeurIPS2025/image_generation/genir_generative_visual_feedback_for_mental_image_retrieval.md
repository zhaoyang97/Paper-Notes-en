---
title: >-
  [Paper Note] GenIR: Generative Visual Feedback for Mental Image Retrieval
description: >-
  [NeurIPS 2025][Image Generation][Interactive Retrieval] This paper proposes GenIR, a multi-round interactive image retrieval framework that leverages text-to-image diffusion models to generate "synthetic visual feedback," explicitly visualizing the system's interpretation of the user's query. This enables users to intuitively identify discrepancies and iteratively refine their queries, achieving substantial improvements over text-only feedback methods on the Mental Image Retr…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Interactive Retrieval"
  - "Visual Feedback"
  - "Diffusion Models"
  - "Mental Image Retrieval"
  - "Multi-round Query Refinement"
date: 2026-05-08
content_hash: 9283e6400cba5f1f
---

# GenIR: Generative Visual Feedback for Mental Image Retrieval

**Conference**: NeurIPS 2025
**arXiv**: [2506.06220](https://arxiv.org/abs/2506.06220)  
**Code**: [mikelmh025/generative_ir](https://github.com/mikelmh025/generative_ir)  
**Area**: Image Retrieval / Image Generation
**Keywords**: Interactive Retrieval, Visual Feedback, Diffusion Models, Mental Image Retrieval, Multi-round Query Refinement

## TL;DR

This paper proposes GenIR, a multi-round interactive image retrieval framework that leverages text-to-image diffusion models to generate "synthetic visual feedback," explicitly visualizing the system's interpretation of the user's query. This enables users to intuitively identify discrepancies and iteratively refine their queries, achieving substantial improvements over text-only feedback methods on the Mental Image Retrieval (MIR) task.

## Background & Motivation

While vision-language models (VLMs) perform impressively on standard text-to-image retrieval benchmarks, a significant gap remains in real-world applications. Authentic human search behavior exhibits two key characteristics:

**Non-one-shot**: Search is an iterative, multi-round process in which users continuously revise queries based on intermediate results.

**Mental-image-driven**: Users are typically re-finding images they have previously seen, relying on memory representations that range from vague to clear—so-called "mental images."

Existing interactive retrieval methods (ChatIR, PlugIR) share a fundamental limitation: **feedback is confined to text**. This indirect, abstract linguistic feedback underperforms in the following respects:

- **Ambiguity**: Textual descriptions struggle to convey precise visual details. For example, asking "Is he wearing a hat?" and receiving "No"—when the subject is actually wearing a helmet—introduces semantic drift that misleads subsequent queries.
- **Unpredictability**: In vision-language embedding spaces such as CLIP, minor textual modifications can produce drastically different retrieval results.
- **Opacity**: The system's internal interpretation of the query—its "visual belief"—is entirely invisible to the user, reducing query refinement to a blind trial-and-error process.

## Core Problem

This paper formally defines the **Mental Image Retrieval (MIR)** task: the user holds a target image in mind (a mental image) and collaborates with an image search engine through multi-round interaction to locate it. MIR is a subtask of interactive text-to-image retrieval, focusing on Known-Item Search (where the user has previously seen the target) rather than exploratory search.

The central challenge is: **how to provide users with clear, interpretable, and actionable feedback that enables effective query improvement?**

## Method

### GenIR Framework Overview

GenIR employs a simple yet powerful iterative pipeline. Each round of interaction comprises four steps:

**Step 1: Query Construction**
The user constructs a text query $q_t$ describing the mental image, and is encouraged to include both high-level descriptions (scene type, overall composition) and fine-grained attributes (colors, object details).

**Step 2: Synthetic Image Generation**
Given query $q_t$, the image generator $G$ produces a synthetic image:
$$I_t^{\text{synthetic}} = G(q_t)$$
This image serves as an **explicit visualization** of the system's understanding of the query—projecting the query's representation in the vision-language latent space into a form that humans can directly perceive.

**Step 3: Image-to-Image Retrieval**
An image encoder (e.g., the CLIP image encoder) embeds both the synthetic image and database images into a shared visual feature space, and retrieval is performed via cosine similarity:
$$I_t^{\text{retrieved}} = \arg\max_{I \in \mathcal{N}} \text{cosine}(\phi(I_t^{\text{synthetic}}), \phi(I))$$
where $\phi$ denotes the image encoder. This transforms retrieval from cross-modal matching (text → image) to same-modal matching (image → image).

**Step 4: Visual Feedback Loop**
The user compares the synthetic image $I_t^{\text{synthetic}}$ against their mental image, identifies discrepancies (missing elements, incorrect attributes, style deviations), and uses these observations to refine the next query $q_{t+1}$.

### Core Advantages of Visual Feedback

The key innovation of GenIR is **externalizing** the system's internal understanding through generated images:

1. **Disambiguation**: Users directly see "what the system thinks you are looking for," rather than guessing how a text query is being interpreted.
2. **Same-modal matching**: Image-to-image retrieval captures spatial relationships and visual attributes that are difficult to express precisely in text.
3. **Model-agnostic**: The framework is compatible with any text-to-image generator (diffusion models, GANs, etc.) and any image retrieval model.

### Dataset Construction Pipeline

GenIR also provides an **automated dataset annotation pipeline**:
1. A VLM generates an initial query $q_0$ from the target image.
2. Each round: generate a synthetic image → retrieve → annotate a correctness label $y_t$ → the VLM refines the query based on observed differences between the target and the synthetic image.
3. Data tuples $(q_t, I_t^{\text{synthetic}}, I_t^{\text{retrieved}}, y_t)$ are stored in the dataset.

## Key Experimental Results

### Datasets and Setup

Evaluation is conducted on 4 cross-domain datasets with varying search space sizes:
- **MS COCO**: 50K validation images, everyday scenes
- **FFHQ**: 70K high-quality face images
- **Flickr30k**: 32K diverse real-world photos
- **Clothing-ADC**: Over 1 million clothing images (12,000 subcategories)

Gemma3 (4B and 12B) is used as the VLM, and 5 diffusion models are tested (Infinity, Lumina-Image-2.0, SD 3.5, FLUX.1, HiDream-I1).

### Main Results (Hits@10)

**MSCOCO (50K search space)**:

| Method | Initial Round | Round 10 |
|--------|--------------|----------|
| ChatIR (text feedback) | ~60% | ~73% |
| Verbal Feedback + Gemma3-12b | — | ~92% |
| Prediction Feedback | — | ~92% |
| **GenIR (Infinity)** | **~90%** | **~98%** |

GenIR reaches approximately 90% at the initial round, far surpassing the final-round performance of all baseline methods.

**Cross-domain (Hits@10, Round 10)**:
- **FFHQ**: GenIR 70% vs. second-best 52% (+18%)
- **Clothing-ADC**: GenIR 73% vs. second-best 50% (+23%, search space >1M)
- **Flickr30k**: GenIR maintains a consistent advantage of 8–15%

### Dataset Quality Validation

Even when using only the **text queries** generated by the GenIR pipeline for text-to-image retrieval (without synthetic images), Round 10 achieves 92.33%, far exceeding ChatIR's 73.64%. This demonstrates that the visual feedback mechanism produces higher-quality query annotations.

### Model Scale Analysis

GenIR + Gemma3-4B consistently outperforms both Prediction Feedback and Verbal Feedback + Gemma3-12B, indicating that the advantage of visual feedback is independent of model scale and permits more efficient deployment.

## Highlights & Insights

1. **Clear problem formulation**: The paper formally defines the MIR task, filling a research gap around the "mental image" scenario in interactive retrieval.
2. **Remarkably simple core idea**: The central insight—"show the user what the system understands"—can be stated in a single sentence, yet yields striking results.
3. **Generator-agnostic superiority**: Even with the weakest generator (HiDream), GenIR significantly outperforms all text-only methods, demonstrating a paradigm-level advantage rather than a model-level one.
4. **Strong cross-domain robustness**: The method performs well across highly diverse domains including faces, clothing, and everyday scenes.
5. **Dual contribution**: The paper contributes both a framework and a multi-round dataset with an automated annotation pipeline.

## Loss & Training

### Diffusion Model Inference Hyperparameters

| Model | Inference Steps | Guidance Scale | Resolution |
|-------|----------------|---------------|------------|
| Infinity | N/A | 3.0 | 1024×1024 |
| Lumina-Image-2.0 | 50 | 4.0 | 1024×1024 |
| Stable Diffusion 3.5 | 28 | 3.5 | 1024×1024 |
| FLUX.1 | 5 | 3.5 | 1024×1024 |
| HiDream-I1-Fast | 16 | 0.0 | 1024×1024 |

- Image retrieval uniformly uses BLIP-2 (feature dimension 256, L2 normalization, cosine similarity).
- VLM (Gemma3-4B/12B): temperature=0.7, top-p=0.9, max_tokens=500, repetition_penalty=1.1.
- Experimental platform: 4× NVIDIA A6000 (48GB); full experiments require approximately 200 GPU hours.

### Computational Cost vs. Performance

| Method | Per-round Latency (s) | Relative GPU Cost | Hits@10 (Round 5) |
|--------|----------------------|------------------|-------------------|
| Verbal Feedback (Gemma3-12b) | 2 | 1.0× | 89.97% |
| Prediction Feedback | 2.5 | 1.2× | 90.70% |
| GenIR (FLUX.1) | 12 | 2.5× | 95.10% |
| GenIR (Infinity) | 16 | 3.0× | 96.85% |
| GenIR (SD 3.5) | 26 | 2.2× | 96.02% |
| GenIR (Lumina) | 27 | 1.3× | 96.55% |

GenIR (Infinity) requires approximately 16 seconds per round—8× slower than Verbal Feedback—but yields an absolute Hits@10 improvement of +6.9%.

### Hybrid Strategy: Balancing Performance and Efficiency

The paper also explores a **hybrid approach** that applies visual feedback to 22.3% of queries and verbal feedback to the rest:
- **Hybrid Oracle** (perfect selection of when to use visual feedback): achieves 98.30% at Round 5, +1.5% over pure visual and +8.35% over pure verbal.
- **Random Select** (randomly applying visual feedback to 22.3% of queries): achieves 91.50% at Round 5, +1.54% over pure verbal.

This indicates that even partial use of visual feedback yields significant gains, and suggests that a future router model could strategically select the feedback type.

## Human Evaluation

- 100 samples, evaluated by 1 annotator, assessing whether the Round 9 synthetic image aids query refinement.
- **86% of synthetic images were judged to be helpful**.
- Visual feedback is particularly effective for refining fine-grained attributes (color, texture, spatial relationships).
- Failure cases are primarily attributable to severe distortion or misinterpretation in the generated images.

## Limitations & Future Work

1. **VLM simulation vs. real users**: Experiments simulate users with Gemma3, assuming a fixed and clear target image. Real users' mental images are often vague and dynamically evolving.
2. **Dynamic nature of mental images**: The paper does not account for the possibility that the search process itself alters the user's memory—retrieval behavior may help users clarify their own recollections.
3. **Three failure modes** (analyzed in detail in the appendix):
    - **Limited Improvement**: In later rounds (7–10), generated images change minimally, indicating insufficient sensitivity to subtle query modifications—e.g., Round 8 and Round 9 images are nearly identical.
    - **Hallucination Content**: Diffusion models tend to "complete" scenes with commonly co-occurring objects, introducing elements absent from the query (e.g., adding a showerhead to a bathroom scene). This is the most harmful failure mode.
    - **Retrieval-Detail Misalignment**: Visually acceptable differences (bench → chair) may be critical distinguishing features in the retrieval space, motivating retrieval-aware generation objectives.
4. **Limited human evaluation scale**: Only 100 samples with a single annotator, conducted under conditions more controlled than real-world search.
5. **Computational overhead**: Each round requires one diffusion model inference (~16–27 seconds per image), posing challenges for real-time applications. The original HiDream-I1 requires 55GB VRAM; 4-bit quantization was used to reduce this below 30GB.

## Related Work & Insights

| Method | Feedback Type | Multi-round | Utilizes Image Space | Core Limitation |
|--------|--------------|-------------|---------------------|-----------------|
| ChatIR | Pure text Q&A | ✓ | ✗ | Redundant/misleading feedback; no visual information |
| PlugIR | Text + retrieval result descriptions | ✓ | Indirect (captioning) | Remains at the linguistic level |
| Imagine-and-Seek | Generated proxy image | ✗ (single-round) | ✓ | No iterative refinement capability |
| **GenIR** | **Synthetic image** | **✓** | **✓ (direct generation)** | Computational cost; generator hallucinations |

GenIR's innovation lies in being the **first to integrate text-to-image generation into an interactive retrieval loop**, realizing a closed-loop system that unifies generation, retrieval, and feedback.

**Broader Implications**:

1. **Generality of the "show it to you" paradigm**: The visual feedback approach can be extended to other retrieval tasks (video retrieval, 3D model retrieval); the core principle is externalizing the system's implicit understanding into a form directly perceivable by users.
2. **Generative models as retrieval middleware**: Rather than serving solely as output mechanisms, generative models can act as mediators for query expression and user interaction—opening a new research direction.
3. **Cross-modal → same-modal**: The technique of converting text-to-image retrieval into image-to-image retrieval via an intermediate generation step merits attention and may offer insights for other cross-modal tasks.
4. **Potential integration with RLHF**: Future work could apply reinforcement learning to optimize the generator, producing images better suited for retrieval feedback rather than optimizing purely for visual quality.

## Rating
- Novelty: ⭐⭐⭐⭐ (Novel task formulation; method is simple but grounded in strong insight)
- Experimental Thoroughness: ⭐⭐⭐⭐ (4 cross-domain datasets, multi-generator comparison, human evaluation included)
- Writing Quality: ⭐⭐⭐⭐ (Clear motivation, thorough comparisons, effective visualizations)
- Value: ⭐⭐⭐⭐ (Opens a new direction in generative visual feedback retrieval; strong practical relevance)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Instance-Level Composed Image Retrieval](instance-level_composed_image_retrieval.md)
- [\[CVPR 2026\] Diffusion Mental Averages](../../CVPR2026/image_generation/diffusion_mental_averages.md)
- [\[NeurIPS 2025\] UniLumos: Fast and Unified Image and Video Relighting with Physics-Plausible Feedback](unilumos_fast_and_unified_image_and_video_relighting_with_physics-plausible_feed.md)
- [\[CVPR 2025\] Generative Image Layer Decomposition with Visual Effects](../../CVPR2025/image_generation/generative_image_layer_decomposition_with_visual_effects.md)
- [\[ECCV 2024\] IRGen: Generative Modeling for Image Retrieval](../../ECCV2024/image_generation/irgen_generative_modeling_for_image_retrieval.md)

</div>

<!-- RELATED:END -->
