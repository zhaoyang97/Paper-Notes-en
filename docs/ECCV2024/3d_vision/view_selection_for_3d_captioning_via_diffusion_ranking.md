---
title: >-
  [Paper Note] View Selection for 3D Captioning via Diffusion Ranking
description: >-
  [ECCV 2024][3D Vision][3D captioning] This work proposes DiffuRank, a method that leverages a pre-trained text-to-3D diffusion model (Shap·E) to score and rank alignment across rendered views of 3D objects, selecting the most representative top-6 views for GPT-4 Vision to generate high-quality captions. This refines approximately 200k incorrect annotations in Cap3D and expands the dataset to 1.5 million captions.
tags:
  - "ECCV 2024"
  - "3D Vision"
  - "3D captioning"
  - "diffusion ranking"
  - "view selection"
  - "hallucination"
  - "Cap3D"
date: 2026-05-08
content_hash: 53b990b39a4fd23b
---

# View Selection for 3D Captioning via Diffusion Ranking

**Conference**: ECCV 2024  
**arXiv**: [2404.07984](https://arxiv.org/abs/2404.07984)  
**Code**: [HuggingFace](https://huggingface.co/datasets/tiange/Cap3D)  
**Area**: 3D Vision / 3D-Text Alignment  
**Keywords**: 3D captioning, diffusion ranking, view selection, hallucination, Cap3D

## TL;DR

This work proposes DiffuRank, a method that leverages a pre-trained text-to-3D diffusion model (Shap·E) to score and rank alignment across rendered views of 3D objects, selecting the most representative top-6 views for GPT-4 Vision to generate high-quality captions. This refines approximately 200k incorrect annotations in Cap3D and expands the dataset to 1.5 million captions.

## Background & Motivation

**Background**: Cap3D provides 660th 3D-text pairs for the Objaverse dataset by rendering 3D objects into multi-view 2D images and generating captions using image captioning models, driving progress in fields such as text-to-3D, image-to-3D, and 3D LLM pre-training.

**Limitations of Prior Work**: (1) A large number of captions in Cap3D contain **hallucinated content**—such as misdescribing a green creature as a "baseball" or an interior design as a "cube". (2) The root cause lies in **accidental views**—fixed horizontal ring cameras inevitably sample projection positions with accidental features that lead to misleading interpretations. (3) These difficult views are challenging even for humans to describe accurately, causing image captioning models to fail easily, with errors being further cascaded and amplified during GPT-4 text summarization.

**Key Challenge**: 3D objects exhibit highly diverse geometries, making it impossible to determine the optimal views using simple geometric rules. However, inappropriate views systematically introduce caption hallucinations.

**Goal**: Automatically select the rendered views that best reflect the characteristics of 3D objects to reduce caption hallucinations and improve description quality.

**Key Insight**: Leveraging pre-trained text-to-3D diffusion models as a 3D prior, the alignment between text and 3D features can be evaluated via diffusion objective functions to rank the views.

**Core Idea**: Captions that are highly aligned with the characteristics of a 3D object should better guide a diffusion model to reconstruct the original 3D features. Thus, the magnitude of the diffusion loss can act as a metric to measure the representativeness of a view.

## Method

### Overall Architecture

3D object $\rightarrow$ Render 28 views (8 grey-background ray-tracing views + 20 transparent-background EEVEE views) $\rightarrow$ BLIP-2 generates 5 captions per view $\rightarrow$ DiffuRank uses Shap·E to score the alignment of captions for each view $\rightarrow$ Select Top-6 views $\rightarrow$ Feed into GPT-4 Vision to generate the final global caption.

### Key Designs

1. **DiffuRank Alignment Scoring**

    - For the caption $c_i^j$ of each view $I_i$, evaluate the conditional diffusion loss using pre-trained Shap·E: $\mathcal{L}_{c} = \|D_{\text{text-to-3D}}(\mathcal{O}_t | c) - \mathcal{O}_0\|$
    - Average the loss over multiple randomly sampled $\{t_k, \epsilon_k\}$ pairs to obtain the alignment score: $\text{Cor}(\mathcal{O}, c_i) = -\mathbb{E}_{j,k} \mathcal{L}_{c_i^j, k}$
    - Design Motivation: Captions highly aligned with the true features of the 3D object provide more effective diffusion guidance, leading to lower denoising loss. Therefore, a lower loss indicates a more representative view.

2. **Dual Rendering Strategy Fusion**

    - Cap3D rendering: 8 grey-background CYCLES views (fixed horizontal ring, default orientations)
    - Shap·E rendering: 20 transparent-background EEVEE views (randomly sampled after normalization)
    - Both background types have distinct advantages depending on the object. DiffuRank automatically selects the most suitable combination of views and backgrounds.
    - Design Motivation: Allowing the ranking algorithm to make decisions automatically is more robust than using hand-crafted rules.

3. **GPT-4 Vision Caption Generation**

    - The top-6 viewed images are directly fed into GPT-4 Vision to generate comprehensive captions.
    - Compared to Cap3D's "multi-caption to GPT-4 text summarization" pipeline, this reduces error cascading.
    - Design Motivation: Utilizing fewer but higher-quality views (6 vs. 28) actually produces more accurate and detailed descriptions.

### Loss & Training

DiffuRank itself requires no training and directly utilizes the pre-trained Shap·E diffusion objective for inference scoring. It adopts an $x_0$-prediction objective: $L_{3D} = \mathbb{E}\|\hat{x}_\theta(x_t, t) - x_0\|_2^2$. For 2D domain expansion, Stable Diffusion's $\epsilon$-prediction objective is used.

## Key Experimental Results

### Main Results

Human A/B evaluation and automatic metric evaluation on 5k Objaverse objects:

| Method | Quality Score↑ | Quality Win%↑ | Halluc. Score↑ | Halluc. Win%↑ | CLIP Score↑ | R@1↑ | R@5↑ |
|------|---------------|---------------|----------------|---------------|-------------|------|------|
| Human | 2.57 | 31.9% | 2.88 | 39.9% | 66.2 | 8.9 | 21.0 |
| Cap3D | 2.62 | 32.7% | 2.43 | 25.8% | 71.2 | 20.5 | 40.8 |
| **Ours (DiffuRank+GPT4V)** | **-** | **-** | **-** | **-** | **74.6** | **26.7** | **48.2** |
| All 28-views | 2.91 | 37.9% | 2.85 | 35.1% | 73.5 | 24.9 | 46.7 |
| Bottom 6-views | 2.74 | 31.1% | 2.61 | 30.1% | 72.8 | 24.6 | - |

In the A/B test, a score < 3 indicates that our method is preferred; Win% represents the percentage of cases favoring our method.

### Ablation Study

| View Selection Strategy | CLIP Score↑ | R@1↑ | Quality Win% |
|-------------|-------------|------|-------------|
| DiffuRank Top-6 | **74.6** | **26.7** | Reference Baseline |
| All 28-views | 73.5 | 24.9 | 43.6% |
| Horizontal 6-views | 73.8 | 25.8 | 44.5% |
| Bottom 6-views | 72.8 | 24.6 | 52.0% |

### Key Findings

- Using **fewer views (6 vs. 28)** yields higher-quality captions because misleading accidental views are filtered out.
- Compared to Cap3D, the hallucination reduction win rate increased from 25.8% to 63.9%, and the caption quality win rate increased from 32.7% to 60.2%.
- Expanding DiffuRank to the 2D domain of VQA tasks outperforms CLIP zero-shot performance.
- The dataset is expanded to 1.5 million captions, each accompanied by 16,384 colored point clouds and 20 rendered views.

## Highlights & Insights

- It is the first to leverage the denoising objective of diffusion models as a **cross-modal alignment metric** rather than a generative tool.
- "Less is more"—6 selected views outperform all 28 views, as validated by human evaluations.
- The method is highly generalizable: it can be expanded to text-to-2D diffusion models for 2D tasks such as VQA.
- Significant data contribution: 1.5 million 3D-text pairs, accompanied by point clouds and camera parameters, released under the ODC-By 1.0 open license.

## Limitations & Future Work

- The method relies on Shap·E as a 3D prior, and the quality of its encoder directly impacts ranking accuracy.
- The API cost and latency of GPT-4 Vision limit large-scale practical deployment.
- More highly efficient view selection strategies (such as active learning or greedy search) have not yet been explored.
- Ethical content filtering still relies on GPT-4 Vision's internal detection mechanism, which may have missed cases.

## Related Work & Insights

- **vs. Cap3D**: This work presents an improved version of Cap3D, addressing caption hallucinations caused by accidental views through DiffuRank.
- **vs. CLIP Score**: CLIP only measures 2D image-text similarity, whereas DiffuRank utilizes 3D priors to model 3D alignment.
- **vs. SDS**: DiffuRank is conceptually related to Score Distillation Sampling (SDS) but evaluates using loss values for scoring instead of gradient optimization.
- Insight: The methodology of utilizing generative models as quality evaluators can be transferred to other modalities.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using a diffusion objective function as an alignment metric is simple yet elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive A/B human evaluations, automatic metric evaluations, ablation studies, and 2D extensions.
- Writing Quality: ⭐⭐⭐⭐ The motivation is very clear, and the methodology exposition is highly intuitive.
- Value: ⭐⭐⭐⭐ The 1.5 million caption dataset is a significant contribution to the 3D-text community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Bi-directional Contextual Attention for 3D Dense Captioning](bi-directional_contextual_attention_for_3d_dense_captioning.md)
- [\[ECCV 2024\] MVDD: Multi-View Depth Diffusion Models](mvdd_multi-view_depth_diffusion_models.md)
- [\[ECCV 2024\] ShapeFusion: A 3D Diffusion Model for Localized Shape Editing](shapefusion_a_3d_diffusion_model_for_localized_shape_editing.md)
- [\[ECCV 2024\] Transferable 3D Adversarial Shape Completion using Diffusion Models](transferable_3d_adversarial_shape_completion_using_diffusion_models.md)
- [\[ECCV 2024\] MVDiffusion++: A Dense High-Resolution Multi-View Diffusion Model for Single or Sparse-View 3D Object Reconstruction](mvdiffusion_a_dense_high-resolution_multi-view_diffusion_model_for_single_or_spa.md)

</div>

<!-- RELATED:END -->
