---
title: >-
  [Paper Note] Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding
description: >-
  [ICLR 2026][Video Understanding][Category Splitting] This paper introduces a new task called *Category Splitting*, which exploits latent compositional structure embedded in video classifier weights to decompose coarse-gr…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "Category Splitting"
  - "Zero-Shot Editing"
  - "Fine-Grained Video Recognition"
  - "Classifier Modification"
  - "Compositional Structure"
date: 2026-05-08
content_hash: 99129099c23c2dfe
---

# Let's Split Up: Zero-Shot Classifier Edits for Fine-Grained Video Understanding

**Conference**: ICLR 2026
**arXiv**: [2602.16545](https://arxiv.org/abs/2602.16545)  
**Code**: [Available](https://kaitingliu.github.io/Category-Splitting/)  
**Area**: Video Understanding
**Keywords**: Category Splitting, Zero-Shot Editing, Fine-Grained Video Recognition, Classifier Modification, Compositional Structure

## TL;DR

This paper introduces a new task called *Category Splitting*, which exploits latent compositional structure embedded in video classifier weights to decompose coarse-grained action categories into fine-grained subcategories under zero-shot conditions, without retraining or additional data.

## Background & Motivation

Video recognition models are typically trained on fixed taxonomies that tend to be overly coarse. For example, a single label "open" may encompass vastly different scenarios such as "open cupboard," "open by pushing," "open quickly," and "open halfway." As application requirements evolve, finer-grained distinctions become necessary.

Existing solutions suffer from three categories of limitations:
- **Re-annotation + retraining**: Expensive, requiring large-scale labeled data and full training cycles.
- **Vision-language models (VLMs)**: Rely on massive video-text corpora; domain-specific data is scarce, and fine-grained temporal cues are difficult to capture.
- **Continual learning**: Requires training data for each new category and focuses on entirely new categories rather than refinement of existing ones.

Core insight: Modern video backbones already encode rich latent structure in their feature spaces, which can be decomposed to distinguish fine-grained variations even without direct supervision signals.

## Method

### Overall Architecture

The core of the approach is to **edit only the classification head** while keeping the backbone frozen, splitting a coarse category $c$ into multiple fine-grained subcategories $\mathcal{S}^c = \{s_1^c, s_2^c, \dots, s_k^c\}$. The updated label space becomes $\mathcal{Y}' = (\mathcal{Y} \setminus \{c\}) \cup \mathcal{S}^c$.

The editing method $E$ must satisfy two properties:
- **Generality**: The edit should correctly classify previously unseen samples of new subcategories.
- **Locality**: The edit should leave predictions for other categories unaffected.

### Key Designs

#### 1. Modifier Retrieval

Each fine-grained subcategory is treated as a composition of a "coarse concept + modifier." For example, "pushing left to right" = "pushing" + "left to right." The key steps are:

**Modifier dictionary construction**: Fine-grained categories already present in the classifier are grouped to identify "pseudo-coarse categories" $\tilde{c}$ that share a common base concept. Their weight vectors are approximated as the mean of subcategory weights:

$$v_{\tilde{c}} = \frac{1}{|\mathcal{S}^{\tilde{c}}|} \sum_{y \in \mathcal{S}^{\tilde{c}}} w_y$$

The modifier vector is obtained by subtracting the shared base: $v_m = w_y - v_{\tilde{c}}$

**Modifier vector transfer**: For target subcategories, cosine similarity of modifier text embeddings is computed via a text encoder $\phi$, incorporating both modifier similarity and full-label similarity for retrieval:

$$v_m^* = \arg\max_{(t_y, t_m, v_m) \in \mathcal{M}_{mod}} \text{sim}(\phi(t_y), \phi(t_s^*)) + \text{sim}(\phi(t_m), \phi(t_m^*))$$

The weight vector for the new subcategory is then: $w_{s_j^c} = w_c + v_m^*$

#### 2. Modifier Alignment

To handle novel modifiers absent from the dictionary, a lightweight alignment module $g_\psi: \mathbb{R}^n \to \mathbb{R}^m$ is trained to directly map text embeddings into the classifier weight space.

Training data are drawn from two sources:
- **Modifier-level pairs**: $\mathcal{D}_{mod} = \{(\phi(t_m), v_m)\}$
- **Category-level pairs**: $\mathcal{D}_{cat} = \{(\phi(t_y), w_y)\} \cup \{(\phi(t_{\tilde{c}}), v_{\tilde{c}})\}$

The module is trained with MSE loss and is implemented as a single-hidden-layer MLP (384-dimensional). Only $\psi$ is updated; the classifier and text encoder remain frozen. The entire process remains zero-shot, requiring no video data.

#### 3. Low-Shot Category Splitting

When a small number of annotated samples is available (as few as one video per subcategory), an **isolated fine-tuning** strategy is employed: only the newly added subcategory weights $\theta_{head}'$ are fine-tuned, while the backbone and original classification head are frozen. Initializing from zero-shot method weights outperforms initialization from coarse-category weights.

### Loss & Training

- Zero-shot stage requires no training data.
- Alignment module: MSE loss, AdamW optimizer, learning rate $1 \times 10^{-3}$, cosine annealing, batch size 10.
- Low-shot fine-tuning: cross-entropy loss, AdamW, learning rate $1 \times 10^{-3}$, weight decay $1 \times 10^{-3}$, batch size 16.
- EMA early stopping ($\beta=0.95$, patience=5, $\delta=1\times10^{-3}$).

## Key Experimental Results

### Main Results

Baseline setup: ViT-Small + MVD pretraining, CLIP ViT-L/14 text encoder. Datasets: SSv2-Split (54 coarse categories) and FineGym-Split (42 coarse categories).

| Method | SSv2-A Gen. | SSv2-A Loc. | FineGym-A Gen. | FineGym-A Loc. |
|------|:-:|:-:|:-:|:-:|
| CLIP | 27.6 | 100.0 | 12.1 | 100.0 |
| FG-CLIP | 30.9 | 100.0 | 19.4 | 100.0 |
| VideoPrism | 28.2 | 100.0 | 21.7 | 100.0 |
| **Ours** | **46.3** | **98.9** | **34.2** | **97.8** |

VLMs exhibit extremely low generality, while the proposed method improves generality on SSv2 by nearly 20 percentage points.

| Fine-tuning Strategy | Initialization | Generality | Locality | Mean |
|---------|-------|:-:|:-:|:-:|
| Full model one-shot | Coarse category | 33.6 | 0.0 | 16.8 |
| $\theta_{head}'$ only one-shot | Coarse category | 48.4 | 98.4 | 73.4 |
| $\theta_{head}'$ only one-shot | Modifier alignment | **52.8** | **98.2** | **75.5** |
| Full-data fine-tuning | Coarse category | 86.7 | 19.2 | 52.9 |

### Ablation Study

- Modifier Retrieval (45.0%) vs. Modifier Alignment (46.3%): alignment improves generality by 1.3%.
- Zero-shot initialization vs. random initialization: +7.8% generality improvement.
- Pretraining impact: MVD (46.3%) > SIGMA (44.1%) > VideoMAE (42.9%) > training from scratch (37.0%).
- Text encoder: CLIP (46.3%) ≈ VideoPrism (46.5%) > RoBERTa (40.9%).

### Key Findings

- Direction-based splits achieve the best performance; splits involving object count, intent, or success are the most challenging.
- Full-data fine-tuning performs worse than one-shot fine-tuning (75.5 vs. 52.9), as it introduces strong bias toward new categories and severely degrades locality.
- Performance is better when the original label space contains analogous categories sharing the same modifier, though the method remains effective even without them.

## Highlights & Insights

1. **Novel task formulation**: Category splitting addresses a natural yet overlooked real-world problem.
2. **Intrinsic structure outperforms VLMs**: Exploiting the weight structure of video classifiers substantially surpasses VLMs for fine-grained video understanding.
3. **Minimal yet effective**: Only the classification head is edited, no backbone update is required, and zero-shot operation is supported.
4. **Counter-intuitive finding**: Full-data fine-tuning underperforms one-shot fine-tuning; isolated fine-tuning with zero-shot initialization is the optimal strategy.

## Limitations & Future Work

- Relies on text labels to identify modifiers, making it difficult to handle purely visual distinctions (e.g., action speed).
- Zero-shot generality still has room for improvement (46% vs. an ideal 86%+).
- Validation is limited to classification tasks; downstream tasks such as detection and segmentation remain unexplored.
- Requires the original classifier to already contain some fine-grained categories for modifier dictionary construction.

## Related Work & Insights

- Directly related to the concept of model editing in NLP, borrowing the generality/locality evaluation framework.
- Naturally connected to compositional action recognition.
- The transferability assumption of modifiers warrants validation in broader domains.
- Extensible to classification refinement scenarios in other visual tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Both the task formulation and zero-shot editing approach are pioneering.
- **Technical Depth**: ⭐⭐⭐⭐ — The method is elegant and concise, with moderate technical complexity.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — A dedicated benchmark is constructed with comprehensive ablations.
- **Value**: ⭐⭐⭐⭐ — Low-cost classifier updating has practical application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Frame2Freq: Spectral Adapters for Fine-Grained Video Understanding](../../CVPR2026/video_understanding/frame2freq_spectral_adapters_for_fine-grained_video_understanding.md)
- [\[CVPR 2026\] Mistake Attribution: Fine-Grained Mistake Understanding in Egocentric Videos](../../CVPR2026/video_understanding/mistake_attribution_fine-grained_mistake_understanding_in_egocentric_videos.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](../../CVPR2026/video_understanding/ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)
- [\[ICCV 2025\] Fine-grained Spatiotemporal Grounding on Egocentric Videos](../../ICCV2025/video_understanding/fine-grained_spatiotemporal_grounding_on_egocentric_videos.md)
- [\[CVPR 2026\] SkeletonContext: Skeleton-side Context Prompt Learning for Zero-Shot Skeleton-based Action Recognition](../../CVPR2026/video_understanding/skeletoncontext_skeleton-side_context_prompt_learning_for_zero-shot_skeleton-bas.md)

</div>

<!-- RELATED:END -->
