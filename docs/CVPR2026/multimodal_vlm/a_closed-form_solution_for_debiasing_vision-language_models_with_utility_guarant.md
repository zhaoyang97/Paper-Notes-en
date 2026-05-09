---
title: >-
  [Paper Note] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks
description: >-
  [CVPR 2026][Multimodal VLM][VLM debiasing] This paper proposes a training-free, annotation-free debiasing method for VLMs that operates in cross-modal embedding spaces. Via orthogonal decomposition, it achieves a Pareto-optimal fairness–utility trade-off with a closed-form solution and provides theoretical upper bounds on utility loss.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLM debiasing
  - fairness
  - closed-form solution
  - Pareto optimality
  - cross-modal alignment
date: 2026-05-08
content_hash: 58706b1cff4c5adc
---

# A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks

**Conference**: CVPR 2026
**arXiv**: [2603.12998](https://arxiv.org/abs/2603.12998)
**Code**: [Available](https://github.com/Supltz/Debias_VLM)
**Area**: Multimodal VLM
**Keywords**: VLM debiasing, fairness, closed-form solution, Pareto optimality, cross-modal alignment

## TL;DR

This paper proposes a training-free, annotation-free debiasing method for VLMs that operates in cross-modal embedding spaces. Via orthogonal decomposition, it achieves a Pareto-optimal fairness–utility trade-off with a closed-form solution and provides theoretical upper bounds on utility loss.

## Background & Motivation

**State of the Field**: VLMs such as CLIP, trained on massive web-scale image–text pairs, achieve strong performance on zero-shot classification, image–text retrieval, and text-to-image generation, but inevitably inherit social biases from training data (e.g., "nurse" is anomalously similar to "female" while "doctor" is anomalously similar to "male").

**Limitations of Prior Work**: Existing debiasing methods suffer from multiple limitations: they require training auxiliary networks (DeAR, FairerCLIP, PromptArray), depend on sensitive-attribute annotations (SFID, CLIP-clip), address only a single modality (SANER and BiasedPrompt debias text only), apply to only a single downstream task (PRISM targets zero-shot classification only), and cannot guarantee that utility does not degrade significantly after debiasing.

**Root Cause**: There is an inherent trade-off between fairness and utility—removing sensitive-attribute information may simultaneously harm semantic content. Prior methods project onto the entire subspace $\mathcal{S}$, which contains not only attribute information but also semantic content (e.g., "doctor") that should be preserved, resulting in over-debiasing.

**Paper Goals**: To design a unified debiasing framework that simultaneously satisfies: training-free, annotation-free, dual-modality debiasing, multi-task applicability, and provably bounded utility loss.

**Starting Point**: The paper formalizes the debiasing problem as an optimization on the unit hypersphere by exploiting the geometric structure of the cross-modal embedding space, and reduces the high-dimensional search space to two dimensions via orthogonal decomposition.

**Core Idea**: Project only onto the **attribute subspace** $\mathcal{A}$ (spanned by inter-group difference directions) rather than the entire group prototype subspace $\mathcal{S}$, thereby removing bias while preserving semantic content. A minimax problem is then solved via Chebyshev scalarization to obtain a closed-form optimal solution robust to arbitrary fairness–utility weighting.

## Method

### Overall Architecture

The method consists of two steps: (1) LLM-guided group prototype construction—building representative text embeddings for each sensitive-attribute group; and (2) closed-form debiased embedding computation—finding the Pareto-optimal debiased vector $\vec{u}^{\star}$ on the unit hypersphere $\mathbb{S}^{d-1}$. No training or labeled data is required, and the method can be directly plugged into any VLM.

### Key Designs

#### 1. LLM-Guided Group Prototype Construction

- **Function**: Construct a robust text prototype $\vec{p}_g$ for each sensitive-attribute group $g$.
- **Mechanism**: Given an input prompt (e.g., "a photo of a doctor"), an LLM (GPT-5) generates group-specific prompts (e.g., "a photo of a male doctor") along with multiple paraphrastic variants (e.g., "a photo of a man doctor," "a photo of a masculine doctor"), and the spherical mean of all variant embeddings is taken as the group prototype.
- **Design Motivation**: Prior methods use a single group prompt as the prototype, ignoring linguistic diversity in attribute expression. Although SANER augments with a corpus, the word set is not conditioned on the input context. This method leverages LLM reasoning to generate context-aligned variants, producing more representative prototypes.

#### 2. Attribute Subspace Construction and Orthogonal Decomposition

- **Function**: Define the attribute subspace $\mathcal{A} = \text{span}\{\vec{a}_2, \ldots, \vec{a}_n\}$, where $\vec{a}_i = \vec{p}_{g_i} - \vec{p}_{g_1}$ denotes the inter-group difference direction.
- **Mechanism**: The original embedding $\vec{e}$ is orthogonally decomposed into $\vec{e}_{\mathcal{A}_\parallel}$ (the attribute-leakage component) and $\vec{e}_{\mathcal{A}_\perp}$ (the neutral semantic component); debiasing amounts to reducing the former while preserving the latter.
- **Design Motivation**: Unlike prior work that projects onto the entire subspace $\mathcal{S}$ (discarding semantic content), projecting only onto $\mathcal{A}$ precisely removes bias without harming content.

#### 3. Closed-Form Optimal Solution

- **Function**: Compute the Pareto-optimal debiased embedding on $\mathbb{S}^{d-1}$ that simultaneously minimizes attribute leakage (fairness) and self-utility loss (utility).
- **Mechanism**:
    - Lemma 1 reduces the search space from the high-dimensional hypersphere to a two-dimensional unit circle in $\text{span}\{\vec{e}_{\mathcal{A}_\parallel}, \vec{e}_{\mathcal{A}_\perp}\}$.
    - The solution is parameterized by a scalar $\alpha$: $\vec{u} = \alpha \frac{\vec{e}_{\mathcal{A}_\parallel}}{\|\vec{e}_{\mathcal{A}_\parallel}\|} + \sqrt{1-\alpha^2} \frac{\vec{e}_{\mathcal{A}_\perp}}{\|\vec{e}_{\mathcal{A}_\perp}\|}$
    - To avoid task-specific hyperparameter tuning, a minimax problem is solved via Chebyshev scalarization, yielding a closed-form optimal $\alpha^{\star}$ (Theorem 1) that is robust to any weight pair $(w_1, w_2)$.
- **Design Motivation**: Prior methods either project completely ($\alpha=0$, optimal fairness but worst utility) or apply no debiasing ($\alpha=\|\vec{e}_{\mathcal{A}_\parallel}\|$, optimal utility but worst fairness). This method finds the minimax-optimal point on the Pareto frontier without manual tuning.

### Loss & Training

The method **requires no training**. The optimization objective is:

$$\min_{0 \leq \alpha \leq \|\vec{e}_{\mathcal{A}_\parallel}\|} \sup_{w_1+w_2=1} \{w_1 L(\alpha) + w_2 V(\alpha)\}$$

where $L(\alpha) = \alpha$ is the attribute leakage and $V(\alpha) = 1 - \alpha\|\vec{e}_{\mathcal{A}_\parallel}\| - \sqrt{1-\alpha^2}\|\vec{e}_{\mathcal{A}_\perp}\|$ is the self-utility loss. This problem admits a closed-form solution requiring no iterative optimization. Theoretical utility guarantees: self-utility loss $\leq 1 - \|\vec{e}_{\mathcal{A}_\perp}\|$; cross-utility loss is bounded above by the self-utility losses of the two modalities via Proposition 1.

## Key Experimental Results

### Main Results

**Zero-shot image classification** (CelebA + FACET, CLIP ViT-L/14):

| Method | CelebA F1↑ | ΔEO_Avg (G×A)↓ | ΔEO_Max (G×A)↓ | FACET F1↑ | ΔEO_Avg (G)↓ | ΔEO_Max (G)↓ |
|--------|-----------|----------------|----------------|----------|--------------|--------------|
| CLIP Baseline | 54.0 | 25.1 | 45.0 | 70.8 | 8.9 | 49.8 |
| RoboShot | 52.3 | **23.3** | **40.0** | 69.3 | 8.5 | **47.3** |
| FairerCLIP | 53.1 | 24.0 | 41.4 | 69.8 | 9.2 | 50.1 |
| **Ours** | **56.5** | 23.6 | 40.1 | **70.7** | **8.3** | 47.5 |

**Image–text retrieval** (COCO2017 + Flickr30K, CLIP ViT-L/14):

| Method | COCO R@5↑ | COCO R@10↑ | MS@1000 (G×ST)↓ | Flickr R@5↑ | Flickr R@10↑ | MS@1000 (G)↓ |
|--------|----------|-----------|-----------------|------------|-------------|--------------|
| CLIP Baseline | 83.8 | 90.1 | 13.4 | 91.0 | 95.4 | 20.3 |
| CLIP-clip | 76.1 | 85.2 | **9.9** | 87.7 | 91.5 | **11.7** |
| FairerCLIP | 76.8 | 85.4 | 10.2 | 87.9 | 92.5 | 12.2 |
| **Ours** | **81.1** | **89.0** | 10.1 | **90.4** | **94.9** | 11.8 |

### Ablation Study

On Flickr30K and CelebA, ablations are conducted on prototype construction and modality-specific debiasing:

| Ablation Condition | MS@1000 (G)↓ | ΔEO_Max (G×A)↓ |
|-------------------|--------------|----------------|
| Baseline | 20.3 | 45.0 |
| Anchor embedding $\vec{p}_a$ only | 13.4 | 41.1 |
| Mean embedding $\vec{p}_m$ only | 14.1 | 41.8 |
| Image debiasing $\vec{u}_I$ only | 13.4 | 41.7 |
| Text debiasing $\vec{u}_T$ only | 13.3 | 41.1 |
| Full method | **11.8** | **40.1** |

Different LLMs (DeepSeek v3.2, Gemini 2.5 Pro) have negligible effect on results, demonstrating the robustness of prototype generation.

### Key Findings

1. **Superior utility preservation**: Zero-shot classification F1 reaches 56.5% (vs. baseline 54.0%, an improvement), and retrieval R@5/R@10 are nearly lossless (81.1 vs. 83.8; 90.4 vs. 91.0), far outperforming the large drops seen in competing methods.
2. **Cross-task consistency**: Best or second-best fairness is achieved across all three tasks: classification, retrieval, and text-to-image generation.
3. **Necessity of dual-modality debiasing**: Debiasing either modality alone yields worse fairness metrics than joint dual-modality debiasing.
4. **No systematic advantage of annotated data**: Methods requiring labeled data (e.g., FairerCLIP, PromptArray) generalize poorly in cross-domain settings (face-centric → full-body).
5. **Text-to-image generation**: SP↓ 39.7 vs. baseline 47.9, while CLIP score (24.2) and AccG (74.6%) are far superior to Orth-Proj (19.7/53.4%) and Orth-Cali (20.7/56.6%).

## Highlights & Insights

- **Mathematical elegance**: VLM debiasing is formalized from a geometric perspective as an optimization on the unit hypersphere; orthogonal decomposition followed by dimensionality reduction yields a closed-form solution, eliminating iterative optimization and hyperparameter search.
- **Provably bounded utility**: This work is the first to provide theoretical upper bounds on both self-utility loss and cross-utility loss for VLM debiasing, rather than relying on empirical observation.
- **Genuine plug-and-play**: Training-free, annotation-free, dual-modality, multi-task, and directly applicable to any VLM (validated on CLIP, BLIP, and other architectures).
- **Clever use of Chebyshev scalarization**: The minimax formulation eliminates task-specific weight tuning, making the solution robust to any fairness–utility preference.
- **Attribute subspace vs. group prototype subspace** is the key insight—prior methods projecting onto $\mathcal{S}$ discard semantics such as "doctor," whereas projecting only onto $\mathcal{A}$ precisely removes gender disparity.

## Limitations & Future Work

1. **Utility guarantees are in embedding space, not task metrics**: The theoretical bounds constrain utility in terms of cosine similarity, not directly in terms of F1, R@K, or other task-specific metrics. Empirically, the two are highly correlated, but a gap remains.
2. **Encoder-side only**: The method operates in the VLM encoder embedding space and has not yet been extended to the decoder side (e.g., directly debiasing the decoder of generative models).
3. **Limited attribute coverage**: Constrained by existing fairness benchmarks, the evaluation covers only a limited set of sensitive attributes (gender, age, skin tone) and does not address more complex intersectional combinations.
4. **Future directions**: Extending the closed-form solution to decoder spaces; incorporating richer attribute taxonomies; exploring adaptive $\alpha$ that adjusts to downstream tasks.

## Related Work & Insights

- **Projection-based debiasing** (Orth-Proj/Orth-Cali, PRISM-mini): The "perfect fairness" extreme case $\alpha=0$ of this paper exactly corresponds to these methods, illustrating that they sacrifice all information along attribute directions.
- **Adversarial training approaches** (DeAR, PromptArray): These require labeled data and training, and their generalization is limited to the training domain.
- **Inspiration from NLP debiasing**: The word embedding debiasing ideas of Bolukbasi et al. are extended to the multimodal setting, with the addition of theoretical utility guarantees.
- **Implications for fairness ML**: Chebyshev scalarization is commonly used in multi-objective optimization; this work is the first to apply it to the fairness–utility balance in VLM debiasing.

## Rating

⭐⭐⭐⭐ A theoretically rigorous and practically effective VLM debiasing method. The closed-form solution is elegant, the utility guarantees are theoretically grounded, and experiments comprehensively cover three downstream tasks with intersectional fairness evaluation. The primary limitation is that utility guarantees remain at the embedding-space level.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] FairLLaVA: Fairness-Aware Parameter-Efficient Fine-Tuning for Large Vision-Language Models](fairllava_fairness-aware_parameter-efficient_fine-tuning_for_large_vision-langua.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)

<!-- RELATED:END -->
