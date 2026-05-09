---
title: >-
  [Paper Note] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks
description: >-
  [CVPR 2026][Multimodal VLM][VLM debiasing] This paper proposes a closed-form debiasing method for VLMs that performs orthogonal decomposition of attribute subspaces in the cross-modal embedding space and solves via Chebyshev scalarization, achieving Pareto-optimal fairness with bounded utility loss. The approach is training-free and annotation-free, and uniformly covers three downstream tasks: zero-shot classification, text-image retrieval, and text-image generation.
tags:
  - CVPR 2026
  - Multimodal VLM
  - VLM debiasing
  - closed-form solution
  - Pareto-optimal fairness
  - training-free annotation-free
  - cross-modal embedding space
date: 2026-05-08
content_hash: 8bf69daa8f6bf4c6
---

# A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks

**Conference**: CVPR 2026
**arXiv**: [2603.12998](https://arxiv.org/abs/2603.12998)
**Code**: [Available](https://github.com/Supltz/Debias_VLM)
**Area**: Multimodal VLM
**Keywords**: VLM debiasing, closed-form solution, Pareto-optimal fairness, training-free annotation-free, cross-modal embedding space

## TL;DR

This paper proposes a closed-form debiasing method for VLMs that performs orthogonal decomposition of attribute subspaces in the cross-modal embedding space and solves via Chebyshev scalarization, achieving Pareto-optimal fairness with bounded utility loss. The approach is training-free and annotation-free, and uniformly covers three downstream tasks: zero-shot classification, text-image retrieval, and text-image generation.

## Background & Motivation

**State of the Field**: VLMs such as CLIP achieve strong cross-modal understanding through contrastive learning on large-scale web image-text pairs, and are widely applied to zero-shot classification, image-text retrieval, and image generation. However, research has shown that VLMs inherit social biases from training data—for example, CLIP's text embeddings place "nurse" anomalously close to "female" and "doctor" close to "male", reflecting gender stereotypes in web corpora. These biases propagate through embeddings to all downstream applications.

**Limitations of Prior Work**: Existing debiasing methods suffer from several shortcomings: (a) many methods (DeAR, FairerCLIP, PromptArray) require training additional debiasing networks, increasing computational and model complexity; (b) some methods (SFID, CLIP-clip) require sensitive attribute annotations, which are difficult to obtain at scale due to privacy and ethical concerns; (c) most methods address only a single downstream task (e.g., PRISM/RoboShot handle only classification, SANER only retrieval and generation); (d) methods that debias only the text modality while ignoring the image modality (Orth-Proj, Orth-Cali, SANER) achieve limited effectiveness, since biases are encoded into both modalities through contrastive learning.

**Root Cause**: There is a fundamental trade-off between debiasing and utility preservation. Debiasing inevitably discards some semantic information, leading to downstream performance degradation. Existing methods either do not explicitly handle utility preservation (BiasedPrompt), or preserve it indirectly via reconstruction losses (DeAR, SANER)—but reconstructing embeddings does not guarantee preservation of cross-modal alignment. No prior method provides a theoretical upper bound on utility loss. Furthermore, existing methods focus only on group fairness while neglecting intersectional fairness (e.g., gender × age combinations).

**Paper Goals**: To propose a unified VLM debiasing framework that simultaneously satisfies five requirements: training-free, annotation-free, joint debiasing of both modalities, coverage of multiple downstream tasks, and theoretical guarantees on utility loss. This paper also introduces systematic evaluation of intersectional fairness in VLM debiasing for the first time.

**Starting Point**: The paper observes that VLM embedding spaces lie on the unit hypersphere $\mathbb{S}^{d-1}$, and any embedding can be decomposed into attribute-related and neutral content components via the orthogonal decomposition theorem. Unlike prior methods (PRISM-mini, Orth-Proj) that project onto the subspace $\mathcal{S}$ spanned by group prototypes—which also contains semantic information (e.g., the meaning of "doctor") and thus inadvertently removes it—this work projects exclusively onto the attribute-difference subspace $\mathcal{A}$, precisely removing bias while preserving semantics.

**Core Idea**: Within the cross-modal embedding space, the debiasing problem is reduced to a two-dimensional unit circle via orthogonal decomposition of the attribute subspace, and a closed-form optimal solution is derived using Chebyshev minimax optimization, simultaneously achieving Pareto-optimal fairness and bounded utility loss.

## Method

### Overall Architecture

Input text/image → VLM encoder produces embedding $\vec{e} \in \mathbb{S}^{d-1}$ → LLM-guided construction of group prototypes $\vec{p}_g$ → compute attribute directions $\vec{a}_i = \vec{p}_{g_i} - \vec{p}_{g_1}$ to build attribute subspace $\mathcal{A}$ → orthogonal decomposition $\vec{e} = \vec{e}_{\mathcal{A}_\parallel} + \vec{e}_{\mathcal{A}_\perp}$ → closed-form solution for optimal debiased embedding $\vec{u}^*$ → output debiased embedding for downstream tasks. The entire pipeline requires no training and no annotated data; it operates purely at inference time.

### Key Designs

**Module 1: LLM-Guided Group Prototype Construction**

- **Function**: Constructs a robust embedding prototype $\vec{p}_g$ for each sensitive attribute group (e.g., male/female) to define the attribute subspace.
- **Mechanism**: Given an input prompt (e.g., "a photo of a doctor"), an LLM (GPT-5) performs two operations: (1) inserting group identifiers to obtain $T_g$ (e.g., "a photo of a male doctor"); (2) generating multiple linguistic variants $\mathcal{T}_g$ for each group (e.g., "a photo of a man doctor", "a photo of a masculine doctor"). The spherical mean of all variant text embeddings is used as the group prototype: $\vec{p}_g = \frac{\vec{e}_g + \sum_i \vec{e}_g^{(i)}}{\|\vec{e}_g + \sum_i \vec{e}_g^{(i)}\|}$.
- **Design Motivation**: Prior methods (PRISM, Orth-Proj) use a single prompt directly as the group prototype, but attribute groups are linguistically non-singular ("man", "gentleman", and "boy" all refer to males but carry different semantics). While SANER constructs a lexicon, it is not conditioned on the input prompt, potentially causing semantic inconsistency. The LLM generates contextually appropriate variants, and the spherical mean ensures maximal representativeness on the hypersphere.

**Module 2: Attribute Subspace Orthogonal Decomposition**

- **Function**: Decomposes the original embedding $\vec{e}$ into an attribute-leakage component (bias) and a neutral content component (semantics), providing a precise operational space for debiasing.
- **Mechanism**: The attribute subspace is defined as $\mathcal{A} = \text{span}\{\vec{a}_2, \dots, \vec{a}_n\}$, where $\vec{a}_i = \vec{p}_{g_i} - \vec{p}_{g_1}$. Projection operators $P_{\mathcal{A}_\parallel} = A(A^\top A)^{-1}A^\top$ and $P_{\mathcal{A}_\perp} = I - P_{\mathcal{A}_\parallel}$ decompose the embedding as $\vec{e} = \vec{e}_{\mathcal{A}_\parallel} + \vec{e}_{\mathcal{A}_\perp}$. The fairness objective is equivalent to making the debiased embedding equidistant from all group prototypes, i.e., $\langle \vec{u}, \vec{a}_i \rangle = 0$.
- **Design Motivation**: Prior methods project onto $\mathcal{S} = \text{span}\{\vec{p}_{g_1}, \vec{p}_{g_2}, \dots\}$, which contains semantic components shared across groups (e.g., the meaning of "doctor"), causing inadvertent semantic deletion. $\mathcal{A}$ contains only inter-group difference directions, with dimensionality $r \leq n-1 \ll d$, far smaller than $d$, thus minimizing the impact of debiasing on semantics.

**Module 3: Chebyshev Scalarization Closed-Form Solver**

- **Function**: Finds the Pareto-optimal point on the frontier between fairness (minimizing attribute leakage $L(\alpha) = \alpha$) and utility (minimizing self-utility loss $V(\alpha)$) that is robust to arbitrary objective weights.
- **Mechanism**: Lemma 1 reduces the search space on the hypersphere to a two-dimensional unit circle in $\text{span}\{\vec{e}_{\mathcal{A}_\parallel}, \vec{e}_{\mathcal{A}_\perp}\}$; Lemma 2 further restricts the search to the first quadrant with $\alpha \in [0, \|\vec{e}_{\mathcal{A}_\parallel}\|]$. The Chebyshev minimax problem $\min_\alpha \sup_{w_1,w_2} \{w_1 L(\alpha) + w_2 V(\alpha)\}$ is then solved in closed form: $\alpha^* = \frac{E - \|\vec{e}_{\mathcal{A}_\perp}\|\sqrt{E^2 - \|\vec{e}_{\mathcal{A}_\parallel}\|^2}}{E^2 + \|\vec{e}_{\mathcal{A}_\perp}\|^2}$, where $E = \|\vec{e}_{\mathcal{A}_\parallel}\| + (1-\|\vec{e}_{\mathcal{A}_\perp}\|)/\|\vec{e}_{\mathcal{A}_\parallel}\|$.
- **Design Motivation**: Since the method must be task-agnostic, it cannot assume knowledge of downstream tasks to tune the preference weights $w_1, w_2$. The Chebyshev approach minimizes the worst-case objective over all weight combinations, achieving robustness across arbitrary tasks. The closed-form solution eliminates the computational overhead and convergence issues of iterative optimization.

### Loss & Training

The method is entirely training-free and operates as a pure inference-time closed-form transformation. Key theoretical guarantees include:

- **Proposition 1**: The cross-utility loss is upper-bounded by $\ell_{cross} \leq \sqrt{2\ell_{self}^{(I)}} + \sqrt{2\ell_{self}^{(T)}}$, reducing cross-modal utility preservation to single-modal self-utility.
- **Theorem 1**: The self-utility loss at the optimal solution satisfies $V(\alpha^*) = (1 - \|\vec{e}_{\mathcal{A}_\perp}\|) \cdot \alpha^* / \|\vec{e}_{\mathcal{A}_\parallel}\|$, and a tighter upper bound on cross-utility loss is also established.

## Key Experimental Results

### Main Results

**Table 1: Zero-Shot Image Classification Results** (CLIP ViT-L/14, values ×100)

| Method | CelebA F1↑ | CelebA ΔEO_Avg(G×A)↓ | CelebA ΔEO_Max(G×A)↓ | FACET F1↑ | FACET ΔEO_Avg(G)↓ |
|--------|-----------|----------------------|----------------------|----------|-------------------|
| Baseline CLIP | 54.0 | 25.1 | 45.0 | 70.8 | 8.9 |
| FairerCLIP | 53.1 | 24.0 | 41.4 | 69.8 | 9.2 |
| RoboShot | 52.3 | 23.3 | 40.0 | 69.3 | 8.5 |
| Orth-Proj | 49.3 | 26.0 | 42.1 | 68.6 | 9.0 |
| **Ours** | **56.5** | **23.6** | **40.1** | **70.7** | **8.3** |

**Table 2: Text-Image Retrieval and Generation Results** (values ×100)

| Method | COCO R@5↑ | COCO MS@1000(G×ST)↓ | Flickr30K R@5↑ | Flickr30K MS@1000(G)↓ | SD v2.1 SP↓ | SD v2.1 AccG↑ |
|--------|----------|---------------------|---------------|----------------------|------------|--------------|
| Baseline | 83.8 | 13.4 | 91.0 | 20.3 | 47.9 | 75.4 |
| SFID | 77.4 | 13.2 | 86.8 | 13.6 | 41.1 | 67.2 |
| CLIP-clip | 76.1 | 9.9 | 87.7 | 11.7 | - | - |
| Orth-Proj | 74.5 | 13.6 | 84.4 | 14.1 | 39.6 | 53.4 |
| **Ours** | **81.1** | **10.1** | **90.4** | **11.8** | **39.7** | **74.6** |

### Ablation Study

**Table 3: Ablation and LLM Sensitivity Analysis** (Flickr30K MS@1000↓ / CelebA ΔEO_Max↓)

| Configuration | MS@1000 | ΔEO_Max |
|--------------|---------|---------|
| Baseline | 20.3 | 45.0 |
| Anchor embedding only as prototype | 13.4 | 41.1 |
| Mean embedding only as prototype | 14.1 | 41.8 |
| Debias image only $\vec{u}_I$ | 13.4 | 41.7 |
| Debias text only $\vec{u}_T$ | 13.3 | 41.1 |
| Replace with DeepSeek v3.2 | 12.0 | 40.1 |
| Replace with Gemini 2.5 Pro | 11.8 | 40.4 |
| **Full method** | **11.8** | **40.1** |

### Key Findings

- **Utility preservation substantially outperforms prior methods**: On CelebA classification, F1 reaches 56.5 (baseline 54.0), while all other debiasing methods fall below the baseline; on retrieval tasks, R@5 degradation is only 2.7 (vs. 6.4 for SFID and 7.7 for CLIP-clip); on generation, AccG reaches 74.6 (close to baseline 75.4), far exceeding Orth-Proj's 53.4.
- **Annotation-free methods outperform annotation-dependent ones**: Through systematic analysis of three research questions, the paper shows that annotation-dependent methods (SFID, FairerCLIP) are constrained by annotation domain (FairFace is a face dataset) and generalize poorly to full-body datasets (FACET, COCO, Flickr30K).
- **Joint dual-modality debiasing is necessary**: Debiasing only the image or only the text modality yields notably worse fairness metrics compared to joint debiasing.
- **LLM choice has minimal impact on results**: Group variants generated by GPT-5, DeepSeek v3.2, and Gemini 2.5 Pro produce nearly identical debiasing performance.

## Highlights & Insights

- The closed-form solution offers determinism, efficiency, and interpretability; this is the first method in VLM debiasing to provide theoretical utility guarantees, establishing a paradigm of "bounded debiasing."
- The distinction between attribute subspace $\mathcal{A}$ and group prototype subspace $\mathcal{S}$ is the central insight: $\mathcal{S}$ contains shared semantics while $\mathcal{A}$ contains only difference directions, enabling precise debiasing.
- Chebyshev minimax optimization achieves task-agnostic robustness, eliminating the need for per-task hyperparameter tuning.
- The systematic three-RQ analysis provides clear design principles for the VLM debiasing field: no annotations needed, no training needed, dual-modality debiasing required.

## Limitations & Future Work

- Utility guarantees are given in embedding space (cosine similarity) rather than task metric space (F1/R@K); the gap between the two may be significant in extreme scenarios.
- The closed-form solution relies on the linearity assumption of the attribute subspace; non-linear bias patterns may not be captured.
- The method only addresses encoder-side embeddings and does not extend to decoders (e.g., Stable Diffusion's U-Net), making debiasing for generation tasks indirect.
- Currently only a limited set of attributes (gender, age, skin tone) are considered; scalability to more complex intersectional attribute combinations remains to be verified.

## Related Work & Insights

- **vs. PRISM/Orth-Proj**: Projecting onto the full group subspace $\mathcal{S}$ destroys semantics; this work projects only onto attribute difference directions $\mathcal{A}$, a more precise operation.
- **vs. SANER/DeAR**: These methods require training additional networks and provide no theoretical utility guarantees; the proposed closed-form solution entirely eliminates training and hyperparameter tuning.
- **vs. FairerCLIP**: Relies on annotated data such as FairFace for training, and generalizes poorly outside face-centric scenarios; this work is entirely annotation-free.
- **Insights**: The paradigm of using orthogonal decomposition to perform precise "surgical" removal of specific attributes from embeddings can be generalized to other settings where targeted attribute removal from embeddings is desired.

## Rating

⭐⭐⭐⭐ The theoretical derivations are rigorous and complete (closed-form solution + Pareto optimality + utility upper bound proofs), and experiments cover three major tasks across multiple datasets and backbones, including human evaluation. However, the core mathematical tools (orthogonal decomposition / Chebyshev scalarization) are individually mature; the novelty lies primarily in problem formalization and their clever integration.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Interpretable Debiasing of Vision-Language Models for Social Fairness](interpretable_debiasing_of_vision-language_models_for_social_fairness.md)
- [\[CVPR 2026\] HulluEdit: Single-Pass Evidence-Consistent Subspace Editing for Mitigating Hallucinations in Large Vision-Language Models](hulluedit_single-pass_evidence-consistent_subspace_editing_for_mitigating_halluc.md)
- [\[CVPR 2026\] GTR-Turbo: Merged Checkpoint is Secretly a Free Teacher for Agentic VLM Training](gtr-turbo_merged_checkpoint_is_secretly_a_free_teacher_for_agentic_vlm_training.md)
- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)
- [\[CVPR 2026\] HiFICL: High-Fidelity In-Context Learning for Multimodal Tasks](hificl_high-fidelity_in-context_learning_for_multimodal_tasks.md)

<!-- RELATED:END -->
