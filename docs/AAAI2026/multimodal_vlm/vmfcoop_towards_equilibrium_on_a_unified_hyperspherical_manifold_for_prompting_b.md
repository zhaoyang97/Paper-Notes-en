---
title: >-
  [Paper Note] vMFCoOp: Towards Equilibrium on a Unified Hyperspherical Manifold for Prompting Biomedical VLMs
description: >-
  [AAAI 2026][Multimodal VLM][Prompt learning] This paper proposes the vMFCoOp framework, which aligns the semantic discrepancy between LLMs and CLIP on a unified hyperspherical manifold via inverse estimation of von Mises…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Prompt learning"
  - "biomedical VLM"
  - "von Mises-Fisher distribution"
  - "hyperspherical manifold"
  - "few-shot learning"
date: 2026-05-08
content_hash: 155466ecb23eeab8
---

# vMFCoOp: Towards Equilibrium on a Unified Hyperspherical Manifold for Prompting Biomedical VLMs

**Conference**: AAAI 2026 Oral  
**arXiv**: [2511.09540](https://arxiv.org/abs/2511.09540)  
**Code**: [GitHub](https://github.com/VinyehShaw/UniEqui)  
**Area**: Medical Imaging / Vision-Language Models (VLM)
**Keywords**: Prompt learning, biomedical VLM, von Mises-Fisher distribution, hyperspherical manifold, few-shot learning

## TL;DR

This paper proposes the vMFCoOp framework, which aligns the semantic discrepancy between LLMs and CLIP on a unified hyperspherical manifold via inverse estimation of von Mises-Fisher distributions, enabling robust few-shot prompt learning for biomedical VLMs.

## Background & Motivation

1. **Background**: Vision-language models such as CLIP achieve strong zero/few-shot generalization through large-scale contrastive learning, yet their effectiveness in the biomedical domain is limited—medical images exhibit highly structured semantics, fine-grained anatomy, strong anatomical priors, and cross-scale variation. Prompt learning methods such as CoOp/CoCoOp have emerged as lightweight adaptation strategies.
2. **Limitations of Prior Work**: BiomedCoOp leverages LLM-generated prompts to guide CLIP's context learning, but suffers from three issues:
    - **Semantic misalignment** between LLMs and CLIP due to differences in training corpora and model architectures
    - **Lack of scalability** to the rapidly evolving family of foundation models
    - Pairwise multimodal alignment in Euclidean space **fails to capture directional semantics and unified representations**
3. **Key Challenge**: Prompt learning must reconcile differing semantic abstractions, representational granularity, and alignment dynamics between LLMs and CLIP; however, existing methods perform independent pairwise matching in flat Euclidean space, which is insufficient for modeling the intrinsic relational geometry.
4. **Goal**: To align the semantic discrepancies of heterogeneous foundation models on a unified hyperspherical manifold, achieving stable, generalizable, and model-agnostic few-shot biomedical prompt learning.
5. **Key Insight**: CLIP and LLM embeddings are naturally $\ell_2$-normalized and reside on the unit hypersphere, making the vMF distribution a natural choice for modeling their directional semantics.
6. **Core Idea**: vMF distributions are estimated separately for CLIP vocabulary embeddings and LLM prompts on the hypersphere, then fused into unified semantic anchors to guide prompt optimization.

## Method

### Overall Architecture

vMFCoOp can be flexibly inserted into any LLM and biomedical CLIP backbone. The framework operates on a shared hyperspherical manifold $S^{d-1}$: (1) inverse estimation of vMF distribution parameters for the CLIP semantic anchor field and the LLM semantic prototype field; (2) fusion into unified semantic anchors; (3) optimization of prompt embeddings via three complementary constraints.

### Key Designs

1. **Unified Semantic Anchors**:
    - **Function**: Fuse CLIP's global semantic direction and LLM's class-specific semantic prototypes into a unified optimization target.
    - **Mechanism**:
     - **CLIP Semantic Anchor Field**: A vMF distribution is fit to CLIP vocabulary embeddings $\{w_i\}$; MLE estimates the mean direction $\mu_C = \bar{w}/R$ and concentration $\kappa_C \approx R(d-R^2)/(1-R^2+\epsilon)$.
     - **LLM Semantic Prototype Field**: A class-conditional vMF is fit to LLM prompt embeddings $T_c$ for each class $c$, yielding $(\mu_{L,c}, \kappa_{L,c})$.
     - **Fusion**: $u_i = (a_C + c_i) / \|a_C + c_i\|_2$, where $a_C = \kappa_C \mu_C$ and $c_i = \kappa_{L,c} \mu_{L,c}$.
    - **Design Motivation**: The mean direction and concentration of vMF jointly encode semantic location and confidence; through weighted fusion, the modality with higher semantic concentration contributes more to the unified anchor.

2. **Three-Level Constraint Optimization**:
    - **Function**: Optimize prompt embeddings on the hypersphere from three complementary perspectives.
    - **Mechanism**:
     - **Semantic Anchor Loss $\mathcal{L}_{anc}$**: Learnable offsets $\delta_i$ and a global scaling factor $\alpha$ dynamically adjust anchor directions; $\mathcal{L}_{anc} = \frac{1}{C} \sum_{i=1}^C \|\tilde{\mathcal{P}}_{c_i} - \tilde{u}_i^d\|_2^2$ draws prompt embeddings toward the unified anchors.
     - **Spherical Contrastive Loss $\mathcal{L}_{sc}$**: A prototype affinity matrix $S = \tau PU^\top$ is constructed; row-wise softmax cross-entropy pulls correct anchors closer and pushes confounding anchors apart; the temperature $\tau$ is annealed via cosine scheduling from $\tau_0$ to $\tau_{max}$ to progressively sharpen angular margins.
     - **Symmetric Cross-Entropy $\mathcal{L}_{SCE}$**: Encourages confident predictions for correct classes (forward CE) while penalizing distributional ambiguity over incorrect classes (reverse CE), enhancing cross-modal alignment.
    - **Design Motivation**: The three losses are complementary—$\mathcal{L}_{anc}$ ensures directional alignment, $\mathcal{L}_{sc}$ ensures inter-class separability, and $\mathcal{L}_{SCE}$ ensures visual-textual consistency.

3. **Model-Agnostic Plug-and-Play Design**:
    - **Function**: Supports arbitrary combinations of CLIP variants and LLMs.
    - **Mechanism**: vMF estimation is a post-processing step independent of specific model internals; unified anchors are fused via distributional parameters, adapting to semantic discrepancies across different models.
    - **Design Motivation**: Foundation models evolve rapidly, making fixed coupling to a specific LLM or CLIP backbone unsustainable.

### Loss & Training

- Total loss: $\mathcal{L} = \lambda_{anc} \mathcal{L}_{anc} + \lambda_{sc} \mathcal{L}_{sc} + \mathcal{L}_{SCE}$
- SGD with cosine learning rate scheduling, initial lr = 0.003, batch size = 4
- Default CLIP backbone: BiomedCLIP (ViT-B/16)
- Default LLM: GPT-4, 50 prompt templates per class
- Learnable context initialized with embeddings of "a photo of a"

## Key Experimental Results

### Main Results

| Few-Shot Setting | vMFCoOp | BiomedCoOp | Gain |
|-----------------|---------|------------|------|
| 1-shot | **57.25±4.75** | 55.08±5.85 | +2.17% |
| 4-shot | **68.29±2.07** | 63.65±3.27 | +4.64% |
| 8-shot | **72.07±1.98** | 71.29±2.19 | +0.78% |
| 16-shot | **75.45±1.48** | 73.63±1.27 | +1.82% |
| 64-shot | **77.49±1.05** | 73.65±3.98 | +3.84% |

Base-to-Novel Generalization (average over 14 datasets):

| Metric | vMFCoOp | BiomedCoOp |
|--------|---------|------------|
| Base | **78.02** | 73.26 |
| Novel | **76.70** | 71.91 |
| HM | **77.35** | 72.58 |

### Ablation Study

| Configuration | 1-shot | 4-shot | 16-shot | Base-to-Novel HM |
|--------------|--------|--------|---------|-----------------|
| No constraints (baseline) | 43.22 | 47.81 | 60.90 | 52.14 |
| $\mathcal{L}_{anc}$ only | 50.29 | 53.68 | 63.87 | 70.98 |
| $\mathcal{L}_{anc} + \mathcal{L}_{sc}$ | 49.87 | 54.23 | 73.25 | 73.42 |
| $\mathcal{L}_{SCE}$ only | 46.35 | 48.98 | 65.72 | 56.39 |
| All three constraints | **57.25** | **68.29** | **75.45** | **77.15** |

### Key Findings

- vMFCoOp achieves a relative improvement of 7.29% in the 4-shot setting, the most clinically meaningful low-data regime.
- Consistent improvements over BiomedCoOp are observed across CLIP backbones (BiomedCLIP, PubMedCLIP, MedCLIP, PMC-CLIP) and LLMs (GPT-4, Qwen2.5, Claude 3.5, DeepSeek R1).
- Significant gains are achieved on clinically challenging datasets such as cardiac MRI and liver MRI from UK Biobank.
- Visualizations show that vMFCoOp correctly localizes lesion regions in rare cases (posterior mediastinal tumors), whereas BiomedCoOp attention drifts toward the cardiac region.
- The three constraints are complementary: $\mathcal{L}_{anc}$ contributes the most, adding $\mathcal{L}_{sc}$ yields sustained gains at higher shot counts, and combining all three is optimal.

## Highlights & Insights

- The vMF distribution as a directional distribution on the hypersphere aligns naturally with CLIP's normalized embeddings, providing a solid theoretical foundation.
- The design of "unified semantic anchors" is elegant: the concentration $\kappa$ automatically weights the contribution of each modality.
- Model-agnosticism is a major practical advantage: no need to redesign alignment methods for each LLM–CLIP combination.
- The evaluation scale—14 datasets, 12 imaging modalities, and 13 anatomical regions—is exceptionally comprehensive for the biomedical VLM domain.
- The temperature annealing strategy in the spherical contrastive loss, transitioning from uniform angular partitioning to large-margin separation, reflects a well-considered training dynamic design.

## Limitations & Future Work

- Dataset-specific tuning of $\lambda_{anc}$ and $\lambda_{sc}$ increases the barrier to deployment.
- The vMF formulation assumes vocabulary embeddings are i.i.d. samples, whereas the internal structure of the vocabulary is considerably more complex in practice.
- Only text-side prompt tuning is explored; visual-side or joint tuning (e.g., MaPLe) remains uninvestigated.
- BUSI is excluded from base-to-novel evaluation due to class count constraints.
- The unified anchor fusion relies on simple weighted averaging; more sophisticated distributional fusion (e.g., vMF mixtures) may yield further improvements.

## Related Work & Insights

- **vs. BiomedCoOp**: BiomedCoOp performs simple token replacement in Euclidean space, ignoring the semantic gap between LLMs and CLIP; vMFCoOp explicitly models and aligns this discrepancy.
- **vs. CoOp/CoCoOp**: CoOp overfits to base classes; CoCoOp mitigates this via conditional prompts but lacks stability; vMFCoOp achieves better generalization through hyperspherical constraints.
- **vs. ProGrad**: ProGrad constrains updates along zero-shot gradient directions, which is stable but limits semantic flexibility; vMFCoOp's anchor-guided optimization is more flexible.
- **vs. MERU**: MERU models hierarchy in hyperbolic space but incurs complex optimization; vMFCoOp employs the hypersphere with vMF distributions for a more direct approach.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The first application of inverse vMF estimation to VLM prompt learning; the unified hyperspherical manifold perspective is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 14 datasets, 7 K-shot settings, cross-backbone validation, and comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Framework diagrams and equations are clear, though some derivations involve non-trivial jumps.
- **Value**: ⭐⭐⭐⭐⭐ Significant practical value for few-shot adaptation of biomedical VLMs; credibility is further strengthened by UK Biobank validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Few-Shot Precise Event Spotting via Unified Multi-Entity Graph and Distillation](few-shot_precise_event_spotting_via_unified_multi-entity_graph_and_distillation.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)
- [\[ICML 2026\] Neutral-Reference Prompting for Vision-Language Models](../../ICML2026/multimodal_vlm/neutral-reference_prompting_for_vision-language_models.md)
- [\[AAAI 2026\] URaG: Unified Retrieval and Generation in Multimodal LLMs for Efficient Long Document Understanding](urag_unified_retrieval_and_generation_in_multimodal_llms_for.md)
- [\[CVPR 2026\] ViKey: Enhancing Temporal Understanding in Videos via Visual Prompting](../../CVPR2026/multimodal_vlm/vikey_enhancing_temporal_understanding_in_videos_via_visual_prompting.md)

</div>

<!-- RELATED:END -->
