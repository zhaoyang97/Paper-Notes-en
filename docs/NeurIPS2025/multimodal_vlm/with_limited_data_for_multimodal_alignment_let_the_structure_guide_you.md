---
title: >-
  [Paper Note] STRUCTURE: With Limited Data for Multimodal Alignment, Let the Structure Guide You
description: >-
  [NeurIPS 2025][Multimodal VLM][Multimodal alignment] This paper proposes STRUCTURE regularization and a representation-similarity-based layer selection strategy that achieves high-quality cross-modal alignment between frozen unimodal foundation models using only tens of thousands of paired samples (less than 1% of conventional data requirements), yielding average improvements of 51.6% and 91.8% across 24 zero-shot classification and retrieval benchmarks.
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Multimodal alignment"
  - "low-data learning"
  - "structure regularization"
  - "frozen-encoder alignment"
  - "layer selection"
date: 2026-05-08
content_hash: 6a12428923b544a5
---

# STRUCTURE: With Limited Data for Multimodal Alignment, Let the Structure Guide You

**Conference**: NeurIPS 2025
**arXiv**: [2506.16895](https://arxiv.org/abs/2506.16895)  
**Code**: [https://brbiclab.epfl.ch/projects/structure](https://brbiclab.epfl.ch/projects/structure)  
**Area**: Multimodal VLM
**Keywords**: Multimodal alignment, low-data learning, structure regularization, frozen-encoder alignment, layer selection

## TL;DR
This paper proposes STRUCTURE regularization and a representation-similarity-based layer selection strategy that achieves high-quality cross-modal alignment between frozen unimodal foundation models using only tens of thousands of paired samples (less than 1% of conventional data requirements), yielding average improvements of 51.6% and 91.8% across 24 zero-shot classification and retrieval benchmarks.

## Background & Motivation
Unimodal foundation models (LLMs, DINOv2, AlphaFold, etc.) have achieved remarkable performance in their respective domains, yet many applications require mapping different modalities into a shared representation space. Multimodal models such as CLIP rely on 400 million paired samples for training, while in domains such as medicine and biology, acquiring large quantities of high-quality paired data is prohibitively expensive.

**Challenges of existing alignment approaches**:
- Supervised alignment methods (Linear/MLP mapping) require tens of millions of paired samples.
- Unsupervised methods (e.g., CKA) do not leverage paired data and perform only instance-level matching, failing to construct a shared embedding space.
- The Platonic Representation Hypothesis (suggesting models converge toward similar internal representations) provides a theoretical basis for low-data alignment, yet how to exploit it remains an open question.

**Core Problem**: Can two frozen unimodal encoders be aligned into a shared space using only tens of thousands of paired samples (<1% of conventional data)?

**Key Insight**: Preserve the rich neighborhood structure in the pre-trained representation space—which encodes relational information derived from millions or billions of samples—by using regularization to prevent excessive distortion during alignment, while selecting intermediate layers best suited for alignment rather than defaulting to the final layer.

## Method

### Overall Architecture
Both pre-trained encoders are frozen, and lightweight alignment functions (linear layers or MLPs) are learned to map each modality into a shared space. STRUCTURE regularization is applied on top of any existing alignment objective $\mathcal{L}_A$, and the optimal layer pair for alignment is selected based on representation similarity.

### Key Designs

1. **STRUCTURE Regularization**: Preserving multi-scale neighborhood geometry

    - For samples in pre-trained space $\mathcal{X}$ and aligned space $\mathcal{A}$, apply $\ell_2$ normalization followed by centering.
    - Compute temperature-scaled similarity matrices $S_X, S_A$, and apply row-wise softmax to obtain probability distributions $P_X, P_A$.
    - Capture $l$-hop relations via matrix exponentiation: $P_X^{(l)} = (P_X)^l$ (analogous to a random walk).
    - Measure structural discrepancy at each scale using Jensen-Shannon divergence.
    - The final regularizer is a weighted average (with higher weights for lower-order hops to counteract distribution concentration):
    $\mathcal{R}_S^{(L)}(X,A) = \frac{1}{L}\sum_{l=1}^L \frac{\text{JS}(P_X^{(l)}, P_A^{(l)})}{l}$
    - Total loss: $\mathcal{L} = \mathcal{L}_A + \lambda(\mathcal{R}_S(X_1, f_1(X_1)) + \mathcal{R}_S(X_2, f_2(X_2)))$

2. **Similarity-Based Layer Selection**:

    - Prior work defaults to aligning the final layer, yet intermediate layers may exhibit higher cross-modal similarity.
    - Mutual kNN is used to measure representation similarity across layer pairs, computed on a small set of paired samples.
    - Experiments demonstrate a strong correlation between layer similarity and downstream performance (high Spearman rank correlation $\rho$).
    - The layer pair with the highest similarity is selected for alignment.

3. **Theoretical Guarantees**:

    - Generalization bound: $|\hat{\mathcal{R}}_N - \mathcal{R}^*| \leq \mathcal{O}(1/\sqrt{N})$
    - STRUCTURE is invariant to global scaling, translation, and orthogonal rotation, depending solely on the intrinsic hierarchical relational structure.

### Loss & Training
- Alignment objective: symmetric contrastive loss $\mathcal{L}_C$ (CLIP-style) + STRUCTURE regularization.
- Training is performed on the COCO training set (80K pairs).
- Compatible with various alignment methods including Linear, MLP, and CSA.

## Key Experimental Results

### Main Results (RoBERTa + DINOv2 ViT-Giant)

| Method | STL10 | CIFAR10 | CIFAR100 | ImageNet | Flickr30 I2T | Flickr30 T2I |
|--------|-------|---------|----------|----------|--------------|--------------|
| Linear + Last | 75.6 | 85.5 | 34.0 | 9.9 | 32.5 | 22.1 |
| Linear + Similar + $\mathcal{R}_S$ | **92.6** | **96.3** | **51.3** | **24.7** | **65.8** | **53.7** |
| MLP + Last | 76.6 | 79.2 | 35.3 | 10.6 | 31.6 | 20.3 |
| MLP + Similar + $\mathcal{R}_S$ | **92.7** | **96.3** | **52.1** | **25.1** | **65.9** | **53.8** |
| CSA + Last | 77.9 | 78.5 | 47.4 | 23.2 | 47.0 | 38.3 |
| CSA + Similar + $\mathcal{R}_S$ | **91.7** | **97.2** | **56.4** | **26.8** | **56.1** | **43.1** |

### Ablation Study

| Component | Avg. Relative Gain (Classification) | Avg. Relative Gain (Retrieval) | Notes |
|-----------|--------------------------------------|-------------------------------|-------|
| Layer selection (Last→Similar) | +2.0%–4.8% | +2.7%–18.3% | Varies by method |
| STRUCTURE regularization | +26.8%–74.0% | +15.9%–137.0% | Largest gains for MLP/Linear |
| Combined | +51.6% avg | +91.8% avg | Synergistic effect |

### Key Findings
- **Feasibility with very few samples**: Only 1,000 paired samples with STRUCTURE still yields significant performance gains.
- **Data efficiency**: Approximately 23× fewer samples are needed on CIFAR100 and Flickr30 to match the performance of the unregularized baseline (data utility: 23.1× and 22.4×).
- **Power of in-domain samples**: Adding only 3–4 target-domain samples per class enables the method to surpass CLIP (trained on 400M samples) on Flowers (95% vs. CLIP's 93%) and CIFAR100.
- **Surpassing CLIP on CIFAR10**: The alignment method using only 0.02% of the data already outperforms end-to-end CLIP on certain datasets.
- **Neighborhood preservation verified**: Without regularization, Trustworthiness and Continuity decline steadily; with regularization, both stabilize at 0.99–1.00.

## Highlights & Insights
- **"Structure as prior"**: The relational structure encoded in pre-trained models, derived from hundreds of millions of samples, serves as an implicit large-scale data prior when preserved through regularization.
- **Multi-scale hierarchical design**: The method preserves not only direct neighbor relations (1-hop) but also multi-step reachability ($l$-hop), analogously characterizing manifold structure at multiple scales via random walks.
- **High practical utility**: Plug-and-play design compatible with any existing alignment method, with negligible computational overhead.

## Limitations & Future Work
- A performance gap with CLIP (trained on billions of samples) remains on complex tasks, requiring a small number of in-domain samples to compensate.
- Only two-modality alignment is explored; extension to three or more modalities is a straightforward but unvalidated direction.
- The regularization hyperparameter $\lambda$ and the number of hierarchy levels $L$ require some tuning.
- STRUCTURE involves matrix exponentiation, which may introduce memory bottlenecks at large batch sizes.

## Related Work & Insights
- **vs. CLIP (Radford et al., 2021)**: CLIP trains end-to-end on 400M samples; STRUCTURE uses frozen encoders with 80K samples and achieves comparable performance on certain tasks.
- **vs. FuseMix (Kim et al.)**: FuseMix also targets low-data alignment but focuses on data augmentation, whereas STRUCTURE emphasizes geometric regularization—the two are complementary.
- **vs. ASIF (Norelli et al.)**: ASIF leverages neighborhood structure for direct matching without learning alignment functions, whereas STRUCTURE learns a parameterized mapping.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of multi-scale neighborhood-preserving regularization and layer selection is novel and elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 24 datasets, 3 alignment methods, data scaling/extension analysis, and multiple model combinations.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation is clear, the core idea is well articulated, and figures and tables are information-dense.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for multimodal alignment in resource-constrained domains such as medicine and biology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Structure-Aware Fusion with Progressive Injection for Multimodal Molecular Representation Learning](structure-aware_fusion_with_progressive_injection_for_multimodal_molecular_repre.md)
- [\[ACL 2026\] STELLA: A Multimodal LLM for Protein Functional Annotation via Unified Sequence-Structure Encoding](../../ACL2026/multimodal_vlm/stella_a_multimodal_llm_for_protein_functional_annotation_via_unified_sequence-s.md)
- [\[NeurIPS 2025\] Aligning by Misaligning: Boundary-aware Curriculum Learning for Multimodal Alignment](aligning_by_misaligning_boundaryaware_curriculum_learning_fo.md)
- [\[ECCV 2024\] AdaShield: Safeguarding Multimodal Large Language Models from Structure-based Attack via Adaptive Shield Prompting](../../ECCV2024/multimodal_vlm/adashield_safeguarding_multimodal_large_language_models_from_structure-based_att.md)
- [\[ICCV 2025\] Oasis: One Image is All You Need for Multimodal Instruction Data Synthesis](../../ICCV2025/multimodal_vlm/oasis_one_image_is_all_you_need_for_multimodal_instruction_data_synthesis.md)

</div>

<!-- RELATED:END -->
