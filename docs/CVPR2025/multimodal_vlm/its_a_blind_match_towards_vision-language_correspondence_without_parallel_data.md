---
title: >-
  [Paper Note] It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data
description: >-
  [CVPR 2025][Multimodal VLM][Unsupervised Alignment] This paper presents the first systematic study on the feasibility of "blind matching" using only the pairwise internal distances within the respective vision and language embedding spaces, in the **complete absence of parallel data**. It proposes a factored Hahn-Grant QAP solver (reducing memory complexity from $O(N^4)$ to $O(N^3)$) and demonstrates the feasibility of this matching through large-scale experiments involving 3…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Unsupervised Alignment"
  - "Quadratic Assignment Problem"
  - "Vision-Language Matching"
  - "Platonic Representation Hypothesis"
  - "Gromov-Wasserstein"
date: 2026-05-08
content_hash: 0f9c15ed88a6c179
---

# It's a (Blind) Match! Towards Vision-Language Correspondence without Parallel Data

**Conference**: CVPR 2025  
**arXiv**: [2503.24129](https://arxiv.org/abs/2503.24129)  
**Code**: [https://dominik-schnaus.github.io/itsamatch](https://dominik-schnaus.github.io/itsamatch) (project page available)  
**Area**: Multimodal VLM  
**Keywords**: Unsupervised Alignment, Quadratic Assignment Problem, Vision-Language Matching, Platonic Representation Hypothesis, Gromov-Wasserstein

## TL;DR

This paper presents the first systematic study on the feasibility of "blind matching" using only the pairwise internal distances within the respective vision and language embedding spaces, in the **complete absence of parallel data**. It proposes a factored Hahn-Grant QAP solver (reducing memory complexity from $O(N^4)$ to $O(N^3)$) and demonstrates the feasibility of this matching through large-scale experiments involving 33 vision models $\times$ 27 language models, even achieving unsupervised image classification.

## Background & Motivation

The Platonic Representation Hypothesis suggests that as models and data scale, the representation spaces of different modalities (vision and language) tend to become geometrically aligned—meaning that the pairwise relationships between the same underlying world concepts are consistent across modalities (e.g., the distance between "cat" and "dog" is similar in both visual and language spaces). Existing cross-modal alignment methods (such as zero-shot stitching) still require a small amount of parallel data. This work pushes this concept to the extreme: **Can vision-language correspondence be achieved in the complete absence of parallel data, relying solely on the pairwise internal distances of each modality?** The value of this study is twofold: (1) providing a tool to investigate the alignment between vision and language models, including on large amounts of unlabeled data; (2) opening up possibilities for unsupervised visual recognition. Core Idea: Modeling unsupervised matching as a Quadratic Assignment Problem (QAP) and developing an efficient solver.

## Method

### Overall Architecture

Given $N$ classes, average embeddings $\mathbf{x}_i$ and $\mathbf{y}_i$ are extracted for each class using a vision model $f_v$ and a language model $f_l$, respectively. Pairwise similarity matrices $\mathbf{X}$ and $\mathbf{Y}$ within each modality are then computed. The goal is to find a permutation $\pi^*$ that minimizes the distortion between $\mathbf{X}$ and the permuted $\mathbf{Y}$. This is an NP-hard Quadratic Assignment Problem (QAP), for which this paper proposes a factored Hahn-Grant solver to obtain an efficient approximate solution.

### Key Designs

1. **QAP Formulation**:
    - Function: Formalizing unsupervised vision-language matching as a mathematical optimization problem.
    - Mechanism: Defining the pairwise distortion $\mathcal{D}_l(\mathbf{X}, \mathbf{Y}) = \sum_{i,j} l(\mathbf{X}_{ij}, \mathbf{Y}_{ij})$, where $l(\cdot,\cdot)$ is a metric function (e.g., Gromov-Wasserstein distance). The goal is to find the optimal permutation $\pi^*$: $\pi^* = \arg\min_{\pi} \sum_{i,j} l(\mathbf{X}_{ij}, \mathbf{Y}_{\pi(i)\pi(j)})$. By replacing the permutation function with a permutation matrix, the standard QAP formulation is obtained.
    - Design Motivation: QAP depends only on the **internal** pairwise distances of each modality, without requiring any cross-modal parallel information. Experiments verify a monotonic relationship between distortion and matching accuracy—as shuffling increases, alignment strictly decreases.

2. **Factored Hahn-Grant Solver**:
    - Function: Efficiently solving the QAP for vision-language matching.
    - Mechanism: The original Hahn-Grant solver requires storing an $O(N^4)$ cost tensor. This paper leverages the fact that common distance metrics (e.g., squared L2 distance, negative Frobenius inner product) can be **factored** into the structure $l(A,B) = f_1(A) + f_2(B) - h_1(A)h_2(B)$. This transforms the QAP into Koopmans-Beckmann form, requiring only two $N \times N$ cost matrices. By separately storing the dual variables $\mathbf{U}, \mathbf{V} \in \mathbb{R}^{N \times N \times (N-1)}$, memory is reduced to $O(N^3)$. Three key improvements are introduced: (1) Primal heuristic search—initialized with FAQ and 2-opt, recycling permutation matrices from each LAP solution; (2) Factored storage to reduce memory; (3) Replacing the Hungarian algorithm with the Jonker-Volgenant algorithm to accelerate LAP optimization.
    - Design Motivation: The $O(N^4)$ memory of the original Hahn-Grant becomes intractable when N > 50, and without primal search, it fails to output high-quality matching results.

3. **Optimal Matching Subproblem Selection**:
    - Function: Identifying the most suitable subset of classes from a large set for matching.
    - Mechanism: Modeling subset selection as a p-dispersion-sum problem: $\mathbf{s}^* = \arg\max_{\mathbf{s}} \sum_{i,j} l(\mathbf{X}_{ij}, \mathbf{Y}_{ij}) s_i s_j$, subject to $\mathbf{s} \in \{0,1\}^L$ and $\mathbf{s}^T \mathbb{1} = N$. This is solved using Gurobi.
    - Design Motivation: Not all classes exhibit similar pairwise relationships in vision and language. Selecting well-aligned subsets allows high matching accuracy to be maintained even under larger problem scales.

### Loss & Training

The proposed method **does not require any training**; it is a pure optimization method. Gromov-Wasserstein distance is used as the distortion metric (which outperforms CKA in experiments). Vision embeddings are obtained by averaging features of images per class, and language embeddings are obtained by averaging text features from multiple prompts, both normalized using $L_2$ normalization.

## Key Experimental Results

### Main Results (Small-scale Matching, CIFAR-10, N=10)

| Solver | Accuracy (%) | GW Cost | Reached Global Optimum (%) |
|--------|----------|---------|---------------|
| Random | 6.5 | 1.814 | 0.0 |
| LocalCKA | 18.5 | 0.530 | 5.0 |
| OT (GW) | 33.5 | 1.311 | 0.0 |
| FAQ | 38.0 | 0.546 | 0.0 |
| MPOpt | 94.0 | 0.325 | 90.0 |
| Gurobi | **100.0** | **0.319** | **100.0** |
| **Ours** | **100.0** | **0.319** | **100.0** |

### Vision Model Comparison (CIFAR-10, N=10)

| Vision Model | Matching Accuracy | Note |
|----------|----------|------|
| DINOv2 | **80-100%** | Self-supervised model performs best |
| CLIP | 60-80% | Vision-language joint training |
| DeiT | 50-70% | Supervised training |
| Random Initialization | ~10% | Baseline |

### Unsupervised Classification (CIFAR-10)

| Vision Model | All-Roberta-large (%) | all-mpnet-base (%) |
|----------|---------------------|-------------------|
| CLIP | 28.6 | 28.8 |
| DeiT | 45.4 | 26.8 |
| DINOv2 | **51.1** | 37.3 |
| Random Baseline | 10.0 | 10.0 |

### Key Findings

- **Only global optima have meaning**: The matching accuracy of local optima (obtained from OT, FAQ) is close to random, indicating that the energy landscape of QAP contains a massive number of low-quality local minima.
- **Pre-training strategy is more important than model size**: DINOv2 outperforms the second-best pre-training strategy by an average of 5.3% (CIFAR-10) and 7.6% (CINIC-10) across all language models.
- **Large-scale matching remains challenging**: On CIFAR-100, when N > 40, Gurobi cannot solve the problem within a 1.5-hour time limit, yet the proposed method still yields the tightest bound.
- **CLIP is superior in fine-grained matching**: On ImageNet-100, CLIP outperforms DINOv2 under larger problem sizes, suggesting that language supervision helps align fine-grained pairwise relations.
- **Unsupervised classification is feasible**: DINOv2 + K-Means + QAP matching reaches 51.1% accuracy (far exceeding the 10% random baseline), which represents the first completely unsupervised image classification.

## Highlights & Insights

- **Bold problem formulation**: Establishing vision-language matching without any parallel data, validating the practical feasibility of the Platonic Representation Hypothesis.
- **Bridge between theory and practice**: Introducing operations research QAP theory into the CV/NLP intersection, with a factored solver that reduces memory consumption by an order of magnitude.
- **Large-scale experimental design**: A systematic study involving 33 $\times$ 27 = 891 model combinations, offering panoramic insights into cross-modal agreement.
- **Proof-of-concept for unsupervised classification**: Vis-a-vis supervised methods, it is still behind, but it proves for the first time the feasibility of completely unsupervised classification.

## Limitations & Future Work

- Computational complexity is $O(N^5)$, making it infeasible when N > 100.
- Abstract concepts (e.g., "freedom of speech") do not have direct visual counterparts, making them fundamentally unmatchable.
- Symmetry issues: Multiple equivalent local minima can cause ambiguity.
- Optimal subset selection depends on the commercial Gurobi solver.
- Unsupervised classification accuracy is still far below supervised methods, limiting its practical utility.

## Related Work & Insights

- Directly validates the core prediction of the Platonic Representation Hypothesis.
- Related to the application of Gromov-Wasserstein in cross-lingual alignment, but applied to vision-language matching for the first time.
- The factored Hahn-Grant solver can be generalized to other CV problems requiring QAP (e.g., graph matching, point cloud registration).
- Insight: As foundation models mature, cross-modal knowledge transfer may rely less and less on explicit parallel data.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ An entirely new problem setup; the first systematic study on fully unsupervised vision-language matching.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 33 $\times$ 27 model combinations, 4 datasets, comparisons with multiple solvers, and unsupervised classification applications.
- Writing Quality: ⭐⭐⭐⭐ Mathematically rigorous, though the QAP formulation details might be challenging for CV readers to parse.
- Value: ⭐⭐⭐⭐ Deep theoretical insights, though practical utility is constrained by computational complexity and problem scale.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)
- [\[CVPR 2025\] Self-Supervised Spatial Correspondence Across Modalities](self-supervised_spatial_correspondence_across_modalities.md)
- [\[CVPR 2025\] Synthetic Data is an Elegant GIFT for Continual Vision-Language Models](synthetic_data_is_an_elegant_gift_for_continual_vision-language_models.md)
- [\[CVPR 2025\] MLLM-as-a-Judge for Image Safety without Human Labeling](mllm-as-a-judge_for_image_safety_without_human_labeling.md)
- [\[CVPR 2025\] Molmo and PixMo: Open Weights and Open Data for State-of-the-Art Vision-Language Models](molmo_and_pixmo_open_weights_and_open_data_for_state-of-the-art_vision-language_.md)

</div>

<!-- RELATED:END -->
