---
title: >-
  [Paper Note] Mass Concept Erasure in Diffusion Models with Concept Hierarchy
description: >-
  [AAAI2026][Image Generation][Concept Erasure] This paper proposes a grouped erasure strategy based on supertype-subtype concept hierarchy and Supertype-Preserving LoRA (SuPLoRA). By freezing the down-projection matrix (orthogonal to the supertype subspace) and training only the up-projection matrix, the method achieves an optimal balance between erasure effectiveness and generation quality in large-scale, multi-domain concept erasure.
tags:
  - AAAI2026
  - Image Generation
  - Concept Erasure
  - Diffusion Models
  - LoRA
  - Concept Hierarchy
  - Safe Generation
date: 2026-05-08
content_hash: dac323c0aa941093
---

# Mass Concept Erasure in Diffusion Models with Concept Hierarchy

**Conference**: AAAI2026
**arXiv**: [2601.03305](https://arxiv.org/abs/2601.03305)
**Authors**: Jiahang Tu, Ye Li, Yiming Wu, Hanbin Zhao, Chao Zhang, Hui Qian (Zhejiang University)
**Code**: [GitHub](https://github.com/TtuHamg/SuPLoRA)
**Area**: Image Generation
**Keywords**: Concept Erasure, Diffusion Models, LoRA, Concept Hierarchy, Safe Generation

## TL;DR

This paper proposes a grouped erasure strategy based on supertype-subtype concept hierarchy and Supertype-Preserving LoRA (SuPLoRA). By freezing the down-projection matrix (orthogonal to the supertype subspace) and training only the up-projection matrix, the method achieves an optimal balance between erasure effectiveness and generation quality in large-scale, multi-domain concept erasure.

## Background & Motivation

### State of the Field
Diffusion models (e.g., Stable Diffusion) learn undesirable concepts from large-scale unfiltered datasets, including copyrighted material, offensive content, and sensitive personal information. Such models may generate unsafe content even after data cleaning. Concept erasure methods suppress the generation of specific concepts through fine-tuning.

### Limitations of Prior Work
- **Poor parameter efficiency**: Each erased concept requires an independent set of fine-tuning parameters, causing parameter count to grow linearly with the number of concepts (e.g., MACE requires 198 MB to erase 64 concepts).
- **Generation quality degradation**: Repeated erasure suppresses visual features that are critical not only to individual concepts but also to supertype concepts (e.g., "person").
- **Cross-domain interference**: Erasing concepts from one domain inadvertently degrades generation capability in another domain.
- **Lack of challenging benchmarks**: Existing benchmarks only erase concepts from a single category.

### Core Idea
The method exploits semantic relationships among concepts to be erased by constructing a hierarchical structure, grouping semantically similar concepts to share parameters during erasure, while protecting the generation capability of supertype concepts through theoretically grounded subspace constraints.

## Core Problem

1. How to simultaneously maintain parameter efficiency and generation quality in large-scale concept erasure?
2. How to prevent degradation of supertype concept generation when erasing subtype concepts?
3. How to construct a unified erasure framework spanning multiple domains (celebrities, objects, and explicit content)?

## Method

### Concept Hierarchy Construction (Sec 3.1)
CLIP is used to compute semantic similarity among concepts → clustering → GPT-4 generates supertype labels:
- Example: {jay, macaw, bald eagle} → supertype "bird"
- Example: {Adam Driver, Adriana Lima, ...} → supertype "person"
- Hierarchical relation: $\mathcal{G}_j = \{c_i^t \in \mathcal{C}^t \mid g(c_i^t) = c_j^p\}$

### Grouped Suppression (Sec 3.2)
Attention suppression is based on MACE but operates at the **supertype level** rather than the individual concept level; concepts within the same group share a single set of LoRA parameters.

Erasure loss — minimizes attention from concept tokens to relevant regions:

$$\mathcal{L}_{\text{attn}} = \mathbb{E}_{c_i \in \mathcal{G}_j, t, l}\left[\|\boldsymbol{\alpha}_{c_i}^{t,l}(\mathbf{A}_j) \odot \mathbf{M}_{c_i}\|_F^2\right]$$

Diffusion regularization — preserves denoising capability in non-erased regions:

$$\mathcal{L}_{\text{Diff}} = \mathbb{E}_{c_i \in \mathcal{G}_j, t, \boldsymbol{\epsilon}}\left[\|(1 - \mathbf{M}_{c_i}) \odot (\boldsymbol{\epsilon} - \epsilon_\theta(\mathbf{z}_t, t, \mathcal{T}_{c_i} | \mathbf{A}_j))\|_2^2\right]$$

Total loss: $\mathcal{L} = \mathcal{L}_{\text{attn}} + \lambda \mathcal{L}_{\text{Diff}}$

### SuPLoRA Design (Sec 3.3)

**Key theoretical derivation**: Comparison between directly modifying $\mathbf{W}$ and training only $\mathbf{A}_j$ (with $\mathbf{B}_j$ frozen).

The weight update when directly modifying $\mathbf{W}$ is:

$$\Delta_{\mathbf{W}}\mathbf{W}' = -\alpha \frac{\partial \mathcal{L}}{\partial \mathbf{o}_j}\mathbf{h}_j^T$$

The weight update when training only $\mathbf{A}_j$ (with $\mathbf{B}_j$ frozen) is:

$$\Delta_{\mathbf{A}_j}\mathbf{W}' = \Delta_{\mathbf{W}}\mathbf{W}' \cdot \mathbf{B}_j^T\mathbf{B}_j$$

**Core Idea**: Training $\mathbf{A}_j$ is equivalent to modifying weights within the subspace $\mathcal{S}_j^\perp$ defined by $\mathbf{B}_j^T\mathbf{B}_j$. If $\mathcal{S}_j^\perp$ is orthogonal to the supertype gradient subspace $\mathcal{S}_j$, erasure updates do not interfere with supertype generation.

**$\mathbf{B}_j$ initialization**:
1. Collect text embeddings $\mathbf{H}_{S_j}$ from supertype concept descriptions.
2. Apply SVD to obtain the supertype gradient subspace $\mathcal{S}_j = \text{span}\{\mathbf{u}_{1,j}, ..., \mathbf{u}_{r,j}\}$.
3. Compute the orthogonal complement $\mathcal{S}_j^\perp$ (null space of $\mathcal{S}_j$).
4. Set $\mathbf{B}_j$ to the basis of $\mathcal{S}_j^\perp$, freeze $\mathbf{B}_j$, and train only $\mathbf{A}_j$.

### Knowledge Distillation Merging
$K$ SuPLoRA modules are merged into a unified weight $\mathbf{W}^*$ via distillation:

$$\min_{\mathbf{W}^*} \underbrace{\mathbb{E}_{i,j}\|\mathbf{W}^*\mathbf{e}_{j,i}^t - (\mathbf{W} + \mathbf{A}_j\mathbf{B}_j)\mathbf{e}_{j,i}^t\|_2^2}_{\text{target alignment}} + \underbrace{\mathbb{E}_l\|\mathbf{W}^*\mathbf{e}_l^g - \mathbf{W}\mathbf{e}_l^g\|_2^2}_{\text{generality consistency}}$$

## Key Experimental Results

### Benchmark Setup
- Model: Stable Diffusion v1.4, DDIM 50 steps
- Erasure scope: 30 celebrities + 30 objects + 4 explicit concepts = 64 concepts total
- Evaluation: ViT-L/16 classifier (88.06% top-1), GCD celebrity classification, NudeNet explicit content detection

### Main Results (64 concepts erased simultaneously)

| Method | Celebrity Acc↓ | Object Acc↓ | NN↓ | In-domain Celebrity Acc↑ | In-domain Object Acc↑ | FID↓ | CLIP Score↑ | Supertype CLIP↑ | Storage (MB)↓ | Time (min)↓ |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| ESD-u | 0.00% | 1.25% | 59 | 0.50% | 7.63% | 34.59 | 25.21 | 22.05 | 3379 | 2166 |
| UCE | 9.87% | 7.81% | 163 | 73.62% | 47.87% | 18.51 | 29.80 | 24.81 | 3379 | 218 |
| MACE | 6.25% | 9.17% | 158 | 78.50% | 50.63% | 18.36 | 30.04 | 25.51 | 198 | 20 |
| SPM | 10.00% | 65.00% | 639 | 78.50% | 63.50% | 21.15 | 30.59 | 26.00 | 218 | 20 |
| **Ours** | **7.50%** | **4.17%** | **121** | **83.38%** | **65.00%** | **17.92** | **30.68** | **26.09** | **154** | **18** |

### Ablation Study on SuPLoRA

| Configuration | In-domain Celebrity/Object Acc↑ | FID↓ | CLIP Score↑ | Supertype CLIP↑ |
|------|:---:|:---:|:---:|:---:|
| Default LoRA (train A+B) | 79.12%/56.50% | 18.18 | 30.18 | 25.19 |
| Default LoRA, freeze random B | 81.12%/59.87% | 18.13 | 30.65 | 26.08 |
| SuPLoRA, train B | 79.83%/57.01% | 18.23 | 30.25 | 25.22 |
| **SuPLoRA (full)** | **83.38%/61.50%** | **17.94** | **30.66** | **26.21** |

### Scalability Experiment (vs. MACE)

| Setting (Celebrity/Object) | Method | In-domain Object Acc↑ | Supertype CLIP↑ |
|:---:|------|:---:|:---:|
| 0/10 | MACE | 92.87% | 26.58 |
| 0/10 | Ours | **93.38%** | **26.97** |
| 20/20 | MACE | 59.12% | 25.91 |
| 20/20 | Ours | **73.88%** | **26.33** |

Under the 20/20 setting, in-domain object retention improves by **+14.76%**.

## Highlights & Insights

- **Concept hierarchy design**: The first work to leverage supertype-subtype semantic structure to organize concepts for erasure, reducing the number of parameter sets from the number of concepts $N$ to the number of groups $K$ (64 → approximately 6 groups).
- **Theoretically guaranteed subspace protection**: SuPLoRA analyzes gradient subspace orthogonality to prove that freezing an orthogonally initialized $\mathbf{B}_j$ prevents supertype degradation.
- **Cross-domain benchmark**: Constructs the first large-scale erasure evaluation spanning three domains simultaneously: celebrities, objects, and explicit content.
- **Significant storage efficiency**: 154 MB vs. MACE 198 MB vs. UCE 3379 MB.
- **Fastest training speed**: 18 min vs. MACE 20 min vs. UCE 218 min.

## Limitations & Future Work

- **Reliance on shared supertype structure**: Grouping effectiveness diminishes when the concepts to be erased lack semantic relationships.
- **Two-level hierarchy constraint**: Only a parent-child two-level hierarchy is constructed; more complex multi-level hierarchies are explored in the appendix but not thoroughly validated.
- **Limited to SD v1.4**: Validation is only conducted on Stable Diffusion v1.4; the method has not been tested on newer architectures such as SDXL or Flux.
- **Style domain not covered**: Artistic style generation is unstable in SD v1.4, so style erasure evaluation is excluded.
- **GPT-4 dependency**: Hierarchy construction and prompt augmentation rely on GPT-4, introducing external API costs.
- **Adversarial robustness not evaluated**: Erasure persistence under red-teaming attacks is not assessed.

## Related Work & Insights

- **vs. ESD**: ESD aligns erased concepts to their supertypes (e.g., "grumpy cat" → "cat"), but aggressive erasure causes generation collapse; this paper protects supertype generation.
- **vs. MACE**: MACE assigns independent LoRA parameters to each concept, causing storage to grow linearly; this paper uses grouped sharing, reducing parameters to approximately one-quarter.
- **vs. UCE**: UCE balances erasure and retention via a closed-form solution but incurs enormous storage overhead (3379 MB); this paper requires only 154 MB.
- **vs. SPM**: SPM protects unrelated concepts via an anchoring loss but performs extremely poorly on explicit content detection (NN = 639 vs. 121 for this paper).
- **vs. ConceptPrune**: ConceptPrune prunes "expert neurons" but is validated on only 10 categories; this paper validates on 64 cross-domain concepts.
- **vs. CE-SDWV**: Inference-time interventions can be bypassed (by disabling the module); this paper modifies model weights irreversibly.

### Broader Implications
- The subspace protection mechanism of SuPLoRA can be generalized to mitigate inter-task interference in continual learning.
- The concept hierarchy construction method is applicable to other model editing tasks that require structured knowledge management.
- The grouped erasure strategy has direct practical value for large-scale safe model deployment.
- Gradient subspace orthogonality analysis provides a theoretical tool for addressing task conflicts in LoRA fine-tuning.

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of concept hierarchy and subspace protection is novel, with rigorous theoretical analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Cross-domain benchmark, multi-baseline comparison, comprehensive ablations, and scalability experiments.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, rigorous theoretical derivations, and well-coordinated equations and figures.
- Value: ⭐⭐⭐⭐ — Addresses practical challenges in safe deployment of diffusion models with strong scalability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Semantic Surgery: Zero-Shot Concept Erasure in Diffusion Models](../../NeurIPS2025/image_generation/semantic_surgery_zero-shot_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] Neighbor-Aware Localized Concept Erasure in Text-to-Image Diffusion Models](../../CVPR2026/image_generation/neighbor-aware_localized_concept_erasure_in_text-to-image_diffusion_models.md)
- [\[CVPR 2026\] Prototype-Guided Concept Erasure in Diffusion Models](../../CVPR2026/image_generation/prototype-guided_concept_erasure_in_diffusion_models.md)
- [\[CVPR 2026\] GrOCE: Graph-Guided Online Concept Erasure for Text-to-Image Diffusion Models](../../CVPR2026/image_generation/groce_graph-guided_online_concept_erasure_for_text-to-image_diffusion_models.md)
- [\[ICLR 2026\] SPEED: Scalable, Precise, and Efficient Concept Erasure for Diffusion Models](../../ICLR2026/image_generation/speed_scalable_precise_and_efficient_concept_erasure_for_diffusion_models.md)

</div>

<!-- RELATED:END -->
