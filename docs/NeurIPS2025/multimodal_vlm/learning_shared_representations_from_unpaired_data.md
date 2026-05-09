---
title: >-
  [Paper Note] Learning Shared Representations from Unpaired Data
description: >-
  [NeurIPS 2025][Multimodal VLM][spectral embedding] This paper proposes SUE (Spectral Universal Embedding), which is the first to demonstrate that cross-modal shared representations can be learned with almost entirely unpaired data. Independent spectral embeddings extract modality-invariant "universal" structure from random walks within each modality; a minimal number of paired samples (~100 pairs) then enables CCA-based linear alignment followed by MMD-based nonlinear fine-tuning. SUE outperforms contrastive learning using the same number of pairs by more than 250% on retrieval benchmarks.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - spectral embedding
  - universal embedding
  - unpaired multimodal learning
  - MMD
  - CCA
date: 2026-05-08
content_hash: a0b45b851c098408
---

# Learning Shared Representations from Unpaired Data

**Conference**: NeurIPS 2025
**arXiv**: [2505.21524](https://arxiv.org/abs/2505.21524)
**Code**: [https://shaham-lab.github.io/SUE_page](https://shaham-lab.github.io/SUE_page)
**Area**: Multimodal VLM / Cross-Modal Representation Learning / Weakly Paired Learning
**Keywords**: spectral embedding, universal embedding, unpaired multimodal learning, MMD, CCA

## TL;DR
This paper proposes SUE (Spectral Universal Embedding), which is the first to demonstrate that cross-modal shared representations can be learned with almost entirely unpaired data. Independent spectral embeddings extract modality-invariant "universal" structure from random walks within each modality; a minimal number of paired samples (~100 pairs) then enables CCA-based linear alignment followed by MMD-based nonlinear fine-tuning. SUE outperforms contrastive learning using the same number of pairs by more than 250% on retrieval benchmarks.

## Background & Motivation

**Background**: Cross-modal shared representation learning is a central task in multimodal learning. Current state-of-the-art methods such as CLIP rely on massive paired datasets (400 million image–text pairs) for contrastive training.

**Limitations of Prior Work**: Acquiring paired data is prohibitively expensive. In domains such as medical imaging, remote sensing, and speech processing, paired annotations require expert involvement or specialized equipment, making large-scale paired data practically infeasible. In contrast, unpaired data from individual modalities is comparatively abundant.

**Key Challenge**: The core supervisory signal in contrastive learning derives from pairing relationships; without pairs, cross-modal correspondence cannot be established. Intuitively, linking modalities without paired information seems impossible.

**Key Insight**: The authors introduce the concept of a *universal embedding*—if pretrained unimodal representations already encode semantic similarity well, then the random walk processes independently constructed from each modality should be highly similar. This similarity can be captured via spectral embeddings without any paired data.

**Core Idea**: The diffusion operators of independent modalities share similar eigenfunctions (modality invariance), and spectral embeddings can extract this universal structure from unpaired data alone.

## Method

### Overall Architecture
SUE operates as a three-stage pipeline:
1. **Spectral Embedding (SE)**: Independently computes spectral embeddings from pretrained unimodal features to extract modality-invariant structure.
2. **CCA**: Applies linear alignment using a minimal number of paired samples to resolve sign/basis ambiguity in SE.
3. **MMD-net**: Further aligns the two distributions using a nonlinear residual network.

**Input**: Large quantities of unpaired unimodal pretrained embeddings $\mathcal{X}, \mathcal{Y}$, plus a small number of paired samples $\mathcal{X}_p, \mathcal{Y}_p$ ($m \ll n$). **Output**: Universal embedding mappings $f_\mathcal{X}, f_\mathcal{Y}$.

### Key Designs

1. **Universality of Spectral Embedding (SE)**:

    - **Function**: Proves that SE computed independently from different modalities can capture the same semantic structure.
    - **Mechanism**: Let $\mathcal{M}$ denote the underlying semantic manifold, and let $f, g$ be mappings that transform $\mathcal{M}$ into two respective modalities. If $f$ and $g$ have bounded distortion and bounded Ricci curvature, then the Laplace-Beltrami operators on $f(\mathcal{M})$ and $g(\mathcal{M})$ possess eigenfunctions that are similar in the $L_\infty$ sense. In practice, the random walk matrix $P = D^{-1}W$ converges to a diffusion operator, and the leading $k$ nontrivial eigenvectors of SE provide a discrete approximation to the eigenfunctions of this operator.
    - **Design Motivation**: Modern pretrained unimodal models (e.g., CLIP visual encoder, BERT text encoder) already encode semantic similarity effectively, so the random walks derived from them are indeed highly similar—as verified experimentally.

2. **Parametric Spectral Embedding (SpectralNet)**:

    - **Function**: Computes SE via deep learning to achieve generalizability and scalability beyond traditional SE.
    - **Mechanism**: Learns a parametric mapping $f: \mathbb{R}^d \to \mathbb{R}^k$ by minimizing the Rayleigh quotient $\mathcal{L}_{\text{spectralnet}}(f) = \frac{1}{n^2}\text{Trace}(f(X)^T L f(X))$ subject to the orthogonality constraint $f(X)^T f(X) = I_k$, where $L = I - P$ is the random walk graph Laplacian. The embeddings $S_\mathcal{X}$ and $S_\mathcal{Y}$ for the two modalities are trained entirely independently.
    - **Design Motivation**: Traditional SE does not generalize to new samples, whereas the parametric mapping learned by SpectralNet can be directly applied to test data.

3. **CCA Linear Alignment**:

    - **Function**: Resolves the non-uniqueness of SE (sign flips and basis rotations).
    - **Mechanism**: CCA is performed on the small set of paired samples $(S_\mathcal{X}(\mathcal{X}_p), S_\mathcal{Y}(\mathcal{Y}_p))$ to obtain projection matrices $Q_\mathcal{X}, Q_\mathcal{Y} \in \mathbb{R}^{k \times r}$. The aligned embeddings are $\tilde{S}_\mathcal{X} = Q_\mathcal{X} \circ S_\mathcal{X}$ and $\tilde{S}_\mathcal{Y} = Q_\mathcal{Y} \circ S_\mathcal{Y}$.
    - **Design Motivation**: The eigenvectors of SE are direction- and basis-ambiguous; CCA resolves this ambiguity using the minimum number of paired samples.

4. **MMD-net Nonlinear Alignment**:

    - **Function**: Fine-tunes the distributional alignment between the two modalities.
    - **Mechanism**: A residual network $F_\theta: \mathbb{R}^r \to \mathbb{R}^r$ is trained to minimize the empirical MMD: $\mathcal{L}_{\text{MMD}} = \frac{1}{m_1^2}\sum_{x_i,x_j}\kappa(\tilde{x_i},\tilde{x_j}) - \frac{2}{m_1 m_2}\sum_{x_i,y_j}\kappa(\tilde{x_i},\tilde{y_j}) + \frac{1}{m_2^2}\sum_{y_i,y_j}\kappa(\tilde{y_i},\tilde{y_j})$, where $\kappa$ is an RBF kernel.
    - **Design Motivation**: CCA performs only linear alignment, which is insufficient for precise correspondence. Since the MMD loss **requires no paired data**, it enables the full unpaired dataset to be exploited.

### Final Mapping
$f_\mathcal{X} = Q_\mathcal{X} \circ S_\mathcal{X}$, $\quad f_\mathcal{Y} = F_\theta \circ Q_\mathcal{Y} \circ S_\mathcal{Y}$

## Key Experimental Results

### Main Results — Cross-Modal Retrieval (Recall@k)

| Dataset | #Pairs | Task | SUE R@1 | SUE R@10 | Contrastive R@1 | Contrastive R@10 | SUE Gain |
|--------|-------|------|---------|----------|-------------|-------------|---------|
| MSCOCO | 100 | I2T | 5.75 | 34.25 | 1.50 | 13.00 | +257% |
| MSCOCO | 100 | T2I | 5.25 | 33.25 | 0.80 | 12.20 | +257% |
| Flickr30k | 500 | I2T | 4.25 | 32.00 | 3.00 | 16.20 | +103% |
| Flickr30k | 500 | T2I | 5.75 | 32.75 | 2.50 | 15.00 | +103% |
| Edges2Shoes | 50 | E2S | 4.00 | 25.25 | 1.00 | 14.00 | +201% |
| Handwritten | 100 | K2P | 25.50 | 79.00 | 4.80 | 28.00 | +284% |

### Ablation Study — Component Contributions (Flickr30k & MSCOCO T2I R@10)

| Configuration | Flickr30k (w/o SE) | Flickr30k (+SE) | MSCOCO (w/o SE) | MSCOCO (+SE) |
|------|-----------------|-----------------|---------------|-------------|
| Raw representations | 2.25 | 8.75 | 1.50 | 4.25 |
| +MMD | 3.75 | 5.50 | 2.00 | 3.75 |
| +CCA | 4.50 | 30.25 | 7.75 | 31.50 |
| +CCA+MMD | 4.75 | **32.75** | 9.75 | **33.25** |

### Key Findings
- **SE is the core component**: Without SE, CCA+MMD achieves only R@10 = 4.75 on Flickr30k; adding SE boosts this to 32.75—SE accounts for the overwhelming majority of performance.
- Replacing SE with an AutoEncoder leads to a significant performance drop, confirming that the universality of SE is irreplaceable.
- SUE with 100 paired samples is approximately equivalent to contrastive learning with 1,000+ pairs, indicating that the value of unpaired data has been severely underestimated.
- Increasing the amount of unpaired data consistently improves retrieval performance (Fig. 5b), whereas increasing paired data beyond the minimum threshold yields negligible gains (Fig. 5c).
- SUE even enables near-text-free text-to-image generation and semantic arithmetic (vector addition = semantic composition).

## Highlights & Insights
- **Paradigm shift**: From "large paired data is necessary" to "almost exclusively unpaired data suffices"—this has profound implications for resource-constrained domains such as medical imaging and low-resource languages.
- The **universality of SE** is striking: independently trained spectral embeddings from different modalities achieve high alignment, validated both theoretically (preservation of diffusion operator eigenfunctions) and empirically.
- **Elegant use of MMD loss**: As a distribution alignment objective requiring no paired data, it enables the full unpaired dataset to be leveraged.
- **Semantic arithmetic** (text embedding + image embedding = compositionally meaningful image) emerges under near-zero pairing conditions, revealing the deep structural properties of the universal embedding space.

## Limitations & Future Work
- Absolute retrieval performance remains far below large-scale paired SOTA methods such as CLIP (R@10 ≈ 33 vs. CLIP >> 90); the authors acknowledge that SUE is not intended to replace large paired models.
- Evaluation is limited to vision–language and vision–vision settings; extension to more complex modalities such as video, speech, and scientific data is needed.
- The computational complexity of SE (eigendecomposition) may become a bottleneck for very large datasets, though SpectralNet partially alleviates this issue.
- Whether the theoretical assumptions of bounded distortion and bounded Ricci curvature hold in practice for pretrained models remains to be rigorously verified.
- The CCA step still requires a minimum of ~100–500 paired samples; fully zero-paired scenarios remain unexplored.

## Related Work & Insights
- **vs. CLIP**: CLIP is trained contrastively on 400 million pairs; SUE achieves meaningful cross-modal representations with only 100 pairs, targeting a fundamentally different data availability regime.
- **vs. CSA**: Although CSA is designed for small paired sets, it operates exclusively on paired data without leveraging unpaired data; SUE's strength derives primarily from unpaired data.
- **vs. MACK**: MACK uses segmentation–text paired models for image–text alignment and still depends on paired-trained submodules; all three components of SUE can operate on unpaired data.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Provides a complete theoretical and empirical demonstration that cross-modal shared representations can be learned from unpaired data; conceptually groundbreaking.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers retrieval, generation, arithmetic, zero-shot, and classification tasks with clear ablations, though absolute performance remains limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Theoretical derivations are rigorous, intuitions are clearly articulated, figures are elegant, and the narrative flows smoothly from theory to experiments.
- **Value**: ⭐⭐⭐⭐ — Opens a new paradigm for unpaired multimodal learning with significant implications for data-scarce domains, though a gap to practical deployment remains.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] MIDAS: Misalignment-based Data Augmentation Strategy for Imbalanced Multimodal Learning](midas_misalignment-based_data_augmentation_strategy_for_imbalanced_multimodal_le.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[NeurIPS 2025\] FOCUS: Internal MLLM Representations for Efficient Fine-Grained Visual Question Answering](focus_internal_mllm_representations_for_efficient_fine-grained_visual_question_a.md)
- [\[ICCV 2025\] BabyVLM: Data-Efficient Pretraining of VLMs Inspired by Infant Learning](../../ICCV2025/multimodal_vlm/babyvlm_data-efficient_pretraining_of_vlms_inspired_by_infant_learning.md)
- [\[NeurIPS 2025\] STRUCTURE: With Limited Data for Multimodal Alignment, Let the Structure Guide You](with_limited_data_for_multimodal_alignment_let_the_structure_guide_you.md)

<!-- RELATED:END -->
