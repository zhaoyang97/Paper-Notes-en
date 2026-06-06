---
title: >-
  [Paper Note] ConstStyle: Robust Domain Generalization with Unified Style Transformation
description: >-
  [ICCV 2025][LLM Pretraining][domain generalization] This paper proposes ConstStyle, a framework that constructs a theoretically grounded **Unified Domain** to which all training samples are style-aligned during training…
tags:
  - "ICCV 2025"
  - "LLM Pretraining"
  - "domain generalization"
  - "style transformation"
  - "unified domain"
  - "distribution alignment"
  - "robustness"
date: 2026-05-08
content_hash: 56730ba87a69603d
---

# ConstStyle: Robust Domain Generalization with Unified Style Transformation

**Conference**: ICCV 2025
**arXiv**: [2509.05975](https://arxiv.org/abs/2509.05975)  
**Code**: [https://github.com/nduongw/ConstStyle](https://github.com/nduongw/ConstStyle)  
**Area**: Domain Generalization
**Keywords**: domain generalization, style transformation, unified domain, distribution alignment, robustness

## TL;DR

This paper proposes ConstStyle, a framework that constructs a theoretically grounded **Unified Domain** to which all training samples are style-aligned during training, while test samples from unseen domains are partially projected toward this unified domain at inference time, effectively reducing the domain gap and improving generalization performance.

## Background & Motivation

Deep neural networks suffer significant performance degradation when the test distribution differs from the training distribution (domain shift). Existing domain generalization (DG) methods broadly follow two strategies: (1) learning domain-invariant features, and (2) data augmentation to increase diversity. However, both have notable limitations:

- Invariant representation learning requires a large number of diverse domains to extract effective invariant features.
- Data augmentation methods assume that training on more domains yields better performance, yet empirical evidence shows this does not always hold — training on fewer but carefully selected domains can sometimes produce better class separation.
- Most existing methods focus exclusively on the training phase, neglecting the test phase when the domain gap is most pronounced.

**Key insight**: Both training and testing should be conducted within a common *unified domain* to effectively reduce the inter-domain gap.

## Method

### Overall Architecture

ConstStyle operates in two phases. The **training phase** consists of (i) determining the unified domain style, (ii) transforming training samples' styles toward the unified domain, and (iii) training the model on the aligned samples. The **inference phase** partially aligns unseen-domain samples to the unified domain before prediction.

### Key Designs

1. **Unified Domain Determination**: The style statistics of each sample are defined as the concatenation of channel-wise mean and variance, $\epsilon_x = \text{concat}(\mu_x, \sigma_x)$. The domain style is modeled as a multivariate Gaussian $\mathcal{P}_S \sim \mathcal{N}(\epsilon_S, \Sigma_S)$. The unified domain style is defined as the Wasserstein barycenter of all observed domain style distributions, with mean $\epsilon_B = \frac{1}{N}\sum_{k=1}^{N}\epsilon_{S_k}$ and covariance solved via iterative optimization. Theorem 1 provides a theoretical bound showing that the empirical loss gap between a model trained on the unified domain and one trained on the original domains is bounded by inter-domain distances: $\sum_k(L^{S_k^T} - L^{S_k}) \leq \beta \times \sum_k(\mathcal{D}_\mu(\mathcal{T}, S_k) + \mathcal{D}_\sigma(\mathcal{T}, S_k))$. When domain labels are unavailable, GMM clustering over style statistics is used to approximate domain partitions.

2. **Two-Stage Training**: In **Stage 1**, the model undergoes initial ERM training on original data to obtain a style feature extractor $\theta_s$; style features of all samples are extracted and the initial unified domain is determined. In **Stage 2**, at each epoch, training samples' style statistics are aligned to the unified domain by sampling $(\mu_s, \sigma_s)$ from $\mathcal{N}(\epsilon_T, \Sigma_T)$ and applying an AdaIN-style transformation: $z_{x_i}^T = \sigma_s \times \frac{z_{x_i} - \mu_x}{\sigma_x} + \mu_s$. The unified domain is updated every $\gamma$ epochs. Over $E$ training epochs with $D$ domains, the procedure generates $E \times D$ distinct style variants, enhancing the model's adaptability to the unified domain style.

3. **Partial Alignment at Inference**: Rather than fully projecting unseen-domain samples onto the unified domain (which may discard original information), a partial projection strategy is employed to balance domain alignment and information preservation: $z_u^T = (\alpha \sigma_u + (1-\alpha)\sigma_T)\frac{z_u^o - \mu_u}{\sigma_u} + (\alpha \mu_u + (1-\alpha)\mu_T)$. The hyperparameter $\alpha \in (0,1)$ controls the degree to which original features are retained. Theorem 2 provides a performance guarantee: $L^{\mathcal{U}^T} - L^{\mathcal{S}^T} \leq \alpha \beta (\mathcal{D}_\mu(\mathcal{U},\mathcal{T}) + \mathcal{D}_\sigma(\mathcal{U},\mathcal{T})) + \epsilon\sqrt{2 \cdot Tr(I)}$, indicating that smaller $\alpha$ reduces the influence of the domain gap while potentially increasing information loss.

### Loss & Training

Standard cross-entropy loss is used for classification. Key training strategies include: initial training for only a few epochs (without full convergence) to efficiently establish the initial unified domain; and periodic updates of the unified domain (every $\gamma$ epochs) to improve quality while maintaining training stability.

## Key Experimental Results

### Main Results

| Dataset | Metric | ConstStyle | Prev. SOTA (CSU) | Gain |
|--------|------|------------|-----------------|------|
| PACS (avg. single unseen domain) | Accuracy | 86.77% | CSU: 85.33% | +1.44% |
| PACS (Sketch domain) | Accuracy | 82.32% | CSU: 78.11% | +4.21% |
| Digits5 (average) | Accuracy | 86.88% | EDFMix: 86.14% | +0.74% |
| PACS In-domain | Accuracy | 96.50% | DSU: 96.30% | +0.20% |

ConstStyle achieves the most significant improvements in scenarios with the largest domain gaps (e.g., the Sketch domain).

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| Multi-unseen-domain avg. (2 domains) | +2.43% accuracy | Advantage maintained even with fewer training domains |
| Cartoon+Sketch unseen domains | +5.91% accuracy | Largest gains under large domain gaps |
| Largest domain gap (Photo→Sketch) | +15.03% over CSU | Validates core advantage of unified domain strategy |
| α=0 (full projection) | Performance drop | Over-alignment discards original information |
| α=1 (no projection) | Performance drop | Unified domain alignment not utilized |
| Few-source-domain training | Up to +19.82% gain | Over second-best method in extreme few-domain settings |

### Key Findings

- More training domains do not always yield better generalization: single-domain training can sometimes produce clearer class boundaries.
- The larger the domain gap, the more pronounced ConstStyle's advantage over competing methods (up to 15%+).
- The partial projection strategy ($\alpha=0.5$–$0.6$) achieves the best balance between information retention and domain alignment.
- In-domain performance is also improved (96.50% vs. 95.94% for ERM), demonstrating that unified domain training does not harm seen-domain performance.

## Highlights & Insights

- The unified domain concept constitutes an elegant theoretical framework, offering a mathematically principled domain selection method via the Wasserstein barycenter.
- The method addresses domain shift in both training and test phases simultaneously, unlike most DG methods that focus solely on training.
- The theoretical analysis is complete (Lemma 1 + Theorem 1/2), providing clear theoretical guidance for the method design.
- The partial projection strategy is simple yet effective, with a transparent physical interpretation of the hyperparameter $\alpha$.

## Limitations & Future Work

- Domain style is characterized solely by channel-wise mean and variance, which may fail to capture higher-order domain characteristics.
- GMM-based estimation of the number of domains introduces sensitivity to hyperparameters.
- Approximation of the Wasserstein barycenter may introduce numerical errors.
- Partial projection may still be insufficient under extreme domain gaps; an adaptive $\alpha$ could be explored.

## Related Work & Insights

- Unlike style augmentation methods such as MixStyle and DSU, which increase style diversity, ConstStyle converges all domains toward a single point — this "convergent" rather than "divergent" paradigm is a relatively distinctive perspective in DG.
- The test-time alignment idea can be combined with test-time training/adaptation approaches.
- The unified domain framework can be extended to other domain shift problems such as object detection and semantic segmentation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The unified domain concept is novel; the joint training-and-test treatment of domain shift is distinctive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluated across multiple datasets and diverse settings (few-source-domain, multi-unseen-domain, varying domain gaps).
- **Writing Quality**: ⭐⭐⭐⭐ Theory and experiments are tightly integrated with a clear overall structure.
- **Value**: ⭐⭐⭐⭐ The method is concise and effective with solid theoretical guarantees, broadly applicable to DG scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Superposition Yields Robust Neural Scaling](../../NeurIPS2025/llm_pretraining/superposition_yields_robust_neural_scaling.md)
- [\[NeurIPS 2025\] Learning the Wrong Lessons: Syntactic-Domain Spurious Correlations in Language Models](../../NeurIPS2025/llm_pretraining/learning_the_wrong_lessons_syntactic-domain_spurious_correlations_in_language_mo.md)
- [\[NeurIPS 2025\] Vocabulary Customization for Efficient Domain-Specific LLM Deployment](../../NeurIPS2025/llm_pretraining/vocabulary_customization_for_efficient_domain-specific_llm_deployment.md)
- [\[ICCV 2025\] ACE-G: Improving Generalization of Scene Coordinate Regression Through Query Pre-Training](aceg_improving_generalization_of_scene_coordinate_regression.md)
- [\[ICLR 2026\] SemHiTok: A Unified Image Tokenizer via Semantic-Guided Hierarchical Codebook](../../ICLR2026/llm_pretraining/semhitok_a_unified_image_tokenizer_via_semantic-guided_hierarchical_codebook_for.md)

</div>

<!-- RELATED:END -->
