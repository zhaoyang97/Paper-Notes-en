---
title: >-
  [Paper Note] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection
description: >-
  [CVPR 2026][Medical Imaging][CLIP adaptation] This paper proposes CHIPS, a curvature-aware hybrid influence-based data selection method that computes Newton-style alignment scores in the CLIP endpoint subspace and combines them with learnability and domain-relevance weights. Using only 30% of the data, CHIPS matches full-dataset continual pre-training (CPT) performance and achieves state-of-the-art results across 17 medical benchmarks.
tags:
  - CVPR 2026
  - Medical Imaging
  - CLIP adaptation
  - data selection
  - curvature-aware
  - continual pre-training
date: 2026-05-08
content_hash: b25c1bdb2dfa72c7
---

# CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection

**Conference**: CVPR 2026
**arXiv**: [2511.18519](https://arxiv.org/abs/2511.18519)
**Code**: [Available](https://github.com/mihara-bot/CHIPS)
**Area**: Medical Imaging
**Keywords**: CLIP adaptation, data selection, curvature-aware, continual pre-training, medical imaging

## TL;DR

This paper proposes CHIPS, a curvature-aware hybrid influence-based data selection method that computes Newton-style alignment scores in the CLIP endpoint subspace and combines them with learnability and domain-relevance weights. Using only 30% of the data, CHIPS matches full-dataset continual pre-training (CPT) performance and achieves state-of-the-art results across 17 medical benchmarks.

## Background & Motivation

### 1. State of the Field

Vision-language models such as CLIP demonstrate strong zero-shot recognition capabilities in general domains, but suffer significant performance degradation in specialized domains (e.g., medical imaging, biology), where vocabulary, acquisition protocols, and label taxonomies shift substantially. Two main paradigms exist for adapting CLIP to specialized domains: **model-centric** methods (probabilistic fine-tuning, PEFT variants, etc., which modify training or parameterization strategies) and **data-centric** methods (continual pre-training on large-scale domain data, ranging from millions to hundreds of millions of samples).

### 2. Limitations of Prior Work

Data-centric methods face severe **data efficiency** problems: collecting, annotating, and processing large-scale domain datasets is extremely costly, and indiscriminately scaling data volume may introduce redundant, uninformative samples that harm learning outcomes.

### 3. Root Cause

The tension between scale and efficiency—does effective CPT truly require extreme data scale? Existing data attribution methods (e.g., TracIn, TRAK) are designed for supervised classification on single-tower models and exhibit three fundamental mismatches when directly applied to CLIP:

- **(A) Cross-modal curvature in dual encoders**: CLIP's dual-encoder architecture produces non-block-diagonal second-order curvature; block-diagonal approximations ignore this coupling and lead to incorrect sample ranking.
- **(B) Non-local gradients under InfoNCE**: Each sample's gradient depends on the softmax normalizer over the entire negative set, making influence batch/globally dependent rather than per-sample additive.
- **(C) Dominance of endpoint projection heads**: Projection heads and temperature parameters drive early shifts in the similarity distribution, rendering full-parameter influence computation unnecessary for CLIP.

### 4. Paper Goals

Design a CLIP-specific data selector that achieves domain adaptation comparable to or better than full-scale CPT with a small data budget, while preserving general-domain capabilities.

### 5. Starting Point

The problem is framed from a **data attribution** perspective: select samples whose one-step update maximally reduces the evaluation loss on the target domain. The key insight is that this alignment score need only be computed in CLIP's endpoint subspace (projection heads + temperature).

### 6. Core Idea

The paper proposes CHIPS (Curvature-aware Hybrid Influence in Projection Subspace), which computes curvature-aware Newton-style alignment scores in CLIP's endpoint geometric space, combines them with an InfoNCE-aware curvature estimator (accelerated via JL sketching) and selection-aware domain-relevance weights, and derives a final per-sample utility score as their product.

## Method

### Overall Architecture

CHIPS computes a composite utility score for each training sample: $\mathcal{I}_{\text{CHIPS}}(z) = \hat{A}_\alpha(z) \cdot w_L(z) \cdot w_R(z)$, consisting of three tightly coupled components:

1. **Curvature-aware proxy alignment score** $\hat{A}_\alpha(z)$: measures the Newton-direction alignment between a sample's gradient and the evaluation gradient in the endpoint subspace.
2. **Learnability weight** $w_L(z)$: favors samples near the decision boundary, discounting samples already solved by the model.
3. **Target domain relevance weight** $w_R(z)$: a soft constraint ensuring the selected distribution does not deviate from the target domain.

The top-$n$ samples by utility score are selected for CPT.

### Key Designs

#### Design 1: Endpoint Subspace Curvature-Aware Alignment (Sec. 2.2)

**Function**: Computes a proxy alignment score over CLIP's endpoint parameters $\vartheta = \{W_v, W_t, \tau\}$ (visual/text projection heads + temperature).

**Mechanism**: The ideal update direction is the Newton step $H_\vartheta^{-1} u_\vartheta$; the alignment score is defined as $A(z) = g_\vartheta(z)^\top M^{-1} u_\vartheta$, where $M$ is a computable curvature proxy. A higher score indicates that the sample's one-step update moves the model further along the descent direction of the evaluation loss.

**Design Motivation**: Local linearization analysis (Theorem 1) establishes a lower bound on the Pearson correlation between endpoint-subspace alignment scores and full-parameter alignment scores. Empirically, the Spearman correlation reaches 0.83, confirming that endpoint-level ranking well preserves full-parameter ranking. The endpoint subspace is far smaller in dimension, substantially reducing computational cost.

#### Design 2: InfoNCE-Aware Curvature Estimation with JL Sketching (Sec. 2.3)

**Function**: Constructs a curvature matrix $M$ that encodes coupling information from both positive and negative pairs.

**Mechanism**: Computes self-curvature (outer product of positive-pair gradients) $\Phi_{\text{pos}}$ and cross-curvature (outer product of negative-pair gradients) $\Phi_{\text{neg}}$, combined with mixing weight $\alpha$:

$$M = (1-\alpha)\Phi_{\text{pos}} + \alpha\Phi_{\text{neg}} + \lambda I$$

JL random projection then compresses the dimension to $k$, yielding the sketched score $\hat{A}_\alpha(z)$.

**Design Motivation**: Symmetric InfoNCE couples each positive pair with multiple negatives through the softmax normalizer, producing cross-sample curvature. Diagonal-only proxies based solely on positive pairs (e.g., TracIn) miss this coupling and introduce ranking bias. Theorem 2 shows the error decomposes into an $O(1/k)$ projection variance term and a curvature bias term—$\alpha > 0$ recovers the off-diagonal mass from negative pairs, reducing curvature bias, while increasing $k$ reduces projection variance. The recommended range is $\alpha \in [0.6, 0.8]$.

#### Design 3: Learnability and Target Domain Relevance Weights (Sec. 2.4)

**Function**: Attaches two multiplicative weights to each sample to modulate the alignment score.

**Learnability $w_L(z)$**: Uses the average correct-pair probability $p_{\text{corr}}(z)$ under CLIP and the margin $m(z)$ of the hardest negative: $w_L(z) = (1 - p_{\text{corr}}(z))(1 + \sigma(-m(z)))$. High-confidence, correctly classified samples are down-weighted; samples near or below the decision boundary (small or negative margin) are up-weighted, as these are the most learnable in a single update step.

**Target Domain Relevance $w_R(z)$**: Computes mean embeddings $\mu_x, \mu_y$ of the evaluation set across both modalities, then $w_R(z) = \sigma((1-\beta)\cos(\hat{x}, \mu_x) + \beta\cos(\hat{y}, \mu_y))$. The sigmoid confines values to $[0.27, 0.73]$, implementing soft reweighting rather than hard filtering so that no sample's weight is zeroed out. Performance is maximized at $\beta = 0.5$.

**Design Motivation**: The alignment score measures gradient direction utility but does not distinguish between "solved" and "boundary" samples, nor does it compensate for distribution mismatch between the training pool and the evaluation set. Learnability focuses selection on the most informative samples; domain relevance prevents the selected distribution from drifting away from the target domain, thereby mitigating catastrophic forgetting.

### Loss & Training

CHIPS is a data selection method, not a training method. The selected subset is used for CPT with the standard symmetric InfoNCE loss:

- Optimizer: AdamW ($\beta_1=0.9, \beta_2=0.98, \epsilon=10^{-6}$)
- Learning rate schedule: cosine annealing (initial $10^{-6}$)
- Batch size: 32,768
- Training duration: fixed 5 epochs
- Hardware: 8× NVIDIA H200 (141 GB)

CHIPS scores are computed once and can be cached for reuse across different architectures and pre-training scales.

## Key Experimental Results

### Main Results

CPT on BIOMEDICA (24M samples) with MetaCLIP-B16-400M; average medical task performance at various retention ratios:

| Method | r=10% Medical Avg | r=20% Medical Avg | r=30% Medical Avg | r=10% General CLS |
|------|:---:|:---:|:---:|:---:|
| Full Dataset | 31.51 | 31.51 | 31.51 | 49.72 |
| Random | 24.78 | 25.00 | 26.28 | 52.21 |
| CLIPScore | 24.16 | 20.01 | 19.01 | 53.39 |
| TracIn | 26.46 | 26.63 | 25.68 | 47.26 |
| TRAK | 25.19 | 24.54 | 23.54 | 48.24 |
| **CHIPS** | **27.03** | **28.20** | **29.96** | 47.88 |

Key results: CHIPS at 10% (27.03) outperforms Random at 50% (26.26); CHIPS at 30% (29.96) achieves 95.1% of full-dataset CPT performance; at r=30%, CHIPS marginally surpasses the dedicated medical model BMCLIP (29.96 vs. 29.86).

Cross-architecture generalization (10% retention, CHIPS scores reused):

| Model | Medical CLS | General CLS | General RET |
|------|:---:|:---:|:---:|
| B32-400M Random | 27.15 | 49.31 | 27.33 |
| B32-400M CHIPS | **27.83** | 47.90 | 25.65 |
| L14-400M Random | 29.33 | 57.07 | 33.35 |
| L14-400M CHIPS | **29.73** | 53.65 | 28.17 |
| H14-CC Random | 35.23 | 61.36 | 32.82 |
| H14-CC CHIPS | **35.48** | 58.24 | 32.09 |

CHIPS achieves the best medical performance across all 7 architecture/pre-training scale configurations, outperforming TracIn by 0.20–2.65 points.

### Ablation Study

Incremental component addition on MetaCLIP-B16-400M:

| Variant | r=10% Med | r=20% Med | r=30% Med | r=10% Gen CLS |
|------|:---:|:---:|:---:|:---:|
| Alignment-only | 25.98 | 27.52 | 27.84 | 48.33 |
| Alignment+Margin | 25.95 | 27.92 | 28.50 | 48.41 |
| **CHIPS (full)** | **27.03** | **28.20** | **29.96** | 47.88 |

The three-component product is optimal across all budgets; at r=30%, it exceeds Alignment+Margin by +1.46 points, indicating that domain relevance is especially critical at larger budgets. The general-domain CLS gap is ≤0.53; the RET gap narrows as $r$ increases (0.99→0.37), suggesting controlled specialization rather than catastrophic forgetting.

### Key Findings

1. **High data efficiency**: 10% data outperforms 50% random sampling; 30% data achieves 95% of full-dataset performance.
2. **Reliable endpoint subspace proxy**: Spearman correlation of 0.83; the text projection head is most important (Text-only retains 99.7%), with the visual projection head being complementary (98.7%).
3. **Optimal curvature mixing range**: $\alpha \in [0.6, 0.8]$ is optimal, validating the importance of negative-pair coupling information for InfoNCE curvature.
4. **Score transferability**: Scores computed once on B16-400M can be directly reused for B32, L14, H14, and different pre-training scales.
5. **Computational cost on par with TRAK** (50.95 vs. 50.95 ×10¹⁵ FLOPs), 3.1% lower than TracIn.

## Highlights & Insights

- **Data-centric perspective on CLIP adaptation**: The first systematic introduction of data selection into CLIP CPT, demonstrating that "curated few" can substitute for "massive accumulation."
- **Solid theoretical foundation**: Theorem 1 establishes a lower bound on the correlation between the endpoint proxy and full-parameter alignment; Theorem 2 provides a bias-variance decomposition for curvature mixing combined with JL projection.
- **Engineering-friendly**: Scores are computed once and reusable across architectures, substantially reducing iteration costs in practical deployment.
- **Elegant three-factor product design**: Alignment (directional utility) × Learnability (boundary samples) × Relevance (domain match) are orthogonal and mutually complementary.

## Limitations & Future Work

1. **Dependence on a labeled target validation set**: A labeled $\mathcal{D}_{\text{eval}}$ is required to compute the evaluation gradient $u_\vartheta$, which is restrictive in annotation-scarce scenarios.
2. **Validation limited to CLIP architecture**: The method has not been extended to other vision-language models such as SigLIP or EVA-CLIP.
3. **Primarily validated in the medical domain**: Although general-domain retention is evaluated, the method has not been tested in other specialized domains (e.g., remote sensing, industrial inspection).
4. **Hyperparameter tuning for α and β**: Although default values are recommended, different domains may require re-tuning.
5. **Label-free target signals unexplored**: The authors themselves suggest exploring label-free or distribution-shift-robust target signals as future work.

## Related Work & Insights

- **TracIn / TRAK**: Data attribution methods for single-tower models; CHIPS builds on these by introducing CLIP-specific curvature estimation and endpoint subspace optimization.
- **BIOMEDICA / MedTrinity**: Large-scale medical multimodal datasets on which CHIPS validates data efficiency.
- **Johnson-Lindenstrauss Lemma**: A classical dimensionality reduction tool used to reduce curvature computation from $O(d^2)$ to near-linear complexity.
- **Insight**: Data selection methods can be combined with model-centric approaches (e.g., PEFT) to form a dual-efficiency strategy of "curated data + efficient fine-tuning."

## Rating

⭐⭐⭐⭐ A theoretically rigorous and experimentally comprehensive data-centric CLIP adaptation work. The three-component design is clear and elegant, and the result that 30% of data matches full-dataset CPT is impressive. The method offers strong practical value for adapting models to specialized domains with scarce data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[CVPR 2026\] MedCLIPSeg: Probabilistic Vision-Language Adaptation for Data-Efficient and Generalizable Medical Image Segmentation](medclipseg_probabilistic_vision-language_adaptation_for_data-efficient_and_gener.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mamba-based_decoder_with_distribution-aware_deep_supe.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle-educated_reinforcement_learning_for_data-efficient_and_factual_radi.md)

</div>

<!-- RELATED:END -->
