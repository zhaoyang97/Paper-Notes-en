---
title: >-
  [Paper Note] CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection
description: >-
  [CVPR 2026][Medical Imaging][CLIP domain adaptation] This paper revisits CLIP domain adaptation from a data-centric perspective and proposes CHIPS, which computes a utility score for each image-text pair by combining three factors: curvature-aware Newton alignment (fidelity), JL sketching-compressed curvature estimation (scalability), and learnability–domain-relevance weighting (retention). Using only 30% of data, CHIPS matches full-dataset CPT; using 10%, it surpasses 50%-data CPT. The method achieves state-of-the-art data selection performance across 17 medical and 31 general benchmarks.
tags:
  - CVPR 2026
  - Medical Imaging
  - CLIP domain adaptation
  - data selection
  - curvature-aware influence functions
  - InfoNCE
  - continual pre-training
date: 2026-05-08
content_hash: 7f380162afe61182
---

# CHIPS: Efficient CLIP Adaptation via Curvature-aware Hybrid Influence-based Data Selection

**Conference**: CVPR 2026
**arXiv**: [2511.18519](https://arxiv.org/abs/2511.18519)
**Code**: [https://github.com/mihara-bot/CHIPS](https://github.com/mihara-bot/CHIPS) (open source)
**Area**: Medical Imaging / Data Selection / CLIP Adaptation
**Keywords**: CLIP domain adaptation, data selection, curvature-aware influence functions, InfoNCE, continual pre-training

## TL;DR
This paper revisits CLIP domain adaptation from a data-centric perspective and proposes CHIPS, which computes a utility score for each image-text pair by combining three factors: curvature-aware Newton alignment (fidelity), JL sketching-compressed curvature estimation (scalability), and learnability–domain-relevance weighting (retention). Using only 30% of data, CHIPS matches full-dataset CPT; using 10%, it surpasses 50%-data CPT. The method achieves state-of-the-art data selection performance across 17 medical and 31 general benchmarks.

## Background & Motivation

**Background**: Vision-language models such as CLIP perform well in general domains but suffer significant performance degradation in specialized domains such as medical imaging. Adaptation methods fall into two categories: model-centric approaches (modifying fine-tuning strategies or parameterization, e.g., PEFT, LoRA) and data-centric approaches (continual pre-training (CPT) on large-scale domain data, e.g., PubMedCLIP, BioMedCLIP).

**Limitations of Prior Work**: The CPT paradigm relies on massive domain-specific data (millions to hundreds of millions of pairs), incurring high data collection costs. Moreover, indiscriminate use of all available data introduces redundant or low-quality samples, which can actually degrade learning. Data itself, as a central factor in CPT effectiveness, has been largely overlooked.

**Key Challenge**: Existing data attribution methods (influence functions, TracIn, TRAK, etc.) are designed for single-encoder supervised classification models and systematically misrank samples when applied to CLIP, for three reasons: (A) CLIP's dual-encoder architecture produces a non-block-diagonal second-order curvature; block-diagonal approximations ignore cross-modal coupling. (B) In InfoNCE, each sample's gradient depends on the softmax normalization over the entire batch, so influence is not per-example additive. (C) Projection heads and the temperature parameter dominate early changes in similarity distributions, making full-parameter influence computation largely wasteful.

**Goal**: Design a CLIP-specific data selector that accurately evaluates each training sample's contribution to target-domain adaptation, achieving optimal CPT performance with minimal data.

**Key Insight**: Adopt a "one-step descent" perspective to compute Newton-direction alignment scores within CLIP's endpoint subspace (projection heads + temperature parameter), while ensuring scalability via InfoNCE-aware curvature estimation and JL sketching.

**Core Idea**: Perform curvature-aware Newton-direction alignment scoring within CLIP's projection-head subspace, combined with learnability and domain-relevance weights, to select high-value samples as a substitute for large-scale blind CPT.

## Method

### Overall Architecture
Given a large target-domain training pool $\mathcal{D}_{\text{pool}}$ (e.g., medical image-text pairs from BIOMEDICA), CHIPS outputs the top-$n$ most valuable samples for CPT. For each candidate sample $z$, CHIPS computes a final utility score $\mathcal{I}_{\text{CHIPS}}(z) = \hat{A}_\alpha(z) \cdot w_L(z) \cdot w_R(z)$ from three multiplicative components, then ranks all samples by score and performs continual pre-training on the top-$n$ selections.

### Key Designs

1. **Curvature-aware Proxy Newton Alignment Score $A(z)$**

   - **Function**: Measures the expected decrease in target-domain evaluation loss after performing one gradient update step on sample $z$.
   - **Mechanism**: Within CLIP's endpoint subspace $\vartheta = \{W_v, W_t, \tau\}$ (two projection heads + temperature), the score is computed as $A(z) = \mathbf{g}_\vartheta(z)^\top \mathbf{M}^{-1} \mathbf{u}_\vartheta$, where $\mathbf{g}_\vartheta(z)$ is the per-sample gradient, $\mathbf{u}_\vartheta$ is the mean gradient over the evaluation set, and $\mathbf{M}$ is the curvature proxy matrix.
   - **Design Motivation**: (1) The endpoint subspace preserves the ranking information of full-parameter alignment (Theorem 1 proves a Spearman correlation of 0.83); (2) computing gradients only for the projection heads yields dimensions far smaller than the full parameter space, substantially reducing computational cost; (3) the Newton direction is more accurate than a simple gradient inner product because it accounts for the geometry of parameter space.
   - **Novelty over Prior Work**: TracIn uses only a first-order gradient inner product (ignoring curvature); TRAK applies random projections but does not account for the special structure of InfoNCE.

2. **InfoNCE-aware Curvature Estimation**

   - **Function**: Constructs the curvature proxy matrix $\mathbf{M}$ to capture cross-sample coupling induced by negative samples in InfoNCE.
   - **Mechanism**: Curvature is decomposed into a self second-order moment $\Phi_{\text{pos}}$ (positive-sample diagonal term) and a cross-sample second-order moment $\Phi_{\text{neg}}$ (negative-sample off-diagonal term), with mixing weight $\alpha$ controlling coupling strength: $\mathbf{M} = (1-\alpha)\Phi_{\text{pos}} + \alpha\Phi_{\text{neg}} + \lambda\mathbf{I}$. JL sketching compresses the high-dimensional matrix to $k$ dimensions, achieving near-linear time and memory complexity.
   - **Design Motivation**: The softmax normalization in InfoNCE causes each sample's gradient to depend on all negative samples in the batch; ignoring this coupling systematically misranks samples. Theorem 2 provides a precise decomposition of the $O(1/k)$ variance and curvature approximation bias introduced by sketching.
   - **Novelty over Prior Work**: Standard influence functions use the Gauss-Newton approximation (positive samples only), completely ignoring the off-diagonal curvature induced by negative samples.

3. **Learnability Weight $w_L(z)$**

   - **Function**: Distinguishes samples that have already been learned from those lying near the decision boundary.
   - **Mechanism**: $w_L(z) = (1-p_{\text{corr}}(z))(1+\sigma(-m(z)))$, where $p_{\text{corr}}$ is the probability of correct matching (high values indicate mastery) and $m(z)$ is the margin to the hardest negative sample (small values indicate proximity to the decision boundary). This focuses the model on samples that are "almost correct but not yet fully learned."
   - **Design Motivation**: A pure alignment score cannot distinguish already-solved easy samples from boundary samples with genuine learning value.

4. **Target-Domain Relevance Weight $w_R(z)$**

   - **Function**: Softly biases selection toward samples whose distribution is close to that of the target-domain evaluation set.
   - **Mechanism**: The cosine similarity between the image/text embeddings of each candidate sample and the mean embedding of the evaluation set is computed and mapped through a sigmoid to a weight $w_R(z) \in [0.27, 0.73]$, serving as a soft weight rather than a hard filter.
   - **Design Motivation**: Prevents the selection of samples that deviate too far from the target domain while avoiding information loss caused by hard filtering, thereby balancing adaptation and retention.

### Loss & Training
- After selecting top-$n$ samples, standard symmetric InfoNCE is used for CPT.
- The training procedure itself is unchanged; the core innovation lies in data selection rather than the training scheme.

## Key Experimental Results

### Main Results (17 Medical Benchmarks, $r$ = 10%–50%)

| Method | $r$=10% Avg | $r$=20% Avg | $r$=30% Avg | Full Dataset |
|--------|-------------|-------------|-------------|--------------|
| Random | 24.78 | 25.00 | 26.28 | 31.51 |
| CLIPScore | 24.16 | 20.01 | 19.01 | — |
| Dot | 25.32 | 26.39 | — | — |
| TracIn | 26.46 | 26.63 | — | — |
| TRAK | 25.19 | 24.54 | — | — |
| **CHIPS** | **27.03** | **28.20** | **31.47** | 31.51 |

- CHIPS at $r$=30% (31.47) matches the full dataset (31.51).
- CHIPS at $r$=10% (27.03) surpasses a random 50% subset of the full dataset.

### Ablation Study

| Configuration | Medical Avg | Note |
|---------------|-------------|------|
| Full CHIPS | 27.03 | Complete model ($r$=10%) |
| w/o curvature ($\alpha$=0) | 25.32 | Degenerates to Dot; ignores negative-sample coupling |
| w/o learnability | 26.15 | Cannot distinguish boundary samples |
| w/o domain relevance | 25.89 | Selects samples deviating from target domain |
| w/o JL sketching | 27.01 | Negligible accuracy change but high memory overhead |

### General-Domain Retention (31 Benchmarks)

| Method | $r$=10% Classification | $r$=10% Retrieval |
|--------|------------------------|-------------------|
| Full CPT | 49.72 | 24.20 |
| Random | 52.21 | 29.28 |
| CHIPS | 47.88 | 25.71 |

- Across all retention ratios, CHIPS exhibits the smallest drop in general-domain performance.

### Key Findings
- **The curvature mixing coefficient $\alpha$ is critical**: $\alpha > 0$ significantly outperforms $\alpha = 0$ (purely diagonal), confirming that InfoNCE negative-sample coupling cannot be ignored.
- **JL sketching dimension $k$**: $k=512$ is sufficient; higher dimensions yield diminishing returns, validating the theoretical $O(1/k)$ convergence.
- **Spearman correlation of 0.83**: Endpoint-subspace alignment scores are highly consistent with full-parameter scores, validating the subspace proxy.
- **10% data surpasses 50% random CPT**: Data quality $\gg$ data quantity.

## Highlights & Insights
- The **endpoint subspace insight** is particularly elegant: the projection heads and temperature parameter are found to dominate early similarity-distribution changes during CPT, so influence function computation within this small subspace suffices. The empirical validation with Spearman 0.83 is convincing. This idea is generalizable to data selection for other dual-encoder models (e.g., retrieval models).
- **Integrating influence functions with the InfoNCE structure** constitutes the most significant technical contribution: the $\alpha$-mixture of positive- and negative-sample second-order moments captures the distinctive curvature structure of contrastive learning, a factor entirely overlooked in prior data attribution work.
- **The finding that 30% data $\approx$ full data** has substantial practical value: in medical domains where data is costly and sensitive, this implies that carefully selected subsets can replace large-scale data collection.

## Limitations & Future Work
- Computing utility scores still requires one forward/backward pass over all candidate samples, incurring non-trivial upfront computational cost (though less than full CPT).
- Experiments are conducted primarily in the medical domain; generalization to other specialized domains (e.g., remote sensing, industrial inspection) remains to be verified.
- The domain relevance weight relies on the mean embedding of the evaluation set; when the target domain is highly heterogeneous, a single mean may be insufficient.
- Theoretical analysis assumes a linear relationship between endpoint gradients and full-parameter gradients, which may not hold in highly nonlinear regions.

## Related Work & Insights
- **vs. TRAK [Park et al., 2023]**: TRAK uses random projection with an EK second-order approximation but does not account for InfoNCE's negative-sample coupling or CLIP's endpoint structure. CHIPS is specifically designed for CLIP and systematically outperforms TRAK in the medical domain.
- **vs. TracIn [Pruthi et al., 2020]**: TracIn performs only first-order gradient tracking, completely ignoring curvature. CHIPS replaces the gradient direction with the Newton direction, yielding more accurate rankings.
- **vs. BioMedCLIP [Zhang et al., 2023]**: BioMedCLIP conducts CPT on large-scale medical data; CHIPS demonstrates that a carefully selected 30% subset suffices to match its performance.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to specifically adapt data attribution methods to the contrastive learning structure of CLIP; both the endpoint subspace and InfoNCE curvature mixing are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 17 medical + 31 general benchmarks, multiple selection ratios, comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations are rigorous, though notation density is high and requires careful reading.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to data-efficient CLIP domain adaptation; the 30% ≈ Full finding has strong practical implications.
- Novelty: ⭐⭐⭐⭐ Curvature-aware influence functions for CLIP data selection represent a carefully considered design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale systematic validation across 48 benchmarks.
- Writing Quality: ⭐⭐⭐⭐ Theoretical and experimental motivations are clearly articulated.
- Value: ⭐⭐⭐⭐ Provides a powerful tool for data-efficient foundation model adaptation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards Efficient Medical Reasoning with Minimal Fine-Tuning Data](towards_efficient_medical_reasoning_with_minimal_fine-tuning_data.md)
- [\[CVPR 2026\] Ultrasound-CLIP: Semantic-Aware Contrastive Pre-training for Ultrasound Image-Text Understanding](ultrasound-clip_semantic-aware_contrastive_pre-training_for_ultrasound_image-tex.md)
- [\[CVPR 2026\] Decoding Matters: Efficient Mamba-Based Decoder with Distribution-Aware Deep Supervision for Medical Image Segmentation](decoding_matters_efficient_mambabased_decoder_with.md)
- [\[CVPR 2026\] OraPO: Oracle-educated Reinforcement Learning for Data-efficient and Factual Radiology Report Generation](orapo_oracle_rl_radiology_report_generation.md)
- [\[CVPR 2026\] RDFace: A Benchmark Dataset for Rare Disease Facial Image Analysis under Extreme Data Scarcity and Phenotype-Aware Synthetic Generation](rdface_a_benchmark_dataset_for_rare_disease_facial_image_analysis_under_extreme_.md)

</div>

<!-- RELATED:END -->
