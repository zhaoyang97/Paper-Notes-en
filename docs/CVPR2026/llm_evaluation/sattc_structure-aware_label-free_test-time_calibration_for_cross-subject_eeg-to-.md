---
title: >-
  [Paper Note] SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval
description: >-
  [CVPR 2026][LLM Evaluation][EEG decoding] This paper proposes SATTC, a label-free test-time calibration head that operates directly on the similarity matrix over frozen EEG and image encoders. It combines a geometric exp…
tags:
  - "CVPR 2026"
  - "LLM Evaluation"
  - "EEG decoding"
  - "cross-subject retrieval"
  - "label-free calibration"
  - "hubness mitigation"
  - "similarity matrix"
date: 2026-05-08
content_hash: 0f3e6dd5cc978b30
---

# SATTC: Structure-Aware Label-Free Test-Time Calibration for Cross-Subject EEG-to-Image Retrieval

**Conference**: CVPR 2026
**arXiv**: [2603.20738](https://arxiv.org/abs/2603.20738)
**Code**: [https://github.com/QunjieHuang/SATTC-CVPR2026](https://github.com/QunjieHuang/SATTC-CVPR2026)
**Area**: LLM Evaluation
**Keywords**: EEG decoding, cross-subject retrieval, label-free calibration, hubness mitigation, similarity matrix

## TL;DR

This paper proposes SATTC, a label-free test-time calibration head that operates directly on the similarity matrix over frozen EEG and image encoders. It combines a geometric expert (subject-adaptive whitening + adaptive CSLS) and a structural expert (mutual nearest neighbors + bidirectional top-k ranking + category popularity) via a product-of-experts fusion, significantly improving Top-1 accuracy and reducing the hubness effect in cross-subject EEG-to-image retrieval.

## Background & Motivation

1. **Background**: EEG-to-image retrieval maps brain signals into a shared embedding space and retrieves corresponding images via nearest-neighbor search. Recent methods (e.g., ATM) train powerful EEG encoders with contrastive learning and achieve competitive zero-shot retrieval on the THINGS-EEG benchmark.
2. **Limitations of Prior Work**: Existing pipelines suffer from three test-time limitations: (1) absence of structure-aware, label-free test-time calibration, reducing inference to bare nearest-neighbor search; (2) no subject-adaptive, density-aware hubness mitigation — globally fixed CSLS neighborhood sizes cannot adapt to local density variations across queries and categories; (3) structural cues such as mutual nearest neighbors and bidirectional rankings are not exploited to diagnose and correct small-k shortlist quality.
3. **Key Challenge**: During cross-subject deployment, EEG feature distributions (mean, variance, covariance structure) exhibit significant statistical shifts across subjects. Combined with the hubness effect in high-dimensional embedding spaces — where a small number of "popular" images dominate the top-k lists of most queries — small-k shortlists become highly unreliable, which is a critical failure mode in practical neural decoding applications.
4. **Goal**: Under the strict constraints of frozen encoders and no target-domain labels, calibrate retrieval rankings solely by manipulating the EEG-image similarity matrix.
5. **Key Insight**: Cross-subject retrieval is reframed as a "similarity matrix calibration" problem — without modifying encoder weights, only the similarity structure itself is adjusted. Two complementary perspectives are adopted: a geometric perspective (density-aware local rescaling) and a structural perspective (consistency patterns in ranking relations).
6. **Core Idea**: A geometric expert mitigates hubness caused by density imbalance, while a structural expert anchors high-confidence matches and penalizes popular hub categories. Their product-of-experts fusion yields the final calibrated retrieval scores.

## Method

### Overall Architecture

**Input**: Frozen EEG encoder $f_{\text{eeg}}$ and image encoder $f_{\text{img}}$, producing a $|Q| \times |C|$ similarity matrix $S_{\text{new}}$ at test time. SATTC operates as a calibration operator $F: S_{\text{new}} \mapsto S_{\text{final}}$ without modifying any network weights. Calibration proceeds in three steps: (1) subject-adaptive whitening (SAW) normalizes EEG embeddings; (2) the geometric expert adjusts similarities in a density-aware manner via adaptive CSLS; (3) the structural expert extracts ranking-consistency priors from the pre-CSLS matrix. The two experts are fused via the product-of-experts rule, yielding the final calibrated score $S_{\text{final}} = \alpha S_{\text{geom}} + \beta S_{\text{struct}}$.

### Key Designs

1. **Subject-Adaptive Whitening (SAW)**:

    - **Function**: Eliminates cross-subject distributional shifts in EEG features.
    - **Mechanism**: For each subject $s$, the mean $\mu_s$ and covariance $\Sigma_s$ are estimated to construct a regularized whitening transform $W_s = (\Sigma_s + \lambda I)^{-1/2}$. EEG embeddings are whitened and L2-normalized. An optional global whitening is applied on the image side. After whitening, features are approximately zero-mean, unit-covariance, and unit-norm, mapping different subjects onto a shared hypersphere.
    - **Design Motivation**: Inter-subject variability in EEG signals is the primary obstacle to cross-subject retrieval. SAW addresses distributional shift via statistical normalization without requiring any labels. Experiments show SAW is the single largest contributor to performance (Top-5 improves from 30.5% to 36.4%).

2. **Adaptive CSLS Geometric Expert**:

    - **Function**: Mitigates hubness with query- and category-adaptive neighborhood sizes.
    - **Mechanism**: Standard CSLS uses a fixed global neighborhood size $k$, but cross-subject EEG embedding density is highly non-uniform. The adaptive variant estimates row density $\rho_{\text{row}}(q)$ per query and maps it to $k_{\text{row}}(q) \in [k_{\min}, k_{\max}]$, and similarly estimates column density $\rho_{\text{col}}(c)$ per category to obtain $k_{\text{col}}(c)$. The CSLS score retains its classical form $S_{\text{geom}}(q,c) = 2s(q,c) - r_q(q) - r_c(c)$, but the neighborhood averages are computed using the respective adaptive neighborhood sizes.
    - **Design Motivation**: A fixed $k$ over-penalizes correct but rare matches in sparse regions and under-penalizes hub regions. The adaptive scheme assigns each query/category a neighborhood size suited to its local density, eliminating the need for global hyperparameter tuning.

3. **Structural Expert (from Pre-CSLS Similarity)**:

    - **Function**: Leverages ranking-consistency patterns in the pre-CSLS similarity matrix to reinforce high-confidence matches and penalize hub categories.
    - **Mechanism**: Row and column rankings are computed from $S_{\text{new}}$, and three types of relationships are identified: (a) **Anchors** — strict mutual nearest neighbor (MNN@1) pairs where $r_{\text{row}}(q,c) = r_{\text{col}}(c,q) = 1$, receiving a positive bias $+\lambda_{\text{anchor}}$; (b) **Bidirectional top-L pairs** — relaxed consistency matches; (c) **Hub candidates** — categories $c$ with low row rank but high column rank that frequently appear in the top-K lists of multiple queries, receiving a negative bias $-\lambda_{\text{pen}} h(c)$ where $h(c)$ is the normalized hubness score. The structural matrix is computed once and held fixed to avoid self-reinforcement.
    - **Design Motivation**: Mutual nearest neighbors are the most reliable matching signal in cross-domain retrieval — if two samples are each other's nearest neighbor, the match is highly trustworthy. Conversely, categories that frequently appear in top-K lists are likely spurious "popular" hubs that warrant active suppression.

### Loss & Training

SATTC involves no training; all operations are performed at test time. The underlying EEG encoder is trained with the AdamW optimizer, batch size 1024, learning rate $5 \times 10^{-4}$, and temperature $\tau = 1.0$. The product-of-experts fusion requires tuning only a single scalar $\beta$ (default 1.9), with $\alpha$ fixed at 1.

## Key Experimental Results

### Main Results

200-way cross-subject retrieval on the THINGS-EEG dataset (LOSO protocol, averaged over all folds and 3 seeds):

| Method | Top-5 (%)↑ | Top-1 (%)↑ |
|--------|-----------|-----------|
| ATM (original) | 20.0 | 5.5 |
| Normalized baseline (cosine+L2+CW) | 30.5 | 9.2 |
| + SAW | 36.4 | 13.7 |
| + SAW + CW | 36.8 | 13.5 |
| + SAW + CW + CSLS (fixed k=12) | 38.1 | 14.1 |
| + SAW + CW + Ada-CSLS | 38.8 | 13.9 |
| **SATTC (full)** | **38.4** | **14.8** |

Cross-encoder plug-and-play generalization (SATTC as a universal calibration layer):

| Encoder | Top-5 Baseline→+SATTC | Top-1 Baseline→+SATTC |
|---------|----------------------|----------------------|
| ATM | 30.5→38.4 (+7.9) | 9.2→14.8 (+5.6) |
| EEGNetV4 | 20.5→34.8 (+14.3) | 5.4→10.8 (+5.4) |
| EEGConformer | 11.6→23.2 (+11.6) | 2.5→6.9 (+4.4) |
| ShallowFBCSPNet | 14.6→30.8 (+16.2) | 3.5→11.1 (+7.6) |

### Ablation Study

| Configuration | Top-5 (%) | Top-1 (%) | Notes |
|---------------|-----------|-----------|-------|
| Normalized baseline | 30.5 | 9.2 | cosine+L2+CW |
| + SAW | 36.4 | 13.7 | Largest single gain (+6.2/+4.5) |
| + SAW + CW | 36.8 | 13.5 | Limited additional gain from CW |
| + Ada-CSLS | 38.8 | 13.9 | Geometric calibration |
| + Structural PoE (SATTC) | 38.4 | 14.8 | Significant Top-1 improvement |

### Key Findings

- SAW is the single largest performance contributor, with an absolute Top-5 gain of 6.2 percentage points, confirming that inter-subject statistical shift is the primary obstacle to cross-subject retrieval.
- The structural expert primarily improves Top-1 (13.9→14.8) without degrading Top-5, indicating that it precisely anchors the single most correct match.
- Adaptive CSLS and fixed CSLS achieve similar accuracy, but the adaptive variant produces a more uniform hubness distribution (flatter category popularity curve).
- SATTC is effective across all four encoder architectural styles (CSP/CNN/Transformer), validating encoder-agnosticism.
- $\beta$ is stable over a wide range; the default value of 1.9 differs from the optimum by only 0.1 percentage points.

## Highlights & Insights

- **Elegant problem reframing**: Recasting cross-subject retrieval from "how to train a better encoder" to "how to calibrate the similarity matrix at test time" fully decouples the method from the encoder. Any new encoder can be immediately improved by attaching SATTC without retraining.
- **Complementary expert design**: The geometric expert addresses hubness from a density perspective, while the structural expert addresses it from a ranking-consistency perspective — the two are complementary rather than conflicting. Their product-of-experts fusion reduces to a simple weighted sum in logit space, which is both elegant and effective.
- **Rigorous experimental design**: Nested LOSO avoids data leakage; the development-set selection strategy (easy/medium/hard subjects) prevents hyperparameter overfitting; and all hyperparameters are shared across encoders, genuinely validating encoder-agnosticism.

## Limitations & Future Work

- Validation is limited to the THINGS-EEG dataset; generalization to other EEG-image datasets remains to be confirmed.
- The structural expert relies on hand-crafted heuristics (rankings, MNN, popularity); learnable alternatives could be explored.
- The current implementation requires precomputing the full similarity matrix and does not support online streaming inference (though SAW and CSLS components can be applied online).
- SATTC has not been combined with training-time domain adaptation methods (e.g., adversarial training), which may be complementary.
- Absolute Top-1 accuracy remains low (14.8%), indicating that EEG-to-image retrieval itself remains an extremely challenging problem.

## Related Work & Insights

- **vs. ATM**: ATM uses non-normalized dot-product similarity; simply switching to cosine+L2+whitening improves Top-5 from 20% to 30.5%, demonstrating that inference pipeline normalization has been severely overlooked.
- **vs. Standard CSLS (Lample et al., 2018)**: Originally proposed for cross-lingual word embedding alignment with a fixed neighborhood size; SATTC's adaptive variant eliminates the need to tune a global $k$.
- **vs. Training-time domain adaptation methods (e.g., MS-MDA)**: These methods align distributions during training, whereas SATTC calibrates at test time — the two approaches are complementary and can be stacked.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The perspective of combining retrieval calibration with the cross-subject EEG problem is novel, though the individual components (whitening, CSLS, MNN) are established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-encoder validation, detailed ablations, and hubness analysis are thorough, but evaluation is limited to a single dataset.
- **Writing Quality**: ⭐⭐⭐⭐ Mathematical derivations are clear, and the incremental comparison clearly attributes the contribution of each component.
- **Value**: ⭐⭐⭐⭐ The encoder-agnostic plug-and-play calibration layer has strong practical value, though the application domain (brain–computer interfaces) is relatively narrow.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HyCal: A Training-Free Prototype Calibration Method for Cross-Discipline Few-Shot Class-Incremental Learning](hycal_training_free_prototype_calibration_for_cross_discipline_fscil.md)
- [\[AAAI 2026\] Graph Out-of-Distribution Detection via Test-Time Calibration with Dual Dynamic Dictionaries](../../AAAI2026/llm_evaluation/graph_out-of-distribution_detection_via_test-time_calibration_with_dual_dynamic_.md)
- [\[CVPR 2026\] Cross-Scale Pansharpening via ScaleFormer and the PanScale Benchmark](cross-scale_pansharpening_via_scaleformer_and_the_panscale_benchmark.md)
- [\[AAAI 2026\] Test-time Diverse Reasoning by Riemannian Activation Steering](../../AAAI2026/llm_evaluation/test-time_diverse_reasoning_by_riemannian_activation_steering.md)
- [\[CVPR 2026\] Free-Grained Hierarchical Visual Recognition](free-grained_hierarchical_visual_recognition.md)

</div>

<!-- RELATED:END -->
