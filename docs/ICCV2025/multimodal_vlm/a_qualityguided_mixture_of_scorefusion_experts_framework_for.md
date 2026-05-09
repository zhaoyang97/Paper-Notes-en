---
title: >-
  [Paper Note] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition
description: >-
  [ICCV 2025][Multimodal VLM][Whole-body biometric recognition] This paper proposes a Quality-guided Mixture of score-fusion Experts (QME) framework that employs a quality-guided MoE strategy to perform learnable fusion of similarity scores from heterogeneous biometric modalities (face, gait, body). Combined with a pseudo-quality loss and a score triplet loss, QME achieves state-of-the-art performance on multiple whole-body biometric recognition benchmarks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Whole-body biometric recognition
  - score fusion
  - mixture of experts
  - quality estimation
  - multimodal fusion
date: 2026-05-08
content_hash: b2b743a0a475be56
---

# A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition

**Conference**: ICCV 2025
**arXiv**: [2508.00053](https://arxiv.org/abs/2508.00053)
**Code**: [GitHub](https://github.com/jiezhu23/QME_ICCV25)
**Area**: Multimodal VLM
**Keywords**: Whole-body biometric recognition, score fusion, mixture of experts, quality estimation, multimodal fusion

## TL;DR

This paper proposes a Quality-guided Mixture of score-fusion Experts (QME) framework that employs a quality-guided MoE strategy to perform learnable fusion of similarity scores from heterogeneous biometric modalities (face, gait, body). Combined with a pseudo-quality loss and a score triplet loss, QME achieves state-of-the-art performance on multiple whole-body biometric recognition benchmarks.

## Background & Motivation

Whole-body biometric recognition requires fusing multiple modalities—including face recognition (FR), gait recognition (GR), and pedestrian re-identification (ReID)—to overcome the limitations of individual modalities under non-ideal conditions such as low illumination, occlusion, and missing features. Conventional fusion strategies fall into three categories:

- **Decision-level fusion**: Each modality independently produces an identity decision, which are then combined.
- **Feature-level fusion**: Feature vectors from different modalities are concatenated, but this approach is constrained by inter-modal inconsistencies and requires paired multi-modal datasets.
- **Score-level fusion**: Similarity scores output by individual models are integrated, offering high computational efficiency and modular flexibility.

**Limitations of Prior Work**:
1. Traditional score fusion methods (weighted averaging, Z-score normalization, Min-Max normalization, etc.) neglect distributional discrepancies across modalities (e.g., cosine similarity from face models vs. Euclidean distance from gait models).
2. Finding optimal fusion weights for each model is highly non-trivial; even exhaustive grid search rarely yields optimal solutions.
3. Existing methods lack input quality awareness and cannot dynamically adjust weights according to the reliability of each modality.

## Core Problem

How can similarity scores from heterogeneous models be adaptively fused based on per-modality input quality—without retraining biometric backbone networks—to improve whole-body biometric recognition performance in both open-set and closed-set scenarios?

## Method

### Overall Architecture

The QME framework consists of two core modules and a two-stage training pipeline.

**Input Processing**: Given $N$ pre-trained models, each computes a similarity score matrix between the query and gallery templates; these are concatenated to form a joint score matrix.

**Normalization Alignment**: A BatchNorm layer aligns the score distribution ranges across modalities and models. For models using Euclidean distance, scores are first mapped via $1/(1+\text{Euc}(q,g))$ to be consistent with cosine similarity (higher values indicating greater similarity).

**Two-Stage Training**:
1. Stage 1: Train the Quality Estimator (QE).
2. Stage 2: Freeze the QE and train the MoE score fusion model.

### Key Designs

#### 1. Quality Estimator (QE) and Pseudo-Quality Loss

**Core Idea**: If the input quality of a given modality is poor (e.g., profile face, blurry gait), the system should reduce reliance on that modality in favor of others.

**QE Architecture**:
- Intermediate features are extracted from the pre-trained model (outputs from multiple intermediate blocks).
- Mean and standard deviation of these features are computed to compress dimensionality.
- The compressed representation is fed into an encoder, which outputs a query-level quality weight $w_n$ via Sigmoid activation.

**Pseudo-Quality Loss** (addressing the absence of manual quality annotations):

$$L_\text{rank} = \sum \text{MSELoss}\!\left(w_i,\ \text{ReLU}\!\left(\frac{\delta - r_i}{\delta - 1}\right)\right)$$

where $r_i$ is the rank of the query feature in the training gallery and $\delta$ is a rank threshold hyperparameter. The intuition is that a higher rank (smaller $r_i$) indicates higher input quality, so the pseudo-quality label is closer to 1.

#### 2. Mixture of Score-Fusion Experts (MoE Layer)

- Contains $Z$ score-fusion experts, each implemented as a 3-layer MLP.
- Each expert receives the normalized concatenated score matrix and outputs a fused score.
- A routing network takes the quality weights $w_n$ from the QE as input and generates contribution weights for each expert.

**Key Difference from Conventional MoE**: In standard MoE, the router predicts assignment probabilities from the input token itself. However, similarity scores are high-level semantic features that lack fine-grained cues about query quality. QME therefore uses the QE's quality weights to guide routing.

**Expert Specialization** (empirically verified):
- Default setting: $Z=2$, with $p_1 = w_n$ and $p_2 = 1 - p_1$.
- $\varepsilon_1$ is preferentially selected when the quality weight is high, learning to emphasize the face modality.
- $\varepsilon_2$ contributes more when the quality weight is low, learning to emphasize the ReID/gait modalities.
- Final fusion: $S' = \sum p_z \cdot S_z$.

#### 3. Score Triplet Loss

Unlike the standard triplet loss, which only constrains relative distances, the score triplet loss directly constrains absolute score values:

$$L_\text{score} = \text{ReLU}(S'_{nm}) + \text{ReLU}(m - S'_{mat})$$

- First term $\text{ReLU}(S'_{nm})$: directly suppresses non-matching scores below zero.
- Second term $\text{ReLU}(m - S'_{mat})$: ensures matching scores exceed non-matching scores by at least $m$ (default: 3).

This design directly optimizes threshold-based evaluation metrics such as TAR@FAR and FNIR@FPIR.

### Loss & Training

**Stage 1** (training QE): Only the pseudo-quality loss $L_\text{rank}$ is used.

**Stage 2** (training MoE): $L_\text{score}$ is used with the QE frozen.

**Training Details**:
- Optimizer: Adam, learning rate $5\times10^{-5}$, weight decay $10^{-2}$.
- Learning rate schedule: Cosine Annealing with Warm-up.
- Each tracklet samples $L=8$ frames, aggregated into a query-level feature.
- Gallery features are pre-computed (following the CAFace strategy).
- Open-set evaluation: 10 random subsets (~20% of test subjects) are constructed; median and standard deviation are reported.

## Key Experimental Results

| Dataset | Metric | QME (Ours) | Best Prior Fusion Method | Notes |
|---------|--------|-----------|--------------------------|-------|
| CCVID | Rank-1 / mAP / TAR@1%FAR | SOTA | Outperforms Farsight, BSSF, etc. | FR model already strong (mostly frontal faces); fusion gain is limited |
| MEVID | Rank-1 / mAP / TAR@1%FAR | SOTA | Significantly outperforms all baselines | Multi-view + long-range; FR performs poorly; fusion gain is large |
| LTCC | Rank-1 / mAP / TAR@1%FAR / FNIR | SOTA | Best in both two-modal and three-modal settings | Clothing-change scenario |
| BRIAR (Face-Incl.) | TAR@0.1%FAR / R20 / FNIR | SOTA | Outperforms Farsight (82.4 TAR) | Large-scale long-range with face included |
| BRIAR (Face-Restr.) | TAR@0.1%FAR / R20 / FNIR | R20=90.6, SOTA | Significantly outperforms all methods | Poor face quality (profile/long-range); QME advantage is largest |

**Key Findings**:
- QME achieves the best or second-best performance across all datasets and metrics.
- Improvements are most pronounced in challenging scenarios (MEVID, BRIAR Face-Restricted).
- QME score fusion outperforms SapiensID (CVPR 2025 SOTA whole-body recognition model).
- Performance is comparable when using different QE sources (AdaFace-QE vs. CAL-QE), demonstrating framework flexibility.

### Ablation Study

1. **Score Triplet Loss vs. Standard Triplet Loss**: $L_\text{score}$ significantly outperforms $L_\text{tri}$ on all metrics; the additional non-matching score suppression term is critical.
2. **Number of Experts $Z$**: Performance improves progressively as $Z$ increases ($Z=1\to2$); multiple experts capture more diverse fusion strategies.
3. **Effect of QE**: Quality-aware weighting via QE further improves performance over simply averaging expert outputs.
4. **Expert Specialization Visualization**: On BRIAR, $\varepsilon_1$ performs better in the Face-Included scenario (TAR@0.1%FAR) while $\varepsilon_2$ dominates in the Face-Restricted scenario, confirming automatic expert specialization.
5. **Cross-Modal Generalization of QE**: A QE trained with CAL (a ReID model) also effectively guides fusion, achieving performance comparable to AdaFace QE.

## Highlights & Insights

1. **No Manual Quality Annotations Required**: Pseudo-quality labels are generated from ranking results, enabling QE training on any dataset.
2. **Plug-and-Play**: No backbone retraining is required; the framework operates purely at the score level and is compatible with arbitrary heterogeneous model combinations.
3. **Quality-Aware Routing in MoE**: Using QE outputs as the router—rather than the raw scores—is semantically more principled, since similarity scores themselves carry little quality information.
4. **Score Triplet Loss**: Directly aligns training objectives with evaluation metrics (TAR@FAR), proving more effective than the standard triplet loss.
5. **Strong Generalizability**: Effective across different modality combinations (two-modal / three-modal), different model combinations, and different datasets.

## Limitations & Future Work

1. **Limited Expert Count Exploration**: Only $Z=1,2$ are evaluated; the trade-off between computational cost and performance gain for larger $Z$ remains unexplored.
2. **Three-Modal Fusion Does Not Always Surpass Two-Modal**: On LTCC, incorporating a weak face modality in three-modal fusion degrades performance, indicating that robustness to extremely weak modalities can be further improved.
3. **QE Depends on Pre-Trained Model Features**: If the pre-trained model itself exhibits severe bias, QE quality estimates may be adversely affected.
4. **Coarseness of Pseudo-Quality Labels**: Rank-based pseudo-labels are an approximation and may not fully reflect true biometric quality.
5. **Validated Only on Biometric Recognition**: Although the framework is claimed to be a general score fusion approach, it has not been evaluated on other multimodal tasks.

## Related Work & Insights

| Method | Type | Quality-Aware | Learnable | Multi-Expert | Applicable Scenarios |
|--------|------|--------------|-----------|--------------|----------------------|
| Z-score / Min-Max | Statistical normalization | ✗ | ✗ | ✗ | General but suboptimal |
| Min/Max Fusion | Fixed rule | ✗ | ✗ | ✗ | Simple scenarios |
| RHE | Histogram equalization | ✗ | ✗ | ✗ | Distribution alignment |
| BSSF (SVM) | Trainable classifier | ✗ | ✓ | ✗ | Requires large training data |
| Farsight | Trainable fusion | ✗ | ✓ | ✗ | Requires manual weight tuning |
| GEFF | Gallery enhancement | ✗ | Partial | ✗ | Limited to two modalities |
| SapiensID | End-to-end model | Implicit | ✓ | ✗ | Requires large-scale training |
| **QME (Ours)** | **MoE score fusion** | **✓** | **✓** | **✓** | **General; any model combination** |

**Broader Implications**:
1. **Quality-Aware Fusion Is Critical for Multimodal Systems**: In real-world deployment, the reliability of different modalities varies with the scene; quality-aware mechanisms are essential for applications such as surveillance and access control.
2. **Generalizability of the Pseudo-Label Strategy**: Using ranking results as pseudo-quality labels is transferable to other quality estimation tasks that lack manual annotations.
3. **Advantages of Score-Level Operations**: Compared to feature-level fusion, score-level fusion requires neither paired datasets nor backbone retraining, making it a practical choice for real-world deployment.
4. **Insights on Applying MoE Beyond NLP**: Adapting MoE to score fusion is an interesting cross-domain transfer; the key innovation lies in replacing the conventional input-driven router with an external quality signal.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — MoE for score fusion + quality-guided routing + pseudo-quality loss; the combination is novel.
- **Technical Depth**: ⭐⭐⭐⭐ — Well-motivated design with clear theoretical justification for each component.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four datasets, diverse baselines, comprehensive ablation studies, and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with high-quality figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ — Plug-and-play, no backbone retraining required, deployment-friendly.
- **Overall**: ⭐⭐⭐⭐ (4/5) — A practically strong multimodal score fusion framework with direct applicability to security and recognition domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)
- [\[ICCV 2025\] DADM: Dual Alignment of Domain and Modality for Face Anti-Spoofing](dadm_dual_alignment_of_domain_and_modality_for_face_anti-spoofing.md)
- [\[ICCV 2025\] Chimera: Improving Generalist Model with Domain-Specific Experts](chimera_improving_generalist_model_with_domain-specific_experts.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)

</div>

<!-- RELATED:END -->
