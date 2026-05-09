---
title: >-
  [Paper Note] A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition
description: >-
  [ICCV 2025][Multimodal VLM][Whole-body biometric recognition] This paper proposes QME (Quality-guided Mixture of score-fusion Experts), a framework that dynamically integrates similarity scores from multiple biometric modalities—including face recognition, gait recognition, and person re-identification—via learnable score fusion strategies and a quality-based MoE routing mechanism, achieving state-of-the-art performance on multiple whole-body recognition benchmarks.
tags:
  - ICCV 2025
  - Multimodal VLM
  - Whole-body biometric recognition
  - score fusion
  - Mixture of Experts
  - quality estimation
  - multimodal recognition
date: 2026-05-08
content_hash: f35ac524f6909e54
---

# A Quality-Guided Mixture of Score-Fusion Experts Framework for Human Recognition

**Conference**: ICCV 2025
**arXiv**: [2508.00053](https://arxiv.org/abs/2508.00053)
**Code**: [Project Link](https://github.com/) (The paper states "Code is available at the Project Link," but no specific URL is provided)
**Area**: Human Recognition & Biometric Fusion
**Keywords**: Whole-body biometric recognition, score fusion, Mixture of Experts, quality estimation, multimodal recognition

## TL;DR
This paper proposes QME (Quality-guided Mixture of score-fusion Experts), a framework that dynamically integrates similarity scores from multiple biometric modalities—including face recognition, gait recognition, and person re-identification—via learnable score fusion strategies and a quality-based MoE routing mechanism, achieving state-of-the-art performance on multiple whole-body recognition benchmarks.

## Background & Motivation
Whole-body biometric recognition integrates multiple modalities such as face recognition (FR), gait recognition (GR), and person re-identification (ReID) to overcome the limitations of individual modalities. This is critical in surveillance and law enforcement scenarios, where gait information can compensate for an obscured face, and facial features can compensate for changes in clothing.

Existing multimodal fusion methods fall into three categories: decision-level fusion, feature-level fusion, and score-level fusion. Feature-level fusion is theoretically optimal but faces two major obstacles: (1) heterogeneous feature spaces across modalities make alignment difficult; and (2) large-scale paired multimodal datasets are scarce—mainstream face datasets lack full-body information, while pedestrian datasets often occlude faces and are too small to support joint training.

Score-level fusion is more flexible, computationally efficient, and robust to missing modalities. However, conventional approaches (e.g., Z-score normalization, weighted averaging) ignore the distributional diversity of similarity scores across modalities and struggle to adaptively assign optimal per-modality weights—even exhaustive grid search is challenging—since queries of different quality should warrant different modality weighting strategies.

The paper's core starting point is to leverage intermediate features from pre-trained unimodal models to estimate input quality, and then use that quality information to guide a MoE router in assigning weights to multiple "experts," each learning a distinct score fusion strategy.

**Core Idea**: Quality estimates from each modality dynamically route inputs to multiple score-fusion experts, with experts corresponding to higher-quality modalities receiving greater weight.

## Method

### Overall Architecture
QME consists of a three-stage pipeline: (1) pre-trained backbone networks for each modality extract features and compute pairwise similarity score matrices; (2) a Quality Estimator (QE) predicts per-modality quality weights from intermediate backbone features; (3) a MoE score-fusion layer routes inputs to different experts based on quality weights, and the weighted sum of expert outputs yields the final fused score. Training proceeds in two stages: the QE is trained first, then frozen while the MoE fusion layer is trained.

### Key Designs
1. **Quality Estimator (QE)**:

    - **Function**: Predicts a modality quality weight $w_n \in \mathbb{R}$ from the intermediate activation features $\mathcal{I}_n \in \mathbb{R}^{L \times U \times P_n \times d_n}$ of pre-trained model $M_n$.
    - **Mechanism**: Per-block mean and standard deviation statistics are extracted from multiple transformer blocks and compressed into a $\mathbb{R}^{L \times 2d_n}$ representation, which is fed into an encoder to predict quality.
    - Training employs a pseudo-quality ranking loss $\mathcal{L}_{rank}$: $$\mathcal{L}_{rank} = \sum_{i \in L} \text{MSELoss}(w_i, \text{ReLU}(\frac{\delta - r_i}{\delta - 1}))$$ where $r_i$ is the gallery rank of query feature $q_i$ and $\delta$ is a rank threshold hyperparameter. A higher rank indicates higher quality.
    - **Design Motivation**: Manual quality annotation is unnecessary—retrieval rank serves as a proxy label. The QE generalizes to any pre-trained model (not limited to face) and can be trained on in-domain or out-of-domain data.

2. **Mixture of Score-Fusion Experts (MoE)**:

    - **Function**: The concatenated score matrix $\mathcal{S} \in \mathbb{R}^{T \times N}$ from $N$ models is processed by multiple fusion experts $\{\varepsilon_1, ..., \varepsilon_Z\}$ (each a 3-layer MLP), each producing a fused score matrix $\mathcal{S}_z \in \mathbb{R}^{1 \times T}$.
    - **Mechanism**: A router $\mathcal{N}_r$ takes QE-predicted quality weights $w_n$ as input and produces per-expert contribution weights $\{p_1, ..., p_Z\}$. The final fused score is the weighted sum: $\mathcal{S}' = \sum_{z \in Z} p_z \mathcal{S}_z$. In experiments, $Z=2$, with $p_1 = w_n$ and $p_2 = 1 - p_1$.
    - **Design Motivation**: Unlike conventional MoE methods that derive routing from input features, similarity scores are high-level semantic representations that lack fine-grained quality cues. A dedicated QE therefore provides quality information to guide routing, ensuring that experts associated with higher-quality modalities receive greater weight.

3. **Score Triplet Loss**:

    - **Function**: Imposes constraints directly in the score domain to optimize verification and open-set retrieval metrics.
    - **Mechanism**: $$\mathcal{L}_{score} = \text{ReLU}(\mathcal{S}'_{nm}) + \text{ReLU}(m - \mathcal{S}'_{mat})$$ The two terms respectively suppress non-match scores (pushing them below zero) and enforce a margin $m$ between match and non-match scores.
    - **Design Motivation**: Conventional triplet loss constrains only relative distances without directly bounding absolute score values. Metrics such as TAR@FAR rely on threshold-based decisions, necessitating suppression of the absolute values of non-match scores. This loss directly aligns the training objective with the evaluation criterion.

### Loss & Training
- **Stage 1** — Train QE using $\mathcal{L}_{rank}$.
- **Stage 2** — Freeze QE; train MoE using $\mathcal{L}_{score}$.
- All pre-trained backbone networks remain frozen; only the lightweight QE encoder and MoE experts (3-layer MLPs) are trained.
- Adam optimizer, learning rate $5 \times 10^{-5}$, cosine annealing with warmup.
- BatchNorm is applied to normalize the concatenated score matrix.
- For models using Euclidean distance, scores are converted to similarities via $1/(1+\text{Euc}(q,g))$.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours (QME) | Prev. SOTA | Gain |
|---------|--------|------------|------------|------|
| CCVID | Rank-1 | **94.1%** | 92.6% (SapiensID) | +1.5% |
| CCVID | TAR@1%FAR | **86.9%** (2 models) | 84.0% (GEFF) | +2.9% |
| MEVID | Rank-1 | **55.7%** | 54.1% (Z-score) | +1.6% |
| MEVID | TAR@1%FAR | **32.9%** | 30.7% (Passive MINT) | +2.2% |
| MEVID | FNIR@1%FPIR | **64.3%** | 65.9% (BSSF) | −1.6% |
| LTCC | Rank-1 | Significant improvement | — | — |
| BRIAR | Multiple metrics | SOTA | Farsight | Substantial improvement |

### Ablation Study

| Configuration | Rank-1 | TAR | Notes |
|---------------|--------|-----|-------|
| QME (full) | **94.1** | **86.9** | Complete framework |
| w/o QE (uniform weights) | Lower | Lower | Quality-guided routing is essential |
| AdaFace-QE routing | 92.6 | 75.0 | Routing via face quality |
| CAL-QE routing | 94.1 | 76.2 | Routing via ReID quality |
| Weighted-sum | 91.7 | 73.6 | Fixed-weight baseline |
| Farsight | 92.0 | 73.9 | Learned fusion baseline |

### Key Findings
- QME yields the largest gains in scenarios with poor face quality (MEVID, BRIAR), as dynamic routing correctly reduces the weight assigned to the face modality.
- On CCVID, where faces are generally clearly visible, improvements are relatively modest yet still achieve state of the art.
- Conventional score fusion methods (Z-score, min-max normalization) exhibit inconsistent performance across metrics, whereas QME maintains optimal or near-optimal performance on all metrics.
- Routing with QEs from different modalities produces different outcomes, confirming that quality estimation meaningfully influences expert selection.

## Highlights & Insights
- **No backbone retraining**: Only the lightweight QE and MoE (3-layer MLPs) are trained; all pre-trained models remain frozen.
- **Pseudo quality labels**: Using retrieval rank as a quality proxy is both elegant and practical, eliminating the need for manual annotation.
- **Strong extensibility**: The framework imposes no constraints on model combinations, modality types, or similarity measures, making it plug-and-play.
- The Score Triplet Loss directly aligns the training objective with evaluation metrics (TAR@FAR), offering a broadly applicable design principle.

## Limitations & Future Work
- The number of experts is fixed at $Z=2$ (corresponding to two modalities); extending the MoE design to more modalities may require architectural revision.
- The QE relies on intermediate features of a specific pre-trained model and must be retrained when the backbone is replaced.
- The paper does not address strategies for handling completely missing modalities (e.g., frames with no face detection results).
- Gains on face-dominated benchmarks such as CCVID are limited, indicating that the method's advantages are concentrated in "challenging fusion" scenarios.

## Related Work & Insights
- **vs. Farsight**: Both are learning-based score fusion methods, but Farsight employs a fixed asymmetric aggregation strategy, whereas QME achieves greater flexibility through quality-guided dynamic routing.
- **vs. SapiensID**: SapiensID is an end-to-end multimodal model requiring large-scale paired data for training; QME operates at the score level without modifying any backbone.
- **vs. GEFF**: GEFF hard-codes the mixing ratio of two modalities via a hyperparameter $\alpha$ and does not naturally extend to three modalities; QME's MoE design inherently supports multiple modalities.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Introducing MoE into score-level fusion and using quality estimation to guide routing is a novel contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Four datasets, diverse baselines, multi-metric evaluation, and comprehensive ablation analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — The framework is clearly described with well-designed figures and tables.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to real-world deployment of multimodal biometric systems.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Chimera: Improving Generalist Model with Domain-Specific Experts](chimera_improving_generalist_model_with_domain-specific_experts.md)
- [\[ICCV 2025\] Synergistic Prompting for Robust Visual Recognition with Missing Modalities](synergistic_prompting_for_robust_visual_recognition_with_missing_modalities.md)
- [\[ICCV 2025\] ProbRes: Probabilistic Jump Diffusion for Open-World Egocentric Activity Recognition](probres_probabilistic_jump_diffusion_for_open-world_egocentric_activity_recognit.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_reward-guided_decoding.md)
- [\[ICCV 2025\] G2D: Boosting Multimodal Learning with Gradient-Guided Distillation](g2d_boosting_multimodal_learning_with_gradient-guided_distillation.md)

<!-- RELATED:END -->
