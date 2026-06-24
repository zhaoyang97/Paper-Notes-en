---
title: >-
  [Paper Note] MAGR: Manifold-Aligned Graph Regularization for Continual Action Quality Assessment
description: >-
  [ECCV 2024][LLM Safety][Continual Learning] Proposes the MAGR method, which utilizes a manifold alignment projector and an Intra-Inter-Joint graph regularizer to address the misalignment between old and current feature manifolds caused by feature replay in Continual Action Quality Assessment (CAQA), significantly outperforming existing baselines across four datasets.
tags:
  - "ECCV 2024"
  - "LLM Safety"
  - "Continual Learning"
  - "Action Quality Assessment"
  - "Feature Replay"
  - "Manifold Alignment"
  - "Graph Regularization"
date: 2026-05-08
content_hash: 72321e9807a2392f
---

# MAGR: Manifold-Aligned Graph Regularization for Continual Action Quality Assessment

**Conference**: ECCV 2024  
**arXiv**: [2403.04398](https://arxiv.org/abs/2403.04398)  
**Code**: [GitHub](https://github.com/ZhouKanglei/MAGR_CAQA)  
**Area**: LLM Security  
**Keywords**: Continual Learning, Action Quality Assessment, Feature Replay, Manifold Alignment, Graph Regularization

## TL;DR

Proposes the MAGR method, which utilizes a manifold alignment projector and an Intra-Inter-Joint graph regularizer to address the misalignment between old and current feature manifolds caused by feature replay in Continual Action Quality Assessment (CAQA), significantly outperforming existing baselines across four datasets.

## Background & Motivation

**Background**: Action Quality Assessment (AQA) quantitatively evaluates action performance in scenarios such as sports and rehabilitation. However, existing methods are trained on static, small-scale datasets and cannot adapt to dynamic changes in skills over time.

**Limitations of Prior Work**: Traditional AQA models require retraining on the full dataset to update. While continual learning (CL) can address non-stationarity, it suffers from catastrophic forgetting. Moreover, CL research primarily focuses on discrete classification tasks, whereas AQA involves continuous quality score regression, presenting unique challenges.

**Key Challenge**: Feature replay methods can preserve privacy (by not storing raw videos), but updating the backbone leads to a severe misalignment between old features and the current feature manifold, worsening catastrophic forgetting. Conversely, freezing the backbone sacrifices the model's adaptability.

**Goal**: Without accessing the original raw data, how to correct the drift of old features while updating the backbone, and ensure consistency between the feature distribution and the quality score distribution.

**Key Insight**: A two-step strategy—first projecting old features onto the current manifold, and then leveraging graph regularization to align the feature space with the quality score space from both local and global perspectives.

**Core Idea**: Correcting old features by learning the manifold shift between sessions, and chunk-regularizing angular distance matrices to ensure consistency between the feature space and the quality score space.

## Method

### Overall Architecture

The MAGR framework consists of two core modules: Manifold Projector (MP) and Intra-Inter-Joint Graph Regularizer (IIJ-GR). During training in session $t$:
- One branch learns new data using the updated encoder $f$.
- Another branch retrieves old features from the memory buffer $\mathcal{M}^{t-1}$, corrects them via MP, and replays them.
- IIJ-GR applies regularization to both new and old features simultaneously.

The overall objective function is:

$$\min_{\Theta} \mathcal{L}_D + \mathcal{L}_M + \lambda_P \mathcal{L}_P + \lambda_R \mathcal{L}_R$$

where $\mathcal{L}_D$ is the regression loss for new data, $\mathcal{L}_M$ is the memory replay loss, $\mathcal{L}_P$ is the projector learning loss, and $\mathcal{L}_R$ is the graph regularization loss.

### Key Designs

1. **Manifold Projector (MP)**: Learns the mapping from the old manifold to the current manifold.

    - **Projector Learning**: At the start of session $t$, the encoder $f'$ from the previous session is frozen. The drift in the manifold is learned using the difference between the initial features $\bar{\boldsymbol{h}}_j^t = f'(\mathbf{x}_j^t)$ and the updated features $\boldsymbol{h}_j^t = f(\mathbf{x}_j^t)$ of the current session's data. The projector utilizes an MLP with residual connections:
    $\hat{\boldsymbol{h}}_j^t = \bar{\boldsymbol{h}}_j^t + p(\bar{\boldsymbol{h}}_j^t)$
    - **Feature Projection**: For each old feature in the memory buffer, it is corrected using the learned projector:
    $\tilde{\boldsymbol{h}}_i^s = \tilde{\boldsymbol{h}}_i^s + p(\tilde{\boldsymbol{h}}_i^s)$
    - **Learning Loss**: $\mathcal{L}_P = \frac{1}{|\mathcal{D}^t|}\sum_j \|\boldsymbol{h}_j^t - \hat{\boldsymbol{h}}_j^t\|_2^2$
    - **Design Motivation**: It is capable of estimating the manifold shift using only current session data without accessing old raw data. Residual connections stabilize learning (ablation confirmed that removing the residual connection decreases $\rho_{avg}$ by 7%).

2. **Intra-Inter-Joint Graph Regularizer (IIJ-GR)**: Aligns the feature distribution with the quality score distribution.

    - **Angular Distance Matrix**: After normalizing the features, angular distance is used instead of Euclidean distance to satisfy the geodesic property:
    $\mathbf{A} = \arccos(\tilde{\mathbf{H}}\tilde{\mathbf{H}}^\top), \quad \tilde{\mathbf{H}} = \mathbf{H}/\|\mathbf{H}\|$
    - **Distance Matrix Partitioning (DMP)**: Partitions the distance matrix into 4 sub-matrices $\mathbf{A}_{11}, \mathbf{A}_{12}, \mathbf{A}_{21}, \mathbf{A}_{22}$, corresponding to the relationships of old-old, old-new, new-old, and new-new, respectively.
    - **Graph Regularization**: Uses the quality score distance matrix $\mathbf{S}$ as supervision, enforced through KL divergence constraint:
    $\mathcal{L}_R = L(\mathbf{A}, \mathbf{S}) + \sum_{i=1}^{2}\sum_{j=1}^{2} L(\mathbf{A}_{ij}, \mathbf{S}_{ij})$
   where $L(\mathbf{P}, \mathbf{Q}) = \frac{1}{N}\sum_{i=1}^{N} \text{KL}(\sigma(\mathbf{P}_i), \sigma(\mathbf{Q}_i))$
    - **Design Motivation**: Euclidean distance does not satisfy the geodesic property of quality scores. KL divergence is more relaxed than MSE and is better suited for correlation metrics (ablation confirmed that using MSE drops $\rho_{avg}$ by 6%).

3. **Ordered Uniform Sampling (OUS)**: Used to select representative features to store in the memory buffer at the end of each session. It sorts features by quality score and then performs uniform sampling to ensure coverage across the entire score range, outperforming random sampling by 4% in terms of $\rho_{avg}$.

### Loss & Training

- Total loss: $\mathcal{L}_D + \mathcal{L}_M + \lambda_P \mathcal{L}_P + \lambda_R \mathcal{L}_R$, where $\lambda_P = \lambda_R = 1$
- Optimizer: Adam, with both learning rate and weight decay set to $10^{-4}$
- Each session is trained for at most 50 epochs, with a batch size of 5 and a mini-batch of 3
- Backbone: I3D (pretrained weights), with BatchNorm frozen
- The MP module uses a two-layer MLP
- 10 representative samples are stored per session

## Key Experimental Results

### Main Results

| Dataset | Metric ($\rho_{avg}$↑) | MAGR | Strongest Baseline | Gain |
|--------|------|------|------|------|
| MTL-AQA | $\rho_{avg}$ | 0.8979 | 0.8720 (MER) | +2.59% |
| FineDiving | $\rho_{avg}$ | 0.8580 | 0.8309 (GEM) | +2.71% |
| UNLV-Dive | $\rho_{avg}$ | 0.7668 | 0.7397 (MER) | +2.71% |
| JDM-MSA | $\rho_{avg}$ | 0.7166 | 0.6689 (MER) | +4.77% |

Joint Training upper bounds are 0.9360 / 0.9075 / 0.8460 / 0.7556.

### Ablation Study

| Configuration | $\rho_{avg}$ | Description |
|------|------|------|
| MAGR (Full) | 0.8979 | Baseline |
| w/o MP | 0.6949 (↓23%) | MP is the most critical component |
| w/o MP Residual Connection | 0.8391 (↓7%) | Residual connection enhances stability |
| w/o IIJ-GR | 0.7362 (↓18%) | Complete regularization is indispensable |
| w/o II-GR (remove local only) | 0.8463 (↓6%) | Local regularization has an independent contribution |
| w/o J-GR (remove global only) | 0.7839 (↓13%) | Global regularization contributes more |
| w/o KL (using MSE) | 0.8447 (↓6%) | KL is superior to MSE |
| w/o OUS (Random sampling) | 0.8619 (↓4%) | OUS sampling strategy is effective |

### Key Findings

- The greater the degree of feature shift, the more pronounced MAGR's advantage: UNLV-Dive has a feature drift MSE of 51.75, where the correlation gain is up to 15.64%.
- Under label scarcity and noisy conditions, MAGR demonstrates higher robustness.
- t-SNE visualization shows that MAGR maintains an ordered distribution of features across different sessions.

## Highlights & Insights

- **Novel problem definition**: Defines the CAQA (Continual Action Quality Assessment) task for the first time, expanding CL from classification to continuous regression.
- **Ingenious manifold projection**: Learns the manifold shift using only current session data without needing old data, balancing privacy and adaptability.
- **Angular distance replacing Euclidean distance**: Based on the insight of geodesic property, this allows feature distances to better reflect relationships with quality scores.
- **Comprehensive benchmark construction**: Proposes a grade-incremental setting and customized evaluation metrics, establishing a foundation for subsequent CAQA research.

## Limitations & Future Work

- The memory buffer still incurs extra storage overhead, which might be restricted in extreme low-storage scenarios.
- The performance of OUS degrades under very small sample sizes (e.g., only 3 per session).
- Validated only on GCN-based AQA models, without exploring other AQA architectures.
- Did not consider cross-domain AQA scenarios (e.g., migrating from diving to gymnastics).

## Related Work & Insights

- **Experience Replay** (MER, DER++) is effective but has privacy issues; feature replay overcomes privacy issues but suffers from manifold shift.
- **NC-FSCIL** avoids shift by freezing the backbone but sacrifices adaptability.
- **SLCA**'s generative replay is unstable in the AQA scenario.
- Insight: For privacy-sensitive continuous regression tasks, manifold alignment is a more critical design than simple feature storage.

## Rating

- Novelty: ⭐⭐⭐⭐ Defines the CAQA task for the first time, with an innovative manifold alignment idea.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets + ablations + robustness + visualizations, highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Logically clear, rich illustrations, and well-articulated motivation.
- Value: ⭐⭐⭐⭐ Opens up a new direction for applying continual learning to regression tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[ACL 2025\] Real-time Factuality Assessment from Adversarial Feedback](../../ACL2025/llm_safety/real-time_factuality_assessment_from_adversarial_feedback.md)
- [\[ICCV 2025\] Adversarial Robust Memory-Based Continual Learner](../../ICCV2025/llm_safety/adversarial_robust_memory-based_continual_learner.md)
- [\[ICML 2025\] Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](../../ICML2025/llm_safety/federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)

</div>

<!-- RELATED:END -->
