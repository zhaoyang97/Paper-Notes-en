---
title: >-
  [Paper Note] TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception
description: >-
  [AAAI 2026][Robotics][Multimodal fusion] This paper proposes TouchFormer, a robust multimodal fusion framework that achieves reliable material perception under vision-impaired conditions through three complementary modules: Modality-Adaptive Gating (MAG), intra- and inter-modal attention mechanisms, and Cross-Instance Embedding Regularization (CER). The approach is validated in a robotic sorting experiment under simulated fire scenarios.
tags:
  - "AAAI 2026"
  - "Robotics"
  - "Multimodal fusion"
  - "material perception"
  - "tactile sensing"
  - "adaptive gating"
  - "Transformer"
date: 2026-05-08
content_hash: a29de52740b248c6
---

# TouchFormer: A Robust Transformer-based Framework for Multimodal Material Perception

**Conference**: AAAI 2026
**arXiv**: [2511.19509](https://arxiv.org/abs/2511.19509)  
**Code**: [https://touchformer.github.io/TouchFormer/](https://touchformer.github.io/TouchFormer/)  
**Area**: Robotics
**Keywords**: Multimodal fusion, material perception, tactile sensing, adaptive gating, Transformer

## TL;DR

This paper proposes TouchFormer, a robust multimodal fusion framework that achieves reliable material perception under vision-impaired conditions through three complementary modules: Modality-Adaptive Gating (MAG), intra- and inter-modal attention mechanisms, and Cross-Instance Embedding Regularization (CER). The approach is validated in a robotic sorting experiment under simulated fire scenarios.

## Background & Motivation

Material perception is a critical capability for robot–environment interaction. Surface material classification (SSMC—known categories) and unknown surface material classification (USMC—unseen categories) are two representative tasks.

**Challenges in vision-failure scenarios**: In environments such as fires, dense fog, or dark factories, vision-based methods suffer severe performance degradation. Non-visual material perception (based on tactile and auditory sensing) therefore becomes particularly important.

**Three key limitations of existing non-visual methods**:

**Naive fusion strategies**: Methods such as MTCNN directly concatenate features from different modalities and assign equal weights to all of them. However, the informative modality varies across material types—some materials are better identified by sound, others by friction. Equal-weight fusion attenuates the advantage of the most discriminative modality.

**Lack of robustness to noise and missing modalities**: In real-world deployments, temporal misalignment arises from differing sensor sampling rates, and sensors may malfunction or produce noise-corrupted data. Existing models exhibit sharp performance drops under such conditions.

**Neglect of fine-grained sub-category discrimination**: Distinguishing, for example, softwood from hardwood requires fine-grained reasoning. Existing methods focus on coarse-grained classification (8 major categories) and have not explored 193 fine-grained sub-categories.

**Core motivation**: How to design a multimodal fusion framework that is robust to noisy inputs, adaptively adjusts modality weights, and handles fine-grained classification?

## Method

### Overall Architecture

TouchFormer receives potentially noisy or incomplete multimodal sequences—sound (S), normal force (N), friction (F), and acceleration (A)—and processes them through three complementary modules:

1. **Modality-Adaptive Gating (MAG)**: Dynamically evaluates modality quality and filters noise at the input stage.
2. **Intra- and Inter-modal Transformer Fusion**: Adaptively integrates cross-modal features in the latent space.
3. **Cross-Instance Embedding Regularization (CER)**: Optimizes the prototype space to enhance inter-class separability.

### Key Designs

#### 1. Modality-Adaptive Gating (MAG)

Conventional methods treat all modalities with equal weight. MAG instead dynamically evaluates the information quality of each modality and assigns importance weights accordingly.

**Step 1**: Intermediate feature computation
$$H_m = \text{ReLU}(W_1 X_m + b_1)$$

**Step 2**: Gating weight generation
$$g_m = \sigma(W_2 H_m + b_2), \quad g_m \in [0, 1]$$

**Step 3**: Low-quality modality filtering—a threshold $gate_{th}$ is introduced; modalities below the threshold are discarded to prevent them from contaminating the fused representation.

**Step 4**: Softmax normalization to obtain final importance weights
$$\alpha_m = \frac{\exp(g_m)}{\sum_k \exp(g_k)}, \quad m \in \{S, N, F, A\}$$

**Step 5**: Weighted summation with positional encoding
$$Z_m = \alpha_m \odot (X_m + PE(T, d))$$

**Design Motivation**: Filtering noisy modalities at the source improves the reliability of subsequent fusion.

#### 2. Intra- and Inter-modal Transformer Fusion

Inspired by MulT but with a key improvement—MAG re-weighting is applied again before final feature integration to block negative transfer.

**Temporal convolution + positional encoding**:
$$\hat{X}_m = \text{Conv1D}(X_m, k_m), \quad Z_m^{[0]} = \hat{X}_m + PE(T_m, d)$$

**Inter-modal Transformer** (Cross-Modal Attention):
$$\hat{Y}_{\alpha \leftarrow \beta} = \text{softmax}\left(\frac{Q_\alpha K_\beta^\top}{\sqrt{d_k}}\right) V_\beta$$

The attention output is modulated by the MAG weight $\alpha_\beta$: $Y_{\alpha \leftarrow \beta} = \alpha_\beta \hat{Y}_{\alpha \leftarrow \beta}$. Source modalities with higher reliability contribute more.

**Intra-modal Transformer** (Self-Attention):
$$\tilde{Z}_\alpha = Z_\alpha^{[0]} + Y_{\alpha \leftarrow \beta}$$
$$Z_\alpha^{\text{intra}} = \text{Transformer}(\tilde{Z}_\alpha, \tilde{Z}_\alpha, \tilde{Z}_\alpha)$$

**Final fusion**: The intra-modal representations of the four modalities are re-weighted and concatenated:
$$Z = \text{Concat}[\alpha_S Z_S^{\text{intra}}, \alpha_N Z_N^{\text{intra}}, \alpha_F Z_F^{\text{intra}}, \alpha_A Z_A^{\text{intra}}]$$

**Key innovation**: Explicit temporal alignment is not required—modalities with different sampling rates are handled implicitly through cross-modal attention.

#### 3. Cross-Instance Embedding Regularization (CER)

Based on contrastive learning principles, CER imposes structural constraints on the embedding space from a cross-instance perspective:

$$\mathcal{L}_C = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{\sum_{j \neq i} \mathbf{I}_{\{y_i = y_j\}} \exp(S_{ij}/\tau)}{\sum_{j \neq i} \exp(S_{ij}/\tau)}$$

where $S_{ij} = z_i^\top z_j$ is the similarity matrix of $\ell_2$-normalized embeddings, and $\tau$ is a temperature hyperparameter.

**Function**: CER pulls same-class embeddings closer and pushes apart embeddings of different classes, enhancing inter-class separability and intra-class compactness. This is particularly beneficial for fine-grained sub-category discrimination (e.g., distinguishing different wood species).

### Loss & Training

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{cls}}(y, \hat{y}) + \lambda \mathcal{L}_C$$

- Joint training with classification loss $\mathcal{L}_{\text{cls}}$ and CER regularization $\mathcal{L}_C$.
- MAG module parameters are updated jointly with the rest of the network.
- Adam optimizer, weight decay 0.1, initial learning rate 0.1, cosine annealing schedule.
- Batch size 32, trained for 50 epochs.
- 5-fold cross-validation for the USMC task.
- Experiments conducted on an A100 GPU.

## Key Experimental Results

### Main Results

Surface material classification performance on the LMTHM dataset:

| Task | # Classes | Method | Accuracy (%) | G_mean |
|---|---|---|---|---|
| SSMC | 8 | DDQN | 92.71 | 0.93 |
| SSMC | 8 | MMUSR (with vision) | 93.8 | 0.94 |
| SSMC | 8 | **TouchFormer (Ours, no vision)** | **95.19 (+2.48)** | **0.95** |
| USMC | 8 | LSTM | 74.23 | 0.73 |
| USMC | 8 | MTCNN | 87.55 | 0.88 |
| USMC | 8 | **TouchFormer (Ours)** | **94.38 (+6.83)** | **0.94** |
| Fine-grained | 193 | MTCNN | 80.21 | 0.80 |
| Fine-grained | 193 | **TouchFormer (Ours)** | **89.77 (+9.56)** | **0.90** |

Most notable results: vision-free TouchFormer (95.19%) outperforms vision-dependent MMUSR (93.8%); on 193-class fine-grained classification, a gain of 9.56% is achieved.

### Ablation Study

| Configuration | SSMC | USMC | Fine-grained | Notes |
|---|---|---|---|---|
| Baseline (Transformer fusion only) | 91.32% | 90.17% | 83.53% | No MAG, no CER |
| Baseline + MAG | 93.15% | 92.56% | 84.88% | + adaptive gating |
| **Baseline + MAG + CER** | **95.19%** | **94.38%** | **89.77%** | Full framework |

MAG contribution: +1.83% / +2.39% / +1.35% (across three tasks).
CER contribution: +2.04% / +1.82% / +4.89% (across three tasks).
CER yields the largest gain on fine-grained classification (+4.89%), validating the effectiveness of contrastive learning for sub-category discrimination.

### Robustness to Missing Modalities

| Multimodal input (S/N/F/A) | LSTM | MTCNN | TouchFormer |
|---|---|---|---|
| ✗/✓/✓/✓ (no sound) | 73.78% | 83.43% | 87.92% |
| ✓/✗/✓/✓ (no normal force) | 71.69% | 83.18% | 88.55% |
| ✓/✓/✗/✓ (no friction) | 69.09% | 84.99% | 89.12% |
| ✓/✓/✓/✗ (no acceleration) | 75.50% | 87.00% | 89.84% |
| ✓/✓/✓/✓ (complete) | 74.23% | 87.55% | **94.38%** |

**Key finding**: Even with any single modality missing, TouchFormer (87.92%–89.84%) still outperforms competing methods using all modalities. The MAG gating mechanism automatically redistributes weights under modality absence.

### Key Findings

1. **No vision > with vision**: Under vision-impaired conditions, TouchFormer without visual input still surpasses MMUSR, which requires visual input.
2. **CER is critical for fine-grained classification**: A 4.89% improvement on 193-class sub-category classification; t-SNE visualization shows substantially improved embedding clustering.
3. **Noise robustness**: TouchFormer consistently outperforms baselines across a wide range of noise ratios (0.0–1.0).
4. **Real robot validation**: A RealMan RM65-B robotic arm equipped with TouchFormer successfully completes vision-free material sorting in a simulated fire scenario.

## Highlights & Insights

1. **Elegant MAG gating design**—beyond re-weighting, MAG filters low-quality modalities via a threshold, ensuring information quality from the input source.
2. **No explicit temporal alignment required**—cross-modal attention implicitly handles modalities with different sampling rates, significantly simplifying data preprocessing.
3. **Complementarity of CER and MAG**: MAG improves input reliability while CER enhances output discriminability, strengthening the model from two orthogonal dimensions.
4. **Engineering value of the FISHM dataset**—a multimodal tactile dataset specifically collected for fire scenarios, filling a data gap in extreme environments.
5. **Closed-loop validation from perception to manipulation**—the work not only evaluates classification accuracy but also demonstrates a sorting application on a real robot.

## Limitations & Future Work

1. **Sensor specificity**: Results are tightly coupled to the uSkin sensor; performance with other tactile sensors remains unknown.
2. **Classification rather than regression**: Only discrete category classification is performed; continuous prediction of material properties (e.g., hardness, roughness) is not explored.
3. **Fire scenario experiments remain controlled simulations**: Object positions are fixed and grasping is executed with fixed parameters; real fire environments are far less predictable.
4. **Evaluation on only two datasets**: LMTHM and FISHM; generalization is insufficiently validated.
5. **Manual tuning of the MAG threshold**: The optimal value of $gate_{th}$ may vary across tasks.

## Related Work & Insights

- **Relationship to MulT**: TouchFormer builds on MulT's cross-modal attention and adds MAG re-weighting as a key improvement.
- **Integration with contrastive learning**: CER is essentially an application of the supervised contrastive loss (SupCon), but its use in tactile material perception is novel.
- **Prospects for emergency robotics**: The ability of fire-rescue robots to perceive materials under zero-visibility conditions has significant practical value.
- **Generality of the sensor fusion paradigm**: The MAG + Transformer fusion pattern is extensible to other multi-sensor settings (e.g., autonomous driving, medical robotics).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combined design of MAG gating and CER regularization is effective, though the individual modules are not entirely novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Ablation, noise robustness, missing modality, fine-grained classification, and real-robot experiments are comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clear; method description is detailed.
- **Value**: ⭐⭐⭐⭐ — Significant engineering value for robotic material perception in extreme environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CUBic: Coordinated Unified Bimanual Perception and Control Framework](../../CVPR2026/robotics/cubic_coordinated_unified_bimanual_perception_and_control_framework.md)
- [\[CVPR 2026\] A Cross-view Fusion Framework for Robust 6-DoF Grasp Pose Estimation](../../CVPR2026/robotics/a_cross-view_fusion_framework_for_robust_6-dof_grasp_pose_estimation.md)
- [\[CVPR 2025\] Perceive What Matters: Relevance-Driven Scheduling for Multimodal Streaming Perception](../../CVPR2025/robotics/perceive_what_matters_relevance-driven_scheduling_for_multimodal_streaming_perce.md)
- [\[AAAI 2026\] Distributionally Robust Online Markov Game with Linear Function Approximation](distributionally_robust_online_markov_game_with_linear_function_approximation.md)
- [\[AAAI 2026\] Robust Out-of-Order Retrieval for Grid-Based Storage at Maximum Capacity](robust_out-of-order_retrieval_for_grid-based_storage_at_maximum_capacity.md)

</div>

<!-- RELATED:END -->
