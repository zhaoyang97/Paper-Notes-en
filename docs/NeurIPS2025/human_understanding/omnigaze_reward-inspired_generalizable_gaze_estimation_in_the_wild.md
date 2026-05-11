---
title: >-
  [Paper Note] OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild
description: >-
  [NeurIPS 2025][Human Understanding][Gaze estimation] This paper proposes OmniGaze, a semi-supervised 3D gaze estimation framework that employs a reward model — fusing visual embeddings…
tags:
  - "NeurIPS 2025"
  - "Human Understanding"
  - "Gaze estimation"
  - "semi-supervised learning"
  - "pseudo-labels"
  - "reward model"
  - "cross-domain generalization"
date: 2026-05-08
content_hash: b3f1a4b5eccc6b28
---

# OmniGaze: Reward-inspired Generalizable Gaze Estimation in the Wild

**Conference**: NeurIPS 2025
**arXiv**: [2510.13660](https://arxiv.org/abs/2510.13660)
**Code**: [GitHub](https://github.com/quhongyu/OmniGaze)
**Area**: Human Understanding
**Keywords**: Gaze estimation, semi-supervised learning, pseudo-labels, reward model, cross-domain generalization

## TL;DR

This paper proposes OmniGaze, a semi-supervised 3D gaze estimation framework that employs a reward model — fusing visual embeddings, MLLM-generated semantic gaze descriptions, and geometric direction vectors — to assess pseudo-label quality. Trained on 1.4 million unlabeled face images, OmniGaze achieves state-of-the-art performance under both within-domain and cross-domain settings across 5 datasets, and demonstrates zero-shot generalization on 4 unseen datasets.

## Background & Motivation

3D gaze estimation aims to directly predict gaze direction from facial images and serves as a foundational technology for VR, human-computer interaction, and medical diagnosis. Existing methods face two core challenges:

**Scarcity and limited diversity of annotated data**: Existing annotated datasets (e.g., ETH-XGaze, Gaze360) simplify real-world scenarios through predefined assumptions (e.g., panoramic cameras, multi-view photography), yet remain limited in scale and scene diversity, causing significant performance degradation on unseen domains.

**Generalization methods constrained by annotation coverage**: Cross-domain generalization approaches (domain adaptation, domain generalization) are effective but fundamentally bounded by the coverage of annotated source-domain data, whereas large quantities of unlabeled face images can be readily obtained through web crawling or generative models.

**Gap in semi-supervised frameworks for gaze estimation**: Weakly supervised methods require annotations from gaze-related domains; self-supervised pretext tasks (e.g., image reconstruction, gaze redirection) exhibit weak semantic relevance to gaze estimation and utilize unlabeled data inefficiently. Critically, existing SSL pseudo-label selection strategies (e.g., FixMatch's fixed threshold, FlexMatch's class-aware threshold) are **designed for classification tasks** and cannot be directly applied to continuous regression targets such as gaze direction.

This gap motivates the design of OmniGaze: a semi-supervised framework tailored for continuous regression tasks that employs a reward model to assess pseudo-label quality while leveraging large-scale unlabeled data to improve generalization.

## Method

### Overall Architecture

OmniGaze follows a standard three-stage semi-supervised training pipeline: (1) the teacher model is trained supervisedly on labeled data; (2) the teacher model generates pseudo-labels for unlabeled data, and a reward model filters high-quality pseudo-labels; (3) the student model is jointly trained on labeled data combined with the filtered pseudo-labeled data. The core innovations lie in the reward model design and multimodal cue fusion.

### Key Designs

1. **Reward model with multimodal cue fusion**: The reward model $h_G$ assesses pseudo-label reliability from three perspectives:

   - **Visual cues**: A CLIP visual encoder extracts visual embeddings $\boldsymbol{f}_k^v$ from input face images.
   - **Semantic cues**: An MLLM (InstructBLIP) is queried with "where is this person looking in 3D space?" to obtain a natural language description, which is encoded via the CLIP text encoder into $\boldsymbol{f}_k^t$.
   - **Geometric cues**: The yaw/pitch angles of the pseudo-label are converted into a 3D direction vector.

   Visual and semantic cues are fused via cross-attention into a semantics-aware gaze representation:
   $\hat{\boldsymbol{f}}_k^v = \text{AvgPool}(\text{LN}(\text{CrossAttn}(\boldsymbol{f}_k^v, \boldsymbol{f}_k^t)))$

   where visual features serve as queries and text features as keys/values, enabling the reward model to dynamically attend to relevant semantic cues when interpreting visual features.

2. **Geometric representation via spherical-to-Cartesian conversion**: Pseudo-labels $(θ_k, ψ_k)$ are converted into 3D direction vectors:
   $\boldsymbol{v}_k = [\cos(ψ_k)\cdot\sin(θ_k),\ \cos(ψ_k),\ \cos(ψ_k)\cdot\cos(θ_k)]$

   This representation is more expressive and continuous than raw angles, facilitating precise alignment with semantics-aware representations. The reward model predicts a confidence score via cross-attention followed by an MLP:
   $\hat{r}_k = \text{Sigmoid}(\text{MLP}(\text{CrossAttn}(\hat{\boldsymbol{f}}_k^v, \boldsymbol{v}_k)))$

3. **Label scorer and final confidence**: The initial confidence $\hat{r}_k$ from the reward model and the cosine similarity between the student model's prediction and the pseudo-label are jointly fed into a lightweight MLP to produce the final confidence:
   $r_k = \text{Sigmoid}(\text{MLP}([(\hat{r}_k, \text{sim}(\hat{y}_k, y_k))]))$

   This design encourages mutual verification between the reward model and the student model, improving assessment robustness.

### Loss & Training

- **Supervised loss** (labeled data): Angular L2 loss $\mathcal{L}^s = \frac{1}{N_L}\sum\|\hat{y} - y_i^l\|_2$

- **Reward model loss**: Binary cross-entropy where ground-truth labels are marked as reliable ($c_k=1$) and pseudo-labels as unreliable ($c_k=0$), training the reward model to discriminate between them:
  $$\mathcal{L}^g = \sum -(c_k\log(r_k) + (1-c_k)\log(1-r_k))$$

- **Unsupervised loss** (pseudo-labeled data): Angular loss with threshold filtering and confidence weighting:
  $$\mathcal{L}^u = \sum \mathbb{1}[r_j \geq 0.5] \cdot r_j \cdot \|h_S(x_j^u) - y_j^u\|_2$$

  Threshold $τ=0.5$: pseudo-labels below the threshold are discarded, while high-confidence pseudo-labels receive larger weights.

- **Periodic pseudo-label update**: Every $K=10$ epochs, the student model weights are transferred to the teacher model and pseudo-labels are regenerated, preventing overfitting to noisy labels at early stages.

- **Overall training objective**: A combined average of $\mathcal{L}^s$ and $\mathcal{L}^u$.

## Key Experimental Results

### Main Results — Within-domain Gaze Estimation (Angular Error °↓)

| Method | MPIIFaceGaze | EyeDiap | RT-Gene | Gaze360 | IVGaze |
|--------|:---:|:---:|:---:|:---:|:---:|
| GazeTR (ICPR'22) | 4.00 | 5.17 | 6.55 | 10.62 | 7.33 |
| AGE-Net (ICIP'24) | 3.61 | 4.78 | - | - | - |
| 3DGazeNet (ECCV'24) | 4.00 | - | - | 9.60 | - |
| **OmniGaze** | **2.97** | **4.07** | **5.40** | **9.12** | **6.72** |
| Gain | -0.64 | -0.71 | -1.15 | -0.48 | -0.61 |

### Ablation Study — Core Components (Zero-shot Setting)

| Configuration | MPIIGaze | EyeDiap | RT-Gene | Notes |
|---------------|:---:|:---:|:---:|-------|
| Labeled data only | 6.17 | 8.45 | 20.73 | Baseline |
| + Unlabeled data (no filtering) | 4.97 | 5.42 | 13.75 | Scale alone yields significant gains |
| + Reward model filtering | **3.44** | **4.31** | **9.09** | Filtering further improves by 1.5°+ |

### Ablation Study — Reward Model Components

| Component | MPIIGaze | EyeDiap | RT-Gene |
|-----------|:---:|:---:|:---:|
| Semantic gaze description only | 3.69 | 4.78 | 10.23 |
| 3D direction vector only | 4.03 | 4.93 | 10.56 |
| Both combined (full) | **3.44** | **4.31** | **9.09** |

### Key Findings

- **Data scale and quality filtering act synergistically**: The three-step progression from labeled-only → adding unlabeled data → adding reward filtering yields significant improvements at each step.
- **Semantic descriptions contribute more than geometric vectors**: Semantic gaze descriptions as a standalone component outperform 3D direction vectors, indicating that MLLM scene understanding provides valuable priors for regression tasks.
- **Zero-shot performance surpasses within-domain SOTA**: OmniGaze-Base's zero-shot performance (3.44° on MPIIFaceGaze) even outperforms prior SOTA requiring within-domain training (3DGazeNet at 4.00°).
- **Scalable with model size**: Performance consistently improves from ViT-S → ViT-B → ViT-L.

## Highlights & Insights

1. **Using MLLMs as semantic priors for regression tasks** is the paper's most elegant design: rather than directly applying MLLMs to gaze estimation, the approach leverages MLLM scene understanding to assist pseudo-label quality assessment — an indirect yet effective form of multimodal knowledge transfer.
2. **Filling the methodological gap of SSL for continuous regression**: Existing SSL pseudo-label selection strategies are almost exclusively designed for classification; the reward model with confidence scoring provides a general framework for regression tasks.
3. **Potential as a scalable data engine**: OmniGaze can generate reliable gaze annotations for arbitrary face images, with performance continuously improving as more unlabeled data is incorporated.

## Limitations & Future Work

- Reward model training relies on a certain amount of labeled data to initialize the teacher model, making it inapplicable in fully unsupervised settings.
- MLLM inference (InstructBLIP description generation) introduces additional data preprocessing costs, presenting efficiency bottlenecks at the million-image scale.
- The binary classification objective of the reward model (ground truth vs. pseudo-label) may be an oversimplification — ground-truth labels can also be noisy.
- Validation is limited to ViT architectures; other backbones such as CNNs remain unexplored.

## Related Work & Insights

- **SemiReward (reward scoring in SSL)**: This work extends the SemiReward concept from classification to regression, with the key distinction of incorporating multimodal cues and geometric representations.
- **Implications for other regression SSL tasks**: Continuous regression tasks such as depth estimation, pose estimation, and optical flow estimation may similarly benefit from reward-driven pseudo-label filtering.
- **MLLMs as auxiliary priors**: MLLMs need not make direct predictions; their semantic understanding capabilities can serve as "meta-knowledge" to assist training in other tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The trimodal fusion of reward model, MLLM semantic descriptions, and geometric representation is novel; the SSL framework for regression tasks represents a methodological contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Five datasets × three settings (within-domain/cross-domain/zero-shot) with extensive ablations across three model scales.
- **Writing Quality**: ⭐⭐⭐⭐ Method description is clear with well-articulated correspondence between challenges and proposed solutions.
- **Value**: ⭐⭐⭐⭐ The scalable data engine paradigm offers strong inspirational value for other visual regression tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Monocular Facial Appearance Capture in the Wild](../../ICCV2025/human_understanding/monocular_facial_appearance_capture_in_the_wild.md)
- [\[ICCV 2025\] GGTalker: Talking Head Synthesis with Generalizable Gaussian Priors and Identity-Specific Adaptation](../../ICCV2025/human_understanding/ggtalker_talking_head_systhesis_with_generalizable_gaussian_priors_and_identity-.md)
- [\[NeurIPS 2025\] RAPTR: Radar-Based 3D Pose Estimation Using Transformer](raptr_radar-based_3d_pose_estimation_using_transformer.md)
- [\[ICCV 2025\] PoseSyn: Synthesizing Diverse 3D Pose Data from In-the-Wild 2D Data](../../ICCV2025/human_understanding/posesyn_synthesizing_diverse_3d_pose_data_from_in-the-wild_2d_data.md)
- [\[CVPR 2026\] RAM: Recover Any 3D Human Motion in-the-Wild](../../CVPR2026/human_understanding/ram_recover_any_3d_human_motion_in-the-wild.md)

</div>

<!-- RELATED:END -->
