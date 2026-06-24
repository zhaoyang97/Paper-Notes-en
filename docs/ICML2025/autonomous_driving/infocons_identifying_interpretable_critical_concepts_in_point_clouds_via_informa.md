---
title: >-
  [Paper Note] InfoCons: Identifying Interpretable Critical Concepts in Point Clouds via Information Theory
description: >-
  [ICML 2025][Autonomous Driving][Point Cloud Explanation] This paper proposes the InfoCons framework, which applies the Information Bottleneck (IB) principle to interpretable point cloud models. By training an attention bottleneck network, the framework decomposes point clouds into 3D concepts of varying importance. It introduces a learnable, unbiased prior to replace the fixed prior, generating conceptually cohesive explanations while ensuring faithfulness to model prediction…
tags:
  - "ICML 2025"
  - "Autonomous Driving"
  - "Point Cloud Explanation"
  - "Information Bottleneck"
  - "Critical Concepts"
  - "Explainable AI"
date: 2026-05-08
content_hash: a40d811c25548d85
---

# InfoCons: Identifying Interpretable Critical Concepts in Point Clouds via Information Theory

**Conference**: ICML 2025  
**arXiv**: [2505.19820](https://arxiv.org/abs/2505.19820)  
**Code**: [infocons-pc](https://github.com/llffff/infocons-pc)  
**Area**: Autonomous Driving / 3D Interpretability  
**Keywords**: Point Cloud Explanation, Information Bottleneck, Critical Concepts, Explainable AI, Autonomous Driving

## TL;DR

This paper proposes the InfoCons framework, which applies the Information Bottleneck (IB) principle to interpretable point cloud models. By training an attention bottleneck network, the framework decomposes point clouds into 3D concepts of varying importance. It introduces a learnable, unbiased prior to replace the fixed prior, generating conceptually cohesive explanations while ensuring faithfulness to model predictions.

## Background & Motivation

**Background**: Point cloud models are widely deployed in safety-critical scenarios such as autonomous driving, where interpretability is crucial for diagnosis and reliability assessment. Existing point cloud explanation methods primarily rely on the "critical subset theory", attempting to extract subsets from point clouds that are most critical to model decisions.

**Limitations of Prior Work**: Existing methods fail to balance faithfulness and conceptual cohesion: (1) **Maxpool-based Critical Points (CP)**: Relying solely on encoder outputs and ignoring the classifier's influence, these methods generate explanations that are unfaithful to the full model. (2) **Gradient-based PCSAM**: This approach approximates point removal by perturbing points toward the centroid, introducing a biased prior of "centroid singularity". For example, even if all 1,024 points overlap at the origin, PointNet still predicts it as a "guitar" with 99.2% confidence. Consequently, the extracted critical subsets always cluster in spatial corners rather than semantically meaningful parts.

**Key Challenge**: Faithfulness requires the critical subset to retain points with causal impact on predictions, while conceptual cohesion requires the subset to form user-understandable semantic structures (e.g., object parts). The spatial clustering of PCSAM is caused by its biased prior rather than model behavior, whereas CP, despite being model-dependent, neglects the classifier.

**Goal**: How to extract critical subsets that are both faithful to the entire model (encoder + classifier) and aligned with human perceptual priors?

**Key Insight**: The authors observe that the VIB-for-Attribution framework, originally designed for 2D image explanations (which selects the most informative features via an information bottleneck), can be adapted to point clouds. However, two key issues must be resolved: (1) The disorder and information redundancy of point clouds necessitate different selection strategies. (2) In hierarchical or attention-based models, point features are highly entangled, rendering simple point-wise masking ineffective.

**Core Idea**: Replacing the fixed uniform prior in VIB with a learnable Gaussian prior, and using an attention bottleneck network to learn point-wise importance masks, thereby performing information selection at the feature level (instead of the input level) to disentangle neighborhood information.

## Method

### Overall Architecture

Given a pre-trained point cloud classification model $\mathcal{G} \circ \mathcal{F}$ (encoder $\mathcal{F}$ + classifier $\mathcal{G}$), InfoCons extracts point features $z = \mathcal{F}^{1:l}(x)$ at an intermediate layer $l$. An attention bottleneck network $f(\cdot|\theta)$ is trained to learn a soft mask $\hat{m} \in (0,1)^{D \times N'}$, which is used to selectively retain features or replace them with noise: $\hat{z} = \hat{m} \odot z + \text{sg}(1-\hat{m}) \odot \epsilon$. Finally, the expectation of $\hat{m}$ along the feature dimension is computed to yield point-wise importance scores $s(x) \in [0,1]^N$, where the top-k points constitute the critical subset.

### Key Designs

1. **Adaptation of IB Objective to Point Clouds (Selective Critical Points → Deep InfoCons)**:

    - **Function**: Adapting the Information Bottleneck objective for scoring point-wise importance.
    - **Mechanism**: The baseline IB objective is $\max_\theta I(\mathcal{C}, y) - \beta I(x, \mathcal{C})$. For point clouds, $I(\mathcal{C}, y)$ is lower-bounded by classification cross-entropy loss $\mathcal{L}_{CE}$, while $I(x, \mathcal{C})$ is upper-bounded by $D_{KL}(\hat{z} \| q(\hat{z}))$. The key improvement is defining the prior $q(\hat{z})$ as $\mathcal{N}(\mu_z, \sigma_z^2)$, whose parameters are determined by the statistics of the point features $z$ (rather than a fixed uniform distribution). Concurrently, "unimportant" points are replaced with noise sampled from this Gaussian prior (instead of being set to zero) to recover coarse-grained neighborhood information.
    - **Design Motivation**: Simple Selective CP works on non-hierarchical models (e.g., PointNet) but fails on hierarchical models (e.g., PointNet++) and attention-based models (e.g., PCT)—because grouping and downsampling operations in feature extraction cause neighboring point features to be highly entangled. Replacing masked features with Gaussian noise preserves neighborhood statistical information while removing only the target point's distinct information.

2. **Attention Bottleneck Network**:

    - **Function**: Learning point-wise importance masks.
    - **Mechanism**: Given intermediate features $z \in \mathbb{R}^{D \times N'}$, channel-level interactions are computed via a query-key-value attention mechanism: $q_z = W_q^T z$, $v_z = \sigma(W_v^T z)$, then $\text{Att}(q_z, z, v_z) = \text{softmax}(q_z^T z / \sqrt{D}) \cdot v_z$. This is expanded back to the original dimension $D$ via an MLP + sigmoid to generate masks $\hat{m} \in (0,1)^{D \times N'}$. For hierarchical models ($N' < N$), distance-weighted spatial interpolation propagates the masks back to the original $N$ points.
    - **Design Motivation**: Channel-level attention (instead of spatial-level) scales well to intermediate features of different sizes $N'$; non-linear attention blocks learn more complex inter-point relationships than linear masking.

3. **Learnable Unbiased Prior**:

    - **Function**: Avoiding spatial bias in PCSAM (the centroid singularity prior).
    - **Mechanism**: The parameters of the prior distribution $q(\hat{z}) = \mathcal{N}(\mu_z, \sigma_z^2)$ are calculated dynamically from the mean and variance of current point features, rather than preset. The KL divergence term $D_{KL}(\hat{z} \| q(\hat{z}))$ encourages the distribution of importance scores to approach the natural data-dependent distribution instead of artificially biasing towards specific spatial locations. A stop-gradient operator prevents gradient backpropagation through the noise branch.
    - **Design Motivation**: Since the gradient direction in PCSAM always points to the centroid, points in spatial corners are inherently preferred, which is a bias of the prior rather than actual model behavior. The learnable prior ensures importance scores are entirely driven by "contributions to predictions".

### Loss & Training

$$\mathcal{L} = \mathcal{L}_{CE}(q(y|\hat{z}), y) + \beta \cdot D_{KL}(\hat{z} \| \mathcal{N}(\mu_z, \sigma_z^2))$$

where $\beta$ controls the compression rate (i.e., how much information is preserved), and $\hat{z} = \hat{m} \odot z + \text{sg}(1-\hat{m}) \odot \epsilon$. Only the attention bottleneck parameters $\theta$ are trained, while the weights of the original point cloud model are frozen.

## Key Experimental Results

### Main Results: Critical Point Dropping Attack (DGCNN on ModelNet40)

| Method | OA↓ after dropping 500 points | Theoretical Time | Actual Time (s) | Parameters |
|------|-------------------|---------|------------|--------|
| CP++ | 75.08% | 1F | 0.01 | 0 |
| PCSAM (1pass) | 89.87% | 1(F+B) | 0.05 | 0 |
| PCSAM (20iter) | 79.86% | 20(F+B) | 0.85 | 0 |
| LIME3D (10³) | **45.22%** | 1000F | 4.54 | 1K |
| InfoCons (1pass) | 73.50% | 1F | 0.01 | 2.4M |
| InfoCons (20iter) | 63.70% | 20F | 0.29 | 2.4M |

### Downstream Applications: Data Augmentation and Adversarial Attack

| Application | Method | Metric | Result |
|------|------|------|------|
| SageMix Data Augmentation | DGCNN + SageMix | OA | 92.79% |
| SageMix + InfoCons | DGCNN + SageMix + InfoCons | OA | **93.19%** (+0.4) |
| SI-Adv Adversarial Attack | SI-Adv | ASR/CD/HD | 99.76% / 5.58 / 6.70 |
| SI-Adv + InfoCons | SI-Adv + InfoCons | ASR/CD/HD | **99.80%** / **5.47** / **6.55** |

### Key Findings

- InfoCons outperforms most baselines in terms of the efficiency-effectiveness trade-off: the 1-pass mode runs at the same speed as CP++ but achieves a much higher capacity (73.50% vs 75.08%), and the 20-iter mode ranks second (63.70%), being surpassed only by LIME3D which requires 1,000 queries.
- Qualitative analysis shows that the critical subsets identified by InfoCons are highly explanatory in misclassification cases. For example, when a "plant" is misclassified as a "flower_pot", InfoCons accurately identifies that the model focuses on the "pot" part while ignoring the "flower".
- The critical subsets of PCSAM are highly similar across different models (all clustering in spatial corners), confirming its biased prior issue; in contrast, InfoCons generates model-dependent subsets, proving higher faithfulness.
- InfoCons successfully scales to real-world datasets such as ScanObjectNN and KITTI object detection scenarios.
- There exists an optimal value for the hyperparameter $\beta$; setting it too high leads to excessive information compression and accuracy drops, while setting it too low yields low-discriminative score maps.

## Highlights & Insights

- **Insight of "Unimportant points cannot be simply removed"**: Unlike "removing background pixels" in images, even "unimportant" points in point clouds carry neighborhood information. Replacing masked points with Gaussian noise (instead of setting them to zero or discarding them) is a key innovation, as it preserves the neighborhood's statistical structure while eliminating only the discriminative information of the target point. This paradigm is transferable to other modalities with highly entangled features.
- **In-depth Analysis of Biased Priors**: The paper clearly exposes the "centroid singularity" issue in PCSAM. The fact that a point-cloud-collapsed-to-a-single-point can still be classified with high confidence reveals that the gradient direction is naturally biased. This not only explains the failure mode of PCSAM but also cautions the community against blindly applying gradient-based methods in other domains.
- **Extensibility to 8 Architectures**: InfoCons generalizes robustly across MLP-based, hierarchical, and self-attention-based models.

## Limitations & Future Work

- The attention bottleneck module must be trained separately for each model to be explained (introducing 2.4M parameters), and the hyperparameter $\beta$ and the choice of intermediate layer $l$ require manual tuning.
- Using the decrease in OA after dropping critical points as a quantitative metric has limitations, as the drop in OA could be caused by the point cloud drifting away from the data manifold rather than purely reflecting faithfulness.
- This work currently focuses on classification tasks; extending the method to segmentation or detection tasks requires redesigning the IB objectives.
- For large-scale point clouds (such as outdoor scenes in KITTI), there is still room to optimize computational efficiency.

## Related Work & Insights

- **vs Critical Points (Qi et al., 2017a)**: CP only considers the activations after the encoder's max pooling, ignoring the classifier's influence. InfoCons considers both the encoder and classifier through end-to-end IB optimization.
- **vs PCSAM (Zheng et al., 2019)**: The gradient direction of PCSAM inherently biases toward the centroid position, resulting in model-independent clusters at spatial corners. InfoCons eliminates this bias by introducing a learnable prior.
- **vs LIME3D (Tan & Kotthaus, 2022)**: LIME3D is a black-box method that achieves the best explanation performance but requires 1,000 queries (~4.5s per point cloud), making it unsuitable for real-time applications. InfoCons is a white-box method, requiring only 0.01s in 1-pass mode.
- **vs VIB for Attribution (Schulz et al., 2020)**: VIB-A is designed for CNNs/pixels and uses a fixed Gaussian prior. InfoCons is specialized for point clouds, employing a learnable prior and Gaussian noise replacement to address feature entanglement.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically adapting the IB principle to point cloud explanation; the learnable prior and noise replacement designs are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Evaluated across 8 models, 3 datasets, and 2 downstream applications, with comprehensive qualitative and quantitative comparisons.
- Writing Quality: ⭐⭐⭐⭐ Deep and thorough problem analysis, particularly outstanding in diagnosing failure modes in the Feature Analysis section.
- Value: ⭐⭐⭐⭐ Holds practical value for model diagnosis in safety-critical scenarios like autonomous driving; the code is open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] RENO: Real-Time Neural Compression for 3D LiDAR Point Clouds](../../CVPR2025/autonomous_driving/reno_real-time_neural_compression_for_3d_lidar_point_clouds.md)
- [\[AAAI 2026\] CompTrack: Information Bottleneck-Guided Low-Rank Dynamic Token Compression for Point Cloud Tracking](../../AAAI2026/autonomous_driving/comptrack_information_bottleneckguided_lowrank_dynamic_token_compres.md)
- [\[AAAI 2026\] Understanding Dynamic Scenes in Egocentric 4D Point Clouds](../../AAAI2026/autonomous_driving/understanding_dynamic_scenes_in_ego_centric_4d_point_clouds.md)
- [\[ICCV 2025\] Robust 3D Object Detection using Probabilistic Point Clouds from Single-Photon LiDARs](../../ICCV2025/autonomous_driving/robust_3d_object_detection_using_probabilistic_point_clouds_from_single-photon_l.md)
- [\[ECCV 2024\] SFPNet: Sparse Focal Point Network for Semantic Segmentation on General LiDAR Point Clouds](../../ECCV2024/autonomous_driving/sfpnet_sparse_focal_point_network_for_semantic_segmentation_on_general_lidar_poi.md)

</div>

<!-- RELATED:END -->
