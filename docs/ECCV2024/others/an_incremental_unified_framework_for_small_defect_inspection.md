---
title: >-
  [Paper Note] An Incremental Unified Framework for Small Defect Inspection
description: >-
  [ECCV 2024][Defect detection] This work proposes an Incremental Unified Framework (IUF), integrating incremental learning into unified reconstruction-based defect detection for the first time. By establishing semantic boundaries via Object-Aware Self-Attention (OASA), compressing non-primary semantic spaces through Semantic Compression Loss (SCL), and protecting old object features using an SVD-based weight update strategy, IUF achieves state-of-the-art incremental defect det…
tags:
  - "ECCV 2024"
  - "Defect detection"
  - "incremental learning"
  - "catastrophic forgetting"
  - "semantic compression"
  - "reconstruction network"
date: 2026-05-08
content_hash: 4569ccc4dbcc8596
---

# An Incremental Unified Framework for Small Defect Inspection

**Conference**: ECCV 2024  
**arXiv**: [2312.08917](https://arxiv.org/abs/2312.08917)  
**Code**: [https://github.com/jqtangust/IUF](https://github.com/jqtangust/IUF)  
**Area**: Other  
**Keywords**: Defect detection, incremental learning, catastrophic forgetting, semantic compression, reconstruction network

## TL;DR

This work proposes an Incremental Unified Framework (IUF), integrating incremental learning into unified reconstruction-based defect detection for the first time. By establishing semantic boundaries via Object-Aware Self-Attention (OASA), compressing non-primary semantic spaces through Semantic Compression Loss (SCL), and protecting old object features using an SVD-based weight update strategy, IUF achieves state-of-the-art incremental defect detection performance at both image and pixel levels on MVTec-AD and VisA.

## Background & Motivation

**Background**: AI-driven small defect detection (anomaly detection) is crucial in industrial manufacturing. Prevailing methods fall into two categories: (1) feature embedding methods (e.g., PaDiM, PatchCore), which locate defects by comparing feature distribution differences between test and normal images; and (2) reconstruction methods (e.g., UniAD), which train models to reconstruct normal samples, thereby identifying anomalies as areas that cannot be accurately reconstructed. Currently, most methods adopt a "one-model-one-object" paradigm, where a dedicated detection model is trained for each product.

**Limitations of Prior Work**: Industrial production involves diverse and dynamically changing products, posing two key challenges: (1) a system must be able to simultaneously detect defects across multiple products; (2) the system must adapt to frequently adjusted production schedules, where new products are constantly introduced, making training from scratch every time impractical. Existing unified models (e.g., UniAD) can handle multiple objects but fail to accommodate the dynamic addition of new objects. Meanwhile, memory-bank-based incremental methods (e.g., CAD) suffer from storage capacity limits and object distribution conflicts—storing features of different objects in the same memory bank leads to mutual interference, and they fail to provide pixel-level localization.

**Key Challenge**: In incremental learning scenarios, the unified reconstruction network suffers from severe "catastrophic forgetting." When learning new objects, the semantic feature space of the network is overwritten by features of the new objects, causing a sharp decline in the reconstruction capabilities for old objects. The fundamental cause is that semantic features of different objects are tightly coupled within the reconstruction network, lacking distinct boundaries to differentiate them.

**Goal**: 1) how to incrementally learn new objects in a unified reconstruction model without forgetting old ones; 2) how to establish independent semantic spaces for different objects to reduce feature conflicts; 3) how to protect the reconstruction capability of old objects without storing historical data/features.

**Key Insight**: The authors approach the problem from the perspective of semantic space management. The overall strategy consists of three steps: first, establishing semantic boundaries for different objects using category information (making the network aware of which object category is currently being processed); second, compressing the non-essential semantic space of each object to reserve capacity for future objects; and finally, suppressing weight updates on the primary semantic spaces of old objects when learning new ones.

**Core Idea**: A three-step "demarcation-compression-protection" semantic space strategy is introduced to achieve anti-forgetting incremental defect detection in a unified reconstruction model without requiring a memory bank.

## Method

### Overall Architecture

IUF is based on a Transformer reconstruction network (with UniAD as the backbone). The overall workflow is as follows: input normal image $\rightarrow$ encoder extracts multi-layer features $\rightarrow$ Object-Aware Self-Attention injects label constraints $\rightarrow$ reconstruction network generates reconstructed features $\rightarrow$ defects are located by comparing differences between input and reconstructed features. During incremental learning, new objects are introduced sequentially for training, while training data of historical objects becomes inaccessible. The framework contains three core components: OASA (establishing semantic boundaries), SCL (compressing semantic spaces), and the update strategy (protecting old semantics).

### Key Designs

1. **Object-Aware Self-Attention (OASA)**:

    - **Function**: Establishes independent semantic boundaries for different objects in the reconstruction network, allowing the network to distinguish feature spaces of different objects.
    - **Mechanism**: An auxiliary discriminator $D(\cdot)$ is introduced, which takes the input image and outputs two results: (1) an object category prediction $y_n$ trained via cross-entropy loss $L_{CE}(y_n, L_n)$; (2) categorical semantic features from key layers $C_{o_n} \in \mathbb{R}^{T \times C \times H \times W}$ (where $T$ represents the number of key layers). Then, $C_{o_n}$ is injected into the self-attention mechanism of the reconstruction network. Specifically, this is done by modulating the Query via Hadamard product: $\text{Attention}(C_{o_n}, Q, K, V) = \text{softmax}(\frac{(C_{o_n} \cdot Q)K^T}{\sqrt{d_k}})V$. Consequently, different object categories yield different Query modulations, forcing the reconstruction network to utilize distinct semantic subspaces when processing different objects.
    - **Design Motivation**: The original reconstruction network of UniAD shares a single semantic space for all objects, which inevitably overwrites old object patterns when learning new ones. Explicitly injecting category information corresponds to "demarcating" the semantic space for each object, thereby forming boundaries.

2. **Semantic Compression Loss (SCL)**:

    - **Function**: Compresses non-primary semantic features of each object to free up network capacity for future new objects.
    - **Mechanism**: Multi-layer intermediate features $M \in \mathbb{R}^{B \times C \times H \times W}$ are first spatially aggregated to obtain $\hat{M} \in \mathbb{R}^{B \times C}$, which is then factorized using SVD: $\hat{M} = USV^T$. The eigenvalues of the diagonal matrix $S$ reflect the importance of channels' semantic information—large eigenvalues correspond to primary semantics, while small eigenvalues correspond to non-primary semantics. The SCL loss function is defined as $L_{sc} = \sum_{i=t}^{C} \sigma_i$, which minimizes the sum of tail eigenvalues. This forces the semantic information of the object to concentrate on a few key channels, releasing the remaining channels for future objects. The hyperparameter $t$ controls the compression intensity.
    - **Design Motivation**: Without compression, each object's features diffuse across all channels, causing inevitable channel-wise conflicts with old objects when new ones are introduced. SVD-driven compression forces each object to only occupy the most crucial channels, reserving space for incremental learning.

3. **Reinforcing Primary Semantic Memory**:

    - **Function**: Protects the key semantic spaces of old objects from being overwritten when learning new objects.
    - **Mechanism**: This is achieved through two parts. (1) **Retaining old weights**: A regularization term on old weights is continuously added during gradient descent: $\theta_j' \leftarrow \theta_j + \nabla\theta_j^* + \beta\theta_j^{old}$, where $\beta$ controls regularization strength. (2) **Suppressing updates on old semantic spaces**: The updated gradient vector $\nabla\theta_j$ is projected onto the old objects' channel feature space (using $V_{old}^T$ from the previous SVD decomposition), and then a log constraint function $\Omega(k,c) = k \times \log(c)$ is applied to suppress updates on important (top-ranked) channels. When channel index $c=1$ (the most important), $\log(1)=0$ completely blocks updates; as $c$ increases, progressively larger updates are allowed. Finally, the constrained gradients are projected back to the original space.
    - **Design Motivation**: Even with semantic boundaries and compression, direct gradient updates can still corrupt old objects' semantic spaces. The log constraint function provides smooth protection—locking the most critical semantic spaces completely while allowing partial updates in secondary spaces to preserve network learning flexibility.

### Loss & Training

The total loss is $L = \lambda_0 L_1 + \lambda_1 L_{CE} + \lambda_2 L_{sc}$, where $L_1$ is the reconstruction loss ($\lambda_0=1$), $L_{CE}$ represents the object classification loss ($\lambda_1=0.5$), and $L_{sc}$ denotes the semantic compression loss ($\lambda_2 \in [1,10]$). Incremental learning proceeds step-by-step: the model is first trained on the base set of objects, followed by adding a set of new objects in each subsequent step. During step-wise training, only current step data and the model weights from the previous step are accessible.

## Key Experimental Results

### Main Results

| Dataset | Setting | Metric | IUF | UniAD (w/o Incremental) | CAD (Memory Bank) |
|--------|------|------|------|----------|------|
| MVTec-AD | 10-1 with 5 Steps | Image ACC↑ | 94.2 | 78.3 | 90.1 |
| MVTec-AD | 10-1 with 5 Steps | Pixel ACC↑ | 95.1 | 83.6 | N/A |
| MVTec-AD | 10-1 with 5 Steps | Image FM↓ | 3.2 | 21.5 | 8.2 |
| MVTec-AD | 3-3 with 4 Steps | Image ACC↑ | 84.2 | 68.7 | 78.4 |
| VisA | 8-1 with 4 Steps | Image ACC↑ | 79.6 | 62.1 | 74.3 |

### Ablation Study

| Config | 10-1/5Steps ACC (Img/Pix) | 10-1/5Steps FM (Img/Pix) |
|------|---------|------|
| w/o OASA | 84.2 / 90.3 | 14.2 / 8.6 |
| w/o SCL | 93.4 / 95.1 | 3.6 / 2.7 |
| w/o US (Update Strategy) | 85.6 / 90.5 | 13.5 / 8.4 |
| Full IUF | 94.2 / 95.1 | 3.2 / 1.0 |

### Key Findings
- **OASA is the foundation of anti-forgetting**: Removing OASA causes the forgetting measure (FM) to surge from 3.2 to 14.2, indicating that establishing semantic boundaries is the cornerstone of the entire framework. Without boundaries, subsequent compression and protection strategies lose their foundation.
- **The update strategy is the second most important component**: Removing it increases the FM from 3.2 to 13.5, showing that even with semantic boundaries, failing to protect the old spaces leads to catastrophic forgetting.
- **SCL's contribution is moderate but essential**: Removing it only slightly reduces the ACC from 94.2 to 93.4, but increases the FM from 3.2 to 3.6, and its efficacy becomes more pronounced in multi-step incremental scenarios.
- **More steps highlight the advantage of IUF**: Under the more challenging 3-3/4Steps setting (with more incremental steps), the performance gap between IUF and UniAD widens further, demonstrating the scalability of the framework in long-sequence increment scenarios.
- IUF is the only incremental method capable of providing both image-level and pixel-level predictions, as CAD can only perform image-level defect detection.

## Highlights & Insights

- **Elegant application of SVD in semantic space management**: Utilizing SVD to quantify channel importance for both compression (SCL minimizing small eigenvalues) and protection (suppressing gradients on crucial channels) is an elegant dual-purpose design.
- **Memory-bank-free incremental learning**: Unlike methods such as CAD that require storing historical features, IUF completely avoids storing historical data or embeddings. It combats forgetting solely through structured management of the semantic space, eliminating storage overhead and cross-object feature interference.
- **Intuitive design of the log constraint function**: $\log(c)$ equals 0 when $c=1$ (full protection) and gradually relaxes as $c$ increases, offering a smooth transition from protection to modification that is more flexible than hard thresholding.

## Limitations & Future Work

- The framework is built upon UniAD's Transformer reconstruction network; its applicability to other architectures (such as GAN- or VAE-based approaches) remains to be validated.
- OASA requires category labels as input—in actual deployment, knowing the type of product under test is necessary, which might restrict fully automated scenarios.
- SVD is performed at each training iteration, and its computational overhead grows with the number of channels, which could become a bottleneck in very wide networks.
- The hyperparameters $\lambda_2$ and $t$ require adjustments based on the number of objects and incremental steps; an automated hyperparameter selection process would be more practical.
- In scenarios with a large number of incremental steps (>10 steps), the semantic space might eventually saturate; performance in long-tail incremental scenarios warrants further investigation.
- All objects in the current experiments are of similar size (industrial parts); the capability to handle a mixture of objects with vast scale differences remains unknown.

## Related Work & Insights

- **vs UniAD**: UniAD is the base model for IUF, which can handle multi-object setups natively but lacks incremental learning support. IUF endows it with incremental capabilities through its three components: OASA, SCL, and the update strategy.
- **vs CAD**: CAD relies on a memory bank to store feature embeddings of all objects, suffering from three drawbacks: capacity limits, cross-object interference, and the inability to localize at the pixel level. IUF entirely circumvents these issues by directly manipulating the internal semantic representation.
- **vs EWC/SI/MAS**: These classical incremental learning methods protect key parameters through regularization, but they are not tailored for reconstruction tasks in defect detection. IUF's semantic space management is a customized solution specifically designed for reconstruction networks.
- This work inspires a key question: Can other reconstruction-based tasks (e.g., image inpainting, anomaly detection, generative models) benefit from similar semantic space management strategies?

## Rating
- Novelty: ⭐⭐⭐⭐ First to integrate incremental learning into unified reconstruction-based defect detection; the SVD-driven semantic space management strategy is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested on two datasets under multiple incremental settings, complemented by extensive ablation studies and qualitative analyses.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear, methodological derivations are logically sound, and charts/tables are intuitive.
- Value: ⭐⭐⭐⭐ Directly addresses real-world pain points in industrial defect detection (frequent production line switches) and holds substantial practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hyperbolic Defect Feature Synthesis for Few-Shot Defect Classification](../../CVPR2026/others/hyperbolic_defect_feature_synthesis_for_few-shot_defect_classification.md)
- [\[ICLR 2026\] GoR: A Unified and Extensible Generative Framework for Ordinal Regression](../../ICLR2026/others/gor_a_unified_and_extensible_generative_framework_for_ordinal_regression.md)
- [\[NeurIPS 2025\] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](../../NeurIPS2025/others/a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [\[ECCV 2024\] A Framework for Efficient Model Evaluation through Stratification, Sampling, and Estimation](a_framework_for_efficient_model_evaluation_through_stratific.md)
- [\[ECCV 2024\] ADMap: Anti-disturbance Framework for Vectorized HD Map Construction](admap_anti-disturbance_framework_for_vectorized_hd_map_construction.md)

</div>

<!-- RELATED:END -->
