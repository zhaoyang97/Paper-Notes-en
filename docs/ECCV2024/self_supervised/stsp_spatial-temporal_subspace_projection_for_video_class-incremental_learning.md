---
title: >-
  [Paper Note] STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning
description: >-
  [ECCV 2024][Self-Supervised Learning][Video Class-Incremental Learning] The Spatial-Temporal Subspace Projection (STSP) method is proposed to address catastrophic forgetting in video class-incremental learning. By representing each class with orthogonal subspace bases using a Temporal-based Subspace Classifier (TSC) and constraining gradients to the null space of old task features using Spatial-based Gradient Projection (SGP), the method achieves SOTA results on HMDB51…
tags:
  - "ECCV 2024"
  - "Self-Supervised Learning"
  - "Video Class-Incremental Learning"
  - "subspace projection"
  - "orthogonal constraints"
  - "gradient projection"
  - "catastrophic forgetting"
date: 2026-05-08
content_hash: 956f3a96bb922830
---

# STSP: Spatial-Temporal Subspace Projection for Video Class-Incremental Learning

**Conference**: ECCV 2024  
**Paper Link**: [ECVA](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/4106_ECCV_2024_paper.php)  
**Code**: None  
**Area**: Other  
**Keywords**: Video Class-Incremental Learning, subspace projection, orthogonal constraints, gradient projection, catastrophic forgetting

## TL;DR
The Spatial-Temporal Subspace Projection (STSP) method is proposed to address catastrophic forgetting in video class-incremental learning. By representing each class with orthogonal subspace bases using a Temporal-based Subspace Classifier (TSC) and constraining gradients to the null space of old task features using Spatial-based Gradient Projection (SGP), the method achieves SOTA results on HMDB51, UCF101, and SSv2.

## Background & Motivation

**Background**: Video Class-Incremental Learning (VCIL) requires models to continuously learn new video classes without forgetting old ones. This is a highly frequent demand in practical applications—surveillance systems need to continuously identify new abnormal behaviors, and robots need to continuously learn new actions. Current incremental learning methods are primarily designed for images, yielding poor performance when directly transferred to the video domain.

**Limitations of Prior Work**: Traditional VCIL methods typically store partial frames or features of old tasks for "exemplar replay," which presents two issues: (1) this practice ignores semantic relations between old and new classes, and simple replay tends to cause class confusion; (2) storing old data may lead to privacy leakage. While regularization-based methods do not store data, they fail to adequately model the complex spatial-temporal dynamics in videos, making it difficult to effectively distinguish actions with similar temporal patterns.

**Key Challenge**: Video data simultaneously contains spatial appearance information and temporal evolution information. Incremental learning must balance protecting old knowledge with learning new knowledge, and this spatial-temporal duality makes the tradeoff significantly more difficult than in image-based incremental learning. Existing methods either focus only on protecting spatial features while ignoring the temporal dimension, or rely on expensive data replay.

**Goal**: (1) To effectively prevent catastrophic forgetting without saving old data; (2) To leverage the spatial-temporal dynamic characteristics in videos to enhance the discriminative capacity of incremental learning.

**Key Insight**: The authors propose addressing this from the perspective of subspace projection—representing the feature distributions of different classes with orthogonal subspace bases, making different classes naturally discriminative; concurrently, utilizing the null space of old task features to constrain the gradient directions of new tasks, fundamentally avoiding interference with old knowledge.

**Core Idea**: Represent each video class as an orthogonal subspace in the feature space, simultaneously achieving class discrimination and knowledge protection through subspace classification in the temporal dimension and gradient projection in the spatial dimension.

## Method

### Overall Architecture
The input to STSP is a sequence of video frames. After extracting spatial features through a shared feature extractor (such as ResNet/ViT), temporal dynamics are captured by a temporal modeling module. At the classification end, the TSC determines classes based on the projection magnitudes of features on each class subspace. During training, the SGP monitors the direction of gradient updates to ensure that learning new knowledge does not disrupt old knowledge. The overall pipeline can be divided into: (1) feature extraction; (2) subspace classification based on TSC; (3) gradient-constrained training based on SGP.

### Key Designs

1. **Temporal-based Subspace Classifier (TSC)**:

    - **Function**: Replaces the traditional fully-connected classification head by representing each video class with orthogonal subspace bases.
    - **Mechanism**: Standard classifiers use a weight vector to represent each class, whereas TSC assigns a set of orthogonal basis vectors to each class, forming a subspace. Given the spatial-temporal feature $f$ of a video, TSC calculates its projection components on each class subspace: $s_c = \|P_c f\|^2$, where $P_c = B_c B_c^T$ is the subspace projection matrix of class $c$ and $B_c$ represents the orthogonal bases of this class. The predicted class corresponds to the subspace with the maximum projection magnitude. Crucially, when extracting subspace bases, TSC considers dynamic changes in the temporal dimension—the positional transitions of features from different frames within the subspace reflect the temporal evolution of actions, offering temporal cues for classification.
    - **Design Motivation**: Compared with fully-connected layers, subspace representation offers two advantages: first, it captures intra-class diversity (a subspace can cover more variation than a single vector); second, the orthogonal constraints between subspaces naturally provide inter-class discriminability. This representation is particularly well-suited for incremental learning—new classes only require the allocation of new orthogonal subspaces, avoiding conflict with old classes.

2. **Inter-class and Intra-class Orthogonality Constraints**:

    - **Function**: Ensures that subspaces of different classes do not overlap and that the basis vectors of the same class are mutually orthogonal.
    - **Mechanism**: Inter-class constraints require the subspace bases of different classes to satisfy $B_i^T B_j = 0$ ($i \neq j$), implying that the subspaces of different classes are orthogonal. This is implemented by adding an orthogonal penalty term to the loss function: $L_{inter} = \sum_{i \neq j} \|B_i^T B_j\|_F^2$. Intra-class constraints ensure that the basis vectors of each class themselves are orthonormal: $B_c^T B_c = I$. Together, these two layers of constraints guarantee that the feature space is "cleanly" partitioned among the classes.
    - **Design Motivation**: In incremental learning, without orthogonal constraints, the subspaces of new classes might invade the feature spaces of old classes, leading to classification confusion. Orthogonal constraints geometrically guarantee a "separate territory" for each class, maintaining the decision boundaries of old classes even when introducing new ones.

3. **Spatial-based Gradient Projection (SGP)**:

    - **Function**: Constrains gradients to the null space of old task spatial features during new task training to prevent forgetting.
    - **Mechanism**: SGP first collects the spatial features of old task data and calculates the principal components of their feature matrix (obtaining the primary subspace of old task features via SVD). When learning a new task, the gradient $g$ of each network layer is projected: $g' = g - P_{old} g$, where $P_{old}$ is the projection matrix of the old task feature subspace. The projected gradient $g'$ lies in the null space of the old task features, meaning that parameter updates will not alter the spatial feature representation already learned for the old tasks. SGP specifically focuses on spatial features rather than temporal features, as spatial appearance information is fundamental for class discrimination.
    - **Design Motivation**: Traditional gradient projection approaches (like GPM) project directly on all features, which is computationally expensive and may over-constrain the network. SGP only protects the core directions of spatial features, leaving more learning flexibility for the temporal dimension. This is because spatial appearance features are typically the most critical information for class discrimination (e.g., the key to "kicking a ball" is the appearance of the "ball" and the "foot"), while temporal models can adapt more flexibly to new classes.

### Loss & Training
The total loss consist of three parts: (1) the subspace projection classification loss $L_{cls}$, which calculates cross-entropy based on projection magnitudes; (2) the inter-class orthogonality loss $L_{inter}$, which penalizes overlap between subspaces of different classes; (3) the intra-class orthogonality loss $L_{intra}$, which maintains the orthonormality of the basis vectors. During training, the subspaces of the initial classes are first learned on the base task, and during subsequent incremental phases, new classes are learned by constraining gradient directions via SGP.

## Key Experimental Results

### Main Results

| Dataset | Metric | STSP | Prev. SOTA | Gain |
|--------|------|------|----------|------|
| UCF101 (5 tasks) | Last Acc | 72.8 | 69.2 | +3.6 |
| UCF101 (10 tasks) | Last Acc | 67.3 | 63.8 | +3.5 |
| HMDB51 (5 tasks) | Last Acc | 51.4 | 47.9 | +3.5 |
| SSv2 (5 tasks) | Last Acc | 38.6 | 35.2 | +3.4 |
| UCF101 (5 tasks) | Avg Acc | 80.1 | 77.5 | +2.6 |

### Ablation Study

| Configuration | UCF101 Last Acc | Description |
|------|----------------|------|
| Full STSP | 72.8 | Full model |
| w/o TSC (using FC) | 68.1 | Swapped with fully-connected classifier |
| w/o Orthogonal Constraints | 69.5 | Removed inter-class / intra-class orthogonal constraints |
| w/o SGP | 70.2 | Without gradient projection |
| w/o Temporal Modeling | 70.9 | TSC does not consider temporal dynamics |
| Replay + FC | 70.4 | Traditional replay method |

### Key Findings
- TSC contributes the most; replacing it with a fully-connected classifier drops the accuracy by 4.7 percentage points, demonstrating that subspace representation is crucial for incremental learning.
- Orthogonal constraints and SGP contribute performance gains of 3.3 and 2.6 percentage points respectively, acting complementarily.
- On longer task sequences (10 tasks), the advantages of STSP become more pronounced, showing that orthogonal subspace constraints maintain better class separation as the number of classes increases.
- On the time-sensitive SSv2 dataset, incorporating temporal modeling yields the largest improvement, validating the value of the temporal subspace in TSC.

## Highlights & Insights
- **Replacing fully-connected layers with subspace representation is an elegant design**: Orthogonal subspaces naturally resolve conflicts between new and old classes in incremental learning without requiring complex knowledge distillation or data replay. This approach can be transferred to other incremental learning scenarios.
- **The decoupled spatial-temporal protection strategy is clever**: SGP protects only spatial features (as spatial appearance is fundamental), leaving room for the temporal dimension to adapt. This "selective protection" is more efficient than comprehensive protection.
- **No requirement to save old data**: Through the combination of orthogonal subspaces and gradient projection, the storage and privacy issues associated with data replay are completely avoided.

## Limitations & Future Work
- As the number of classes increases, the dimensionality of the feature space remains finite, placing an upper limit on the number of orthogonal subspaces that can be allocated. When there are very many classes, orthogonal constraints might not be fully satisfiable.
- SGP requires storing additional feature statistics of old tasks (though much smaller than storing raw data); in extremely long-sequence incremental learning, storage overhead still grows.
- The paper is primarily validated on action recognition; adaptation may be required for more fine-grained video understanding tasks (such as incremental learning for temporal action localization or video question answering).
- The dimensionality of the subspace bases is a fixed hyperparameter; different classes might require subspaces of different dimensions to represent their intra-class diversity.

## Related Work & Insights
- **vs LUCIR**: LUCIR uses a cosine classifier + knowledge distillation and requires storing old exemplars for replay. STSP completely avoids data storage using orthogonal subspaces, while providing richer class representation.
- **vs GPM**: GPM introduces gradient projection in image incremental learning but processes all features across all layers. STSP's SGP only protects core directions of spatial features, which is more efficient and leaves more room for temporal learning.
- **vs vCLIMB**: vCLIMB is the first work to systematically study VCIL, but mainly transfers image-based incremental learning methods, ignoring video's spatial-temporal characteristics. STSP's TSC models spatial-temporal dynamics directly.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of subspace classifier and selective gradient projection is novel in VCIL.
- Experimental Thoroughness: ⭐⭐⭐⭐ Validated on three datasets under multiple incremental settings with sufficient ablation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and design, with a complete logical chain.
- Value: ⭐⭐⭐⭐ Holds practical value for the video incremental learning community, where the freedom from data replay is a major benefit.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Temporal Imbalance of Positive and Negative Supervision in Class-Incremental Learning](../../CVPR2026/self_supervised/temporal_imbalance_of_positive_and_negative_supervision_in_class-incremental_lea.md)
- [\[CVPR 2026\] Exemplar-Free Class Incremental Learning via Preserving Class-Discriminative Structure](../../CVPR2026/self_supervised/exemplar-free_class_incremental_learning_via_preserving_class-discriminative_str.md)
- [\[CVPR 2025\] SMILE: Infusing Spatial and Motion Semantics in Masked Video Learning](../../CVPR2025/self_supervised/smile_infusing_spatial_and_motion_semantics_in_masked_video_learning.md)
- [\[ECCV 2024\] ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](vic-mae_self-supervised_representation_learning_from_images_and_video_with_contr.md)
- [\[ECCV 2024\] Self-supervised Video Copy Localization with Regional Token Representation](self-supervised_video_copy_localization_with_regional_token_representation.md)

</div>

<!-- RELATED:END -->
