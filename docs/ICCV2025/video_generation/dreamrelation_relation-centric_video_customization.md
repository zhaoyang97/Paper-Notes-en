---
title: >-
  [Paper Note] DreamRelation: Relation-Centric Video Customization
description: >-
  [ICCV 2025][Video Generation][Relation video customization] DreamRelation is proposed as the first relation-centric video customization method. Through a Relation LoRA Triplet combined with Hybrid Mask Training, it achieves disentanglement between relation and appearance, and enhances relational dynamics learning via a spatiotemporal relational contrastive loss, enabling animals to imitate human interactions.
tags:
  - "ICCV 2025"
  - "Video Generation"
  - "Relation video customization"
  - "MM-DiT"
  - "LoRA"
  - "contrastive learning"
  - "disentangled learning"
date: 2026-05-08
content_hash: 4b27469e92ffa8b9
---

# DreamRelation: Relation-Centric Video Customization

**Conference**: ICCV 2025
**arXiv**: [2503.07602](https://arxiv.org/abs/2503.07602)  
**Code**: Available (coming soon)  
**Area**: Video Generation / Video Customization
**Keywords**: Relation video customization, MM-DiT, LoRA, contrastive learning, disentangled learning

## TL;DR

DreamRelation is proposed as the first relation-centric video customization method. Through a Relation LoRA Triplet combined with Hybrid Mask Training, it achieves disentanglement between relation and appearance, and enhances relational dynamics learning via a spatiotemporal relational contrastive loss, enabling animals to imitate human interactions.

## Background & Motivation

Existing video customization methods can personalize subject appearance and single-object motion, but cannot customize high-order interaction relations between two subjects (e.g., handshaking, hugging). The core challenges in relation customization are:

**Spatial Complexity**: Relations involve complex spatial layouts and positional variations.

**Temporal Complexity**: Relations contain subtle temporal dynamics.

**Appearance Entanglement**: Models tend to overfit appearance details while neglecting the semantics of the interaction itself.

Generating unconventional relations (e.g., animals shaking hands) via text descriptions is also infeasible — general-purpose T2V models such as Mochi cannot produce such non-standard interactions. More precise relational dynamics (e.g., "two persons walking toward each other from designated positions") also exceed the descriptive capacity of text.

This paper defines the **relation video customization task**: given a reference video demonstrating a relational pattern ⟨subject, relation, subject⟩, generate a video exhibiting that relation on novel subjects.

## Method

### Overall Architecture

DreamRelation builds upon Mochi (MM-DiT architecture) and comprises two parallel processes:
1. **Relational Decoupling Learning**: Decouples relation from appearance via Relation LoRA Triplet and Hybrid Mask Training.
2. **Relational Dynamics Enhancement**: Reinforces relational dynamics learning via a spatiotemporal relational contrastive loss.

The training objective for the video diffusion Transformer is: $\mathcal{L}(\theta) = \mathbb{E}_{\bm{z}, \epsilon, \bm{c}, t}[\|\epsilon - \epsilon_\theta(\bm{z}_t, \bm{c}, t)\|_2^2]$, where $\bm{z}_0 \in \mathbb{R}^{f \times h \times w \times c}$ denotes the video latent encoded by a 3D VAE.

### Key Designs

**1. Relation LoRA Triplet**

The relational pattern is decomposed into a triplet ⟨$S_1$, $R$, $S_2$⟩, and a composite LoRA set is designed accordingly:
- **Relation LoRAs**: Injected into the **Query** and **Key** matrices of MM-DiT full attention → captures relational information.
- **Subject LoRA 1 & 2**: Injected into the **Value** matrix → captures appearance information of the two subjects.
- **FFN LoRA**: Injected into the attention linear layers → refines the outputs of Relation and Subject LoRAs.

Note that text and vision tokens in MM-DiT are handled by separate LoRA sets.

**Theoretical Basis for LoRA Placement — Q/K/V Role Analysis**:

This constitutes the most critical insight of the paper. Through visualization and singular value decomposition (SVD), the distinct roles of Q, K, and V in MM-DiT are revealed:

- **Value features**: Contain rich appearance information (e.g., blue glasses, birthday hats, clothing), with relational information (e.g., handshaking regions) entangled with appearance.
- **Query/Key features**: Exhibit highly abstract and homogeneous patterns, showing cross-video similarity distinct from Value features.
- **Subspace similarity quantification**: SVD is applied to Q, K, V weight matrices; the top-r left singular vectors are used to compute normalized subspace similarity. Results show high Q-K similarity and very low Q/K–V similarity.

Consequently, Relation LoRAs are placed on Q/K (where information is decoupled from appearance, facilitating disentanglement), and Subject LoRAs on V (aligned with appearance information). This makes DreamRelation the **first relation video generation framework with interpretable components**.

**2. Hybrid Mask Training Strategy (HMT)**

Grounding DINO and SAM are employed to generate segmentation masks $M_{S_1}$, $M_{S_2}$ for the two subjects in the reference video. The relation mask $M_R$ is defined as the union of the two subject masks (minimum bounding rectangle, following a classic convention in relation detection). Due to temporal compression by factor $T_c$ in the 3D VAE, masks are averaged within every $T_c$ frames.

Two key strategies are incorporated during training:

**LoRA Selection Strategy**: At each iteration, only the Relation LoRAs or one Subject LoRA is randomly selected for updating. When Relation LoRAs are selected, both Subject LoRAs are simultaneously trained to provide appearance cues, helping Relation LoRAs focus on relational information. FFN LoRAs participate in all iterations.

**Enhanced Diffusion Loss**: The loss weight is amplified over the target region using the corresponding mask:

$$\mathcal{L}_{rec} = \mathbb{E}_{\bm{z}, \epsilon, \bm{c}, t} (\lambda_m \mathbf{M}_l + 1) \cdot \|\epsilon - \epsilon_\theta(\bm{z}_t, \bm{c}, t)\|_2^2$$

where $l \in \{S_1, S_2, R\}$ denotes the selected mask type and $\lambda_m = 50$ is the mask weight.

**At inference**: Subject LoRAs are excluded (to prevent importing appearance from reference videos), and only Relation LoRAs and FFN LoRAs are used, enhancing generalization.

**3. Space-Time Relational Contrastive Loss (RCL)**

Relational dynamics learning is explicitly enhanced to reduce dependence on appearance details. The core idea is that relational dynamics manifest in inter-frame variations, while appearance information is static within individual frames.

- **Anchor feature $A \in \mathbb{R}^{(f-1) \times c}$**: Computed by taking frame-wise differences $\bar{\epsilon}$ along the temporal dimension of model outputs, then spatially averaged.
- **Positive samples $P$**: $n_{pos}$ relational dynamics features sampled from other videos exhibiting the same relation.
- **Negative samples $N$**: $n_{neg}$ appearance features sampled from single-frame model outputs.

The contrastive loss is based on InfoNCE:

$$\mathcal{L}_{RCL} = \log \sum_{i=1}^{f-1} \frac{-\sum_{j=1}^{n_{pos}} \exp(A_i^\top P_{ij} / \tau)}{\sum_{j} \exp(A_i^\top P_{ij} / \tau) + \sum_{k} \exp(A_i^\top N_{ik} / \tau)}$$

A FIFO memory bank (size 64) is used to store and dynamically update positive and negative samples, expanding contrastive learning effectiveness and training stability.

### Loss & Training

Total loss: $\mathcal{L}_{total} = \mathcal{L}_{rec} + \lambda_1 \mathcal{L}_{RCL}$

Training configuration:
- Base model: Mochi; AdamW optimizer; lr=2e-4; weight decay=0.01
- 2400 training iterations; LoRA rank=16; $\lambda_m=50$; $\lambda_1=0.01$
- $n_{pos}=4$; $n_{neg}=10$; $\tau=0.07$
- Generation resolution 61×480×848; batch size=1
- Inference: Euler Discrete scheduler; 64 steps; CFG scale=6.0

Dataset: 26 human interaction relations from NTU RGB+D; approximately 20 videos per relation for training; 40 designed prompts for evaluation.

## Key Experimental Results

### Main Results — Relation Video Customization

| Method | Relation Accuracy↑ | CLIP-T↑ | Temporal Consistency↑ | FVD↓ |
|------|:---:|:---:|:---:|:---:|
| Mochi (base) | 0.2623 | 0.3237 | 0.9888 | 2047 |
| Direct LoRA finetuning | 0.3258 | 0.2966 | 0.9945 | 2229 |
| ReVersion | 0.2690 | 0.3013 | 0.9921 | 2683 |
| MotionInversion | 0.3151 | 0.3217 | 0.9855 | 2085 |
| **DreamRelation** | **0.4452** | **0.3248** | **0.9954** | **2080** |

### Ablation Study — Component-wise Contribution

| Configuration | Relation Accuracy↑ | CLIP-T↑ | FVD↓ |
|------|:---:|:---:|:---:|
| w/o HMT | 0.3574 | 0.3244 | 2249 |
| w/o RCL | 0.3416 | 0.3185 | 2137 |
| w/o Relation LoRAs | 0.3626 | 0.3035 | 2318 |
| w/o Subject LoRAs | 0.3769 | 0.3147 | 2409 |
| w/o FFN LoRAs | 0.4021 | 0.3241 | 2370 |
| **Full Model** | **0.4452** | **0.3248** | **2080** |

LoRA placement ablation:

| Relation LoRA | Subject LoRA | Relation Accuracy↑ |
|:---:|:---:|:---:|
| V | Q, K | 0.3444 |
| Q | K, V | 0.3921 |
| K, V | Q | 0.3937 |
| **Q, K** | **V** | **0.4452** |

### Key Findings

1. **Substantial lead in Relation Accuracy**: 0.4452 vs. the second-best 0.3258 (Direct LoRA), a 36.6% improvement.
2. **All components are indispensable**: Removing any single component significantly degrades performance; HMT and RCL each contribute approximately 0.08–0.10 in accuracy.
3. **Q/K/V role analysis validated**: Placing Relation LoRAs on Q/K yields the best result (0.4452), while placement on V yields the worst (0.3444).
4. **Attention visualization**: The proposed method accurately focuses attention on the hand interaction region for "shaking hands."
5. **User study consistently favorable**: 15 annotators consistently preferred the proposed method in relation alignment, text alignment, and overall quality.
6. **RCL is transferable**: Applying RCL to MotionInversion also improves Relation Accuracy (0.3151→0.3633).

## Highlights & Insights

1. **Pioneering task definition**: The paper is the first to systematically define the relation video customization task and propose a complete solution.
2. **Interpretable architectural design**: SVD analysis provides theoretical grounding for LoRA placement decisions, rather than relying on empirical trial and error.
3. **Frame differences as relational dynamics**: Frame-wise differences capture temporal variations, while spatial averaging filters out appearance — an elegant and concise design.
4. **Impressive domain transfer**: Human relations (handshaking, hugging) can be transferred to animals such as cats and dogs.

## Limitations & Future Work

1. The Relation Accuracy metric relies on a VLM (Qwen-VL-Max) and is therefore constrained by its capabilities.
2. Training data covers only 26 human interaction relations; extension to more relation types remains to be explored.
3. The current method is limited to dual-subject relations; multi-subject scenarios require further design.
4. Computational overhead is substantial due to the large parameter count of the Mochi base model.
5. Excluding Subject LoRAs at inference enhances generalization but may not always be optimal.

## Related Work & Insights

- **MotionInversion**: A representative method for single-object motion customization relying on temporal attention layers absent in MM-DiT, making it unsuitable for relational modeling.
- **ReVersion**: A pioneering method for image relation customization that captures relations via textual inversion, but static images cannot represent dynamic interactions.
- **MM-DiT (Mochi)**: The foundational video DiT architecture; this paper provides the first in-depth analysis of Q/K/V functional differences within this architecture in the context of customization.
- **Insight**: Understanding the functional distinctions among Transformer components is a critical prerequisite for designing effective LoRA strategies.

## Rating

| Dimension | Score |
|------|:---:|
| Novelty | ⭐⭐⭐⭐⭐ |
| Technical Depth | ⭐⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] DualReal: Adaptive Joint Training for Lossless Identity-Motion Fusion in Video Customization](dualreal_adaptive_joint_training_for_lossless_identity-motion_fusion_in_video_cu.md)
- [\[ICCV 2025\] OCK: Unsupervised Dynamic Video Prediction with Object-Centric Kinematics](ock_unsupervised_dynamic_video_prediction_with_object-centric_kinematics.md)
- [\[CVPR 2025\] TokenMotion: Decoupled Motion Control via Token Disentanglement for Human-centric Video Generation](../../CVPR2025/video_generation/tokenmotion_decoupled_motion_control_via_token_disentanglement_for_human-centric.md)
- [\[CVPR 2026\] HandWorld: Hand-Centric Unified Video Action Generation](../../CVPR2026/video_generation/handworld_hand-centric_unified_video_action_generation.md)
- [\[AAAI 2026\] Mask2IV: Interaction-Centric Video Generation via Mask Trajectories](../../AAAI2026/video_generation/mask2iv_interaction-centric_video_generation_via_mask_trajectories.md)

</div>

<!-- RELATED:END -->
