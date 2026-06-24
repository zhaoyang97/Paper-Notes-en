---
title: >-
  [Paper Note] Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases
description: >-
  [ECCV 2024][Human Understanding][Kinematic Phrases] This paper proposes Kinematic Phrases (KP) as an intermediate representation between human motion and action semantics. KP is based on objective kinematic facts, possesses appropriate abstraction, interpretability, and generalization capabilities, and is used to construct a motion understanding system and a white-box motion generation evaluation benchmark named KPG.
tags:
  - "ECCV 2024"
  - "Human Understanding"
  - "Kinematic Phrases"
  - "Action Semantics"
  - "Motion Generation Evaluation"
  - "Motion Understanding"
  - "Intermediate Representation"
date: 2026-05-08
content_hash: e55d211f46638af0
---

# Bridging the Gap Between Human Motion and Action Semantics via Kinematic Phrases

**Conference**: ECCV 2024  
**arXiv**: [2310.04189](https://arxiv.org/abs/2310.04189)  
**Code**: [https://foruck.github.io/KP/](https://foruck.github.io/KP/)  
**Area**: Human Understanding / Motion Generation  
**Keywords**: Kinematic Phrases, Action Semantics, Motion Generation Evaluation, Motion Understanding, Intermediate Representation

## TL;DR

This paper proposes Kinematic Phrases (KP) as an intermediate representation between human motion and action semantics. KP is based on objective kinematic facts, possesses appropriate abstraction, interpretability, and generalization capabilities, and is used to construct a motion understanding system and a white-box motion generation evaluation benchmark named KPG.

## Background & Motivation

**Background**: Motion understanding aims to establish a reliable mapping between motion and action semantics. Currently, text-to-motion generation is developing rapidly, with representative methods including MLD, T2M-GPT, and MoMask. These methods directly map text descriptions to motion sequences, or utilize a motion latent space compressed by a VAE as an intermediary.

**Limitations of Prior Work**: There is a severe many-to-many relationship between motion and action semantics. An abstract action semantic (e.g., "walking forward") can be expressed by motions with huge perceptual differences (e.g., walking with arms raised vs. walking with arms swinging). Conversely, the same motion can carry different semantics in different contexts. This database leads to two specific issues: (1) The reliability of the direct mapping paradigm is insufficient, as models cannot guarantee that the generated motion is consistent with the specified semantics; (2) Existing automated evaluation metrics (FID, R-Precision) have become ineffective—they rely on the latent space of black-box pre-trained models and cannot reliably evaluate motion-semantic consistency, with multiple methods even outperforming the Ground Truth in terms of R-Precision.

**Key Challenge**: There is a huge modal gap between the motion space and the action semantic space. Motion is low-level, continuous, and represents high-dimensional skeleton sequences, while action semantics are high-level, discrete, and represent abstract natural language descriptions. Directly mapping these two spaces is extremely challenging.

**Goal**: (1) How to narrow the modal gap between motion and action semantics; (2) How to construct a reliable motion-semantic consistency evaluation method to replace ineffective black-box metrics.

**Key Insight**: Inspired by kinematics research and qualitative pose representations (Posebits), the authors argue that an intermediate representation is needed to bridge these two spaces. This intermediate representation should: (1) be based on objective kinematic facts rather than subjective descriptions; (2) have an appropriate level of abstraction to eliminate the impact of motion perturbations; (3) be able to automatically convert between motion and text.

**Core Idea**: Propose Kinematic Phrases (KP) as an intermediate representation to bridge motion and semantics using objective kinematic facts (sign changes in joint position variations, joint pair distances, limb angles, etc.), and construct a white-box motion generation evaluation benchmark based on it.

## Method

### Overall Architecture

The method consists of two major parts: (1) KP definition and knowledge base construction—defining 6 types of KPs and extracting them from 140K motion sequences to build a large-scale knowledge base containing motion, text, and KP modalities; (2) KP application—building a motion understanding system (motion interpolation, modification, generation) and a motion generation evaluation benchmark KPG based on KPs.

### Key Designs

1. **Kinematic Phrases Definition**:

    - **Function**: Categorically and symbolically represent objective kinematic facts in motion sequences.
    - **Mechanism**: KPs cover human motion from four kinematic levels. For each phrase in each frame, a scalar indicator is calculated, and its sign determines the category of the phrase (positive/negative/zero). The six types of KPs are:
        - Position Phrase (PP, 34 types): Motion direction of a joint relative to a reference direction.
        - Pairwise Relative Position Phrase (PRPP, 242 types): Relative position relationship between joint pairs.
        - Pairwise Distance Phrase (PDP, 81 types): Distance changes between joint pairs.
        - Limb Angle Phrase (LAP, 8 types): Limb bending/extension.
        - Limb Orientation Phrase (LOP, 24 types): Limb orientation.
        - Global Velocity Phrase (GVP, 3 types): Global velocity direction.
    - A total of 392 phrases capture motion through sign changes, minimizing human-defined bias (using only positive/negative/zero signs as the standard).
    - **Design Motivation**: KPs focus only on sign changes of objective kinematic facts, avoiding the bias of subjective semantic annotations. Meanwhile, they are robust to small perturbations—minor motion changes do not change the category of the KP.

2. **Joint Motion-KP Latent Space Learning**:

    - **Function**: Leverage the clarity and interpretability of KPs to guide the structure of the motion latent space.
    - **Mechanism**: Train two VAEs—Motion VAE $\{\mathcal{E}_m, \mathcal{D}_m\}$ and KP VAE $\{\mathcal{E}_p, \mathcal{D}_p\}$, using Transformer architectures. The latent spaces of the two VAEs are aligned using a distribution alignment loss $\mathcal{L}_{da} = KL(\phi_m, \phi_p) + KL(\phi_p, \phi_m)$ and an embedding alignment loss $\mathcal{L}_{emb} = \|z_m - z_p\|_1$. During training, up to 20% of KPs are randomly zeroed out to enhance robustness. The joint space supports cross-decoding: any combination of motion codes and KP codes can be input to either decoder.
    - **Design Motivation**: A pure motion VAE latent space lacks semantic structure. With the guidance of KPs, the latent space can encode semantic information at the kinematic level.

3. **Kinematic Prompt Generation Benchmark (KPG)**:

    - **Function**: Provide a white-box, reliable evaluation benchmark for text-to-action generation.
    - **Mechanism**: KPs are automatically converted into 7,796 text prompts via templates, divided into 4 groups: atomic prompts (252, involving a single KP like "left hand moves upward"), repetitive prompts (492, e.g., "left arm bends twice"), sequential prompts (3,912, two KPs executed sequentially), and simultaneous prompts (3,120, two KPs executed simultaneously). During evaluation, KPs are extracted after motion generation, and the hit rate is judged by detecting whether specific KP patterns occur. Key threshold: the target KP must appear continuously for at least 5 frames in the extracted KP sequence.
    - **Design Motivation**: Existing metrics (FID, R-Precision) rely on black-box models and can be overfitted. KPG evaluation is entirely rule-based, requires no pre-trained models, and achieves a fully white-box evaluation. By reducing semantic complexity (only requiring the generation of specific kinematic facts), the basic capabilities of the model can be evaluated more accurately.

### Loss & Training

The overall loss is $\mathcal{L} = \lambda_1 \mathcal{L}_{rec} + \lambda_2 \mathcal{L}_{KL} + \lambda_3 \mathcal{L}_{da} + \lambda_4 \mathcal{L}_{emb}$, where $\lambda_i = 1$. Reconstruction loss includes L1 loss of motion representation, KP, skeletal joints, downsampled mesh vertices, and joint accelerations. The joint space is frozen after training for 6000 epochs, and the text-to-motion latent diffusion model is trained for 3000 epochs. The model has only 45.1M parameters (compared to 228M for T2M-GPT).

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours | Prev. SOTA | Note |
|--------|------|------|----------|------|
| HumanML3D | R-P@1 | 0.496 | 0.521(MoMask) | Traditional metric is close |
| HumanML3D | FID | 0.275 | 0.045(MoMask) | But this paper points out R-P/FID has become ineffective |
| KPG | Overall Accuracy | 57.86% | 55.59%(T2M-GPT) | KP space method is superior |
| KPG | Atomic Accuracy | 98.80% | 97.22%(T2M-GPT) | Best basic capability |
| KPG | Sequential Accuracy | 71.32% | 70.24%(T2M-GPT) | Best compositional capability |

### Ablation Study

| Configuration | KPG Acc.% | Diversity | Note |
|------|-----------|-----------|------|
| Full Method | 57.86 | 6.048 | - |
| W/o KP | 39.94 | 5.526 | Verifies the core role of KP |
| W/o Joint KP | 50.03 | 5.685 | Joint KP contributes the most |
| W/o Joint Pair KP | 47.24 | 5.772 | Joint pair KP is important |
| W/o Limb KP | 55.92 | 5.934 | Low impact |
| W/o Body KP | 56.84 | 5.871 | Low impact |

### Key Findings

- Existing methods perform far worse on KPG than expected; even the simplest atomic prompts have a failure rate of about 5%.
- There is a contradiction between R-Precision and user evaluation: a method with high R-Precision does not necessarily satisfy users.
- The consistency between KPG evaluation and human evaluation reaches 84%, validating its effectiveness as a white-box evaluation metric.
- Different methods exhibit different failure modes: MDM/MLD have insufficient motion magnitude, T2M-GPT has redundant motion, and ReMoDiffuse is limited by the retrieval library.
- A user study (36 volunteers × 600 sentences) shows that the proposed method is comparable to T2M-GPT in semantic consistency, but with only 1/5 of the parameters.

## Highlights & Insights

- **Elegant design of intermediate representation**: KP captures kinematic facts using only sign changes, which is both objective and interpretable, and perfectly bridges the motion and semantic spaces.
- **Innovative evaluation paradigm**: KPG provides a white-box alternative for motion generation evaluation, revealing the unreliability of existing black-box metrics.
- **Large-scale knowledge base**: Collected 140K motion sequences to build KP Base, showing excellent scalability.
- **Deep insights**: Reveals the deficiencies of existing text-to-motion models in basic kinematic understanding through KPG.

## Limitations & Future Work

- KP currently only captures sign changes, losing magnitude and velocity information.
- Skeleton granularity limits the expression of fine-grained movements like fingers.
- The KPG prompt structure is relatively simple (at most binary combinations), offering limited evaluation capability for complex daily actions.
- LLMs can be introduced to enhance the semantic representation capabilities of KPs.
- KP Base can be extended to other modalities such as 2D pose and egocentric perspectives.

## Related Work & Insights

- **Action Representation**: Posebits (static pose boolean relationships) $\to$ KP (dynamic kinematic facts), which is a natural extension from static to dynamic.
- **Motion Generation**: MLD, T2M-GPT, and MoMask represent the current SOTA, but KPG reveals their shortcomings in basic capabilities.
- **Evaluation Methods**: FID and R-Precision are borrowed from image generation but have shown signs of becoming ineffective in the motion domain.
- **Insights**: In other domains requiring generation quality evaluation (such as 3D generation and video generation), building white-box evaluation benchmarks similar to KPG can also be considered.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The KP intermediate representation and the KPG white-box evaluation benchmark are both entirely new contributions with a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes both traditional and KPG evaluations, detailed ablations, and a large-scale user study.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, convincing derivation of motivation, and deep insights into domain issues.
- Value: ⭐⭐⭐⭐⭐ KP and KPG have a profound impact on the motion understanding field, potentially changing the evaluation paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] InclusiveVidPose: Bridging the Pose Estimation Gap for Individuals with Limb Deficiencies in Videos](../../ICLR2026/human_understanding/inclusivevidpose_bridging_the_pose_estimation_gap_for_individuals_with_limb_defi.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](../../ICCV2025/human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[CVPR 2025\] Stochastic Human Motion Prediction with Memory of Action Transition and Action Characteristic](../../CVPR2025/human_understanding/stochastic_human_motion_prediction_with_memory_of_action_transition_and_action_c.md)
- [\[ICLR 2026\] From Pixels to Semantics: Unified Facial Action Representation Learning for Micro-Expression Analysis](../../ICLR2026/human_understanding/from_pixels_to_semantics_unified_facial_action_representation_learning_for_micro.md)
- [\[ECCV 2024\] HUMOS: Human Motion Model Conditioned on Body Shape](humos_human_motion_model_conditioned_on_body_shape.md)

</div>

<!-- RELATED:END -->
