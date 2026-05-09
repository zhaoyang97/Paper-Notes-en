---
title: >-
  [Paper Note] Generating Attribute-Aware Human Motions from Textual Prompt
description: >-
  [AAAI 2026][Human Understanding][Human motion generation] This paper proposes AttrMoGen, a framework that decouples action semantics from human attributes (age, gender, etc.) via a Structural Causal Model (SCM)-based Causal Information Bottleneck, enabling attribute-aware human motion generation from text prompts. The authors also introduce HumanAttr, the first large-scale text-motion dataset with extensive attribute annotations.
tags:
  - AAAI 2026
  - Human Understanding
  - Human motion generation
  - attribute-awareness
  - causal decoupling
  - VQVAE
  - text-driven
date: 2026-05-08
content_hash: ada7807d97e57037
---

# Generating Attribute-Aware Human Motions from Textual Prompt

**Conference**: AAAI 2026
**arXiv**: [2506.21912](https://arxiv.org/abs/2506.21912)
**Code**: None
**Area**: Human Understanding
**Keywords**: Human motion generation, attribute-awareness, causal decoupling, VQVAE, text-driven

## TL;DR

This paper proposes AttrMoGen, a framework that decouples action semantics from human attributes (age, gender, etc.) via a Structural Causal Model (SCM)-based Causal Information Bottleneck, enabling attribute-aware human motion generation from text prompts. The authors also introduce HumanAttr, the first large-scale text-motion dataset with extensive attribute annotations.

## Background & Motivation

Text-driven human motion generation has seen significant progress in recent years. However, existing methods overlook a fundamental factor: **human attributes (e.g., age, gender, weight, height) have a pronounced effect on motion patterns**.

**Attribute dependency of motion patterns**: Elderly individuals and adolescents walk in distinctly different ways; even for the same action "walking," people of different ages, genders, and body types exhibit natural variation in stride length, joint range, and movement amplitude.

**Semantic-attribute coupling**: A motion sequence simultaneously encodes action semantics (walking, running) and human attribute information, yet textual descriptions typically focus only on action semantics. Existing methods align text and motion in a shared space without distinguishing between the two, which may impede alignment quality.

**Dataset gap**: Large-scale text-motion datasets with comprehensive human attribute annotations are lacking. Existing datasets either omit attribute labels (e.g., HumanML3D), cover a narrow attribute range (e.g., 90% of KIT subjects are aged 18–45), or are extremely small in scale.

**Core observation**: Human motion can be factored into two components—action semantics and human attributes. Since textual descriptions correspond only to action semantics, explicit decoupling of the two is necessary.

## Method

### Overall Architecture

AttrMoGen consists of two main components:

1. **Semantic-Attribute Decoupling VQVAE (Decoup-VQVAE)**: The encoder removes attribute information from raw motion via a Causal Information Bottleneck to obtain attribute-free semantic tokens; the decoder reconstructs motion from semantic tokens and attribute labels.
2. **Semantics Generative Transformer**: Predicts semantic tokens from text; at inference time, combines with user-specified attributes to generate motion.

### Key Designs

#### 1. **SCM-based Decoupling**

The problem is formulated as causal factor disentanglement. Definitions:
- $X$: raw motion
- $Y$: target semantic tokens
- $S$: action semantics (causal factor of $Y$)
- $A$: human attributes (non-causal factor of $Y$, yet essential to the constitution of $X$)

The Causal Information Bottleneck (CIB) objective:
$$CIB(X,Y,S,A) = I(X;S,A) + I(Y;S) - I(S;A) - \lambda I(X;S)$$

Role of each term:
- $I(X;S,A)$: ensures $S$ and $A$ are jointly sufficient to reconstruct $X$ → reconstruction loss
- $I(Y;S)$: ensures $S$ is sufficient to derive $Y$ → quantization process
- $-I(S;A)$: minimizes mutual information between $S$ and $A$ → **core of decoupling**
- $-\lambda I(X;S)$: restricts information flow from $X$ to $S$ → information bottleneck

#### 2. **Decoupling Implementation (Decoupling Term $-I(S;A)$)**

Decoupling is achieved via an upper-bound estimate of mutual information:

$$I(S;A) \leq \log|A| - \mathbb{E}_{s\sim p(S)}H(A|S=s)$$

A surrogate attribute classifier $h$ is introduced to classify human attribute $A$ from semantic embedding $S$, with its output serving as the conditional probability $p(A|s_i) = h(s_i)$. The following loss is minimized:

$$\mathcal{L}_{entropy} = -\sum_{i=1}^{B} H(A|S=s_i)$$

Minimizing $\mathcal{L}_{entropy}$ reduces $I(S;A)$, thereby eliminating attribute information from $S$. The classifier $h$ is updated alternately with encoder $f$ and decoder $g$.

#### 3. **Information Bottleneck Implementation (Bottleneck Term $-\lambda I(X;S)$)**

Counterfactual motion is used to align encoder outputs. Core idea: if the encoder correctly decouples the two factors, motions sharing **the same action semantics but different attributes** should yield identical semantic embeddings.

- Counterfactual motion is generated via the decoder: $X^- = g(S, A^-)$, where $A^-$ is a randomized attribute.
- A similarity matrix is computed between the encoder outputs of the original and counterfactual motions.
- Bottleneck loss: $\mathcal{L}_{bottleneck} = \|\tilde{D}(X, X^-) - I\|_F^2$

This enforces proximity between $f(X)$ and $f(X^-)$ while maintaining inter-channel independence.

#### 4. **Semantics Generative Transformer**

The model adopts the MoMask (masked Transformer) architecture. During training, semantic tokens are randomly masked and predicted conditioned on CLIP text features. At inference:
1. Text → Semantics Generative Transformer → semantic tokens
2. Semantic tokens + attribute input → Decoup-VQVAE decoder → attribute-aware motion

### Loss & Training

The overall loss function:
$$\mathcal{L}_{overall} = \mathcal{L}_{vqvae} + \alpha\mathcal{L}_{entropy} + \lambda\mathcal{L}_{bottleneck}$$

where $\mathcal{L}_{vqvae} = \mathcal{L}_{rec} + \mathcal{L}_{commit} + \mathcal{L}_{embed}$, with default $\alpha=0.01$, $\lambda=0.5$.

Training strategy: encoder $f$, decoder $g$, and surrogate attribute classifier $h$ are updated alternately; the classifier is supervised with cross-entropy loss $\mathcal{L}_{CE}$.

## Key Experimental Results

### HumanAttr Dataset

| Subset | Subjects | Motions | Duration (min) | Age Range |
|---------|---------|--------|----------|---------|
| BMLmovi | 86 | 1,801 | 161.8 | [17, 33] |
| ETRI-Activity3D | 100 | 3,727 | 691.8 | [21, 88] |
| KIT | 55 | 4,231 | 463.1 | [15, 55] |
| Nymeria | 264 | 6,850 | 552.4 | [18, 50] |
| **Total** | **640** | **18,199** | **2,135.4** | **[5, 88]** |

### Main Results

| Method | R-Precision Top-1↑ | Top-3↑ | FID↓ | MM-Dist↓ | Diversity→ | MModality↑ |
|------|-------------------|--------|------|----------|-----------|------------|
| T2M | 0.592 | 0.859 | 1.909 | 3.827 | 18.856 | 2.627 |
| MotionDiffuse | 0.670 | 0.928 | 0.416 | 2.704 | 18.968 | 2.435 |
| MoMask | 0.685 | 0.925 | 0.245 | 2.602 | 18.981 | 1.438 |
| GenMoStyle | 0.680 | 0.925 | 0.332 | 2.649 | 19.118 | 1.588 |
| **AttrMoGen** | **0.705** | **0.940** | **0.089** | **2.266** | 19.268 | **1.250** |

AttrMoGen reduces FID from MoMask's 0.245 to 0.089 (−63.7%) and MM-Dist from 2.602 to 2.266.

### Ablation Study

| Configuration | R-Precision Top-1↑ | FID↓ | MM-Dist↓ | Note |
|------|-------------------|------|----------|------|
| MoMask (baseline) | 0.685 | 0.245 | 2.602 | No attribute information |
| w/ attr test only | 0.603 | 0.957 | 3.815 | Adding attribute text at test only → severe degradation |
| w/ attr train+test | 0.689 | 0.203 | 2.518 | Adding attribute text at both stages → limited gain |
| w/o entropy | 0.686 | 0.489 | 2.523 | Removing decoupling term → FID doubles |
| w/o bottleneck | 0.686 | 0.184 | 2.486 | Removing information bottleneck |
| $\lambda=0.25$ | 0.698 | 0.098 | 2.326 | Smaller bottleneck weight |
| $\lambda=1$ | 0.701 | 0.088 | 2.332 | Larger bottleneck weight |
| **AttrMoGen ($\lambda=0.5, \alpha=0.01$)** | **0.705** | **0.089** | **2.266** | Optimal configuration |

**Key ablation findings**:
- Naively appending attribute information to text at test time severely degrades performance (FID: 0.245→0.957), as the model has never encountered this format during training.
- The decoupling term $\mathcal{L}_{entropy}$ has the largest impact on FID (0.089→0.489), validating the central role of causal decoupling.

### Attribute Control Verification

| Attribute | Group | MoMask Acc | AttrMoGen Acc | Note |
|------|------|-----------|---------------|------|
| Gender | male | 0.747 | **0.992** | Extremely high control accuracy |
| Gender | female | 0.546 | **0.985** | Extremely high control accuracy |
| Age | 5–18 | 0.314 | **0.422** | Age control is more difficult |
| Age | 60–88 | 0.556 | **0.787** | High discriminability for elderly group |

### Key Findings

1. **Attribute information is decisive for motion quality**: A 63.7% FID reduction demonstrates the substantial value of attribute-awareness.
2. **Direct concatenation of attribute text is not an effective strategy**: Explicit decoupling mechanisms are required rather than simple text-level fusion.
3. **Causal decoupling is the core contribution**: Ablation of $\mathcal{L}_{entropy}$ shows that the decoupling term contributes most to performance.
4. **Counterfactual alignment is effective**: The bottleneck loss enforces attribute invariance in semantic embeddings via counterfactual motions.
5. **Gender control accuracy exceeds 98%**, whereas age control is comparatively more difficult, as the influence of age on motion patterns is more subtle.

## Highlights & Insights

- **Pioneering integration of human attributes into text-driven motion generation**, filling a significant gap in the field.
- **Elegant application of causal modeling**: The SCM framework formally characterizes action semantics and attributes as causal and non-causal factors respectively—theoretically principled and practically effective.
- **Clever use of counterfactual motion generation**: The decoder itself is leveraged to synthesize "same semantics, different attributes" counterfactual samples to supervise training.
- **HumanAttr dataset** constitutes an important community contribution (640 subjects, age range 5–88).
- A fundamental distinction from style-based motion generation: style reflects subjective intent (proud/depressed), whereas attributes are objective biomechanical characteristics.

## Limitations & Future Work

- Attribute labels are discretized (4 age groups / 2 gender categories), discarding continuous attribute information.
- Age distribution in HumanAttr is imbalanced (young adults dominate), with limited data for extreme age groups (5–18, 60–88).
- Only age and gender are employed as attributes; weight and height, though annotated, are not utilized in the main experiments.
- The surrogate attribute classifier may introduce additional bias.
- Motion representation is restricted to SMPL parameterization, without modeling fine-grained hand motion.

## Related Work & Insights

- **Relationship to MoMask**: AttrMoGen augments the MoMask architecture with an attribute decoupling layer.
- A comparative experiment replacing style labels with attribute labels in style-based methods (GenMoStyle) confirms the necessity of dedicated decoupling over style transfer approaches.
- The Causal Information Bottleneck (CIB) has been applied to image debiasing and fairness learning; this work represents its first introduction into motion generation.
- The framework holds direct value for virtual human and digital twin applications, where different characters must exhibit motion patterns consistent with their respective attributes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic incorporation of human attributes into motion generation; the SCM-based decoupling framework is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive ablation and attribute group experiments; the new dataset is well-constructed.
- Writing Quality: ⭐⭐⭐⭐ — Problem formulation is clear; causal modeling is rigorously presented.
- Value: ⭐⭐⭐⭐⭐ — Dual contributions of dataset and method; significant long-term impact.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] QuaMo: Quaternion Motions for Vision-based 3D Human Kinematics Capture](../../ICLR2026/human_understanding/quamo_quaternion_motion_kinematics.md)
- [\[AAAI 2026\] CLIP-FTI: Fine-Grained Face Template Inversion via CLIP-Driven Attribute Conditioning](clip-fti_fine-grained_face_template_inversion_via_clip-driven_attribute_conditio.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[ICCV 2025\] KinMo: Kinematic-Aware Human Motion Understanding and Generation](../../ICCV2025/human_understanding/kinmo_kinematic-aware_human_motion_understanding_and_generation.md)
- [\[ACL 2026\] ForgeryTalker: Generating Attribution Reports for Manipulated Facial Images](../../ACL2026/human_understanding/generating_attribution_reports_for_manipulated_facial_images_a_dataset_and_basel.md)

<!-- RELATED:END -->
