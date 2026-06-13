---
title: >-
  [Paper Note] KinMo: Kinematic-Aware Human Motion Understanding and Generation
description: >-
  [ICCV 2025][Human Understanding][Human motion generation] This paper proposes the KinMo framework, which decomposes human motion into six kinematic groups and their interactions as a hierarchically describable representa…
tags:
  - "ICCV 2025"
  - "Human Understanding"
  - "Human motion generation"
  - "text-motion alignment"
  - "kinematic grouping"
  - "hierarchical representation"
  - "fine-grained control"
date: 2026-05-08
content_hash: 7ddf23e6fee7e544
---

# KinMo: Kinematic-Aware Human Motion Understanding and Generation

**Conference**: ICCV 2025  
**arXiv**: [2411.15472](https://arxiv.org/abs/2411.15472)  
**Code**: [https://andypinxinliu.github.io/KinMo](https://andypinxinliu.github.io/KinMo)  
**Area**: Human Body Understanding  
**Keywords**: Human motion generation, text-motion alignment, kinematic grouping, hierarchical representation, fine-grained control

## TL;DR

This paper proposes the KinMo framework, which decomposes human motion into six kinematic groups and their interactions as a hierarchically describable representation. An automatic annotation pipeline generates fine-grained textual descriptions at multiple granularities. Combined with hierarchical text-motion alignment (HTMA) and a coarse-to-fine motion generation strategy, KinMo significantly improves motion understanding and fine-grained motion generation.

## Background & Motivation

Existing text-driven human motion generation methods rely on global action descriptions (e.g., "run") and suffer from a fundamental **modality gap** problem:

**Many-to-many mapping ambiguity**: The same motion can be described in multiple ways ("pick up an object" vs. "bend down to reach something"), and the same text can correspond to multiple motion variants ("running" may refer to fast running, jogging, or running with raised arms, etc.)

**Lack of local motion controllability**: Existing models are capable of generating coherent whole-body motions from global descriptions but cannot independently control specific body parts — for example, specifying "raise the right hand while walking."

**Absence of spatial detail**: Global descriptions fail to capture details such as velocity, limb positioning, and kinematic dynamics.

Prior methods (e.g., LGTM, FG-MDM) attempt to generate supplementary text using LLMs, but such text is arbitrary and lacks a systematic, kinematics-based formulation.

## Method

### Overall Architecture

The KinMo framework consists of four components: (1) a describable motion representation that decomposes motion into six kinematic groups and their interactions; (2) the KinMo dataset, generated via a semi-supervised annotation pipeline with three-level textual descriptions; (3) Hierarchical Text-Motion Alignment (HTMA), which encodes text hierarchically and aligns it with motion at each level; and (4) coarse-to-fine motion generation conditioned on hierarchical embeddings.

### Key Designs

1. **Describable Motion Representation**: Human joints are organized along the kinematic tree into six kinematic groups $G = \{\text{Torso, Neck, Left Arm, Right Arm, Left Leg, Right Leg}\}$. For each group $g$, the following quantities are defined:

    - Group position: $\mathbf{P}_g(t) = \frac{1}{|J_g|} \sum_{j \in J_g} \mathbf{p}_j(t)$
    - Limb angles: $\Theta_g(t) = \{r_j(t) | j \in J_g\}$
    - Group velocity: $\mathbf{V}_g(t) = \frac{1}{|J_g|} \sum_{j \in J_g} \mathbf{v}_j(t)$

   Inter-group interactions are defined as positional differences, connecting joint angles, and relative velocities. This representation is a **linear transformation** of existing joint-level representations and can be losslessly converted back to the original form, while being naturally amenable to natural language description.

2. **Semi-Supervised Annotation Pipeline**: A three-step process generates fine-grained annotations:

    - **Spatial information**: PoseScript is used to generate detailed textual descriptions for each frame.
    - **Keyframe selection**: sBERT computes cosine similarity between frame descriptions; frames with similarity below a threshold of 0.8 are marked as keyframes.
    - **LLM inference**: GPT-4o-mini infers motion descriptions for each kinematic group and their interactions based on keyframe pose descriptions.

   Two human annotators iteratively refined the prompts until Cohen's Kappa exceeded 0.8. The total annotation cost was approximately $23.

3. **Hierarchical Text-Motion Alignment (HTMA)**: The core innovation. Rather than encoding all descriptions jointly, HTMA encodes them level by level and progressively refines representations via cross-attention:

    $\mathbf{h_c} = E_c(\text{emb}(T_c))$
    $\mathbf{h_g} = E_g(\text{CrossAttn}(\text{emb}(T_g), \mathbf{h_c}))$
    $\mathbf{h_i} = E_i(\text{CrossAttn}(\text{emb}(T_i), \mathbf{h_g}))$

   Each level uses a VAE-based ACTOR encoder with a shared architecture; cross-attention establishes connections between adjacent levels of description. InfoNCE contrastive learning loss is applied at each level for text-motion alignment.

4. **Coarse-to-Fine Motion Generation**: Built on the MoMask generation architecture. The generation process proceeds in three stages: initial tokens are generated conditioned on the global description embedding $\mu_c$; intermediate tokens are produced by re-feeding into the generator conditioned on the group-level embedding $\mu_g$; and final tokens are generated conditioned on the interaction-level embedding $\mu_i$. The generator shares weights across all three levels.

5. **Motion Reasoner**: Fine-tuned on LLaMA-3, this module automatically generates group-level and interaction-level descriptions from a global action description, so that users only need to provide simple text at inference time.

### Loss & Training

- Alignment training: InfoNCE + KL divergence + cross-modal embedding similarity + motion reconstruction loss
- Generation training: MoMask masked reconstruction loss; shared weights are used for three-level conditioning
- Motion Reasoner: Standard next-token prediction loss conditioned on the global description

## Key Experimental Results

### Main Results

**Text-Motion Retrieval** (HumanML3D, Protocol (a) — full test set):

| Method | R@1 ↑ | R@3 ↑ | R@10 ↑ | MedR ↓ |
|------|-------|-------|--------|--------|
| TMR | 5.68 | 14.04 | 30.94 | 28.00 |
| KinMo (DistilBERT) | 8.13 | 19.69 | 39.18 | 18.00 |
| **KinMo (RoBERTa)** | **9.05** | **20.47** | **41.60** | **16.00** |

*R@1 improves by 59% (5.68→9.05); MedR decreases by 43% (28→16).*

**Text-Motion Generation** (HumanML3D):

| Method | R-Prec Top3 ↑ | FID ↓ | MM-Dist ↓ |
|------|--------------|-------|----------|
| MoMask | 0.807 | 0.045 | 2.958 |
| FineMoGen | 0.784 | 0.151 | 2.998 |
| ParCo | 0.801 | 0.109 | 2.927 |
| **KinMo (HTMA)** | **0.821** | **0.039** | **2.901** |

*FID decreases by 13% (0.045→0.039); Top3 R-Prec improves by 1.7%.*

### Ablation Study

| Semantic Level | R@1 ↑ | R@3 ↑ | MedR ↓ |
|---------|-------|-------|--------|
| Global only | 3.67 | 10.32 | 40.00 |
| + Group-level | 7.58 | 16.97 | 22.00 |
| + Group + Interaction | **9.05** | **20.47** | **16.00** |
| − Cross-attention | 7.63 | 16.94 | 22.00 |

*Each additional level of description yields substantial improvements. Without cross-attention, adding interaction-level descriptions provides almost no benefit, demonstrating the necessity of hierarchical encoding.*

| Embedder | Config | FID ↓ | R-Prec Top3 ↑ |
|--------|------|-------|--------------|
| CLIP | Global only | 0.115 | 0.499 |
| CLIP | + Group + Interaction | 0.098 | 0.512 |
| HTMA | Global only | 0.056 | 0.512 |
| HTMA | + Group + Interaction | **0.044** | **0.527** |

*HTMA outperforms CLIP under all configurations, and the coarse-to-fine strategy proves effective with both embedders.*

### Key Findings

- User study (20 participants × 320 samples): KinMo achieves the highest MOS scores in realism, text alignment, and overall impression.
- KinMo is the only method capable of performing local temporal editing (e.g., modifying only the right arm motion) while preserving the naturalness of whole-body motion.
- In motion trajectory control experiments, KinMo achieves the lowest average control error (0.1657) and the lowest FID (0.103).
- The six-group kinematic decomposition substantially outperforms the two-group decomposition (upper/lower body) used in ParCo.

## Highlights & Insights

- Kinematic grouping serves as an elegant bridge between motion and language: the six-group division is consistent with human kinematics and naturally suited for natural language description.
- The semi-supervised annotation pipeline is remarkably efficient: annotating the entire HumanML3D dataset (44,970 motion sequences) costs approximately $23.
- The design philosophy of hierarchical alignment (coarse-to-fine, progressive refinement) offers broadly transferable insights for other multi-granularity text-X alignment tasks.

## Limitations & Future Work

- Dependence on HumanML3D: the dataset remains limited in scale and action diversity.
- The Motion Reasoner is susceptible to error propagation from LLM-generated descriptions.
- The number of inter-group interaction descriptions grows quadratically with the number of groups (6 groups → 15 interaction pairs), potentially yielding overly verbose description text.

## Related Work & Insights

- Comparison with TMR demonstrates that fine-grained descriptions improve alignment quality more substantially than enhanced encoders.
- LGTM and FG-MDM rely on LLMs to generate arbitrary supplementary descriptions, whereas KinMo systematically generates descriptions grounded in kinematic knowledge, yielding superior results (Table 3).
- Single-frame pose descriptions extracted by PoseScript can serve as a general intermediate representation bridging motion and language.

## Rating

- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GenM3: Generative Pretrained Multi-path Motion Model for Text Conditional Human Motion Generation](genm3_generative_pretrained_multi-path_motion_model_for_text_conditional_human_m.md)
- [\[AAAI 2026\] SOSControl: Enhancing Human Motion Generation through Saliency-Aware Symbolic Orientation and Timing Control](../../AAAI2026/human_understanding/soscontrol_enhancing_human_motion_generation_through_saliency-aware_symbolic_ori.md)
- [\[AAAI 2026\] ReAlign: Text-to-Motion Generation via Step-Aware Reward-Guided Alignment](../../AAAI2026/human_understanding/realign_text-to-motion_generation_via_step-aware_reward-guided_alignment.md)
- [\[ICCV 2025\] GENMO: A GENeralist Model for Human MOtion](genmo_a_generalist_model_for_human_motion.md)
- [\[NeurIPS 2025\] MOSPA: Human Motion Generation Driven by Spatial Audio](../../NeurIPS2025/human_understanding/mospa_human_motion_generation_driven_by_spatial_audio.md)

</div>

<!-- RELATED:END -->
