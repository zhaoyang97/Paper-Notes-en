---
title: >-
  [Paper Note] FineXtrol: Controllable Motion Generation via Fine-Grained Text
description: >-
  [AAAI 2026][Self-Supervised Learning][motion generation] This paper proposes FineXtrol, a framework that leverages temporally annotated, fine-grained body-part text descriptions as control signals. By combining a dual-branch ControlNet architecture with hierarchical contrastive learning to enhance the discriminability of the text encoder, FineXtrol achieves efficient, user-friendly, and precise controllable human motion generation, significantly outperforming existing methods on multi-body-part control benchmarks on HumanML3D.
tags:
  - AAAI 2026
  - Self-Supervised Learning
  - motion generation
  - controllable generation
  - fine-grained text
  - contrastive learning
  - diffusion model
date: 2026-05-08
content_hash: a32881dae5c40876
---

# FineXtrol: Controllable Motion Generation via Fine-Grained Text

**Conference**: AAAI 2026
**arXiv**: [2511.18927](https://arxiv.org/abs/2511.18927)
**Code**: N/A
**Area**: Self-Supervised
**Keywords**: motion generation, controllable generation, fine-grained text, contrastive learning, diffusion model

## TL;DR

This paper proposes FineXtrol, a framework that leverages temporally annotated, fine-grained body-part text descriptions as control signals. By combining a dual-branch ControlNet architecture with hierarchical contrastive learning to enhance the discriminability of the text encoder, FineXtrol achieves efficient, user-friendly, and precise controllable human motion generation, significantly outperforming existing methods on multi-body-part control benchmarks on HumanML3D.

## Background & Motivation

Text-driven human motion generation is widely applied in animation and digital human domains. Existing controllable motion generation methods fall into two categories: (1) approaches that use LLMs to expand text descriptions to improve generation precision, but whose expanded descriptions are often misaligned with ground-truth motions and lack explicit temporal cues (e.g., specifying *when* to raise a hand); and (2) approaches that use global 3D coordinate sequences as control signals, which are precise but computationally expensive (requiring coordinate system conversion) and difficult for users to provide directly. The root cause is a fundamental tension between precise controllability, user-friendliness, and computational efficiency. The paper's starting point is to use temporally annotated, fine-grained text descriptions from the FineMotion dataset—strictly aligned with ground-truth motions (e.g., "move the left hand toward the left thigh during 1.0–1.5s")—as control signals, combined with the ControlNet paradigm and hierarchical contrastive learning to achieve efficient controllable generation.

## Method

### Overall Architecture

FineXtrol adopts a dual-branch architecture (analogous to ControlNet). The inputs are coarse-grained text $\boldsymbol{p}$, fine-grained text control signal $\boldsymbol{c}$, and a noisy motion sequence $\boldsymbol{x_t}$; the output is the denoised clean motion sequence $\boldsymbol{x_0}$. The lower branch reuses a pretrained MDM (Motion Diffusion Model) to maintain stable coarse-grained text-conditioned generation; the upper branch is a trainable copy of MDM that receives modulation from the fine-grained control signal via conditional feature adaptation. The two branches are connected through zero-initialized linear layers.

### Key Designs

**1. Text-Based Fine-Grained Control Signal Injection**

Rather than directly concatenating fine-grained and coarse-grained text into a single input (the "Direct" approach), FineXtrol introduces the control signal as residual guidance to modulate motion features. Specifically, the upper branch first constructs a text–motion embedding $\boldsymbol{e}'$ identical to that of the lower branch; the control signal $\boldsymbol{c}$ is then encoded by the text encoder to produce embedding $\boldsymbol{e}_c$, which is aligned via a linear layer and added to $\boldsymbol{e}'$:

$$\boldsymbol{h}_0^{\text{ctrl}} = \boldsymbol{e}' + \text{Linear}(\boldsymbol{e}_c)$$

The interaction between the two branches at layer $l$ is implemented through a zero-initialized linear layer $\mathcal{P}_l$:

$$\boldsymbol{h}_l^{\text{out}} = \boldsymbol{h}_l^{\text{ori}} + \mathcal{P}_l(\boldsymbol{h}_l^{\text{ctrl}})$$

During training, random masking is applied to the control signal (replacing random temporal intervals with `<Mask>`) to improve the model's robustness to partial control.

**2. Hierarchical Contrastive Learning for Text Encoder Enhancement**

Pretrained text encoders such as CLIP and T5 lack discriminability for fine-grained motion descriptions. The paper analyzes the three-level structure of control signals (sentence-level → snippet-level → sequence-level) and designs a hierarchical contrastive learning module built on T5, trained progressively:

- **Sentence-level**: A corpus of body-part motion sentences is constructed; DeepSeek-V2 is used to rewrite sentences and generate positive pairs, trained with InfoNCE loss.
- **Snippet-level**: Sentences within a single temporal interval are randomly replaced or shuffled to generate positive pairs.
- **Sequence-level**: Snippet-level augmentation is applied to each temporal interval while preserving the global temporal order.

Each level is initialized from the weights learned at the previous level. The contrastive loss is:

$$\mathcal{L}_i = -\log \frac{\exp(\text{sim}(\boldsymbol{z}_i, \boldsymbol{z}_j)/\tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(\boldsymbol{z}_i, \boldsymbol{z}_k)/\tau)}$$

**3. Efficient Inference Design**

Since text rather than coordinates serves as the control signal, FineXtrol requires no coordinate transformation between pose representations, resulting in faster inference and fewer trainable parameters.

### Loss & Training

The training objective inherits the simple reconstruction loss from MDM: $\mathcal{L}_\theta = \|\epsilon_\theta(\boldsymbol{x}_t, t, \boldsymbol{p}; \theta) - \boldsymbol{\hat{x}_0}\|_2^2$. The text encoder is first trained via the three-stage progressive contrastive learning procedure, after which the encoder is frozen and the FineXtrol framework is trained. All experiments are conducted on a single A100 40G GPU.

## Key Experimental Results

### Main Results

Comparison with existing controllable motion generation methods on the HumanML3D test set:

| Method | Control Signal | User-Friendly | FID ↓ | R-Top3 ↑ | Diversity | MM-Dist ↓ |
|--------|---------------|--------------|-------|----------|-----------|-----------|
| Real | - | - | 0.002 | 0.796 | 9.503 | 2.965 |
| MDM | - | - | 0.544 | 0.611 | 9.559 | 5.432 |
| OmniControl | Coordinates | ✗ | 0.255 | 0.680 | 9.735 | 5.054 |
| InterControl | Coordinates | ✗ | 0.209 | 0.684 | 9.301 | 5.164 |
| CoMo | Text | ✓ | 0.347 | 0.625 | 9.568 | 5.588 |
| **FineXtrol** | **Text** | **✓** | **0.245** | **0.685** | **9.492** | **5.087** |

Multi-body-part cross control (more challenging setting):

| Method | FID ↓ | R-Top3 ↑ | MM-Dist ↓ |
|--------|-------|----------|-----------|
| OmniControl | 0.624 | 0.601 | 5.252 |
| CoMo | 0.606 | 0.611 | 5.638 |
| **FineXtrol** | **0.351** | **0.676** | **5.146** |

Inference efficiency comparison:

| Method | Inference Time (s) ↓ | Trainable Params |
|--------|---------------------|-----------------|
| OmniControl | 168.51 | 48.79M |
| InterControl | 159.72 | 42.00M |
| GMD | 153.25 | 238.63M |
| **FineXtrol** | **128.57** | **23.39M** |

### Ablation Study

| Ablation | FID ↓ | R-Top3 ↑ |
|----------|-------|----------|
| Direct concatenation paradigm | 1.383 | 0.601 |
| **Ours (residual guidance)** | **0.245** | **0.685** |

| Text Encoder | FID ↓ | R-Top3 ↑ | MM-Dist ↓ |
|-------------|-------|----------|-----------|
| CLIP | 0.579 | 0.603 | 5.927 |
| T5 | 0.374 | 0.659 | 5.483 |
| **Ours (hierarchical contrastive)** | **0.245** | **0.685** | **5.087** |

### Key Findings

- FineXtrol exhibits only marginal performance degradation in multi-body-part cross control, whereas OmniControl and CoMo degrade substantially.
- In a user study with 33 participants, 78.79% (without control signals) and 74.24% (with control signals) preferred FineXtrol.
- The Direct text concatenation paradigm performs far worse than the residual guidance paradigm, demonstrating that a single branch struggles to process dense information.

## Highlights & Insights

- The idea of replacing coordinate sequences with fine-grained text as control signals is novel: it preserves precise controllability while substantially reducing computational cost and lowering the barrier for end users.
- The hierarchical contrastive learning module designs data augmentation strategies tailored to the three-level structure of control signals, effectively addressing the insufficient fine-grained semantic discriminability of pretrained encoders.
- Zero-initialized connections ensure no noise is injected during early training, enabling progressive learning of control signal semantics.

## Limitations & Future Work

- The approach relies on the FineMotion dataset for ground-truth-aligned fine-grained annotations; its generalizability to open-domain text control remains to be validated.
- Evaluation is conducted solely on HumanML3D, without assessment on additional motion datasets.
- The precision of control signals depends on the quality of textual descriptions, and manually authoring fine-grained descriptions still imposes a non-trivial burden on users.

## Related Work & Insights

- **vs. OmniControl / InterControl**: These methods use 3D coordinate sequences as control signals—precise but requiring coordinate transformation and unfriendly to users. FineXtrol replaces coordinates with text, achieving inference speeds more than 30 seconds faster and roughly half the parameter count of OmniControl.
- **vs. CoMo**: CoMo uses LLM-expanded text but lacks temporal annotations and alignment with ground-truth motions. FineXtrol uses aligned descriptions from FineMotion and surpasses CoMo by 0.060 on R-Top3.

## Rating

- Novelty: ⭐⭐⭐⭐ — The paradigm of replacing coordinate sequences with fine-grained text as control signals is novel; the hierarchical contrastive learning design is well motivated.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Quantitative comparisons, ablations, user studies, and visualizations are all covered with systematic experimental design.
- Writing Quality: ⭐⭐⭐⭐ — The paper is clearly structured with well-articulated motivation and rich figures and tables.
- Value: ⭐⭐⭐⭐ — Provides a practical and efficient new paradigm for controllable motion generation with clear deployment potential.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning](../../ICCV2025/self_supervised/mosic_optimal-transport_motion_trajectory_for_dense_self-supervised_learning.md)
- [\[CVPR 2026\] Text-Phase Synergy Network with Dual Priors for Unsupervised Cross-Domain Image Retrieval](../../CVPR2026/self_supervised/text-phase_synergy_network_with_dual_priors_for_unsupervised_cross-domain_image_.md)
- [\[NeurIPS 2025\] TabSTAR: A Tabular Foundation Model for Tabular Data with Text Fields](../../NeurIPS2025/self_supervised/tabstar_a_tabular_foundation_model_for_tabular_data_with_text_fields.md)
- [\[ICCV 2025\] A Token-level Text Image Foundation Model for Document Understanding (TokenFD/TokenVL)](../../ICCV2025/self_supervised/a_tokenlevel_text_image_foundation_model_for_document_unders.md)
- [\[AAAI 2026\] MovSemCL: Movement-Semantics Contrastive Learning for Trajectory Similarity (Extension)](movsemcl_movement-semantics_contrastive_learning_for_trajectory_similarity_exten.md)

</div>

<!-- RELATED:END -->
