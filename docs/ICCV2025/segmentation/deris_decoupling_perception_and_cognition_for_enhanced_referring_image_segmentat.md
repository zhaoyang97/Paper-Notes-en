---
title: >-
  [Paper Note] DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy
description: >-
  [ICCV 2025][Segmentation][referring image segmentation] This paper proposes DeRIS, a framework that decouples referring image segmentation into two branches — perception and cognition — and introduces a Loopback Synergy mechanism to iteratively enhance cross-branch interaction. A non-referent sample conversion augmentation strategy is also introduced. DeRIS achieves state-of-the-art performance on RefCOCO/+/g and gRefCOCO benchmarks.
tags:
  - "ICCV 2025"
  - "Segmentation"
  - "referring image segmentation"
  - "perception-cognition decoupling"
  - "loopback synergy"
  - "non-referent"
  - "GRES"
date: 2026-05-08
content_hash: 95aa6cd3a0309584
---

# DeRIS: Decoupling Perception and Cognition for Enhanced Referring Image Segmentation through Loopback Synergy

**Conference**: ICCV 2025
**arXiv**: [2507.01738](https://arxiv.org/abs/2507.01738)  
**Code**: Available (link mentioned in paper)  
**Area**: Image Segmentation
**Keywords**: referring image segmentation, perception-cognition decoupling, loopback synergy, non-referent, GRES

## TL;DR
This paper proposes DeRIS, a framework that decouples referring image segmentation into two branches — perception and cognition — and introduces a Loopback Synergy mechanism to iteratively enhance cross-branch interaction. A non-referent sample conversion augmentation strategy is also introduced. DeRIS achieves state-of-the-art performance on RefCOCO/+/g and gRefCOCO benchmarks.

## Background & Motivation
Referring image segmentation (RIS) requires segmenting targets in an image according to natural language expressions, demanding both fine-grained perceptual capability and multimodal cognitive understanding. Existing methods fall into two categories: perception-centric approaches (e.g., Mask2Former-based) preserve fine spatial information but lack strong multimodal understanding; cognition-centric approaches (e.g., BEiT3/CLIP-based) excel at multimodal comprehension but lose fine spatial detail due to the quadratic complexity of Transformers. This raises a core question: what is the primary bottleneck in RIS — perception or cognition? Through quantitative analysis, the authors find that enhancing perception yields only +1.2% cIoU improvement, whereas enhancing cognition yields +12.9% cIoU, establishing that **cognitive capability is the primary bottleneck**. Accordingly, this paper proposes to decouple RIS into two independent branches, each leveraging its respective strength, connected via the Loopback Synergy mechanism.

## Method

### Overall Architecture
DeRIS consists of three core components: (1) **Perception Branch**: employs Swin Transformer + FPN to extract multi-scale features and generate high-precision instance-level masks; (2) **Cognition Branch**: employs the BEiT3 vision-language pretrained model to process low-resolution images and text, providing multimodal semantic understanding; (3) **Loopback Synergy**: iteratively passes object queries over multiple rounds to facilitate progressive information exchange between the two branches.

### Key Designs
1. **Loopback Synergy Mechanism**:

    - Function: Establishes strong interaction between the perception and cognition branches; each round comprises one cognition layer and one perception layer.
    - Mechanism: The perception layer generates object queries $Q_p$ and masks $M_p$, which are passed to the cognition layer. The cognition layer allows $Q_p$ to interact with visual-linguistic semantic information, producing cognitive queries $Q_c$ and referring confidence scores $S_r$. A C1 operation then fuses them as $Q_f = \text{MLP}(\text{Concat}(Q_p, Q_r))$, serving as input to the next round. The default number of iterations is $N_r = 3$, with supervision applied at each round.
    - Design Motivation: Unidirectional transfer (e.g., C-to-P) causes slow convergence in the perception branch, whereas the P-to-C direction flows more naturally. The loopback design allows both branches to continuously reinforce each other, enabling progressive understanding.

2. **Perception Layer and Cognition Layer Design**:

    - Function: The perception layer is responsible for fine-grained mask generation; the cognition layer handles referring classification.
    - Mechanism: The perception layer, similar to Mask2Former, processes queries via deformable cross-attention and self-attention, and integrates perception and cognition features through a fused feature map $f_m = \text{Conv}(\text{Concat}(f_{h4}, f_v))$. The cognition layer models Instance-Instance relationships (self-attention with mask priors) and Instance-Text relationships (cross-attention with text features) to produce semantically aligned $Q_c$.
    - Design Motivation: Incorporating cognitive features $f_v$ into mask prediction generates text-informed candidate regions; inter-instance relationship modeling in the cognition layer helps each object query attend to spatial context.

3. **Non-referent Sample Conversion (NSC) Augmentation**:

    - Function: Addresses the long-tail distribution problem caused by the ~9% proportion of non-referent samples in gRefCOCO.
    - Mechanism: Image-text pairs containing targets are dynamically converted into non-referent samples by replacing the textual description. A three-level filtering process ensures no false non-referents are generated: (1) the image corresponding to the selected sentence must differ from the current image; (2) the sentence length must exceed a threshold $N_w = 2$; (3) the sentence similarity must be below a threshold $T_s = 0.6$, where similarity is the mean of Jaccard and cosine similarity.
    - Design Motivation: The scarcity of non-referent samples causes the model to over-predict target existence, resulting in poor performance on the N-acc metric.

### Loss & Training
The total loss consists of three components: segmentation loss $\mathcal{L}_{mask}$ (BCE + Dice), referring classification loss $\mathcal{L}_r$ (BCE), and non-referent judgment loss $\mathcal{L}_{nr}$ (BCE). The loss per Loopback round is $\mathcal{L}^i = \lambda_m \mathcal{L}_{mask}^i + \lambda_r \mathcal{L}_r^i + \lambda_{nt} \mathcal{L}_{nt}^i$, with all weights set to 1.0. The auxiliary loss weight is $\lambda_{aux} = 0.2$. During inference, a threshold $\mathcal{T}_{ref} = 0.7$ is applied to filter referring classifications.

## Key Experimental Results

### Main Results

| Method | RefCOCO val | RefCOCO+ val | RefCOCOg val(U) | Notes |
|--------|------------|-------------|----------------|-------|
| PolyFormer-L | 76.94 | 72.15 | 71.15 | Perception-centric |
| C3VG | 81.37 | 77.05 | 76.34 | Cognition-centric |
| OneRef-L | 81.26 | 76.60 | 75.68 | Cognition-centric |
| **DeRIS-B** | 81.99 | 75.62 | 76.30 | Swin-S + BEiT3-B |
| **DeRIS-L** | **85.72** | **81.28** | **80.01** | Swin-B + BEiT3-L |

gRefCOCO (GRES) Results:

| Method | Val gIoU | Val cIoU | Val N-acc | TestA gIoU | TestB gIoU |
|--------|----------|----------|-----------|------------|------------|
| SAM4MLLM-8B | 71.86 | 67.83 | 66.08 | 74.15 | 65.29 |
| **DeRIS-B** | 74.10 | 68.06 | **77.03** | 73.72 | 65.63 |
| **DeRIS-L** | **77.67** | **72.00** | **82.22** | **75.30** | **67.99** |

### Ablation Study

| Configuration | gIoU | cIoU | Notes |
|--------------|------|------|-------|
| Query: P-to-C (baseline) | 69.98 | 65.49 | Perception → Cognition |
| Query: C-to-P | 56.77 | 54.80 | Cognition → Perception (slow convergence) |
| Hierarchical Combined | 70.13 | 66.32 | Feature-level fusion (18% slower training) |
| **Loopback Synergy** | **71.37** | **67.27** | Loopback synergy (only 3% slower training) |

Cognition vs. Perception Bottleneck Analysis:

| Cognition Model | Perception Model | cIoU Change |
|----------------|-----------------|-------------|
| BERT-B → BEiT3-B | Swin-S | +10.05 (large gain from enhanced cognition) |
| BEiT3-B | Swin-T → Swin-B | +1.20 (limited gain from enhanced perception) |

NSC Augmentation Effect:

| Configuration | N-acc | gIoU | cIoU |
|--------------|-------|------|------|
| w/o NSC | 60.19 | 66.09 | 63.98 |
| w/ NSC (Rc=15%) | 75.36 (+15.17) | 71.82 (+5.73) | 66.33 (+2.35) |

### Key Findings
- Cognitive capability is the primary bottleneck in RIS (+12.9% vs. +1.2%), not perceptual capability.
- Qualitative analysis shows that object queries can produce accurate masks, but referring classification frequently fails.
- The NSC strategy improves N-acc by over 15 percentage points, effectively stabilizing non-referent judgment during training.
- DeRIS naturally accommodates non-referent and multi-referent scenarios without requiring special architectural modifications.

## Highlights & Insights
- This work is the first to systematically quantify the respective contributions of perception and cognition in RIS, identifying cognition as the primary bottleneck — a finding with significant implications for the field.
- The Loopback Synergy design is elegant and efficient: strong cross-branch interaction is established solely through iterative object query passing, with negligible additional training overhead.
- The NSC augmentation strategy is simple yet effective, with a three-level filtering process ensuring conversion quality.
- The framework demonstrates strong extensibility: the cognition branch can be replaced with more powerful models such as Qwen2-7B.

## Limitations & Future Work
- The perception branch operates at 384×384 resolution; higher resolutions may further improve fine-grained segmentation.
- The cognition branch uses low-resolution 224×224 inputs, which may discard some spatial information.
- The NSC conversion ratio $R_c$ requires manual tuning; an adaptive strategy may be preferable.
- Extension of the method to referring video segmentation remains unexplored.

## Related Work & Insights
- Mask2Former provides strong perceptual priors, while BEiT3 provides strong cognitive priors; the decoupled design effectively leverages both.
- The Loopback Synergy concept can be extended to other multimodal tasks requiring perception-understanding co-adaptation.
- The long-tail distribution problem of non-referent samples is prevalent in many multimodal grounding tasks, and the NSC strategy is broadly applicable.

## Rating
- Novelty: ⭐⭐⭐⭐ The perception-cognition decoupling framework is conceptually clear and well-motivated; the Loopback Synergy design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive validation across four datasets with rich ablation studies; the bottleneck analysis is highly persuasive.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with a problem-driven narrative that is engaging and easy to follow.
- Value: ⭐⭐⭐⭐⭐ The cognitive bottleneck finding provides meaningful guidance for the RIS community; the method achieves SOTA with strong generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Latent Expression Generation for Referring Image Segmentation and Grounding](latent_expression_generation_for_referring_image_segmentation_and_grounding.md)
- [\[NeurIPS 2025\] SaFiRe: Saccade-Fixation Reiteration with Mamba for Referring Image Segmentation](../../NeurIPS2025/segmentation/safire_saccade-fixation_reiteration_with_mamba_for_referring_image_segmentation.md)
- [\[ICCV 2025\] TinyViM: Frequency Decoupling for Tiny Hybrid Vision Mamba](tinyvim_frequency_decoupling_for_tiny_hybrid_vision_mamba.md)
- [\[ECCV 2024\] ReMamber: Referring Image Segmentation with Mamba Twister](../../ECCV2024/segmentation/remamber_referring_image_segmentation_with_mamba_twister.md)
- [\[ICCV 2025\] Enhancing Transformers Through Conditioned Embedded Tokens](enhancing_transformers_through_conditioned_embedded_tokens.md)

</div>

<!-- RELATED:END -->
