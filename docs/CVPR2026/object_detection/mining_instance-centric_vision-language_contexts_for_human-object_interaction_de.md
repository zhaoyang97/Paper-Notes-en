---
title: >-
  [Paper Note] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection
description: >-
  [CVPR 2026][Object Detection][Human-object interaction detection] This paper proposes InCoM-Net, which extracts intra-instance, inter-instance, and global context features separately for each instance from VLM features, and achieves state-of-the-art HOI detection on HICO-DET and V-COCO (HICO-DET Full mAP 43.96, V-COCO AP_role^S1 73.6) via progressive context aggregation and fusion with detector features.
tags:
  - CVPR 2026
  - Object Detection
  - Human-object interaction detection
  - vision-language model
  - instance-centric context
  - multi-context features
  - attention mechanism
date: 2026-05-08
content_hash: 9164050a4b52f3cf
---

# Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection

**Conference**: CVPR 2026
**arXiv**: [2604.02071](https://arxiv.org/abs/2604.02071)
**Code**: [https://github.com/nowuss/InCoM-Net](https://github.com/nowuss/InCoM-Net)
**Area**: Object Detection / Human-Object Interaction Detection
**Keywords**: Human-object interaction detection, vision-language model, instance-centric context, multi-context features, attention mechanism

## TL;DR

This paper proposes InCoM-Net, which extracts intra-instance, inter-instance, and global context features separately for each instance from VLM features, and achieves state-of-the-art HOI detection on HICO-DET and V-COCO (HICO-DET Full mAP 43.96, V-COCO AP_role^S1 73.6) via progressive context aggregation and fusion with detector features.

## Background & Motivation

1. **Background**: HOI detection aims to localize human-object pairs in images and classify their interaction relationships, serving as a fundamental task in visual understanding. Recent Transformer- and VLM-based methods (e.g., CLIP, BLIP) have significantly advanced performance.
2. **Limitations of Prior Work**: Existing VLM integration methods either use scene-level VLM features solely as global semantic priors (e.g., HOICLIP, UniHOI), or restrict VLM features to object bounding boxes via RoI alignment (e.g., ADA-CM, BCOM), failing to fully exploit contextual cues distributed across different levels of the scene.
3. **Key Challenge**: HOI reasoning requires simultaneous understanding of an instance's own visual cues, its relationships with surrounding instances, and the global scene context. However, existing methods apply context information uniformly to all instances, lacking instance-specific context modeling.
4. **Goal**: To extract multi-level contextual information from VLM features for each instance individually and effectively fuse it into the detector's instance features.
5. **Key Insight**: The authors observe that human judgment of HOI relies on three types of cues—the visual features within the target instance, its relationships with other instances, and surrounding scene information—and accordingly design an instance-centric multi-context mining scheme.
6. **Core Idea**: Extract three types of context (intra-instance, inter-instance, and global) from VLM features via masked self-attention, then progressively fuse them into detector queries.

## Method

### Overall Architecture

InCoM-Net adopts a dual-branch architecture: (1) a DETR detector branch that extracts instance-level features $q^l$; and (2) a CLIP visual encoder that extracts VLM features $V^l$. The core module is Instance-centric Context Mining, comprising two sub-modules—ICR (Instance-Centric Context Refinement) and ProCA (Progressive Context Aggregation)—processed iteratively across $L$ layers. Finally, an HO Pair Generator constructs human-object pair features, which are fed into an interaction decoder for HOI classification.

### Key Designs

1. **Instance-centric Context Refinement (ICR)**:

    - **Function**: Generates three types of context features from VLM features for each instance individually.
    - **Mechanism**: Masked self-attention is applied to VLM features $V^l$. For the $i$-th instance, an instance mask $M_i^R$ (marking the instance region) and a surrounding mask $M_i^C$ (the union of other instances) are constructed. Unmasked self-attention produces global context $G^l$; self-attention restricted by $M_i^R$ produces intra-instance context $R_i^l$; and self-attention restricted by $M_i^C$ produces inter-instance context $C_i^l$. Each output is encoded through a separate FFN.
    - **Design Motivation**: Different levels of context provide complementary information—intra-instance features capture target appearance, inter-instance features model object relationships, and global features provide scene context. Separate encoding preserves semantic diversity.

2. **Progressive Context Aggregation (ProCA)**:

    - **Function**: Progressively fuses multi-context features produced by ICR into detector query features.
    - **Mechanism**: The detector query $q_i^l$ is summed with the aggregated feature from the previous layer $f_i^{l-1}$ to form the query, which then performs cross-attention separately over $G^l$, $R_i^l$, and $C_i^l$. The three resulting outputs are concatenated and passed through an FFN to produce the current-layer aggregated feature $f_i^l$. This process iterates across $L$ layers, with each layer consuming features from a different VLM layer.
    - **Design Motivation**: Progressive multi-layer aggregation enables the model to gradually integrate VLM information at different semantic levels, enhancing alignment between instance appearance and context.

3. **Masked Feature Training (MFT)**:

    - **Function**: Balances the utilization of the two heterogeneous feature sources—VLM and detector.
    - **Mechanism**: During training, three input configurations are constructed with equal probability—full input (VLM + detector), detector-only, and VLM-only. Features from the masked branch are set to zero and the corresponding cross-attention is disabled. The total loss is the sum of focal losses from all three configurations.
    - **Design Motivation**: Two heterogeneous feature sources can cause the model to over-rely on a single source. Random masking forces the model to learn to exploit complementary information under varying conditions, improving robustness.

### Loss & Training

- Interaction classification uses focal loss.
- Each of the three masked configurations (full / detector-only / VLM-only) produces its own focal loss; the total loss is their sum.
- Both DETR and CLIP are frozen; only ICR, ProCA, the HO Pair Generator, and the interaction decoder are trained.
- AdamW optimizer is used with an initial learning rate of $10^{-4}$, decayed by a factor of 5 every 10 epochs, with training completed in 30 epochs.

## Key Experimental Results

### Main Results

| Dataset | Metric | InCoM-Net (ViT-L) | Prev. SOTA (NMSR) | Gain |
|--------|------|------|----------|------|
| HICO-DET | Full mAP | **43.96** | 42.93 | +1.03 |
| HICO-DET | Rare mAP | **45.61** | 42.41 | +3.20 |
| HICO-DET | Non-rare mAP | **43.46** | 43.11 | +0.35 |
| V-COCO | AP_role^S1 | **73.6** | 69.8 | +3.8 |
| V-COCO | AP_role^S2 | **75.4** | 72.1 | +3.3 |

ViT-B version: HICO-DET Full 39.53 (surpassing HORP by +0.92), V-COCO S1 72.2 (surpassing SCTC by +5.1).

### Ablation Study

| Configuration | Full mAP | Rare mAP | Note |
|------|---------|----------|------|
| Baseline (no ICR/ProCA) | 36.17 | 33.11 | Detector features only |
| + ICR | 37.42 | 34.47 | +1.25, multi-context is effective |
| + ProCA | 38.42 | 36.80 | +1.00, progressive aggregation is effective |
| + MFT | **39.53** | **38.87** | +1.11, balances heterogeneous features |

Ablation on context types (with ICR + ProCA):

| Context Configuration | Full mAP | Rare mAP |
|-----------|---------|----------|
| V only (raw VLM) | 38.30 | 37.31 |
| + G (global) | 38.65 | 36.76 |
| + R (intra-instance) | 39.19 | 38.78 |
| + C (inter-instance) | **39.53** | **38.87** |

### Key Findings

- MFT contributes the largest gain (+1.11 mAP), especially on Rare categories (+2.07), indicating that balancing heterogeneous features is critical for low-frequency interactions.
- Intra-instance context $R$ contributes most significantly to Rare categories (+2.02), suggesting that fine-grained instance information is particularly important for rare interaction inference.
- InCoM-Net also achieves state-of-the-art results in zero-shot settings; Unseen mAP under RF-UC and NF-UC reaches 37.69/39.45 (ViT-L), demonstrating strong generalization.
- ProCA performs best at 3 layers; gains from additional layers tend to saturate.

## Highlights & Insights

- **Instance-level multi-context decomposition**: The masked mechanism adaptively extracts three types of context from shared VLM features in an elegant and effective manner. Separating context by semantic role before fusion captures finer-grained relationships than directly using global VLM features.
- **MFT strategy**: The training strategy of randomly masking heterogeneous feature sources is a creative contribution—applying a dropout-like idea to multi-modal feature fusion to effectively prevent over-reliance on any single source.
- **Transfer potential**: This instance-centric multi-context mining paradigm is transferable to tasks requiring instance-level relationship modeling, such as scene graph generation and relational reasoning.

## Limitations & Future Work

- Both DETR and CLIP are frozen, limiting the potential of end-to-end optimization; partial fine-tuning of the VLM encoder could be explored.
- The quality of context masks depends on the detector's detection quality; missed or erroneous detections would degrade context accuracy.
- Only static image context is considered; temporal action cues (e.g., video HOI) are not exploited.
- The three masked configurations are sampled with equal probability; adaptive sampling strategies could be explored.

## Related Work & Insights

- **vs. BCOM (CVPR24)**: BCOM employs a dual branch to encode RoI features and VLM features separately but lacks inter-instance context modeling. InCoM-Net unifies multi-level context extraction via ICR, outperforming BCOM by +4.62 mAP (ViT-L).
- **vs. ADA-CM (ICCV23)**: ADA-CM injects detection signals via adapters and performs RoI pooling, but applies uniform context to all instances. Instance-specific context modeling is the key differentiator in InCoM-Net.
- **vs. NMSR (ICCV25)**: The previous state of the art; InCoM-Net surpasses it by +1.03 on HICO-DET and +3.8 on V-COCO, with advantages primarily attributable to multi-context refinement and progressive aggregation.

## Rating

- Novelty: ⭐⭐⭐⭐ The instance-level multi-context decomposition idea is novel, and the MFT strategy is creative; however, the overall framework remains a standard DETR + CLIP dual-branch design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two datasets, regular and zero-shot settings, detailed ablations, and visualizations—very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, intuitive figures, and coherent motivation derivation.
- Value: ⭐⭐⭐⭐ State-of-the-art HOI detection with a method design transferable to other relational reasoning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward](saliency-r1_enforcing_interpretable_and_faithful_vision-language_reasoning_via_s.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](../../ICCV2025/object_detection/visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[NeurIPS 2025\] DetectiumFire: A Comprehensive Multi-modal Dataset Bridging Vision and Language for Fire Understanding](../../NeurIPS2025/object_detection/detectiumfire_a_comprehensive_multi-modal_dataset_bridging_vision_and_language_f.md)
- [\[CVPR 2026\] PHAC: Promptable Human Amodal Completion](phac_promptable_human_amodal_completion.md)
- [\[CVPR 2026\] Show, Don't Tell: Detecting Novel Objects by Watching Human Videos](show_dont_tell_detecting_novel_objects_by_watching.md)

</div>

<!-- RELATED:END -->
