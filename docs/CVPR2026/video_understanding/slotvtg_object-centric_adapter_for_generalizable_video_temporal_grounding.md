---
title: >-
  [Paper Note] SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding
description: >-
  [CVPR 2026][Video Understanding][Video Temporal Grounding] This paper proposes SlotVTG, a framework that inserts a lightweight Slot Adapter into the early layers of an MLLM decoder to decompose visual tokens into object-…
tags:
  - "CVPR 2026"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "Object-Centric Learning"
  - "Slot Attention"
  - "Out-of-Domain Generalization"
  - "Multimodal Large Language Models"
date: 2026-05-08
content_hash: 865bf04f6ec59cde
---

# SlotVTG: Object-Centric Adapter for Generalizable Video Temporal Grounding

**Conference**: CVPR 2026
**arXiv**: [2603.25733](https://arxiv.org/abs/2603.25733)  
**Code**: None  
**Area**: Video Understanding / Temporal Grounding
**Keywords**: Video Temporal Grounding, Object-Centric Learning, Slot Attention, Out-of-Domain Generalization, Multimodal Large Language Models

## TL;DR

This paper proposes SlotVTG, a framework that inserts a lightweight Slot Adapter into the early layers of an MLLM decoder to decompose visual tokens into object-level slot representations. A Slot Alignment Loss guided by DINOv2 priors encourages semantically coherent slot formation, substantially improving out-of-domain (OOD) generalization for video temporal grounding (up to +4.3 OOD R1@0.5), while introducing only ~0.25% additional trainable parameters.

## Background & Motivation

1. **Background**: Multimodal large language models (MLLMs) have become the dominant paradigm for video temporal grounding (VTG), yet fine-tuning on specific datasets is required to acquire fine-grained temporal understanding.
2. **Limitations of Prior Work**: VTG annotation demands precise start/end timestamps, making large-scale data collection extremely costly and limiting training data availability. Fine-tuning on limited data causes models to memorize dataset-specific shortcuts (temporal position bias, query text bias, appearance bias, etc.), leading to severe performance degradation on OOD test sets.
3. **Key Challenge**: Models perform well in-domain (ID) but degrade substantially OOD—after training on Charades-STA, ID R1@0.5 reaches 63.4 but OOD drops to 43.6 (−31.2%). Noise perturbation experiments confirm that models no longer attend to target visual content under OOD conditions.
4. **Goal**: Enable fine-tuned MLLMs to perform grounding based on actual visual content rather than domain-specific patterns, thereby improving OOD generalization.
5. **Key Insight**: Object-centric learning decomposes scenes into entity-level representations that inherently extract domain-invariant visual features. Measurements show that the inter-domain MMD distance of slot representations is reduced by 49.6% compared to the baseline.
6. **Core Idea**: A Slot Attention bottleneck forces visual information to pass through object-level decomposition before entering the LLM, suppressing domain-specific associations.

## Method

### Overall Architecture

Video frames are processed by a frozen visual encoder to extract tokens, which are projected into the LLM decoder space. A Slot Adapter is inserted into early decoder layers: visual tokens are first projected to a lower dimension, then iteratively decomposed into a small number of abstract slots via slot attention, and finally reconstructed back into the original token sequence via cross-attention with a residual connection. The reconstructed tokens pass into deeper layers (fine-tuned with LoRA) for temporal reasoning and answer generation. Text tokens bypass the Slot Adapter throughout.

### Key Designs

1. **Slot Adapter**:

    - **Function**: Decomposes dense visual tokens into a small number ($N_s=4$) of abstract slots and reconstructs the original sequence.
    - **Mechanism**: Visual tokens are first projected to a lower dimension ($W_{down}$: $D \to d$, $d=512$). Then $N_s$ learnable slot queries compete to bind with tokens over $I=3$ iterative rounds of slot attention—softmax along the slot axis implements winner-take-all assignment, followed by normalization along the token axis and weighted aggregation to update slots via GRU recurrence. In the reconstruction stage, cross-attention (original tokens as queries, slots as keys/values) is applied, followed by $W_{up}$ to restore dimensionality $D$. A zero-initialized projection with residual connection ensures an identity mapping at initialization.
    - **Design Motivation**: The competitive mechanism in slot attention forces each slot to focus on a single semantic entity (person, object, background), producing entity-level representations that are more domain-invariant than raw per-patch tokens. The bottleneck structure naturally filters domain-specific noise.

2. **Early-Layer Insertion Strategy**:

    - **Function**: Slot Adapter is inserted into decoder layers 1–7; deeper layers are fine-tuned with LoRA.
    - **Mechanism**: Research shows that cross-frame interaction occurs in early layers, while deeper layers handle language integration and answer generation. Insertion in early layers allows each slot to capture temporally consistent semantics across frames, rather than decomposing each frame independently. LoRA in deeper layers performs temporal reasoning over the already-decomposed representations.
    - **Design Motivation**: Inserting at deeper layers would place slot decomposition after features have already been substantially fused with language information, making it difficult to isolate visual domain-specific patterns.

3. **Slot Alignment Loss**:

    - **Function**: Guides slot attention maps to form semantically coherent groupings.
    - **Mechanism**: A token-pair similarity matrix $M_{slot} = 2(\bar{A}\bar{A}^T) - 1$ is computed from slot attention weights $A$. Simultaneously, DINOv2 features are extracted from a frozen model to compute $M_{dino} = \bar{F}_{dino}\bar{F}_{dino}^T$. The two matrices are aligned via cosine similarity: $\mathcal{L}_{SA} = 1 - \frac{1}{T}\sum_t \cos(M_{slot}^{(t)}, M_{dino}^{(t)})$. This leverages the objectness prior learned by DINOv2 through self-supervision to guide semantically meaningful slot formation.
    - **Design Motivation**: Without this supervision, the bottleneck alone may produce arbitrary clusterings. DINOv2 features naturally reflect object/background boundaries and serve as a teacher signal for meaningful decomposition.

### Loss & Training

$\mathcal{L}_{total} = \mathcal{L}_{CE} + \lambda \mathcal{L}_{SA}$, with $\lambda=0.1$.

The visual encoder is frozen; the Slot Adapter and LoRA are jointly optimized. The 3B model has ~7.6M trainable parameters (0.25%), and the 7B model ~23.3M (0.33%). AdamW optimizer, learning rate $5 \times 10^{-5}$, 5 training epochs, batch size 32, 8×3090/4090 GPUs.

## Key Experimental Results

### Main Results

Charades-STA → Others (3B backbone, R1@0.5):

| Method | Cha. (ID) | ANet (OOD) | QVH (OOD) |
|--------|-----------|------------|-----------|
| Chrono-Qwen | 63.4 | 26.3 | 43.6 |
| SlotVTG | 64.0 | **28.7** | **47.9** |
| Δ | +0.6 | **+2.4** | **+4.3** |

QVHighlights → Others (3B backbone, R1@0.5):

| Method | QVH (ID) | Cha. (OOD) | ANet (OOD) |
|--------|----------|------------|------------|
| Chrono-Qwen | 79.1 | 45.7 | 35.3 |
| SlotVTG | 79.5 | **46.6** | **35.7** |

OOD gains are larger with the 7B model: Cha.→ANet +4.0, Cha.→QVH +4.1 R1@0.5.

### Ablation Study

| Component | Cha. (ID) R1@0.5 | ANet (OOD) R1@0.5 |
|-----------|------------------|-------------------|
| LoRA only | 63.4 | 26.3 |
| Self-attention adapter | 63.5 | 26.5 |
| Slot Adapter | 64.0 | **28.7** |
| Slot Adapter w/o $\mathcal{L}_{SA}$ | 63.3 | 28.0 |
| Slot Adapter + $\mathcal{L}_{SA}$ ($\lambda$=0.1) | 64.0 | **28.7** |
| Slot Adapter + $\mathcal{L}_{SA}$ ($\lambda$=0.2) | 64.3 | 26.1 |

Insertion layer range:

| Layer Range | ANet (OOD) R1@0.5 |
|-------------|-------------------|
| 1–7 (early) | **28.7** |
| 10–17 (middle) | 27.5 |
| 20–36 (deep) | 28.4 |

### Key Findings

- **Slot Adapter vs. standard self-attention adapter**: Substantial OOD improvement (28.7 vs. 26.5), confirming that the entity decomposition mechanism of slot attention—rather than simple bottlenecking—is the driving factor.
- **Sensitivity of $\lambda$ in SA Loss**: $\lambda=0.1$ is optimal; $\lambda=0.2$ causes OOD performance to drop (26.1), indicating that an overly strong objectness prior restricts model flexibility.
- **Early-layer insertion is optimal**: Layers 1–7 > 10–17 > 20–36, consistent with the hypothesis that early layers handle cross-frame visual interaction.
- **Cross-attention reconstruction outperforms simple copy + projection**: OOD R1@0.7 reaches 14.9 vs. 13.7.
- Slot visualizations show decomposition into semantically meaningful regions (person, object, background) on both ID and OOD data, without any target-domain supervision.
- Inter-domain MMD distance decreases from 0.192 to 0.097 (−49.6%), quantitatively confirming reduced domain discrepancy.

## Highlights & Insights

- **Diagnostic experiments are highly compelling**: Noise perturbation experiments clearly reveal that models under OOD conditions do not attend to visual content—performance degradation from adding noise to GT segments and random segments is nearly identical (12.6% vs. 12.1%), pinpointing the root cause more convincingly than simply reporting OOD performance drops.
- **Significant OOD gains at minimal parameter cost**: A 4+ point OOD improvement is achieved with only 0.25% trainable parameters. The adapter is plug-and-play on top of existing fine-tuned MLLMs without requiring retraining of the vision-language alignment.
- **Slots as domain-invariant bottleneck**: Object-level decomposition is inherently more robust to domain shift than patch-level tokens—an insight generalizable to other vision-language tasks requiring OOD generalization.

## Limitations & Future Work

- Only 4 slots are used, which may be insufficient for complex scenes (multiple people and objects); however, increasing to 8 slots yields slightly worse performance.
- Validation is limited to VTG; whether the Slot Adapter improves OOD generalization in other video tasks (e.g., video QA, video captioning) remains to be explored.
- The SA Loss is applied only at the last adapter layer; joint constraints across multiple layers may be more effective.
- Temporal slot consistency constraints are not explored (e.g., requiring the same slot to track the same entity across adjacent frames).
- OOD gains are smaller when QVHighlights serves as the source dataset, as its domain distribution is already broader.

## Related Work & Insights

- **vs. Slot-VLM**: Uses a dual-branch object-event slot decomposition of video tokens but requires training the entire VL pipeline from scratch. SlotVTG's adapter-based approach is plug-and-play and substantially reduces training cost.
- **vs. Chrono**: Achieves generative VTG via interleaved frame-timestamp representations. SlotVTG adds object-centric decomposition on top of this framework, demonstrating complementary improvement.
- **vs. DETR-based methods (EaTR, CG-DETR)**: These specialized models exhibit even more severe OOD degradation, confirming that MLLMs combined with object-centric adapters represent a more promising paradigm.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Introducing slot attention as an adapter into MLLMs for OOD VTG is a novel and well-motivated perspective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Diagnostic analysis, cross-domain evaluation, detailed ablations, visualizations, and quantitative domain distance measurements.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The logical chain from problem diagnosis to solution design to validation is exceptionally clear.
- **Value**: ⭐⭐⭐⭐ — Meaningfully advances OOD generalization for VTG; the adapter design is transferable to other video tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Reconstruction-Guided Slot Curriculum: Addressing Object Over-Fragmentation in Video Object-Centric Learning](reconstruction-guided_slot_curriculum_addressing_object_over-fragmentation_in_vi.md)
- [\[CVPR 2026\] How Should Video LLMs Output Time? An Analysis of Efficient Temporal Grounding Paradigms](how_should_video_llms_output_time.md)
- [\[CVPR 2026\] CVA: Context-aware Video-text Alignment for Video Temporal Grounding](cva_context-aware_video-text_alignment_for_video_temporal_grounding.md)
- [\[CVPR 2026\] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)
- [\[ICLR 2026\] From Vicious to Virtuous Cycles: Synergistic Representation Learning for Unsupervised Video Object-Centric Learning](../../ICLR2026/video_understanding/from_vicious_to_virtuous_cycles_synergistic_representation_learning_for_unsuperv.md)

</div>

<!-- RELATED:END -->
