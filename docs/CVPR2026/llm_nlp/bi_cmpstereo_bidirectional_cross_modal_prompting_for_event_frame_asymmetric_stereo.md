---
title: >-
  [Paper Note] Bi-CMPStereo: Bidirectional Cross-Modal Prompting for Event-Frame Asymmetric Stereo
description: >-
  [CVPR 2026][LLM/NLP][event camera] Bi-CMPStereo is a bidirectional cross-modal prompting framework that alternately designates event and frame as the target domain for stereo canonicalization and cross-domain embedding a…
tags:
  - "CVPR 2026"
  - "LLM/NLP"
  - "event camera"
  - "stereo matching"
  - "cross-modal"
  - "asymmetric stereo"
  - "depth estimation"
date: 2026-05-08
content_hash: ddccd50cd0f174be
---

# Bi-CMPStereo: Bidirectional Cross-Modal Prompting for Event-Frame Asymmetric Stereo

**Conference**: CVPR 2026  
**arXiv**: [2604.15312](https://arxiv.org/abs/2604.15312)  
**Code**: [github.com/xnh97/Bi-CMPStereo](https://github.com/xnh97/Bi-CMPStereo)  
**Area**: LLM / NLP (Other)  
**Keywords**: event camera, stereo matching, cross-modal, asymmetric stereo, depth estimation

## TL;DR

Bi-CMPStereo is a bidirectional cross-modal prompting framework that alternately designates event and frame as the target domain for stereo canonicalization and cross-domain embedding adaptation, while leveraging cost volumes from both directions to achieve robust event-frame asymmetric stereo matching.

## Background & Motivation

The high temporal resolution and high dynamic range of event cameras complement the rich contextual information of frame cameras, making event-frame asymmetric stereo promising for high-speed motion and extreme lighting conditions. However, the modality gap is severe: existing methods either use domain-level alignment (unified representation + Siamese feature extraction) or feature-level alignment (independent encoders + shared embeddings), but both risk marginalizing domain-specific discriminative cues. The key challenge is learning expressive representations without information-lossy marginalization.

## Method

### Overall Architecture

The asymmetric stereo inputs are alternately designated as target and source domains. CMPStereo learns aligned stereo representations in the target domain's canonical space. Two complementary configurations are instantiated (evCMPStereo with events as target, imgCMPStereo with images as target), and Bi-CMPStereo simultaneously leverages cost volumes from both directions for robust disparity estimation.

### Key Designs

1. **Stereo Canonicalization Constraint (SCC)**: Regularizes the network to learn target-domain discriminative features from both modalities, achieving high-fidelity cross-modal alignment in the target domain's canonical space. This ensures that features extracted from the source domain also possess target-domain discriminative expressiveness.

2. **Cross-Domain Embedding Adapter (CDEA)**: Enhances target-domain cues that are weakly encoded in the source domain. A lightweight adapter performs initial source-to-target adaptation at the feature level, which is further refined by domain-specific encoders.

3. **Hierarchical Visual Transform (HVT)**: Employs HVT when extracting contextual features from frame images to avoid shortcut learning and enhance generalization. Cascaded ConvGRU iteratively refines disparity. Bidirectional cost volumes fuse complementary information for final robust estimation.

### Loss & Training

Disparity loss with iterative refinement; bidirectional cost volumes each produce disparity estimates before fusion. The SCC constraint is imposed as a regularization term during training.

## Key Experimental Results

### Main Results

Evaluated on DSEC and MVSEC benchmarks:

| Benchmark | Metric | Prev. SOTA | Bi-CMPStereo |
|-----------|--------|-----------|-------------|
| DSEC | All metrics | Baseline | **Significantly surpasses** |
| MVSEC | All metrics | Baseline | **Significantly surpasses** |

Significantly outperforms SOTA in both accuracy and generalization.

### Ablation Study

- The bidirectional framework outperforms either unidirectional configuration
- SCC constraint is critical for improving cross-modal feature quality
- CDEA effectively supplements target-domain cues missing from the source domain

### Key Findings

- Alternating target domains effectively prevents information marginalization
- Bidirectional cost volumes provide complementary matching confidence
- Preserving domain-specific cues is more effective than pursuing unified representations

## Highlights & Insights

- The "alternating target domain" bidirectional design philosophy is novel — rather than seeking a compromise common space, each direction is fully exploited
- SCC combines stereo matching geometric constraints with cross-modal alignment
- HVT prevents the model from taking shortcuts by directly using frame information to bypass event information

## Limitations & Future Work

- The bidirectional framework implies twice the computational overhead
- The choice of event representation (event concentration) may not be optimal
- Validated on only two stereo benchmarks

## Related Work & Insights

- The alternating target domain framework can be generalized to other cross-modal fusion tasks
- SCC's domain prompting idea draws inspiration from prompting in NLP
- This systematic approach to event-frame fusion provides a reference for neuromorphic sensor applications

## Rating

7/10 — Systematically complete method design with significant experimental improvements, representing a strong advance in asymmetric stereo.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] GASP: Guided Asymmetric Self-Play For Coding LLMs](../../ICLR2026/llm_nlp/gasp_guided_asymmetric_self-play_for_coding_llms.md)
- [\[ACL 2026\] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction](../../ACL2026/llm_nlp/hcre_llm-based_hierarchical_classification_for_cross-document_relation_extractio.md)
- [\[ICLR 2026\] ELLMob: Event-Driven Human Mobility Generation with Self-Aligned LLM Framework](../../ICLR2026/llm_nlp/ellmob_event-driven_human_mobility_generation_with_self-aligned_language_models.md)
- [\[NeurIPS 2025\] PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs](../../NeurIPS2025/llm_nlp/presto_preimage-informed_instruction_optimization_for_prompting_black-box_llms.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)

</div>

<!-- RELATED:END -->
