---
title: >-
  [Paper Note] HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding
description: >-
  [CVPR 2026][Multimodal VLM][3D Affordance] This paper proposes the HAMMER framework, which extracts contact-aware intention embeddings from an MLLM, enhances point cloud features via hierarchical cross-modal fusion…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "3D Affordance"
  - "MLLM"
  - "Cross-Modal Fusion"
  - "Point Cloud"
  - "Intention Understanding"
date: 2026-05-08
content_hash: 9a13f2ca95007e7d
---

# HAMMER: Harnessing MLLM via Cross-Modal Integration for Intention-Driven 3D Affordance Grounding

**Conference**: CVPR 2026
**arXiv**: [2603.02329](https://arxiv.org/abs/2603.02329)  
**Code**: [https://rayyoh.github.io/Hammer/](https://rayyoh.github.io/Hammer/)  
**Area**: Multimodal VLM
**Keywords**: 3D Affordance, MLLM, Cross-Modal Fusion, Point Cloud, Intention Understanding

## TL;DR
This paper proposes the HAMMER framework, which extracts contact-aware intention embeddings from an MLLM, enhances point cloud features via hierarchical cross-modal fusion, and injects 3D spatial information into the intention embeddings through a multi-granular geometry lifting module. The framework achieves interaction-image-based 3D affordance grounding and comprehensively outperforms existing methods on the PIAD benchmark.

## Background & Motivation
**Background**: Intention-driven 3D affordance grounding—predicting operable regions on point clouds from interaction images—is a critical task bridging visual understanding and physical interaction, with applications in robotic manipulation and imitation learning.

**Limitations of Prior Work**: GREAT relies on hand-crafted templates and two-stage training, using MLLMs only to generate textual descriptions as intermediate representations. InteractVLM renders point clouds as multi-view images, applies a 2D segmentor to predict contact maps, and back-projects them to 3D, introducing geometric inconsistencies and detail loss.

**Key Challenge**: Existing methods either fail to fully exploit the powerful understanding capabilities of MLLMs (using them only for text generation) or introduce unavoidable information loss through intermediate representations.

**Goal**: To fully leverage the multimodal understanding capabilities of MLLMs for 3D affordance grounding, avoiding the information loss associated with intermediate text or 2D mask representations.

**Key Insight**: Directly extract embeddings encoding interaction intent from the hidden layers of an MLLM, and inject MLLM knowledge into point cloud features via cross-modal attention.

**Core Idea**: A special `[CONT]` token aggregates the MLLM's interaction intent; hierarchical cross-modal fusion and multi-granular geometry lifting then enable 3D affordance grounding.

## Method

### Overall Architecture
Interaction image $\mathbf{I}$ → MLLM extracts intention embedding $\bm{f}_c$ and hidden states $\bm{h}$ → hierarchical cross-modal fusion enhances point cloud features → multi-granular geometry lifting injects 3D information into $\bm{f}_c$ to obtain $\bm{f}_c^{3D}$ → decoder generates affordance map $\bm{p}$.

### Key Designs

1. **Affordance-Guided Intention Embedding**:

    - *Function*: Extract a compact interaction-intent representation from the MLLM.
    - *Mechanism*: A new `[CONT]` token is added to the MLLM vocabulary; an object-centric prompt guides the model to focus on relevant semantics. The last-layer hidden state of `[CONT]` is extracted and projected via an MLP: $\bm{f}_c = \psi_c(\bm{h}_{[\text{CONT}]})$. An auxiliary task additionally trains the model to generate textual affordance labels.
    - *Design Motivation*: Directly aggregating interaction information from the MLLM eliminates the need for intermediate text or 2D masks.

2. **Hierarchical Cross-Modal Integration**:

    - *Function*: Enhance point cloud features using MLLM hidden states.
    - *Mechanism*: A two-stage process is employed. In the encoder bottleneck stage, cross-attention fuses point cloud features with MLLM hidden states: $\tilde{\bm{f}}_p^{enc} = \text{CrossAttn}(\bm{f}_p^{enc}, \bm{f}_h, \bm{f}_h)$. In the decoder stage, a gating mechanism adaptively weights MLLM tokens to produce a global descriptor, which is then concatenated with full-resolution point cloud features.
    - *Design Motivation*: MLLM hidden states carry rich visual understanding and world knowledge, compensating for the semantic limitations of a pure 3D backbone.

3. **Multi-Granular Geometry Lifting**:

    - *Function*: Inject 3D geometric awareness into the 2D intention embedding.
    - *Mechanism*: Multi-scale point cloud features $\{\bm{f}_p^{(i)}\}$ from each decoder layer are used to progressively fuse geometric information via a layer-wise attention mechanism: $\bm{f}_c^{(i)} = \bm{f}_c^{(i-1)} + \text{Softmax}\!\left(\frac{\bm{q}^{(i)}(\bm{k}^{(i)})^T}{\sqrt{d}}\right) \bm{v}^{(i)}$, injecting structural and surface features from coarse to fine.
    - *Design Motivation*: The 2D intention embedding lacks 3D spatial information. Unlike the back-projection approach of InteractVLM, directly lifting the embedding's 3D awareness is more general and does not depend on camera parameters.

### Loss & Training
$$\mathcal{L} = \lambda_{txt}\mathcal{L}_{txt} + \lambda_{aff}\mathcal{L}_{aff}, \quad \mathcal{L}_{aff} = \mathcal{L}_{focal} + \mathcal{L}_{dice}$$

## Key Experimental Results

### Main Results (PIAD Benchmark)

| Method | Conference | Seen aIOU↑ | Seen AUC↑ | Unseen aIOU↑ | Unseen AUC↑ |
|--------|------------|-----------|----------|-------------|------------|
| IAGNet | ICCV'23 | 20.51 | 84.85 | 7.95 | 71.84 |
| GREAT | CVPR'25 | 19.61 | 85.22 | 8.32 | 67.46 |
| **HAMMER** | - | **22.20** | **88.43** | **13.71** | **80.92** |

### Ablation Study

| Configuration | Seen aIOU | Unseen aIOU | Note |
|---------------|----------|------------|------|
| Full HAMMER | 22.20 | 13.71 | Complete model |
| w/o Hierarchical Fusion | ↓ | ↓ | No MLLM knowledge injection |
| w/o Geometry Lifting | ↓ | ↓ | Intention embedding lacks 3D information |
| w/o Text Auxiliary | ↓ | ↓ | Weaker task-level understanding |

### Key Findings
- The most significant gains appear in the Unseen setting (aIOU: 8.32→13.71; AUC: 67.46→80.92), demonstrating strong generalization.
- The hierarchical fusion and geometry lifting modules contribute complementarily: the former improves semantic understanding while the latter improves spatial localization.
- The model also demonstrates strong robustness under noise-injected point cloud evaluations.

## Highlights & Insights
- **Full exploitation of multi-layer MLLM information**: Rather than relying solely on final-layer outputs, intermediate hidden states are used to enhance point cloud features—a substantially richer use of the MLLM than text generation alone.
- The **`[CONT]` token design** for intent aggregation is concise and effective, analogous to the `[SEG]` token in LISA but applied to 3D affordance grounding.
- The multi-granular geometry lifting module avoids the geometric inconsistencies inherent to 2D-to-3D back-projection.

## Limitations & Future Work
- Reliance on a pretrained MLLM (requiring LoRA fine-tuning) incurs substantial computational overhead.
- The point cloud encoder adopts the relatively simple PointNet++; a stronger 3D backbone could yield further improvements.
- Validation is limited to tabletop manipulation objects; scene-level affordance understanding is not addressed.
- Object category labels are used as prior inputs, which may be difficult to obtain in fully open-world settings.

## Related Work & Insights
- **vs. GREAT**: GREAT uses an MLLM to generate textual descriptions for subsequent fusion; HAMMER directly exploits MLLM hidden states, avoiding information loss during text conversion.
- **vs. InteractVLM**: InteractVLM back-projects 2D masks to 3D; HAMMER directly enhances the 3D awareness of intention embeddings via multi-granular geometry lifting, requiring no camera parameters.
- The paradigm of fusing MLLM hidden states with 3D representations to enhance cross-modal understanding is broadly extensible to other 3D understanding tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ — The fusion pathway combining MLLM hidden states with point clouds is original.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Multi-benchmark evaluation with ablation and robustness tests.
- Writing Quality: ⭐⭐⭐⭐ — Architectural comparison figures against prior methods are intuitive and effective.
- Value: ⭐⭐⭐⭐ — Provides a valuable design paradigm for applying MLLMs to 3D understanding tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[CVPR 2026\] VGGDrive: Empowering Vision-Language Models with Cross-View Geometric Grounding for Autonomous Driving](vggdrive_empowering_vision-language_models_with_cross-view_geometric_grounding_f.md)
- [\[CVPR 2026\] CRIT: Graph-Based Automatic Data Synthesis to Enhance Cross-Modal Multi-Hop Reasoning](crit_graph-based_automatic_data_synthesis_to_enhance_cross-modal_multi-hop_reaso.md)
- [\[ICML 2026\] CG-MLLM: Captioning and Generating 3D Content via Multi-modal Large Language Models](../../ICML2026/multimodal_vlm/cg-mllm_captioning_and_generating_3d_content_via_multi-modal_large_language_mode.md)
- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)

</div>

<!-- RELATED:END -->
