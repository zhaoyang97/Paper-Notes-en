---
title: >-
  [Paper Note] AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models
description: >-
  [CVPR 2026][3D Vision][VFM evaluation] This paper proposes AVA-Bench, which decomposes the evaluation of vision foundation models (VFMs) into 14 "atomic visual abilities" (AVAs). Through train/test distribution alignment and single-ability isolation testing, AVA-Bench precisely identifies the strengths and weaknesses of VFMs. A key finding is that a 0.5B LLM preserves the same VFM ranking as a 7B LLM, reducing evaluation cost by $8\times$.
tags:
  - CVPR 2026
  - 3D Vision
  - VFM evaluation
  - atomic visual abilities
  - benchmark
  - ability disentanglement
  - visual question answering
date: 2026-05-08
content_hash: 93479e3d76cb09c5
---

# AVA-Bench: Atomic Visual Ability Benchmark for Vision Foundation Models

**Conference**: CVPR 2026
**arXiv**: [2506.09082](https://arxiv.org/abs/2506.09082)
**Code**: [https://zheda-mai.github.io/AVA-Bench/](https://zheda-mai.github.io/AVA-Bench/) (open source)
**Area**: 3D Vision / Vision Foundation Model Evaluation
**Keywords**: VFM evaluation, atomic visual abilities, benchmark, ability disentanglement, visual question answering

## TL;DR
This paper proposes AVA-Bench, which decomposes the evaluation of vision foundation models (VFMs) into 14 "atomic visual abilities" (AVAs). Through train/test distribution alignment and single-ability isolation testing, AVA-Bench precisely identifies the strengths and weaknesses of VFMs. A key finding is that a 0.5B LLM preserves the same VFM ranking as a 7B LLM, reducing evaluation cost by $8\times$.

## Background & Motivation

**State of the Field**: Vision foundation models (e.g., DINOv2, CLIP, SAM) have proliferated rapidly. Evaluation approaches fall into two categories: task-specific evaluation (e.g., ImageNet classification, COCO detection) and general VQA evaluation (using an LLM as a universal head and testing on VQA benchmarks).

**Limitations of Prior Work**: General VQA evaluation has two blind spots: (i) the instruction fine-tuning data distribution is misaligned with the VQA test distribution, so failures may stem from data mismatch rather than insufficient visual ability; (ii) VQA questions typically require multiple visual abilities simultaneously, making it impossible to attribute errors to the absence of all required abilities versus only one critical one.

**Root Cause**: Composite-task evaluation does not support ability attribution — when a VFM underperforms on VQA, the specific failing ability cannot be identified with existing methods.

**Paper Goals**: To design a benchmark that isolates individual visual abilities for evaluation, while ensuring train/test distribution consistency to eliminate noise introduced by data mismatch.

**Starting Point**: Decompose complex visual reasoning into 14 "atomic visual abilities" (analogous to the periodic table of elements), evaluating each ability independently.

**Core Idea**: Through atomic decomposition and distribution alignment, transform VFM evaluation from "speculative selection" to "principled engineering selection."

## Method

### Overall Architecture
The inputs are various VFMs (DINOv2, CLIP, SigLIP, SAM, MiDaS, AIMv2, RADIO, etc.), and the outputs are "ability fingerprints" for each VFM across 14 atomic visual abilities. The pipeline proceeds as follows: (1) define 14 AVAs; (2) construct independent train/test data for each AVA with aligned distributions; (3) freeze the VFM and use an LLM as a universal head, fine-tuning the connector and LoRA independently per AVA; (4) evaluate on the corresponding AVA test set.

### Key Designs

1. **Definition and Selection of 14 Atomic Visual Abilities**:

    - Function: Decompose complex visual reasoning into 14 fundamental perceptual abilities.
    - Mechanism: The AVA set is determined by cross-referencing two sources — (a) visual primitives from synthetic text-to-image (T2I) benchmarks (quantity, color, texture, spatial relations, etc.), and (b) GPT-4 analysis of which visual skills are required by questions in VQA datasets. The intersection yields 14 AVAs: localization, counting, spatial reasoning, orientation, color, texture, emotion, absolute/relative depth estimation, action/fine-grained/object/scene recognition, and OCR.
    - Design Motivation: Non-perceptual reasoning (e.g., mathematical reasoning) is excluded to focus purely on visual perception, ensuring evaluation results genuinely reflect VFM visual capability.

2. **Distribution-Aligned Data Construction**:

    - Function: Construct datasets with fully aligned train/test distributions for each AVA.
    - Mechanism: Training and test sets are split 80/20 per AVA, ensuring that object categories and answer ranges seen during training are exactly consistent with those at test time. For example, counting tasks balance sample counts across quantity intervals; depth tasks ensure uniform distribution of object categories across depth intervals.
    - Design Motivation: Eliminate the confound in which test failures arise from missing training samples rather than insufficient VFM visual ability, ensuring that failures are genuinely attributable to the VFM.
    - Novelty: Traditional VQA benchmarks frequently have misaligned train/test distributions, introducing data-shift noise that this work explicitly removes.

3. **Bounding Box Isolation Strategy**:

    - Function: Disentangle the "localization" ability from other abilities by providing target object bounding boxes.
    - Mechanism: For example, when evaluating depth estimation, the target object's bounding box is provided to avoid requiring the model to first localize and then estimate depth. For spatial reasoning, two objects are marked with red and blue bounding boxes so that only the spatial relationship judgment is tested.
    - Design Motivation: Effectiveness is validated empirically — when bounding boxes are provided, all VFMs perform comparably and well on spatial reasoning; when bounding boxes are removed, performance diverges sharply and correlates strongly with localization ability rankings, demonstrating that failures in composite tasks often originate from a specific AVA bottleneck.

4. **Lightweight LLM Evaluation Strategy**:

    - Function: Demonstrate that a 0.5B LLM suffices to preserve VFM ranking consistency.
    - Mechanism: When the goal is comparing relative VFM rankings (rather than maximizing absolute accuracy), a 0.5B Qwen2 model and a 7B Vicuna-1.5 model produce nearly identical VFM rankings, at approximately $8\times$ lower computational cost.
    - Design Motivation: Makes large-scale VFM evaluation practically feasible, eliminating the requirement for expensive large models as evaluation heads.

### Data Construction Details
- A total of **218K** image-question pairs drawn from **26** diverse datasets.
- Coverage spans general scenes, wildlife, vehicles, indoor/outdoor environments, remote sensing, and more.
- Rigorous quality control per AVA: sample balancing, minimum bounding box area thresholds, and object category diversity.

### Evaluation Metrics
- Absolute depth and counting: normalized MAE
- Localization: GIoU
- Color: CIEDE2000 color difference
- OCR: ANLS (normalized Levenshtein similarity)
- All other AVAs: standard accuracy

## Key Experimental Results

### Main Results

| VFM | Pre-training Paradigm | Avg. Rank | Strongest AVA | Weakest AVA |
|-----|----------------------|-----------|---------------|-------------|
| SigLIP-2 | Language-supervised (Sigmoid contrastive) | 1–2 | Superior across most AVAs | — |
| SigLIP-1 | Language-supervised (Sigmoid contrastive) | 1–2 | OCR and language-related AVAs | — |
| AIMv2 | Multimodal autoregressive | 2–3 | Strong generalization | — |
| InternVL-2.5 | Language-supervised | Mid | — | — |
| CLIP | Language-supervised (contrastive) | Mid | — | — |
| RADIO | Multi-teacher distillation | Mid-high | Robustly balanced | — |
| DINOv2 | Self-supervised | Mid | Orientation, localization | OCR and language-related tasks |
| SAM | Segmentation-supervised | Lower | Color recognition | Most other AVAs |
| MiDaS | Depth-supervised | Lower | Depth-related AVAs | Most other AVAs |

### Ablation Study

| Configuration | Key Performance | Notes |
|---------------|----------------|-------|
| Spatial reasoning with bbox | All VFMs perform well and comparably | Bbox successfully isolates localization ability |
| Spatial reasoning without bbox | VFM performance diverges sharply | Rankings highly correlated with localization ability |
| 0.5B LLM evaluation head | VFM rankings consistent with 7B | Evaluation cost reduced by $8\times$ |
| 7B LLM evaluation head | Baseline rankings | Higher absolute accuracy, rankings unchanged |
| Large object localization (0.3–0.5) | Minimal differences across VFMs | Large-object localization is low-level; all models handle it well |
| Small object localization (<0.1) | MiDaS and SAM degrade significantly | Small-object localization reveals genuine capability gaps |

### Key Findings
- **Language supervision is critical**: SigLIP-2 and AIMv2 lead across most AVAs, indicating that language supervision is essential for general visual ability.
- **Specialized VFMs have niche advantages**: DINOv2 matches or surpasses language-supervised models on orientation recognition; SAM achieves the best color recognition.
- **Low-level abilities are saturated**: All VFMs perform well on texture recognition, relative depth estimation, and object recognition, suggesting that VQA failures primarily originate from bottlenecks in specific higher-level AVAs.
- **Connector is the bottleneck for non-language-aligned VFMs**: DINOv2's linear probing accuracy is 66.3%, which drops to 25.67% after passing through the connector, indicating that critical visual information is lost during modality alignment.

## Highlights & Insights
- The **atomic evaluation paradigm** is conceptually elegant: it transforms the vague question "is this VFM good?" into the precise question "how good is it across 14 dimensions?", analogous to factor analysis in psychometrics. This approach is transferable to any scenario requiring fine-grained capability diagnosis.
- The **bounding box isolation experiment** elegantly validates the methodological rationale: comparable performance with bounding boxes versus sharp divergence without them constitutes a well-designed contrastive experiment worth emulating.
- The **0.5B vs. 7B finding** has direct practical value: when relative rankings are the target rather than absolute accuracy, small models are entirely sufficient — an insight generalizable to other benchmark evaluation settings.
- The **distribution alignment and bias elimination strategies** in data construction (e.g., interval balancing for counting, category-interval distribution balancing for depth) set a standard for benchmark design.

## Limitations & Future Work
- Although the selection of 14 AVAs is literature-grounded, it may be incomplete — abilities such as illumination understanding, occlusion reasoning, and perspective understanding are absent.
- The evaluation data volume varies substantially across AVAs (from 8.5K to 44.9K), potentially undermining the reliability of evaluations for smaller AVAs.
- The evaluation pipeline depends on the LLaVA architecture; different LLM integration approaches may affect conclusions.
- Only static image abilities are evaluated; temporal visual abilities in video understanding are not addressed.
- Non-language-aligned VFMs suffer significant information loss through connector mapping, which is an inherent limitation of the evaluation pipeline rather than purely a property of the VFMs themselves.

## Related Work & Insights
- **vs. MMBench/SEED and similar general VQA benchmarks**: Those benchmarks serve as "comprehensive examinations," whereas AVA-Bench functions as a "targeted diagnostic test"; the two are complementary rather than mutually exclusive.
- **vs. Platonic Representation Hypothesis [Huh et al.]**: AVA-Bench experiments partially support this hypothesis (convergence at low-level abilities) while revealing significant divergence at higher-level abilities, providing a more nuanced empirical validation.
- **vs. RADIO [Ranzinger et al.]**: RADIO, which fuses multiple VFMs via multi-teacher distillation, performs robustly on AVA-Bench, validating the effectiveness of the model fusion approach.

## Rating
- Novelty: ⭐⭐⭐⭐ The atomic decomposition evaluation concept is intuitively motivated yet had not been systematically implemented before; execution quality is high.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 9 VFMs × 14 AVAs; ablation studies (bounding box, LLM size, object scale) are elegantly designed.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures are visually polished (radar charts, heat maps, ranking plots); logic is clear and the motivating examples are vivid.
- Value: ⭐⭐⭐⭐ Directly informative for VFM selection in practice; sustained community adoption and benchmark maintenance remain to be observed.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Foundry: Distilling 3D Foundation Models for the Edge](foundry_distilling_3d_foundation_models_for_the_edge.md)
- [\[CVPR 2026\] ArtHOI: Taming Foundation Models for Monocular 4D Reconstruction of Hand-Articulated-Object Interactions](arthoi_taming_foundation_models_for_monocular_4d_reconstruction_of_hand-articula.md)
- [\[CVPR 2026\] Sky2Ground: A Benchmark for Site Modeling under Varying Altitude](sky2ground_a_benchmark_for_site_modeling_under_varying_altitude.md)
- [\[CVPR 2026\] Pano360: Perspective to Panoramic Vision with Geometric Consistency](pano360_perspective_to_panoramic_vision_with_geome.md)
- [\[ICLR 2026\] GIQ: Benchmarking 3D Geometric Reasoning of Vision Foundation Models with Simulated and Real Polyhedra](../../ICLR2026/3d_vision/giq_benchmarking_3d_geometric_reasoning_of_vision_foundation_models_with_simulat.md)

<!-- RELATED:END -->
