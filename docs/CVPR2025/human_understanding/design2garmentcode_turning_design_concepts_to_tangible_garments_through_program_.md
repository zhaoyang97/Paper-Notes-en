---
title: >-
  [Paper Note] Design2GarmentCode: Turning Design Concepts to Tangible Garments Through Program Synthesis
description: >-
  [CVPR 2025][Human Understanding][Program Synthesis] Proposes Design2GarmentCode, the first neuro-symbolic framework that translates multi-modal design inputs (text/image/sketch) into parameterized garment drafting programs (GarmentCode DSL). This achieves a 100% simulation success rate and 88.67% user satisfaction, with the generated programs being fully editable and highly parameterizable.
tags:
  - "CVPR 2025"
  - "Human Understanding"
  - "Program Synthesis"
  - "Garment Design"
  - "Neuro-symbolic Method"
  - "GarmentCode DSL"
  - "Parameterized Patterns"
date: 2026-05-08
content_hash: 890cd68c9f6dc0ec
---

# Design2GarmentCode: Turning Design Concepts to Tangible Garments Through Program Synthesis

**Conference**: CVPR 2025  
**arXiv**: [2412.08603](https://arxiv.org/abs/2412.08603)  
**Code**: [https://style3d.github.io/design2garmentcode](https://style3d.github.io/design2garmentcode)  
**Area**: Human Understanding / Garment Generation  
**Keywords**: Program Synthesis, Garment Design, Neuro-symbolic Method, GarmentCode DSL, Parameterized Patterns

## TL;DR

Proposes Design2GarmentCode, the first neuro-symbolic framework that translates multi-modal design inputs (text/image/sketch) into parameterized garment drafting programs (GarmentCode DSL). This achieves a 100% simulation success rate and 88.67% user satisfaction, with the generated programs being fully editable and highly parameterizable.

## Background & Motivation

### Background

**Background**: Generating digital garments from design concepts to wearable 3D clothes requires sequential steps of pattern design, sewing, and simulation. Existing methods (e.g., DressCode, Sewformer) directly predict discrete representations of pattern shapes (e.g., point sequences), but the generated sewing patterns often suffer from structurally incorrect topology, leading to simulation failures.

### Limitations of Prior Work

**Limitations of Prior Work**: (1) Low simulation success rate for discrete pattern prediction (DressCode only achieves 84%) due to sewing topology errors that crash the simulator. (2) Non-editable output: users cannot modify design parameters (e.g., lengthening sleeves, changing necklines). (3) Complex garments (composed of multiple sewn patterns) are represented as long token sequences (1500+), making prediction highly challenging.

### Key Challenge

**Key Challenge**: Directly predicting geometric coordinates of sewing patterns is flexible but highly error-prone, whereas generating parameterized programs guarantees structural correctness but requires the LLM to understand a domain-specific language (DSL) for garments.

### Key Insight

**Key Insight**: Cast the garment generation problem as a program synthesis task. Instead of predicting pattern coordinates, the framework predicts the parameterized programs and parameter configurations that generate patterns. Executing the program automatically yields structurally correct patterns.

### Core Idea

**Core Idea**: Multi-modal Understanding Agent + Finetuned LLM Program Generation Agent $\rightarrow$ Parameterized garment pattern program $\rightarrow$ 100% structural correctness.

## Method

### Key Designs

1. **Multi-modal Understanding Agent (MMUA)**:

    - Function: Translates multi-modal design inputs into structured design choices.
    - Mechanism: A pretrained LMM (e.g., GPT-4V) receives text/image/sketch inputs and answers design questions (e.g., "Neckline type? A. Crewneck B. V-neck C. Mandarin collar") in a multiple-choice format (instead of numerical estimation). All design decisions are converted into discrete choices and quantized parameters.
    - Design Motivation: Empirical findings show that LMMs are far more accurate at answering multiple-choice questions than estimating numerical values.

2. **DSL Generation Agent (DSL-GA)**:

    - Function: Generates GarmentCode parameterized programs from design choices.
    - Mechanism: Fine-tunes an LLM using LoRA on pairs of GarmentCode programs and natural language comments. It generates a fixed-length token sequence (only 122 tokens vs. 1500 for DressCode) containing program types and quantized parameters. A projector (decoder-only transformer) translates parameter tokens into actual parameter values for the DSL program.
    - Design Motivation: The 122 tokens sequence is 1/10th the length of DressCode's representation, significantly reducing sequence prediction difficulty. Quantizing parameters into integers, booleans, and enums eliminates cumulative errors from float prediction.

3. **GarmentCode Program Execution**:

    - Function: Translates programs $\rightarrow$ patterns $\rightarrow$ 3D garments.
    - Mechanism: GarmentCode is a parameterized garment drafting DSL, where each program defines pattern shapes and sewing topology. Executing the program automatically generates topologically correct patterns, guaranteeing simulated outcomes.
    - Design Motivation: The structural correctness of the program is guaranteed by the DSL grammar, avoiding the invalid patterns often produced by coordinate prediction.

### Loss & Training

Standard autoregressive loss is utilized for LoRA fine-tuning of the LLM. A quantization function unifies boolean, integer, float, and enum parameters into discrete tokens. Float parameters are scaled by $\lambda=100$ and rounded to achieve centimeter-level precision.

## Key Experimental Results

### Main Results

| Metric | Design2GarmentCode | DressCode | Sewformer |
|------|-------------------|-----------|-----------|
| Simulation Success Rate (SSR) | **100%** | 84% | 65% |
| User Satisfaction (Image) | **88.67%** | - | 3.33% |
| User Aesthetics (Image) | **77%** | - | 5.33% |
| Chamfer Distance | **2.091**| - | 9.7 |
| Avg No. of Patterns | **11.02** | 5.11 | 10.11 |
| Token Sequence Length | **122** | 1500 | - |

### Key Findings
- **100% Simulation Success Rate**: The program synthesis method fundamentally eliminates topological errors.
- **More Complex Garments**: An average of 11 sewing patterns vs. DressCode's 5 patterns, allowing for more diverse design expressions.
- **10x Sequence Compression**: 122 vs. 1500 tokens makes predictions more accurate and highly efficient.
- **Editability**: Users can directly modify program parameters (e.g., sleeve length, neckline type) to regenerate garments, enabling a preview-edit-regenerate workflow.

## Highlights & Insights
- **Program Synthesis Guarantees Correctness**: Relying on the syntactic rules of a DSL to ensure structural correctness is inherently more reliable than end-to-end coordinate prediction.
- **Multiple-Choice > Numerical Estimation**: Discretizing the continuous design space leverages LMMs' strength in multiple-choice reasoning.
- **Parameterization Implies Editability**: The program-plus-parameter representation naturally supports editing, which is impossible with coordinate representations.

## Limitations & Future Work
- Constrained by the expressiveness of the GarmentCode DSL; it cannot generate garment structures undefined in the DSL.
- Fixed standard body shape, lacking support for custom sizing.
- Sketch input requires high drawing quality.
- Limited training data; specific garment styles may require additional fine-tuning.
- DSL scalability: supporting new garment types requires expanding the DSL syntax.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ A classic example of neuro-symbolic methods, introducing program synthesis to garment generation for the first time.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid user studies across text, image, and sketch modal inputs.
- Writing Quality: ⭐⭐⭐⭐ Clear methodological descriptions and elegant system design.
- Value: ⭐⭐⭐⭐⭐ Provides an editable, 100% correct, and practical solution for garment design automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ConceptScope: Characterizing Dataset Bias via Disentangled Visual Concepts](../../NeurIPS2025/human_understanding/conceptscope_characterizing_dataset_bias_via_disentangled_visual_concepts.md)
- [\[CVPR 2026\] Gaze Target Estimation Anywhere with Concepts](../../CVPR2026/human_understanding/gaze_target_estimation_anywhere_with_concepts.md)
- [\[CVPR 2025\] Shape My Moves: Text-Driven Shape-Aware Synthesis of Human Motions](shape_my_moves_text-driven_shape-aware_synthesis_of_human_motions.md)
- [\[ECCV 2024\] ReLoo: Reconstructing Humans Dressed in Loose Garments from Monocular Video in the Wild](../../ECCV2024/human_understanding/reloo_reconstructing_humans_dressed_in_loose_garments_from_monocular_video_in_th.md)
- [\[CVPR 2026\] Miburi: Towards Expressive Interactive Gesture Synthesis](../../CVPR2026/human_understanding/miburi_towards_expressive_interactive_gesture_synthesis.md)

</div>

<!-- RELATED:END -->
