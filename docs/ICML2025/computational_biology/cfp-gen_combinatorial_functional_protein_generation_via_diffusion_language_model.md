---
title: >-
  [Paper Note] CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models
description: >-
  [ICML 2025][Computational Biology][Protein Design] Proposing CFP-Gen—a large-scale diffusion language model that achieves combinatorial protein generation under multimodal functional constraints (functional annotations + sequence motifs + 3D structures) via Annotation-Guided Feature Modulation (AGFM) and Residue-level Control Function Encoding (RCFE), improving the F1 score by 30% compared to ESM3.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Protein Design"
  - "Diffusion Language Models"
  - "Multi-functional Constraints"
  - "Combinatorial Conditional Generation"
  - "Functional Annotation"
date: 2026-05-08
content_hash: 43fcc6c6294f9113
---

# CFP-Gen: Combinatorial Functional Protein Generation via Diffusion Language Models

**Conference**: ICML 2025  
**arXiv**: [2505.22869](https://arxiv.org/abs/2505.22869)  
**Code**: None  
**Area**: Medical Imaging/Protein Design  
**Keywords**: Protein Design, Diffusion Language Models, Multi-functional Constraints, Combinatorial Conditional Generation, Functional Annotation

## TL;DR
Proposing CFP-Gen—a large-scale diffusion language model that achieves combinatorial protein generation under multimodal functional constraints (functional annotations + sequence motifs + 3D structures) via Annotation-Guided Feature Modulation (AGFM) and Residue-level Control Function Encoding (RCFE), improving the F1 score by 30% compared to ESM3.

## Background & Motivation

**Background**: Protein language models (PLMs) have shown immense potential in de novo protein design. Most existing approaches generate proteins based on a single condition (functional label only, structure only, or sequence pattern only).

**Limitations of Prior Work**:
   - Single-condition generation cannot satisfy multiple functional constraints simultaneously—practical protein engineering requires satisfying various constraints such as catalytic activity (EC number), domain (IPR domain), and subcellular localization (GO term) at the same time.
   - Iterative filtering or multi-step optimization pipelines are highly inefficient and suffer from low success rates under multiple constraints.
   - Such pipelines become even more infeasible when data is limited.

**Key Challenge**: Real-world protein engineering is a multi-objective optimization problem, but existing generative models only support single-objective targets.

**Goal**: Model protein generation under multimodal functional constraints simultaneously within a unified model.

**Key Insight**: Combine diffusion language models with multimodal conditional encoding—different types of functional constraints are injected into the generation process via dedicated modules.

**Core Idea**: Combining AGFM (annotation $\rightarrow$ feature distribution modulation) + RCFE (critical residues $\rightarrow$ residue-level control) + structure encoder (3D backbone $\rightarrow$ geometric constraints) to achieve multi-objective protein design.

## Method

### Overall Architecture
CFP-Gen is based on a diffusion language model with the ESM architecture:
1. Input: Noisy protein sequences + multiple functional conditions (GO terms, IPR domains, EC numbers, sequence motifs, 3D structures).
2. Iteratively denoise to generate the target protein sequence.
3. Functional conditions are injected into each ESM block via three dedicated modules.

### Key Designs

1. **Annotation-Guided Feature Modulation (AGFM)**:

    - **Function**: Dynamically modulates functional annotations (GO/IPR/EC) into the feature distribution of the noisy sequence.
    - **Mechanism**: Functional labels are encoded as one-hot vectors $\rightarrow$ learnable affine transformations yield scale $\gamma$ and shift $\beta$ $\rightarrow$ perform $\gamma \cdot x + \beta$ on features after LayerNorm (similar to AdaIN/FiLM).
    - **Key Advantages**: Multiple annotation labels can be freely combined—joint optimization during training ensures strict alignment between function and sequence, while supporting flexible annotation combinations during inference.
    - **Design Motivation**: More direct than classifier-guided methods—AGFM is embedded inside the model, eliminating the need for an external classifier.

2. **Residue Control Functional Encoding (RCFE)**:

    - **Function**: Provides precise residue-level control over critical amino acid residues (sequence motifs/functional domains).
    - **Mechanism**: Employs an ESM-like Transformer encoder to process partial sequences marked with functional domains, capturing epitope relationships and evolutionary associations among residues, which are then injected into the main network as conditions.
    - **Design Motivation**: Certain residues are critical to function (e.g., active sites)—global functional annotations are insufficient for precise control over these local positions.

3. **3D Structural Conditions**:

    - **Function**: Utilizes the 3D backbone atomic coordinates of reference proteins as geometric constraints.
    - **Mechanism**: Uses an off-the-shelf structural encoder (such as ESM-IF) to encode backbone coordinates into feature vectors $\rightarrow$ injected into the generation process.
    - **Design Motivation**: Inverse folding scenario—optimizing sequence functions while keeping the structure intact.

### Loss & Training
- Discrete Diffusion: Adding and removing noise in the token space.
- Cross-Entropy Loss: Predicting the denoised amino acids.
- Multi-conditional joint training: Randomly masking some conditions (dropout) to enhance robustness.
- Extended based on the ESM2-650M architecture.

## Key Experimental Results

### Main Results
Functional protein generation (evaluated by leading functional predictors):

| Method | F1 (GO-MF) ↑ | F1 (EC) ↑ | Sequence Novelty ↑ |
|------|------------|----------|------------|
| EvoDiff | 0.35 | 0.28 | High |
| DPLM | 0.42 | 0.38 | High |
| ESM3 | 0.48 | 0.45 | Medium |
| **CFP-Gen** | **0.62** | **0.58** | **High** |

### Inverse Folding

| Method | AAR (Amino Acid Recovery) ↑ | Structural Consistency (TM-Score) |
|------|-------|-----|
| ProteinMPNN | 0.45 | 0.89 |
| DPLM | 0.48 | 0.87 |
| **CFP-Gen** | **0.57** | **0.91** |

### Multi-functional Protein Design

| Constraint Combinations | Success Rate ↑ | Description |
|---------|---------|------|
| EC + GO (Bifunctional enzyme) | 72.5% | Catalytic activity + Subcellular localization |
| EC + IPR (Enzyme + Domain) | 68.3% | Catalytic activity + Specific domain |
| EC + Sequence motif | 78.1% | Catalytic activity + Conserved site |

### Ablation Study

| Configuration | F1 (GO-MF) | Description |
|------|-----------|------|
| Without AGFM (no annotation modulation) | 0.45 | Degenerates to unconditional generation |
| Without RCFE (no residue control) | 0.55 | Lacks precise local control |
| Without structural conditions | 0.58 | No geometric constraints |
| **Complete CFP-Gen** | **0.62** | Three conditions are complementary |

### Key Findings
- CFP-Gen's F1 is improved by 30% compared to ESM3—joint optimization of multimodal constraints is far superior to single-constraint methods.
- AAR (inverse folding) improved by 9%—structural conditions effectively guide sequence design.
- The success rate of multi-functional protein design reaches 68-78%—demonstrating the feasibility of large-scale multi-functional enzyme design for the first time.
- Conditional dropout training enables support for arbitrary condition combinations during inference—offering extremely high flexibility.
- The generated protein sequences possess high novelty—it does not merely copy known proteins from the training set.

## Highlights & Insights
- **Unified processing of multimodal conditions** is the key contribution—integrating scattered functional, sequence, and structural constraints into a single generative framework.
- The FiLM-style modulation of AGFM is simple and effective—a mature technique borrowed from the image generation domain.
- The residue-level control of RCFE bridges the gap between global annotations and local precision.
- Multi-functional protein design (such as enzymes with dual catalytic activities) represents an important frontier in protein engineering.
- The foundation on the ESM architecture ensures scalability and compatibility with the existing protein AI ecosystem.

## Limitations & Future Work
- Functional validation is solely conducted through computational predictors; physical experimental validation (wet lab) is missing.
- Whether the generated proteins are actually foldable and expressible has not been validated.
- The training data is biased toward known protein functions; generalization capabilities to completely novel functions remain unknown.
- 3D structural conditions rely on the quality of external encoders.
- The trade-off between the diversity of generated sequences and functional satisfaction has not been thoroughly explored.

## Related Work & Insights
- **vs ESM3**: SOTA in single-condition generation; CFP-Gen significantly improves performance through multi-conditional constraints.
- **vs ProteinMPNN**: Focuses on inverse folding and does not support functional constraints; CFP-Gen performs better even in inverse folding task.
- **vs RFdiffusion/Chroma**: Structure-centric design methods, whereas CFP-Gen's function-centric approach is more practical for function-guided protein engineering.
- **Key Insight**: The ideas behind multimodal conditional generation can be extended to other molecular designs (drugs, materials, etc.).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Combinatorial functional protein generation is an important new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple tasks (functional generation / inverse folding / multi-functional design) but lacks wet-lab validation.
- Writing Quality: ⭐⭐⭐⭐ Clear framework and reasonable module design.
- Value: ⭐⭐⭐⭐⭐ Significant practical value for protein engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MMCP-GEN: A Modality-Extensible Diffusion Language Model for Conditional Protein Sequence Generation](../../CVPR2026/computational_biology/mmcp-gen_a_modality-extensible_diffusion_language_model_for_conditional_protein_.md)
- [\[ICML 2025\] Steering Protein Language Models](steering_protein_language_models.md)
- [\[ACL 2025\] Concept Bottleneck Language Models For Protein Design](../../ACL2025/computational_biology/concept_bottleneck_language_models_for_protein_design.md)
- [\[ICML 2025\] Training Flexible Models of Genetic Variant Effects from Functional Annotations using Accelerated Linear Algebra](training_flexible_models_of_genetic_variant_effects_from_functional_annotations_.md)
- [\[ICML 2025\] Elucidating the Design Space of Multimodal Protein Language Models](elucidating_the_design_space_of_multimodal_protein_language_models.md)

</div>

<!-- RELATED:END -->
