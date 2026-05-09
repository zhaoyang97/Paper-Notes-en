---
title: >-
  [Paper Note] EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering
description: >-
  [ICLR 2026 (Workshop on Foundation Models for Science)][Medical Imaging][Protein Engineering] EvoFlows proposes an edit-based flow matching approach that learns mutational trajectories between evolutionarily related protein sequences, enabling controllable numbers of edits (insertions, deletions, substitutions) on a template sequence while jointly predicting *what* to mutate and *where* to mutate.
tags:
  - ICLR 2026 (Workshop on Foundation Models for Science)
  - Medical Imaging
  - Protein Engineering
  - Flow Matching
  - Edit Operations
  - Sequence-to-Sequence
  - Evolutionary Trajectories
date: 2026-05-08
content_hash: 21e2ac4d71e453dc
---

# EvoFlows: Evolutionary Edit-Based Flow-Matching for Protein Engineering

**Conference**: ICLR 2026 (Workshop on Foundation Models for Science)  
**arXiv**: [2603.11703](https://arxiv.org/abs/2603.11703)  
**Code**: None  
**Area**: Biomedical / Protein Design  
**Keywords**: Protein Engineering, Flow Matching, Edit Operations, Sequence-to-Sequence, Evolutionary Trajectories

## TL;DR

EvoFlows proposes an edit-based flow matching approach that learns mutational trajectories between evolutionarily related protein sequences, enabling controllable numbers of edits (insertions, deletions, substitutions) on a template sequence while jointly predicting *what* to mutate and *where* to mutate.

## Background & Motivation

The core objective of protein engineering is to generate functional variants from a known template sequence, requiring models to introduce biologically meaningful mutations. Existing protein language models exhibit several limitations in optimization tasks:

**Autoregressive models (e.g., ESM, ProtGPT2)**: Generate complete sequences from scratch, precluding local edits on a template and offering no control over the mutational distance from the template.

**Masked language models / discrete diffusion models (e.g., ESM-MLM, EvoDiff)**: Rely on pre-specified mutation sites (masked positions), whereas in practical protein engineering the optimal mutation sites are typically unknown. These methods cannot autonomously identify mutation positions.

**Lack of insertion and deletion (indel) support**: The vast majority of existing methods handle only fixed-length substitution mutations, despite the fact that a large proportion of adaptive changes in natural evolution arise from sequence length variation—i.e., insertions and deletions.

In summary, existing methods either do not support template-conditioned generation, require known mutation positions, or ignore indels—creating a substantial gap relative to the demands of real-world protein engineering.

## Method

### Overall Architecture

EvoFlows is a **variable-length sequence-to-sequence** modeling framework. Its core idea is to treat protein engineering as an "edit flow" from a template sequence to a target variant—a continuous trajectory of edit operations (insertions, deletions, substitutions). Under the flow matching framework, the model learns these evolutionary trajectories and, at inference time, can apply a controllable number of edit operations to the template.

### Key Designs

1. **Edit-Based Representation**:

    - **Function**: Represent the differences between two protein sequences as a sequence of edit operations.
    - **Mechanism**: Given a template sequence $A$ and a target sequence $B$, the minimal edit distance is computed via sequence alignment (e.g., the Needleman–Wunsch algorithm), yielding a series of operations: substitution, insertion, and deletion. Each operation encodes a position and the corresponding amino acid change.
    - **Design Motivation**: Explicitly representing the mutational process as edit operations allows the model to jointly predict *where* and *what* to mutate, and naturally accommodates sequence length variation through indels—making it far more flexible than fixed-length masking or substitution paradigms.

2. **Evolutionary Trajectory Learning via Flow Matching**:

    - **Function**: Leverage the flow matching framework to learn a continuous edit flow from template to variant.
    - **Mechanism**: Sequence pairs are sampled from evolutionarily related protein families (e.g., distinct sequences within the same UniRef cluster) to construct a probability flow in the edit operation space. Flow matching fits a velocity field in this space; starting from the template and integrating the learned ODE, the model generates continuous and plausible variant sequences.
    - **Design Motivation**: Flow matching offers more stable training and better sample efficiency compared to discrete diffusion. Performing flow matching in the edit operation space (rather than the sequence space) natively supports variable-length sequences and maintains consistency with natural evolutionary trajectories.

3. **Controllable Number of Mutations**:

    - **Function**: Control the edit distance between generated variants and the template at inference time.
    - **Mechanism**: By adjusting the step size or termination time of ODE integration, one can regulate how far the trajectory departs from the template—short distances yield conservative mutations (few substitutions), while long distances produce more aggressive mutations (more substitutions plus indels).
    - **Design Motivation**: Controlling the number of mutations is critical in protein engineering—too few may be insufficient to improve function, while too many may disrupt folding stability. This controllability is a central practical feature of EvoFlows.

### Loss & Training

- **Flow Matching Objective**: Standard conditional flow matching loss minimizing the MSE between the predicted velocity field and the ground-truth velocity.
- **Training Data**: Evolutionarily related protein families are extracted from UniRef (universal protein reference clusters) and OAS (Observed Antibody Space) to construct sequence pairs as training trajectories.
- **Edit Alignment**: A preprocessing step prior to training—optimal edit alignments are computed for each sequence pair and used as target trajectories for flow matching.

## Key Experimental Results

### Experimental Setup
- Evaluation data: Diverse protein families from UniRef and OAS.
- Evaluation paradigm: In silico (computational evaluation); no wet-lab validation.
- Core evaluation dimensions: Naturalness of generated variants (consistency with native protein family distributions) and exploratory range (distance from the template).

### Main Results

| Method | Family Consistency | Template Distance | Indel Support | Position Prediction |
|--------|--------------------|-------------------|---------------|---------------------|
| Autoregressive models | Moderate | Uncontrollable | Limited | N/A (full sequence generation) |
| Masked language models | High (conservative) | Requires pre-specified positions | Not supported | Not supported (positions must be specified) |
| Discrete diffusion models | Moderate–High | Requires pre-specified positions | Not supported | Not supported (positions must be specified) |
| **EvoFlows** | **High** | **Larger range & controllable** | **Supported** | **Automatically predicted** |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---------------|-----------|---------|
| Substitution only (no indels) | Limited exploratory range | Validates the necessity of indel support |
| Varying ODE integration steps | Continuously controllable mutation count | Validates smoothness of the learned velocity field |
| Different protein families | Consistently strong performance | Demonstrates reliable generalization |

### Key Findings

1. **Variants generated by EvoFlows are consistent with native protein family distributions**: The learned edit flow genuinely captures patterns of natural evolution.
2. **Exploratory range substantially exceeds baselines**: The model can generate variants farther from the template while maintaining plausibility, implying a larger functional search space.
3. **Joint prediction of "where" and "what"**: No prior knowledge of mutation sites is required, which is of critical importance for practical protein engineering.

## Highlights & Insights

- **Precise problem formulation**: The paper accurately identifies three core shortcomings of existing protein language models in engineering tasks (no template conditioning, requirement of known positions, no indel support) and addresses all three within a unified framework.
- **Flow matching in edit space**: Transferring flow matching from the sequence space to the edit operation space is an elegant modeling choice—it handles variable-length sequences naturally and carries more intuitive physical meaning.
- **Controllability**: Mutation extent is controlled via ODE integration step size, providing a practical dial that practitioners can adjust between conservative and aggressive regimes.
- **Connecting evolution and generation**: Using evolutionarily related sequences as training signal implicitly constrains the generative process to respect natural selection pressures.

## Limitations & Future Work

1. **In silico evaluation only**: All experiments are computational; wet-lab validation is absent. The actual functional properties of generated variants (enzymatic activity, binding affinity, etc.) remain uncharacterized.
2. **Workshop paper**: As a workshop contribution, the methodological and experimental depth is limited and large-scale evaluation is not yet comprehensive.
3. **Edit alignment quality**: Training depends on sequence-alignment-derived edit operations; alignment quality may affect the learned velocity field. For highly divergent sequence pairs, the optimal edit path is not unique.
4. **Absence of structural information**: The current method operates solely at the sequence level without incorporating three-dimensional structural information. Structural constraints could further improve variant plausibility.
5. **Scalability**: Efficiency and quality for very long sequences (>1000 residues) require further investigation.
6. **Combinatorial effects of multi-step edits**: Trajectories derived from individual sequence pairs may fail to capture the epistatic effects of cooperative mutations that emerge over multiple evolutionary steps.

## Related Work & Insights

- **Relationship to EvoDiff**: EvoDiff employs discrete diffusion to generate directly in sequence space and requires pre-specified mutation positions; EvoFlows performs continuous flow matching in edit space and requires no position priors.
- **Relationship to ESM models**: ESM-based masked language models excel at evaluating mutational effects but are not designed for mutation proposal; EvoFlows directly targets mutation design.
- **Flow matching in biology**: This represents an early application of flow matching to protein sequence modeling, paralleling flow matching methods developed for molecular conformation generation.
- **Implications for drug design**: Controllable sequence editing is particularly valuable in applications such as antibody affinity maturation and enzyme engineering.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Learning Patient-Specific Disease Dynamics with Latent Flow Matching for Longitudinal Imaging Generation](learning_patient-specific_disease_dynamics_with_latent_flow_matching_for_longitu.md)
- [\[ICLR 2026\] How to Make the Most of Your Masked Language Model for Protein Engineering](how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](../../NeurIPS2025/medical_imaging/energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[AAAI 2026\] Ambiguity-aware Truncated Flow Matching for Ambiguous Medical Image Segmentation](../../AAAI2026/medical_imaging/ambiguity-aware_truncated_flow_matching_for_ambiguous_medica.md)
- [\[NeurIPS 2025\] Prior-Guided Flow Matching for Target-Aware Molecule Design with Learnable Atom Number](../../NeurIPS2025/medical_imaging/prior-guided_flow_matching_for_target-aware_molecule_design_with_learnable_atom_.md)

<!-- RELATED:END -->
