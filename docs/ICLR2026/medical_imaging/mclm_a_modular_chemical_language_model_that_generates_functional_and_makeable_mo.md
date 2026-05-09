---
title: >-
  [Paper Note] mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules
description: >-
  [ICLR 2026][Medical Imaging][chemical language model] This paper proposes mCLM (Modular Chemical Language Model), which represents molecules as sequences of synthesizable building blocks, enabling LLMs to generate molecules that simultaneously satisfy pharmacological function and automated synthesis feasibility. mCLM achieves significant improvements in pharmacokinetic and toxicity properties across 430 FDA-approved drugs.
tags:
  - ICLR 2026
  - Medical Imaging
  - chemical language model
  - molecular optimization
  - automated synthesis
  - modular design
  - drug discovery
date: 2026-05-08
content_hash: f6c8e28af529858a
---

# mCLM: A Modular Chemical Language Model that Generates Functional and Makeable Molecules

**Conference**: ICLR 2026  
**arXiv**: [2505.12565](https://arxiv.org/abs/2505.12565)  
**Code**: Available (provided in the paper)  
**Area**: Medicine / Molecular Generation  
**Keywords**: chemical language model, molecular optimization, automated synthesis, modular design, drug discovery

## TL;DR
This paper proposes mCLM (Modular Chemical Language Model), which represents molecules as sequences of synthesizable building blocks, enabling LLMs to generate molecules that simultaneously satisfy pharmacological function and automated synthesis feasibility. mCLM achieves significant improvements in pharmacokinetic and toxicity properties across 430 FDA-approved drugs.

## Background & Motivation

**Background**: LLMs have demonstrated the ability to understand chemical knowledge, yet remain limited in generating functional small molecules—generated molecules are typically incompatible with automated synthesis workflows.

**Limitations of Prior Work**: Existing molecular generation methods rely on atom-level or fragment-level representations (e.g., SMILES). While the generated molecules may satisfy pharmacological objectives, they are rarely manufacturable through automated synthesis pipelines, creating a substantial gap between computational prediction and experimental validation.

**Key Challenge**: Molecular "function" (pharmacological activity, toxicity, etc.) and "makeability" (known synthesis routes, available building blocks) are two independent optimization objectives, and conventional methods focus only on the former.

**Goal**: To enable LLMs to learn a novel molecular language such that generated molecules simultaneously exhibit optimized chemical function and guaranteed synthetic feasibility.

**Key Insight**: Molecules are represented as combinatorial sequences drawn from a predefined building block library, where each block corresponds to a chemically synthesizable fragment compatible with automated synthesis.

**Core Idea**: Replace conventional SMILES with a modular molecular language, allowing LLMs to search for functionally optimal molecules within a constrained synthetic space.

## Method

### Overall Architecture
mCLM decomposes target molecules into building block sequences, with each block drawn from a library compatible with automated synthesis. The input is the modular decomposition of an initial molecule; the output is an optimized module sequence. Training employs paired data (original → optimized) in a seq2seq fine-tuning paradigm.

### Key Designs

1. **Modular Molecular Representation**:

    - Function: Converts molecules from atom-level SMILES to building-block-level sequences.
    - Mechanism: A building block library $\mathcal{B} = \{b_1, ..., b_N\}$ is defined, where each $b_i$ is an automatically synthesizable chemical fragment. A molecule $M$ is decomposed as $M = b_{i_1} \oplus b_{i_2} \oplus ... \oplus b_{i_k}$, where $\oplus$ denotes chemical bond connectivity. Retrosynthetic analysis is used to determine the decomposition.
    - Design Motivation: Block-level representation inherently guarantees synthetic feasibility—as long as each block is synthesizable and connectivity rules are valid, the entire molecule is synthesizable.

2. **Function-Guided Editing Training**:

    - Function: Trains the model to learn how to edit molecular modules to improve pharmacological properties.
    - Mechanism: Paired training data are constructed by decomposing FDA-approved drugs into modules, then using property predictors to evaluate the pharmacological improvement of various module substitutions. Substitutions that preserve scaffold similarity while improving target properties (e.g., AMES toxicity ↓, BBBP permeability ↑, HIA absorption ↑) are selected as positive examples.
    - Design Motivation: De novo generation from scratch is overly challenging; edit-based learning retains the effective structural scaffold of the original molecule while enabling directed optimization.

3. **Generalization Beyond the Training Module Library**:

    - Function: Enables the model to utilize novel building blocks unseen during training.
    - Mechanism: At test time, the module library is expanded to include out-of-distribution (OOD) building blocks. The model generalizes by leveraging learned semantic relationships between modules (chemical property similarity).
    - Design Motivation: A fixed module library restricts the searchable chemical space; generalization to new modules substantially extends practical utility.

### Loss & Training
Standard seq2seq cross-entropy loss, with fine-tuning based on the LLaMA architecture. Training data are filtered using pharmacological property predictors.

## Key Experimental Results

### Main Results

| Model | AMES ↓ | BBBP ↑ | CYP3A4 ↓ | DILI ↓ | HIA ↑ | PGP ↓ | Avg. Improvement |
|------|--------|--------|----------|--------|-------|-------|---------|
| FDA Drugs (Original) | 47.8 | 61.4 | - | - | - | - | - |
| mCLM | Improved | Improved | Improved | Improved | Improved | Improved | Significant |
| MoleculeSTM | - | - | - | - | - | - | Far below mCLM |

### Ablation Study

| Configuration | Avg. Improvement | Notes |
|------|---------|------|
| Full mCLM | Best | Full modularization + function guidance |
| OOD Module Library (122 drugs) | Still effective | Generalizes to unseen modules |
| SMILES Baseline | No synthesis guarantee | May improve function but not manufacturable |
| Without Editing Training | Random substitution | No directed optimization |

### Key Findings
- mCLM improves all 6 pharmacokinetic/toxicity metrics across 430 FDA-approved drugs.
- Remains effective on 122 OOD drugs using only automated-synthesis-compatible modules, demonstrating generalization.
- Substantially outperforms text-based molecular editing baselines such as MoleculeSTM.
- Modular representation makes synthesis routes automatically available, eliminating the computation-to-experiment gap.

## Highlights & Insights
- **Synthetic Feasibility as a Hard Constraint**: Makeability is elevated from a post hoc check to an architectural guarantee—a critical practical requirement for drug discovery pipelines.
- **Modular Language Converts Chemical Search to Token Sequence Optimization**: The search over a continuous chemical space is reformulated as editing over discrete module sequences, which is naturally suited to LLMs.

## Limitations & Future Work
- Coverage of the module library constrains the reachable chemical space.
- The accuracy of property predictors directly affects training data quality.
- Optimization is limited to pharmacokinetics/toxicity; pharmacodynamic properties (e.g., binding affinity) are not addressed.
- The practical success rate of executing the proposed synthesis routes has not been experimentally validated.

## Related Work & Insights
- **vs. MoleculeSTM**: Conventional text–molecule methods operate in SMILES space and provide no guarantee of synthetic feasibility.
- **vs. RetroGPT**: Retrosynthetic planning addresses "how to synthesize a given molecule," whereas mCLM addresses "how to find the optimal molecule within a synthesizable space"—these are complementary directions.

## Rating
- Novelty: ⭐⭐⭐⭐ Novel combination of modular molecular representation and LLMs.
- Experimental Thoroughness: ⭐⭐⭐ Multi-property evaluation, but lacks experimental wet-lab validation.
- Writing Quality: ⭐⭐⭐⭐ Interdisciplinary yet clearly explained.
- Value: ⭐⭐⭐⭐⭐ Directly applicable to AI-assisted drug discovery pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reverse Distillation: Consistently Scaling Protein Language Model Representations](reverse_distillation_consistently_scaling_protein_language_model_representations.md)
- [\[ICLR 2026\] How to Make the Most of Your Masked Language Model for Protein Engineering](how_to_make_the_most_of_your_masked_language_model_for_protein_engineering.md)
- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](../../NeurIPS2025/medical_imaging/mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [\[ICLR 2026\] HistoPrism: Unlocking Functional Pathway Analysis from Pan-Cancer Histology via Gene Expression Prediction](histoprism_unlocking_functional_pathway_analysis_from_pan-cancer_histology_via_g.md)
- [\[ICLR 2026\] AFD-INSTRUCTION: A Comprehensive Antibody Instruction Dataset with Functional Annotations for LLM-Based Understanding and Design](afd-instruction_a_comprehensive_antibody_instruction_dataset_with_functional_ann.md)

</div>

<!-- RELATED:END -->
