---
title: >-
  [Paper Note] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models
description: >-
  [NeurIPS 2025][Computational Biology][Functional Groups] This paper presents FGBench, a dataset comprising 625K molecular property reasoning questions focused on functional group-level reasoning evaluation. Through three…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "Functional Groups"
  - "Molecular Property Reasoning"
  - "Chemistry Benchmark"
  - "Structure-Activity Relationship"
  - "LLM Reasoning"
date: 2026-05-08
content_hash: 204e4b75bfa6d5c4
---

# FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2508.01055](https://arxiv.org/abs/2508.01055)  
**Code**: [https://github.com/xuanliugit/FGBench](https://github.com/xuanliugit/FGBench)  
**Area**: Medical Imaging
**Keywords**: Functional Groups, Molecular Property Reasoning, Chemistry Benchmark, Structure-Activity Relationship, LLM Reasoning

## TL;DR

This paper presents FGBench, a dataset comprising 625K molecular property reasoning questions focused on functional group-level reasoning evaluation. Through three dimensions (single functional group effect, multi-functional group interaction, and molecular comparison), it systematically reveals the severe deficiencies of current LLMs in fine-grained chemical reasoning.

## Background & Motivation

**Background**: LLMs are increasingly applied in chemistry, including molecular property prediction, molecular description generation, and molecular generation. However, existing datasets (e.g., MoleculeNet) primarily focus on molecule-level property prediction, providing molecular-level labels without finer-grained structural information.

**Limitations of Prior Work**: (1) Functional groups are the core structural units governing physicochemical properties (e.g., hydroxyl -OH confers polarity and hydrogen bonding capability; carboxyl -COOH participates in esterification), yet existing databases lack explicit associations between functional groups and molecular properties. (2) Existing functional group annotation methods (e.g., CheckMol) rely on direct pattern matching, which fails when two functional groups overlap and cannot directly identify functional group differences between two molecules. (3) No benchmark exists specifically for evaluating LLM reasoning at the functional group level.

**Key Challenge**: Chemists typically follow a "three-step reasoning" process when predicting molecular properties—identifying structurally similar molecules, observing functional group differences, and inferring property changes based on prior knowledge—yet LLMs lack this fine-grained reasoning capability, and no suitable training or evaluation resources are available.

**Goal**: (1) How to construct a reliable functional group-level molecular pairing dataset? (2) How do LLMs perform on functional group reasoning tasks of varying granularity? (3) Can chemistry-specific fine-tuning improve functional group-level reasoning?

**Key Insight**: The chemist's reasoning process is emulated by constructing QA pairs through comparison of molecularly similar but functionally distinct molecule pairs. A validation-by-reconstruction strategy is adopted to ensure data quality.

**Core Idea**: A validation-by-reconstruction strategy is employed to construct a functional group-level molecular pairing dataset, systematically evaluating LLM reasoning across three dimensions (single/multi-functional group effects and molecular comparison), thereby revealing critical deficiencies in current models.

## Method

### Overall Architecture

The FGBench construction pipeline proceeds as follows: starting from 10 MoleculeNet datasets (ESOL, Lipophilicity, FreeSolv, HIV, BACE, BBBP, Tox21, SIDER, ClinTox, QM9) → SMILES canonicalization → filtering similar molecule pairs based on Tanimoto similarity (512-bit Morgan fingerprints, threshold > 0.7) → extracting functional group differences and precise positions using the AccFG tool → applying the validation-by-reconstruction strategy → generating QA pairs from templates. The final dataset contains 42,967 molecule pairs and 625,936 QA pairs, covering 245 functional group types and 8 molecular properties.

### Key Designs

1. **Validation-by-Reconstruction Strategy**:

    - **Function**: Ensures correctness of functional group difference annotations and chemical validity of molecular editing operations.
    - **Mechanism**: For a molecule pair $(M_1, M_2)$ with functional group differences $(FG_1, FG_2)$, $FG_1$ is removed from $M_1$ and replaced by $FG_2$ at the same position; the reconstructed molecule is verified to match $M_2$ and be chemically valid. This process simultaneously generates the information needed for QA construction: atom-indexed molecular SMILES, atom-indexed functional group SMILES, and the attachment positions of functional groups.
    - **Design Motivation**: Direct functional group pattern matching is error-prone in complex structures (especially in cases of overlap and isomerism). The reconstruction process provides end-to-end correctness verification and generalizes to other molecular property datasets.

2. **Three-Dimensional Reasoning Tasks**:

    - **Function**: Comprehensively evaluates LLM functional group reasoning at different levels of granularity.
    - **Mechanism**:
        - **Dimension 1 – Single Functional Group Effect**: Molecule pairs with only one functional group difference are selected to assess the model's understanding of single functional group effects (e.g., the impact of removing a hydroxyl group on solubility).
        - **Dimension 2 – Multi-Functional Group Interaction**: Molecule pairs with multiple functional group differences are retained to evaluate the model's ability to understand additive/interaction effects of multiple functional groups.
        - **Dimension 3 – Molecular Comparison**: Two complete molecules are directly provided for property comparison without functional group editing information, serving as a reference baseline.
    - **Design Motivation**: Tasks are structured from simple to complex. Dimension 3 as a reference reveals whether models rely solely on memorized molecule-level knowledge rather than functional group-level reasoning.

3. **Boolean and Numerical Dual-Category QA Pairs**:

    - **Function**: Separately evaluates qualitative judgment and quantitative reasoning capabilities.
    - **Mechanism**: Boolean QA queries whether a functional group modification changes the direction of a property (e.g., from active to inactive); numerical QA queries the exact magnitude of change (e.g., solubility difference). Each QA includes atom-indexed SMILES, property name and initial property value, and detailed functional group editing instructions.
    - **Design Motivation**: Trend judgment and precise prediction represent two fundamentally distinct reasoning capabilities.

### Evaluation Design

A subset of 7,146 samples is curated from 625K QA pairs for evaluation (up to 25 pairs per task), with balanced distribution across dimensions and categories. Nine models are evaluated (2 closed-source, 4 general open-source, 3 chemistry-specific); classification tasks use ACC and regression tasks use RMSE. Chemistry-specific models use a dedicated answer parser to accommodate their limited instruction-following capability.

## Key Experimental Results

### Main Results

| Model | Single-Bool | Inter-Bool | Comp-Bool | Single-Value RMSE |
|-------|------------|-----------|----------|------------------|
| o3-mini | **0.687** | **0.693** | **0.703** | 101.886 |
| GPT-4o | 0.667 | 0.488 | 0.614 | 77.990 |
| Llama-3.1 70B | 0.683 | 0.530 | 0.456 | 84.119 |
| Llama-3.1 8B | 0.548 | 0.547 | 0.474 | 162.351 |
| Qwen2.5-7B | 0.590 | 0.396 | 0.664 | 63.511 |
| ChemLLM-7B | 0.233 | 0.235 | 0.250 | 209.584 |
| nach0 | 0.606 | 0.543 | 0.041 | 104.534 |
| LlaSMol-Mistral-7B | 0.387 | 0.298 | 0.239 | 266.720 |

### Ablation Study

| Analysis Dimension | Finding |
|-------------------|---------|
| Single FG → Multi-FG Interaction | Most models show significant accuracy drops (GPT-4o: 0.667→0.488; Llama 70B: 0.683→0.530) |
| Chemistry-specific vs. General Models | ChemLLM-7B (0.233) substantially underperforms same-scale Llama-3.1 8B (0.548) |
| nach0 Bias Analysis | 97.7% of Single-Bool predictions are False; high ACC results from class imbalance bias |
| Qwen2.5 Anomaly | Comp-Bool 0.664 vs. Inter-Bool 0.396, suggesting possible prior exposure to MoleculeNet molecules but lack of FG knowledge |
| Reasoning Model Advantage | o3-mini achieves best performance on 4/6 tasks; reasoning capability is important for chemical reasoning |

### Key Findings

- **LLMs severely underperform at functional group-level reasoning**: The best model, o3-mini, achieves only 68.7% on Single-Bool, with further degradation on multi-functional group interaction tasks.
- **Chemistry-specific fine-tuning is counterproductive**: ChemLLM, despite extensive training on molecule-level data, substantially underperforms general-purpose models on FGBench—molecule-level knowledge does not transfer to the functional group level.
- **Reasoning capability is critical**: The reasoning training of o3-mini enables consistent superiority in chemical property reasoning, particularly on multi-FG interaction tasks requiring multi-step inference.
- **Failure case analysis**: When reasoning about logD for benzonitrile vs. benzene, o3-mini correctly identifies polarity (partially correct reasoning) but reverses the direction (factual error), suggesting that retrieval augmentation may be a promising improvement direction.

## Highlights & Insights

- **Filling an evaluation gap**: The first dataset specifically targeting functional group-level molecular property reasoning (625K QA pairs, 245 functional group types), with excellent scale and comprehensiveness.
- **Generalizability of the validation-by-reconstruction strategy**: The data construction pipeline can be directly applied to other molecular property datasets.
- **Revealing the generalization dilemma of chemistry-tuned LLMs**: Molecule-level training not only fails to transfer to the functional group level but may also degrade the general reasoning capability of the base model.
- **Multimodal potential of fine-grained annotations**: Precise positional annotations of functional groups enable the dataset to be directly used for molecular graph–text multimodal learning.

## Limitations & Future Work

- Positional isomerism (ortho/meta/para substitution), carbon chain isomerism, and stereoisomerism are not considered.
- Only 10 datasets from MoleculeNet are used, limiting the variety of properties covered.
- Evaluation is conducted in a zero-shot setting; few-shot or fine-tuned performance is not explored.
- 3D molecular structure information is absent; certain properties (e.g., the effect of chirality on pharmacological activity) require 3D perception.
- SMILES as an input format is inherently unintuitive, which may constrain LLM chemical reasoning capability.

## Related Work & Insights

- MoleculeNet is the classical molecular property benchmark but provides only molecule-level labels; FGBench extends evaluation downward to the functional group level.
- The AccFG tool addresses the overlapping functional group annotation problem and is a key dependency for FGBench construction.
- SciBench and MolPuzzle evaluate chemical mathematical reasoning and molecular structure elucidation, respectively; FGBench focuses on structure–property relationship reasoning.
- Functional groups have been widely used in molecular representation pre-training (Li et al. 2023, Nguyen et al. 2024), but as tokens rather than as targets for reasoning evaluation.

## Rating

⭐⭐⭐⭐ (4/5)

The data construction methodology is rigorous (validation-by-reconstruction strategy), the task design is well-motivated (three dimensions × two categories), and the scale is substantial (625K QA pairs). The paper surfaces an important finding regarding the generalization dilemma of chemistry-tuned LLMs. As a benchmark paper, model evaluation is comprehensive. The primary limitation is that this represents a relatively conventional benchmark contribution with limited methodological novelty.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models](mol-llama_towards_general_understanding_of_molecules_in_large_molecular_language.md)
- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[ACL 2026\] BioTool: A Comprehensive Tool-Calling Dataset for Enhancing Biomedical Capabilities of Large Language Models](../../ACL2026/computational_biology/biotool_a_comprehensive_tool-calling_dataset_for_enhancing_biomedical_capabiliti.md)
- [\[ICLR 2026\] Tracing Pharmacological Knowledge in Large Language Models](../../ICLR2026/computational_biology/tracing_pharmacological_knowledge_in_large_language_models.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)

</div>

<!-- RELATED:END -->
