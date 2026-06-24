---
title: >-
  [Paper Note] Structural Reasoning Improves Molecular Understanding of LLM
description: >-
  [LLM (Other)][molecular reasoning] This paper proposes the Molecular Structural Reasoning (MSR) framework. By explicitly incorporating six key structural elements of molecules (molecular formula, longest carbon chain, aromatic rings, ring compounds, functional groups, and chiral centers) as intermediate reasoning steps, it significantly improves LLM performance on molecular understanding tasks.
tags:
  - "LLM (Other)"
  - "molecular reasoning"
  - "structural information"
  - "chain-of-thought"
  - "SMILES"
  - "molecule-to-text"
date: 2026-05-08
content_hash: 6c35d6aa9f0b24ac
---

# Structural Reasoning Improves Molecular Understanding of LLM

| Conference/Journal | Year | Paper Link | Code |
|----------|------|---------|------|
| ACL 2025 | 2025 | [arXiv 2410.05610](https://arxiv.org/abs/2410.05610) | - |

**Area**: LLMs + Chemistry / Molecular Understanding  
**Keywords**: molecular reasoning, structural information, chain-of-thought, SMILES, molecule-to-text

## TL;DR

This paper proposes the Molecular Structural Reasoning (MSR) framework. By explicitly incorporating six key structural elements of molecules (molecular formula, longest carbon chain, aromatic rings, ring compounds, functional groups, and chiral centers) as intermediate reasoning steps, it significantly improves LLM performance on molecular understanding tasks.

## Background & Motivation

**Problem Definition**: LLMs are increasingly applied in chemistry (molecular description generation, retrosynthesis, text-to-molecule, etc.), but even state-of-the-art LLMs (GPT-4o, Llama3) **cannot accurately infer key structural information of molecules**. For example, the accuracy of counting aromatic rings is only about 50%-75%.

**Why Structural Information Matters**:
- Molecular properties (toxicity, solubility, boiling point, etc.) **heavily depend on structural features**.
- Chemists reason about molecules **starting from structures**: they first identify rings and carbon chain backbones, and then locate functional groups.
- Injecting accurate structural information into LLMs can improve the correctness of molecular generation.

**Motivation**: To design a framework that allows LLMs to "sketch" molecular structures before answering, resembling a chain-of-thought approach specifically tailored for chemical structural reasoning.

## Method

### Overall Architecture

MSR consists of two modules: the **Reasoning Module** (generating structural information) and the **Answering Module** (generating the final answer based on the original input and structural information). Depending on whether the molecule is provided as an input, it operates in two modes:

- **Analytic Reasoning**: Input contains the molecule $\rightarrow$ Use external tools (RDKit) to extract precise structural information $\rightarrow$ Fine-tune the answering module.
- **Synthetic Reasoning**: Input does not contain the molecule (e.g., text descriptions) $\rightarrow$ Fine-tune the reasoning module to infer structural information from text $\rightarrow$ Fine-tune the answering module to generate the molecule.

### Key Designs

- **Six Key Structural Elements**: Mimicking the reasoning process of chemists, defined from coarse to fine:
  1. Molecular formula (atom types and counts)
  2. Longest carbon chain length (backbone information)
  3. Number of aromatic rings (stability and electronic properties)
  4. Ring compounds (ring system types)
  5. Functional groups (chemical reactivities)
  6. Chiral centers (stereochemistry)
- **Match Ratio Reject Sampling**: In synthetic reasoning, $k$ candidate molecules are first generated using beam search. The structural match ratio between each candidate and the MSR is calculated, and the molecule with the highest match ratio is selected as the output.
- **Reliability Filtering**: In synthetic reasoning, only structural components with sufficiently high inference accuracy are retained, while unreliable inference results are discarded.

### Loss & Training

Standard sequence-to-sequence cross-entropy loss, where MSR is added to the training data as an additional input (for analytic reasoning) or an intermediate output (for the reasoning module of synthetic reasoning).

## Experiments

### Main Results 1: Molecule-to-Text

**L+M Dataset**:

| Model | BLEU-2 | BLEU-4 | ROUGE-L | METEOR |
|------|--------|--------|---------|--------|
| MolT5-base | 0.738 | 0.535 | 0.539 | 0.718 |
| MolT5-base + MSR | **0.805** | **0.592** | **0.642** | **0.822** |
| MolT5-large | 0.769 | 0.556 | 0.557 | 0.743 |
| MolT5-large + MSR | **0.832** | **0.622** | **0.691** | **0.878** |

**ChEBI-20 Dataset** (with general LLMs):

| Model | BLEU-4 | ROUGE-L | METEOR |
|------|--------|---------|--------|
| GPT-4o | 0.128 | 0.307 | 0.291 |
| GPT-4o + MSR | **0.174** | **0.313** | **0.341** |
| ChemT5-base + MSR | **0.560** | **0.626** | **0.657** |
| BioT5 (SOTA baseline) | 0.556 | 0.633 | 0.656 |

### Main Results 2: Text-to-Molecule

| Model | BLEU | Exact Match | MACCS FTS | Morgan FTS | FCD↓ |
|------|------|------------|-----------|-----------|------|
| MolT5-large | 0.564 | 0.000 | 0.757 | 0.395 | 17.50 |
| MolT5-large + MSR | **0.710** | **0.111** | **0.837** | **0.560** | **1.54** |
| MolT5-base | 0.684 | 0.000 | 0.760 | 0.475 | NaN |
| MolT5-base + MSR | **0.706** | **0.052** | **0.825** | **0.548** | **1.45** |

### Ablation Study

**Inference Module Accuracy (Synthetic Mode)**:

| Component | MolT5-base (L+M) | MolT5-base (ChEBI) | GPT-4o | Llama3 |
|------|-------------------|--------------------|----|--------|
| Aromatic rings | 0.825 | 0.926 | 0.718 | 0.593 |
| Molecular formula | 0.426 | 0.458 | 0.298 | 0.084 |
| Functional groups | 0.889 | 0.957 | 0.298 | 0.137 |

### Key Findings

1. MSR consistently improves performance across **all models and tasks**, validating the universality of the framework.
2. **Chemistry-specific LLM + MSR** can outperform baselines pre-trained on much larger datasets (e.g., ChemT5-base + MSR $\approx$ BioT5).
3. General LLMs (GPT-4o, Llama3) exhibit much lower structural reasoning accuracy than fine-tuned chemistry LLMs, explaining their performance bottleneck on chemistry tasks.
4. **Reject sampling in synthetic reasoning** effectively enhances the alignment between the generated molecules and MSR.
5. MSR enables models to **reach good performance faster** (improving training efficiency).

## Highlights & Insights

- Accurately diagnoses the limitations of LLMs in understanding molecular structures and proposes targeted solutions.
- The dual-mode design of analytic/synthetic reasoning elegantly covers both scenarios where molecules serve as either inputs or outputs.
- Match ratio reject sampling leverages the deterministic nature of molecular structural information, cleverly integrating reasoning with verification.
- Comprehensive experimental coverage: 3 tasks, 3 datasets, chemistry LLMs + general LLMs.

## Limitations & Future Work

- The six structural elements are manually defined, which may not cover all crucial chemical characteristics.
- There is still significant room for improvement in the accuracy of the reasoning module for synthetic reasoning (e.g., molecular formula accuracy is only 42%-47%).
- Dependence on external tools (RDKit) subjects analytic reasoning to the limitations of the tool.
- Evaluation is only conducted on English chemical text, leaving cross-lingual applicability unknown.
- Reject sampling increases computational overhead during inference.

## Related Work & Insights

- **Chemistry LLMs**: MolT5 (Edwards et al., 2022), ChemT5 (Christofidellis et al., 2023), BioT5 (Pei et al., 2023).
- **Chain-of-Thought Distillation**: Ho et al. (2023), Magister et al. (2023) distill CoT into smaller models.
- **Molecular Representations**: SMILES (Weininger, 1988), SELFIES.

## Rating

| Dimension | Score (1-10) |
|------|-----------|
| Novelty | 8 |
| Practicality | 7 |
| Experimental Thoroughness | 9 |
| Writing Quality | 8 |
| Overall Score | 8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Boosting LLM's Molecular Structure Elucidation with Knowledge Enhanced Tree Search Reasoning](boosting_llms_molecular_structure_elucidation_with_knowledge_enhanced_tree_searc.md)
- [\[ACL 2025\] GAMEBoT: Transparent Assessment of LLM Reasoning in Games](gamebot_transparent_assessment_of_llm_reasoning_in_games.md)
- [\[ACL 2025\] Reason from Future: Reverse Thought Chain Enhances LLM Reasoning](reason_from_future_reverse_thought_chain_enhances_llm_reasoning.md)
- [\[ACL 2025\] Understanding Silent Data Corruption in LLM Training](understanding_silent_data_corruption_in_llm_training.md)
- [\[ACL 2025\] Exploring Explanations Improves the Robustness of In-Context Learning](exploring_explanations_improves_the_robustness_of_in-context_learning.md)

</div>

<!-- RELATED:END -->
