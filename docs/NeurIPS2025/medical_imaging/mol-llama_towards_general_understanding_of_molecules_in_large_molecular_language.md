---
title: >-
  [Paper Note] Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models
description: >-
  [NeurIPS 2025][Medical Imaging][molecular language model] This paper proposes Mol-LLaMA, a large molecular language model for general molecular understanding. By designing three types of instruction data and a 2D-3D molecular representation fusion module, Mol-LLaMA surpasses GPT-4o in molecular feature understanding while exhibiting interpretability and reasoning capabilities.
tags:
  - NeurIPS 2025
  - Medical Imaging
  - molecular language model
  - multimodal instruction tuning
  - 2D-3D molecular representation fusion
  - drug discovery
  - molecular reasoning
date: 2026-05-08
content_hash: 7546e639666d270c
---

# Mol-LLaMA: Towards General Understanding of Molecules in Large Molecular Language Models

**Conference**: NeurIPS 2025
**arXiv**: [2502.13449](https://arxiv.org/abs/2502.13449)
**Code**: [Project Page](https://mol-llama.github.io/)
**Area**: Medical Imaging / AI for Science
**Keywords**: molecular language model, multimodal instruction tuning, 2D-3D molecular representation fusion, drug discovery, molecular reasoning

## TL;DR

This paper proposes Mol-LLaMA, a large molecular language model for general molecular understanding. By designing three types of instruction data and a 2D-3D molecular representation fusion module, Mol-LLaMA surpasses GPT-4o in molecular feature understanding while exhibiting interpretability and reasoning capabilities.

## Background & Motivation

Molecular understanding is fundamental to chemistry and biology and is critical for drug discovery. Existing molecular LLMs face two core challenges:

**Insufficient knowledge and reasoning capability**: Existing models rely on task-specific public databases (e.g., brief descriptions from PubChem), resulting in narrow knowledge coverage, lack of causal explanations for molecular features, and frequent errors in predicting molecular classes and properties.

**Limited structural understanding**: Models use only a single type of molecular encoder (2D or 3D graph), unable to simultaneously capture bond information/connectivity (2D advantage) and spatial arrangement/surface area/volume (3D advantage).

Through a case study, the authors demonstrate that GPT-4o and 3D-MoLM misclassify bromazepam (a benzodiazepine) as a quinazoline, while Mol-LLaMA correctly identifies its benzodiazepine scaffold and explains its interaction mechanism with GABA-A receptors.

## Method

### Overall Architecture

Mol-LLaMA consists of four components:
1. Molecular encoders (MoleculeSTM-2D + UniMol-3D) → capture complementary molecular representations
2. 2D-3D Blending Module → integrates complementary information via cross-attention
3. Q-Former projector → maps unified molecular representations into LLM space
4. LLM backbone (Llama-2-7B-Chat or Llama-3.1-8B-Instruct) + LoRA

Training proceeds in two stages: molecular representation learning (aligning molecular embeddings with text) → end-to-end instruction tuning.

### Key Designs

1. **Three-Type Instruction Data Design (Mol-LLaMA-Instruct)**

   Core insight: Molecular features exhibit a hierarchical relationship — structure determines chemical properties, and chemical + structural properties jointly determine biological features. Based on this, three data types are designed:

   - **Detailed structural descriptions (S)**: GPT-4o is prompted with IUPAC names to generate detailed descriptions of functional groups and connectivity, building the foundation for structural understanding.
   - **Structure-to-feature relationship explanations (S2F)**: Directly links structural information to chemical/biological features, enabling the model to learn causal relationships (e.g., "why does this structure lead to a specific activity"), naturally endowing the model with reasoning and interpretability.
   - **Comprehensive conversations (Conv.)**: Progressively deepens from structure → chemistry → biology in a hierarchical manner, cultivating the ability to handle diverse user queries and step-by-step reasoning.

   Data generation pipeline: GPT-4o is used with IUPAC names and PubChem descriptions to generate data, followed by GPT-4o-as-judge filtering to remove factual errors, yielding approximately 284K instruction samples.

2. **2D-3D Blending Module**

   Core mechanism: The 2D encoder (MoleculeSTM) excels at modeling bond information and connectivity, while the 3D encoder (UniMol) excels at capturing atomic spatial arrangement. The two encoders provide complementary information but encode independently, requiring effective fusion.

   Specific design:
   - Graph embeddings and node embeddings from each encoder are concatenated.
   - Self-attention is applied first, followed by cross-attention to integrate complementary information.
   - The fused 2D and 3D embeddings are concatenated and fed into the Q-Former projector.

   Ablation experiments confirm that simple concatenation (Concat), while able to detect the presence of atoms and functional groups, fails to correctly predict connectivity, whereas the Blending Module accurately predicts molecular structure.

3. **Q-Former Projector**

   - A SciBERT-initialized Q-Former models the interaction between learnable query tokens and unified molecular representations via cross-attention.
   - The cross-attention mechanism in Q-Former naturally ensures permutation invariance, making it well-suited for graph-structured data.

### Loss & Training

- **Stage 1 (Molecular Representation Learning)**: The 2D/3D encoders are frozen; the Blending Module and Q-Former are trained using three objectives: molecule-text contrastive learning, molecule-text matching, and molecule-anchored text generation, with IUPAC names as text anchors.
- **Stage 2 (End-to-End Instruction Tuning)**: The encoders are frozen; the Blending Module, Q-Former, and LLM (LoRA) are jointly trained on the Mol-LLaMA-Instruct dataset.

## Key Experimental Results

### Main Results

**General molecular understanding (GPT-4o evaluation, relative scores; >1 indicates better than GPT-4o)**:

| Model | Structure Overall | Chemistry Overall | Biology Overall | Note |
|-------|------------------|------------------|-----------------|------|
| Mol-LLaMA (Llama3.1) | 1.125 | 1.251 | 1.744 | Comprehensively surpasses GPT-4o |
| Mol-LLaMA (Llama2) | 1.098 | 1.232 | 1.631 | Also surpasses GPT-4o |
| 3D-MoLM† (Llama3.1) | 0.749 | 0.875 | 1.191 | Only biology approaches GPT-4o |
| LLaMo† (Llama3.1) | 0.442 | 0.425 | 0.705 | Far below GPT-4o |
| GPT-4o | 1.000 | 1.000 | 1.000 | Baseline |

**MoleculeQA molecular understanding benchmark**:

| Model | Structure | Source | Property | Application | Total |
|-------|-----------|--------|----------|-------------|-------|
| Mol-LLaMA (Llama3.1) | **77.81** | **75.50** | **49.63** | **49.30** | **70.76** |
| 3D-MoLM† | 76.31 | 73.64 | 47.93 | 47.33 | 69.10 |
| MolCA-1.3B | 71.12 | 70.98 | 47.81 | 43.17 | 64.79 |

**PAMPA zero-shot molecular property prediction**:

| Model | Default Acc | CoT Acc | w/ Task Info Acc | Fidelity | Helpfulness |
|-------|------------|---------|-----------------|----------|-------------|
| Mol-LLaMA (Llama3.1) | 63.55% | 64.37% | **72.48%** | 0.927 | 0.966 |
| GPT-4o | 48.65% | 58.23% | 47.17% | - | - |

### Ablation Study

| Configuration | Structure Overall | Chemistry Overall | Biology Overall | Note |
|---------------|------------------|------------------|-----------------|------|
| S (structure only) | 1.119 | 1.166 | 1.328 | Best structural understanding |
| S+S2F | 1.172 | **1.285** | **1.754** | Best chemistry/biology |
| Conv. (conversation only) | 1.166 | 0.689 | 0.887 | Diverse but lacks depth |
| Full (complete data) | 1.125 | 1.251 | 1.744 | Best overall balance |
| 2D only | 0.907 | 1.137 | 1.526 | Weak structural understanding |
| 3D only | 1.071 | 1.195 | 1.632 | Good but not optimal |
| 2D+3D Concat | 1.037 | 1.210 | 1.741 | Structural understanding degrades |
| 2D+3D Blended | **1.125** | **1.251** | **1.744** | Fusion module is effective |

### Key Findings

- Mol-LLaMA comprehensively surpasses GPT-4o across all five evaluation dimensions (all Overall scores >1.0).
- Structure-to-feature relationship explanation (S2F) data contributes most to understanding chemical and biological features.
- The Blending Module resolves the "structural understanding degradation" caused by simple concatenation.
- The model is robust to diverse molecular conformations (performance is consistent across Diverse vs. Fixed conformations).
- CoT and task-information prompting consistently improve performance, indicating genuine reasoning capability.

## Highlights & Insights

- **Hierarchical data design philosophy**: Inspired by the hierarchical relationship among molecular features (structure → chemistry → biology), the three-type progressive instruction data design is more effective than simply aggregating task-specific data.
- **Surpassing GPT-4o**: In the specialized domain of molecular understanding, carefully designed domain data and modality fusion enable a 7B/8B model to comprehensively outperform GPT-4o.
- **Practical LLM-as-judge quality control**: GPT-4o is used to filter factual errors from its own generated data, ensuring data quality.
- **Illustrative 2D-3D complementarity cases**: Ablations include concrete examples where the 2D encoder misses atoms, the 3D encoder confuses bond types, and simple concatenation misidentifies connectivity.

## Limitations & Future Work

- Instruction data is generated by GPT-4o, potentially introducing its biases and knowledge boundaries.
- Evaluation also primarily relies on GPT-4o-as-judge, creating an overlap between the evaluator and the data generator.
- Training and evaluation are limited to PubChem molecules, without coverage of macromolecules (proteins, nucleic acids).
- No comparison is made with the latest molecular foundation models (e.g., MoLFormer).
- 3D conformations are generated via RDKit/OpenBabel rather than experimental structures, potentially introducing noise.

## Related Work & Insights

- **3D-MoLM**: 3D molecular encoder + LLM; the primary comparison baseline in this work.
- **LLaMo**: 2D molecular encoder + LLM; exhibits insufficient structural understanding.
- **MoleculeSTM / UniMol**: Adopted in this work as the 2D and 3D molecular encoders, respectively.
- Insight: Careful design of domain-specific data is more impactful than architectural innovation; multi-representation fusion requires explicit cross-attention mechanisms, while simple concatenation can be detrimental.

## Rating

- Novelty: ⭐⭐⭐⭐ The three-type hierarchical instruction data design is novel; the fusion module is simple yet effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive qualitative and quantitative analysis with detailed ablations (data types, encoders, conformation robustness).
- Writing Quality: ⭐⭐⭐⭐ Compelling case studies and rich tables.
- Value: ⭐⭐⭐⭐ A general-purpose assistant for practical molecular analysis; surpassing GPT-4o demonstrates strong application potential.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models](fgbench_a_dataset_and_benchmark_for_molecular_property_reasoning_at_functional_g.md)
- [\[NeurIPS 2025\] EDBench: Large-Scale Electron Density Data for Molecular Modeling](edbench_large-scale_electron_density_data_for_molecular_modeling.md)
- [\[NeurIPS 2025\] EndoBench: A Comprehensive Evaluation of Multi-Modal Large Language Models for Endoscopy Analysis](endobench_a_comprehensive_evaluation_of_multi-modal_large_language_models_for_en.md)
- [\[NeurIPS 2025\] Position: Thematic Analysis of Unstructured Clinical Transcripts with Large Language Models](position_thematic_analysis_of_unstructured_clinical_transcripts_with_large_langu.md)
- [\[NeurIPS 2025\] Consistent Sampling and Simulation: Molecular Dynamics with Energy-Based Diffusion Models](consistent_sampling_and_simulation_molecular_dynamics_with_energy-based_diffusio.md)

<!-- RELATED:END -->
