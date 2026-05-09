---
title: >-
  [Paper Note] MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation
description: >-
  [ICLR 2026][Graph Learning][molecular recognition] This paper introduces MolLangBench, a benchmark constructed via automated tools and expert annotation to provide high-quality, unambiguous evaluation datasets for the molecule-language interface. It covers three task types (recognition / editing / generation) and three modalities (SMILES / image / graph), evaluates 16+ commercial LLMs and 5 chemistry-specific models, and reveals that even GPT-5 falls significantly short on basic molecular operations (generation accuracy only 43%).
tags:
  - ICLR 2026
  - Graph Learning
  - molecular recognition
  - molecule editing
  - molecule generation
  - molecule-language alignment
  - benchmark
date: 2026-05-08
content_hash: 99ed43011555f317
---

# MolLangBench: A Comprehensive Benchmark for Language-Prompted Molecular Structure Recognition, Editing, and Generation

**Conference**: ICLR 2026
**arXiv**: [2505.15054](https://arxiv.org/abs/2505.15054)
**Code**: [GitHub](https://github.com/TheLuoFengLab/MolLangBench) / [HuggingFace](https://huggingface.co/datasets/ChemFM/MolLangBench)
**Area**: AI for Chemistry
**Keywords**: molecular recognition, molecule editing, molecule generation, molecule-language alignment, benchmark

## TL;DR

This paper introduces MolLangBench, a benchmark constructed via automated tools and expert annotation to provide high-quality, unambiguous evaluation datasets for the molecule-language interface. It covers three task types (recognition / editing / generation) and three modalities (SMILES / image / graph), evaluates 16+ commercial LLMs and 5 chemistry-specific models, and reveals that even GPT-5 falls significantly short on basic molecular operations (generation accuracy only 43%).

## Background & Motivation

**Background**: Recent work has extensively explored molecule-language alignment, but these approaches typically target downstream chemical tasks (e.g., property prediction, reaction prediction) directly, bypassing fundamental structure-level capabilities. Analogous to the success of vision-language modeling—where VLMs align text with visually observable content—current molecular-language models attempt to align symbolic molecular structures with unobservable chemical properties, a mismatch that makes alignment considerably more difficult.

**Limitations of Prior Work**: (1) No benchmark systematically evaluates AI capabilities in basic molecular structure operations (recognition, editing, generation); (2) existing molecular benchmarks focus on high-level tasks (drug design, property prediction) while neglecting the prerequisite—whether models truly "understand" molecular structure; (3) existing datasets vary in quality and may contain ambiguities and uncertainties.

**Key Challenge**: If AI cannot reliably perform basic molecular structure recognition and manipulation, more complex chemical reasoning tasks (drug discovery, materials design) cannot be trusted. The chemist's workflow invariably begins with structural understanding.

**Goal**: To provide the first systematic, high-quality evaluation tool for fundamental molecule-language capabilities.

**Key Insight**: Grounded in the actual chemist workflow—first recognize structure, then manipulate structure, then generate structure—yielding a three-tier progressive task design.

**Core Idea**: Evaluate AI's fundamental molecular structure capabilities using deterministic, unambiguous, high-quality data, thereby exposing deficiencies in current models.

## Method

### Overall Architecture

MolLangBench evaluates three core capabilities of increasing difficulty:
1. **Molecular Structure Recognition**: Given a molecule, answer structural questions in natural language (neighboring atoms, bond types, functional groups, ring structures, stereochemistry)
2. **Molecular Editing**: Modify a given molecular structure according to natural language instructions
3. **Molecular Generation**: Generate a molecule from scratch based solely on a textual description

Three molecular representations are supported: SMILES strings, molecular images (2D structure diagrams), and molecular graphs.

### Key Designs

1. **Automated Construction Pipeline for Recognition Tasks**:
    - Function: Guarantee deterministic and unambiguous answers
    - Mechanism: RDKit is used to automatically compute ground truth (single-hop neighbors, bond types, functional group identification, ring structures, stereochemistry, etc.), covering three categories: local topology, functional groups, and stereochemistry
    - Design Motivation: Automated tools ensure each question has a unique, definitive answer, avoiding subjectivity introduced by human annotation
    - Sampling Strategy: Label-balanced sampling from 10,000 candidate molecules, with deliberate selection of harder examples (e.g., cases where bonded atoms are non-adjacent in SMILES)

2. **Expert Annotation Pipeline for Editing and Generation Tasks**:
    - Function: Construct precise mappings between natural language instructions and molecular structures
    - Mechanism: A three-stage pipeline—(1) annotators with chemistry backgrounds draft instructions/descriptions; (2) a second annotator conducts peer review with iterative revision until consensus is reached; (3) two independent validators reconstruct the molecular structure from text alone, and a sample is accepted only if both succeed
    - Design Motivation: Accurate molecular reconstruction from text alone constitutes the strongest validation of instruction/description unambiguity
    - Effort: Over 500 hours of expert annotation and validation

3. **Anti-Leakage and Robustness Design**:
    - Function: Ensure evaluation results are not affected by data leakage or memorization
    - Mechanism: (1) Unique hash canary strings to detect leakage; (2) SMILES enumeration augmentation (enumerating from different starting atoms) to test robustness—editing accuracy across 5 different augmentations is $0.773\pm0.027$, indicating high consistency

### Loss & Training

MolLangBench does not involve model training. Evaluation metrics: exact-match accuracy for recognition and editing tasks; accuracy (whether the generated molecule satisfies all specified conditions) for generation tasks, supplemented by Tanimoto similarity (molecular fingerprints) and pass@k metrics.

## Key Experimental Results

### Main Results

**Evaluation of 16 commercial LLMs** (SMILES modality, core test set):

| Model | Recognition Acc. | Editing (Valid/Sim./Acc.) | Generation (Valid/Sim./Acc.) |
|-------|-----------------|--------------------------|------------------------------|
| GPT-5 | 0.862 | 0.960/0.923/**0.855** | 0.920/0.741/**0.430** |
| o3 | 0.918 | 0.945/0.903/0.785 | 0.670/0.546/0.290 |
| o4-mini | 0.872 | 0.930/0.885/0.740 | 0.820/0.651/0.350 |
| Gemini-2.5-Pro | 0.852 | 0.930/0.881/0.745 | 0.865/0.737/0.430 |
| Claude-Opus-4.1 | 0.814 | 0.950/0.884/0.705 | 0.920/0.725/0.330 |
| Llama-4-Maverick | 0.614 | 0.895/0.772/0.545 | 0.875/0.511/0.115 |
| Qwen3-Max | 0.486 | 0.690/0.561/0.360 | 0.465/0.104/0.000 |

### Ablation Study

**Chemistry-specific models vs. general-purpose LLMs**:

| Model Type | Recognition | Editing Acc. | Generation Acc. |
|-----------|-------------|-------------|----------------|
| ChemDFM-13B | 0.300 | 0.025 | 0.000 |
| Galactica-120B | 0.290 | 0.040 | 0.000 |
| HIGHT (graph-language) | 0.127 | 0.000 | 0.000 |
| GPT-4o (general) | 0.593 | 0.525 | 0.115 |

**SMILES vs. SELFIES representation** (o3 model):

| Representation | Recognition | Editing Acc. | Generation Acc. |
|---------------|-------------|-------------|----------------|
| SMILES | 0.918 | 0.785 | 0.290 |
| SELFIES | 0.528 | 0.195 | 0.000 |

**pass@k results** (o3 model):

| Task | pass@1 | pass@3 | pass@5 |
|------|--------|--------|--------|
| Editing (core) | 0.785 | 0.856 | 0.900 |
| Generation (core) | 0.290 | 0.485 | 0.545 |

### Key Findings

1. **Generation is extremely challenging**: The strongest model, GPT-5, achieves only 43.0%, and pass@5 reaches only 54.5%—current AI is severely limited in constructing molecular structures from textual descriptions.
2. **Six error categories for the o3 model**: invalid SMILES syntax (11/66 editing/generation cases), stereochemistry errors (9/15), chain length errors (4/8), substituent misplacement (13/42), ring structure errors (10/23), and extra/missing groups (1/3)—atom counting and enumeration issues arising from BPE tokenization are one root cause.
3. **SELFIES is far inferior to SMILES**: Under the same o3 model, generation accuracy with SELFIES drops to 0%—due to the scarcity of SELFIES in LLM training data.
4. **Chemistry-specific models lag comprehensively**: ChemDFM, Galactica, and others fall far below general-purpose GPT-4o, indicating that scale effects outweigh domain-specific knowledge.
5. **Structural understanding improves downstream reasoning**: GPT-4o achieves approximately 5% improvement on property prediction when first describing structure before predicting properties compared to direct prediction (BBBP: 0.551→0.603, BACE: 0.583→0.632).

## Highlights & Insights

- **Fills an important gap**: The first comprehensive benchmark derived from the chemist's workflow that systematically evaluates AI's fundamental molecular structure capabilities.
- **High-quality data construction**: 500+ hours of expert annotation and a three-stage verification process guarantee unambiguity—this itself constitutes a core contribution.
- **Reveals a path deviation**: Current molecular-language research may be heading in the wrong direction by skipping structural understanding and directly targeting property prediction, analogous to a VLM performing reasoning without recognizing objects in images.
- **Accompanying training data**: MolLangData provides large-scale training data, forming a complete ecosystem.

## Limitations & Future Work

- The editing/generation core sets each contain only 200 samples, which is relatively small (though the 500-hour annotation cost limits expansion).
- Molecules are restricted to fewer than 40 heavy atoms (covering 93% of UniChem molecules); biomacromolecules are not addressed.
- Evaluation relies on the Mathpix API to convert generated images back to SMILES, introducing an additional source of error.
- Evaluation is primarily conducted on OpenAI models; coverage of open-source models could be more comprehensive.

## Related Work & Insights

- **vs. MoleculeNet**: Focuses on property prediction; MolLangBench targets language-molecular structure interaction—different levels of abstraction.
- **vs. MolX/Uni-MRL**: These works perform property prediction and caption annotation while bypassing structural understanding as a prerequisite.
- **Analogy to GPQA**: A "diamond set" of only 198 samples still serves as a standard benchmark for LLM scientific reasoning; quality outweighs scale.
- **Insight**: AI for Science requires evaluation of fundamental capabilities before advanced tasks—this represents a "GLUE moment" for the chemistry domain.

## Rating

- Novelty: ⭐⭐⭐⭐ First comprehensive benchmark for the molecule-language structure interface, with a clearly defined problem that aligns well with real chemical workflows.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 16+ LLMs + 5 chemistry models + 3 modalities + SELFIES + pass@k + error analysis + downstream property experiments—extremely comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated, and thoroughly argued.
- Value: ⭐⭐⭐⭐⭐ Provides a much-needed standardized evaluation tool for AI chemistry research, with potential to reorient the field's research priorities.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] Entropy-Guided Dynamic Tokens for Graph-LLM Alignment in Molecular Understanding](entropy-guided_dynamic_tokens_for_graph-llm_alignment_in_molecular_understanding.md)
- [\[ICLR 2026\] RAS: Retrieval-And-Structuring for Knowledge-Intensive LLM Generation](ras_retrieval-and-structuring_for_knowledge-intensive_llm_generation.md)
- [\[NeurIPS 2025\] From Sequence to Structure: Uncovering Substructure Reasoning in Transformers](../../NeurIPS2025/graph_learning/from_sequence_to_structure_uncovering_substructure_reasoning_in_transformers.md)
- [\[ICLR 2026\] GraphUniverse: Synthetic Graph Generation for Evaluating Inductive Generalization](graphuniverse_synthetic_graph_generation_for_evaluating_inductive_generalization.md)
- [\[CVPR 2026\] Mario: Multimodal Graph Reasoning with Large Language Models](../../CVPR2026/graph_learning/mario_multimodal_graph_reasoning_with_large_language_models.md)

<!-- RELATED:END -->
