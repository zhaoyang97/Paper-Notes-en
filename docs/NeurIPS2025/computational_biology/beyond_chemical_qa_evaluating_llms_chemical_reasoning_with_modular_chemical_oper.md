---
title: >-
  [Paper Note] Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations
description: >-
  [NeurIPS 2025][Computational Biology][chemical reasoning] This paper introduces ChemCoTBench, the first CoT-based benchmark for evaluating chemical reasoning in LLMs. It decomposes complex chemical problems into modular chemical operations (adding/deleting/substituting functional groups), and is accompanied by ChemCoTDataset — a large-scale dataset of 22,000 expert-annotated CoT samples — enabling systematic evaluation of both reasoning and non-reasoning LLMs across molecular…
tags:
  - "NeurIPS 2025"
  - "Computational Biology"
  - "chemical reasoning"
  - "Chain-of-Thought"
  - "benchmark"
  - "molecular operations"
  - "SMILES"
date: 2026-05-08
content_hash: 8f626f2d76b48be2
---

# Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations

**Conference**: NeurIPS 2025
**arXiv**: [2505.21318](https://arxiv.org/abs/2505.21318)  
**Code**: [https://github.com/IDEA-XL/ChemCoTBench/](https://github.com/IDEA-XL/ChemCoTBench/)  
**Area**: LLM Reasoning
**Keywords**: chemical reasoning, Chain-of-Thought, benchmark, molecular operations, SMILES

## TL;DR
This paper introduces ChemCoTBench, the first CoT-based benchmark for evaluating chemical reasoning in LLMs. It decomposes complex chemical problems into modular chemical operations (adding/deleting/substituting functional groups), and is accompanied by ChemCoTDataset — a large-scale dataset of 22,000 expert-annotated CoT samples — enabling systematic evaluation of both reasoning and non-reasoning LLMs across molecular understanding, editing, optimization, and reaction prediction.

## Background & Motivation

**Background**: LLMs have demonstrated strong performance in mathematical and code reasoning (aided by CoT), yet their chemical reasoning capabilities have not been systematically evaluated. Existing chemistry benchmarks (e.g., ChemLLM, MolPuzzle) primarily test knowledge retrieval and factual recall rather than step-by-step reasoning.

**Limitations of Prior Work**: (1) A lack of structured, step-by-step chemical reasoning tasks — existing benchmarks conflate reasoning, knowledge recall, and numerical computation, making it impossible to attribute performance to specific factors; (2) insufficient alignment with real-world applications such as drug design and reaction engineering; (3) an almost complete absence of CoT training data for chemical reasoning.

**Key Challenge**: Chemistry demands rigorous structural analysis and multi-step reasoning (e.g., molecular optimization in pharmaceutical design), yet LLM chemical reasoning ability is underestimated or mischaracterized by simple QA tasks.

**Goal**: To establish a systematic benchmark for evaluating chemical reasoning in LLMs, supporting comprehensive assessment from foundational tasks (molecular understanding/editing) to applied tasks (molecular optimization/reaction prediction).

**Key Insight**: Chemical problem-solving is analogized to mathematical arithmetic — modular chemical operations are defined (addition = adding a functional group, subtraction = deleting a functional group, substitution = replacing a functional group), enabling step-wise evaluation of chemical reasoning.

**Core Idea**: Chemical reasoning is formalized as a stepwise, verifiable workflow through modular chemical operations, yielding the first chemical CoT benchmark alongside a large-scale CoT training dataset.

## Method

### Overall Architecture
ChemCoTBench comprises 4 major tasks, 22 sub-tasks, and 1,495 test samples, organized in a hierarchical progression from foundational to applied: Molecular Understanding → Molecular Editing → Molecular Optimization → Reaction Prediction. The reasoning steps for each task are defined as sequences of modular chemical operations. ChemCoTDataset (22,000 CoT-annotated training samples) is provided as a companion resource.

### Key Designs

1. **Foundational Task: Molecule Understanding**

    - **Function**: Evaluates LLMs' basic comprehension of molecular SMILES representations.
    - **Mechanism**: Covers functional group identification and counting, ring structure recognition, Murcko scaffold extraction, ring system identification, and SMILES equivalence judgment — collectively probing LLMs' perception of molecular topology.
    - **Design Motivation**: Molecular understanding is the foundation of all chemical reasoning; failure to correctly parse SMILES structures renders all subsequent operations unreliable.

2. **Foundational Task: Molecule Editing**

    - **Function**: Evaluates whether LLMs can execute basic molecular modification operations following natural language instructions.
    - **Mechanism**: Three atomic operations are defined — Add, Delete, and Substitute functional groups — analogous to addition and subtraction in mathematics, constituting the basic arithmetic of chemical reasoning. Two core capabilities are assessed: maintaining chemical validity after editing, and correctly executing the specified modification.
    - **Design Motivation**: Complex tasks such as molecular optimization can be decomposed into sequences of editing operations, making editing ability a fundamental competency for chemical reasoning.

3. **Applied Task: Molecule Optimization**

    - **Function**: Given a source molecule and a target property, generate an optimized molecule.
    - **Mechanism**: Organized at two levels — physicochemical properties (LogP, solubility, QED) and target-based properties (DRD2, GSK3-beta, JNK3 binding affinity). Target-based optimization is more challenging, requiring understanding of drug–target interactions.
    - **Design Motivation**: This is a central problem in drug design, requiring LLMs not only to parse molecular structures but also to reason about how structural modifications affect target properties.

4. **Applied Task: Reaction Prediction**

    - **Function**: Evaluates LLMs on forward reaction prediction, retrosynthesis, reaction condition recommendation, and reaction mechanism understanding.
    - **Mechanism**: Encompasses 4 sub-tasks — forward prediction (major/minor product prediction), single-step retrosynthesis (inferring reactants from a known product), reaction condition recommendation, and reaction mechanism understanding (predicting the next intermediate or selecting a mechanistic pathway).
    - **Design Motivation**: Reaction prediction is a core task in chemistry; spanning from overall product prediction to detailed mechanistic insights, it comprehensively tests chemical reasoning ability.

5. **ChemCoTDataset (22K CoT Training Data)**

    - **Function**: Provides large-scale chemical reasoning training data.
    - **Mechanism**: Reasoning chains are distilled from Gemini-2.5-pro, DeepSeek-R1, and Claude-3.7-sonnet-thinking; CoT annotations are generated by DeepSeek-R1 and manually reviewed by 13 chemistry PhD students. IUPAC nomenclature is incorporated to assist molecular comprehension.
    - **Design Motivation**: Scarcity of chemical reasoning data is one of the primary reasons for poor LLM performance in chemistry.

### Loss & Training
The benchmark itself does not involve training. ChemCoTDataset is used for SFT fine-tuning to validate the effectiveness of data augmentation.

## Key Experimental Results

### Main Results: Foundational Tasks

| Model | FG Count (MAE) | Ring Count (MAE) | Murcko (Sim) | Ring System (Acc) | Add | Delete | Sub |
|---|---|---|---|---|---|---|---|
| Gemini-2.5-pro-think | **0.11** | **0.60** | **0.51** | **87.5** | **100** | **85** | **81.7** |
| Claude3.7-sonnet-think | 0.21 | 1.60 | 0.40 | 80.0 | 85 | 80 | 83.4 |
| DeepSeek-R1 | 0.27 | 1.55 | 0.34 | 45.0 | 70 | 70 | 68.3 |
| o3-mini | 0.13 | 0.60 | 0.39 | 75.0 | 65 | 55 | 80.0 |
| GPT-4o | 0.17 | 1.35 | 0.21 | 80.0 | 80 | 80 | 65.0 |

### Ablation Study: Effect of Training Data

| Configuration | Description |
|---|---|
| Base model | Limited chemical reasoning ability, especially in open-source and distilled models |
| + ChemCoTDataset fine-tuning | Significant improvement in chemical reasoning performance |
| Strong math/code ability ≠ strong chemistry ability | Lack of chemical reasoning data is the primary bottleneck |

### Key Findings
- **Gemini-2.5-pro leads comprehensively in chemical reasoning**: achieves top performance on nearly all sub-tasks, reaching 100% on the molecular editing Add operation.
- **Reasoning vs. non-reasoning models show limited differences**: indicating that the bottleneck in chemical reasoning lies not in the reasoning framework but in chemical knowledge and SMILES comprehension.
- **Open-source/distilled reasoning models remain weak in chemistry**: e.g., Llama-Nemo-49B scores 0% on the molecular editing Add task, demonstrating that distilled reasoning ability does not transfer to the chemistry domain.
- **ChemCoTDataset is effective**: fine-tuning with this dataset yields significant gains in chemical reasoning performance, confirming that data scarcity is the core issue.
- **SMILES comprehension is the foundational bottleneck**: ring count MAE ranges broadly from 0.6 to 1.9 across models, indicating that LLMs' perception of SMILES topological structure remains poor.

## Highlights & Insights
- **The modular chemical operations analogy** is particularly elegant: mapping chemical reasoning to mathematical arithmetic (add/delete/substitute = addition/subtraction/multiplication) enables stepwise evaluation and training. This paradigm is transferable to other specialized domains (e.g., bioinformatics, materials science).
- **From a quality control perspective**, the hybrid annotation pipeline combining 13 chemistry PhD reviewers with LLM-based generation is a noteworthy methodological contribution.
- **IUPAC nomenclature as an auxiliary input for SMILES comprehension** is a practical finding: IUPAC names encode functional group and structural information, and providing them as auxiliary input helps LLMs better understand molecular structures.

## Limitations & Future Work
- **Limitations of SMILES as a molecular representation**: SMILES is a linear string representation that is inherently ill-suited for encoding 3D structural information. Future work could incorporate molecular graphs or 3D coordinates.
- **Evaluation primarily based on SMILES matching**: the same molecule can have multiple valid SMILES representations, and strict string matching may underestimate model performance.
- **Lack of in-depth evaluation of multi-step reasoning**: although CoT operation sequences are defined, evaluation focuses primarily on final outputs, without systematically assessing the correctness of intermediate steps.
- **Recommended direction**: developing multimodal chemical reasoning benchmarks that integrate molecular graph representations.

## Related Work & Insights
- **vs. ChemLLM**: ChemLLM provides a chemistry-specialized framework but focuses on knowledge recall; ChemCoTBench focuses on reasoning.
- **vs. MolPuzzle**: MolPuzzle addresses spatial reasoning for spectral interpretation; ChemCoTBench offers broader coverage (understanding/editing/optimization/reaction).
- **vs. mathematical benchmarks (MATH/GSM8K, etc.)**: the benchmark design methodology is analogous (hierarchical, verifiable), but ChemCoTBench faces the additional challenge of SMILES comprehension.

## Rating
- Novelty: ⭐⭐⭐⭐ — First chemical CoT benchmark; the modular chemical operations concept is novel
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers 13+ models, 4 major tasks, 22 sub-tasks, with expert review
- Writing Quality: ⭐⭐⭐⭐ — Task design is clearly presented, though the paper is lengthy
- Value: ⭐⭐⭐⭐ — Fills a gap in chemical reasoning evaluation; the CoT dataset is practically useful

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway](../../ACL2026/computational_biology/toxreason_a_benchmark_for_mechanistic_chemical_toxicity_reasoning_via_adverse_ou.md)
- [\[ICLR 2026\] Exploring Synthesizable Chemical Space with Iterative Pathway Refinements](../../ICLR2026/computational_biology/exploring_synthesizable_chemical_space_with_iterative_pathway_refinements.md)
- [\[ICLR 2026\] CP-Agent: Context-Aware Multimodal Reasoning for Cellular Morphological Profiling under Chemical Perturbations](../../ICLR2026/computational_biology/cp-agent_contextaware_multimodal_reasoning_for_cellular_morphological_profiling_.md)
- [\[ICML 2026\] SIGMA: Structure-Invariant Generative Molecular Alignment for Chemical Language Models via Autoregressive Contrastive Learning](../../ICML2026/computational_biology/sigma_structure-invariant_generative_molecular_alignment_for_chemical_language_m.md)
- [\[NeurIPS 2025\] Evaluating Multiple Models Using Labeled and Unlabeled Data](evaluating_multiple_models_using_labeled_and_unlabeled_data.md)

</div>

<!-- RELATED:END -->
