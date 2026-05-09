---
title: >-
  [Paper Note] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding
description: >-
  [ACL 2026][chemical vision understanding] This paper proposes ChemVLR, the first reasoning-oriented VLM for the chemical domain. It constructs a 760K reasoning dataset via a cross-modal reverse engineering strategy and employs a three-stage training pipeline of continued pre-training → SFT → RL, achieving substantial improvements over proprietary models and domain-specialized VLMs on molecular recognition and reaction prediction tasks.
tags:
  - ACL 2026
  - chemical vision understanding
  - reasoning VLM
  - cross-modal reverse engineering
  - three-stage training
  - molecular recognition
date: 2026-05-08
content_hash: 63a41b53d6237776
---

# ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding

**Conference**: ACL 2026
**arXiv**: [2604.06685](https://arxiv.org/abs/2604.06685)
**Code**: [https://github.com/xxlllz/ChemVLR](https://github.com/xxlllz/ChemVLR)
**Area**: Interpretability
**Keywords**: chemical vision understanding, reasoning VLM, cross-modal reverse engineering, three-stage training, molecular recognition

## TL;DR
This paper proposes ChemVLR, the first reasoning-oriented VLM for the chemical domain. It constructs a 760K reasoning dataset via a cross-modal reverse engineering strategy and employs a three-stage training pipeline of continued pre-training → SFT → RL, achieving substantial improvements over proprietary models and domain-specialized VLMs on molecular recognition and reaction prediction tasks.

## Background & Motivation

**Background**: Chemical VLMs (e.g., ChemVLM, TinyChemVL) have made notable progress but primarily adopt an end-to-end direct-answer paradigm relying on SFT. Meanwhile, RLVR has demonstrated powerful reasoning enhancement capabilities in domains such as mathematics and programming.

**Limitations of Prior Work**: Existing chemical VLMs operate as "black-box" systems—jumping directly from molecular images to answers without generating interpretable reasoning chains. They fail to leverage the capacity of LLMs to infer underlying reaction mechanisms and perform poorly on complex visual chemistry problems. Furthermore, high-quality chemical reasoning data is extremely scarce, particularly visual grounded reasoning annotations.

**Key Challenge**: Chemical image understanding requires fine-grained substructure analysis (e.g., functional group recognition), yet general-purpose VLMs lack domain-specific chemical knowledge, and direct SFT is insufficient to fully activate pre-trained knowledge.

**Goal**: To build a chemical VLM that prioritizes reasoning during perception—explicitly identifying fine-grained chemical descriptors (e.g., functional groups) before deriving the final answer.

**Key Insight**: Leveraging textual chemical queries paired with ground-truth answers, the reasoning process is reverse-engineered via an LLM and combined with image rendering to produce visual reasoning data.

**Core Idea**: Large-scale reasoning data generation via cross-modal reverse engineering, combined with a progressive CPT → SFT → RL three-stage training pipeline.

## Method

### Overall Architecture
For data construction, reasoning data is generated through reverse engineering starting from textual SMILES QA pairs. Gemini-2.5-Flash reconstructs the reasoning process, augmented with IUPAC names, RDKit-computed functional groups, and expert demonstrations as semantic anchors. A three-stage filtering pipeline yields 760K high-quality samples. For training, a three-stage pipeline is employed: CPT (chemical-visual alignment) → SFT (mixed reasoning and instruction training) → RL (DAPO optimization).

### Key Designs

1. **Cross-Modal Reverse Engineering for Data Generation**

   - **Function**: Generate large-scale visual reasoning data from scarce reasoning annotations.
   - **Mechanism**: Given textual SMILES queries and answers, three types of auxiliary semantic anchors are integrated—IUPAC names retrieved from PubChem, functional groups computed by RDKit, and manually curated expert demonstrations—to prompt Gemini-2.5-Flash to reverse-engineer the reasoning process. A three-stage filtering pipeline is applied: structural filtering (retaining visual reasoning patterns) → answer consistency checking (verifying that derived results match ground-truth SMILES) → external LLM validation (independent verification by GPT-4.1-mini). The final output comprises 360K reasoning samples, 400K description samples, and 1.4M instruction samples.
   - **Design Motivation**: Providing SMILES sequences alone is insufficient for LLMs to generate accurate reasoning. Semantic anchors raise data retention rates from 55%–78% to 73%–95%.

2. **Three-Stage Progressive Training**

   - **Function**: Systematically build chemical perception and reasoning capabilities.
   - **Mechanism**: The CPT stage trains the ViT and projector (with the LLM backbone frozen) using 500K chemical image-text pairs for visual-chemical domain alignment. The SFT stage performs full-parameter fine-tuning with mixed data comprising 360K reasoning and 1.4M instruction samples, using `<think>/<answer>` tag formatting. The RL stage applies DAPO optimization with SMILES accuracy rewards (Tanimoto similarity of 1.0 required for reward) and format rewards.
   - **Design Motivation**: General-purpose VLMs lack chemical visual perception capabilities (experimentally confirmed that direct SFT yields poor results); CPT is necessary to bridge the domain gap first.

3. **IUPAC Knowledge Activation**

   - **Function**: Enhance chemical understanding by leveraging pre-trained knowledge.
   - **Mechanism**: 300K image-to-IUPAC conversion samples are constructed as part of the instruction data. Since IUPAC nomenclature appears far more frequently than SMILES in general pre-training corpora, this effectively activates the model's existing chemical knowledge.
   - **Design Motivation**: Direct training with SMILES yields limited gains, whereas incorporating IUPAC data raises the data generation retention rate from 78% to 92%.

### Loss & Training
The RL stage employs DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) with binary rewards (accuracy + format). The SFT model is used to filter a medium-difficulty subset of 100K samples for training.

## Key Experimental Results

### Main Results

| Model | MMChemOCR Avg Sim. | MMChemOCR Tani@1.0 | img2smiles Tani@1.0 | ChemRxn-V Pred |
|-------|-------------------|-------------------|-------------------|---------------|
| ChemVLR-8B | **93.8** | **84.6** | **92.7** | **67.8** |
| TinyChemVL | 91.2 | 77.4 | 75.6 | 52.4 |
| Gemini-3-Flash | 77.6 | 61.2 | 63.8 | 51.7 |
| ChemDFM-X | 70.9 | 36.5 | 77.6 | 0.7 |

### Ablation Study

| Configuration | Performance | Note |
|---------------|-------------|------|
| SFT only | Baseline | Lacks chemical visual understanding capability |
| CPT + SFT | Improved | Visual alignment improves perception |
| CPT + SFT + RL | **Best** | RL yields an average gain of 9% across all tasks |
| RL only | Nearly ineffective | Cannot optimize effectively without domain foundation |

### Key Findings
- ChemVLR is the first VLM to achieve accuracy on par with specialized SMILES OCR models (e.g., Decimer).
- RL training exhibits an "aha moment"—reward rises sharply between steps 200 and 400.
- IUPAC data serves as a critical catalyst, significantly activating pre-trained chemical knowledge.

## Highlights & Insights
- The **reverse engineering data generation strategy** is highly practical—reconstructing reasoning from answers with multi-level validation to ensure quality, and generalizable to other professional domains with scarce annotations.
- The **finding on IUPAC knowledge activation** is instructive: the degree to which pre-trained knowledge can be leveraged depends on whether the representation format of the training data aligns with the pre-training distribution.
- The **RL "aha moment"** further validates the effectiveness of RLVR for reasoning enhancement in specialized domains.

## Limitations & Future Work
- Training requires 16× H800 GPUs, imposing high resource demands.
- The correctness of reasoning chains depends on filtering quality; reasoning paths may be technically correct but logically imprecise.
- Validation is limited to organic chemistry molecules and reactions; inorganic chemistry and more complex reaction mechanisms have not been addressed.

## Related Work & Insights
- **vs. TinyChemVL**: Also a chemistry-domain VLM but trained with SFT only; ChemVLR further enhances reasoning capability through RL.
- **vs. ChemDFM-R/Chem-R**: These works enhance reasoning in the text domain, whereas ChemVLR extends this to multimodal visual reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐ First reasoning VLM for the chemical domain; reverse engineering data generation strategy is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple benchmarks, multiple baselines, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Data construction and training pipeline are clearly described.
- Value: ⭐⭐⭐⭐ Significant contribution to chemical AI and scientific reasoning.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Efficient Vision-Language Reasoning via Adaptive Token Pruning](../../NeurIPS2025/interpretability/efficient_vision-language_reasoning_via_adaptive_token_pruning.md)
- [\[ACL 2026\] Understanding or Memorizing? A Case Study of German Definite Articles in Language Models](understanding_or_memorizing_a_case_study_of_german_definite_articles_in_language.md)
- [\[ICLR 2026\] Cross-Modal Redundancy and the Geometry of Vision-Language Embeddings](../../ICLR2026/interpretability/cross-modal_redundancy_and_the_geometry_of_vision-language_embeddings.md)
- [\[ACL 2026\] The Reasoning Trap: How Enhancing LLM Reasoning Amplifies Tool Hallucination](the_reasoning_trap_how_enhancing_llm_reasoning_amplifies_tool_hallucination.md)
- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)

<!-- RELATED:END -->
