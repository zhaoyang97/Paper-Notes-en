---
title: >-
  [Paper Note] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding
description: >-
  [ACL 2026][Multimodal VLM][Chemical vision understanding] This paper proposes ChemVLR, the first reasoning-oriented VLM in the chemical field. By constructing a 760K reasoning dataset through a cross-modal reverse engine…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Chemical vision understanding"
  - "reasoning VLM"
  - "cross-modal reverse engineering"
  - "three-stage training"
  - "molecular recognition"
date: 2026-05-08
content_hash: e63eb57186797890
---

# ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding

**Conference**: ACL 2026  
**arXiv**: [2604.06685](https://arxiv.org/abs/2604.06685)  
**Code**: [https://github.com/xxlllz/ChemVLR](https://github.com/xxlllz/ChemVLR)  
**Area**: Interpretability  
**Keywords**: Chemical vision understanding, reasoning VLM, cross-modal reverse engineering, three-stage training, molecular recognition

## TL;DR
This paper proposes ChemVLR, the first reasoning-oriented VLM in the chemical field. By constructing a 760K reasoning dataset through a cross-modal reverse engineering strategy and employing a three-stage training pipeline (CPT-SFT-RL), the model significantly outperforms proprietary models and domain-expert VLMs in molecular recognition and reaction prediction tasks.

## Background & Motivation

**Background**: Chemical VLMs (e.g., ChemVLM, TinyChemVL) have made progress but primarily follow an end-to-end direct answering paradigm relying on SFT. Meanwhile, RLVR has demonstrated powerful reasoning enhancement capabilities in domains such as mathematics and programming.

**Limitations of Prior Work**: Existing chemical VLMs are "black-box" systems—jumping directly from molecular images to answers without generating interpretable reasoning paths. They fail to fully utilize the LLM's capacity to infer underlying reaction mechanisms and perform poorly on complex visual chemistry problems. Furthermore, high-quality chemical reasoning data, especially visually grounded reasoning annotations, is extremely scarce.

**Key Challenge**: Chemical image understanding requires fine-grained substructure analysis (e.g., functional group recognition), but general VLMs lack domain-specific chemical knowledge, and direct SFT fails to sufficiently activate pre-trained knowledge.

**Goal**: To build a chemical VLM that prioritizes reasoning during the perception process—explicitly identifying fine-grained chemical descriptors (e.g., functional groups) before deriving the final answer.

**Key Insight**: Utilize textual chemical queries and ground-truth answers to reverse-engineer reasoning processes via LLMs, combined with image rendering to generate visual reasoning data.

**Core Idea**: Large-scale reasoning data generation via cross-modal reverse engineering combined with a CPT $\rightarrow$ SFT $\rightarrow$ RL progressive training pipeline.

## Method

### Overall Architecture
Regarding data construction, reasoning processes are reconstructed from textual SMILES QA pairs using Gemini-2.5-Flash, utilizing IUPAC names, RDKit functional groups, and expert demonstrations as semantic anchors. High-quality samples (760K) are generated after three stages of filtering. Regarding training, a three-stage process is adopted: CPT (chemical-visual alignment) $\rightarrow$ SFT (mixed reasoning and instruction training) $\rightarrow$ RL (DAPO optimization).

### Key Designs

1.  **Cross-modal Reverse Engineering Data Generation**:
    *   **Function**: Generates large-scale visual reasoning data from scarce reasoning annotations.
    *   **Mechanism**: Given textual SMILES queries and answers, three types of auxiliary semantic anchors (IUPAC names retrieved from PubChem, functional groups calculated by RDKit, and manually curated expert demonstrations) are integrated to reverse-engineer the reasoning process using Gemini-2.5-Flash. Three stages of filtering: structural filtering (retaining visual reasoning patterns) $\rightarrow$ answer consistency check (verifying derivation matches ground-truth SMILES) $\rightarrow$ external LLM verification (independent validation by GPT-4.1-mini). Final output includes 360K reasoning, 400K description, and 1.4M instruction samples.
    *   **Design Motivation**: Providing SMILES sequences alone is insufficient for LLMs to generate accurate reasoning; semantic anchors increased the data retention rate from 55%-78% to 73%-95%.

2.  **Three-stage Progressive Training**:
    *   **Function**: Systematically builds chemical perception and reasoning capabilities.
    *   **Mechanism**: The CPT stage trains the ViT+Projector (freezing the LLM backbone) using 500K chemical image-text pairs for visual-chemical domain alignment. The SFT stage involves full-parameter fine-tuning on a mix of 360K reasoning and 1.4M instruction data using the `<think>/<answer>` tag format. The RL stage uses DAPO optimization with binarized rewards based on SMILES accuracy (rewarded only if Tanimoto similarity is 1.0) and formatting.
    *   **Design Motivation**: General VLMs lack chemical visual perception (experiments confirmed direct SFT performs poorly), requiring CPT first to bridge the domain gap.

3.  **IUPAC Knowledge Activation**:
    *   **Function**: Leverages pre-trained knowledge to enhance chemical understanding.
    *   **Mechanism**: A portion of the instruction data consists of 300K image-to-IUPAC conversion samples. Since IUPAC naming appears far more frequently in general pre-training corpora than SMILES, it effectively activates the model's existing chemical knowledge.
    *   **Design Motivation**: The paper found that training directly with SMILES yielded limited results, but adding IUPAC data increased the data generation retention rate from 78% $\rightarrow$ 92%.

### Loss & Training
The RL stage utilizes DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization) with binary rewards (accuracy + format). The SFT model is used to filter out 100K medium-difficulty samples for training.

## Key Experimental Results

### Main Results

| Model | MMChemOCR Avg Sim. | MMChemOCR Tani@1.0 | img2smiles Tani@1.0 | ChemRxn-V Pred |
| :--- | :--- | :--- | :--- | :--- |
| ChemVLR-8B | **93.8** | **84.6** | **92.7** | **67.8** |
| TinyChemVL | 91.2 | 77.4 | 75.6 | 52.4 |
| Gemini-3-Flash | 77.6 | 61.2 | 63.8 | 51.7 |
| ChemDFM-X | 70.9 | 36.5 | 77.6 | 0.7 |

### Ablation Study

| Configuration | Effect | Description |
| :--- | :--- | :--- |
| SFT only | Baseline | Lacks chemical visual understanding capabilities |
| CPT + SFT | Gain | Visual alignment improves perception |
| CPT + SFT + RL | **Optimal** | RL improves average performance across tasks by 9% |
| RL only | Ineffective | Optimization fails without domain foundation |

### Key Findings
*   ChemVLR is the first VLM to achieve precision comparable to specialized SMILES OCR models (e.g., Decimer).
*   RL training exhibits an "aha moment"—rewards rise sharply between 200-400 steps.
*   IUPAC data acts as a key catalyst, significantly activating pre-trained knowledge.

## Highlights & Insights
*   The **reverse engineering data generation strategy** is highly practical—deriving the reasoning process from the answer and ensuring quality through multiple verifications can be generalized to other data-scarce professional domains.
*   The discovery of **IUPAC knowledge activation** is insightful—the utilization of pre-trained knowledge depends on whether the representation of training data matches the pre-trained distribution.
*   The **RL "aha moment"** re-validates the effectiveness of RLVR for reasoning enhancement in specialized domains.

## Limitations & Future Work
*   Training requires 16xH800 GPUs, entailing high resource demands.
*   The correctness of the reasoning process depends on filtering quality; there may be cases where reasoning paths are correct but the logic is imprecise.
*   Validation was only performed on organic chemical molecules/reactions; inorganic chemistry and more complex reaction mechanisms have not yet been explored.

## Related Work & Insights
*   **vs TinyChemVL**: Both are chemical domain VLMs, but TinyChemVL only uses SFT; ChemVLR further enhances reasoning through RL.
*   **vs ChemDFM-R/Chem-R**: These enhance reasoning in the text domain, whereas ChemVLR extends this to multimodal visual reasoning.

## Rating
*   Novelty: ⭐⭐⭐⭐ (First reasoning VLM in chemistry; novel reverse engineering data strategy)
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple benchmarks, baselines, and detailed ablations)
*   Writing Quality: ⭐⭐⭐⭐ (Clear descriptions of data construction and training pipeline)
*   Value: ⭐⭐⭐⭐ (Significant contribution to AI for Science and scientific reasoning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)
- [\[ACL 2026\] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning](a_survey_of_multimodal_mathematical_reasoning_from_perception_alignment_to_reaso.md)
- [\[AAAI 2026\] TinyChemVL: Advancing Chemical Vision-Language Models via Efficient Visual Token Reduction and Complex Reaction Tasks](../../AAAI2026/multimodal_vlm/tinychemvl_advancing_chemical_vision-language_models_via_efficient_visual_token_.md)
- [\[ICML 2026\] Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](../../ICML2026/multimodal_vlm/bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)

</div>

<!-- RELATED:END -->
