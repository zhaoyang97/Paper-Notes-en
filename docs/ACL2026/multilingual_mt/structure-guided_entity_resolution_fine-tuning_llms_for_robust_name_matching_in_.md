---
title: >-
  [Paper Note] Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts
description: >-
  [ACL2026][Multilingual & Machine Translation][Entity Resolution] SGER proposes a two-stage curriculum learning framework to fine-tune Llama 3 8B for personal name entity matching: Phase 1 trains the model to parse name s…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "Entity Resolution"
  - "Name Matching"
  - "Curriculum Learning"
  - "KYC"
  - "LoRA Fine-Tuning"
  - "Multilingual"
date: 2026-05-08
content_hash: 5c9d8cb538955b8e
---

# Structure-Guided Entity Resolution: Fine-Tuning LLMs for Robust Name Matching in Complex Linguistic Contexts

**Conference**: ACL2026
**arXiv**: [2605.23597](https://arxiv.org/abs/2605.23597)
**Code**: TBD
**Area**: multilingual_mt
**Keywords**: Entity Resolution, Name Matching, Curriculum Learning, KYC, LoRA Fine-Tuning, Multilingual

## TL;DR

SGER proposes a two-stage curriculum learning framework to fine-tune Llama 3 8B for personal name entity matching: Phase 1 trains the model to parse name structures (outputting JSON), and Phase 2 trains a binary classification matcher starting from the Phase 1 checkpoint. It achieves 99.02% accuracy and 0.994 F1 on a 50,000-pair Indian KYC dataset and has been deployed in the Dream11 (250 million users) production environment.

## Background & Motivation

Name matching is a core task in entity resolution, directly impacting operations and regulation in KYC/AML compliance scenarios. In India, one of the world's most linguistically diverse environments, the challenges are particularly prominent: naming conventions vary by region and community, inconsistent multi-script transcription (Devanagari $\rightarrow$ Latin), data entry errors (OCR-merged tokens, missing spaces), and inconsistent honorific suffixes. Traditional methods (Edit Distance, Jaro-Winkler, Phonetic Encoding) struggle to handle these structural variations, while directly fine-tuning LLMs for binary classification forces the model to learn both name structures and decision boundaries simultaneously, which limits performance.

## Method

### Overall Architecture

SGER adopts a two-stage curriculum learning fine-tuning approach for Llama 3 8B: Phase 1 teaches the model to understand the internal grammatical structure of names, and Phase 2 trains a binary entity matcher based on the Phase 1 checkpoint. During inference, the system takes two name strings as input and outputs "Yes" or "No".

### Key Designs

1.  **Phase 1 - Name Structure Understanding**: Maps a single name string to a JSON object (`first_name`, `middle_name`, `last_name`). The model is trained using approximately 10,000 manually annotated Indian names, covering multi-regional, multi-lingual, and multi-cultural naming patterns. LoRA SFT fine-tuning is used to establish stable internal representations of name components.
2.  **Phase 2 - Binary Matching**: Continues training from the Phase 1 checkpoint using annotated name pairs to develop a binary classifier. During inference, logits for the "Yes"/"No" tokens are extracted, and a matching probability is calculated via softmax. A pair is judged as the same person if the probability exceeds a predefined threshold.
3.  **Data Augmentation Strategies** (inspired by CV augmentation): (a) **Pair Swapping** (analogous to mirror flipping) — ensures order invariance; (b) **Component Permutation** (analogous to geometric transformation) — permutes first/middle/last components to generate structural variants; (c) **Random Space Deletion** (analogous to noise injection) — simulates data entry errors. This expanded the dataset from 20K pairs to over 50K.

### Loss & Training

Both stages utilize LoRA (rank=4, alpha=8) and mixed-precision (fp16) SFT. Early stopping is based on the validation set F1 score. Training was performed on a single NVIDIA A100 80GB GPU.

## Key Experimental Results

### Main Results

Comparison on an independent test set of 50,000 pairs (ensuring no name-level overlap):

| Method | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Levenshtein (Th=0.8) | 57.4% | 75.1% | 70.2% | 0.726 |
| BERT (Fine-Tuned) | 69.1% | 81.2% | 80.1% | 0.806 |
| GPT-4o (Few-Shot) | 85.4% | 90.1% | 92.1% | 0.911 |
| LLM SFT (Llama 3 8B) | 91.2% | 93.3% | 95.9% | 0.946 |
| LLM SFT + Aug. | 95.7% | 97.6% | 97.0% | 0.973 |
| **SGER (Ours)** | **99.02%** | **99.95%** | **98.9%** | **0.994** |

### Ablation Study

-   **Effect of Data Augmentation**: Expanding from the original 20K pairs to over 50K improved the F1 score from 0.946 to 0.973.
-   **Effect of Curriculum Learning**: Incorporating Phase 1 structural pre-training on top of the augmented data improved the F1 score from 0.973 to 0.994, demonstrating the complementarity of these techniques.
-   **Error Analysis**: Primary failure modes include (1) ambiguous phonetics ("Saurabh" vs. "Sorab") and (2) composite errors (where missing spaces, inverted order, and OCR corruption occur simultaneously).

### Key Findings

-   Curriculum learning providing the greatest improvement on difficult samples containing multiple combined perturbations—Phase 1 initializes stable internal representations for name components.
-   Production Deployment: Deployed using 3× NVIDIA L4 GPUs with the vLLM inference framework, achieving 10K RPM and a P99 latency of 120ms.
-   Business Impact: Eliminated manual review processes, saving over $500K in annual operational costs.

## Highlights & Insights

-   **Decomposing Structure and Decision**: Making the implicit "name grammar" an explicit pre-training task before downstream matching is a simple yet highly effective design philosophy.
-   **Transferring CV Augmentation to NLP**: Applying concepts like mirror/rotation/noise injection from the image domain to name data via order swapping, component permutation, and space deletion.
-   **Industrial Implementation Integrity**: Provides an end-to-end closed loop from data augmentation and two-stage training to deployment architecture and commercial impact.

## Limitations & Future Work

-   The approach was validated only on Indian KYC data; its transferability to other multilingual environments (e.g., Arabic or East Asian names) has not been tested.
-   The Phase 1 JSON schema only supports three fields: `first_name`, `middle_name`, and `last_name`, which may not cover the name structures of all cultures.
-   The method relies on LoRA fine-tuning of Llama 3 8B; the performance of smaller or larger models remains to be explored.

## Related Work & Insights

-   **Peeters & Bizer (2023)**: A comprehensive survey on the use of LLMs for entity matching.
-   **Steiner et al. (2024)**: Demonstrated the efficiency of fine-tuning for ER, which this paper extends by introducing curriculum learning.
-   **Curriculum Learning** (Feng et al., 2023; Soviany et al., 2022): This work is the first to validate the effectiveness of two-stage curriculum learning specifically for name entity resolution.

## Rating

| Dimension | Score (1-10) |
|---|---|
| Novelty | 6 |
| Utility | 10 |
| Clarity | 9 |
| Experimental Thoroughness | 7 |

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multilingual Language Models Encode Script Over Linguistic Structure](multilingual_language_models_encode_script_over_linguistic_structure.md)
- [\[ACL 2026\] Exploring Two-Phase Continual Instruction Fine-tuning for Multilingual Adaptation in Large Language Models](exploring_continual_fine-tuning_for_enhancing_language_ability_in_large_language.md)
- [\[ICML 2026\] Toward Robust Multilingual Adaptation of LLMs for Low-Resource Languages](../../ICML2026/multilingual_mt/toward_robust_multilingual_adaptation_of_llms_for_low-resource_languages.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](../../ICLR2026/multilingual_mt/sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[ACL 2026\] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)

</div>

<!-- RELATED:END -->
