---
title: >-
  [Paper Note] Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases
description: >-
  [ACL 2025][LLM Pretraining][pre-pretraining] Proposes performing "pre-pretraining" on formal languages prior to natural language pre-training, demonstrating that formal languages with hierarchical dependency structures (such as k-Shuffle Dyck) provide effective inductive biases for Transformers, enabling a 1B-parameter model to achieve the same language modeling loss with 33% fewer tokens.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "pre-pretraining"
  - "formal languages"
  - "Chomsky hierarchy"
  - "circuit complexity"
  - "inductive bias"
date: 2026-05-08
content_hash: c14ddfa89feb8e66
---

# Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases

**Conference**: ACL 2025  
**arXiv**: [2502.19249](https://arxiv.org/abs/2502.19249)  
**Code**: [GitHub](https://github.com/michahu/pre-pretraining)  
**Area**: LLM Pre-training  
**Keywords**: pre-pretraining, formal languages, Chomsky hierarchy, circuit complexity, inductive bias

## TL;DR

Proposes performing "pre-pretraining" on formal languages prior to natural language pre-training, demonstrating that formal languages with hierarchical dependency structures (such as k-Shuffle Dyck) provide effective inductive biases for Transformers, enabling a 1B-parameter model to achieve the same language modeling loss with 33% fewer tokens.

## Background & Motivation

**Background**: Although language models show impressive performance across various tasks, they remain highly "data-hungry"—requiring 5 to 6 orders of magnitude more data than humans to reach human-level performance. Data efficiency is an important research frontier.

**Limitations of Prior Work**: Training models is difficult in low-resource settings and data-constrained scenarios. As the majority of natural language data has already been utilized, continuous improvement of data efficiency has become a key challenge.

**Key Challenge**: While formal languages and natural language seem completely unrelated, prior work indicates that pre-training on formal languages can improve natural language acquisition. The core problem lies in determining what features of formal languages enable effective transfer.

**Goal**: To investigate which properties of formal languages allow pre-pretraining to transfer effectively to natural language.

**Key Insight**: Analyzing the intersection of two dimensions: linguistics (Chomsky hierarchy) and computational complexity theory (circuit depth / C-RASP).

**Core Idea**: The optimal pre-pretraining language should reside at the intersection of the Chomsky hierarchy and circuit complexity hierarchy—capturing the hierarchical dependency structures in natural language while remaining learnable within the computational limits of Transformer architectures.

## Method

### Overall Architecture

Prior to natural language pre-training, the model is trained on formal language data (pre-pretraining), after which weights are directly transferred to the natural language pre-training stage. Four formal languages are systematically tested: 1-Dyck (nested parentheses), k-Dyck (multi-type parentheses), k-Shuffle Dyck (cross-dependent parentheses), and ww (copying language), along with random strings and natural language as baselines.

### Key Designs

1. **Chomsky × Circuit Hypothesis**: Effective transfer occurs when two conditions are met simultaneously: (a) the formal language captures the hierarchical dependency structures present in natural language (context-sensitive or above in the Chomsky hierarchy); (b) the formal language is learnable within the computational limits of Transformer architectures (definable in C-RASP). k-Shuffle Dyck fulfills both conditions.
2. **Marginal Rate of Substitution (MRS)**: Quantifies the efficiency of formal language tokens—how many natural language tokens a single formal language token can replace. In experiments, the MRS of k-Shuffle Dyck is significantly greater than 1, implying that formal language tokens are far more efficient than natural language tokens.
3. **Subnetwork Analysis**: Pruning methods are employed to identify the sparse subnetwork learned during pre-pretraining, verifying that these attention heads remain crucial after natural language training. This provides mechanistic evidence of transfer.

### Loss & Training

- Standard autoregressive language modeling loss (next-token prediction cross-entropy)
- Independent learning rate warmups for the pre-pretraining and pre-training stages
- Keeping pre-training hyperparameters and training steps fixed, varying only the pre-pretraining data and duration

## Key Experimental Results

### Main Results

Performance of Pythia 160M models after different pre-pretraining languages:

| Pre-pretraining Language | C4 Val Loss | BLiMP Grammar ↑ | Verbatim Retrieval ↓ | C-RASP? | Hierarchical Dependency? |
|---------------------|----------|-----------|---------|---------|---------|
| None | Baseline | Baseline | Baseline | - | - |
| Random Binary/Integer Strings | Worse than baseline | - | - | - | No |
| Natural Language | Better than baseline | No improvement | Improved | - | - |
| 1-Dyck | Better than baseline | Improved | Improved | ✓ | Partial |
| k-Dyck | Better than baseline | Improved | Improved | ✗ | Yes |
| ww (Copying) | Worse than baseline | Improved | Degraded | ✗ | No |
| **k-Shuffle Dyck** | **Optimal** | **Improved** | **Optimal** | ✓ | Yes |

Pythia 1B experiments: After pre-pretraining on 1.6B tokens of k-Shuffle Dyck, the model requires only 1.10B total tokens to reach the final loss of the baseline (which requires 1.63B), demonstrating a 33% improvement in token efficiency.

### Ablation Study

| Experiment | Validation Loss | Findings |
|------|---------|------|
| n-gram fitted data (unigram/bigram/trigram) | Worse than k-Shuffle Dyck | Transfer effects do not stem from local statistical properties |
| Vocabulary size k=32/64/128/256 | k=128 optimal | Optimal hyperparameters exist |
| Subnetwork pruning 50% | ℳ significantly outperforms random pruning | Pre-pretraining attention heads remain critical in natural language |

### Key Findings

- **Hierarchical dependency is the core of transfer**: Only languages with hierarchical dependencies (k-Dyck, k-Shuffle Dyck) achieve positive transfer.
- **Random data is harmful**: Pre-pretraining on random strings degrades performance instead.
- **Formal languages outperform equivalent amounts of natural language**: MRS > 1, meaning formal language tokens are more "valuable" than natural language tokens.
- **All formal languages improve grammatical judgment**: Even though ww is harmful to overall loss, it still improves BLiMP grammar accuracy.
- **Mechanistic traceability**: Attention heads learned during pre-pretraining are retained and reused in subsequent natural language training.

## Highlights & Insights

- Elegant theoretical hypothesis—the intersection of Chomsky hierarchy and circuit complexity hierarchy precisely pinpoints the optimal pre-pretraining language.
- Surprising finding: The token efficiency of formal languages is actually higher than that of natural language (MRS >> 1), challenging the intuition of statistical learning theory.
- Subnetwork analysis provides mechanistic evidence of transfer from formal to natural language, going beyond superficial metric observations.
- Practical value: Pre-pretraining weights can be distributed independently and easily integrated into existing pre-training pipelines.

## Limitations & Future Work

- Only tested staged training (formal first, then natural), leaving open whether mixed training is superior.
- Findings are validated on English (a high-resource language); effects on low-resource languages might differ or be even more pronounced.
- Scaling laws beyond the 1B parameter and 1.6B token regime remain unknown.
- Only Transformers were considered; findings may vary on RNNs and State Space Models (SSMs).
- Optimal hyperparameters for k-Shuffle Dyck (e.g., vocabulary size) require more efficient automatic search methods.

## Related Work & Insights

- Related to curriculum learning (Bengio et al., 2009) but with a novel direction—traditional curriculum learning shows mostly negative results in the BabyLM challenge, whereas the formal language approach in this paper succeeds.
- Consistent with the phenomenon of code pre-training transferring to natural language—the structured nature of code might offer a similar inductive bias.
- Insight: Formal language pre-pretraining can be analogized to a "cognitive warm-up," establishing foundational circuits for structured reasoning before learning natural language content more efficiently.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The Chomsky × Circuit hypothesis presents a unique perspective, and the finding that formal language token efficiency outperforms natural language is highly striking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive cross-language comparisons and ablations, though lacking validation at a larger scale.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical derivation, ingenious experimental design, and fluent narrative.
- Value: ⭐⭐⭐⭐⭐ Introduces a completely new direction for pre-training data efficiency, with the potential to influence future LLM training practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Chinese Grammatical Error Correction With Pre-trained Models and Linguistic Clues](chinese_grammatical_error_correction_with_pre-trained_models_and_linguistic_clue.md)
- [\[ACL 2025\] How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training](how_do_llms_acquire_new_knowledge_a_knowledge_circuits_perspective_on_continual_.md)
- [\[ACL 2025\] Splintering Nonconcatenative Languages for Better Tokenization](splintering_nonconcatenative_languages_for_better_tokenization.md)
- [\[ICML 2025\] Evaluating Morphological Alignment of Tokenizers in 70 Languages](../../ICML2025/llm_pretraining/evaluating_morphological_alignment_of_tokenizers_in_70_languages.md)
- [\[ACL 2025\] Data Caricatures: On the Representation of African American Language in Pretraining Corpora](data_caricatures_on_the_representation_of_african_american_language_in_pretraini.md)

</div>

<!-- RELATED:END -->
