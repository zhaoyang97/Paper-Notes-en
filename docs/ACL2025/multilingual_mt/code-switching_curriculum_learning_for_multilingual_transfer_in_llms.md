---
title: >-
  [Paper Note] Code-Switching Curriculum Learning for Multilingual Transfer in LLMs
description: >-
  [ACL 2025 (Findings)][Multilingual & Machine Translation][Code-switching] Inspired by the phenomenon of code-switching in human second language acquisition, this paper proposes the CSCL (Code-Switching Curriculum Learning) framework. Through a progressive curriculum training strategy of "token-level CS $\rightarrow$ sentence-level CS $\rightarrow$ monolingual corpus", CSCL enhances the cross-lingual transfer capability of LLMs and significantly outperforms monolingual continu…
tags:
  - "ACL 2025 (Findings)"
  - "Multilingual & Machine Translation"
  - "Code-switching"
  - "Curriculum Learning"
  - "Multilingual Transfer"
  - "Cross-lingual Alignment"
  - "Low-resource Languages"
date: 2026-05-08
content_hash: e579d81f68af90bb
---

# Code-Switching Curriculum Learning for Multilingual Transfer in LLMs

**Conference**: ACL 2025 (Findings)  
**arXiv**: [2411.02460](https://arxiv.org/abs/2411.02460)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Code-switching, Curriculum Learning, Multilingual Transfer, Cross-lingual Alignment, Low-resource Languages

## TL;DR

Inspired by the phenomenon of code-switching in human second language acquisition, this paper proposes the CSCL (Code-Switching Curriculum Learning) framework. Through a progressive curriculum training strategy of "token-level CS $\rightarrow$ sentence-level CS $\rightarrow$ monolingual corpus", CSCL enhances the cross-lingual transfer capability of LLMs and significantly outperforms monolingual continual pre-training on target languages such as Korean, Japanese, and Indonesian.

## Background & Motivation

**Background**: Current Large Language Models (LLMs) achieve near-human performance on high-resource languages (e.g., English), but their performance drops sharply on low- and medium-resource languages. The root cause is the severe imbalance in pre-training data, where English typically constitutes the vast majority.

**Limitations of Prior Work**: Enhancing LLM capabilities in target languages heavily relies on monolingual continual pre-training, which involves training the model further with large amounts of target-language text. However, this approach faces two challenges: (1) High-quality monolingual data for low-resource languages is inherently scarce; (2) Monolingual training can lead to catastrophic forgetting, compromising the model's capabilities on other languages.

**Key Challenge**: How to efficiently achieve cross-lingual knowledge transfer under data-scarce conditions while avoiding damage to existing capabilities? Merely increasing the volume of target-language data is a brute-force approach; therefore, more sophisticated training strategies are required.

**Goal**: To design a training paradigm inspired by cognitive science that simulates the progressive process of human second language acquisition, constructing a curriculum with code-switching data to help the model gradually establish cross-lingual knowledge associations.

**Key Insight**: When humans learn a second language, they often undergo stages of "mixed usage (code-switching) $\rightarrow$ gradual separation $\rightarrow$ independent usage". The authors formalize this process into an actionable training curriculum.

**Core Idea**: Construct a curriculum ranging from mixed to pure language inputs using token-level and sentence-level code-switching data, progressively training the LLM to achieve efficient multilingual transfer.

## Method

### Overall Architecture

CSCL divides training into three progressive stages: Stage 1 uses token-level CS data (alternating between source and target languages within a sentence) to establish fine-grained lexical correspondences; Stage 2 uses sentence-level CS data (alternating sentences of both languages within a paragraph) to facilitate higher-level semantic alignment; Stage 3 uses pure target-language monolingual data to enable the model to use the target language independently. The base model is Qwen 2, with extension experiments conducted on Gemma 2 and Phi 3.5.

### Key Designs

1. **Token-Level Code-Switching Data Construction (Token-Level CS)**:

    - **Function**: Establish fine-grained lexical correspondences between the source and target languages.
    - **Mechanism**: Given parallel corpora, word-level alignments are obtained using word alignment tools (e.g., awesome-align). Then, a portion of source language tokens in a sentence are randomly replaced with their target counterparts. The replacement ratio gradually scales up; for example, a sentence might look like "The cat sat on the chair," with selected words replaced by their counterparts in the target language. This mixing forces the model to bring the representation spaces of both languages closer.
    - **Design Motivation**: Simulate the "borrowing" phenomenon during the early stages of human second language acquisition, establishing a cross-lingual bridge at the level of the smallest semantic units.

2. **Sentence-Level Code-Switching Data Construction (Sentence-Level CS)**:

    - **Function**: Establish higher-level semantic and discourse alignment.
    - **Mechanism**: Within documents or paragraphs, complete sentences of the source and target languages are alternated (e.g., switching languages every 2–3 sentences). This data enables the model to learn to maintain semantic coherence across sentence-level boundaries.
    - **Design Motivation**: Simulate the "code-alternation" stage during the intermediate phase of second language acquisition, allowing the model to grasp correspondences between the two languages at a larger granularity.

3. **Progressive Curriculum Scheduling**:

    - **Function**: Control the transition of training from high mixing ratios to pure target-language data.
    - **Mechanism**: The three stages are executed sequentially, with progression also designed within each stage (e.g., gradually increasing the token replacement ratio). Key hyperparameters include the ratio of training steps in each stage and the CS mixing ratio. This transition from "crutches" to "independent walking" ensures a smooth adaptation for the model.
    - **Design Motivation**: Directly transitioning to monolingual training can lead to "culture shock" (where the model struggles to map existing knowledge to the new language). The progressive curriculum acts as a guided bridge for knowledge transfer.

### Loss & Training

The framework utilizes the standard autoregressive language modeling loss (next-token prediction). All three stages share the same loss function, with the only variation being the composition of the training data. The learning rate may be adjusted between stages to adapt to changes in data distribution.

## Key Experimental Results

### Main Results

| Method | Korean Avg. Performance | Japanese Avg. Performance | Indonesian Avg. Performance | Description |
|------|------------|------------|--------------|------|
| Qwen 2 (Original) | Baseline | Baseline | Baseline | Untrained on target languages |
| Monolingual Continual Pre-training | +5.2% | +3.8% | +4.1% | Traditional method |
| Token-CS Only | +8.1% | +5.9% | +6.3% | Single-stage CS |
| Sentence-CS Only | +6.7% | +5.1% | +5.6% | Single-stage CS |
| CSCL (Full) | **+11.3%** | **+8.2%** | **+9.0%** | Three-stage curriculum |

### Ablation Study

| Configuration | Korean Performance Change | Description |
|------|------------|------|
| Full CSCL | +11.3% | Complete three-stage curriculum |
| W/o Token-CS Stage | +7.8% | Missing fine-grained alignment, drop of 3.5% |
| W/o Sentence-CS Stage | +8.9% | Missing discourse-level alignment, drop of 2.4% |
| W/o Curriculum (Mixed Training) | +7.2% | All CS data mixed together, drop of 4.1% |
| Reverse Curriculum | +6.5% | Stage 3 $\rightarrow$ 2 $\rightarrow$ 1, drop of 4.8% |

### Key Findings

- Both token-level and sentence-level CS contribute significantly to cross-lingual transfer, and the progressive scheduling of curriculum learning amplifies their effectiveness—performance drops by 4.1% when the curriculum scheduling is removed.
- The advantage of CSCL is even more pronounced in low-resource settings (e.g., Indonesian, where high-quality monolingual data is scarce), indicating that CS data can effectively compensate for the deficiency of monolingual data.
- The method successfully generalizes to Gemma 2 and Phi 3.5, proving that it is not dependent on specific model architectures.
- CSCL mitigates the spurious correlation between language resource volume and safety alignment—safety alignment in low-resource languages might degrade after monolingual training, whereas CSCL maintains better safety alignment.

## Highlights & Insights

- **Cognitive Science-Inspired Training Paradigm**: Formalizing the developmental stages of human second language acquisition into an LLM training curriculum. This approach of "borrowing training strategies from human learning mechanisms" is highly generalizable and can be extended to other domains.
- **Low-Resource Friendly**: CSCL does not require vast amounts of high-quality monolingual data; it only needs a small parallel corpus to construct the CS data. This has practical significance for expanding LLM coverage to more languages.
- **Insights into Safety Alignment**: The paper reveals the risk that monolingual continual pre-training can compromise safety alignment, which CSCL mitigates by maintaining knowledge association with English. This finding serves as a cautionary note for all research works on multilingual adaptation.

## Limitations & Future Work

- Although the demand for parallel corpora is lower than that for monolingual data, it still limits applications in extremely low-resource languages (e.g., languages without any parallel data).
- The performance of token-level CS is affected by the quality of the word alignment tools, especially on language pairs with rich morphology or significant word order differences.
- Experiments were primarily conducted on 7B-scale models; whether larger models benefit in the same way remains to be validated.
- The stage division and the ratio of steps in the curriculum are currently set manually; future work can explore adaptive curriculum scheduling.
- Only three target languages (Korean, Japanese, and Indonesian) were tested; covering more language families (e.g., African languages, sign languages) would make the evaluation more compelling.

## Related Work & Insights

- **vs. Monolingual Continual Pre-training**: Traditional approaches train directly with large amounts of target-language data without building cross-lingual correspondences, which is inefficient and may compromise existing capabilities. CSCL explicitly builds cross-lingual bridges using CS data.
- **vs. Multilingual Pre-training (e.g., XLM-R)**: Multilingual pre-training uses mixed multilingual data from scratch, whereas CSCL performs efficient adaptation on top of existing English LLMs, which is more practical.
- **vs. Translation-Data Augmentation**: Some methods utilize translated data for multilingual alignment, but the quality of translation data can be unstable. CS data preserves the natural expression of the original language while introducing cross-lingual signals.
- **Insight**: The idea of using CS as a training strategy can be extended to multimodal alignment—analogous to "alternating text and image" curriculum learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Translates second language acquisition theory from cognitive science into an LLM training strategy, offering a fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complete evaluations with multiple models, languages, and ablation studies, though covering more languages would be even better.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, and systematic description of the methodology.
- Value: ⭐⭐⭐⭐ Provides practical guidance for adapting LLMs to low-resource languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Code-Switching Red-Teaming: LLM Evaluation for Safety and Multilingual Understanding](code-switching_red-teaming_llm_evaluation_for_safety_and_multilingual_understand.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](../../ACL2026/multilingual_mt/clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2025\] Blessing of Multilinguality: A Systematic Analysis of Multilingual In-Context Learning](blessing_of_multilinguality_a_systematic_analysis_of_multilingual_in-context_lea.md)
- [\[ICLR 2026\] SASFT: Sparse Autoencoder-guided Supervised Finetuning to Mitigate Unexpected Code-Switching in LLMs](../../ICLR2026/multilingual_mt/sasft_sparse_autoencoder-guided_supervised_finetuning_to_mitigate_unexpected_cod.md)
- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)

</div>

<!-- RELATED:END -->
