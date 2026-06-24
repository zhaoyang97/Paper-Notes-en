---
title: >-
  [Paper Note] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu
description: >-
  [ACL 2025][Multilingual & Machine Translation][Low-resource machine translation] This work systematically investigates the impact of various language resources (dictionaries, parallel corpora, grammar books, CoT prompts) on translation quality in LLM in-context machine translation. Using Manchu as a case study, it finds that high-quality dictionaries and retrieved parallel examples are the most valuable, while grammar books are almost useless. Through character-encryption exp…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Low-resource machine translation"
  - "In-context learning"
  - "Manchu"
  - "Language resource ablation"
  - "Data augmentation"
date: 2026-05-08
content_hash: 766f34cbabc7983b
---

# Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu

**Conference**: ACL 2025  
**arXiv**: [2502.11862](https://arxiv.org/abs/2502.11862)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Low-resource machine translation, In-context learning, Manchu, Language resource ablation, Data augmentation

## TL;DR

This work systematically investigates the impact of various language resources (dictionaries, parallel corpora, grammar books, CoT prompts) on translation quality in LLM in-context machine translation. Using Manchu as a case study, it finds that high-quality dictionaries and retrieved parallel examples are the most valuable, while grammar books are almost useless. Through character-encryption experiments, the study proves that LLMs primarily rely on in-context learning capabilities rather than prior knowledge. Finally, it demonstrates the effectiveness of utilizing in-context translation to generate synthetic parallel data for training traditional NMT models.

## Background & Motivation

**Background**: Large language models (LLMs), with their in-context learning (ICL) capabilities, can perform machine translation without specialized training. For low-resource languages, this approach is particularly attractive—by simply integrating language resources such as dictionaries, grammar books, and parallel examples into the prompt, LLMs can translate directly. Existing works (e.g., LingoLLM) have verified the viability of this paradigm across multiple endangered languages.

**Limitations of Prior Work**: Although various language resources can be incorporated into the prompt, the relative importance of these resources remains unclear. Which are the most crucial among dictionaries, grammar books, and parallel examples? What is the performance gap between high-quality and low-quality resources of the same type? Furthermore, LLMs might have already "seen" some low-resource language data during the pre-training stage. The confounding factor of whether in-context translation performance stems from ICL capability or prior knowledge has not been systematically isolated.

**Key Challenge**: Due to the limited prompt length, it is impossible to cram all language resources into it. Knowing which resources are truly useful and how to incorporate them for optimal results is self-evident for utilizing the limited context window efficiently. Meanwhile, only by separating the contributions of ICL and prior knowledge can the potential and limitations of this translation paradigm be truly understood.

**Goal**: (1) Systematically ablate the impact of various language resources on translation quality; (2) isolate the contributions of prior knowledge and ICL via encryption experiments; (3) explore the practical application value of in-context translation as a tool for data augmentation.

**Key Insight**: Manchu is selected as a case study—a critically endangered Tungusic language that, due to its historical importance, possesses relatively rich linguistic research resources (dictionaries, grammar books, parallel texts), making it highly suitable as a research subject.

**Core Idea**: By conducting rigorous sequential ablation and character-encryption experiments, this study answers two core questions: "which resources are most important?" and "ICL vs. prior knowledge", and applies in-context translation to data augmentation to bootstrap the training of traditional NMT models.

## Method

### Overall Architecture

The overall pipeline is as follows: given a Manchu sentence, it is first decomposed into stems and suffixes by a morphological analyzer. Then, lexical definitions are retrieved from a dictionary, similar examples are retrieved from a parallel corpus, related grammatical explanations are extracted from a grammar book, and finally, this information is integrated into a prompt for the LLM to generate an English translation. The authors formalize the prompt as $\pi(\mu(\mathbf{x}), D, P, G, C)$, where each parameter corresponds to morphological analysis, dictionary, parallel examples, grammar, and CoT instructions, respectively. The contribution of each component is evaluated one by one through sequential ablation experiments.

### Key Designs

1. **Rule-Based Morphological Analyzer**:

    - **Function**: Decompose Manchu sentences into stem and suffix sequences, serving as the basis for subsequent retrieval.
    - **Mechanism**: Manchu is an agglutinative language that extensively uses suffixes to mark grammatical features, and the boundary between stems and suffixes is relatively clear. The analyzer recursively strips known suffixes from the end of the word until the remaining portion matches a known stem. For ambiguous words with multiple potential analyses (e.g., *tere* can be the pronoun "that" or the inflected verb form *te-re* "sitting"), all possible analyses are preserved to allow the LLM to choose the most appropriate interpretation in context.
    - **Design Motivation**: Morphological analysis is a preliminary step for dictionary lookup and example retrieval. Prior work (such as LingoLLM) simply queried dictionaries using whole words, leading to limited coverage. A rule-based analyzer can handle the rich suffix variations in Manchu, significantly improving dictionary match rates.

2. **Multi-level Dictionary Information Integration and Sequential Ablation**:

    - **Function**: Systematically evaluate the contribution of different levels of dictionary information to translation.
    - **Mechanism**: Three incremental dictionary variants were designed: $D_l$ containing only lexical definitions; $D_{l+s}$ adding suffix explanations; and $D_{l+s+c}$ further incorporating phrase collocations. Experiments show that adding suffix explanations improves BLEU from 7.40 to 7.47, and adding collocations further increases it to 7.55. Similarly, three methods were designed for parallel examples: random selection $P_r$, dictionary-entry-retrieved $P_d$, and BM25-retrieved $P_{bm}$. Grammar was designed as a simplified version $G_s$, a detailed version $G_l$, and a detailed version with examples $G_{l+p}$. CoT was designed as an annotated version $C_a$ and an annotated + syntactic analysis version $C_{a+s}$. Ablations were executed in order of expected contribution: dictionary $\rightarrow$ parallel examples $\rightarrow$ grammar $\rightarrow$ CoT. At each step, the optimal configuration was selected as the baseline for the next step.
    - **Design Motivation**: A fully combinative ablation is intractable due to the excessive number of variants. Sequential ablation allows evaluating the marginal contribution of each component under controllable complexity. Ordering from "expected most useful $\rightarrow$ most uncertain" ensures research efficiency.

3. **Character-Level Encryption Experiments**:

    - **Function**: Isolate the contributions of the LLM's prior knowledge and in-context learning capabilities.
    - **Mechanism**: Through simple character mapping (e.g., vowels mapped cyclically like a $\rightarrow$ e, e $\rightarrow$ i, ..., u $\rightarrow$ a, and consonants similarly), Manchu text is "encrypted" into "pseudo-Manchu" that the LLM has never encountered. Encryption is applied to all Manchu content (input sentences, dictionary entries, parallel examples), while English text remains unchanged. Consequently, the LLM must rely solely on the information in the prompt and its ICL capacity to translate, without utilizing any prior knowledge of Manchu. The contribution of prior knowledge can then be quantified by comparing the translation result differences between the encrypted and original Manchu.
    - **Design Motivation**: Existing studies cannot determine how much of the LLM's in-context translation performance stems from having "seen" target language data during pre-training. The encryption experiment serves as a concise and effective method for controlled variable analysis.

### Loss & Training

The in-context translation phase does not involve training. For the data augmentation experiments, 42,240 synthetic parallel data sentences were generated using in-context MT and mixed with 3,520 real parallel sentences to fine-tune mT5-small. The model was trained with a learning rate of 5e-4, active batch size of 16, and an early stopping strategy (stopping if validation loss did not decrease for 2 consecutive steps).

## Key Experimental Results

### Main Results

| Model | BLEU | chrF | SBERT |
|------|------|------|-------|
| Llama3-1B | 0.27 | 9.95 | 16.37 |
| Llama3-3B | 1.81 | 21.95 | 38.46 |
| Llama3-8B | 3.05 | 26.59 | 49.10 |
| Llama3-70B | 6.31 | 31.01 | 56.82 |
| GPT-4o | 8.84 | 33.72 | 61.35 |
| DeepSeek-V3 | **12.35** | **37.93** | **65.64** |

### Ablation Study (GPT-4o, marginal contribution of each component)

| Component | BLEU | chrF | SBERT | Description |
|------|------|------|-------|------|
| $\pi(\mathbf{x})$ Direct Translation | 3.10 | 21.68 | 33.49 | baseline |
| $+\mu(\mathbf{x})$ Morphological Analysis | - | - | - | No improvement when added alone |
| $+D_{l+s+c}$ Full Dictionary | 7.55 | 32.71 | 61.07 | **Largest contribution** |
| $+P_{bm}$ BM25 Parallel Examples | 8.84 | 33.72 | 61.35 | Significant improvement |
| $+G_{l+p}$ Grammar with Examples | 8.90 | 33.77 | 60.40 | **Almost no improvement** |
| $+C_{a+s}$ CoT | 8.49 | 33.43 | 59.01 | **Performance decreased instead** |

### Key Findings

- **Dictionary is the most critical resource**: After adding the full dictionary, BLEU leaped from 3.10 to 7.55, contributing over 60% of the total improvement. In particular, suffix explanations and phrase collocations both provided additional value.
- **High-quality parallel examples provide significant help**: Similar examples retrieved via BM25 outperform random selection and word-retrieved examples, raising the BLEU score from 7.55 to 8.84.
- **Grammar books hardly assist translation**: The improvements offered by the three grammar variants are marginal, consistent with the findings of Aycock et al.
- **CoT is actually harmful**: Explicitly asking the LLM to perform grammatical tagging and syntactic analysis introduces more errors, ultimately degrading translation quality.
- **Model size is important**: The translation quality of the Llama3 family continuously improves from 1B to 70B, with DeepSeek-V3 performing the best (likely because its training data contains more Chinese/Manchu data).
- **LLMs rely primarily on ICL rather than prior knowledge**: The encryption experiments show that most models experience minor performance degradation (except for DeepSeek-V3), proving that the translation ability stems primarily from in-context learning.
- **Data augmentation is highly effective**: The mT5-small model (300M parameters) trained on synthetic data can match or even exceed the in-context translation performance of Llama3-70B.

## Highlights & Insights

- **The character-encryption experiment** is the most ingenious design of this paper—it effectively isolates the contributions of prior knowledge and ICL via simple character mapping, with a concise methodology and clear conclusions. This experimental paradigm can be transferred to studies of other low-resource languages.
- **The finding that "grammar is useless"** carries significant implications—even though grammatical information is highly useful for human learners, LLMs do not seem to utilize formalized grammar rules effectively. This suggests that prompt design should focus on vocabulary and examples rather than grammatical explanations.
- **The application of data augmentation** demonstrates the practical value of in-context translation—not just translation itself, but also as a tool to generate synthetic training data, enabling a 300M small model to achieve performance comparable to a 70B large model.

## Limitations & Future Work

- Only investigated one language (Manchu), which is an agglutinative language with clear boundaries between stems and suffixes; whether the study's conclusions hold for languages with other typological characteristics (such as isolating or inflectional languages) remains uncertain.
- Only explored the Manchu $\rightarrow$ English direction, without investigating the reverse direction.
- The CoT experiments tested only limited strategies; more optimized CoT designs might yield different results.
- The character-encryption did not alter the typological characteristics of Manchu (e.g., word order, agglutinative structure); more aggressive "pseudo-language" designs might lead to different findings.
- Future work can extend this framework to more low-resource languages, especially those with different distributions of corpus resources.

## Related Work & Insights

- **vs LingoLLM (Zhang et al., 2024b)**: LingoLLM verified the feasibility of in-context translation across multiple low-resource languages but did not systematically ablate the contribution of each component. This paper's ablation experiments are more rigorous and comprehensive.
- **vs Tanzer et al. (2024)**: This ICLR 2024 work proposed a benchmark for learning to translate from a grammar book but did not differentiate the relative contributions of dictionaries and grammar. This paper clearly demonstrates that dictionaries are far more important than grammar.
- **vs Hus & Anastasopoulos (2024)**: EMNLP 2024's "Back to School" also explored utilizing grammar books for translation but did not conduct encryption experiments to isolate prior knowledge.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The encryption experiment is ingeniously designed, and the ablation analysis is comprehensive, though the method itself (prompt + retrieval) is not novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely detailed ablations, multi-model comparisons, encryption experiments, and data augmentation applications make it highly complete.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logically rigorous, clear figures and tables, and detailed appendices, making it a high-quality empirical research paper.
- **Value**: ⭐⭐⭐⭐ Provides a clear practical guide for low-resource in-context translation (prioritize dictionaries and examples, ignore grammar and CoT).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning](grammamt_improving_machine_translation_with_grammar-informed_in-context_learning.md)
- [\[ACL 2025\] The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages](the_esethu_framework_reimagining_sustainable_dataset_governance_and_curation_for.md)

</div>

<!-- RELATED:END -->
