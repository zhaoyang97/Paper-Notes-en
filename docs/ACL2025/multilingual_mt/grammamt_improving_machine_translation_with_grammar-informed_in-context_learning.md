---
title: >-
  [Paper Note] GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning
description: >-
  [ACL 2025][Multilingual & Machine Translation][Machine Translation] GrammaMT is proposed to leverage grammatical information from Interlinear Glossed Text (IGT) to enhance few-shot machine translation in LLMs, achieving an average improvement of 12+ BLEU on endangered languages and consistent improvements on medium- and high-resource languages.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Machine Translation"
  - "Grammatical Information"
  - "In-Context Learning"
  - "Low-Resource Languages"
  - "Interlinear Glossed Text"
date: 2026-05-08
content_hash: 63f2ef999efb8fd3
---

# GrammaMT: Improving Machine Translation with Grammar-Informed In-Context Learning

**Conference**: ACL 2025  
**arXiv**: [2410.18702](https://arxiv.org/abs/2410.18702)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Machine Translation, Grammatical Information, In-Context Learning, Low-Resource Languages, Interlinear Glossed Text  

## TL;DR

GrammaMT is proposed to leverage grammatical information from Interlinear Glossed Text (IGT) to enhance few-shot machine translation in LLMs, achieving an average improvement of 12+ BLEU on endangered languages and consistent improvements on medium- and high-resource languages.

## Background & Motivation

**Background**: LLMs perform exceptionally well in machine translation for high-resource languages, but the translation quality for low-resource languages, especially endangered ones, remains poor. Leveraging existing LLMs to translate low-resource languages requires satisfying: (i) no training required; (ii) only a small amount of data is needed; (iii) data is easy to collect.

**Limitations of Prior Work**: Existing methods either rely on fine-tuning on large-scale corpora (unfeasible for low-resource languages) or require comprehensive resources like grammar books and dictionaries (e.g., LingoLLM), which are expensive to acquire. Standard few-shot methods are simple but yield limited performance.

**Key Challenge**: There is an urgent demand for translating low-resource/endangered languages, yet available data and linguistic resources are extremely scarce. The core challenge is how to maximize translation quality using minimal and most easily accessible linguistic information.

**Goal**: Design a training-free method requiring only a small number of linguistic annotations to enhance the multilingual translation capability of LLMs for low-resource languages.

**Key Insight**: Utilize Interlinear Glossed Text (IGT)—a common morpheme-level annotation format in linguistics containing triples of source sentences, gloss lines, and target translations—to inject grammatical information into LLM prompts.

**Core Idea**: Embed IGT grammatical annotations into few-shot prompts to exchange minimal linguistic annotation costs for significant improvements in low-resource translation.

## Method

### Overall Architecture

GrammaMT proposes three prompting strategies, all training-free and requiring only 21 IGT examples:

### Key Designs

#### 1. Gloss-shot

Add IGT annotation triples (source sentence + gloss + translation) into few-shot examples, allowing the LLM to learn source language structures from grammatical examples:

$$(\mathbf{g}_1, \cdots, \mathbf{g}_N, \mathbf{x}) \rightarrow \mathbf{y}$$

#### 2. Chain-gloss

Similar to Chain-of-Thought, require the LLM to first generate the gloss of the input sentence and then perform translation:

$$(\mathbf{g}_1, \cdots, \mathbf{g}_N, \mathbf{x}) \rightarrow (\mathbf{y}_g, \mathbf{y})$$

This increases interpretability, but relies on the LLM's own ability to generate glosses.

#### 3. Model-gloss

Use an external gloss generation model (such as GlossLM) to generate the gloss for the input sentence, avoiding inaccuracies in LLM's self-generated gloss:

$$(\mathbf{g}_1, \cdots, \mathbf{g}_N, \mathbf{x}, \mathbf{y}_{ge}) \rightarrow \mathbf{y}$$

#### IGT Format Description

IGT annotation is a standard format for linguistic description, such as Swahili:
- **Source**: (yeye) alimwona (yeye)
- **Gloss**: 3SG-PST-see-FV 3SG (uppercase = grammatical morpheme, lowercase = lexical morpheme)
- **Translation**: S/he saw him/her

### Experimental Setup

- **Models**: Meta-Llama-3-70B-Instruct (4-bit quantized) is primarily used, while 8B, Mixtral-8x22B, and GPT-4o are also tested
- **N-shot**: All strategies uniformly use 21 examples (proven optimal by ablation studies)
- **Evaluation Metrics**: BLEU (SacreBLEU), chrF++, xCOMET-XXL
- **Decoding**: Greedy decoding, temperature=1

## Key Experimental Results

### Main Results — Endangered/Unseen Languages (SIGMORPHON 2023)

| Method | BLEU Avg. | chrF++ Avg. | xCOMET Avg. |
|---|---|---|---|
| NLLB-200 | 0.55 | 13.80 | 12.82 |
| zero-shot | 0.88 | 18.05 | 15.21 |
| few-shot | 3.94 | 21.85 | 16.76 |
| gloss-shot | 3.41 | **22.50** | 18.21 |
| chain-gloss | 4.25 | 20.84 | 16.78 |
| **model-gloss** | **15.97** | **41.45** | **40.83** |
| LingoLLM (GPT-4) | 14.1 | — | — |

**model-gloss improves by an average of 12.03 BLEU compared to few-shot, and outperforms LingoLLM, which utilizes more linguistic resources.**

### Main Results — Low-Resource Languages (GlossLM)

| Method | BLEU Avg. (5 lang) | chrF++ Avg. | xCOMET Avg. |
|---|---|---|---|
| few-shot | 16.69 | 36.96 | 34.52 |
| gloss-shot | 16.39 | 36.88 | 35.65 |
| **chain-gloss** | **17.06** | **37.10** | **35.77** |

Most significant improvement on Yoruba: few-shot 11.98 → gloss-shot 16.32 (+4.34 BLEU).

### Main Results — Medium-to-High Resource Languages

| Method | BLEU Avg. (7 lang) |
|---|---|
| few-shot | 18.95 |
| gloss-shot | 18.61 |
| **chain-gloss** | **19.75** |

chain-gloss outperforms few-shot by 2.5+ BLEU on Urdu and Russian.

### Ablation Study

| Analysis | Key Findings |
|---|---|
| **N-shot count** | N=21 is optimal; excessive increases lead to a performance plateau |
| **Gloss Accuracy** | Llama achieves only 21% word accuracy on Tsez, whereas GlossLM reaches 88% → directly explaining the advantage of model-gloss |
| **Oracle Experiment** | Using gold gloss improves performance by an average of 17.46 BLEU (±6.6); zero-gloss also outperforms few-shot → proving that gloss itself is highly valuable |
| **Effect of Grammatical Annotations** | Performance degrades when grammatical annotations are removed to keep only lexical morphemes → indicating grammatical information is more than word-to-word translation |
| **Cross-model Generalization** | model-gloss achieves 18.69 average BLEU on SIGMORPHON with GPT-4o; the smaller Llama-8B model is also effective |
| **Out-of-Domain Generalization (FLORES)** | gloss-shot performs best (avg 21.64 vs few-shot 20.69); chain-gloss performs unstably on out-of-domain data |

### Key Findings

- **model-gloss performs best on endangered languages**: relies on the accuracy of an external gloss model but offers massive improvements
- **chain-gloss is more practical for low/medium/high-resource languages**: does not rely on an external model and yields improvements in most languages
- **gloss-shot is the most robust in out-of-domain settings**: only uses glosses as prompt examples without generating them, offering the widest applicability
- **Oracle experiments reveal the upper bound**: accurate glosses can bring an improvement of 17+ BLEU, showing that the development of automatic gloss generation models is highly worthwhile

## Highlights & Insights

- **Extremely low-threshold linguistic enhancement**: requires only 21 IGT triples, and such annotations are very common in linguistic descriptions, making them much easier to obtain than grammar books and dictionaries
- **Linguistic instantiation of Chain-of-Thought**: chain-gloss is a natural variant of CoT in translation tasks, and grammatical annotations provide a more explicit structure than simply "letting the model think step-by-step"
- **Three strategies covering different scenarios**: model-gloss is suitable for languages with gloss models, chain-gloss is suitable for general scenarios, and gloss-shot is suitable for out-of-domain transfer

## Limitations & Future Work

- Evaluates primarily translation into English (→en) directions; reverse translation (en→) is only explored preliminarily
- Interpretability of gloss-shot is limited; it remains unclear how glosses in unrelated examples affect the translation
- chain-gloss is unstable on out-of-domain data, pointing likely to mismatched distributions between short-sentence glosses in GlossLM and long sentences in FLORES
- Limited by the availability of IGT data, it does not cover all language families

## Related Work & Insights

- LingoLLM (Zhang et al., 2024): uses grammar books + dictionaries + morphological analyzers → resource requirements are significantly higher than GrammaMT
- GlossLM (Ginn et al., 2024): gloss corpus of 250K sentences × 1800 languages → directly supporting the model-gloss strategy
- Tanzer et al. (2024): grammar book-assisted translation → similar idea but high resource requirements
- Chain-of-Thought (Wei et al., 2022): GrammaMT's chain-gloss is essentially CoT in the linguistic domain

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Utilizing IGT, a standard linguistic tool, to enhance LLM translation is a novel and natural idea
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 3 datasets × 16 languages × 4 models × multiple ablations, highly comprehensive
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, deep ablation analysis, and elegantly designed ablations
- **Value**: ⭐⭐⭐⭐⭐ — Significant practical value for endangered language translation; the method is simple, effective, and has a very low threshold

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] Blessing of Multilinguality: A Systematic Analysis of Multilingual In-Context Learning](blessing_of_multilinguality_a_systematic_analysis_of_multilingual_in-context_lea.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](../../ACL2026/multilingual_mt/clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2025\] THOR-MoE: Hierarchical Task-Guided and Context-Responsive Routing for Neural Machine Translation](thor-moe_hierarchical_task-guided_and_context-responsive_routing_for_neural_mach.md)

</div>

<!-- RELATED:END -->
