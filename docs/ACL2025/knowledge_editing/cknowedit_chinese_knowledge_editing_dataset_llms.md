---
title: >-
  [Paper Note] CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs
description: >-
  [ACL 2025][Knowledge Editing][Chinese Dataset] Constructs CKnowEdit, the first knowledge editing dataset oriented towards Chinese linguistic characteristics. It covers three major categories (linguistics (pinyin/ancient poetry/classical Chinese/idioms/proverbs), facts (history and geography), and logical traps (homophones/reasoning/wordplay)) with a total of 1,854 samples. It systematically evaluates the performance of five mainstream knowledge editing methods on four Chinese…
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Chinese Dataset"
  - "Linguistics"
  - "Logical Traps"
  - "Cultural Knowledge"
date: 2026-05-08
content_hash: de89376495374576
---

# CKnowEdit: A New Chinese Knowledge Editing Dataset for Linguistics, Facts, and Logic Error Correction in LLMs

**Conference**: ACL 2025  
**arXiv**: [2409.05806](https://arxiv.org/abs/2409.05806)  
**Code**: [https://github.com/zjunlp/EasyEdit](https://github.com/zjunlp/EasyEdit)  
**Area**: Knowledge Editing / Chinese NLP  
**Keywords**: Knowledge Editing, Chinese Dataset, Linguistics, Logical Traps, Cultural Knowledge

## TL;DR
Constructs CKnowEdit, the first knowledge editing dataset oriented towards Chinese linguistic characteristics. It covers three major categories (linguistics (pinyin/ancient poetry/classical Chinese/idioms/proverbs), facts (history and geography), and logical traps (homophones/reasoning/wordplay)) with a total of 1,854 samples. It systematically evaluates the performance of five mainstream knowledge editing methods on four Chinese LLMs, revealing unique editing challenges in Chinese.

## Background & Motivation

**Background**: Knowledge Editing aims to correct erroneous knowledge in LLMs without full retraining. Existing datasets (ZsRE, CounterFact, KnowEdit, etc.) are primarily based on factual triples from English Wikipedia, exhibiting an obvious English-centric bias. Although a few multilingual datasets attempt cross-lingual editing, most are obtained by translating English corpora, failing to capture the deep characteristics of the target language.

**Limitations of Prior Work**:

- (a) Translation cannot preserve unique Chinese linguistic phenomena—polyphonic characters, parallelism, classical Chinese, and idioms/allusions are completely lost during translation.
- (b) Existing multilingual datasets primarily evaluate cross-lingual editing consistency, which is unsuitable for studying Chinese-specific knowledge editing methods.
- (c) Three unique challenges of the Chinese language system: linguistic complexity (integration of form, sound, and meaning), culture-bearing factual knowledge (untranslatable geographical and historical concepts), and language-specific logical structures (reliance on implicit conjunctions and topic-prominent structures).

**Key Challenge**: Current knowledge editing research overlooks language specificity, leading to a drastic performance drop of editing methods in Chinese scenarios, particularly when dealing with knowledge involving culture, phonetics, and classical literature.

**Key Insight**: Starting from three unique dimensions of the Chinese language (linguistic features, cultural facts, and logical traps), original Chinese data is natively collected instead of relying on translation, and an open-ended generation + LLM-as-judge evaluation paradigm is adopted to replace traditional token-level automatic evaluation.

## Method

### Overall Architecture
The construction workflow of CKnowEdit is: **Multi-source data collection → Qwen-7B-Chat filtering (retaining samples where the model answered incorrectly) → GPT-4 assisted labeling + human verification → Five-step quality assurance process**. Ultimately, 1,854 high-quality samples were selected from 11,981 raw data points. The dataset contains comprehensive fields including prompt, target_new, target_old, generalization (weak/strong generalization), and locality (related but distinct knowledge).

### Key Designs

1. **A taxonomic system of Chinese knowledge with three major categories and ten subcategories**:

    - **Linguistics (48.4%)**: Pinyin (polyphonic character ambiguity), Ancient poetry (strict metrics + rare characters), Classical Chinese (differences between ancient and modern meanings for polysemous words), Idioms (reversion of literal and actual meanings), Proverbs (metaphorical understanding).
    - **Facts (5.97%)**: Widespread gaps in LLM knowledge regarding Chinese history and geography.
    - **Logic (45.63%)**: homophonic misunderstandings (e.g., "the captain/the queue is died/long after vaccine"), reasoning errors, wordplay (semantic absurdity caused by word segmentation ambiguity).
    - Design Motivation: Each type of knowledge poses challenges to LLMs in different dimensions—linguistics tests cultural memory, facts test knowledge coverage, and logic tests reasoning and disambiguation capabilities.

2. **Strict Evaluation Design for Generalization and Locality**:

    - **Weak Generalization**: Paraphrasing the prompt with synonyms to test whether the edited model can still output correct answers under different phrasings.
    - **Strong Generalization**: Divided into "context migration" (e.g., migrating a character with the same meaning in classical Chinese to a new context) and "logical one-step reasoning" (using edited knowledge as a premise for a single-step deduction).
    - **Locality**: Instead of using completely unrelated knowledge for comparison, choosing knowledge that is "related to the target knowledge but with different facts" (such as sharing the same subject) to construct a more rigorous side-effect detection.
    - Design Motivation: The polysemy and context-dependency of Chinese demand more granular generalization testing; simple prompt replacement is insufficient to verify true knowledge acquisition.

3. **Open-ended Generation + LLM-as-Judge Evaluation Paradigm**:

    - Giving up traditional token-level / logit-level teacher-forcing automatic evaluation (where ROUGE-L is heavily affected by length bias).
    - Adopting open-ended text generation + GPT-4o scoring (1-10 scale) with customized evaluation prompts for each knowledge category.
    - Human evaluation validation: 70 samples × 20 categories (4 models × 5 methods), yielding a correlation coefficient of 0.70 with GPT-4 scores.

### Evaluation Metrics
Four standard knowledge editing metrics: Editing Success (ES), Generalization (Gen), Portability (Por), and Locality (Loc), each scored 1-10 by GPT-4o.

## Key Experimental Results

### Main Results

| Editing Method | Type | Best ES Count | Best Gen Count | Best Por Count | Features |
|---------|------|-----------|-------------|-------------|------|
| AdaLoRA | Parameter Fine-tuning | 70%+ cases | ~70% cases | ~86% cases | Globally optimal, suitable for long-text editing |
| AlphaEdit | Parameter Modification | 4 cases | Sub-optimal | Sub-optimal | Null-space constrained editing |
| FT-M | Parameter Fine-tuning | 3 cases | Average | Average | Simple fine-tuning baseline |
| ROME | Locate-and-Edit | Poor | Poor | Poor | Local parameter modification, unsuitable for long text |
| GRACE | External Parameter | Average | Average | Average | Discrete key-value adapter |

Evaluated Models: Qwen-7B-Chat, Qwen2-7B-Instruct, DeepSeek-LLM-7B-Chat, Baichuan2-7B-Chat

### Ablation Study

| Analysis Dimension | Key Findings |
|---------|---------|
| Ancient Poetry Editing | All methods performed worst; Portability is almost always < 1. Reason: Weak representation of rare characters + distributional shift in ancient/modern grammar |
| Chinese vs. English | Linguistic knowledge editing severely distorts after translation to English (classical poetry translates back to modern prose); factual knowledge shows minor Chinese-English differences; English performs better on logic (translation eliminates Chinese-specific traps) |
| Cross-lingual Generalization | Editing in English yields poor performance on Chinese queries—neurons for different languages do not overlap in LLMs, forming a natural cross-lingual barrier |
| ROME vs AdaLoRA | ROME's local parameter modification is suitable for short factual triples but breaks long-text generation distribution; AdaLoRA adaptively adjusts multiple modules to maintain contextual consistency |

### Key Findings
- **AdaLoRA is globally optimal in Chinese long-text editing**, overturning prior conclusions of ROME's superiority on English datasets—reflecting the unique demands of Chinese editing.
- **Chinese linguistic knowledge is the hardest to edit**: Ancient poetry and idioms involve deep binding of form, sound, and meaning, which cannot be reached by symbol-level editing.
- **Translation cannot replace native Chinese data**: Linguistic knowledge and logical traps are completely dissolved during translation.
- **Human evaluation validates the effectiveness of GPT-4o as a judge** (correlation coefficient of 0.70).

## Highlights & Insights
- **The first native Chinese knowledge editing dataset**: Collected from diverse sources such as classical literature and Baidu Tieba (Ruozhiba), truly reflecting the depth and cultural complexity of the Chinese language.
- **Exquisitely designed data taxonomy**: The taxonomy of three major and ten sub-categories not only covers the unique challenges of Chinese but also provides a paradigm for constructing language-specific datasets in other languages.
- **Upgraded evaluation methodology**: Open-ended generation + LLM-as-judge is closer to practical applications than traditional token-level evaluations, and its reliability is confirmed through human validation.
- **Crucial insight—the choice of editing method is strongly correlated with language/knowledge type**: ROME is effective in English factual editing but fails in Chinese cultural knowledge editing.

## Limitations & Future Work
- Imbalanced data distribution: Linguistic and logical data account for >94%, while factual data is only 5.97%, affecting the adequacy of evaluating factual editing.
- Experiments conducted only under single-edit settings; batch editing and sequential editing scenarios remain unexplored (limited by computational resources).
- There may be biases in GPT-4 evaluating GPT-4 (though this paper evaluates other models).
- Larger models (>7B) are not covered, leaving the performance of editing methods on large-scale models unknown.
- Data filtering used Qwen-7B-Chat as a baseline; with model capability improvements, some samples might be answered correctly.

## Related Work & Insights
- **vs. KnowEdit / ZsRE**: English factual editing datasets; CKnowEdit complements them with linguistic and logical dimensions, being native to Chinese.
- **vs. Bi-ZsRE / MzsRE**: Multilingual datasets constructed through translation; CKnowEdit proves that translation cannot preserve language specificity.
- **vs. EasyEdit Framework**: CKnowEdit is integrated into EasyEdit, enabling direct replica of all experiments.
- **Insight**: For low-resource or culture-specific languages, knowledge editing research must construct native datasets rather than relying on translation.

## Rating
- Novelty: ⭐⭐⭐⭐ The first native Chinese knowledge editing dataset, with innovations in both taxonomy and evaluation methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 methods × 4 models, Chinese-English comparison, cross-lingual evaluation, human validation.
- Writing Quality: ⭐⭐⭐⭐ Detailed linguistic analysis with rich and intuitive examples.
- Value: ⭐⭐⭐⭐ Fills the gap of Chinese knowledge editing datasets and provides a paradigm for non-English knowledge editing research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database](../../ICLR2026/knowledge_editing/scaling_knowledge_editing_in_llms_to_100000_facts_with_neural_kv_database.md)
- [\[ACL 2025\] ToxEdit: Adaptive Detoxification Safeguarding General Capabilities of LLMs through Toxicity-Aware Knowledge Editing](adaptive_detoxification_safeguarding_general_capabilities_of_llms_through_toxici.md)
- [\[ICML 2025\] WikiBigEdit: Understanding the Limits of Lifelong Knowledge Editing in LLMs](../../ICML2025/knowledge_editing/wikibigedit_understanding_the_limits_of_lifelong_knowledge_editing_in_llms.md)
- [\[ACL 2025\] SAKE: Steering Activations for Knowledge Editing](sake_steering_activations_for_knowledge_editing.md)
- [\[ACL 2025\] Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)

</div>

<!-- RELATED:END -->
