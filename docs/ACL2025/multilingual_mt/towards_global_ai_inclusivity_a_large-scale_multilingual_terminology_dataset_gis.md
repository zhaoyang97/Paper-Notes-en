---
title: >-
  [Paper Note] Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)
description: >-
  [ACL 2025][Multilingual & Machine Translation][Multilingual Terminology] The authors construct GIST, the first large-scale multilingual AI terminology dataset (approximately 5K terms across 5 languages), using a hybrid framework of LLM extraction + human crowdsourced translation + LLM selection. They demonstrate that prompting-based post-translation optimization consistently improves the translation quality of AI terminology in machine translation across metrics such as BLEU…
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Multilingual Terminology"
  - "AI Terminology Translation"
  - "Crowdsourced Translation"
  - "Post-Translation Optimization"
  - "Hybrid LLM-Human Framework"
date: 2026-05-08
content_hash: df9ba718d9174373
---

# Towards Global AI Inclusivity: A Large-Scale Multilingual Terminology Dataset (GIST)

**Conference**: ACL 2025  
**arXiv**: [2412.18367](https://arxiv.org/abs/2412.18367)  
**Code**: [GitHub](https://github.com/jiarui-liu/MultilingualAITerminology)  
**Institutions**: Carnegie Mellon University, University of Michigan, University of Toronto, Max Planck Institute  
**Area**: NLP / Machine Translation / Terminology Translation  
**Keywords**: Multilingual Terminology, AI Terminology Translation, Crowdsourced Translation, Post-Translation Optimization, Hybrid LLM-Human Framework  

## TL;DR

The authors construct GIST, the first large-scale multilingual AI terminology dataset (approximately 5K terms across 5 languages), using a hybrid framework of LLM extraction + human crowdsourced translation + LLM selection. They demonstrate that prompting-based post-translation optimization consistently improves the translation quality of AI terminology in machine translation across metrics such as BLEU and COMET.

## Background & Motivation

Although machine translation has made significant progress, the translation of specialized AI terminology remains a major bottleneck. Terms like "Coreference Resolution" and "Explain-Away Effect" are frequently mistranslated in general systems such as Google Translate, causing comprehension difficulties or even misunderstandings for non-English researchers.

- **Severe shortage of existing resources**: The ACL 60-60 initiative covers only about 250 terms, which far fails to meet practical needs.
- **Inconsistent LLM translation**: The three-model agreement rate among Claude 3 Sonnet, GPT-3.5-Turbo, and Google Translate is only about 15% (higher for Chinese at 42.71%), and the two-model agreement rate is around 40%.
- **Manual curation is hard to scale**: Relying entirely on domain experts to build multilingual terminology databases is costly and time-consuming.
- **Broad impact**: Non-English speakers make up a significant portion of the global AI community. Translation errors of terms in platform documents like Hugging Face model cards and data cards can lead to the misuse of models and datasets.

## Method

### Overall Architecture

The GIST construction pipeline consists of: **Term Extraction → Human Crowdsourced Translation → LLM Best Candidate Selection → Post-Translation Integration**. Terms are extracted from 879 award-winning papers across 18 top-tier conferences (AAAI/IJCAI/CVPR/ECCV/ICCV/ICLR/ICML/NeurIPS/KDD/ACL/EMNLP/NAACL/EACL/LREC/COLING/CoNLL/SIGIR/WWW) and translated into Arabic, Chinese, French, Japanese, and Russian.

### Key Designs

1. **LLM + Multi-round Quality Assurance for Term Extraction**: LLaMA-3-70B-Instruct is used to extract terms from award-winning papers under strict definition criteria—they must be nouns/noun phrases, specific to AI, and either meaningless or having a different meaning in non-AI domains. The processing granularity is sentence chunks of up to 64 words. This is followed by multi-round filtering: removing terms appearing in only 1 paper → removing abbreviations and those starting with special characters → further screening with GPT-4o → manual review by 3 domain experts. External resources such as Wikipedia AI glossaries and government AI terminology dictionaries are also integrated.

2. **Hybrid Work of Human Crowdsourcing + LLM Validation**: Automatic translation feasibility is first tested using Claude 3 Sonnet, GPT-3.5-Turbo, and Google Translate. Since the three-model agreement rate is extremely low (~15% for most languages), Amazon Mechanical Turk is utilized for crowdsourcing. For each term, 10 human translations and 1 Google Translate result are collected, and then GPT-4o selects the best translation among the 11 candidates. The crowdsourcing process includes strict qualification tests and daily quality monitoring.

3. **Training-Free Terminology Integration Methods**: Three post-translation integration strategies are explored—(a) **Prompting Optimization**: GPT-4o-mini is used to refine the initial translation with the term dictionary provided as context; (b) **Word Alignment + Replacement**: Multilingual BERT is used to perform word alignment, locating the source term in the translation and replacing it with the GIST translation; (c) **Constrained Decoding**: Including lexicologically constrained beam search and token-level logits adjustment.

## Key Experimental Results

### GIST Dataset Statistics

| Metric | Arabic | Chinese | French | Japanese | Russian |
|--------|---------|------|------|------|------|
| Number of Terms | 4,844 | 6,426 | 6,527 | 4,770 | 5,167 |
| Unique English Words | 2,470 | 3,244 | 3,470 | 2,424 | 2,615 |
| Unique Target Words | 3,161 | 2,838 | 4,036 | 2,050 | 4,210 |
| English Words/Term | 2.02±0.59 | 2.05±0.68 | 2.07±0.67 | 2.02±0.58 | 2.01±0.59 |
| Target Characters/Term | 15.22±5.66 | 4.66±1.96 | 21.27±8.49 | 6.89±3.16 | 20.20±7.83 |

### Prompting Post-Translation Optimization Performance (60-60 Evaluation Set, BLEU Gain)

| Model | Arabic | Chinese | French | Japanese | Russian |
|------|---------|------|------|------|------|
| gpt-4o-mini | 23.58 → +1.07 | 32.64 → +1.60 | 40.80 → +3.08 | 21.46 → +0.64 | 17.25 → +1.07 |
| aya-expanse | 20.11 → +1.23 | 27.31 → +1.33 | 33.05 → +2.46 | 14.59 → +0.61 | 16.59 → +1.59 |
| nllb | 22.38 → +1.37 | 17.29 → +1.92 | 34.93 → +2.86 | 6.19 → +2.42 | 17.30 → +1.54 |
| seamless | 23.13 → +1.16 | 26.26 → +0.97 | 40.04 → +2.08 | 14.56 → +0.74 | 17.18 → +1.71 |
| aya-23-8B | 19.98 → +0.54 | 26.08 → +0.47 | 33.85 → +2.28 | 15.06 → +0.87 | 15.77 → +1.05 |

### GPT-4o Selection vs. Majority Voting (Task 1 Human Evaluation)

| Evaluation Result | Arabic | Chinese | French | Japanese | Russian |
|---------|---------|------|------|------|------|
| Both translations are good | 45.76% | 50.59% | 48.67% | 56.99% | 54.43% |
| GPT-4o selection is better | 28.54% | 28.76% | 30.44% | 24.37% | 30.26% |
| Majority voting is better | 20.37% | 17.62% | 18.89% | 15.44% | 13.04% |
| Both are bad | 4.46% | 2.70% | 1.89% | 2.62% | 1.22% |

### GIST vs. 60-60 Terminology Translation Quality (Task 2 Human Evaluation)

| Evaluation Result | Arabic | Chinese | French | Japanese | Russian |
|---------|---------|------|------|------|------|
| Both translations are good | 46.42% | 37.17% | 39.48% | 57.28% | 39.09% |
| GIST is better | 29.38% | 43.02% | 43.64% | 31.46% | 45.00% |
| 60-60 is better | 17.65% | 16.04% | 13.77% | 6.99% | 8.64% |
| Both are bad | 5.68% | 3.21% | 2.60% | 4.08% | 5.68% |

### Key Findings

1. **Prompting Method is Consistently Effective**: It improves translation quality across almost all languages, models, and evaluation metrics (BLEU/COMET/ChrF/ChrF++/TER), with statistical significance $p = 0.00$.
2. **Performance of Word Alignment Varies by Language**: It is effective for Chinese and Japanese (minimal morphological changes, direct replacement doesn't disrupt grammar), but sometimes degrades quality for Arabic, French, and Russian (which require morphological agreement such as gender, number, and case).
3. **GPT-4o Selection Outperforms Majority Voting**: Across all five languages, the translation candidates selected by GPT-4o are significantly better than those chosen by majority voting.
4. **GIST Translation Quality Outperforms 60-60**: In manual pairwise evaluations across all five languages, GIST translations are consistently and significantly better than the ACL 60-60 evaluation set.
5. **Adequate Dataset Coverage**: Rarefaction curve analysis shows that a 60% subset of the papers can cover over 80% of the terms ($t\text{-statistic} = 64.78$, $p = 0$). The domain distribution spans statistics (13.31%), mathematics (12.24%), CS (11.74%), NLP (11.50%), DS (9.98%), CV (6.57%), etc.

## Highlights & Limitations

### Highlights

- First large-scale (5K) multilingual AI terminology dataset, vastly exceeding the scale of 60-60 (~250 terms).
- A hybrid framework of LLM extraction + human translation + LLM validation balances efficiency and quality.
- The training-free post-translation integration scheme is highly practical.
- Interactive translation demo system provided on the ACL Anthology website (acl6060.org), allowing real-time comparison between original and optimized translations.
- Experiments cover 5 models × 5 languages × 5 metrics × 2 evaluation sets, demonstrating rigorous experimental design.

### Limitations

- It assumes a one-to-one mapping between English terms and target languages, overlooking cases where a single term may have multiple equally valid translations.
- It only covers 5 languages, falling far short of representing global linguistic diversity.
- The boundaries of the AI domain are blurry, making exhaustive terminology coverage impossible.
- Dataset updates rely on human judgment regarding the relevance and impact of new terms, and LLMs are constrained by knowledge cutoff dates, making full automation difficult.
- Constrained decoding and logits adjustment methods are about 100 times slower and yield poor generation quality (repeating terms or ignoring them entirely).

## Related Work & Insights

- **ACL 60-60 Initiative**: An evaluation set of only about 250 terms. GIST scales this up by ~20 times, and its translation quality is verified to be superior through manual evaluation.
- **Purely Manual Construction Methods**: Costly and hard to scale; GIST reduces costs via a hybrid LLM + crowdsourcing approach.
- **Fully Automated Methods**: Lacks accuracy; the three-model agreement rate is only ~15% (42.71% for Chinese).
- **Training-based Terminology Integration**: Requires data augmentation + fine-tuning or modifying model architecture, which cannot easily adapt to new terms; GIST's post-translation methods require no training.
- **Constrained Decoding Methods**: Flexible but slow, sacrificing translation accuracy; the prompting method is superior in both effectiveness and efficiency.
4. **Peculiarities of Chinese and Japanese**: The number of target characters per term in Chinese and Japanese is very small (4.66 and 6.89), reflecting the high information density of Chinese characters.
5. **Instability of Constrained Decoding Methods**: Although they guarantee the appearance of the target terms, they sometimes degrade overall translation fluency.

## Highlights & Insights

- The **hybrid human-AI collaborative pipeline** shows great scalability: LLMs handle large-scale extraction and screening, while human experts guarantee overall quality.
- The strategy of extracting terms from award-winning papers is clever—balancing representativeness and quality control.
- The training-free term integration methods carry great value for practical applications, especially given the simplicity of the prompting method.
- The developed ACL Anthology website demo showcases practical application scenarios.
- Covering the time span from 2000 to 2023 ensures the temporal completeness of the terms.

## Limitations & Future Work

- Only 5 target languages are covered; this can be extended to more languages (e.g., Korean, Hindi).
- Terms are sourced only from award-winning papers, which may miss important terms in non-award-winning papers.
- Maintaining temporal updates of term extraction and translation is a long-term challenge.
- Constrained decoding methods require customization based on model architectures, offering limited generalizability.
- The actual impact of terminology translation on downstream NLP tasks has not been evaluated.

## Related Work & Insights

- Compared with the ACL 60-60 initiative, GIST scales up by an order of magnitude.
- Complementary to the LLM terminology translation study by Feng et al. (2024)—GIST provides large-scale validation.
- The advantages of the prompting method in terminology integration echo the paradigm of LLM few-shot learning.
- Insight: The construction of domain-specific multilingual resources should employ a hybrid LLM + human pipeline, rather than relying purely on manual or purely automated approaches.

## Rating

- **Novelty**: ⭐⭐⭐ — The first large-scale multilingual dataset of AI terminology, with contributions primarily at the resource level.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multilingual coverage, automatic + human evaluation, and comparison of three integration methods.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured, with a detailed description of the dataset construction workflow.
- **Value**: ⭐⭐⭐⭐ — Holds practical significance for promoting the global popularization and multilingual accessibility of AI knowledge.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SIFT-50M: A Large-Scale Multilingual Dataset for Speech Instruction Fine-Tuning](sift-50m_a_large-scale_multilingual_dataset_for_speech_instruction_fine-tuning.md)
- [\[ACL 2025\] Group then Scale: Dynamic Mixture-of-Experts Multilingual Language Model](group_then_scale_dynamic_mixture-of-experts_multilingual_language_model.md)
- [\[ACL 2025\] LangMark: A Multilingual Dataset for Automatic Post-Editing](langmark_a_multilingual_dataset_for_automatic_post-editing.md)
- [\[ACL 2025\] LEMONADE: A Large Multilingual Expert-Annotated Abstractive Event Dataset for the Real World](lemonade_a_large_multilingual_expert-annotated_abstractive_event_dataset_for_the.md)
- [\[ACL 2025\] An Expanded Massive Multilingual Dataset for High-Performance Language Technologies (HPLT)](an_expanded_massive_multilingual_dataset_for_high-performance_language_technolog.md)

</div>

<!-- RELATED:END -->
