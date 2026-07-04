---
title: >-
  [Paper Note] Lost in Literalism: How Supervised Training Shapes Translationese in LLMs
description: >-
  [ACL 2025][LLM (Other)][Machine Translation] This paper systematically investigates the phenomenon of translationese in machine translation compiled by Large Language Models (LLMs), revealing that translationese bias in supervised fine-tuning (SFT) data is the root cause of unnatural translations in LLMs. To mitigate this issue, the paper proposes approaches that polish training reference translations and filter out unnatural training instances.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Machine Translation"
  - "Translationese"
  - "Supervised Fine-Tuning"
  - "Data Quality"
  - "Naturalness"
date: 2026-05-08
content_hash: 5d93d877ec38f888
---

# Lost in Literalism: How Supervised Training Shapes Translationese in LLMs

**Conference**: ACL 2025  
**arXiv**: [2503.04369](https://arxiv.org/abs/2503.04369)  
**Code**: [github](https://github.com/yafuly/LLM_Translationese)  
**Area**: LLM/NLP  
**Keywords**: Machine Translation, Translationese, Supervised Fine-Tuning, Data Quality, Naturalness

## TL;DR

This paper systematically investigates the phenomenon of translationese in machine translation compiled by Large Language Models (LLMs), revealing that translationese bias in supervised fine-tuning (SFT) data is the root cause of unnatural translations in LLMs. To mitigate this issue, the paper proposes approaches that polish training reference translations and filter out unnatural training instances.

## Background & Motivation

Although Large Language Models have achieved outstanding performance in machine translation tasks, the issue of "translationese" remains prevalent. Translationese refers to translated texts that are translated too literally at the phrase or sentence level, deviating from the natural expression habits of the target language and making the translation sound unnatural to native speakers.

A key paradox is that while LLMs are exposed to a vast amount of natural language data during pre-training and should theoretically possess the ability to generate natural text, they still produce unnatural translations in practice. For instance, when translating the English phrase "suffer night blindness" into Chinese, the model generates a literal translation instead of a more natural expression. The authors argue that this "unexpected" unnatural translation stems from biases introduced during the supervised fine-tuning phase.

More interestingly, when LLMs are tasked with "polishing" existing translations, they generate significantly more natural outputs. This further confirms that LLMs inherently possess the capability to produce natural translations, but this potential is not fully unlocked under the "translation" task format.

## Method

### Overall Architecture

The workflow of this paper consists of three core components: (1) systematically evaluating the phenomenon of translationese in LLM translations; (2) tracing the origins of translationese in supervised training data; and (3) proposing training strategies to mitigate translationese.

### Key Designs

1. **Translationese Annotation and Quantification (TSR Metric)**: The authors collected documents from four writing domains: news, academic papers, Wikipedia, and social media. Using models like ALMA, GPT-3.5/4, and Mistral, they performed English-to-Chinese and German-to-English translations. Three professional translators were invited to annotate translationese errors (including unnatural sentence flow and unnatural phrase flow) on the Label Studio platform, calculating the Translationese Span Ratio (TSR) to quantify the severity of translationese.

2. **Correlation between Perplexity and Translationese**: Utilizing Llama-3.1-8B to calculate the perplexity (PPL) of translations, the study found a positive correlation between PPL and manually annotated TSR—the higher the perplexity, the more severe the translationese. This both validates the hypothesis that LLMs inherently prefer natural generation and provides an automatic metric for detecting translationese.

3. **Translationese Analysis of Training Data**: The authors sampled 500 English-to-Chinese and German-to-English translation instances from the ALMA training set and had professional translators annotate translationese. The results reveal that over 34% of the training instances exhibit translationese patterns (40.4% in English-to-Chinese, 34.2% in German-to-English). This indicates that during SFT, LLMs are guided to understand "translation" as a direct mapping from source to target language, overemphasizing fidelity at the expense of naturalness.

4. **SFT-Polished (Polished Training References)**: This approach leverages the "polishing" capabilities of GPT-4 to improve translation references in training data. Unlike having GPT-4 translate directly (SFT-KD), SFT-Polished instructs GPT-4 to polish existing translations, which preserves the original translation quality while enhancing naturalness. The key insight of this method is that while LLMs may generate translationese under the "translation" task frame, they can leverage their advantages in natural language generation when given a "polishing" task frame.

5. **Filtering Unnatural Training Instances**: Using perplexity as a metric for naturalness, the training instances are ranked and the most unnatural subset is removed. Experiments show that filtering out 20% of the instances improves both translation naturalness and translation quality.

### Loss & Training

The training employs a standard supervised fine-tuning (SFT) workflow based on the training configurations of ALMA, utilizing parallel training data from WMT'17 to WMT'21 and Flores-200 (totaling 31,621 instances). Llama-3.1-8B and Qwen-2.5-7B are selected as base models. The core strategic difference lies in the preprocessing of training data:
- **SFT**: Training with the original reference translations.
- **SFT-KD**: Replacing reference translations with translation results directly generated by GPT-4.
- **SFT-Polished**: Using reference translations polished by GPT-4.

## Key Experimental Results

### Main Results

| Training Method | Metric | Llama-3.1-8B (En-Zh) | Qwen-2.5-7B (En-Zh) |
|---------|------|----------------------|---------------------|
| SFT | PPL(Doc) | 13.8 | 13.8 |
| SFT-KD | PPL(Doc) | 14.3 | 13.9 |
| SFT-Polished | PPL(Doc) | **11.9** | **12.1** |
| SFT | PPL(Sent) | 103.3 | 101.6 |
| SFT-Polished | PPL(Sent) | **90.0** | **87.3** |

Translation quality evaluation (COMET-QE):

| Training Method | Llama En-Zh | Llama De-En | Qwen En-Zh | Qwen De-En |
|---------|------------|------------|------------|------------|
| SFT | 80.0 | 80.5 | 73.8 | 74.0 |
| SFT-Polished | **81.8** | 81.0 | 74.2 | **75.6** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Filtering 0% (Baseline) | PPL Baseline | No filtering |
| Filtering 20% | Naturalness ↑ Translation Quality ↑ | Optimal balance point |
| Filtering 40% | Naturalness ↑ Translation Quality ↓ | Naturalness continues to improve but quality begins to drop |
| SFT-KD vs SFT-Polished | KD brings no improvement, Polished significantly improves | Proves that polishing, rather than distillation, is key |

### Key Findings

- All LLMs exhibit significant translationese; even for GPT-4, over 40% of its translations show clear translationese patterns.
- Having the LLM polish its own translations (GPT-4 Polishing) reduces the ratio of translationese documents from 43% to 25%.
- Sentence-level translationese is more prevalent than phrase-level translationese (annotation count ratio is approximately 2:1).
- Simply adding style requirements to the prompt (Specified) cannot effectively reduce translationese and may even worsen it.
- SFT-Polished consistently improves translation naturalness across both base models and both translation directions.
- Polishing during the training phase yields better results than post-polishing during the inference phase.

## Highlights & Insights

1. **Deep Insight**: The root cause of translationese does not lie in the deficiency of LLM capabilities, but rather in the biases within the training data during the SFT phase. While LLMs acquire the ability to generate natural text through pre-training, the "translation" task format activates a literal translation mode that overemphasizes fidelity.
2. **Simple and Effective Solution**: The method of polishing training data is both intuitive and highly effective, requiring no modifications to the model architecture or the training pipeline.
3. **Perplexity as a Translationese Metric**: This work establishes a quantitative correlation between perplexity and translationese, delivering a practical tool for automatic detection of translationese.
4. **Importance of Task Format**: The discovery that "translation" and "polishing" task formats exert fundamentally different impacts on the naturalness of LLM generation highlights the critical importance of task framing.

## Limitations & Future Work

- The primary experiments are conducted on English-to-Chinese and German-to-English translation directions. Although there are generalization experiments on more languages, the coverage remains limited.
- Polishing relies on powerful models like GPT-4, increasing the cost of data preparation.
- While perplexity is effective as a translationese metric, it might confound with other dimensions of text quality.
- The paper does not thoroughly explore differentiated processing strategies for different types of translationese (sentence-level vs. phrase-level).
- Filtering out unnatural instances may inadvertently remove valuable training signals as well.

## Related Work & Insights

- This work introduces the concept of translationese from traditional machine translation quality research into the LLM era, representing the first systematic study of LLM-generated translationese.
- It complements existing works on translation-specific LLMs (such as ALMA), focusing on translation style rather than merely translation accuracy.
- It offers valuable insights for the data-centric AI paradigm, proving that training data quality impacts not only accuracy but also generation style.
- It provides a novel perspective and methodology for SFT data cleaning and preprocessing.

## Rating

- Novelty: ⭐⭐⭐⭐ It is the first systematic study of translationese in LLMs, presenting a unique perspective, though the solution is relatively straightforward.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluations covering human evaluation, automatic metrics, multiple models, and multiple language directions.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical argumentation is clear, progressing elegantly from phenomenal observation to causal analysis and ultimately to proposed solutions.
- Value: ⭐⭐⭐⭐ It reveals an overlooked yet significant problem in LLM translation, offering a highly practical and reproducible methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] How LLMs Comprehend Temporal Meaning in Narratives: A Case Study in Cognitive Evaluation of LLMs](how_llms_comprehend_temporal_meaning_in_narratives_a_case_study_in_cognitive_eva.md)
- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [\[ACL 2025\] SSUF: A Semi-supervised Scalable Unified Framework for E-commerce Query Classification](a_semi-supervised_scalable_unified_framework_for_e-commerce_query_classification.md)
- [\[ACL 2025\] How Humans and LLMs Organize Conceptual Knowledge: Exploring Subordinate Categories in Italian](conceptual_knowledge_org.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)

</div>

<!-- RELATED:END -->
