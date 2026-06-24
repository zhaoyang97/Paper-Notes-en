---
title: >-
  [Paper Note] Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation
description: >-
  [ACL2025][Multilingual & Machine Translation][Cross-lingual Transfer] This paper systematically investigates the cross-lingual transfer effects of English debiasing/detoxification fine-tuning across 7 LLMs and 20 languages. The study finds that SFT is effective for debiasing and DPO for detoxification, but transfer to non-English languages is generally accompanied by a decline in language generation capabilities (impaired language consistency, fluency…
tags:
  - "ACL2025"
  - "Multilingual & Machine Translation"
  - "Cross-lingual Transfer"
  - "Debiasing"
  - "Detoxification"
  - "Multilingual LLMs"
  - "DPO"
  - "Supervised Fine-Tuning"
date: 2026-05-08
content_hash: f3caf82a205cc311
---

# Cross-Lingual Transfer of Debiasing and Detoxification in Multilingual LLMs: An Extensive Investigation

**Conference**: ACL2025  
**arXiv**: [2412.14050](https://arxiv.org/abs/2412.14050)  
**Code**: [GitHub](https://github.com/Veranep/crosslingualdetoxdebias)  
**Area**: Multilingual Translation  
**Keywords**: Cross-lingual Transfer, Debiasing, Detoxification, Multilingual LLMs, DPO, Supervised Fine-Tuning

## TL;DR

This paper systematically investigates the cross-lingual transfer effects of English debiasing/detoxification fine-tuning across 7 LLMs and 20 languages. The study finds that SFT is effective for debiasing and DPO for detoxification, but transfer to non-English languages is generally accompanied by a decline in language generation capabilities (impaired language consistency, fluency, and diversity). Furthermore, the transfer performance can be predicted by the pre-training data volume of the target language.

## Background & Motivation

**Background**: Generative LLMs are primarily trained on English data but are used by a global, multilingual user base. Consequently, models exhibit higher social bias and toxicity in non-English languages.

**Limitations of Prior Work**: Separately fine-tuning debiasing/detoxification for each language is highly costly and requires language-specific datasets (often obtained only through translation from English). While existing research demonstrates that English fine-tuning can transfer cross-lingually, its side effects on generation quality have not been fully investigated.

**Key Challenge**: Cross-lingual transfer of English debiasing/detoxification is a double-edged sword: although it reduces biases/toxicity in other languages, catastrophic forgetting may severely impair the generation capabilities of non-English languages (e.g., models may start replying in English to non-English prompts).

**Goal**: To systematically compare the cross-lingual transfer performance of different fine-tuning methods (SFT vs. DPO) on both debiasing and detoxification tasks, while quantifying their negative impacts on language generation capabilities.

**Key Insight**: Concurrently evaluate three dimensions—bias, toxicity, and language capability—within a unified experimental framework, covering a large-scale combination of 7 models $\times$ 4 datasets $\times$ 20 languages.

**Core Idea**: The success of cross-lingual debiasing/detoxification transfer depends on the proportion of the target language in the pre-training data, and successful transfer is almost always accompanied by the degradation of language generation capabilities. Therefore, directly mitigating bias/toxicity in the target language should be prioritized.

## Method

### Overall Architecture

**Fine-tuning Methods**:
1. **Supervised Fine-Tuning (SFT)**: Fine-tuning the model on harmless texts.
2. **Direct Preference Optimization (DPO)**: Utilizing preference data containing harmful/harmless contrasts to simultaneously maximize the probability of harmless outputs and minimize the probability of harmful outputs.

**Fine-tuning Datasets** (all in English):

| Type | Debiasing | Detoxification |
|:--|:--|:--|
| SFT | Panda (95K samples) | Jigsaw (95K samples) |
| DPO | BiasDPO (1.1K samples) | DetoxDPO (25K samples) |

- Additional control experiment: SFT on the preferred completions of the DPO dataset to decouple the effects of the fine-tuning method from the dataset.

**Model Selection** (7 instruction-tuned models):
- Aya 23 8B, Aya Expanse 8B (multilingual design, 23 languages)
- Gemma 2 2B IT, Gemma 2 9B IT
- Llama 3 8B Instruct, Llama 3.1 8B Instruct (8 languages)
- Mistral 7B Instruct v0.3

Parameter-efficient fine-tuning is performed using QLoRA.

### Key Designs

**Three-Dimensional Evaluation System**:

**Bias Evaluation** (3 benchmarks):
- CrowS-Pairs: Minimal-pair sentences covering 9 bias types (ideal score is 50)
- StereoSet: Gender/occupation/race/religion biases (ideal score is 50)
- MBBQ: Question-answering-based bias evaluation containing ambiguous and disambiguated contexts (ideal score is 0)

**Toxicity Evaluation**:
- RTP-LX: Multilingual translated version of RealToxicityPrompts
- 25 continuations generated per prompt using nucleus sampling ($T=0.9$, $\text{top-}p=0.8$)
- Scored via Perspective API, calculating Expected Maximum Toxicity (EMT)

**Language Generation Capability Evaluation** (4 metrics):
- Language Consistency: Whether the generated text is in the same language as the prompt (Tatoeba 1000 sentences/language + fastText detection)
- Fluency: Conditional perplexity calculated via mT5-XL (taking the median to avoid extreme values)
- Diversity: The proportion of distinct unigrams that do not appear in the input
- QA Capability: Global-MMLU 5-shot evaluation

### Initial Model Selection
Models with excessively low language consistency (Llama 3 Instruct with only 15.1%, Mistral with 30.5%, and Gemma 2 2B with 54.4%) were excluded. Ultimately, Aya 23, Aya Expanse, Llama 3.1 Instruct, and Gemma 2 9B IT were selected for the fine-tuning experiments.

## Key Experimental Results

### Initial Model Bias/Toxicity Levels

| Model | CrowS-Pairs | StereoSet | Toxicity (EMT) | Language Consistency |
|:--|:--|:--|:--|:--|
| Aya 23 | 57.90±5.25 | 51.89±0.91 | **0.541±0.066** | 72.3% |
| Gemma 2 9B IT | **62.19±4.88** | 53.17±1.14 | 0.481±0.075 | 82.7% |
| Llama 3.1 Instruct | 59.30±6.06 | **57.41±2.45** | 0.539±0.069 | 80.8% |

### Debiasing Performance

**SFT on Panda (Most Effective)**:
- Almost universally reduces bias scores across the three benchmarks: CrowS-Pairs, StereoSet, and MBBQ.
- Age and gender biases (types covered by the Panda dataset) decrease most significantly.
- **Cost**: Question-answering capability, language consistency, and diversity all decline drastically.

**DPO on BiasDPO**:
- Effective only for the Aya series models, with limited efficacy on other models (likely because the dataset size of 1.1K is too small).
- Advantage: Diversity increases instead.

### Detoxification Performance

**DPO on DetoxDPO (Only Effective Method)**:
- Reduces toxicity in English as well as all non-English languages.
- The toxicity reduction evaluated via GPT-4o is consistent across all languages.

**SFT on Jigsaw / DetoxDPO (Counterproductive)**:
- Toxicity unexpectedly increases! The Jigsaw dataset only contains non-toxic comments, which fails to neutralize toxicity in the model.

### Cross-Lingual Discrepancies

- Indo-European Latin-script languages (French, Portuguese, German, Swedish) show the best transfer performance.
- Low-resource languages (Dutch, Maltese, Catalan) exhibit poor transfer performance.
- The data volume of the target language in the pre-training data is moderately and significantly correlated with the transfer performance ($-0.6 < r < -0.4$), providing better predictive power than bilingual sentence similarity.

### Language Generation Capability Degradation

- Following SFT fine-tuning, the language consistency of non-English languages generally drops drastically (the model starts replying in English to non-English prompts).
- As shown in Figure 1: after English detoxification, the generation for a German prompt is no longer toxic, but it is written in English.
- DPO shows less degradation in language consistency, though the fluency of Aya models declines.

### Key Findings
1. Debiasing and detoxification require different methods: SFT is suitable for debiasing, whereas DPO is suitable for detoxification.
2. When cross-lingual transfer succeeds, **it is almost inevitably accompanied by the degradation of at least one language generation metric**.
3. The transfer of bias mitigation is category-specific: bias types covered in the fine-tuning dataset exhibit better transfer effects.
4. During cross-lingual transfer to other languages, bias types not covered by the dataset can also achieve some degree of mitigation ("spillover effect").

## Highlights & Insights

1. **Prominent Experimental Scale**: 7 models $\times$ 4 datasets $\times$ 2 fine-tuning methods $\times$ 20 languages. This constitutes the most comprehensive cross-lingual debiasing/detoxification study in this field to date.
2. **Crucial Findings from Three-Dimensional Evaluation**: It is the first to systematically quantify the trade-off between bias/toxicity mitigation and language capability degradation, rather than merely reporting reductions in bias scores.
3. **Exemplary Decoupling of Methodology**: By conducting SFT control experiments on DPO datasets, the study isolates the effects of the fine-tuning method from those of the dataset.
4. **Predictive Power of Pre-training Data Volume**: It predicts cross-lingual transfer performance better than bilingual sentence similarity, offering a simple and actionable decision-making guide for practical applications.
5. **Valuable Qualitative Analysis**: It reveals interesting behavioral differences among different fine-tuning methods (e.g., DPO models tend to treat toxic prompts as "quotes").

## Limitations & Future Work

1. Bias benchmarks are mainly translated based on US cultural stereotypes, potentially overlooking culture-specific biases.
2. The Perspective API is a black-box model, which possesses inherent biases and is continuously updated, affecting reproducibility.
3. GPU resource constraints restricted some models to being trained for only a limited number of epochs or on subsets of the data.
4. Language coverage is constrained by the benchmark datasets, excluding some low-resource languages that warrant attention.
5. CrowS-Pairs covers only 8 non-English languages, which limits the statistical significance of the cross-lingual correlation analysis.

## Related Work & Insights

- **Relationship with Li et al. (2024)**: The latter found that DPO detoxification achieves cross-lingual transfer by reducing activations in language-agnostic regions. This work extends the validation across more models and tasks, and reveals that the predictive power of bilingual sentence similarity does not hold for most models (it is only effective for Gemma).
- **Relationship with Catastrophic Forgetting Research**: Meade et al. (2022) observed that fine-tuning-based debiasing results in minor forgetting; this work further demonstrates that the forgetting issue is much more severe in cross-lingual scenarios.
- **Insights**: (1) Priority should be given to developing debiasing/detoxification datasets in target languages, rather than relying on English transfer. (2) The "quoting strategy" behavior of DPO suggests that the model might have learned evasion rather than a true understanding of toxicity. (3) Pre-training data volume can serve as a practical metric to decide whether cross-lingual transfer is dependable.

## Rating

- Novelty: ⭐⭐⭐ — The research question is not entirely new, but the system scale and the three-dimensional evaluation framework are key contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 7 models $\times$ 20 languages $\times$ multiple benchmarks $\times$ multiple fine-tuning methods, with comprehensive ablation and correlation analyses.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, with highly logical analysis and visualization of experimental results.
- Value: ⭐⭐⭐⭐ — Significant practical guidance for the safe deployment of multilingual LLMs. The conclusion that "transfer comes with a cost" deserves widespread attention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs](mid_layer_crosslingual_alignment.md)
- [\[ACL 2025\] Cross-Lingual Optimization for Language Transfer in Large Language Models](cross-lingual_optimization_for_language_transfer_in_large_language_models.md)
- [\[ACL 2025\] Cross-Lingual Transfer of Cultural Knowledge: An Asymmetric Phenomenon](cross-lingual_transfer_of_cultural_knowledge_an_asymmetric_phenomenon.md)
- [\[ACL 2025\] Semantic Aware Linear Transfer by Recycling Pre-trained Language Models for Cross-Lingual Transfer](semantic_aware_linear_transfer_by_recycling_pre-trained_language_models_for_cros.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)

</div>

<!-- RELATED:END -->
