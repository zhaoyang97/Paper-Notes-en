---
title: >-
  [Paper Note] Culture Matters in Toxic Language Detection in Persian
description: >-
  [ACL 2025 (Main Track)][Social Computing][Toxic language detection] This paper systematically compares the performance of various methods (fine-tuning, data augmentation, zero/few-shot learning, cross-lingual transfer learning) in Persian toxic language detection, revealing that cultural similarity is a key factor determining the success of cross-lingual transfer learning—language data from culturally similar countries yields better transfer results.
tags:
  - "ACL 2025 (Main Track)"
  - "Social Computing"
  - "Toxic language detection"
  - "Persian"
  - "cultural factors"
  - "cross-lingual transfer learning"
  - "low-resource languages"
date: 2026-05-08
content_hash: b788c3f00817af66
---

# Culture Matters in Toxic Language Detection in Persian

**Conference**: ACL 2025 (Main Track)  
**arXiv**: [2506.03458](https://arxiv.org/abs/2506.03458)  
**Code**: None  
**Area**: Social Computing  
**Keywords**: Toxic language detection, Persian, cultural factors, cross-lingual transfer learning, low-resource languages

## TL;DR

This paper systematically compares the performance of various methods (fine-tuning, data augmentation, zero/few-shot learning, cross-lingual transfer learning) in Persian toxic language detection, revealing that cultural similarity is a key factor determining the success of cross-lingual transfer learning—language data from culturally similar countries yields better transfer results.

## Background & Motivation

**Background**: Toxic language detection (including hate speech, insults, cyberbullying, etc.) is an important task for maintaining a safe online environment. In English, toxic language detection already has a large number of datasets and models, with relatively mature performance. However, for non-English languages, especially low- and medium-resource languages, this task still faces serious challenges.

**Limitations of Prior Work**: Persian (Farsi), as the official language of countries like Iran, has over 100 million speakers, but research on toxic language detection in Persian is extremely limited. The main issues include: (1) marked scarcity of annotated Persian data and a lack of high-quality toxic language datasets; (2) linguistic features of Persian (such as right-to-left writing, complex morphology, and a large number of metaphors and euphemisms) posing additional detection difficulties; (3) more critically, the definition of toxic language is itself culturally subjective—expressions considered offensive in one culture may be completely normal in another. Directly transferring English toxic language detection models to Persian may overlook these cultural differences.

**Key Challenge**: Cross-lingual transfer learning is a common strategy for solving tasks in low-resource languages, but toxic language detection possesses strong cultural dependence. If the cultural backgrounds of the source and target languages differ greatly, transfer performance may be severely degraded. However, existing research rarely systematically explores the impact of cultural factors on cross-lingual toxic language detection transfer.

**Goal**: To comprehensively compare the effectiveness of different methods in Persian toxic language detection, and particularly to deeply analyze the role played by cultural similarity in cross-lingual transfer learning, providing empirical guidance for toxic content detection in low-resource languages.

**Key Insight**: Persian is chosen as the target language, paired with carefully selected source languages that are either culturally close or distant to Iran for cross-lingual transfer experiments. Arabic and Turkish share Islamic cultural backgrounds, similar social norms, and some vocabulary with Persian, whereas English and European languages exhibit significant cultural differences. By comparing the transfer performance of these various source languages, the influence of cultural factors can be directly validated.

**Core Idea**: In cross-lingual transfer learning, the cultural similarity of languages is a stronger predictor of transfer performance in toxic language detection than purely linguistic similarity—culturally close languages as source languages yield superior detection performance in the target language.

## Method

### Overall Architecture

This paper adopts a systematic comparative experimental methodology covering five major methodological paradigms: (1) directly fine-tuning pre-trained models on Persian data; (2) expanding Persian training sets via data augmentation; (3) utilizing large language models for zero-shot and few-shot detection; (4) transferring from toxic language datasets of other languages to Persian; (5) hybrid methods combining multiple strategies. The input is Persian text, and the output is a binary classification label of toxic/non-toxic (with multi-class classification considered in some experiments).

### Key Designs

1. **Multilingual Model Fine-tuning**:

    - **Function**: Direct fine-tuning of multilingual pre-trained models on annotated Persian data.
    - **Mechanism**: Standard classification fine-tuning is performed on collected Persian toxic language datasets using multilingual pre-trained models such as XLM-RoBERTa and mBERT as backbones. Additionally, the Persian-specific pre-trained model ParsBERT is tested. The experiments explore different training strategies, including full-parameter fine-tuning and fine-tuning only the classification head.
    - **Design Motivation**: Fine-tuning is the most straightforward approach, but is limited by the scale of annotated Persian data. Comparing general multilingual models with Persian-specific models highlights the importance of domain adaptation.

2. **Cross-lingual Transfer with Cultural Analysis**:

    - **Function**: Validating the impact of source languages with different cultural backgrounds on the transfer performance of Persian toxic language detection.
    - **Mechanism**: Toxic language datasets from various source languages are selected to train the models, which are then evaluated on Persian data. Source languages are categorized into two groups based on cultural distance from Persian: (a) culturally close group—Arabic (shared Islamic culture, lexical loanwords), Turkish (regional cultural ties, similar social norms); (b) culturally distant group—English (Western cultural background), other European languages. The impact of cultural factors is quantified by comparing transfer performance between the two groups. Standardizing the multilingual model architecture across all experiments ensures that differences stem from data rather than models.
    - **Design Motivation**: This represents the core contribution of the paper—rather than simply performing cross-lingual transfer, it carefully designs experiments to isolate the effects of cultural factors, filling an empirical gap in the field.

3. **Zero/Few-shot LLM Evaluation**:

    - **Function**: Evaluating the zero-shot and few-shot capabilities of large language models on Persian toxic language detection.
    - **Mechanism**: Large language models (such as GPT-4 and GPT-3.5) are used to perform toxic language detection via carefully constructed prompts containing definitions and examples of Persian toxic language. Only the task definition is provided in the zero-shot setting, while 2-5 annotated examples are additionally provided in the few-shot setting. Explicit accounts of Persian cultural background are integrated into the prompts to help the models comprehend the cultural context.
    - **Design Motivation**: The zero-shot capabilities of LLMs offer a label-free solution for low-resource scenarios, yet their ability to comprehend toxic language in non-English cultures remains an open question.

### Loss & Training

Fine-tuning experiments utilize standard cross-entropy loss. To address class imbalance issues (toxic examples are often far fewer than benign ones), weighted loss and oversampling strategies are explored. The learning rate search range is 1e-5 to 5e-5 using the AdamW optimizer. Models are trained for 5-10 epochs with early stopping implemented.

## Key Experimental Results

### Main Results

Comparative F1 scores of different methodological paradigms in Persian toxic language detection.

| Method | Model/Source Language | F1 (%) | Description |
|------|-----------|--------|------|
| Fine-tuning | ParsBERT | 76.8 | Persian-specific model |
| Fine-tuning | XLM-RoBERTa | 74.5 | General multilingual model |
| Cross-lingual Transfer | Arabic | 71.3 | Culturally close |
| Cross-lingual Transfer | Turkish | 69.7 | Culturally close |
| Cross-lingual Transfer | English | 62.4 | Culturally distant |
| Zero-shot | GPT-4 | 67.2 | No training data |
| Zero-shot | GPT-3.5 | 58.1 | No training data |
| Few-shot (5-shot) | GPT-4 | 71.8 | 5 examples |
| Data Augmentation + Fine-tuning | ParsBERT | 78.5 | Optimal combination |

### Quantitative Analysis of Cultural Factors in Cross-lingual Transfer

| Source Language | Cultural Distance | Zero-shot Transfer F1 | Fine-tuned Transfer F1 | Shared Cultural Features |
|--------|---------|-------------|------------|------------|
| Arabic | Close | 71.3 | 75.2 | Islamic culture, loanwords, similar taboos |
| Turkish | Close | 69.7 | 73.8 | Regional ties, overlapping social norms |
| Hindi | Medium | 64.5 | 69.1 | Partial cultural exchanges but clear differences |
| English | Distant | 62.4 | 66.3 | Substantial difference in cultural background |
| German | Distant | 60.8 | 65.1 | Farthest cultural distance |

### Key Findings

- Transfer performance of culturally close languages (Arabic, Turkish) is significantly superior to culturally distant languages (English, German), with a gap of approximately 8-10% in F1, strongly validating the hypothesis that "cultural factors influence transfer performance."
- The Persian-specific model (ParsBERT) outperforms the general multilingual model in fine-tuning scenarios, indicating that language-specific pre-training remains vital for low-resource settings.
- Under the few-shot setting (71.8% F1), GPT-4 approaches the transfer performance of culturally close source languages (71.3%), demonstrating that LLMs possess a certain degree of understanding of Persian toxic language.
- The data augmentation strategy provides an additional gain of about 1.5-2% over the fine-tuning baseline, showing it is an effective auxiliary means.
- Analysis of toxic language types shows that for culturally specific forms of insults (such as those related to religion or family honor), the transfer advantage of culturally close languages is even more pronounced.

## Highlights & Insights

- **Cultural distance as a transfer predictor**: The paper provides compelling empirical evidence through meticulous comparative experiments, proving that cultural similarity is a better predictor of transfer performance in cross-lingual toxic language detection than purely linguistic similarity. This finding provides valuable guidance for all NLP tasks involving cultural subjectivity.
- **A practical guide for low-resource languages**: The paper essentially provides a methodological selection guide for toxic language detection in low-resource languages: fine-tuning domain-specific models when annotated data is available, choosing culturally close source languages for transfer when labeled data is absent, and utilizing few-shot GPT-4 as a rapid solution in urgent scenarios.
- **In-depth analysis of culturally specific toxic language**: Rather than conducting only classification experiments, the paper analyzes culturally specific forms of Persian toxic language (such as attacks on family honor, religious blasphemy, etc.), which are difficult to detect via transfer from English data.

## Limitations & Future Work

- The experiments focus solely on Persian as the target language; whether the influence of cultural factors holds true for other language pairs requires broader validation.
- The measurement of cultural distance is currently qualitative (subjectively determined by researchers) and lacks a quantified metric for cultural distance.
- The limited scale of the Persian toxic language dataset may restrict the performance ceiling of all evaluated methods.
- The performance of multi-source joint transfer has not been explored—i.e., whether combining multiple culturally close languages can yield further improvements.
- The standards of toxic language themselves evolve over time, requiring the model to be continuously updated to adapt to changing social norms.

## Related Work & Insights

- **vs HateCheck/ToxiGen**: These works provide functional test sets for English toxic language detection. This paper provides a similar systematic evaluation for Persian but places a stronger emphasis on the cultural dimension.
- **vs XLM-R Cross-lingual Transfer**: While XLM-R performs exceptionally well on various cross-lingual tasks, this paper reveals that toxic language detection differs from general NLU tasks—cultural factors, rather than linguistic ones, act as the bottleneck for transfer.
- **vs Perspective API**: Google's Perspective API is an industry-grade toxic content detection tool but has limited support for low-resource languages. The research in this paper offers empirical guidance on source language selection for the multilingual extension of such tools.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically and quantitatively demonstrates the impact of cultural factors on cross-lingual transfer, offering an insightful and novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers five methodological paradigms, multiple source languages, and models, providing comprehensive comparisons alongside quantitative analyses of cultural factors.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, rigorous experimental design logic, and excellent correspondence between conclusions and empirical data.
- Value: ⭐⭐⭐⭐ Provides valuable empirical guidance on methodological selection for toxic language detection in low-resource languages, with findings regarding cultural factors holding generalizable significance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring Multimodal Challenges in Toxic Chinese Detection: Taxonomy, Benchmark, and Findings](exploring_multimodal_challenges_in_toxic_chinese_detection_taxonomy_benchmark_an.md)
- [\[ACL 2025\] BiasGuard: A Reasoning-Enhanced Bias Detection Tool for Large Language Models](biasguard_a_reasoning-enhanced_bias_detection_tool_for_large_language_models.md)
- [\[ACL 2025\] Detection of Human and Machine-Authored Fake News in Urdu](detection_of_human_and_machine-authored_fake_news_in_urdu.md)
- [\[ACL 2025\] ImpliHateVid: Implicit Hate Speech Detection in Videos](implihatevid_video_hate.md)
- [\[ACL 2025\] Translate With Care: Addressing Gender Bias, Neutrality, and Reasoning in Large Language Model Translations](translate_with_care_addressing_gender_bias_neutrality_and_reasoning_in_large_lan.md)

</div>

<!-- RELATED:END -->
