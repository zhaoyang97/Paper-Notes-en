---
title: >-
  [Paper Note] Detection of Human and Machine-Authored Fake News in Urdu
description: >-
  [ACL 2025][Social Computing][Fake News Detection] This paper proposes a 4-way fake news detection task for Urdu (Human Fake / Human True / Machine Fake / Machine True), constructs the first Urdu machine-generated news dataset, and introduces a hierarchical detection framework that decomposes the 4-way classification into two sub-tasks: machine-generated text detection and fake news detection. It consistently outperforms baselines in both in-domain and cross-domain settings.
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Fake News Detection"
  - "Urdu"
  - "Machine-Generated Text"
  - "Hierarchical Classification"
  - "Low-Resource Languages"
date: 2026-05-08
content_hash: 2c251d469f48d04c
---

# Detection of Human and Machine-Authored Fake News in Urdu

**Conference**: ACL 2025  
**arXiv**: [2410.19517](https://arxiv.org/abs/2410.19517)  
**Code**: [GitHub](https://github.com/zainali93/UrduHMFND2024)  
**Area**: Fake News Detection / Low-Resource NLP  
**Keywords**: Fake News Detection, Urdu, Machine-Generated Text, Hierarchical Classification, Low-Resource Languages

## TL;DR

This paper proposes a 4-way fake news detection task for Urdu (Human Fake / Human True / Machine Fake / Machine True), constructs the first Urdu machine-generated news dataset, and introduces a hierarchical detection framework that decomposes the 4-way classification into two sub-tasks: machine-generated text detection and fake news detection. It consistently outperforms baselines in both in-domain and cross-domain settings.

## Background & Motivation

Fake news detection faces a dual challenge in the active era of social media:

**New Threats in the LLM Era**: Models like ChatGPT can generate high-quality, error-free disinformation, rendering traditional detection methods based on linguistic cues increasingly obsolete. Concurrently, journalists and media organizations also utilize LLMs, further blurring the line between real and fake news.

**Plight of Low-Resource Languages**: Current detectors primarily focus on binary classification tasks in English, leaving low-resource languages like Urdu severely under-researched. Existing Urdu fake news datasets only contain human-written texts, failing to address the challenges posed by machine-generated content.

**Limitations of Classification Schemas**: Traditional binary classification (true/fake) cannot differentiate between machine-generated true news and fake news. Under a 4-way classification scenario, direct multi-class classifiers perform poorly on machine-generated categories.

The core observation of this paper is: **the 4-way classification problem inherently comprises two independent sub-tasks—"who wrote it" (Human/Machine) and "is it true" (True/Fake). Decomposing them can enhance the accuracy of each sub-task.**

## Method

### Overall Architecture

The overall process consists of three stages: (1) using GPT-4o to generate machine-written versions of four existing Urdu fake news datasets to construct a 4-way classification dataset; (2) benchmarking against baselines (LSVM, 4-way fine-tuned xlm-RoBERTa); (3) proposing a hierarchical approach to decompose the 4-way classification into two binary sub-tasks.

### Key Designs

1. **Dataset Construction - Machine-Generated News**:

    - Using GPT-4o to rewrite each article in four existing datasets, maintaining the same narrative stance.
    - Designing 5 different prompts (1 human-written + 4 GPT-4o generated), randomly assigned to each article.
    - The original label changes from True/Fake to Human True/Human Fake, and the machine-generated version receives the Machine True/Machine Fake label.
    - **Quality Control**: Native speaker review + 20% token count threshold filtering. Three types of issues were identified: failure to rewrite (GPT-4o requesting input), short text hallucinations, and generation of prefixes. These were resolved through prompt engineering.

2. **Four Datasets Covering Diverse Scenarios**:

    - Dataset 1 (Ax-to-Grind): 10,083 samples, short headlines, 15 domains.
    - Dataset 2 (UFN2023): 4,097 samples, short headlines, 9 domains.
    - Dataset 3 (UFN Augmented): 2,000 samples, long articles, translated from English.
    - Dataset 4 (Bend the Truth): 1,300 samples, long articles, counterfactual rewrites by journalists.

3. **Hierarchical Detection Architecture**:
   Core Innovation - Decomposing 4-way classification into two sub-tasks:
    - **First Layer: Machine-Generated Text (MGT) Detection**: Labels simplified to Human/Machine, fine-tuning xlm-RoBERTa-base.
    - **Second Layer: Fake News Detection**: Labels simplified to Fake/True, fine-tuning xlm-RoBERTa-base.
    - **Inference Phase**: Prediction of both models $\rightarrow$ Concatenation of the two predicted labels $\rightarrow$ Mapping back to 4-way labels.
    - Both sub-models use identical hyperparameters (lr=$2\times 10^{-5}$, weight decay=0.01, 10 epochs) to ensure fair comparison.

4. **Cross-Domain Evaluation Design**:
   Training on 4 independent datasets + short-text combinations + long-text combinations + all combinations, executing 49 cross-evaluations across all test sets to thoroughly test generalization capability.

### Loss & Training

- Both baseline and hierarchical models fine-tune xlm-RoBERTa-base.
- Learning rate $2\times 10^{-5}$, weight decay $0.01$, and $10$ epochs are used.
- `load_best_model_at_end=True` to select the best model.
- Inference uses softmax probabilities and takes `argmax`.

## Key Experimental Results

### Main Results - 4-Way Classification Detection (Table 3)

| Dataset | Model | HF F1 | HT F1 | MF F1 | MT F1 | Acc |
|-------|------|-------|-------|-------|-------|-----|
| Dataset1 | LSVM | 0.73 | 0.61 | 0.64 | 0.52 | 0.63 |
| Dataset1 | XLM-R | 0.83 | 0.71 | 0.77 | 0.69 | 0.75 |
| Dataset1 | **Hierarchical** | **0.85** | 0.69 | **0.80** | **0.74** | **0.77** |
| Dataset2 | XLM-R | 0.93 | 0.66 | 0.88 | 0.70 | 0.82 |
| Dataset2 | **Hierarchical** | 0.93 | **0.80** | **0.90** | **0.77** | **0.87** |
| Dataset3 | XLM-R | 0.91 | 0.91 | 0.88 | 0.89 | 0.90 |
| Dataset3 | **Hierarchical** | **0.96** | **0.95** | **0.92** | **0.91** | **0.94** |
| Dataset4 | XLM-R | 0.76 | 0.73 | 0.58 | 0.65 | 0.68 |
| Dataset4 | **Hierarchical** | **0.85** | **0.85** | **0.74** | **0.79** | **0.81** |

### Combined Dataset Experiments

| Combination | Model | HF | HT | MF | MT | Acc |
|------|------|------|------|------|------|------|
| Short(1+2) | XLM-R | 0.88 | 0.68 | 0.83 | 0.72 | 0.78 |
| Short(1+2) | Hierarchical | **0.93** | **0.85** | **0.91** | **0.86** | **0.89** |
| Long(3+4) | XLM-R | 0.89 | 0.88 | 0.74 | 0.77 | 0.82 |
| Long(3+4) | Hierarchical | **0.94** | **0.94** | **0.89** | **0.90** | **0.92** |
| All | XLM-R | 0.89 | 0.77 | 0.83 | 0.74 | 0.81 |
| All | Hierarchical | **0.91** | **0.85** | **0.88** | **0.83** | **0.87** |

### Key Findings

1. **Hierarchical method consistently outperforms baselines**: Across all four datasets and combinations, the hierarchical approach outperforms baselines, with accuracy improvements ranging from 2% to 13%.
2. **Successfully bridging the Human/Machine F1 gap**: In the baselines, Machine F1 is significantly lower than Human F1. The hierarchical method drastically narrows this gap, demonstrating that the decomposition strategy effectively resolves the issue of insufficient machine-generated text detection.
3. **Cross-domain generalization remains a challenge**: Accuracy drops significantly off-diagonal (e.g., training on Dataset 3 and testing on Dataset 4 yields only 32%), indicating poor model generalization.
4. **Text length becomes a misleading feature**: Models trained on short texts fail when applied to long texts. This is because, in short datasets, the average token count of fake news is significantly higher than that of real news, causing the model to inadvertently learn length features.
5. **Data augmentation is effective for the MGT module**: Enhancing the MGT module of Dataset 1 using the M4 dataset improves MGT accuracy by 3% and overall accuracy by 4%.

## Highlights & Insights

- **Forward-looking task definition**: In an era inundated with LLM-generated content, the 4-way classification framework (distinguishing Human/Machine $\times$ Real/Fake) is much more aligned with practical needs than traditional binary classification.
- **Simple yet effective decomposition paradigm**: Breaking down the complex 4-way classification into two simpler binary tasks substantially improves performance without requiring complex model architectures.
- **Attention to low-resource languages**: Urdu has 230 million speakers but remains severely under-researched in NLP. This work fills an important gap.
- **Real-world issues revealed by cross-domain analysis**: The finding that the model relies on text length as a feature serves as a warning regarding robustness in real-world deployment.

## Limitations & Future Work

- Only xlm-RoBERTa-base was used; stronger multilingual models (such as mBERT, XLM-R-large) were not explored.
- TF-IDF features may not be optimal for the LSVM baseline, and news features like NELA could perform better but are costly to implement.
- Poor cross-domain generalization necessitates exploring domain adaptation techniques.
- The issue of text length being used as a shortcut feature has not been addressed at the methodological level.
- The MGT module only covers outputs from GPT-4o and does not encompass content generated by other LLMs.
- The dataset scale is relatively small (at most 10,000 samples); performance in large-scale scenarios remains unknown.

## Related Work & Insights

- **Su et al. (2023)**: Proposed Structured Mimicry Prompting to simultaneously generate machine-written real/fake news; this study adopts a similar paradigm to generate Urdu data.
- **Zellers et al. (2019) GROVER**: Pioneer work on simultaneously generating and detecting fake news articles.
- **Wang et al. (2024) M4**: A multilingual machine-generated text detection dataset, of which the Urdu subset was used in this study for data augmentation.
- The hierarchical classification approach can be extended to other multi-dimensional classification tasks (e.g., joint sentiment and topic classification).

## Rating

- **Novelty**: 7/10 — The 4-way framework and hierarchical decomposition are reasonable innovations, but the technical formulation is relatively baseline.
- **Experimental Thoroughness**: 8/10 — Conducted extensively across four datasets, multiple combinations, and 49 sets of cross-domain evaluations, with in-depth analysis.
- **Writing Quality**: 7/10 — Well-structured and appropriately analyzed, though some narratives are slightly redundant.
- **Value**: 7/10 — Makes critical contributions to low-resource fake news detection, though the technical depth of the methodology is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Synergizing LLMs with Global Label Propagation for Multimodal Fake News Detection](llm_label_propagation.md)
- [\[AAAI 2026\] FactGuard: Event-Centric and Commonsense-Guided Fake News Detection](../../AAAI2026/social_computing/factguard_event-centric_and_commonsense-guided_fake_news_detection.md)
- [\[ICML 2026\] IDO: Incongruity-Aware Distribution Optimization for Multimodal Fake News Detection](../../ICML2026/social_computing/ido_incongruity-aware_distribution_optimization_for_multimodal_fake_news_detecti.md)
- [\[ACL 2026\] LiveFact: A Dynamic, Time-Aware Benchmark for LLM-Driven Fake News Detection](../../ACL2026/social_computing/livefact_a_dynamic_time-aware_benchmark_for_llm-driven_fake_news_detection.md)
- [\[ICLR 2026\] Human or Machine? A Preliminary Turing Test for Speech-to-Speech Interaction](../../ICLR2026/social_computing/human_or_machine_a_preliminary_turing_test_for_speech-to-speech_interaction.md)

</div>

<!-- RELATED:END -->
