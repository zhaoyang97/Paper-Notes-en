---
title: >-
  [Paper Note] What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma
description: >-
  [ACL 2025 (Oral, SAC Highlight)][Mental health stigma] Constructs an expert-annotated mental health stigma interview corpus (4,141 utterances, 684 participants) based on attribution theory, covering 7 fine-grained stigma types and socio-cultural background information, and benchmarks multiple SOTA neural models on the performance and challenges of the stigma detection task.
tags:
  - "ACL 2025 (Oral, SAC Highlight)"
  - "Mental health stigma"
  - "corpus construction"
  - "attribution theory"
  - "expert annotation"
  - "text classification"
date: 2026-05-08
content_hash: f09744ae18f7ea59
---

# What is Stigma Attributed to? A Theory-Grounded, Expert-Annotated Interview Corpus for Demystifying Mental-Health Stigma

**Conference**: ACL 2025 (Oral, SAC Highlight)  
**arXiv**: [2505.12727](https://arxiv.org/abs/2505.12727)  
**Code**: [https://github.com/HanMeng2004/Mental-Health-Stigma-Interview-Corpus](https://github.com/HanMeng2004/Mental-Health-Stigma-Interview-Corpus)  
**Area**: Others  
**Keywords**: Mental health stigma, corpus construction, attribution theory, expert annotation, text classification

## TL;DR

Constructs an expert-annotated mental health stigma interview corpus (4,141 utterances, 684 participants) based on attribution theory, covering 7 fine-grained stigma types and socio-cultural background information, and benchmarks multiple SOTA neural models on the performance and challenges of the stigma detection task.

## Background & Motivation

**Background**: Mental health stigma is a pervasive societal issue that impedes patients from seeking treatment and recovery. There have been some attempts in the NLP field to computationally detect stigmatizing expressions in online texts, but the available training data remains severely insufficient.

**Limitations of Prior Work**: Existing stigma detection datasets primarily originate from social media posts or synthetic data, suffering from two key issues: (1) lack of theoretical grounding—annotation schemes are often defined ad-hoc without mapping to mature psychological theoretical frameworks; (2) homogeneous and noisy data sources—social media texts have casual styles and lack demographic background information of participants, making it impossible to analyze the socio-cultural causes of stigma.

**Key Challenge**: Mental health stigma is a highly complex, fine-grained socio-psychological phenomenon—the same utterance might be perceived as stigmatizing in one culture but as caring in another. Without solid theoretical grounding and expert judgment, it is difficult to build detection systems that are helpful for practical applications.

**Goal**: (1) Construct a theory-grounded, expert-annotated stigma corpus containing participants' socio-cultural backgrounds based on mature psychological theory; (2) evaluate the capabilities and limitations of current SOTA models on this corpus.

**Key Insight**: Corrigan's (2000) Attribution Theory is adopted as the theoretical framework, which classifies mental health stigma into 7 dimensions: Responsibility, Social Distance, Anger, Helping, Pity, Coercive Segregation, and Fear. Naturally expressed stigmatizing attitudes are collected through structured human-computer interviews.

**Core Idea**: Use a chatbot to guide participants in discussing a vignette about a patient with depression, naturally eliciting their stigmatizing attitudes during structured interviews, which are then annotated by psychology experts according to the 7 dimensions of attribution theory.

## Method

### Overall Architecture

The research workflow consists of three stages: (1) data collection—designing a chatbot to guide 684 participants through structured interviews to collect their attitudes toward patients with depression; (2) expert annotation—trained annotators with psychology backgrounds annotate interview segments into one of the 7 stigma types or "non-stigma"; (3) benchmark evaluation—training and evaluating multiple text classification models on the annotated corpus.

### Key Designs

1. **Attribution Theory-Based Annotation Framework**:

    - Function: Provides a fine-grained, theory-driven stigma classification framework.
    - Mechanism: Incorporates the 7 stigma dimensions of Corrigan's attribution theory as the label space. Each interview segment is labeled as one of 8 classes: non-stigma (53.9%), Responsibility (9.5%), Social Distance (9.2%), Fear (8.9%), Anger (7.2%), Coercive Segregation (6.5%), Helping (3.8%), and Pity (1.0%). Detailed annotation guidelines and multiple rounds of training were introduced during the annotation process, with inter-annotator agreement evaluated via Krippendorff's $\alpha$.
    - Design Motivation: Unlike previous binary classifications of "stigma vs. non-stigma," theory-grounded fine-grained classification can reveal different dimensions and causes of stigma, providing a basis for subsequent targeted interventions. For instance, "fear"-type stigma needs to be addressed through public education, whereas "responsibility attribution" stigma must be corrected by emphasizing the involuntary nature of the condition.

2. **Human-Machine Structured Interview Collection Paradigm**:

    - Function: Elicits participants' natural expressions of stigmatizing attitudes under controlled conditions.
    - Mechanism: Designs a chatbot to present a vignette about "Zhang suffering from depression" to participants, then naturally elicits their attitudes through a series of guiding questions (e.g., "Do you think this is his own fault?", "Would you be willing to be his neighbor?"). The question design covers all 7 dimensions of the attribution theory. Sociodemographic information such as age, sexual orientation, gender, educational background, and cultural background of the participants is simultaneously collected.
    - Design Motivation: Compared with social media data, the interview approach ensures that every participant has an opportunity to express their attitude toward each dimension, resulting in a more balanced data distribution. Meanwhile, the structured design guarantees comparability across different participants. Utilizing a chatbot mitigates social desirability bias, which is stronger in front of human interviewers.

3. **Sociocultural Context Metadata**:

    - Function: Supports association analysis between stigmatizing attitudes and socio-cultural factors.
    - Mechanism: Records detailed demographic information for each participant, including age, gender, educational level, nationality, religious beliefs, and mental health-related experiences. This information is released alongside the stigma annotations as part of the data set, enabling researchers to analyze "which socio-cultural factors predict stronger stigmatizing attitudes."
    - Design Motivation: Stigma is not universally and uniformly distributed—individuals from different cultural backgrounds may hold strikingly different attitudes toward mental illness. Providing background metadata enables this dataset to not only train classification models but also support social science research.

### Loss & Training

In the benchmark evaluation, standard classification training is applied: the dataset is split 70/15/15 into train/validation/test sets. Models are fine-tuned on the 8-class classification task, with evaluation metrics including Macro-F1, Weighted-F1, and per-class F1.

## Key Experimental Results

### Main Results

| Model | Macro-F1 | Weighted-F1 | Non-Stigma F1 | Responsibility F1 | Fear F1 |
|------|----------|-------------|-----------|---------|---------|
| RoBERTa-base | 42.3 | 58.2 | 74.1 | 38.5 | 35.2 |
| RoBERTa-large | 45.8 | 61.7 | 76.3 | 41.2 | 38.1 |
| LLaMA-2-7B (zero-shot) | 28.5 | 39.4 | 52.8 | 22.1 | 19.6 |
| LLaMA-2-7B (few-shot) | 36.2 | 48.9 | 63.5 | 31.8 | 27.4 |
| Mistral-7B (zero-shot) | 31.2 | 42.7 | 56.1 | 24.8 | 22.3 |
| Mixtral-8x7B (few-shot) | 39.8 | 53.6 | 68.2 | 35.6 | 31.0 |
| GPT-4 (few-shot) | 48.3 | 63.5 | 78.6 | 43.7 | 40.5 |

### Ablation Study

| Experimental Configuration | Macro-F1 | Description |
|---------|----------|------|
| 8-class fine-grained classification | 45.8 | Full task |
| 2-class (Stigma vs. Non-stigma) | 72.4 | Performance vastly improves when simplified to binary classification |
| Remove "Pity" class (least data) | 47.2 | Slight improvement after removing the minority class |
| Incorporating background info as input | 47.1 | Participant background information is slightly helpful |
| Using English-participant data only | 46.5 | Minor but existing impact of cultural diversity |

### Key Findings

- Fine-grained stigma classification is highly challenging: even GPT-4 achieves a Macro-F1 of less than 50% on the 8-class task, indicating that different types of stigmatizing expressions are highly similar and subtle at the textual level.
- Class imbalance is a core challenge: the F1 scores for the "Pity" (1.0%) and "Helping" (3.8%) classes are significantly lower than those for the "Non-stigma" (53.9%) class.
- Fine-tuned medium-sized models (RoBERTa-large) achieve comparable performance to few-shot LLMs (GPT-4), indicating that this task requires domain knowledge rather than simple scaling.
- Incorporating participant background information provides some benefits, implying that stigmatizing expressions are associated with socio-cultural factors.

## Highlights & Insights

- **Theory-driven dataset design** is the most prominent highlight of this paper—instead of defining labels post-hoc, it starts with a mature psychological theoretical framework and designs data collection and annotation pipelines accordingly. This ensures the scientific validity and interpretability of the dataset.
- The **human-computer interview paradigm** cleverly combines structure and naturalness: the chatbot ensures questions cover all dimensions while exhibiting lower social desirability bias compared to face-to-face interviews, making participants more willing to express genuine attitudes.
- The value of this dataset extends far beyond NLP classification tasks—it simultaneously supports social science research (analyzing the relationship between cultural factors and stigma) and dialogue system development (training more empathetic chatbots).

## Limitations & Future Work

- The data volume is relatively limited (4,141 utterances), which may be insufficient for training very large models.
- The vignette is only designed for a single mental illness, namely "depression"; stigma patterns for other conditions (e.g., schizophrenia, bipolar disorder) might differ.
- Interviews were conducted via a chatbot, so participants' responses might be shorter and more constrained than in natural human-to-human conversations.
- Annotations only cover category labels without annotating specific spans of stigmatizing expressions, limiting the potential for interpretability analysis.

## Related Work & Insights

- **vs. SBIC (Social Bias Inference Corpus)**: SBIC focuses on broad social biases in social media, whereas this work narrows down to the specific domain of mental health stigma, offering a more solid theoretical foundation and a more controlled data collection method.
- **vs. HateXplain**: HateXplain provides hate speech detection and rationales (spans), but its source is social media, lacking participant background information and a theoretical framework.
- **vs. Synthetic Data Approaches**: Uses LLMs to generate stigmatizing text for data augmentation, but synthetic data struggles to capture the subtlety and diversity of real human expressions.

## Rating

- Novelty: ⭐⭐⭐⭐ The first mental-health stigma interview corpus based on attribution theory, filling an important gap.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models are evaluated with diverse analytical angles (class distribution, cultural factors, etc.).
- Writing Quality: ⭐⭐⭐⭐⭐ This interdisciplinary work is clearly written, with smooth transitions between the theoretical and technical sections.
- Value: ⭐⭐⭐⭐ The dataset is publicly released and designated as an ACL Oral, which significantly advances interdisciplinary research between NLP and social sciences.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ACORD: An Expert-Annotated Retrieval Dataset for Legal Contract Clause Retrieval](acord_an_expert-annotated_retrieval_dataset_for_legal_contract_drafting.md)
- [\[ACL 2025\] VAQUUM: Are Vague Quantifiers Grounded in Visual Data?](vaquum_are_vague_quantifiers_grounded_in_visual_data.md)
- [\[ACL 2025\] CoAM: Corpus of All-Type Multiword Expressions](coam_corpus_of_all-type_multiword_expressions.md)
- [\[ACL 2025\] LAQuer: Localized Attribution Queries in Content-grounded Generation](laquer_localized_attribution.md)
- [\[CVPR 2025\] Foundations of the Theory of Performance-Based Ranking](../../CVPR2025/others/foundations_of_the_theory_of_performance-based_ranking.md)

</div>

<!-- RELATED:END -->
