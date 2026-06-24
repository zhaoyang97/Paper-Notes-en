---
title: >-
  [Paper Note] Machine Translation Models are Zero-Shot Detectors of Translation Direction
description: >-
  [ACL 2025][Multilingual & Machine Translation][Translation Direction Detection] An unsupervised translation direction detection method based on NMT translation probabilities is proposed: if $p(\text{translation}|\text{original}) > p(\text{original}|\text{translation})$, the original translation direction of parallel texts can be determined in a zero-shot manner, achieving 96% document-level detection accuracy for NMT translations.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Translation Direction Detection"
  - "Translationese"
  - "Unsupervised Methods"
  - "Machine Translation Probability"
  - "Forensic Linguistics"
date: 2026-05-08
content_hash: 9aedc486e51d47ca
---

# Machine Translation Models are Zero-Shot Detectors of Translation Direction

**Conference**: ACL 2025  
**arXiv**: [2401.06769](https://arxiv.org/abs/2401.06769)  
**Code**: [GitHub](https://github.com/ZurichNLP/translation-direction-detection)  
**Area**: Multilingual Translation  
**Keywords**: Translation Direction Detection, Translationese, Unsupervised Methods, Machine Translation Probability, Forensic Linguistics  

## TL;DR

An unsupervised translation direction detection method based on NMT translation probabilities is proposed: if $p(\text{translation}|\text{original}) > p(\text{original}|\text{translation})$, the original translation direction of parallel texts can be determined in a zero-shot manner, achieving 96% document-level detection accuracy for NMT translations.

## Background & Motivation

**Background**: The original translation direction of parallel texts is often ignored in the MT community, but research shows it affects both training and evaluation. In forensic linguistics, determining the original language of a document is crucial for resolving plagiarism or forgery allegations.

**Limitations of Prior Work**:
   - **Supervised methods**: Rely on features like n-gram frequencies and POS tags to train classifiers, requiring large amounts of labeled data and suffering from severe performance drops across domains.
   - **Unsupervised clustering methods**: Require expert labeling of clustering results; in multi-domain scenarios, clustering is easily dominated by domain differences rather than translation status differences.
   - Both types of methods rely on domain-specific data and have poor applicability in the open domain.

**Key Challenge**: Translation direction detection has clear demands in practical applications (data filtering, evaluation calibration, forensic identification), but existing methods lack sufficient cross-domain generalization capabilities.

**Goal**: To design a method that detects translation direction using off-the-shelf NMT models without requiring any task-specific training data.

**Key Insight**: Leveraging the "simplification effect" of translated text (translationese/machine-translationese)—translated texts tend to have lower lexical diversity and more high-frequency expressions compared to original texts, so the generation probability of the translated text by an NMT model should be higher than that of the reverse direction.

**Core Idea**: NMT models naturally tend to assign higher conditional probabilities to translated texts; thus, the translation direction can be detected unsupervised by utilizing the difference in bidirectional translation probabilities.

## Method

### Overall Architecture

Given a parallel sentence pair $(x, y)$, a multilingual NMT model is used to calculate the bidirectional translation probabilities; the direction with the higher probability is determined as the translation direction.

### Key Designs

#### Sentence-level detection

Calculate the average token-level log probability (to avoid the effect of sequence length):

$$P_{\text{tok}}(y|x) = P(y|x)^{\frac{1}{|y|}} = \left[\prod_{j=1}^{|y|} p(y_j | y_{<j}, x)\right]^{\frac{1}{|y|}}$$

Translation direction determination:

$$\text{OTD} = \begin{cases} X \to Y, & \text{if } P_{\text{tok}}(y|x) > P_{\text{tok}}(x|y) \\ Y \to X, & \text{otherwise} \end{cases}$$

#### Document-level detection

Compute the global average over the probabilities of all sentence pairs in the document:

$$P_{\text{tok}}(y|x) = \left[\prod_{i=1}^{n}\prod_{j=1}^{|y_i|} p(y_{i,j}|y_{i,<j}, x_i)\right]^{\frac{1}{|y_1|+\cdots+|y_n|}}$$

#### Direction bias metric

$$B = |acc(X \to Y) - acc(Y \to X)|$$

$B=0$ indicates no bias, while $B=1$ indicates complete bias toward one direction.

### Experimental Setup

- **NMT Models**: M2M-100-418M, SMaLL-100, NLLB-200-1.3B
- **Data**: WMT16/22/23 News Translation Tasks (14 translation directions, 44K+ HT sentence pairs, 55K+ NMT sentence pairs), FLORES-101 indirect translation subset
- **Supervised Baseline**: Fine-tuned XLM-R (base), inspired by the COMET architecture, using the sum, absolute difference, and product of bidirectional representations as classification features

## Key Experimental Results

### Main Results — Sentence-level Classification (M2M-100)

| Translation Type | Macro-Avg Accuracy |
|---|---|
| Human Translation (HT) | **66.5%** |
| NMT Translation | **75.0%** |
| Pre-NMT Translation | 41.5% (below random) |
| LLM Translation (GPT-4) | **73.1%** |

**NMT translation detection performs the best, followed by human translation, while pre-NMT systems perform below the random level (since their outputs are often ungrammatical, making NMT models assign low probabilities).**

### Main Results — Document-level Classification (M2M-100, ≥10 sentences)

| Translation Type | Macro-Avg Accuracy |
|---|---|
| Human Translation (HT) | **80.5%** |
| NMT Translation | **95.5%** |

The highest NMT document-level accuracy across language pairs: en↔cs 98.0%, en↔ru 96.5%.

### Supervised vs. Unsupervised Comparison

| Method | HT Avg. | NMT Avg. |
|---|---|---|
| Supervised (XLM-R) | **69.5%** | 72.1% |
| Unsupervised (M2M-100) | 64.0% | **74.5%** |

The supervised method is better on in-domain HT, but the unsupervised method is superior on NMT and requires no training data.

### Ablation Study — Model Comparison

| Model | HT Macro-Avg | Remarks |
|---|---|---|
| M2M-100-418M | **66.5%** | Best on HT |
| SMaLL-100 | 66.4% | Close |
| NLLB-200-1.3B | 59.4% | Largest model but worst on HT |

Interestingly, the strongest translation model, NLLB, performs the worst in detecting the HT direction.

### Key Findings

- **Simplification effect is the core driver**: The lower lexical diversity and tendency toward high-frequency expressions in NMT output result in higher translation probabilities being assigned.
- **Pre-NMT assumption does not hold**: Old rule-based or phrase-based systems often output ungrammatical text, for which NMT models assign low probabilities, causing direction detection to flip.
- **Direction bias is non-negligible**: The direction bias for de→fr is $B=0.39$, and for zh→en is $B=0.30$; NLLB shows a bias as high as $B=0.64$ on en↔zh.
- **Indirect translation**: In cases where both sides are translated from English (FLORES), predictions for cs↔uk and xh↔zu are relatively balanced, whereas de↔fr is biased towards de→fr.
- **Sentence length influence**: Only sentences with more than 60-70 characters reach average accuracy; short sentences (e.g., "Mit freundlichen Grüßen") are difficult to detect.
- **Forensic case verification**: In a forgery allegation case involving a German doctoral thesis vs. an English book, the method supported the forgery hypothesis with a significance of $p=0.0002$.

## Highlights & Insights

- **An extremely simple method solving practical problems**: Requires only an off-the-shelf NMT model and zero training data to detect translation direction at both sentence and document levels.
- **Elegant theoretical intuition**: Translation simplification effect $\to$ probability asymmetry $\to$ direction detection; the logical chain is clear and thoroughly validated by experiments.
- **Real-world forensic application**: Applying the academic method to a public academic plagiarism/forgery case in Germany in 2022, increasing the societal impact of the paper.
- **Revealing model biases**: Different NMT models suffer from systematic direction biases on certain language pairs; the strongest model is not necessarily the most suitable for detection.

## Limitations & Future Work

- Requires sentence-aligned parallel texts, whereas in practice, one-to-many/many-to-many alignments may require preprocessing.
- Ineffective on pre-NMT system outputs (accuracy is below random), indicating that the method's assumption relies on the reasonable grammaticality of translation outputs.
- Primarily tested on high-resource languages; low-resource languages have not been validated due to a lack of bidirectional test data.
- Direction bias may lead to unreliable results on certain language pairs, requiring additional bias correction or specific model selection.
- Sentence-level accuracy (66%) might be insufficient for forensic scenarios demanding high precision, requiring aggregation at the document level.

## Related Work & Insights

- Junczys-Dowmunt (2018): Bidirectional translation probabilities used for noisy parallel corpus filtering $\to$ Extended to translation direction detection in this work.
- Thompson & Post (2020): Translation probabilities used for MT evaluation $\to$ Probability symmetry/asymmetry analysis framework.
- Sominsky & Wintner (2019): Supervised feature classification methods $\to$ The proposed unsupervised method requires no labeled data.
- Vanmassenhove et al. (2019): Machine translation reduces lexical diversity $\to$ Provided the theoretical foundation for the core assumption.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Clever application of a simple hypothesis, converting the inherent asymmetry of NMT probabilities into a detection tool.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 20 translation directions, 3 NMT models, 4 translation types, supervised baseline comparison, and a forensic case study.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Clear motivation, engaging forensic case study, and in-depth qualitative analysis.
- **Value**: ⭐⭐⭐⭐ — Simple and practical method, valuable for data filtering, translation evaluation, and forensic linguistics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Translation and Fusion Improves Zero-shot Cross-lingual Information Extraction](translation_and_fusion_improves_cross-lingual_information_extraction.md)
- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Trans-Zero: Self-Play Incentivizes Large Language Models for Multilingual Translation](trans-zero_self-play_incentivizes_large_language_models_for_multilingual_transla.md)
- [\[ACL 2025\] Exploring In-context Example Generation for Machine Translation](exploring_in-context_example_generation_for_machine_translation.md)
- [\[ACL 2025\] Has Machine Translation Evaluation Achieved Human Parity?](mt_eval_human_parity.md)

</div>

<!-- RELATED:END -->
