---
title: >-
  [Paper Note] On the Acquisition of Shared Grammatical Representations in Bilingual Language Models
description: >-
  [ACL 2025][LLM (Other)][Cross-lingual Transfer] By training small, controlled bilingual language models, this paper investigates the mechanisms of shared cross-lingual grammatical representations using the structural priming paradigm from psycholinguistics. The study finds that the cross-lingual structural priming effect is asymmetric across language pairs and significantly weaker for typologically more distant language pairs (e.g., English-Greek).
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Cross-lingual Transfer"
  - "Bilingual Models"
  - "Structural Priming"
  - "Shared Grammatical Representations"
  - "Linguistic Typology"
date: 2026-05-08
content_hash: be190acb88d5acce
---

# On the Acquisition of Shared Grammatical Representations in Bilingual Language Models

**Conference**: ACL 2025  
**arXiv**: [2503.03962](https://arxiv.org/abs/2503.03962)  
**Code**: None  
**Area**: LLM / NLP Understanding  
**Keywords**: Cross-lingual Transfer, Bilingual Models, Structural Priming, Shared Grammatical Representations, Linguistic Typology

## TL;DR

By training small, controlled bilingual language models, this paper investigates the mechanisms of shared cross-lingual grammatical representations using the structural priming paradigm from psycholinguistics. The study finds that the cross-lingual structural priming effect is asymmetric across language pairs and significantly weaker for typologically more distant language pairs (e.g., English-Greek).

## Background & Motivation

**Background**: Cross-lingual transfer is a core capability of contemporary multilingual language models—models trained on English often perform well on languages such as French and German. However, how this transfer occurs and whether models genuinely establish shared cross-lingual grammatical representations remain unclear.

**Limitations of Prior Work**: Existing research primarily utilizes pre-trained large-scale multilingual models (e.g., mBERT, XLM-R), where key variables such as the volume of training data per language and exposure order cannot be controlled. This prevents the isolation of specific factors influencing the formation of shared representations. Furthermore, while previous cross-lingual structural priming studies have found evidence of shared representations, they failed to control for confounding variables like training data volume and language exposure order.

**Key Challenge**: Understanding the mechanisms of cross-lingual transfer requires strictly controlled experimental conditions (such as data volume, exposure order, and language pair selection), which are inherently absent in large-scale pre-trained models. Additionally, a methodology capable of detecting "shared grammatical representations" is needed, as simple evaluations on downstream tasks cannot directly isolate sharing at the syntactic level.

**Goal**: (1) Systematically investigate the formation of shared grammatical representations by training bilingual models under strictly controlled conditions; (2) Explore how language similarity affects cross-lingual grammatical sharing; (3) Provide mechanistic guidance for practical applications of cross-lingual transfer.

**Key Insight**: The authors borrow the concept of "structural priming" from cognitive psycholinguistics. In human studies, if an individual is more inclined to produce a passive sentence after processing one, it indicates that the grammatical structure of passive sentences has been "primed." If a passive sentence in Language A primes the preference for a passive sentence in Language B, it demonstrates that the two languages share a representation of the passive structure.

**Core Idea**: Train controlled small bilingual GPT-2 models by manipulating language pairs (English-Dutch, English-French, English-Greek) and training data proportions, and employ cross-lingual structural priming experiments to detect the presence and strength of shared grammatical representations.

## Method

### Overall Architecture

Experimental workflow: (1) Train small bilingual GPT-2 models (~125M parameters) from scratch, controlling the data ratio between the two languages (e.g., 50:50, 80:20) and the order of exposure (simultaneous learning vs. L1 prior to L2); (2) Construct active/passive sentence pairs as prime and target stimuli (e.g., English prime sentence -> Dutch target sentence); (3) Compare the model's probability preference for the passive/active structure in target sentences under different priming conditions; (4) If the psycholinguistic structural priming effect is replicated in the models, it indicates that the models have formed cross-lingually shared grammatical representations.

### Key Designs

1. **Controlled Bilingual Model Training Scheme**:

    - **Function**: Eliminate confounding variables in large-scale pre-trained models to isolate the effects of specific independent variables.
    - **Mechanism**: Train multiple GPT-2 models for each language pair (English-Dutch, English-French, English-Greek). Three independent variables are manipulated: (a) the data ratio between the two languages (ranging from 50:50 to 80:20); (b) the order of language exposure (simultaneous exposure vs. learning L1 first and then L2); (c) maintaining a consistent total volume of training data as a control. Multiple models are trained with different random seeds for each condition to perform statistical tests.
    - **Design Motivation**: The mixture ratios and exposure orders of multilingual data in large models are unknown and uncontrollable. Training small models from scratch is the only way to strictly control these variables and draw causal conclusions.

2. **Cross-Lingual Structural Priming Paradigm**:

    - **Function**: Detect whether the language model has formed abstract grammatical representations shared across languages.
    - **Mechanism**: Borrow a classic paradigm from psycholinguistics. Sentence pairs are constructed: the prime sentence is a passive sentence in Language A (e.g., English "The cake was eaten by the girl"), and the target sentence is in Language B. The model is observed to see if it assigns a higher probability to the passive structure in Language B. The strength of the shared representation is quantified by calculating the priming effect size—the differences in the probability of passive versus active target sentences following passive priming. Priming is tested bidirectionally (A→B and B→A).
    - **Design Motivation**: Structural priming is the "gold standard" method in cognitive science for detecting shared grammatical representations. If the model exhibits cross-lingual priming effects, it directly proves the sharing of grammatical representations across languages.

3. **Typological Gradient Design of Three Language Pairs**:

    - **Function**: Investigate how typological similarity affects the formation of shared representations.
    - **Mechanism**: Select three language pairs with increasing typological distance from English: English-Dutch (same Germanic family, highly similar), English-French (same Indo-European family but different branches), and English-Greek (same Indo-European family but distinct writing systems). Dutch shares a similar word order and the same writing system with English, whereas Greek uses a completely different alphabet system with very little lexical overlap.
    - **Design Motivation**: If the strength of shared representations degrades as typological distance increases, it demonstrates that linguistic similarity is a key driving factor for cross-lingual transfer, which has direct implications for developing models in low-resource languages.

### Loss & Training

Standard autoregressive language modeling loss (next-token prediction) is used to train GPT-2. The data is sourced from Wikipedia in each language, followed by cleaning and tokenization. Bilingual data is mixed via simple concatenation, with separate BPE tokenizers trained for each language pair.

## Key Experimental Results

### Main Results — Cross-Lingual Structural Priming Effects

| Language Pair Direction | Priming Effect Strength | Significant | Description |
|-----------|------------|---------|------|
| English → Dutch | Strong | ✅ p<0.001 | Strongest priming effect, highly similar languages |
| Dutch → English | Stronger | ✅ p<0.001 | Effect is stronger when English is the target |
| English → French | Moderate | ✅ p<0.01 | Moderate typological distance |
| French → English | Relatively Strong | ✅ p<0.001 | Similarly, stronger when English is the target |
| English → Greek | Weak | ❌ Not always significant | Maximum typological distance |
| Greek → English | Moderate | ✅ p<0.05 | Different writing systems impact sharing |

### Ablation Study — Effect of Data Proportions and Exposure Order

| Condition | English → Dutch Priming Effect | Dutch → English Priming Effect | Description |
|------|-------------|-------------|------|
| 50:50 Simultaneous Exposure | Strong | Stronger | Baseline condition |
| 80:20 (More English) | Slightly Weaker | Strong | More English data does not equate to better sharing |
| English first, then Dutch | Moderate | Strong | Partially retained sharing despite catastrophic forgetting |
| 80:20 (More Dutch) | Relatively Strong | Moderate | Limited impact of minority language data proportion |

### Key Findings

- **Asymmetry in Priming Effects**: Across all language pairs, the priming effect is stronger when English serves as the target language. This asymmetry persists even after controlling for data volume and exposure order, hinting that this might be an inherent characteristic of how grammatical representations are organized, rather than being caused by data volume biases.
- **Crucial Role of Typological Distance**: The priming effect is strongest for English-Dutch, followed by English-French, and weakest for English-Greek. This directly demonstrates that typological similarity—including syntactic structures and orthography—governs the formation of shared representations.
- **Catastrophic Forgetting Does Not Fully Erase Shared Representations**: Even when the model undergoes severe degradation in L1 performance due to sequential training (L2 training after L1), cross-lingual priming effects can still be preserved, particularly in similar language pairs. This indicates that shared representations at the syntactic level are more robust than those at the lexical level.
- **Implications for Human Psycholinguistics**: The asymmetric priming effects observed in the models can help explain similar asymmetries observed in human bilinguals. Previous explanations of this asymmetry in human studies could not rule out confounds regarding data quantity/exposure, which are successfully eliminated in controlled model experiments.

## Highlights & Insights

- **A Paradigm for Integrating Cognitive Science Methodology into NLP**: Utilizing cognitive science "probes" like structural priming to inspect internal model representations is far more precise and targeted than coarse evaluations on downstream tasks. This interdisciplinary methodological translation is highly worth promoting.
- **Causal Inference Advantages of Controlled Small Models**: Confounding variables that cannot be controlled in large models can be isolated in small models. Although small models are less performant, the causal evidence they provide holds stronger scientific value than the correlational evidence obtained from large models. This paradigm of "mechanistic studies using small models" is exemplary.
- **Practical Recommendations for Low-Resource Model Development**: The results suggest that when performing continual pre-training for low-resource languages, selecting a source base model that is typologically closer is more effective than simply choosing the largest available English model.

## Limitations & Future Work

- **Only One Grammatical Structure Examined**: The study only tests active/passive alternation to detect shared representations, leaving out other grammatical structures (e.g., dative alternation, relative clause nesting).
- **Small Model Scale**: The 125M parameter GPT-2 is far smaller than multilingual models used in practice, and cross-lingual representations in larger models might exhibit different dynamics.
- **Limited Language Pairs**: Only three Indo-European languages were tested. It remains unknown whether these findings generalize to language pairs with larger language family divides (e.g., English-Chinese, English-Japanese).
- Future work can extend this methodology to more grammatical structures and language pairs, especially non-Indo-European languages, to chart a comprehensive map of shared cross-lingual representations.

## Related Work & Insights

- **vs. Chang et al. (2024, Multilinguality)**: While Chang et al. investigate macro-level effects of how multilingual data improves target language performance, this work provides a mechanism explanation at the micro-syntactic level—the foundation of cross-lingual transfer is shared grammatical representations.
- **vs. Probing-Based Analysis**: Traditional approaches use probing classifiers to detect linguistic features in encoders, but probing cannot differentiate "information encoded in representations" from "representations actually used by the model during processing." Structural priming more directly detects whether the model genuinely utilizes the shared representations.
- **Relevance to Continual Pre-training**: The catastrophic forgetting experiments in this paper directly address practical issues surrounding the pipeline of "English base model pre-training followed by target language fine-tuning"—grammatical sharing may still be preserved even after L1 forgetting occurs.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing the structural priming paradigm to LM analysis is a novel interdisciplinary contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Sufficient statistical testing across three language pairs and multiple conditions, but limited to a single grammatical structure.
- Writing Quality: ⭐⭐⭐⭐ This interdisciplinary work is clearly written with thorough background details.
- Value: ⭐⭐⭐⭐ Provides significant contributions to our mechanical understanding of multilingual models and offers useful guidance for low-resource language practices.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Acquisition and Application of Novel Knowledge in Large Language Models](acquisition_and_application_of_novel_knowledge_in_large_language_models.md)
- [\[ACL 2025\] Bilingual Zero-Shot Stance Detection](bilingual_zero-shot_stance_detection.md)
- [\[ACL 2025\] Language-Codec: Bridging Discrete Codec Representations and Speech Language Models](language_codec_bridging_discrete_codec_speech_language_models.md)
- [\[ACL 2025\] Representations of Fact, Fiction and Forecast in Large Language Models: Epistemics and Attitudes](representations_of_fact_fiction_and_forecast_in_large_language_models_epistemics.md)
- [\[ACL 2025\] Cross-model Transferability among Large Language Models on the Platonic Representations of Concepts](cross_model_transferability_sv.md)

</div>

<!-- RELATED:END -->
