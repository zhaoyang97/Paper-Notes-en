---
title: >-
  [Paper Note] Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models
description: >-
  [ACL 2025][Multilingual & Machine Translation][Statement-Tuning] This work extends the Statement-Tuning method to multilingual scenarios, demonstrating that mDeBERTa, an encoder-only model with only 276M parameters, can achieve cross-lingual zero-shot generalization across unseen tasks and unseen languages after multilingual Statement-Tuning, matching or even surpassing generative LLMs with 70B+ parameters on multiple NLU tasks.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Statement-Tuning"
  - "Encoder Models"
  - "Cross-lingual Generalization"
  - "Zero-shot Learning"
  - "Parameter-efficient"
date: 2026-05-08
content_hash: 709fcea4d6f30a8d
---

# Statement-Tuning Enables Efficient Cross-lingual Generalization in Encoder-only Models

**Conference**: ACL 2025  
**arXiv**: [2506.01592](https://arxiv.org/abs/2506.01592)  
**Code**: [Yes](https://github.com/mbzuai-nlp/Multilingual-ST)  
**Area**: NLP / Multilingual / Zero-shot Generalization  
**Keywords**: Statement-Tuning, Encoder Models, Cross-lingual Generalization, Zero-shot Learning, Parameter-efficient

## TL;DR

This work extends the Statement-Tuning method to multilingual scenarios, demonstrating that mDeBERTa, an encoder-only model with only 276M parameters, can achieve cross-lingual zero-shot generalization across unseen tasks and unseen languages after multilingual Statement-Tuning, matching or even surpassing generative LLMs with 70B+ parameters on multiple NLU tasks.

## Background & Motivation

LLMs perform exceptionally well in zero-shot/few-shot scenarios, but encoder models (such as BERT, RoBERTa) struggle to directly perform zero-shot task generalization due to architectural limitations—specifically, their pre-training via masked language modeling and the requirement of task-specific classification heads.

However, encoder models possess three core advantages:

**More lightweight**: Parameter sizes are significantly smaller than those of LLMs, leading to low computational and memory requirements.

**Better semantic embeddings**: Encoder models outperform decoder models on semantic understanding tasks.

**More efficient inference**: Non-autoregressive architectures yield faster inference in tasks like sequence labeling.

Existing Statement-Tuning methods have only been validated in English, leaving key open questions:
- Can encoder models achieve zero-shot cross-lingual task generalization in a multilingual environment?
- Can they serve as an efficient alternative to LLMs in low-resource language scenarios?

These questions are crucial for billions of low-resource language speakers worldwide, who often lack the computational resources to run large LLMs.

## Method

### Overall Architecture

The three-step pipeline of multilingual Statement-Tuning:
1. Multilingual Task Verbalization: Converts tasks into declarative statements.
2. Statement Fine-Tuning: Trains the encoder to predict whether a statement is True or False.
3. Zero-shot Inference: Generates statements for each possible label of a new task and selects the one with the highest probability of being True.

### Key Designs

1. **Task Verbalization**

    - Converts any discriminative task into a finite number of natural language declarative statements.
    - Each label corresponds to a statement template.
    - Example: `"{{target_word}}" means the same in "{{context_1}}" and "{{context_2}}"`
    - The statement corresponding to the correct label is labeled as True, while the others are labeled as False.
    - **Design Motivation**: Replaces task-specific classification heads with a unified True/False classification head to achieve cross-task generalization.

2. **Multilingual Training Data Construction**

    - Covers 9 NLU tasks across 25 languages (including high-resource and low-resource languages).
    - Randomly selects 1,500 training instances per language per task (750 positive and 750 negative samples).
    - Specifically introduces machine translation (MT) tasks to enhance cross-lingual capability.
    - Incorporates multiple template variations for each task to improve robustness.

3. **Model Selection**

   | Model | Parameters | Pre-training Corpus |
   |------|--------|-----------|
   | mBERT base | 110M | Wikipedia |
   | mDeBERTa-v3 | 276M | CC-100 |
   | XLM-R base | 250M | CC-100 |
   | XLM-R large | 560M | CC-100 |

4. **Ablation Design**

    - Language scale ablation: English-only vs. 11 languages vs. 25 languages.
    - Template language ablation: English templates vs. Machine-translated templates.
    - MT data ablation: With vs. without machine translation data.
    - Model scale ablation: 110M $\rightarrow$ 560M.

### Loss & Training

Standard binary cross-entropy loss (True/False) is used, with QLoRA for parameter-efficient fine-tuning. During inference, statements are generated for each possible label, and the label with the highest probability of being True is selected.

## Key Experimental Results

### Main Results (Zero-shot generalization on unseen tasks, averaged cross-lingual accuracy)

| Model | Parameters | XCOPA | XNLI | XStoryCloze | XWinoGrad |
|------|------|-------|------|-------------|-----------|
| Qwen2 | 72B | 67.84 | 42.10 | 66.70 | 84.02 |
| Llama3.1 | 70B | 62.24 | 41.68 | 68.32 | 82.69 |
| Gemma 2 | 9B | 66.29 | 46.50 | 67.41 | 83.93 |
| Aya 23 | 35B | 57.24 | 44.09 | 63.65 | 72.69 |
| **mDeBERTa** | **276M** | **65.52** | **47.84** | **73.53** | 54.75 |
| **XLM-R large** | **560M** | **64.36** | **45.76** | **78.78** | 54.26 |

### Key Comparison (mDeBERTa 276M vs. LLMs)

| Task | mDeBERTa (276M) | Best LLM | Gap |
|------|-----------------|---------|------|
| XNLI | **47.84** | 46.50 (Gemma 9B) | +1.34 |
| XStoryCloze | **73.53** | 68.32 (Llama 70B) | +5.21 |
| XCOPA | 65.52 | **67.84** (Qwen 72B) | -2.32 |

XLM-R large (560M) outperforms Llama 3.1 70B (68.32) on XStoryCloze with an accuracy of 78.78, demonstrating an approximate **130x parameter scale gap**.

### Language Scale Ablation

| Training Setup | XCOPA | XNLI | XStoryCloze |
|----------|-------|------|-------------|
| English-only (+MT) | 98.6% of 25-lang | 95.1% | 96.0% |
| 11 Languages | ~100% | ~100% | ~100% |
| 25 Languages | 100% | 100% | 100% |

### Inference Efficiency

| Model | Max Batch Size | Inference Speed Advantage |
|------|----------------|-------------|
| mDeBERTa (276M) | **Largest** | **Fastest** |
| Qwen2 (500M) | Smaller | Slower |
| Gemma 2 (9B) | Most limited | Slowest |

### Key Findings

1. **Encoder models are effective cross-task generalizers**: mDeBERTa outperforms all evaluated LLMs, including those with 72B parameters, on XNLI and XStoryCloze.
2. **English-only training yields most of the cross-lingual performance** (reaching 95-98.6% of 25-language training), indicating that multilingual pre-training itself is sufficient to support cross-lingual generalization.
3. **No significant difference between English and translated templates**: This simplifies prompt design.
4. **Crucial role of MT data**: Incorporating machine translation data significantly boosts cross-lingual transfer performance.
5. **Failure of mBERT and XLM-R base to generalize**: This demonstrates that cross-lingual generalization is an emergent capability driven by both model size and pre-training quality.
6. **Failure on the XWinoGrad task**: Coreference resolution lacks sufficient correlation with the training tasks, showing that Statement-Tuning heavily depends on the selection of training tasks.

## Highlights & Insights

1. **Counter-intuitive conclusion**: A 276M parameter encoder model outperforms 70B+ decoder models on multiple NLU tasks.
2. **Immense practical value**: Provides a viable NLU solution for low-resource languages and computationally constrained environments.
3. **Analysis of cross-lingual generalization mechanism**: Not just a matter of model size; pre-training quality (e.g., mDeBERTa vs. XLM-R base) is equally critical.
4. **Elegant design of Statement-Tuning**: Formulates arbitrary classification tasks into a single format via a unified True/False framework.
5. **Open discovery**: English training data combined with multilingual pre-training is sufficient for cross-lingual generalization.

## Limitations & Future Work

1. Statement-Tuning is highly dependent on the selection of training tasks, showing poor performance when the target task has low correlation (e.g., XWinoGrad).
2. Requires manual design of statement templates; although English templates are proven sufficient, template engineering still incurs overhead.
3. Inapplicable to tasks with extremely large label spaces—a statement must be generated for each potential label.
4. Fails to precisely pinpoint the emergence mechanism of cross-lingual generalization capabilities in encoder models.
5. Pre-training and instruction-tuning data for some generative models are not fully transparent, posing potential risks of data leakage.

## Related Work & Insights

- Elshabrawy et al. (2025): The original Statement-Tuning method, validated only in English.
- FLAN (Wei et al., 2022): Pioneering work on instruction tuning, leveraging a 137B parameter decoder-only model.
- T0 (Sanh et al., 2022): Instruction tuning for T5 encoder-decoder models.
- Xu et al. (2023): Demonstrates that DeBERTa outperforms generative LLMs under a zero-shot NLI framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to extend Statement-Tuning to multilingual scenarios with systematic validation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 4 encoder models, 10+ decoder baselines, 4 evaluation tasks, with multiple ablation studies.
- **Writing Quality**: ⭐⭐⭐⭐ In-depth analysis and ablation designs with clear conclusions.
- **Value**: ⭐⭐⭐⭐⭐ Provides an extremely practical solution for resource-constrained multilingual NLU.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cross-Lingual Generalization and Compression: From Language-Specific to Shared Neurons](cross_lingual_neurons_compression.md)
- [\[ACL 2025\] A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs](a_case_study_of_cross-lingual_zero-shot_generalization_for_classical_languages_i.md)
- [\[ACL 2025\] Language Fusion for Parameter-Efficient Cross-lingual Transfer (FLARE)](flare_crosslingual_lora.md)
- [\[ACL 2025\] Cross-Lingual Representation Alignment Through Contrastive Image-Caption Tuning](cross-lingual_representation_alignment_through_contrastive_image-caption_tuning.md)
- [\[ACL 2025\] CC-Tuning: A Cross-Lingual Connection Mechanism for Improving Joint Multilingual Supervised Fine-Tuning](cc-tuning_a_cross-lingual_connection_mechanism_for_improving_joint_multilingual_.md)

</div>

<!-- RELATED:END -->
