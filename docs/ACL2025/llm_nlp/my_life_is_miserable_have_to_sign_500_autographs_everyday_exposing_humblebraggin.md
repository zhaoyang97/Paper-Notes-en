---
title: >-
  [Paper Note] "My life is miserable, have to sign 500 autographs everyday": Exposing Humblebragging, the Brags in Disguise
description: >-
  [ACL 2025][LLM (Other)][Humblebragging] This work introduces humblebragging detection to the field of computational linguistics for the first time, proposing a 4-tuple formal definition, constructing the HB-24 synthetic dataset, and conducting a comprehensive benchmark evaluation across ML/DL/LLM. GPT-4o achieves a 0.88 F1 under the zero-shot + definition setting, outperforming human annotators.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Humblebragging"
  - "Text Classification"
  - "Sarcasm Detection"
  - "Synthetic Data"
  - "LLM"
date: 2026-05-08
content_hash: fb582b5e5594e77b
---

# "My life is miserable, have to sign 500 autographs everyday": Exposing Humblebragging, the Brags in Disguise

**Conference**: ACL 2025  
**arXiv**: [2412.20057](https://arxiv.org/abs/2412.20057)  
**Code**: [Yes (GitHub)](https://github.com/SharathHN/HB-24)  
**Area**: Other  
**Keywords**: Humblebragging, Text Classification, Sarcasm Detection, Synthetic Data, LLM

## TL;DR

This work introduces humblebragging detection to the field of computational linguistics for the first time, proposing a 4-tuple formal definition, constructing the HB-24 synthetic dataset, and conducting a comprehensive benchmark evaluation across ML/DL/LLM. GPT-4o achieves a 0.88 F1 under the zero-shot + definition setting, outperforming human annotators.

## Background & Motivation

Humblebragging is a linguistic phenomenon of self-promotion masked by complaints or humility. For example: "Oh my god, I actually got promoted to lead the entire team, the pressure is so immense!" — superficially complaining about pressure, but actually boasting about the promotion.

While this phenomenon has been studied in psychology, tourism studies, and advertising, it has never been explored in computational linguistics. Automatically detecting humblebragging is crucial for several NLP downstream tasks:

**Sentiment Analysis**: A humblebrag seems negative on the surface, but the actual intent is positive; misjudgment severely affects sentiment polarity classification.

**Intent Recognition**: Distinguishing genuine complaints from disguised bragging is essential.

**Dialogue Understanding**: In social media monitoring and customer feedback analysis, distinguishing actual complaints from bragging in disguise is necessary.

Similar to sarcasm and irony, humblebragging relies on the incongruity between literal meaning and actual intent, but its uniqueness lies in hiding self-promotion within statements of humility or complaints. Prior to this, no computational linguistics dataset or method existed to address this issue.

## Method

### Overall Architecture

The paper unfolds across three levels: (1) proposing a formal definition, (2) constructing a dataset, and (3) designing and evaluating detection methods.

### Key Designs

#### 1. **4-Tuple Formal Definition**

Humblebragging is defined as $HB = \langle B, BT, HM, MT \rangle$:

- $B$ (Brag): The explicit self-promotion/bragging part in the text.
- $BT$ (Brag Theme): The thematic category of the brag (e.g., wealth, fame, job performance, etc.).
- $HM$ (Humble Mask): The humility/complaint part used to disguise the brag.
- $MT$ (Mask Type): The type of mask, either humility or complaint.

This definition is adapted from the 6-tuple framework of sarcasm but removes Speaker and Hearer (since humblebragging usually does not target a specific listener). The design motivation is to structurally decompose humblebragging, enabling machines to understand the dual-layer semantics of "literal meaning vs. actual intent."

#### 2. **HB-24 Dataset Construction**

Due to the lack of existing datasets, the authors adopt a "synthetic training, real testing" strategy:

- **Training Set**: Generated using GPT-4o via zero-shot and few-shot prompting, yielding 11,000 candidate samples, which were filtered to 3,340 synthetic humblebrags after human review.
- **Test Set**: 558 real-world humblebrags from Wittels (2012).
- **Negative Samples**: Includes sarcasm (SARC dataset), irony (SemEval-2018), direct brags, complaints, and neutral sentences. All negative samples are human-written.

The meticulous design of negative samples is key: humblebragging is easily confused with sarcasm and irony, so incorporating these "confusing items" in the training set enhances the model’s discriminative capacity.

#### 3. **Classification Method Design**

Two task settings are proposed:

- **Binary Classification**: Standard encoder classification methods (BERT, RoBERTa).
- **Sentence Completion/Question Answering**: Formulates the detection task as a Yes/No question answering task, running decoder models in zero-shot (Z) and zero-shot + definition (Z+D) settings with the input format `<definition><question><x><answer>`.

### Loss & Training

- Encoder models use Adam optimizer + 5-fold cross-validation for hyperparameter tuning.
- Decoder models perform zero-shot inference under Z and Z+D settings; some models are fine-tuned using LoRA.
- Notably, in the Z+D setting, the 4-tuple definition is included as part of the system prompt to guide the model in understanding the structure of humblebragging.

## Key Experimental Results

### Main Results

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Human Average | 0.80 | 0.86 | 0.71 | 0.77 |
| SVM | 0.62 | 0.72 | 0.61 | 0.56 |
| BERT-Large (F) | 0.68 | 0.76 | 0.50 | 0.61 |
| RoBERTa-Large (F) | 0.78 | 0.91 | 0.62 | 0.74 |
| GPT-4o (Z) | 0.84 | 0.78 | 0.94 | 0.85 |
| **GPT-4o (Z+D)** | **0.89** | **0.91** | **0.85** | **0.88** |
| Llama-3.1-8B (Z) | 0.49 | 0.49 | 0.99 | 0.66 |
| Llama-3.1-8B (Z+D) | 0.68 | 0.62 | 0.88 | 0.72 |
| Llama-3.1-8B (F) | 0.81 | 0.87 | 0.72 | 0.79 |

### Ablation Study: 4-Tuple Definition vs. Others

| Model Setting | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Llama (Z+D) 4-Tuple Definition | 0.68 | 0.62 | 0.88 | 0.72 |
| Llama (Z+SOM) Sentiment Opposition Model | 0.66 | 0.64 | 0.74 | 0.68 |

### Downstream Application: Sentiment Polarity Classification

| Model | Accuracy | F1 |
|---|---|---|
| R-SST2 (Original) | 0.53 | 0.51 |
| R-HBSC (+humblebragging detection) | 0.82 | 0.83 |

### Key Findings

1. **GPT-4o (Z+D) outperforms humans**: F1 reaches 0.88, while the best human annotator only achieves 0.85, indicating that large language models can outperform humans when equipped with sufficient linguistic and world knowledge.
2. **Universal efficacy of the 4-tuple definition**: All decoder-based models perform better under the Z+D setting compared to the Z setting, proving that formal definitions can effectively aid detection.
3. **Finetuning on synthetic data is effective**: After finetuning on HB-24, most models show significant F1 improvements; the finetuned RoBERTa even outperforms most 7-8B decoder models.
4. **The task is challenging even for humans**: One of the three human annotators achieved an F1 of only 0.63, reflecting the inherent difficulty of humblebragging detection.
5. **Significant gains for downstream tasks**: Incorporating humblebragging and sarcasm detection boosts sentiment classification F1 from 0.51 to 0.83.

## Highlights & Insights

- **Interdisciplinary Perspective**: Maslow's Hierarchy of Needs from psychology is introduced into the motivational analysis — humblebragging simultaneously satisfies the need for belongingness (level 3, via humility) and esteem (level 4, via boasting).
- **Drawing Inspiration from Sarcasm Research**: The 4-tuple definition is adapted from the 6-tuple framework of sarcasm; this formalization strategy of "standing on the shoulders of giants" is highly commendable.
- **Reasonable Evaluation Design**: Using synthetic data for training and real-world data for testing nicely balances data acquisition costs with the realism of evaluation.

## Limitations & Future Work

1. Distribution shift exists between synthetic and real data (e.g., lack of colloquialisms, elongated words, etc.), which may limit generalization capability.
2. The dataset size is relatively small (3,340 training + ~1,100 testing).
3. Currently limited to English text, without considering cross-lingual/cross-cultural differences in humblebragging.
4. Only binary detection is investigated, without deep-diving into component extraction (e.g., automatically extracting B, HM, etc.).
5. Future work can extend to multimodal scenarios (such as image+text humblebragging detection).

## Related Work & Insights

- Extensive research in sarcasm and irony detection can be drawn upon, especially incongruity-based detection methods.
- Synthetic data generation methods (GPT-4o prompt engineering) offer valuable references for low-resource tasks.
- The paradigm of reformatting classification tasks into generation/QA tasks is widely applicable to various text classification scenarios in the LLM era.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to introduce humblebragging detection in computational linguistics, with a novel and practical 4-tuple formal definition.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers ML/DL/LLM/human comparisons, with comprehensive ablation studies and downstream application validations.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured, well-motivated, and with intuitive examples.
- **Value**: ⭐⭐⭐⭐ — Initiates a new research direction, releases dataset and code, and achieves practical improvements in downstream tasks like sentiment analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Does Time Have Its Place? Temporal Heads Where Language Models Recall Time-specific Information](does_time_have_its_place_temporal_heads_where_language_models_recall_time-specif.md)
- [\[ACL 2025\] DiffLM: Controllable Synthetic Data Generation via Diffusion Language Models](difflm_controllable_synthetic_data_generation_via_diffusion_language_models.md)
- [\[ACL 2025\] Enhancing Input-Label Mapping in In-Context Learning with Contrastive Decoding](enhancing_input-label_mapping_in_in-context_learning_with_contrastive_decoding.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Zero-Shot Belief: A Hard Problem for LLMs](zero-shot_belief_a_hard_problem_for_llms.md)

</div>

<!-- RELATED:END -->
