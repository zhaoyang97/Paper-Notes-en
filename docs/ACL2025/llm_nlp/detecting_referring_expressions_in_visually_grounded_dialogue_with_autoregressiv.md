---
title: >-
  [Paper Note] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models
description: >-
  [ACL 2025][LLM (Other)][mention detection] This paper models the detection of referring expressions in visual dialogue as an autoregressive token prediction task. Through parameter-efficient fine-tuning (QLoRA) of Llama 3.1-8B, the authors demonstrate that textual context alone is highly effective for detecting mention spans in visually grounded dialogues, achieving F1 scores of 0.90 and 0.94 on the AGOS and PhotoBook datasets, respectively.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "mention detection"
  - "referring expressions"
  - "visually grounded dialogue"
  - "autoregressive language models"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: 62914c58a7d33af7
---

# Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models

**Conference**: ACL 2025  
**arXiv**: [2506.21294](https://arxiv.org/abs/2506.21294)  
**Code**: [GitHub](https://github.com/willemsenbram/mention-detection-vgd)  
**Area**: LLM/NLP  
**Keywords**: mention detection, referring expressions, visually grounded dialogue, autoregressive language models, parameter-efficient fine-tuning  

## TL;DR

This paper models the detection of referring expressions in visual dialogue as an autoregressive token prediction task. Through parameter-efficient fine-tuning (QLoRA) of Llama 3.1-8B, the authors demonstrate that textual context alone is highly effective for detecting mention spans in visually grounded dialogues, achieving F1 scores of 0.90 and 0.94 on the AGOS and PhotoBook datasets, respectively.

## Background & Motivation

- **Core Problem**: In visually grounded dialogues, speakers often use words or phrases to refer to objects in the visual scene (i.e., referring expressions). Effectively detecting these mentions is a prerequisite for downstream coreference resolution and visual grounding.
- **Limitations of Prior Work**: Early rule-based + dependency-parsing methods required extensive feature engineering; although BERT-like encoder models are effective, their framework is based on sequence labeling, which lacks flexibility. Currently, generative information extraction based on autoregressive LLMs has not yet been applied to mention detection in visually grounded dialogues.
- **Key Insight**: To what extent can textual context alone support tasks that are inherently multimodal? How does dialogue history affect the performance of mention detection?

## Method

### Overall Architecture

The mention detection is modeled as a **generative paraphrasing task**: given the current utterance and dialogue history, the model autoregressively generates a copy of the current utterance, inserting boundary markers `>>` and `<<` at the start and end of mention spans. For example, input "I have a dog" $\rightarrow$ output "I have >>a dog<<".

### Key Designs

1. **Context-conditioned Generation**: The generation target is $u_i' = f(u_i, H)$, where $H = (u_{i-h}, ..., u_{i-1})$ represents historical messages of configurable length. The impact of different context window sizes (0, 3, 7, and 19 historical messages) is compared experimentally.

2. **Parameter-efficient Fine-tuning**: Llama 3.1-8B is fine-tuned using QLoRA (4-bit quantization + LoRA) and trained on two small-scale datasets: AGOS (15 dialogues, 1,486 mentions) and PhotoBook (50 dialogues, 2,111 mentions).

3. **Span Boundary Marker Design**: Special markers for the start and end of mentions are added to the tokenizer's vocabulary, allowing the model to naturally segment mention spans during generation without requiring extra CRF or sequence-labeling layers.

### Evaluation Protocol

Cross-validation is used to evaluate in-dataset performance, along with cross-dataset transfer tests (training on AGOS and testing on PB, and vice versa), and comparison against NP extraction baselines and BERT sequence-labeling baselines.

## Key Experimental Results

### Main Results (In-dataset Cross-validation)

| Model | Context Window | AGOS F1 | PB-GOLD F1 |
|------|:---:|:---:|:---:|
| Llama 3.1-8B | 0 | .863 | .930 |
| Llama 3.1-8B | 3 | .892 | .930 |
| Llama 3.1-8B | 7 | .900 | .937 |
| Llama 3.1-8B | 19 | **.902** | **.940** |
| NP Baseline | - | Lower | Lower |

### Cross-dataset Transfer

| Train set → Test set | F1 |
|------|:---:|
| AGOS → PB | Performance drops but remains reasonable |
| PB → AGOS | Limited transfer performance |

Cross-dataset transfer presents challenges due to the different features of mention distributions between the two datasets (17.94% of messages in AGOS contain >1 mention, compared to only 1.95% in PB).

### Ablation Study: Impact of Dialogue History

| History Window | AGOS F1 Change | PB F1 Change |
|------|:---:|:---:|
| 0 → 3 | +0.029 | +0.000 |
| 3 → 7 | +0.008 | +0.007 |
| 7 → 19 | +0.002 | +0.003 |

### Key Findings

- Textual context alone is sufficient to achieve high mention detection performance (F1 > 0.90), highlighting the rich information contained in language context.
- Dialogue history consistently improves performance on AGOS, but provides less help for PB—where mentions are more independent descriptive expressions.
- The combination of a small-scale dataset + parameter-efficient fine-tuning + a medium-sized LLM is remarkably effective.
- The gap in cross-dataset transfer indicates that the referring language in task-oriented dialogues is domain-specific.
- As a text-only method, there are fundamental limitations on boundary cases that strictly require visual information to determine referentiality.

## Highlights & Insights

- Elegantly formulates mention detection as an "annotated paraphrasing" generation task, eliminating the need for sequence-labeling architectures.
- Represents the first application of autoregressive LLMs + generative information extraction to mention detection in visually grounded dialogues.
- Clearly analyzes the boundaries of both text-only and multimodal approaches with an honest discussion.

## Limitations & Future Work

- Text-only approaches cannot resolve ambiguous mentions requiring visual context (e.g., whether "that" refers to an object in the image).
- The dataset sizes are small (15 and 50 dialogues), so the generalizability of results needs further validation.
- Only Llama 3.1-8B was evaluated, without comparison to other scales or architectures of LLMs.
- The model only detects mention spans and does not perform coreference resolution or visual grounding.
- The task is limited to coarse-grained mention detection and does not differentiate between mention types.

## Related Work & Insights

- **Mention Detection**: Lee et al. (2013); Devlin et al. (2019) — Evolution from rule-based systems to BERT.
- **Generative Information Extraction**: Cao et al. (2021); Zhang et al. (2025) — Modeling structured prediction as autoregressive generation.
- **Visually Grounded Dialogue**: PhotoBook (Haber et al., 2019); AGOS (Willemsen et al., 2022) — Dialogue tasks that guide participants to refer to images.
- **Parameter-efficient Fine-tuning**: LoRA (Hu et al., 2022); QLoRA (Dettmers et al., 2023) — Fine-tuning large models in low-resource settings.

## Rating

| Dimension | Score |
|------|:---:|
| Novelty | ★★★☆☆ |
| Practicality | ★★★☆☆ |
| Experimental Thoroughness | ★★★★☆ |
| Writing Quality | ★★★★☆ |
| Overall | ★★★☆☆ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Comparing Linguistic Acceptability Judgments of Autoregressive Language Models](comparing_linguistic_acceptability_judgments_of_autoregressive_language_models.md)
- [\[ACL 2025\] Deontological Keyword Bias: The Impact of Modal Expressions on Normative Judgments of Language Models](deontological_keyword_bias.md)
- [\[ACL 2025\] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model](beyond_dialogue_a_profile-dialogue_alignment_framework_towards_general_role-play.md)
- [\[ACL 2025\] Towards Geo-Culturally Grounded LLM Generations](geocultural_grounded_llm.md)
- [\[ICML 2025\] B-score: Detecting biases in large language models using response history](../../ICML2025/llm_nlp/b-score_detecting_biases_in_large_language_models_using_response_history.md)

</div>

<!-- RELATED:END -->
