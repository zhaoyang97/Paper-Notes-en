---
title: >-
  [Paper Note] The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages
description: >-
  [ACL 2025][Multilingual & Machine Translation][Low-resource languages] Proposes the Esethu framework—a community-driven, sustainable data governance approach that enables circular reinvestment of data revenue through innovative community-centric licensing, validated using the isiXhosa speech dataset, ViXSD.
tags:
  - "ACL 2025"
  - "Multilingual & Machine Translation"
  - "Low-resource languages"
  - "data governance framework"
  - "community-driven"
  - "isiXhosa speech dataset"
  - "ASR"
date: 2026-05-08
content_hash: 324bfa44715a4954
---

# The Esethu Framework: Reimagining Sustainable Dataset Governance and Curation for Low-Resource Languages

**Conference**: ACL 2025  
**arXiv**: [2502.15916](https://arxiv.org/abs/2502.15916)  
**Code**: None  
**Area**: Multilingual Translation  
**Keywords**: Low-resource languages, data governance framework, community-driven, isiXhosa speech dataset, ASR

## TL;DR

Proposes the Esethu framework—a community-driven, sustainable data governance approach that enables circular reinvestment of data revenue through innovative community-centric licensing, validated using the isiXhosa speech dataset, ViXSD.

## Background & Motivation

### Limitations of Prior Work

**Background**: African low-resource languages are severely underrepresented in NLP due to several root causes:

1. **Data Scarcity**: isiXhosa has over 9 million speakers, but publicly available speech data amounts to only about 61 hours, far below the baseline for languages of comparable population size.
2. **Inadequacy of Existing Licenses**: Traditional open licenses (like CC BY) assume all users have equal resource access capability. In practice, resource-rich, non-African entities are more likely to exploit these resources, exacerbating inequalities, while closed licenses stifle research and innovation.
3. **Neglect of Community Benefits**: Existing dataset creation workflows rarely provide continuous economic benefits or governance power to data contributors (i.e., members of the speaker community). For instance, the Oshiwambo dataset project could not be released due to the lack of an appropriate Afro-centric license.
4. **Sustainability Challenges**: The creation of low-resource language datasets is typically a one-off effort, lacking a continuous investment mechanism to expand and maintain the datasets.

### Goal

**Goal**: 
### Overall Architecture

The Esethu framework comprises three core elements: a community-driven data curation process, the innovative Esethu License, and a sustainable revenue reinvestment model.


## Method

### Overall Architecture

The Esethu framework comprises three core elements: a community-driven data curation process, the innovative Esethu License, and a sustainable revenue reinvestment model. The framework ensures that the ownership and licensing of the data remain strictly under the control of the native speaker community.

### Key Designs

1. **Esethu License**: A dual-component license is designed, consisting of an open permit (allowing non-commercial research use, similar to CC BY-NC-SA) and a commercial license (free for African entities; licensing fees required for non-African commercial entities). "African entities" are defined as organizations headquartered in Africa or with majority African ownership. Licensing fees are strictly mandated to be reinvested into further data creation.

2. **Circular Reinvestment Model**: Licensing revenues are systematically reinvested into expanding the dataset. Simulations show that with monthly licensing revenue equivalent to 1% of the initial cost and a 20% quarterly growth rate, the dataset can scale from 10 hours to 893 hours within 12 months (approximately a 50x increase) while creating stable community employment (from 1 to 4 full-time transcribers).

3. **Community-Empowered Data Curation Process**: Participants retain governance rights over how their data is used. Rich demographic and linguistic metadata are collected (age, gender, education, birthplace, dialectal features, etc.), ensuring gender balance and representative regional diversity.

### Loss & Training

ASR validation experiments employ the MMS (Massively Multilingual Speech) model with adapter fine-tuning:
- Based on the `mms-1b-fl102` checkpoint
- Batch size 2, gradient accumulation steps 16, total effective batch size 32
- Learning rate 0.001
- Fine-tuned for 5, 10, and 15 epochs respectively
- Audio preprocessing: converted to mono, resampled to 16kHz, with punctuation and special characters removed

## Key Experimental Results

### Main Results

The ViXSD dataset includes 8 speakers (4 male, 4 female), 395 recordings, and approximately 10 hours of read speech.

| Model | WER | CER | Description |
|------|-----|-----|------|
| mms-1b-fl102 (zero-shot) | 0.356 | 0.066 | Baseline |
| mms-1b-all (zero-shot) | 0.372 | 0.068 | More adapters performed worse |
| mms-1b-fl102-xho-5 | 0.335 | 0.058 | 5-epoch fine-tuning |
| mms-1b-fl102-xho-10 | **0.310** | **0.052** | Optimal at 10 epochs |
| mms-1b-fl102-xho-15 | 0.321 | 0.052 | Slight overfitting at 15 epochs |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Fine-tuning vs zero-shot | WER decreased by 4.6% | Validates training effectiveness on the dataset |
| Insertion error reduction | t=10.27, p=0.002 | Statistically significant |
| Substitution error reduction | t=8.06, p=0.004 | Statistically significant |
| Deletion error reduction | t=0.91, p=0.429 | Not significant |

### Key Findings

1. The `mms-1b-fl102` model (utilizing fewer adapter weights, i.e., 102 adapters) outperforms `mms-1b-all` (1162 adapters) in the zero-shot setting, indicating that smaller models can be highly competitive when focusing fine-tuning.
2. Fine-tuning for 10 epochs yields the best performance, while 15 epochs show slight overfitting, reflecting a typical challenge in low-resource scenarios.
3. Fine-tuning significantly mitigates "hallucination" phenomena in ASR (with a significant reduction in insertion errors), though improvements in deletion errors remain limited.

## Highlights & Insights

- **Institutional Innovation Over Technical Innovation**: The core contribution of this work lies not in algorithms or models, but in proposing a replicable data governance and licensing framework, which is more fundamental for the sustainable development of low-resource languages.
- **Self-Sustaining Ecosystem via Circular Reinvestment**: Licensing revenue $\rightarrow$ dataset expansion $\rightarrow$ better models $\rightarrow$ increased commercial usage $\rightarrow$ further revenue, establishing a virtuous cycle.
- **Balancing Openness and Protection**: The Esethu License delicately balances open research access and the protection of commercial interests, avoiding resource asymmetry issues typical of purely open licenses.
- **Richness of Metadata**: The dataset contains extensive demographic information (birthplace, raised region, current residence, etc.), which is valuable for studies on dialects and accents.

## Limitations & Future Work

1. **Limited Dataset Size**: Containing only 10 hours of speech from 8 speakers, the dataset might cause models to overfit to specific speaker characteristics.
2. **Insufficient Regional Coverage**: Speakers are predominantly located in the Eastern Cape and Western Cape, which fails to cover all dialectal variations of isiXhosa.
3. **Monotonous Text Source**: Data originates solely from news articles, leading to potential style and topical biases.
4. **Legal Enforceability of the License**: The international legal enforceability of the Esethu License has not been fully verified, and the definition of an "African entity" might face disputes in borderline cases.
5. **Validation of the Reinvestment Model**: The projection of 50x growth in 12 months relies heavily on assumptions (1% initial revenue, 20% quarterly growth), necessitating long-term tracking to verify actual sustainability.

## Related Work & Insights

- The **Nwulite Obodo Data License** and the **Kaitiakitanga Māori Data Sovereignty License** serve as inspirations for the Esethu License, illustrating global demands from indigenous communities for data sovereignty.
- While African speech datasets such as **AfriSpeech-200** and **BibleTTS** have made strides, most follow a top-down approach and lack community-led governance mechanisms.
- The Data Trust model presents an alternative governance structure, yet it entails significantly higher legal complexity.
- Takeaways for the NLP community: Technical progress must run parallel with institutional innovation; otherwise, the digital divide in low-resource languages will continue to widen.

## Rating

| Dimension | Score (1-5) |
|------|-----------|
| Novelty | 4 |
| Theoretical Depth | 3 |
| Experimental Thoroughness | 3 |
| Value | 5 |
| Writing Quality | 4 |
| Total Score | 3.8 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Accessible Machine Translation Evaluation For Low-Resource Languages](accessible_machine_translation_evaluation_for_low-resource_languages.md)
- [\[ACL 2025\] Dictionaries to the Rescue: Cross-Lingual Vocabulary Transfer for Low-Resource Languages Using Bilingual Dictionaries](dictionaries_to_the_rescue_cross-lingual_vocabulary_transfer_for_low-resource_la.md)
- [\[ACL 2025\] Multilingual Encoder Knows More Than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages](multilingual_encoder_knows_more_than_you_realize_shared_weights_pretraining_for_.md)
- [\[ACL 2025\] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu](understanding_in-context_machine_translation_for_low-resource_languages_a_case_s.md)
- [\[ACL 2025\] Read it in Two Steps: Translating Extremely Low-Resource Languages with Code-Augmented Grammar Books](low_resource_translation.md)

</div>

<!-- RELATED:END -->
