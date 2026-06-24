---
title: >-
  [Paper Note] Cultural Learning-Based Culture Adaptation of Language Models
description: >-
  [ACL 2025][LLM (Other)][cultural adaptation] This paper proposes the CLCA framework, which draws on cultural learning theory to generate culturally adapted dialogue data through simulated social interactions. By combining this with intention understanding for multi-task training, the framework significantly improves the cultural value alignment of various LLMs on the World Values Survey.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "cultural adaptation"
  - "cultural learning"
  - "role-playing"
  - "social interaction"
  - "value alignment"
date: 2026-05-08
content_hash: 602cfecce7087f95
---

# Cultural Learning-Based Culture Adaptation of Language Models

**Conference**: ACL 2025  
**arXiv**: [2504.02953](https://arxiv.org/abs/2504.02953)  
**Code**: [https://github.com/UKPLab/CLCA](https://github.com/UKPLab/CLCA)  
**Area**: LLM/NLP  
**Keywords**: cultural adaptation, cultural learning, role-playing, social interaction, value alignment

## TL;DR
This paper proposes the CLCA framework, which draws on cultural learning theory to generate culturally adapted dialogue data through simulated social interactions. By combining this with intention understanding for multi-task training, the framework significantly improves the cultural value alignment of various LLMs on the World Values Survey.

## Background & Motivation

**Background**: LLMs default to aligning with WEIRD values, resulting in poor cross-cultural adaptability. Existing methods rely heavily on prompt engineering.

**Limitations of Prior Work**: Prompts rely on cultural knowledge that the model has already learned; direct fine-tuning on generic corpora fails to achieve controlled cultural adaptation (Choenni et al. 2024).

**Key Challenge**: How can LLMs internalize the implicit values of a specific culture rather than merely performing surface-level imitation?

**Goal**: Generate implicit cultural signals through simulated social interactions for training.

**Key Insight**: Cultural learning theory (Tomasello) — humans acquire culture through social interaction and intention understanding.

**Core Idea**: Utilize LLM role-playing to generate culturally adapted social interaction dialogues, combined with intention understanding training, to achieve behavior-driven value adaptation.

## Method

### Overall Architecture
Culturally adapted social scenario generation -> LLM role-playing dialogue -> Quality filtering -> Intention labeling -> Multi-task training (dialogue + intention understanding) -> WVS evaluation.

### Key Designs

1. **Culturally Adapted Scenario Generation**

    - Based on the Sotopia framework, utilizing GPT-4 for cultural localization (names, locations, customs).
    - Incorporating Hofstede cultural dimensions and the Inglehart-Welzel evolutionary cultural map.
    - **Design Motivation**: Transmit implicit cultural norms through specific scenarios.

2. **Dialogue Generation and Filtering**

    - Two LLMs perform role-playing, each having private social goals.
    - Two-stage LLM-as-Judge filtering: generation quality and cultural consistency.
    - **Design Motivation**: Ensure that the data reflects cultural values.

3. **Intention Understanding Training**

    - Generate free-text intentions for each dialogue turn (generic vs. culturally relevant).
    - Culturally relevant intentions serve as "instructed learning" signals.
    - **Design Motivation**: Distinguish between essential behaviors and accidental behaviors.

4. **Multi-Task Training (CLCA)**

    - Task 1: Multi-turn dialogue (imitative learning)
    - Task 2: Intention understanding (instructed learning)
    - **Design Motivation**: Correspond to the two basic forms of cultural learning.

## Key Experimental Results

### Main Results -- WVS Cultural Value Alignment

| Model | Method | US | China | Japan | Saudi Arabia | Average Gain |
|------|------|------|------|------|------|---------|
| Llama-3 | Baseline | Baseline | Baseline | Baseline | Baseline | - |
| Llama-3 | CLCA | +5% | +8% | +6% | +10% | **+7.3%** |
| Mistral | CLCA | +4% | +7% | +5% | +9% | **+6.3%** |

### Ablation Study

| Configuration | WVS Alignment | Description |
|------|-----------|------|
| Dialogue only | +4% | Imitative learning is effective but insufficient |
| Intention only | +2% | Instructed learning alone has limited effectiveness |
| Full CLCA | **+7%** | The combination of both works best |
| Random dialogue | +1% | Cultural adaptation is key |

### Key Findings
- **CLCA is effective across models**: Improvements are observed across multiple architectures.
- **Intention understanding is key**: Performance drops significantly when it is removed.
- **Greater gains in non-Western cultures**: The gains in Saudi Arabia and China are more pronounced.

## Highlights & Insights
- **Transfer from cultural learning theory to AI**—applying "imitation + instruction + intention understanding" to LLM cultural adaptation.
- **Behavior-driven vs. value-driven** is an important distinction: modifying values by simulating behavior.
- **Culturally adapted social scenario generation** can be transferred to other behavioral adaptation tasks.

## Limitations & Future Work
- Evaluation relies solely on the WVS questionnaire.
- Synthetic data quality is influenced by GPT-4.
- Collaborative learning (the third form of cultural learning) is not addressed.

## Related Work & Insights
- **vs. Prompt Engineering**: Prompts are adjusted during inference, whereas CLCA internalizes values during training.
- **vs. Choenni et al.**: They found generic corpora to be ineffective, whereas CLCA shows that social interaction data is effective.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Interdisciplinary innovation combining cultural learning theory and AI.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-model and multi-cultural validation with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Solid theoretical foundation.
- **Value**: ⭐⭐⭐⭐ Highly significant for global LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Culture is Not Trivia: Sociocultural Theory for Cultural NLP](culture_is_not_trivia_sociocultural_theory_for_cultural_nlp.md)
- [\[ACL 2025\] DenseLoRA: Dense Low-Rank Adaptation of Large Language Models](denselora_dense_low-rank_adaptation_of_large_language_models.md)
- [\[ACL 2025\] Commonsense Reasoning in Arab Culture](commonsense_arab_culture.md)
- [\[ACL 2025\] TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)
- [\[ACL 2025\] Towards Style Alignment in Cross-Cultural Translation](towards_style_alignment_in_cross-cultural_translation.md)

</div>

<!-- RELATED:END -->
