---
title: >-
  [Paper Note] Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model
description: >-
  [ACL 2025][LLM (Other)][Role-Playing] The Beyond Dialogue framework is proposed to eliminate the bias between profiles and dialogues in role-playing training through Profile-Dialogue alignment, and introduces a sentence-level fine-grained alignment task to help models better understand and perform character traits.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Role-Playing"
  - "Profile-Dialogue Alignment"
  - "Fine-grained Alignment"
  - "Multi-task Training"
  - "Automated Evaluation"
date: 2026-05-08
content_hash: c91be9af0a0d6787
---

# Beyond Dialogue: A Profile-Dialogue Alignment Framework Towards General Role-Playing Language Model

**Conference**: ACL 2025  
**arXiv**: [2408.10903](https://arxiv.org/abs/2408.10903)  
**Code**: [github.com/yuyouyu32/BeyondDialogue](https://github.com/yuyouyu32/BeyondDialogue)  
**Area**: LLM/NLP  
**Keywords**: Role-Playing, Profile-Dialogue Alignment, Fine-grained Alignment, Multi-task Training, Automated Evaluation

## TL;DR

The Beyond Dialogue framework is proposed to eliminate the bias between profiles and dialogues in role-playing training through Profile-Dialogue alignment, and introduces a sentence-level fine-grained alignment task to help models better understand and perform character traits.

## Background & Motivation

### Limitations of Prior Work

**Limitations of Prior Work**: **Background**: Role-playing LLMs have become a hot direction in large model applications in recent years, with typical products like Character AI and Replika showing great potential. However, current role-playing training faces two core issues:

**Bias between Profile and Dialogue**: During training, a predefined complete character profile is used to guide dialogue generation under specific scenarios, but dialogues in a single scenario usually only reflect a portion of the profile's traits. For example, Hermione's speaking style is defined by four categories, but dialogue in a specific scenario might only reflect two of them. In the HPD dataset, 83.2% of dialogues deviate from the predefined profile. This bias can mislead model training.

**Lack of Fine-grained Alignment**: Models only learn a coarse mapping from profile to dialogue through dialogue tasks, lacking sentence-level understanding of "how personality traits are manifested in specific dialogue segments," resulting in a lack of depth in role-playing.

These two issues jointly limit the performance of general role-playing models and serve as the main motivation of this paper.

## Method

### Overall Architecture

The Beyond Dialogue framework consists of three core phases:

1. **Alignment Dataset Construction**: Scenario-level alignment and adjustment of character profiles
2. **Supervised Finetuning**: Training the model with a mixture of alignment data, dialogue data, and chit-chat data
3. **Automated Dialogue Evaluation**: Generating random scenarios for multi-turn dialogues to perform objective and quantitative evaluation

### Key Designs

1. **Coarse-grained Dialogue Dataset Construction Pipeline**: Fully automated extraction of role-playing dialogue data from novels/scripts. Text is first segmented and filtered by character frequency, an open-source model is utilized to extract dialogue scenarios and evaluate quality, GPT-4o is then used for dialogue extraction and scenario rendering, and finally a high-quality dataset containing 280 Chinese characters and 31 English characters is constructed (3,552 scenario dialogues, 23,247 turns).

2. **Profile Alignment & Adjustment**: This is the core contribution of this work. Using GPT-4o along with an innovative prompting mechanism, a sentence-level alignment analysis is performed on each multi-turn dialogue across five dimensions (Character, Style, Emotion, Relationship, Personality, or CSERP). Scenario profiles are then dynamically adjusted based on alignment results: removing traits not manifested in the dialogue, while supplementing scenario-related emotional and relational attributes. This ensures consistency between the training input (profile prompt) and output (dialogue label). After alignment, only 4.2% of dialogues are fully aligned with predefined profiles across all dimensions, validating the ubiquity of the bias.

3. **CSERP Fine-grained Alignment Task**: Five alignment training tasks are derived from each dialogue session, allowing the model to learn to explicitly associate profile attributes with dialogue sentences, enhancing the model's awareness and adherence to character profiles.

4. **Objectified Evaluation Pipeline**: Translating all evaluation tasks into objective questions (e.g., multiple-choice, true/false) and combining them with the "LLMs as Judges" method to quantify the model's profile adherence, resulting in lower variance than subjective grading and higher consistency with human evaluation.

### Loss & Training

- Uses a standard SFT negative log-likelihood loss
- Training data mix ratio: Alignment role-playing dialogue data $D_r$ : CSERP alignment data $D_a$ : Chit-chat data $D_c$ = 1:5:4
- Base models: Qwen2-7B-Instruct and Mistral-Nemo-Instruct-2407

## Key Experimental Results

### Main Results

| Model | Character Recall↑ | Style Recall↑ | Emotion NMAPE↓ | Relationship NMAPE↓ | Human-likeness↑ | Win-Rate vs GPT-4o↑ |
|------|-------------------|---------------|----------------|---------------------|-----------------|---------------------|
| GPT-4o | 74.32 | 81.67 | 16.31 | 12.13 | 67.33 | N/A |
| Baichuan-NPC-Turbo | 75.19 | 79.15 | 17.24 | 13.10 | 56.00 | 65.00 |
| Qwen2 + RPA & CC & CSERP | **78.04** | **81.58** | **16.29** | **11.37** | **64.33** | **71.67** |
| Mistral + RPA & CC & CSERP | 74.58 | 78.47 | 16.62 | 11.38 | 59.00 | 69.33 |

### Ablation Study

| Configuration | Character↑ | Style↑ | Win-Rate↑ | Description |
|------|-----------|--------|-----------|------|
| Base + RP & CC (No alignment) | 74.91 | 78.59 | 64.00 | Original dialogue training |
| Base + RPA & CC (With profile adjustment) | 76.43 | 80.93 | 67.33 | Significant improvement after profile alignment |
| Base + RPA & CC & CSERP (Full scheme) | 78.04 | 81.58 | 71.67 | Further improvement by incorporating CSERP |

### Key Findings

- Profile-Dialogue bias indeed severely impacts training: when training with unaligned data (RP & CC), the character adherence metrics actually drop.
- Profile Adjustment (RPA) brings significant improvements, validating the necessity of eliminating bias.
- The CSERP fine-grained alignment task further enhances the model's role-playing capability.
- The final 7B model trained on the Qwen2 base outperforms GPT-4o and the professional role-playing model Baichuan-NPC-Turbo in multiple dimensions such as Character and Style.
- Contrasting against GPT-4o, the Win-Rate reaches 71.67%, proving the effectiveness of the framework.

## Highlights & Insights

1. **Precise Problem Definition**: This work is the first to identify and quantify the profile-dialogue bias problem in role-playing training (83.2% of data contains bias), providing an important perspective for the field.
2. **Fully Automated & Low Cost**: The entire data construction and alignment pipeline is fully automated, requiring no human annotations.
3. **Objectified Evaluation Framework**: By translating subjective evaluation tasks into objective questions, it resolves the issue of inconsistent evaluation criteria in role-playing.
4. **Novel Sentence-level Alignment Approach**: Inspired by actor learning-methods, it enhances role-playing capability by learning "how character traits are manifested in dialogues."

## Limitations & Future Work

- Data construction relies on GPT-4o; though cost-effective, API calls are still required.
- Only verified on 7B scale models; performance on larger models remains unknown.
- Characters are sourced exclusively from novels and scripts, lacking validation on real characters like historical figures.
- GPT-4o is used as both dialogue conversationalist and judge in the evaluation, which might introduce evaluation bias.
- Whether the selection of the five CSERP dimensions is optimal and whether important dimensions are omitted can be further explored.

## Related Work & Insights

- **Difference from DITTO**: DITTO also focuses on general role-playing but does not address the profile-dialogue bias issue, nor does it perform fine-grained alignment.
- **Difference from RoleLLM**: RoleLLM generates dialogue data via GPT, but the generated dialogues lack human authenticity, and it also fails to resolve the bias problem.
- **Inspiration from Mixed-task Training**: Mixing tasks that are directly or indirectly related can significantly enhance downstream task performance; the CSERP alignment task is a successful practice of this concept.
- **Inspiration from Evaluation Methods**: The paradigm of objectifying evaluation tasks is worth adopting and can be applied to other scenarios that require evaluating LLM instruction-following capabilities.

## Rating

- Novelty: ⭐⭐⭐⭐ First to systematically identify and address the profile-dialogue bias problem, with a novel sentence-level alignment approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across multiple base models and dimensions; ablation study is logically designed, though only tested on 7B models.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, complete framework description, and intuitive illustrations.
- Value: ⭐⭐⭐⭐ Significant contribution to the role-playing domain, with open-sourced data and code, offering a practical methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)
- [\[CVPR 2025\] Chat-based Person Retrieval via Dialogue-Refined Cross-Modal Alignment](../../CVPR2025/llm_nlp/chat-based_person_retrieval_via_dialogue-refined_cross-modal_alignment.md)
- [\[ACL 2025\] X-Turing: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents](xturing_enhanced_turing_test.md)
- [\[ACL 2025\] Beyond Profile: From Surface-Level Facts to Deep Persona Simulation in LLMs](beyond_profile_from_surface-level_facts_to_deep_persona_simulation_in_llms.md)
- [\[ACL 2025\] Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases](can_llms_simulate_l2-english_dialogue_an_information-theoretic_analysis_of_l1-de.md)

</div>

<!-- RELATED:END -->
