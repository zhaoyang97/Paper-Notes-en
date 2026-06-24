---
title: >-
  [Paper Note] When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation
description: >-
  [ACL 2025][Dialogue Systems][Persona-based Dialogue] This paper systematically investigates the impact of **interlocutor information** on target speaker generation quality in persona-based dialogue. Through an evaluation framework that masks/reveals interlocutor information, the study discovers that models effectively adapt to the interlocutor's persona, show weaker generalization to unfamiliar interlocutors than to unfamiliar topics, and that LLMs tend to "copy-paste" person…
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "Persona-based Dialogue"
  - "Interlocutor Influence"
  - "Role-Play LLM"
  - "Speaker Identification"
  - "Dialogue Evaluation"
date: 2026-05-08
content_hash: ca2e91fbc0739ea2
---

# When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation

**Conference**: ACL 2025  
**arXiv**: [2505.24613](https://arxiv.org/abs/2505.24613)  
**Code**: None  
**Area**: Dialogue Generation / Role-Play  
**Keywords**: Persona-based Dialogue, Interlocutor Influence, Role-Play LLM, Speaker Identification, Dialogue Evaluation

## TL;DR

This paper systematically investigates the impact of **interlocutor information** on target speaker generation quality in persona-based dialogue. Through an evaluation framework that masks/reveals interlocutor information, the study discovers that models effectively adapt to the interlocutor's persona, show weaker generalization to unfamiliar interlocutors than to unfamiliar topics, and that LLMs tend to "copy-paste" persona details in zero-shot settings.

## Background & Motivation

Role-Play LLMs have made significant progress recently, enabling the simulation of specific characters' dialogue styles and features. Existing studies primarily focus on the persona consistency of the **target speaker**—i.e., whether the model can faithfully express the designated character. However, in natural conversations, how one speaks depends not only on "who I am" but also on "who I am talking to". For example, Harry's style when chatting with Sally (from the movie *When Harry Met Sally...*) should be distinctly different from his style when chatting with Superman.

**The influence of interlocutor information on dialogue generation** remains severely understudied in prior research:
- Liu et al. (2020), Gu et al. (2021), and Xu et al. (2022) only observed interlocutor adaptation to some extent without systematic evaluation.
- There is no research framework that directly measures the impact of interlocutor information on target speaker identification.
- The model's generalization capability across different interlocutor pairings and topics has not been explored.

This paper systematically addresses these gaps through three research questions:
- **Q1**: Are models better at adapting to the target speaker or the interlocutor?
- **Q2**: How is the models' generalization capability to unfamiliar interlocutors and unfamiliar topics?
- **Q3**: Is additional fine-tuning actually beneficial?

## Method

### Overall Architecture

The study adopts a four-step pipeline:
1. Fine-tune Llama 3.1 8B Instruct on the PRODIGy movie dialogue dataset.
2. Construct various speaker pairing combinations (familiar/unfamiliar interlocutor $\times$ familiar/unfamiliar topic).
3. Generate dialogues using the fine-tuned model and zero-shot settings, respectively.
4. Perform systematic evaluations using LLM-as-a-judge and human evaluations, with the core task being **speaker identification**.

The core idea of the evaluation is: if the target speaker is harder to identify after interlocutor information is masked, it indicates that the model actually utilized interlocutor information effectively for adaptation.

### Key Designs

1. **Four Information Disclosure Configurations**: This is the core evaluation paradigm design of this paper:
    - **BothDisc**: Both the interlocutor's biography and dialogue turns are visible—the full-information baseline.
    - **BioDisc**: Only the interlocutor's biography is visible, and dialogue turns are masked—tests the effect of the biography alone.
    - **TurnsDisc**: Only the interlocutor's turns are visible, and the biography is masked—tests the effect of dialogue content alone.
    - **BothMask**: Interlocutor information is completely masked—tests identification capability without interlocutor clues.

2. **Multi-dimensional Speaker Pairing Design**:
    - **Familiar pairings**: Character pairs from the same movie (e.g., Harry-Sally) or newly created non-PRODIGy character pairs.
    - **Unfamiliar pairings**: Cross-movie character mixes (e.g., Harry-Superman) or a mix of PRODIGy and new characters.
    Topics are also divided into familiar (character-related topics) and unfamiliar (character-unrelated topics), resulting in several experimental conditions.

3. **Non-PRODIGy Character Generation**: Using GPT-4o-mini combined with the Persona Hub dataset (1 billion persona descriptions) to generate entirely new character biographies to ensure no data contamination. This representative set of everyday people (instead of fictional characters) is used to test generalization to completely unseen personas.

### Loss & Training

- **Fine-tuning**: Fine-tuned Llama 3.1 8B Instruct on the PRODIGy dataset, which contains 5,660 annotated two-party dialogues, partitioned with an 80:10:10 split.
- **Dialogue Generation**: Turn-by-turn generation, incorporating historical dialogue into the prompt each turn, with each dialogue limited to 8 turns (as the study found persona consistency decays after 8 turns).
- **Biography Truncation**: Consistently uses the first 5 sentences of each character's biography to ensure fairness.
- **Evaluation LLM-as-a-judge**: Fine-tuned Llama 3.1 8B Instruct as the evaluator, using greedy decoding to ensure consistency, evaluating each dialogue twice (once for the target speaker, once for the interlocutor).
- **Human Evaluation**: Recruited native English speakers via Prolific, with 3 annotators per dialogue.

## Key Experimental Results

### Main Results

**Evaluation of Gold Dialogues (Human-written reference dialogues):**

| Disclosure | Accuracy | Precision | Recall | F1 |
|---------|----------|-----------|--------|-----|
| BothDisc | 0.820 | 0.838 | 0.817 | 0.819 |
| BioDisc | 0.805 | 0.822 | 0.801 | 0.802 |
| TurnsDisc | 0.588 | 0.629 | 0.583 | 0.581 |
| BothMask | 0.577 | 0.619 | 0.571 | 0.568 |

**Dialogues Generated by the Fine-tuned Model:**

| Disclosure | Accuracy | F1 |
|---------|----------|-----|
| BothDisc | 0.594 | 0.588 |
| BioDisc | 0.594 | 0.589 |
| TurnsDisc | 0.517 | 0.508 |
| BothMask | 0.515 | 0.505 |

**Dialogues Generated by the Zero-shot Model:**

| Disclosure | Accuracy | F1 |
|---------|----------|-----|
| BothDisc | 0.755 | 0.754 |
| BothMask | 0.735 | 0.733 |

### Ablation Study

| Configuration | Accuracy | Description |
|------|----------|------|
| Familiar Interlocutor Pairing (FT) | 0.703 | Model performs well on familiar pairings |
| Unfamiliar Interlocutor Pairing (FT) | 0.496 | Performance drops significantly on unfamiliar pairings |
| Familiar Topic (FT) | 0.566 | Minimal impact from topic differences |
| Unfamiliar Topic (FT) | 0.561 | Model shows strong generalization across topics |
| Familiar Interlocutor (Zero-shot) | 0.816 | Zero-shot yields higher identification for familiar pairings |
| Unfamiliar Interlocutor (Zero-shot) | 0.696 | But mainly due to "copy-pasting" |
| Zero-shot Rare Word Copy Rate | 40.69% | Zero-shot heavily copies rare words from biographies |
| FT Model Rare Word Copy Rate | 5.18% | FT model copies less, indicating deeper adaptation |
| Gold Dialogue Rare Word Copy Rate | 5.84% | FT results are close to human references |

### Key Findings

1. **Interlocutor Biography is More Crucial than Dialogue Turns**: The performances of BioDisc and BothDisc are similar, but performance degrades significantly after masking the biography (TurnsDisc, BothMask).
2. **"Who to Speak to" Matters More than "What to Talk About"**: The model's generalization to unfamiliar interlocutors is significantly worse than to unfamiliar topics (pairing gap of 20%+ vs. topic gap of <1%).
3. **High Zero-shot Scores are an Illusion of "Copy-Pasting"**: Zero-shot METEOR score (0.140) is much higher than fine-tuned (0.043), and the rare word rate is 8x higher, showing that zero-shot only captures surface-level features.
4. **Fine-tuning Learns Deeper Features**: The copy rate of the fine-tuned model is close to the gold data, indicating that it captures deeper persona features rather than simple replication.
5. **Human and LLM Evaluation Consistency**: Human evaluation trends align with LLM evaluation, but overall scores are lower, suggesting LLMs outperform humans on this task.

## Highlights & Insights

- **Innovative Evaluation Paradigm**: Evaluating the model's adaptive capability indirectly by masking/revealing interlocutor information reveals deeper mechanisms better than traditional direct-matching evaluations.
- **Discovery of the "Copy-Paste" Phenomenon**: Reveating that zero-shot LLMs tend to directly copy biographical details from the prompt. While resulting in high scores, this indicates shallow adaptation—offering an important warning for role-play evaluation.
- **Systematic Experimental Design**: The cross-experimental design of familiar/unfamiliar interlocutors $\times$ topics effectively answers the question of "which matters more".
- **Dialogue Quality vs. Speaker Variation**: Clearly demonstrates that "who I am talking to" is more critical than "what topic I am talking about", aligning with sociolinguistic intuition.
- **Non-PRODIGy Character Design**: Prevents data contamination effects by introducing newly generated characters, ensuring rigorous experimental design.

## Limitations & Future Work

1. **Single Model Limitation**: Only Llama 3.1 8B Instruct was used; results have not been validated across other model families, where behavior may differ.
2. **English-centric Scope**: Based on Western movie characters and English dialogues; character adaptation under different cultural/linguistic backgrounds may vary.
3. **Dialogue Turns Constraint**: Fixed at 8 turns; the dynamics of persona drift and adaptation in longer dialogues remain unexplored.
4. **Lack of Automated Deep Evaluation**: Currently primarily relies on speaker identification tasks, lacking a comprehensive evaluation of dialogue quality, coherence, and emotional adaptation.
5. Future work can explore **multi-turn interlocutor switching** scenarios and **how interlocutor information can be actively utilized** to improve generation strategies.

## Related Work & Insights

- **PRODIGy (Occhipinti et al., 2024)**: Provides a movie dialogue dataset with rich speaker annotations, which is the foundation for the systematic evaluations in this paper.
- **RoleLLM (Wang et al., 2024)** and **CharacterEval (Tu et al., 2024)**: Focus on the evaluation of role-play but do not explore the impact of the interlocutor.
- **CAMEL (Li et al., 2023)** and **Park et al. (2023)**: Multi-character interaction systems, but without a systematic study on interlocutor adaptation.
- **Insight**: Role-play evaluations should not only look at target-character consistency; the adaptive capability to the interlocutor is equally important.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to systematically investigate the effect of the interlocutor on persona-based dialogue generation, with an innovative evaluation framework design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive cross-experiments of FT/Zero-shot $\times$ 4 information disclosures $\times$ familiar/unfamiliar pairings & topics, including human evaluations.
- **Writing Quality**: ⭐⭐⭐⭐ Clear research questions (Q1-Q3), with each experimental section directly addressing one question, ensuring a well-structured paper.
- **Value**: ⭐⭐⭐⭐ Unveils the illusion of zero-shot "copy-pasting" and the significance of interlocutor adaptation, providing practical guidance for developing and evaluating role-play LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](persona_sentiment_dialogue.md)
- [\[ACL 2025\] KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors](kokorochat_a_japanese_psychological_counseling_dialogue.md)
- [\[ACL 2026\] SPASM: Stable Persona-driven Agent Simulation for Multi-turn Dialogue Generation](../../ACL2026/dialogue/spasm_stable_persona-driven_agent_simulation_for_multi-turn_dialogue_generation.md)
- [\[ACL 2025\] Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching](wizard_of_shopping_target-oriented_e-commerce_dialogue_generation_with_decision_.md)
- [\[ACL 2025\] UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](uniconv_retrieval_response_gen.md)

</div>

<!-- RELATED:END -->
