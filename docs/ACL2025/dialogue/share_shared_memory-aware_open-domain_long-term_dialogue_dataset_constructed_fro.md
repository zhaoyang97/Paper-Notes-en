---
title: >-
  [Paper Note] SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script
description: >-
  [ACL 2025][Dialogue Systems][Long-term dialogue] This paper proposes SHARE, a long-term dialogue dataset constructed from movie scripts, introducing the concept of "shared memory" for the first time. It also designs the EPISODE dialogue framework to manage personal information, personal events, and shared memories, making long-term dialogues more intimate and engaging.
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "Long-term dialogue"
  - "Shared memory"
  - "Movie script"
  - "Dialogue memory management"
  - "Multi-session dialogue"
date: 2026-05-08
content_hash: abe9b225df68774f
---

# SHARE: Shared Memory-Aware Open-Domain Long-Term Dialogue Dataset Constructed from Movie Script

**Conference**: ACL 2025  
**arXiv**: [2410.20682](https://arxiv.org/abs/2410.20682)  
**Code**: [github](https://github.com/e1kim/SHARE)  
**Area**: Other  
**Keywords**: Long-term dialogue, Shared memory, Movie script, Dialogue memory management, Multi-session dialogue

## TL;DR

This paper proposes SHARE, a long-term dialogue dataset constructed from movie scripts, introducing the concept of "shared memory" for the first time. It also designs the EPISODE dialogue framework to manage personal information, personal events, and shared memories, making long-term dialogues more intimate and engaging.

## Background & Motivation

Dialogue memory plays a critical role in building relationships and facilitating ongoing conversations. Limitations of existing research in long-term dialogue include:

**Focus solely on individual memory**: Existing methods only utilize personal information (e.g., "I am a K-Pop fan") or short-term events (e.g., "going to the doctor") to support long-term dialogues, which are mostly suitable for initial interactions or casual chitchat.

**Neglect of shared memory**: In reality, conversations between old friends revolve heavily around shared memories (e.g., "Do you remember the music festival we went to last year?"), but research on this is virtually non-existent.

**Difficulties in data collection**: Crowdsourced dialogues require manually creating role-playing scenarios, which is costly. LLM-generated data primarily focuses on explicitly stated events, ignoring implicit events that must be inferred from the dialogue.

**Comparison with existing datasets** (Table 1): Datasets such as MSC, LoCoMo, and CONVERSATION CHRONICLES do not contain information on shared memories.

**Key Innovation**: Movie scripts naturally contain shared memories between characters; dialogues not only depict the characters and their relationships but also convey shared memory information that is not directly shown in the scenes.

## Method

### Overall Architecture

SHARE dataset construction + EPISODE dialogue framework. Data is extracted from movie scripts and contains four types of information: persona, personal events, mutual events, and shared memories. Based on this, the EPISODE framework manages and utilizes these memories for response generation.

### Key Designs

1. **Dataset Construction (SHARE)**:

    - **Data Source**: 1,201 movie scripts collected from IMSDB, DailyScript, and Simply Scripts, covering various genres such as romance, comedy, and action.
    - **Script Preprocessing**: A movie script parser is used to structure scripts into dialogues. Only dyadic dialogues are retained, where each scene is treated as a session, and multiple sessions of the same character pair constitute an episode ($\ge 3$ sessions).
    - **Information Extraction** (using GPT-4):
        - Persona: Personality, interests, etc.
        - Personal events: Transient information such as current health status.
        - Mutual events: Real-time events between the two participants (explicitly inferable from the current session).
        - Shared memories: Past events shared by the two participants prior to the current session (requiring implicit inference).
    - **Annotation**: Each utterance is linked to its corresponding memory set information using GPT-3.5-turbo.

2. **EPISODE Dialogue Framework**:

    - **Memory Selection**: A memory selector is trained based on Llama-3-8B to select memories relevant to the current context from the growing memory set $\mathcal{M}_{(u,v)} = \{\mathcal{P}_u, \mathcal{P}_v, \mathcal{E}_u, \mathcal{E}_v, \mathcal{S}_{(u,v)}\}$. An "Everyday Language" option is added to the candidate pool for chit-chat scenarios that do not require specific memories.
    - **Response Generation**: Gemma (2B) and Llama-3.1-Instruct (8B) are used as backbone models to generate responses by combining the selected memory $\mathbf{m}$ and the current context $\mathbf{c}$.

3. **Memory Management (Asynchronous Update)**:

    - **Information Extraction**: After each session, new information is extracted using a trained Llama-3-8B.
    - **Memory Updating Strategy**:
        - Accumulation: Newly extracted information is directly added if it is independent of existing memories.
        - Sequential Update: Updated when there is a causal/sequential relationship between the new information and existing memories.
        - Conflict Update: Updated when the new information contradicts existing memories.
        - Deduplication: Not updated if the new information duplicates existing memories.

### Loss & Training

Standard supervised fine-tuning (SFT) is employed, with the dataset split 8:1:1 into training/validation/test sets. The memory selector and response generator are fine-tuned on Llama-3-8B and the respective backbone models using the collected training data.

## Key Experimental Results

### Main Results

**Automatic Evaluation Metrics** (Llama-3.1-Instruct 8B):

| Method | BLEU-3/4 | ROUGE-1/2 | ROUGE-L | BertSim | Dist-1/2 |
|------|----------|-----------|---------|---------|----------|
| Zero-shot | 0.0122/0.0099 | 0.0997/0.0213 | 0.0923 | 0.8474 | 0.5458/0.8470 |
| SHARE w/o memory | 0.0168/0.0135 | 0.1146/0.0329 | 0.1085 | 0.8592 | 0.6372/0.8432 |
| SHARE w/ predicted memory | **0.0267/0.0200** | **0.1392/0.0508** | **0.1290** | **0.8632** | 0.5676/0.8179 |
| SHARE w/ gold memory | 0.0500/0.0377 | 0.2205/0.1000 | 0.2040 | 0.8806 | 0.6171/0.8389 |

**Multi-Session GPT-4o Evaluation** (Llama-3.1-Instruct 8B, Session 6, 0-3 scale):

| Method | Coherence | Engagement | Intimacy | Reflectiveness |
|------|--------|--------|--------|--------|
| SHARE (w/o shared memory) | 2.5979 | 2.3538 | 1.6771 | — |
| SHARE + ACCUMULATION | 2.5958 | 2.3125 | 1.7271 | 1.3937 |
| SHARE + COMEDY | 2.0625 | 1.2521 | 1.0042 | — |
| SHARE + EPISODE | **2.6042** | **2.3625** | **1.7583** | **1.7604** |

### Ablation Study

| Configuration | Key Metrics | Note |
|------|---------|------|
| Predicted memory vs. Persona-only memory | BLEU-4: 0.0200 vs. 0.0116 | Using complete memory (including shared memory) significantly outperforms using persona-only memory. |
| EPISODE vs. ACCUMULATE | Reflectiveness: 1.76 vs. 1.39 (Session 6) | The memory updating strategy significantly outperforms simple accumulation. |
| EPISODE vs. COMEDY | Engagement: 2.36 vs. 1.25 (Session 6) | The compression strategy of COMEDY leads to information loss. |
| w/ vs. w/o shared memory | Reflectiveness: w/ vs. w/o | Relationship reflectiveness cannot be evaluated without shared memory. |

### Key Findings

1. **Importance of Shared Memory**: 61.57% of episodes contain at least one shared memory, demonstrating its importance in sustaining dialogues.
2. **Predicted Memory vs. Persona-only Memory**: Integrating shared memories generates richer and more diverse dialogues than using individual persona information alone.
3. **EPISODE Consistently Leads in Reflectiveness**: As sessions increase, the relationship reflectiveness of EPISODE continuously improves and outperforms all baselines.
4. **Memory Management Outperforms Simple Accumulation**: The structured memory strategy of EPISODE outperforms simple accumulation in both coherence and engagement.

## Highlights & Insights

- **Clever Choice of Movie Scripts as Data Source**: It leverages the implicit shared memory information naturally embedded in movie dialogues, addressing high crowdsourcing costs and the neglect of implicit information in LLM-generated data.
- **Introduction of the "Shared Memory" Concept**: It fills an important gap in long-term dialogue research. While existing works focus on "what you know" and "what happened," SHARE also focuses on "what we experienced together."
- **Complete End-to-End Framework**: It forms a complete closed loop from data construction to memory management and response generation.
- **Diverse Dialogue Styles**: It is not limited to daily conversations but also includes various genres such as fantasy.

## Limitations & Future Work

1. **Data Quality Dependency on GPT-4**: Information extraction relies entirely on GPT-4, which may introduce extraction biases.
2. **Monolingual English**: The dataset only covers English.
3. **Scripts vs. Real Dialogues**: The conversational style in movie scripts differs from real-world daily conversations.
4. **Subjective Evaluation**: Criteria for metrics like Reflectiveness are relatively subjective.
5. **Accuracy of Implicit Inference**: "Implicit inference" of shared memories is inherently a difficult task to verify.
6. **Limited Scale**: With 3,216 episodes and 17,679 sessions, the dataset is still relatively small compared to large-scale dialogue datasets.

## Related Work & Insights

- **Difference from MSC and LoCoMo**: These datasets only focus on personae and personal events, whereas SHARE introduces the dimension of shared memory.
- **Difference from CONVERSATION CHRONICLES**: Although the latter includes relationship information, it uses synthetic data, whereas SHARE is derived from real movie scripts.
- **Insights into Memory Management**: The four updating strategies of EPISODE (accumulation, sequential, conflict, and deduplication) provide a more granular memory management scheme than simple compression.
- **Future Directions**: The concept of shared memory can be extended to scenarios requiring relationship maintenance, such as game NPC dialogues, online customer services, and personal assistants.

## Rating

- Novelty: ⭐⭐⭐⭐ The concept of "shared memory" is novel, and the choice of movie scripts as the data source is clever.
- Experimental Thoroughness: ⭐⭐⭐⭐ Combines automatic evaluation, GPT-4o evaluation, and human evaluation, with multi-backbone comparisons.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, fitting quotation of Saint-Exupéry at the beginning, and rich tables.
- Value: ⭐⭐⭐⭐ Fills the gap of shared memory in long-term dialogue; both the dataset and framework are open-source.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](../../ACL2026/dialogue/apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[ACL 2025\] KokoroChat: A Japanese Psychological Counseling Dialogue Dataset Collected via Role-Playing by Trained Counselors](kokorochat_a_japanese_psychological_counseling_dialogue.md)
- [\[ACL 2025\] Enabling Chatbots with Eyes and Ears: An Immersive Multimodal Conversation System](enabling_chatbots_with_eyes_and_ears_an_immersive_multimodal_conversation_system.md)
- [\[ACL 2025\] Dialogue Systems for Emotional Support via Value Reinforcement](dialogue_systems_for_emotional_support_via_value_reinforcement.md)
- [\[ACL 2025\] EnSToM: Enhancing Dialogue Systems with Entropy-Scaled Steering Vectors for Topic Maintenance](enstom_enhancing_dialogue_systems_with_entropy-scaled_steering_vectors_for_topic.md)

</div>

<!-- RELATED:END -->
