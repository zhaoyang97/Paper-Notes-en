---
title: >-
  [Paper Note] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles
description: >-
  [ACL 2025][Dialogue Systems][User Simulator] This paper proposes the USP (User Simulator with Implicit Profiles) framework. By extracting implicit user profiles from human-machine dialogues and combining conditional supervised fine-tuning with cycle-consistency-based reinforcement learning, USP significantly outperforms baseline methods across three dimensions: authenticity, consistency, and diversity, improving semantic similarity and style similarity by approximately 34% an…
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "User Simulator"
  - "Implicit Profiles"
  - "Cycle Consistency"
  - "Reinforcement Learning"
  - "Multi-turn Dialogue"
date: 2026-05-08
content_hash: 3e5bda6a1b4a65ec
---

# Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles

**Conference**: ACL 2025  
**arXiv**: [2502.18968](https://arxiv.org/abs/2502.18968)  
**Code**: [Available](https://github.com/wangkevin02/USP)  
**Area**: NLP / Dialogue Systems / User Simulation  
**Keywords**: User Simulator, Implicit Profiles, Cycle Consistency, Reinforcement Learning, Multi-turn Dialogue

## TL;DR

This paper proposes the USP (User Simulator with Implicit Profiles) framework. By extracting implicit user profiles from human-machine dialogues and combining conditional supervised fine-tuning with cycle-consistency-based reinforcement learning, USP significantly outperforms baseline methods across three dimensions: authenticity, consistency, and diversity, improving semantic similarity and style similarity by approximately 34% and 43%, respectively.

## Background & Motivation

User simulators are crucial in dialogue system research, acting as agents for real users to interact with dialogue systems and supporting collaborative training and automated evaluation. However, current methods face three major challenges:

**Limitations of Role-Playing Methods**: LLM-based role-playing methods rely on predefined well-known personas, lacking **utterance-level realism** and **user-level diversity**. LLMs are typically trained as polite, helpful assistants, leading to "role confusion" where they still exhibit assistant characteristics when simulating users.

**Limitations of Direct Simulation**: Although models like PlatoLM and Parrot learn from real human-machine dialogues, they only focus on surface-level textual features, ignoring implicit user traits (such as personality and conversational consistency) and lacking conversational control.

**Lack of User Diversity Distribution**: Existing methods fail to capture the distribution characteristics of real-world user profiles, which is essential for analyzing cohort behaviors.

Core Idea: **To simulate a user well, one must understand the user first** ("Know You First and Be You Better"). User simulation is formulated as a dialogue reconstruction task, where implicit user profiles are inferred from dialogues, and personalized conversations are subsequently generated based on these profiles.

## Method

### Overall Architecture

USP consists of four phases: (1) user profile extractor construction; (2) conditional supervised fine-tuning (utterance-level); (3) diverse profile sampling; and (4) cycle-consistency reinforcement learning (conversation-level).

### Key Designs

1. **Implicit User Profile Construction**:

    - **Profile Schema Design**: Based on interpersonal interaction theory, a profile schema is designed with two dimensions:
        - **Objective Facts (OF)**: Scenario-consistent dimensions (9 attributes such as age, gender, occupation, education, etc.) + Scenario-specific details (goals, task details)
        - **Subjective Characteristics (SC)**: Inner traits (Big Five personality traits) + Outer expressions (language style)
    - **Profile Extractor**: GPT-4o is utilized to extract the aforementioned attributes from human-machine dialogues, and the discrete attributes are then integrated into natural language descriptions.
    - **Profile Quality Verification**: The Dialogue Profile Consistency (DPC) metric is proposed, evaluating the consistency between profiles and dialogues using an F1-based retrieval paradigm.
    - Design Motivation: Unlike existing work that uses discrete attributes, natural language descriptions enhance generalization and flexibility.

2. **Conditional Supervised Fine-Tuning (Conditional SFT)**:

    - Conditioning on the extracted user profile $p_i$ and conversational context $c_{i,j}$, the LLM is trained to generate user utterances.
    - The loss function only optimizes the user utterance portion (excluding system responses) to address the objective misalignment between user simulation and assistant response modeling.
    - Design Motivation: To learn profile-conditioned generation capability at the utterance level.

3. **Diverse Profile Sampling**:

    - Embed profiles into a semantic space using SimCSE $\rightarrow$ UMAP dimensionality reduction $\rightarrow$ fit the distribution using Gaussian Kernel Density Estimation (KDE).
    - Supports probabilistic sampling to generate profiles that conform to the real-world distribution.
    - Synthesizes virtual profiles by combining nearest-neighbor OF and SC descriptions, thereby increasing profile diversity.
    - Design Motivation: To capture the distribution of real user profiles, ensuring representation of both majority and minority cohorts.

4. **Cycle-Consistency Reinforcement Learning (RLCC)**:

    - Core Idea: SFT ensures forward consistency (from profile $\rightarrow$ utterance) but does not guarantee backward consistency (i.e., whether the generated dialogue reflects the profile).
    - Execution: USP interacts with GPT-4o based on the target profile $p_i$ to generate a simulated dialogue $d_i'$, then extracts a reconstructed profile $p_i'$ from $d_i'$, maximizing the semantic similarity between $p_i$ and $p_i'$.
    - Reward Function: $r_{i,j} = \lambda r^{cc}_{i,j} + (1-\lambda) r^{ai\_detect}_{i,j}$, where $\lambda=0.8$.
    - The AI detection reward serves as an auxiliary component to prevent reward hacking.
    - Optimized using PPO.
    - Design Motivation: To ensure the complete expression of profiles at the conversation level.

### Loss & Training

- Base Model: LLaMA-3-8B-Base
- SFT: 4 $\times$ A100 40GB, 3 epochs, learning rate 5e-5, ~2 days
- RLCC: 2 $\times$ H20 96GB, KL coefficient 0.01, learning rate 5e-7, 1 epoch, ~5 days
- Sampling: Selecting the 5,000 profiles least similar to the training set from 1 million generated profiles.

## Key Experimental Results

### Main Results I: Utterance-level Evaluation

| Model | Semantic Similarity↑ | Style Similarity↑ | AVA↑ | r-DP.P↑ | P.Cover↑ |
|------|-----------|-----------|------|---------|----------|
| GPT-4o (w/o Profile) | 40.24 | 13.75 | 11.28 | - | - |
| PlatoLM | 39.37 | 43.11 | 40.29 | - | - |
| GPT-4o (w/ Profile) | 41.66 | 5.74 | 9.87 | 92.73 | 73.34 |
| LLaMA3 (w/ Profile) | 39.82 | 14.88 | 13.47 | 82.19 | 72.29 |
| USP (w/o RLCC) | 54.25 | 46.57 | 43.61 | 71.30 | 71.56 |
| **USP** | **53.38** | **46.60** | **43.35** | **72.61** | **71.23** |

### Main Results II: Conversation-level Evaluation

| Model | ESR↓ | Semantic Similarity↑ | Style Similarity↑ | r-DPC↑ | SC.Score↑ |
|------|------|-----------|-----------|--------|----------|
| GPT-4o (w/ Profile) | 32 | 48.87 | 10.15 | 55.66 | 4.56 |
| PlatoLM | 18 | 43.24 | 32.43 | - | - |
| CharacterGLM | 44 | 40.19 | 10.86 | 33.85 | 3.64 |
| USP (w/o RLCC) | 12 | 66.17 | 40.01 | 61.13 | 3.24 |
| **USP** | **10** | **65.39** | **46.23** | **64.05** | **3.35** |

### Ablation Study

| Configuration | ESR↓ | Semantic Similarity↑ | r-DPC↑ |
|------|------|-----------|--------|
| USP (w/o RLCC, w/o Sampling) | 17 | 64.22 | 57.47 |
| USP (w/o RLCC) | 12 | 66.17 | 61.13 |
| USP ($\lambda=5:5$) | 14 | 66.28 | 60.39 |
| USP ($\lambda=8:2$) | **10** | **65.39** | **64.05** |
| USP ($\lambda=9:1$) | 12 | 66.91 | 63.90 |

### Key Findings

1. **USP Leads Significantly in Authenticity**: Semantic similarity of 53.38 vs. PlatoLM's 39.37 ($\uparrow$34%), and style similarity of 46.60 vs. PlatoLM's 43.11.
2. **RLCC Significantly Boosts Conversation Consistency**: r-DPC improved from 61.13 to 64.05, and the win rate for consistency in human evaluation is 43 vs. 30.
3. **Lowest Early Stop Rate (10)**: USP avoids common issues such as repetitive generation and thank-you loops.
4. **60% of Samples with ADV < 5%**: The distribution of dialogues generated by USP is highly aligned with the target distribution.

## Highlights & Insights

- The **"understand the user first, then become the user"** philosophy is elegant and powerful, featuring a clever paradigm design that combines implicit profile extraction and conditional generation.
- **Cycle Consistency** is the core innovation: ensuring the complete expression of the profile through a "generate $\rightarrow$ extract $\rightarrow$ compare" loop.
- The **multi-level evaluation framework** is a valuable reference: utterance-level (semantic/style similarity, AVA) + conversation-level (ESR, r-DPC) + human evaluation.
- The design of the DPC metric (an F1-based method using atomic fact verification) is of generic value for evaluating profile-dialogue consistency.
- Downstream Application Validation: Applying USP to evaluate the multi-turn dynamic performance of LLMs aligns highly with mainstream benchmarks.

## Limitations & Future Work

1. The base model is limited to LLaMA-3-8B; using larger models may yield further improvements.
2. The training duration is relatively long (SFT: 2 days + RLCC: 5 days), implying a high computational cost.
3. Evaluated only on English dialogues, leaving cross-lingual generalization capability unknown.
4. Profile extraction depends on GPT-4o, which might be restricted in privacy-sensitive scenarios.
5. The AI detection reward might introduce undesirable signals due to detection model biases.

## Related Work & Insights

- Unlike CharacterLLM (Shao et al., 2023) and CharacterGLM (Zhou et al., 2024), USP does not rely on predefined profiles.
- The concept of cycle consistency is inspired by CycleGAN, and its application in NLP warrants further exploration.
- The diverse profile sampling method can be adapted for other personalized generation tasks.
- Provides a more reliable user agent for Sim2Real (simulation-to-reality) applications.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Implicit profile extraction combined with cycle-consistency reinforcement learning; the innovations are clear and effective.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Multi-dimensional metrics (authenticity/consistency/diversity/continuity) and multi-level evaluation (utterance/conversation/human).
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured with thorough methodology explanations, though the heavy use of mathematical formulations makes it slightly dense to read.
- **Value**: ⭐⭐⭐⭐ — Offers effective and highly practical tools for dialogue system evaluation and user simulation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling](know_your_mistakes_towards_preventing_overreliance_on_task-oriented_conversation.md)
- [\[ACL 2026\] Your Students Don't Use LLMs Like You Wish They Did](../../ACL2026/dialogue/your_students_dont_use_llms_like_you_wish_they_did.md)
- [\[ACL 2025\] DEMO: Reframing Dialogue Interaction with Fine-grained Element Modeling](demo_reframing_dialogue_interaction_with_fine-grained_element_modeling.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[ICML 2026\] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives](../../ICML2026/dialogue/is_your_llm_overcharging_you_tokenization_transparency_and_incentives.md)

</div>

<!-- RELATED:END -->
