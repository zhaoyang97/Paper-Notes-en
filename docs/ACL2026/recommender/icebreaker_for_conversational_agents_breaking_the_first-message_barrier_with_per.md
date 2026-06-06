---
title: >-
  [Paper Note] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters
description: >-
  [ACL 2026][Recommender Systems][Proactive Dialogue Initiation] This paper proposes IceBreaker, which addresses the "first-message barrier" of conversational agents through a two-step "handshake"—Resonance-Aware Interest…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Proactive Dialogue Initiation"
  - "Personalized Conversation Starters"
  - "Interest Distillation"
  - "Preference Alignment"
  - "Cold Start"
date: 2026-05-08
content_hash: 54a71e831dbf8ce2
---

# IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters

**Conference**: ACL 2026  
**arXiv**: [2604.18375](https://arxiv.org/abs/2604.18375)  
**Code**: N/A (Industrial deployment)  
**Area**: Recommender Systems / Dialogue Systems  
**Keywords**: Proactive Dialogue Initiation, Personalized Conversation Starters, Interest Distillation, Preference Alignment, Cold Start

## TL;DR
This paper proposes IceBreaker, which addresses the "first-message barrier" of conversational agents through a two-step "handshake"—Resonance-Aware Interest Distillation to capture triggered interests and Interaction-Oriented Starter Generation coupled with personalized preference alignment. A/B testing in one of the world's largest conversational products demonstrated an increase in user active days by +1.84‰ and CTR by +94.25‰.

## Background & Motivation

**Background**: Conversational agents (e.g., ChatGPT, Doubao) are transitioning from passive response to proactive engagement. Prior research focus on proactivity within dialogues, such as generating follow-up questions or topic guidance, but these occur after a conversation has already started.

**Limitations of Prior Work**: There is a neglected product bottleneck during the initiation phase—the "first-message barrier." Users may have vague needs without explicit query intent and lack awareness of the agent's capabilities, leading approximately 20% of users to exit the product without initiating any dialogue.

**Key Challenge**: The initiation phase lacks immediate context to guide responses. Unlike the mid-dialogue phase, initiation must operate at a "cold start" moment without explicit user intent. Furthermore, user preferences are highly personalized and follow a long-tail distribution; uniform alignment targets tend to generate generalized starters that fail to resonate with individuals.

**Goal**: Formulate proactive initiation as a "conversational starter generation" task to generate personalized starter questions that guide users to begin a dialogue.

**Key Insight**: Imitate how humans initiate conversations in cold-start scenarios—first identify potential points of resonance, then use appropriate phrasing to trigger interaction.

**Core Idea**: A two-step handshake framework—first extracting triggered interests from session summaries via Resonance-Aware Interest Distillation, then generating a list of starters via an interaction-oriented generator, optimized for personalized interaction utility and intra-list diversity through list-level multi-dimensional preference alignment.

## Method

### Overall Architecture
IceBreaker consists of two stages: (1) Resonance-Aware Interest Distillation (RID)—filtering interests from historical session summaries most likely to trigger re-interaction; (2) Interaction-Oriented Starter Generation (ISG)—generating a diverse list of starters conditioned on distilled interests, aligned with personalized interaction utility using DPO.

### Key Designs

1.  **Resonance-Aware Interest Distillation (RID)**:
    - **Function**: Extracts "triggered interests" from historical session summaries most likely to provoke interaction.
    - **Mechanism**: Learns a personalized resonance scorer $s_\phi(u, h_t) = \cos(\mathbf{u}, \mathbf{z}_t)$, measuring the match between user features and summary embeddings using cosine similarity. Training signals come from "interest re-visitation"—if a user revisits a historical interest in a subsequent session, the summary is a positive sample. It uses intra-user and cross-user negative sampling for contrastive learning. During inference, an adaptive threshold $\tau_u$ based on user activity groups filters the triggered interest set $\mathcal{I}^*$.
    - **Design Motivation**: Not all historical interests trigger new dialogues. "Interest re-visitation" is the most direct resonance signal—topics users are willing to re-discuss are effective triggers. Adaptive gating ensures higher thresholds for active users (avoiding information overload) and more lenient thresholds for low-activity users (increasing recall).

2.  **Supervised Interest Expansion Instruction Tuning (SIT)**:
    - **Function**: Establishes a solid initialization for the generator to ensure generation quality and topic coverage.
    - **Mechanism**: A teacher LLM generates high-quality starter instruction corpora $\mathcal{D}_{\text{cov}}$ from triggered interests $\mathcal{I}^*$, followed by fine-tuning the generator using standard language modeling loss.
    - **Design Motivation**: SIT expands the training distribution beyond limited observational data, providing a stable initialization for subsequent preference optimization.

3.  **Personalized Multi-Dimensional Alignment (PMA)**:
    - **Function**: Aligns the generator with user-specific interaction utility and intra-list diversity.
    - **Mechanism**: Preference pairs are iteratively constructed through hybrid reward list searches. Candidate pools are sampled from the current generator, and lists are expanded position-by-position using beam search, combining interaction utility reward $R_{\text{util}}$ and diversity reward $R_{\text{div}}$. Three types of negative samples are constructed—utility negatives, diversity negatives, and joint failure negatives—to decouple preference signals across dimensions for DPO training. The policy is resampled after each update to mine new preference pairs (self-augmented iterative optimization).
    - **Design Motivation**: Constructing preference pairs directly from user feedback faces extreme sparsity. By combining model self-generation with dual-reward guidance, effective preference supervision can be mined under sparse feedback. Decoupling the three negative sample types ensures optimization does not conflict between utility and diversity.

### Loss & Training
RID uses contrastive learning to train the resonance scorer. ISG is first warmed up with SFT, then aligned with DPO, where DPO preference pairs are iteratively mined via hybrid reward searching. After deployment, self-augmented optimization is performed periodically to track user preference drift.

## Key Experimental Results

### Main Results (Online A/B Test, >1 Month)

| Method | Active Days ‰ | Avg Sessions ‰ | CTR ‰ | Dialogue Start Rate ‰ |
| :--- | :--- | :--- | :--- | :--- |
| PE (Prompt Engineering) | -0.01 | -0.26 | -16.16* | -0.17 |
| SFT | +0.20 | +0.33 | +6.97* | -0.05 |
| SFT + DPO | +1.16 | +0.42* | +56.41* | +0.68 |
| **Ours (IceBreaker)** | **+1.84*** | **+1.59*** | **+94.25*** | **+1.27*** |

### Ablation Study (Offline, Doubao1.5-Lite backbone)

| Method | R-User ↑ | R-Score ↑ | Lexical Diversity ↑ | Semantic Diversity ↑ |
| :--- | :--- | :--- | :--- | :--- |
| PE | +0.56 | +0.08 | 29.45 | 6.23 |
| PE + RID | +0.71 | +0.38 | 25.13 | 4.86 |
| SFT | +0.78 | +0.44 | 28.97 | 5.59 |
| SFT + DPO | +0.79 | +0.52 | 12.94 | 2.37 |
| **Ours (IceBreaker)** | **+0.89** | **+0.80** | 28.83 | 5.28 |

### Key Findings
- RID distillation significantly enhances personalization: ranking consistency and scores improved markedly from prompt engineering to adding RID.
- Standard DPO improves utility but severely damages diversity (semantic diversity dropped from 5.59 to 2.37); IceBreaker resolves this through multi-dimensional preference alignment.
- Distribution analysis shows: RID filters out functional/generic head topics, shifting toward triggerable long-tail topics. ISG further leans toward interactive consumption topics (psychology, ACG, entertainment).
- All metrics in the online A/B test were significant ($p<0.05$), with the +94.25‰ CTR improvement being particularly prominent.

## Highlights & Insights
- Using "interest re-visitation" as a proxy for resonance signals is clever—topics users are willing to repeat naturally reflect deep interests. This signal construction can be transferred to interest modeling in recommender systems.
- The design of three types of negative samples (utility, diversity, and joint failure) elegantly decouples multi-dimensional optimization targets, avoiding the "utility vs. diversity seesaw" common in DPO.
- Moving dialogue systems from "passive answering" to "proactive initiation" is a significant paradigm shift. The 20% user loss due to the first-message barrier demonstrates that this is a critical product bottleneck.

## Limitations & Future Work
- The paper lacks open-source code and reproducible offline datasets; online experiments depend on industrial environments.
- The quality of triggered interests relies heavily on the accuracy of session summaries; poor historical summaries affect the entire pipeline.
- The current system assumes sufficient historical interaction data; cold-start scenarios for completely new users (zero history) are not discussed.
- Self-augmented iterative optimization may carry the risk of cumulative bias.

## Related Work & Insights
- **vs. Proactive Dialogue Methods**: Traditional proactivity (follow-up, clarification, topic transitions) relies on existing dialogue context; IceBreaker addresses proactive initiation before a dialogue begins, which is a more fundamental problem.
- **vs. General Preference Alignment (DPO/RLHF)**: General alignment targets cannot handle user-level personalization and feedback sparsity. IceBreaker's multi-dimensional alignment effectively solves these by iteratively mining preference pairs via model-generated candidates and hybrid reward search.
- **vs. Recommender Systems**: Starter generation is essentially "recommending dialogue entry points," but unlike traditional recommendations, it requires generating natural language rather than selecting existing items.

## Rating
- Novelty: ⭐⭐⭐⭐ The definition of the "first-message barrier" is novel; the two-step handshake and triple-negative sample design are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive offline and online A/B testing over a month, with rich distribution and case analyses.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation from a product perspective is compelling; methodology is clearly described.
- Value: ⭐⭐⭐⭐⭐ Already deployed in one of the world's largest conversational products, offering direct industrial application value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)
- [\[ACL 2026\] Intent-Driven Semantic ID Generation for Grounded Conversational News Recommendation](intent-driven_semantic_id_generation_for_grounded_conversational_news_recommenda.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)

</div>

<!-- RELATED:END -->
