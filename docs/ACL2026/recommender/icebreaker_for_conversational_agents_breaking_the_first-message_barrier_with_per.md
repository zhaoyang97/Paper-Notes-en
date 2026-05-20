---
title: >-
  [Paper Note] IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters
description: >-
  [ACL 2026][Recommender Systems][proactive conversation initiation] This paper proposes IceBreaker, a two-step "handshake" framework—resonance-aware interest distillation to capture trigger interests…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "proactive conversation initiation"
  - "personalized session starter"
  - "interest distillation"
  - "preference alignment"
  - "cold start"
date: 2026-05-08
content_hash: b64d3f04a4261c0d
---

# IceBreaker for Conversational Agents: Breaking the First-Message Barrier with Personalized Starters

**Conference**: ACL 2026
**arXiv**: [2604.18375](https://arxiv.org/abs/2604.18375)  
**Code**: N/A (industrial deployment system)  
**Area**: Recommender Systems / Dialogue Systems
**Keywords**: proactive conversation initiation, personalized session starter, interest distillation, preference alignment, cold start

## TL;DR
This paper proposes IceBreaker, a two-step "handshake" framework—resonance-aware interest distillation to capture trigger interests, followed by interaction-oriented starter generation with personalized preference alignment—to address the "first-message barrier" in conversational agents. A/B testing on one of the world's largest conversational products yields +1.84‰ active days and +94.25‰ CTR.

## Background & Motivation

**State of the Field**: Conversational agents (e.g., ChatGPT, Doubao) are evolving from passive responders to proactive participants. Existing research addresses proactivity within ongoing dialogues—such as follow-up question generation and topic steering—but these approaches presuppose that a conversation has already begun.

**Limitations of Prior Work**: A largely overlooked product bottleneck exists at the conversation initiation stage: the "first-message barrier." Users may have vague needs but lack explicit query intent, and are often unaware of the agent's capabilities, causing approximately 20% of users to exit the product without initiating any conversation.

**Root Cause**: The initiation stage lacks immediate context to guide responses. Unlike mid-conversation scenarios, initiation must operate at a "cold start" moment with no explicit user intent. Furthermore, user preferences are highly personalized and long-tail distributed, while uniform alignment objectives tend to produce generic starters that fail to resonate with individual users.

**Paper Goals**: Formalize proactive initiation as a "conversational starter generation" task, generating personalized opening questions to guide users into dialogue.

**Starting Point**: Emulate how humans initiate conversations in cold-start scenarios—first identifying a few interest points likely to resonate, then crafting appropriate phrasing to elicit engagement.

**Core Idea**: A two-step handshake framework: first, resonance-aware interest distillation extracts trigger interests from session summaries; then, an interaction-oriented generator produces a list of starters, optimized via list-level multi-dimensional preference alignment for personalized interaction utility and intra-list diversity.

## Method

### Overall Architecture
IceBreaker consists of two stages: (1) Resonance-Aware Interest Distillation (RID), which selects from historical session summaries the interest points most likely to trigger re-engagement; and (2) Interaction-oriented Starter Generation (ISG), which conditions on the distilled interests to produce a diverse list of starters, aligned to personalized interaction utility via DPO.

### Key Designs

1. **Resonance-Aware Interest Distillation (RID)**:

    - Function: Extract "trigger interests" from historical session summaries that are most likely to elicit re-engagement.
    - Mechanism: A personalized resonance scorer $s_\phi(u, h_t) = \cos(\mathbf{u}, \mathbf{z}_t)$ is learned via cosine similarity between user representations and session summary embeddings. Training signals derive from "interest revisits"—if a user's subsequent sessions revisit a historical interest, the corresponding session summary serves as a positive sample. Contrastive learning is applied with both intra-user and cross-user negatives. At inference, an activity-stratified adaptive threshold $\tau_u$ filters the trigger interest set $\mathcal{I}^*$.
    - Design Motivation: Not all historical interests can trigger new conversations. "Interest revisits" serve as the most direct resonance signal—topics users voluntarily revisit reflect genuine trigger points. Adaptive gating ensures higher thresholds for active users (avoiding information overload) and more relaxed thresholds for low-activity users (improving recall).

2. **Supervised Interest-Expansion Tuning (SIT)**:

    - Function: Establish a strong initialization for the generator, ensuring generation quality and topic coverage.
    - Mechanism: A teacher LLM generates high-quality starter instruction data $\mathcal{D}_{\text{cov}}$ conditioned on trigger interests $\mathcal{I}^*$, and the generator is fine-tuned with standard language modeling loss.
    - Design Motivation: SIT expands the training distribution beyond limited observed data, providing a stable initialization for subsequent preference optimization.

3. **Personalized Multi-Dimensional Alignment (PMA)**:

    - Function: Align the generator with user-specific interaction utility and intra-list diversity.
    - Mechanism: Preference pairs are iteratively constructed via mixed-reward list search. A candidate pool is sampled from the current generator, and lists are expanded position-by-position via beam search, combining interaction utility reward $R_{\text{util}}$ and diversity reward $R_{\text{div}}$ at each step. Three types of negatives are constructed—utility negatives, diversity negatives, and joint-failure negatives—to disentangle preference signals across dimensions, and DPO loss is applied. After each round, new preference pairs are mined by re-sampling from the updated policy (self-augmenting iterative optimization).
    - Design Motivation: Directly constructing preference pairs from user feedback suffers from extreme sparsity. Self-generated candidates guided by dual-dimensional rewards enable continuous mining of effective preference supervision under sparse feedback. The three decoupled negative types prevent optimization conflicts between utility and diversity.

### Loss & Training
RID trains the resonance scorer via contrastive learning. ISG first warms up with SFT, then applies DPO alignment, where preference pairs are iteratively mined through mixed-reward search. After deployment, self-augmenting optimization is periodically executed to track user preference drift.

## Key Experimental Results

### Main Results (Online A/B Test, >1 Month)

| Method | Active Days‰ | Avg. Sessions‰ | CTR‰ | Conversation Start Rate‰ |
|--------|-------------|----------------|------|--------------------------|
| PE (direct prompting) | -0.01 | -0.26 | -16.16* | -0.17 |
| SFT | +0.20 | +0.33 | +6.97* | -0.05 |
| SFT + DPO | +1.16 | +0.42* | +56.41* | +0.68 |
| **IceBreaker** | **+1.84*** | **+1.59*** | **+94.25*** | **+1.27*** |

### Ablation Study (Offline, Doubao1.5-Lite backbone)

| Method | R-User ↑ | R-Score ↑ | Lexical Diversity ↑ | Semantic Diversity ↑ |
|--------|---------|-----------|--------------------|--------------------|
| PE | +0.56 | +0.08 | 29.45 | 6.23 |
| PE + RID | +0.71 | +0.38 | 25.13 | 4.86 |
| SFT | +0.78 | +0.44 | 28.97 | 5.59 |
| SFT + DPO | +0.79 | +0.52 | 12.94 | 2.37 |
| **IceBreaker** | **+0.89** | **+0.80** | 28.83 | 5.28 |

### Key Findings
- RID distillation substantially improves personalization: transitioning from direct prompting to RID-augmented prompting yields clear improvements in ranking consistency and scores.
- Standard DPO improves utility but severely degrades diversity (semantic diversity drops from 5.59 to 2.37); IceBreaker's multi-dimensional preference alignment resolves this issue.
- Distribution analysis reveals that RID filters out functional/generic head topics and shifts toward trigger-oriented long-tail topics; ISG further biases toward interaction-friendly consumption topics (psychology, anime, entertainment).
- All metrics in the online A/B test are statistically significant (p<0.05), with CTR improvement of +94.25‰ being particularly notable.

## Highlights & Insights
- Using "interest revisits" as a proxy resonance signal is elegant—topics that users voluntarily revisit naturally reflect deep-seated interests. This signal construction paradigm is transferable to interest modeling in recommender systems.
- The three-type negative sample design (utility negatives, diversity negatives, joint-failure negatives) cleanly decouples multi-dimensional optimization objectives, avoiding the common "utility–diversity seesaw" problem in DPO.
- Advancing conversational systems from "passive response" to "proactive initiation" represents an important paradigm shift. The 20% user attrition attributed to the first-message barrier confirms this as a genuinely critical product bottleneck.

## Limitations & Future Work
- The paper lacks publicly available code and reproducible offline datasets; online experiments rely on an industrial environment.
- The quality of trigger interests is heavily dependent on the accuracy of session summaries; poor-quality historical summaries undermine the entire pipeline's foundation.
- The current system assumes sufficient historical interaction data per user; cold-start scenarios for entirely new users (zero history) are not addressed.
- Iterative self-augmenting optimization may risk accumulating bias over rounds.

## Related Work & Insights
- **vs. mid-conversation proactivity methods**: Traditional proactive dialogue (follow-up questions, clarification, topic steering) relies on existing conversational context. IceBreaker addresses cold-start proactive initiation before any conversation begins, representing a more fundamental problem.
- **vs. general preference alignment (DPO/RLHF)**: General alignment objectives cannot handle user-level personalization or feedback sparsity. IceBreaker's multi-dimensional preference alignment addresses both issues through self-generated candidates and iterative mixed-reward search.
- **vs. recommender systems**: Starter generation is essentially "recommending a dialogue entry point," but unlike traditional recommendation it requires generating natural language rather than selecting from existing items.

## Rating
- Novelty: ⭐⭐⭐⭐ The "first-message barrier" problem formulation is novel; the two-step handshake framework and three-type negative sample design are creative.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Offline and online A/B testing over more than one month, with rich distribution analysis and case studies.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation from a product perspective is highly convincing; method description is clear.
- Value: ⭐⭐⭐⭐⭐ Already deployed on one of the world's largest conversational products, demonstrating direct industrial applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] Where and What: Reasoning Dynamic and Implicit Preferences in Situated Conversational Recommendation](where_and_what_reasoning_dynamic_and_implicit_preferences_in_situated_conversati.md)
- [\[ACL 2026\] Personalized Benchmarking: Evaluating LLMs by Individual Preferences](personalized_benchmarking_evaluating_llms_by_individual_preferences.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)

</div>

<!-- RELATED:END -->
