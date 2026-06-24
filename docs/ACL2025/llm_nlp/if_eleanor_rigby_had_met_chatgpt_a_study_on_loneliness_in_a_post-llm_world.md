---
title: >-
  [Paper Note] If Eleanor Rigby Had Met ChatGPT: A Study on Loneliness in a Post-LLM World
description: >-
  [ACL 2025][LLM (Other)][loneliness] A qualitative and quantitative analysis of 79,951 ChatGPT conversations (WildChat dataset) is conducted to investigate how lonely users utilize LLM services. It is discovered that lonely users engage in much longer conversations (12 vs. 5 turns) and 37% seek advice or a listening ear. However, ChatGPT responds inappropriately in severe scenarios like suicidal ideation, and toxic content in lonely conversations reaches up to 55% (compared to…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "loneliness"
  - "human-AI interaction"
  - "ChatGPT"
  - "toxicity"
  - "mental health"
  - "social impact"
date: 2026-05-08
content_hash: 0a0953fc732532e1
---

# If Eleanor Rigby Had Met ChatGPT: A Study on Loneliness in a Post-LLM World

**Conference**: ACL 2025  
**arXiv**: [2412.01617](https://arxiv.org/abs/2412.01617)  
**Code**: [GitHub](https://github.com/adewynter/EleanorRigby)  
**Area**: LLM/NLP  
**Keywords**: loneliness, human-AI interaction, ChatGPT, toxicity, mental health, social impact

## TL;DR

A qualitative and quantitative analysis of 79,951 ChatGPT conversations (WildChat dataset) is conducted to investigate how lonely users utilize LLM services. It is discovered that lonely users engage in much longer conversations (12 vs. 5 turns) and 37% seek advice or a listening ear. However, ChatGPT responds inappropriately in severe scenarios like suicidal ideation, and toxic content in lonely conversations reaches up to 55% (compared to 20% in the main corpus), with females being targeted 22 times more often than males.

## Background & Motivation

**Background**: Loneliness is recognized by the WHO as a global public health issue, with a 2018 survey showing that approximately 20% of adults frequently feel lonely. Studies indicate that customized LLM chatbots (e.g., CareCall) can help alleviate loneliness under controlled environments, but these systems are supervised by professionals and deployed in controlled situations.

**Limitations of Prior Work**: In reality, users are more likely to seek companionship through free, easily accessible general-purpose services like ChatGPT, rather than professional, customized solutions. These services are positioned as "productivity tools" rather than mental health aids, lacking the necessary safety mechanisms and disclaimers.

**Key Challenge**: The anthropomorphic nature of LLM services naturally leads lonely users to treat them as confidants. However, the services themselves are neither professionally designed for mental health nor specially equipped to handle sensitive interactions (such as suicidal ideation or trauma). This "unintended large-scale use" presents severe risks.

**Goal**: To empirically study the actual usage patterns and potential risks of general-purpose LLM services in scenarios involving loneliness.

**Key Insight**: Starting from real WildChat conversation data, GPT-4o labeling and Reflexive Thematic Analysis (RTA) are used to distinguish lonely conversations from general ones, followed by a quantitative and qualitative analysis of their differences.

**Core Idea**: By analyzing real ChatGPT conversation data to reveal the usage patterns and risks of lonely users, it is shown that while companionship has potential, the safety mechanisms are severely inadequate, with toxic content and inappropriate responses being prominent issues.

## Method

### Overall Architecture

This paper is an empirical social study. The overall pipeline is: (1) randomly sample 79,951 ChatGPT conversations from the WildChat dataset; (2) use GPT-4o to classify them by interaction types (writing, coding, Q&A, chat, etc.); (3) extract non-task-oriented conversations (relevant corpus); (4) apply the taxonomy of Jiang et al. 2022 to perform loneliness assessment; (5) conduct quantitative statistics, Reflexive Thematic Analysis (RTA), and toxicity analysis specifically on the lonely conversations.

### Key Designs

1. **Multi-layer Corpus Filtering and Labeling**:

    - **Function**: Filter the subset of lonely conversations step-by-step from 79,951 dialogues.
    - **Mechanism**: Main corpus (79,951 dialogues) $\rightarrow$ remove task-oriented dialogues $\rightarrow$ relevant corpus $\rightarrow$ loneliness assessment $\rightarrow$ lonely corpus (approx. 8%). Using GPT-4o for automatic labeling, human statistical verification shows an accuracy of $86.4 \pm 4.7\%$ (intent) and $99.2 \pm 1.2\%$ (cause and target).
    - **Design Motivation**: Real-world data reflects the actual impact of LLM services better than laboratory data.

2. **Loneliness Assessment Taxonomy**:

    - **Function**: Assess characteristics of loneliness in conversations from multiple dimensions.
    - **Mechanism**: Adopting the classification framework designed by Jiang et al. 2022 based on the UCLA Loneliness Scale and DLS, which includes 5 dimensions: whether lonely (Yes/No), temporality (Transient/Enduring/Ambiguous), interaction type (seeking advice/offering help/seeking validation/reaching out/unfocused), context (social/physical/somatic/romantic), and interpersonal relationship (romantic/friendship/family/coworker).
    - **Design Motivation**: Loneliness is a multidimensional phenomenon; simple binary classification cannot capture its complexity.

3. **Qualitative Analysis: Reflexive Thematic Analysis (RTA)**:

    - **Function**: Perform an in-depth qualitative interpretation of lonely conversations.
    - **Mechanism**: Conduct Reflexive Thematic Analysis (Braun and Clarke 2006) on the top 500 overlapping conversations between the lonely corpus and relevant corpus. Semantic coding is applied as corpus labels to identify three main themes: seeking advice (37%), mental health (35% treating ChatGPT as a therapist), and toxic behavior (55%).
    - **Design Motivation**: Quantitative analysis cannot capture subtle differences in the quality of interactions; qualitative analysis reveals the appropriateness of user intentions and ChatGPT's responses.

4. **Toxicity Analysis**:

    - **Function**: Quantify and analyze patterns of toxic content in lonely conversations.
    - **Mechanism**: Label the types of toxic content (sexual, violent, racist, etc.) and target groups (female, male, minor) for each conversation. Toxic content accounts for 55% of lonely conversations (vs. 20% in the main corpus). Females are 22 times more likely to be targeted than males, and toxic content related to minors increases from 5% to 28%.
    - **Design Motivation**: Toxic behaviors may be amplified in the context of loneliness, requiring special attention.

## Key Experimental Results

### Main Results

| Metric | Main Corpus | Lonely Corpus |
|------|--------|---------|
| Dialog turns (average) | 5 | 12 |
| Toxic content ratio | 20% | 55% |
| Toxic content targeting females | 11% | 41% |
| Toxic content targeting minors | 5% | 28% |
| Toxic content targeting males | 14% | 7% |

### Characteristics of Lonely Conversations

| Characteristic | Percentage/Value | Explanation |
|------|----------|------|
| Seeking advice/listening | 37% | Lonely conversations excluding toxic content |
| Treating ChatGPT as a therapist | 35% | Among conversations seeking advice |
| Involving suicidal ideation | 5 cases | Only 1 provided a specific helpline number |
| Adversarial toxic dialogues | 40% | Non-roleplay parts within toxic content |
| Average length of adversarial dialogues | +3 turns (longest 67 turns) | vs. non-toxic lonely conversations |

### Key Findings
- ChatGPT performs acceptably in normal confiding scenarios—it can provide empathetic responses and advice (e.g., suggesting communication with family), but almost always recommends "consulting a psychotherapist."
- In severe scenarios (suicidal ideation, trauma), its response is severely inadequate—suggesting "self-care" or "outdoor activities," and only in 1 case was a suicide prevention helpline number provided.
- Behavioral guardrails are effective in standard dialogue (ChatGPT never generates toxic content on its own during normal chat), but are bypassed in role-playing/novel writing (26%).
- Adversarial users engage more persistently—potentially indicating that ChatGPT's "evasive" response strategy instead prolongs toxic interactions.
- One user expressed disappointment upon discovering that ChatGPT could not remember them, hinting at the risk of emotional dependency caused by anthropomorphism.

## Highlights & Insights
- The research perspective is uniquely distinct—instead of evaluating the "technical capabilities" of LLMs, it investigates the social consequences when they are "misused" by lonely groups as a general service, carrying significant ethical and policy implications.
- The four proposed recommendations (transparency standards, aligning emotional responses, studying real-world impact, legislative regulation) offer a concrete direction for AI governance. The paper cites real cases, such as a US teenager passing away due to a chatbot, to bolster its persuasiveness.

## Limitations & Future Work
- WildChat data comes from the HuggingFace API, which may not represent the entire user base of ChatGPT.
- Using GPT-4o for labeling carries potential biases (accuracy of 86.4%, with a 5% margin of error for label dependency).
- The analysis is restricted to dialogue text and cannot evaluate the real-world impact outside the conversations (such as subsequent user behavior).
- The boundary between the LLM and the service is blurred—conclusions depend on the entire service stack (UI, content moderation, etc.) rather than the LLM itself.

## Related Work & Insights
- **Ours vs. Jo et al. 2023 (CareCall)**: Evaluates the effect of LLMs on mitigating elderly loneliness in a controlled environment ($n=34$), whereas this paper studies the risks of uncontrolled use "in the wild."
- **Ours vs. Zhao et al. 2024 (WildChat)**: This paper adds a loneliness dimension analysis based on the WildChat data, revealing social risks not covered by the original paper.
- **Ours vs. Jakesch et al. 2023**: Studies the impact of LLMs on changing user opinions, whereas this paper observes similar echo-chamber and polarization risks within the context of loneliness.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Investigates LLM usage from a social impact perspective, filling an important research gap between general-purpose LLM services and mental health.
- Experimental Thoroughness: ⭐⭐⭐⭐ Large-scale analysis of nearly 80,000 conversations combined with deep qualitative analysis, but lacks longitudinal tracking and causal inference.
- Writing Quality: ⭐⭐⭐⭐⭐ Handles sensitive topics appropriately, with a vivid yet rigorous narrative, and correct anonymization of cases.
- Value: ⭐⭐⭐⭐⭐ Directly instructive for AI ethics, LLM deployment strategies, and public policy, representing significant social value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Expert Evaluation of LLM World Models: A High-Tc Superconductivity Case Study](../../ICML2025/llm_nlp/expert_evaluation_of_llm_world_models_a_high-t_c_superconductivity_case_study.md)
- [\[ACL 2025\] A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)
- [\[ACL 2025\] Can LLMs Interpret and Leverage Structured Linguistic Representations? A Case Study with AMRs](can_llms_interpret_and_leverage_structured_linguistic_representations_a_case_stu.md)
- [\[ACL 2025\] Mind the (Belief) Gap: Group Identity in the World of LLMs](mind_the_belief_gap_group_identity_in_the_world_of_llms.md)
- [\[NeurIPS 2025\] Q♯: Provably Optimal Distributional RL for LLM Post-Training](../../NeurIPS2025/llm_nlp/qsharp_provably_optimal_distributional_rl_for_llm_post-training.md)

</div>

<!-- RELATED:END -->
