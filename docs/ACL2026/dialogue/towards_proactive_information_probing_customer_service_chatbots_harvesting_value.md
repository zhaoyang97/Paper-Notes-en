---
title: >-
  [Paper Note] Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation
description: >-
  [ACL 2026][Dialogue Systems][Proactive Information Probing] This paper proposes the ProChatIP framework, which transforms customer service chatbots from passive answering tools into proactive information harvesting engin…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Proactive Information Probing"
  - "Customer Service Chatbots"
  - "Dialogue Strategy"
  - "Reinforcement Learning"
  - "Business Intelligence"
date: 2026-05-08
content_hash: 5092562a350f31d7
---

# Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation

**Conference**: ACL 2026  
**arXiv**: [2604.11077](https://arxiv.org/abs/2604.11077)  
**Code**: [https://github.com/SCUNLP/PROCHATIP](https://github.com/SCUNLP/PROCHATIP)  
**Area**: Dialogue Systems  
**Keywords**: Proactive Information Probing, Customer Service Chatbots, Dialogue Strategy, Reinforcement Learning, Business Intelligence

## TL;DR

This paper proposes the ProChatIP framework, which transforms customer service chatbots from passive answering tools into proactive information harvesting engines. Through a specialized dialogue strategy module, the system learns "when to probe" the user for preset target information while minimizing dialogue turns and user friction.

## Background & Motivation

**Background**: AI customer service chatbots have become an integral part of modern business operations, primarily focusing on answering user queries, understanding user intent, and extracting Q&A pairs. Chatbots in the LLM era can effectively handle complex interaction dynamics.

**Limitations of Prior Work**: Existing customer service bots are inherently passive—they only respond to user questions and lack the ability to proactively collect valuable business information. For instance, a chatbot on a commodity trading platform could proactively ask about market trends during service ("Have you been monitoring inventory levels recently?"), thereby obtaining market intelligence for pricing models via crowdsourcing.

**Key Challenge**: Users enter customer service dialogues to solve their own problems. If the bot's information probing appears abrupt or context-irrelevant, users are likely to refuse or ignore it, potentially reducing satisfaction. Excessive repetitive questioning can prolong dialogues and damage the user experience.

**Goal**: Define the "Proactive Information Probing" task and optimize probing timing ("when to probe") to maximize information acquisition while minimizing dialogue turns and user friction.

**Key Insight**: Decouple the information probing decision into a separate Chat Strategy (CS) module, isolated from the LLM backbone, and train the strategy using a two-stage curriculum learning approach with SFT and RL.

**Core Idea**: In each dialogue turn, a lightweight strategy module determines whether to probe ($d_t \in \{0, 1\}$). If a probe is decided, the LLM naturally incorporates the probing question into its response. The strategy is trained with dual rewards (Proactive Probe Reward + Passive Wait Reward) to learn to act when user receptivity is highest.

## Method

### Overall Architecture

ProChatIP consists of two components: (1) A dialogue strategy module CS (implemented with Qwen3-0.6B) that analyzes dialogue history and target information to output a binary decision $d_t$; (2) An LLM backbone (response generation) that generates a reply based on history, target info, and the strategy decision, integrating a probe if $d_t=1$. The CS module is trained via two stages: SFT and RL.

### Key Designs

1.  **SFT Stage—Acquisition of Basic Probing Capabilities**:
    *   **Function**: Enables the strategy module to learn basic patterns of when it is or isn't appropriate to probe.
    *   **Mechanism**: Uses an LLM to generate a rule-based synthetic customer service dataset where each dialogue corresponds to a clear rule (e.g., "probe when the user shows willingness to expand the topic" or "do not probe when the user explicitly refuses"). Full-parameter fine-tuning is performed on this labeled data to establish baseline probing capabilities.
    *   **Design Motivation**: Existing customer service datasets lack proactive probing samples, requiring rule-based synthesis for bootstrapping. SFT provides initial heuristic pattern recognition.

2.  **RL Stage—Dual Reward Strategy Optimization**:
    *   **Function**: Learns optimal probing timing from dynamic interactions.
    *   **Mechanism**: Uses an LLM to build a user simulator (with randomized resistance behavior instructions) for multi-turn interactions with ProChatIP. Dual rewards are designed: (a) Proactive Probe Reward—successful info acquisition (large positive), refusal (penalty), and ignore (small penalty); (b) Passive Wait Reward—choosing to wait after a previous refusal (large positive, "smart retreat"), waiting after being ignored (small positive, "pause and retry"), and continuous inaction (large penalty to prevent "lazy strategy"). Optimization is performed using the REINFORCE algorithm.
    *   **Design Motivation**: Only using probing rewards leads to over-aggressive bots. Adding wait rewards teaches the strategy the value of "stepping back"—retreating after refusal is more effective than persistent questioning.

3.  **Strategy-Generation Decoupled Architecture**:
    *   **Function**: Adds probing capabilities while maintaining service quality.
    *   **Mechanism**: Strategy decisions are made by an independent lightweight module (Qwen3-0.6B), while the LLM backbone is only responsible for execution—integrating probes naturally when receiving $d_t=1$ and answering normally when $d_t=0$. The functions are completely decoupled.
    *   **Design Motivation**: If the same LLM handles both decision-making and generation, probing intent might interfere with service quality. Decoupling ensures the core customer service task remains unaffected.

### Loss & Training

SFT uses cross-entropy loss to train the binary classification of probing decisions. RL uses the REINFORCE algorithm ($$\theta \leftarrow \theta + \alpha \nabla_\theta \log \text{CS}_\theta(d_t | h_t, \mathcal{I}) \cdot G_t$$) to maximize cumulative discounted rewards.

## Key Experimental Results

### Main Results

| Method | TSR↑ | AvgT↓ | RPR↓ | QRR↑ |
| :--- | :--- | :--- | :--- | :--- |
| Vanilla (GPT-4o-mini) | 0.00% | 7.57 | 0.00% | 99.65% |
| Proactive Baseline | 85.79% | 3.09 | 50.33% | 99.41% |
| ICL-AIF Baseline | 64.42% | 4.10 | 65.08% | 99.37% |
| **ProChatIP** | **87.38%** | **1.92** | **24.59%** | **98.67%** |

*TSR=Target Success Rate, AvgT=Average Dialogue Turns, RPR=Refusal Rate, QRR=Query Response Rate*

### Ablation Study

| Configuration | TSR | AvgT | Description |
| :--- | :--- | :--- | :--- |
| SFT Only | Mid | Mid | Basic pattern recognition |
| SFT + RL | 87.38% | 1.92 | RL significantly optimizes timing |
| w/o Wait Reward | Down | Up | No retreat strategy; refusal rate rises |

### Key Findings

*   **ProChatIP significantly outperforms baselines**: Compared to the strongest baseline, target information success rate increased by 11.36%, dialogue turns decreased by 39.50%, and user satisfaction increased by 23.73%.
*   **Criticality of Wait Reward**: "Smart retreat" (choosing to wait and re-probe after refusal) is key to strategy success—purely aggressive probing leads to high refusal rates.
*   **Human Evaluation Consistency**: Results from LLM simulators and human participants are highly consistent, validating the effectiveness of the simulator-based training environment.
*   **Effective Decoupled Architecture**: The strategy module uses only 0.6B parameters without compromising the LLM backbone's quality (QRR remains >98%).

## Highlights & Insights

*   **Paradigm Shift in Business Value**: Transforms customer service interactions from a "cost center" to a "profit center"—making every dialogue a low-cost channel for business intelligence collection.
*   **Dual Reward Design**: The mechanism of proactive probe + passive wait rewards cleverly encodes strategic wisdom ("knowing when to advance and retreat"), avoiding a singular aggressive strategy.
*   **Deployment Friendly**: The strategy module is lightweight (0.6B) and compatible with any LLM backbone, allowing for plug-and-play deployment in existing systems.

## Limitations & Future Work

*   Currently supports only a single target information probe; multi-target prioritization is not handled.
*   The simulator's resistance behavior is rule-generated and may not fully cover the complex psychology of real users.
*   The quality of probing questions depends entirely on the LLM backbone and lacks targeted optimization.
*   Evaluations were limited to Finance and General domains; broader applicability remains to be verified.

## Related Work & Insights

*   **vs. Proactive Dialogue Agents**: Existing proactive systems (e.g., clarifying ambiguity, recommendation) serve the user's interest. ProChatIP introduces a "system-centric utility" dimension—leveraging proactivity for business intelligence, creating a multi-objective challenge.
*   **vs. Traditional Customer Service**: Traditional service focuses on query resolution and empathy. ProChatIP adds information harvesting without damaging these dimensions.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ (First to define "Proactive Information Probing" with clear business value)
*   Experimental Thoroughness: ⭐⭐⭐⭐ (Dual verification with simulators and humans, multiple baselines)
*   Writing Quality: ⭐⭐⭐⭐ (Clear motivation and complete methodological description)
*   Value: ⭐⭐⭐⭐⭐ (Directly alters the business positioning of chatbots with high practical value)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ACL 2026\] Frame of Reference: Addressing the Challenges of Common Ground Representation in Dialogue](frame_of_reference_addressing_the_challenges_of_common_ground_representation_in_.md)
- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[ACL 2026\] Template-assisted Contrastive Learning of Task-oriented Dialogue Sentence Embeddings](template-assisted_contrastive_learning_of_task-oriented_dialogue_sentence_embedd.md)
- [\[ACL 2026\] Reasoning Gets Harder for LLMs Inside A Dialogue](reasoning_gets_harder_for_llms_inside_a_dialogue.md)

</div>

<!-- RELATED:END -->
