---
title: >-
  [Paper Note] Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation
description: >-
  [ACL 2026][Dialogue Systems][Proactive Information Probing] This paper proposes ProChatIP, a framework that transforms customer service chatbots from passive response tools into proactive information harvesting engines. A dedicated dialogue policy module learns *when to probe* users for preset target information while minimizing conversation turns and user friction.
tags:
  - ACL 2026
  - Dialogue Systems
  - Proactive Information Probing
  - Customer Service Chatbot
  - Dialogue Policy
  - Reinforcement Learning
  - Business Intelligence
date: 2026-05-08
content_hash: 42be1748cdabe105
---

# Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation

**Conference**: ACL 2026
**arXiv**: [2604.11077](https://arxiv.org/abs/2604.11077)
**Code**: [https://github.com/SCUNLP/PROCHATIP](https://github.com/SCUNLP/PROCHATIP)
**Area**: Dialogue Systems
**Keywords**: Proactive Information Probing, Customer Service Chatbot, Dialogue Policy, Reinforcement Learning, Business Intelligence

## TL;DR

This paper proposes ProChatIP, a framework that transforms customer service chatbots from passive response tools into proactive information harvesting engines. A dedicated dialogue policy module learns *when to probe* users for preset target information while minimizing conversation turns and user friction.

## Background & Motivation

**Background**: AI-powered customer service chatbots have become a critical component of modern business operations, primarily focusing on answering user queries, understanding user intent, and extracting QA pairs. In the LLM era, customer service bots can effectively handle complex interaction dynamics.

**Limitations of Prior Work**: Existing customer service chatbots are inherently passive—they only respond to user inquiries and lack the ability to proactively collect valuable business information. For instance, a customer service agent on a commodity trading platform could proactively ask users for their views on market conditions (e.g., "Have you been following inventory levels recently?"), thereby crowd-sourcing market intelligence needed for pricing models.

**Key Challenge**: Users enter customer service conversations to resolve their own issues. If the bot's information probing appears abrupt or contextually irrelevant, users are likely to refuse or ignore it, potentially reducing satisfaction. Excessive or repetitive inquiries also prolong conversations and degrade the user experience.

**Goal**: Define the task of "proactive information probing," optimize the timing of probing (i.e., *when* to probe), and maximize information acquisition while minimizing conversation turns and user friction.

**Key Insight**: Decouple the probing decision into an independent policy module (CS) separate from the LLM backbone, and train the policy using a two-stage curriculum of SFT followed by RL.

**Core Idea**: At each conversation turn, a lightweight policy module determines whether to probe ($d_t \in \{0,1\}$). If probing is chosen, the LLM naturally incorporates the probing question into its response. A dual reward signal—proactive probing reward and passive waiting reward—trains the policy to act at the moment of highest user receptivity.

## Method

### Overall Architecture

ProChatIP consists of two components: (1) a Dialogue Policy Module CS (implemented with Qwen3-0.6B) that analyzes conversation history and target information to output a binary decision $d_t$; and (2) an LLM backbone for customer service response generation that produces replies conditioned on history, target information, and the policy decision—embedding a probing question when $d_t=1$ and responding normally when $d_t=0$. The CS module is trained via two-stage SFT + RL.

### Key Designs

1. **SFT Stage — Acquiring Baseline Probing Capability**:

    - **Function**: Teach the policy module to recognize basic patterns of when probing is appropriate or inappropriate.
    - **Mechanism**: An LLM is used to generate a rule-based synthetic customer service dataset, where each dialogue corresponds to an explicit rule (e.g., "probe when the user shows willingness to expand the topic" or "do not probe when the user explicitly refuses"). Full-parameter fine-tuning on these annotated samples establishes a baseline probing capability.
    - **Design Motivation**: Existing customer service datasets contain no proactive probing examples; bootstrapping via rule-based synthesis is necessary. SFT provides initial heuristic pattern recognition.

2. **RL Stage — Dual-Reward Policy Optimization**:

    - **Function**: Learn optimal probing timing through dynamic interaction.
    - **Mechanism**: An LLM-based user simulator (with randomized resistance behavior instructions) is constructed to interact with ProChatIP over multiple turns. A dual reward is designed: (a) *Proactive probing reward*—large positive reward for successfully obtaining information, penalty for being refused, small penalty for being ignored; (b) *Passive waiting reward*—large positive reward for waiting after a prior refusal ("smart retreat"), small positive reward for waiting after being ignored ("pause and retry"), large penalty for consecutive inaction (to prevent a "lazy policy"). The REINFORCE algorithm is used for optimization.
    - **Design Motivation**: A probing-only reward causes the bot to behave too aggressively. The waiting reward teaches the policy the value of "stepping back"—retreating after a refusal is more effective than continued pressure.

3. **Policy-Generation Decoupled Architecture**:

    - **Function**: Add probing capability while preserving LLM customer service quality.
    - **Mechanism**: The probing decision is made by an independent lightweight module (Qwen3-0.6B), while the LLM backbone is solely responsible for execution—naturally embedding a probing question upon receiving $d_t=1$ or answering normally upon $d_t=0$. The two components are fully decoupled.
    - **Design Motivation**: Having a single LLM handle both decision-making and generation risks probing intent interfering with service quality. Decoupling ensures the core customer service function is unaffected.

### Loss & Training

SFT applies cross-entropy loss for binary classification of the probing decision. RL applies the REINFORCE algorithm ($\theta \leftarrow \theta + \alpha \nabla_\theta \log \text{CS}_\theta(d_t | h_t, \mathcal{I}) \cdot G_t$) to maximize cumulative discounted reward.

## Key Experimental Results

### Main Results

| Method | TSR↑ | AvgT↓ | RPR↓ | QRR↑ |
|--------|------|-------|------|------|
| Vanilla (GPT-4o-mini) | 0.00% | 7.57 | 0.00% | 99.65% |
| Proactive Baseline | 85.79% | 3.09 | 50.33% | 99.41% |
| ICL-AIF Baseline | 64.42% | 4.10 | 65.08% | 99.37% |
| ProChatIP | 87.38% | 1.92 | 24.59% | 98.67% |

TSR = Target information Success Rate, AvgT = Average Turns, RPR = Refusal Rate, QRR = Query Reply Rate

### Ablation Study

| Configuration | TSR | AvgT | Notes |
|---------------|-----|------|-------|
| SFT only | Moderate | Moderate | Basic pattern recognition |
| SFT + RL | 87.38% | 1.92 | RL substantially improves timing |
| w/o Waiting Reward | Decreases | Increases | Without retreat strategy, refusal rate rises |

### Key Findings

- **ProChatIP substantially outperforms baselines**: Compared to the strongest baseline, it achieves an 11.36% improvement in target information success rate, a 39.50% reduction in conversation turns, and a 23.73% improvement in user satisfaction.
- **Critical role of the waiting reward**: The "smart retreat" behavior—waiting and re-probing after a refusal—is key to policy success; purely aggressive probing leads to high refusal rates.
- **Consistency with human evaluation**: Results from LLM-based simulators and human participants are highly consistent, validating the effectiveness of the simulated training environment.
- **Effectiveness of decoupled architecture**: The policy module requires only 0.6B parameters and does not compromise the LLM backbone's customer service quality (QRR remains >98%).

## Highlights & Insights

- **A paradigm shift in business value**: The work repositions customer service interactions from a "cost center" to a "profit center"—each service conversation becomes a low-cost channel for business intelligence collection.
- **Dual-reward design**: The combination of proactive probing and passive waiting rewards elegantly encodes the strategic wisdom of "knowing when to advance and when to retreat," avoiding purely aggressive policies.
- **Deployment-friendly**: The lightweight policy module (0.6B) is compatible with any LLM backbone and can be plug-and-play integrated into existing customer service systems.

## Limitations & Future Work

- Currently supports probing for only a single target information item; multi-target priority ranking is not addressed.
- Resistance behaviors in the user simulator are rule-generated and may not fully capture the complex psychology of real users.
- The quality of generated probing questions depends entirely on the LLM backbone, with no targeted optimization.
- Evaluation is limited to financial and general customer service domains; broader domain applicability remains to be verified.

## Related Work & Insights

- **vs. Proactive Dialogue Agents**: Existing proactive dialogue systems (e.g., for ambiguity clarification or recommendation steering) direct proactivity toward user benefit. ProChatIP introduces a "system-centric utility" dimension—leveraging proactivity to collect business intelligence—creating a multi-objective challenge.
- **vs. Traditional Customer Service**: Traditional customer service focuses on user experience dimensions such as query resolution and empathetic engagement. ProChatIP augments these with information harvesting without degrading them.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First to define the "proactive information probing" task; the problem formulation is novel and carries direct commercial value.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual validation via simulator and human participants, with multiple baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clearly articulated; method description is complete.
- Value: ⭐⭐⭐⭐⭐ Directly reshapes the commercial positioning of customer service bots; practical value is exceptionally high.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Agentic Conversational Search with Contextualized Reasoning via Reinforcement Learning](agentic_conversational_search_with_contextualized_reasoning_via_reinforcement_le.md)
- [\[ACL 2026\] APEX-MEM: Agentic Semi-Structured Memory with Temporal Reasoning for Long-Term Conversational AI](apex-mem_agentic_semi-structured_memory_with_temporal_reasoning_for_long-term_co.md)
- [\[ACL 2026\] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review](author-in-the-loop_response_generation_and_evaluation_integrating_author_experti.md)
- [\[ACL 2026\] VoxMind: An End-to-End Agentic Spoken Dialogue System](voxmind_an_end-to-end_agentic_spoken_dialogue_system.md)
- [\[ACL 2026\] Disambiguation-Centric Finetuning Makes Enterprise Tool-Calling LLMs More Realistic and Less Risky](disambiguation-centric_finetuning_makes_enterprise_tool-calling_llms_more_realis.md)

<!-- RELATED:END -->
