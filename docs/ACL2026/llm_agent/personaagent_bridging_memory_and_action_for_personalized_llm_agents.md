---
title: >-
  [Paper Note] PersonaAgent: Bridging Memory and Action for Personalized LLM Agents
description: >-
  [ACL2026][LLM Agent][Personalized Agent] PersonaAgent connects user history and tool actions using "personalized memory + personalized actions + test-time optimizable persona prompts…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "Personalized Agent"
  - "Long-term Memory"
  - "Persona Prompt"
  - "Test-time Alignment"
  - "LaMP"
date: 2026-05-08
content_hash: 519905693cf13b99
---

# PersonaAgent: Bridging Memory and Action for Personalized LLM Agents

**Conference**: ACL2026  
**arXiv**: [2506.06254](https://arxiv.org/abs/2506.06254)  
**Code**: Code not explicitly released in the paper  
**Area**: LLM Agent / Personalized Agents  
**Keywords**: Personalized Agent, Long-term Memory, Persona Prompt, Test-time Alignment, LaMP

## TL;DR
PersonaAgent connects user history and tool actions using "personalized memory + personalized actions + test-time optimizable persona prompts," significantly outperforming baselines such as RAG, PAG, ReAct, and MemBank on multiple LaMP personalized decision-making tasks.

## Background & Motivation
**Background**: LLM agents are capable of tool calling, maintaining memory, and multi-step reasoning, but most agents remain biased toward general task execution. Personalization is commonly seen in user profiling, retrieval augmentation, or user-specific fine-tuning, but these methods typically utilize personal information only during the text generation phase.

**Limitations of Prior Work**: General agent action spaces do not change with the user, often leading to "one-size-fits-all" strategies. User-specific fine-tuning is difficult to scale for large numbers of users and frequent updates. While fixed RAG/PAG workflows can read user data, they lack agentic decision-making capabilities and cannot continuously adjust tool calling and behavioral strategies.

**Key Challenge**: True personal intelligence needs to simultaneously satisfy agentic intelligence, real-world deployment usability, personal data utilization, and real-time preference alignment. Existing methods often cover only one or two of these aspects. Personalization should not only occur in the final response text but also influence which tools the agent selects, what memories it retrieves, and how it interprets the current task.

**Goal**: The authors aim to establish a unified framework that allows LLM agents to read user history, abstract long-term preferences, call personalized tools, and dynamically update the user persona based on recent interactions during test-time.

**Key Insight**: The paper defines a persona as a unique system prompt for each user. It is not a static profile but a mediator between memory and action modules: memory provides evidence for the persona, the persona controls actions, and action results in turn update both memory and persona.

**Core Idea**: Use a persona prompt as the central controller of the personalized agent and optimize this controller at test-time via textual feedback from recent interactions.

## Method
The design of PersonaAgent can be understood as adding a "user-level operating system" on top of a general LLM agent. While an ordinary agent selects tools based on task context, PersonaAgent first compresses user history into an actionable persona, let the persona influence tool selection, memory retrieval, reasoning paths, and final decisions.

### Overall Architecture
The framework consists of two complementary modules and a mediator. The personalized memory module stores user interactions, divided into episodic memory and semantic memory. The personalized action module adjusts tool calling and behavioral strategies based on the persona. The persona prompt transforms user evidence from memory into behavioral constraints usable at each step by the agent.

When a new query arrives, the system first retrieves similar history from episodic memory and combines it with the stable user profile in semantic memory to form the context. The agent then selects actions under persona modulation—such as using external knowledge, retrieving personalized history, updating memory, or performing persona-guided reasoning. The test-time alignment module simulates recent user interactions, compares the difference between agent responses and ground-truth user responses, and uses an LLM to generate a "textual gradient" to update the persona.

### Key Designs
1. **Dual-layer Personalized Memory**:

	- **Function**: Simultaneously maintains fine-grained historical events and long-term stable user preferences.
	- **Mechanism**: Episodic memory stores $(q_i,r_i^{gt},m_i)$ for each user, with new queries retrieving Top-K history via embedding similarity. Semantic memory uses a summarization prompt to abstract the event set into a user profile $P^u=f_s(S_t,D^u)$.
	- **Design Motivation**: Relying solely on event retrieval introduces noise and long contexts, while using only a profile loses specific behavioral evidence. Dual-layer memory allows the agent to see specific precedents while adhering to long-term preferences.

2. **Persona-controlled Personalized Action Space**:

	- **Function**: Shifts personalization forward from "response content" to "action selection."
	- **Mechanism**: General agents select actions from a universal set $A$; PersonaAgent expands this to $\hat{A}=A\cup D$, where $D$ includes user data and historical access tools. The action policy is denoted as $a_t\sim\pi_P(\cdot|c_t)$, modulated by persona $P$.
	- **Design Motivation**: For many personalized tasks, "speaking like the user" is insufficient; the agent must know when to retrieve personal history, when to rely on external knowledge, and when to let historical preferences override general judgments.

3. **Test-time Preference Alignment**:

	- **Function**: Allows the persona to evolve in real-time with recent user behavior rather than being fixed in the initialization prompt.
	- **Mechanism**: Given a recent batch $D_{batch}=\{(q_j,\hat{r}_j,r_j^{gt})\}$, the system lets an LLM generate textual loss feedback based on the difference between simulated and real responses, followed by an `LLM_update` to rewrite the persona. This optimization is executed asynchronously without affecting real-time response latency.
	- **Design Motivation**: User preferences change and cannot always be captured by a one-time profile summary. Test-time textual optimization avoids retraining models for every user while retaining individual-level adaptation.

### Loss & Training
PersonaAgent does not rely on user-level model fine-tuning; instead, it performs test-time optimization via prompts and textual feedback. The paper formalizes persona optimization as $$P^*=\arg\min_P\sum_j L(\hat{r}_j,r_j^{gt}|q_j)$$, but the actual gradient is represented in natural language feedback by `LLM_grad` and rewritten by `LLM_update`. Claude-3.5 Sonnet is used as the unified execution model in experiments, maintaining consistent input/output formats to isolate gains from the framework design.

## Key Experimental Results

### Main Results
| Task | Metric | Strong Baseline | PersonaAgent | Gain |
|------|------|--------|--------------|------|
| LaMP-1 Citation Identification | Acc / F1 | MemBank 0.862 / 0.861 | 0.919 / 0.918 | Significant improvement in citation personalization |
| LaMP-2M Movie Tagging | Acc / F1 | MemBank 0.470 / 0.391 | 0.513 / 0.424 | Better capture of user movie preferences |
| LaMP-2N News Categorization | Acc / F1 | PAG 0.768 / 0.509 | 0.796 / 0.532 | Profile + action integration > fixed workflows |
| LaMP-3 Product Rating | MAE / RMSE | ICL 0.277 / 0.543 | 0.241 / 0.509 | Lowest numerical rating error |

### Ablation Study
| Configuration | LaMP-1 Acc/F1 | LaMP-2M Acc/F1 | LaMP-2N Acc/F1 | LaMP-3 MAE/RMSE | Description |
|------|---------------|----------------|----------------|-----------------|------|
| Full PersonaAgent | 0.919 / 0.918 | 0.513 / 0.424 | 0.796 / 0.532 | 0.241 / 0.509 | Full System |
| w/o alignment | 0.894 / 0.893 | 0.487 / 0.403 | 0.775 / 0.502 | 0.259 / 0.560 | Overall decline without test-time alignment |
| w/o persona | 0.846 / 0.855 | 0.463 / 0.361 | 0.769 / 0.483 | 0.277 / 0.542 | Persona mediator is key to memory-action bridge |
| w/o Memory | 0.821 / 0.841 | 0.460 / 0.365 | 0.646 / 0.388 | 0.348 / 0.661 | Missing user context hurts performance |
| w/o Action | 0.764 / 0.789 | 0.403 / 0.329 | 0.626 / 0.375 | 0.375 / 0.756 | Reasoning alone is insufficient; actions are critical |

### Key Findings
- PersonaAgent is the top performer across four decision-making tasks. Specifically, LaMP-1 Acc increased from MemBank's 0.862 to 0.919, indicating that persona-guided memory/action is highly effective for topic-level user interests.
- Ablations show the action module has the greatest impact; removing it causes LaMP-3 MAE to degrade from 0.241 to 0.375. This suggests personalized tool actions are more important than simply stuffing user profiles into prompts.
- Test-time scaling yields gains: increasing the alignment batch size, adding a few alignment iterations, or retrieving more memory entries enhances personalization on LaMP-2M, though gains plateau after ~3 iterations.
- Efficiency analysis shows PersonaAgent averages 1.79s per sample, slower than PAG (1.24s) but significantly faster than ReAct (2.61s) and MemBank (2.92s). The authors emphasize that persona optimization is asynchronous.
- Cold-start experiments show that even when limiting each user to 10 historical interactions, PersonaAgent remains optimal across four LaMP tasks (e.g., LaMP-1 Acc 0.845, LaMP-3 MAE 0.301).

## Highlights & Insights
- The most valuable contribution of the paper is elevating the persona from "text describing the user" to a "strategy mediator controlling agent actions." This ensures personalization permeates retrieval, tool selection, and reasoning paths.
- Test-time textual gradients are well-suited for personalization. They require no parameter training for each user and do not demand explicit preference writing; personas can be iteratively rewritten as long as recent interaction ground truth exists.
- The closed-loop design of memory and action is intuitive. Action results update memory, memory rewrites the persona, and the persona controls the next round of actions, which is closer to a long-term personal assistant than a fixed RAG pipeline.
- Ablation results provide a clear signal: for personalized agents, simply adding memory is insufficient; memory must influence the action policy.

## Limitations & Future Work
- Authors acknowledge that textual feedback may overlook implicit or multimodal user signals such as emotion, visual preferences, or dwell time. Future work could incorporate clicks, voice, or physiological feedback.
- Frequent use of personalized data for memory retrieval and persona optimization poses privacy risks. While the paper mentions exploring privacy-preserving mechanisms like federated learning, these are not yet implemented.
- Experiments focused on LaMP tasks; user preference drift, malicious feedback, data expiration, and cross-device synchronization in real-world long-term deployments have not been systematically evaluated.
- Automatic persona updates might accumulate errors. If ground truth for an interaction is noisy, textual gradients may push the persona toward incorrect preferences, necessitating robust update and rollback mechanisms.

## Related Work & Insights
- **vs RAG / PAG**: RAG retrieves history and PAG uses profiles, but both are often fixed workflows. PersonaAgent lets the persona modulate the action policy, deciding when, what, and how to use evidence.
- **vs ReAct**: ReAct possesses tool-use and reasoning capabilities but lacks user-level alignment. PersonaAgent adds personal memory and persona control onto an agentic loop.
- **vs MemBank**: MemBank emphasizes long-term memory but lacks strong personalized action control. PersonaAgent's ablations show that while memory is vital, the action module and persona bridge are the performance cores.
- **vs User-specific Fine-tuning**: Fine-tuning provides individual alignment but at high maintenance and compute costs. PersonaAgent avoids parameter updates in large-scale scenarios via test-time prompt optimization.

## Rating
- **Novelty**: ⭐⭐⭐⭐ High conceptual integration of memory, action, and persona prompts into an optimizable personalized agent framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes main experiments, ablations, persona analysis, scaling, model variations, and cold-starts; real-world user studies are still needed.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation and comprehensive tables; some algorithmic descriptions lean toward prompt engineering and could be more specific.
- **Value**: ⭐⭐⭐⭐⭐ Highly insightful for building personal assistants, recommendation agents, and long-term user interaction systems, specifically the persona-as-controller design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)
- [\[ACL 2026\] ProPer Agents: Proactivity Driven Personalized Agents for Advancing Knowledge Gap Navigation](proper_agents_proactivity_driven_personalized_agents_for_advancing_knowledge_gap.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](../../ICLR2026/llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ACL 2026\] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](shopping_companion_a_memory-augmented_llm_agent_for_real-world_e-commerce_tasks.md)
- [\[ACL 2026\] RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents](recmem_recurrence-based_memory_consolidation_for_efficient_and_effective_long-ru.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2026\] ProPer Agents: Proactivity Driven Personalized Agents for Advancing Knowledge Gap Navigation](proper_agents_proactivity_driven_personalized_agents_for_advancing_knowledge_gap.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)
- [\[ICLR 2026\] FingerTip 20K: A Benchmark for Proactive and Personalized Mobile LLM Agents](../../ICLR2026/llm_agent/fingertip_20k_a_benchmark_for_proactive_and_personalized_mobile_llm_agents.md)
- [\[ACL 2026\] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](shopping_companion_a_memory-augmented_llm_agent_for_real-world_e-commerce_tasks.md)
- [\[ACL 2026\] RecMem: Recurrence-based Memory Consolidation for Efficient and Effective Long-Running LLM Agents](recmem_recurrence-based_memory_consolidation_for_efficient_and_effective_long-ru.md)

</div>

<!-- RELATED:END -->
