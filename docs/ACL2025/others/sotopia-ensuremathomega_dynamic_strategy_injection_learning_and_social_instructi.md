---
title: >-
  [Paper Note] SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents
description: >-
  [ACL 2025 (Main)][Social Agents] This paper proposes the SOTOPIA-Ω framework, which dynamically injects multi-step reasoning strategies and direct strategies from negotiation theory into expert agents to automatically construct high-quality social dialogue training corpora. It defines a new concept "Social Instruction Following (S-IF)" along with two evaluation metrics, enabling a 7B model to outperform GPT-4 expert agents in social goal achievement.
tags:
  - "ACL 2025 (Main)"
  - "Social Agents"
  - "Strategy Injection"
  - "Social Instruction Following"
  - "Dialogue Data Construction"
  - "Negotiation Theory"
date: 2026-05-08
content_hash: 5e9695f2ce979063
---

# SOTOPIA-Ω: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents

**Conference**: ACL 2025 (Main)  
**arXiv**: [2502.15538](https://arxiv.org/abs/2502.15538)  
**Code**: None  
**Area**: Others  
**Keywords**: Social Agents, Strategy Injection, Social Instruction Following, Dialogue Data Construction, Negotiation Theory

## TL;DR

This paper proposes the SOTOPIA-Ω framework, which dynamically injects multi-step reasoning strategies and direct strategies from negotiation theory into expert agents to automatically construct high-quality social dialogue training corpora. It defines a new concept "Social Instruction Following (S-IF)" along with two evaluation metrics, enabling a 7B model to outperform GPT-4 expert agents in social goal achievement.

## Background & Motivation

**Background**: LLM-driven social agents represent an important direction in current AI research, requiring models to engage in natural, strategic interactions with humans or other agents in open-ended social scenarios. Benchmark platforms like SOTOPIA provide standardized environments for evaluating social agents across various social contexts, such as negotiation, persuasion, and information exchange. Existing works primarily rely on direct prompting of strong models like GPT-4 to achieve social interactions.

**Limitations of Prior Work**: While humans naturally employ various strategies in social interactions (e.g., concession, pressure, empathy, and information exchange), existing social agents lack explicit learning and application of these strategies. When deadlocks occur (where both parties repeatedly state their positions without progress), agents without strategic guidance often fail to break the impasse. Furthermore, constructing high-quality social dialogue training data remains expensive.

**Key Challenge**: Human social strategies are rich and effective, but transferring them to AI agents faces two major challenges: (1) when to use and how to combine these strategies is highly dynamic and cannot be simply hard-coded; (2) there is a lack of large-scale social dialogue corpora labeled with strategy usage for supervised learning.

**Goal**: (1) To design a method that automatically injects human social strategies into LLM agents and generates high-quality training corpora; (2) to define the concept of "Social Instruction Following" (S-IF) to supplement blind spots in existing social capability evaluations; (3) to train a 7B-scale model to achieve or even surpass the social performance of GPT-4.

**Key Insight**: Grounded in negotiation theory, this work systematizes common social strategies into multi-step reasoning strategies (complex strategies that unfold step-by-step, such as interest-based negotiation and problem diagnostic methods) and direct strategies (simple strategies like direct concession or holding positions). It then designs a mechanism to dynamically select and inject these strategies before each turn of dialogue.

**Core Idea**: Treat social strategies as "injectable prompt modules" that are dynamically selected via multi-step reasoning to guide the agent's next action before generating dialogue turns, thereby automatically constructing a high-quality strategy-guided dialogue corpus.

## Method

### Overall Architecture

The workflow of SOTOPIA-Ω is divided into three stages: (1) Strategy Design: extracting and formalizing multi-step reasoning strategies and direct strategies from negotiation theory and social psychology; (2) Dynamic Strategy Injection: having agents analyze the current dialogue state to select the most appropriate strategy before each round, then generating replies under the guide of the selected strategy; (3) Corpus Construction & Model Training: using the strategy-enhanced GPT-4 as expert agents for large-scale dialogue generation, and training a 7B model using the generated corpus.

### Key Designs

1. **Multi-step Reasoning Strategies**:

    - **Function**: Provides structured reasoning guidance for complex social scenarios
    - **Mechanism**: Extracts classic strategies from negotiation theory, such as "Interest-Based Negotiation" (identifying core interests of both parties instead of superficial positions before seeking win-win solutions) and "Cost-Benefit Analysis" (evaluating potential gains and risks of options before responding). Each strategy is formalized as a multi-step reasoning template, containing a chain of thought path: "Analyze Current Situation $\rightarrow$ Clarify Goal $\rightarrow$ Plan Strategy $\rightarrow$ Generate Action." Such strategies are suitable for complex scenarios with conflicts of interest that require creative solutions.
    - **Design Motivation**: Simple tactics like "persuade the other party" or "insist on one's position" demonstrate limited effectiveness in complex negotiations. Multi-step reasoning strategies simulate the deep thinking process of human experts at critical negotiation nodes to break dialogue deadlocks.

2. **Direct Strategies**:

    - **Function**: Provides rapid-response guidance for simple social scenarios
    - **Mechanism**: Two simple and direct styles are designed: "Cooperative Concession" (making reasonable compromises at appropriate times to advance the dialogue) and "Holding Positions" (standing firm on core interests but maintaining a friendly tone). These strategies do not require complex reasoning and are activated via brief prompt prefixes.
    - **Design Motivation**: Not all social situations require complex reasoning; sometimes direct and explicit strategies are more natural. The combination of direct and multi-step reasoning strategies offers a complete spectrum of strategies.

3. **Dynamic Strategy Selection and Injection Mechanism**:

    - **Function**: Automatically selects the most appropriate strategy in each round of dialogue
    - **Mechanism**: Before generating a response in each round, the system analyzes the current dialogue state (such as completed rounds, target progress of both parties, whether a deadlock exists, etc.) and dynamically selects the best strategy from the strategy pool to inject into the agent's prompt. The selection process is itself conducted by an LLM—given the dialogue history and available strategy list, the LLM determines which strategy is most likely to advance the conversation. This two-stage "select strategy then generate reply" process ensures context sensitivity in strategy application.
    - **Design Motivation**: Static strategy allocation (e.g., using a fixed strategy) cannot adapt to dynamic changes in dialogue. Dynamic selection aligns strategy usage patterns closely with human experts, who flexibly adjust strategies based on the current context.

### Loss & Training

Supervised fine-tuning (SFT) is conducted on 7B models (e.g., Llama 2 7B, Mistral 7B) using the strategy-enhanced GPT-4 dialogue corpus. The training objective is standard next-token prediction, but the training data includes the strategy reasoning process (as part of the chain of thought), enabling the model to learn strategy selection concurrently with response generation.

## Key Experimental Results

### Main Results

Social goal achievement rate on SOTOPIA benchmarks:

| Model | Social Goal Score | S-IF Score | Outperforms GPT-4 |
|------|-----------|----------|--------------|
| GPT-4 (Expert Agent) | Baseline | Baseline | — |
| Llama 2 7B + SOTOPIA-Ω | **Significantly Outperforms** | **Improved** | ✅ |
| Mistral 7B + SOTOPIA-Ω | **Significantly Outperforms** | **Improved** | ✅ |
| Llama 2 7B (No Strategy Corpus) | Lower than GPT-4 | Lower | ❌ |
| Direct SFT (No Strategy Injection) | Close to GPT-4 | Moderate | Close |

### Ablation Study

| Configuration | Social Goal | Deadlock Frequency ↓ | Description |
|------|---------|----------|------|
| Full SOTOPIA-Ω | **Optimal** | **Lowest** | Dynamic combination of multi-step and direct strategies |
| Multi-step Reasoning Strategies Only | Sub-optimal | Lower | Lacks rapid response in simple scenarios |
| Direct Strategies Only | Decreased | Moderate | Weak in complex negotiation scenarios |
| Static Strategy Allocation | Significant decrease | Higher | Cannot adjust dynamically based on dialogue |
| No Strategy (Pure SFT) | Lowest | Highest | Frequently falls into deadlocks |

### Key Findings

- **7B Models Can Outperform GPT-4 Experts**: The 7B model trained on high-quality, strategy-guided corpora significantly outperforms direct prompting of GPT-4 as an agent in achieving social goals, indicating that "appropriate strategy" matters more than "model size."
- **Dynamic Construction is Key**: Dynamic strategy injection shows substantial advantages over static strategy allocation, particularly in breaking deadlocks. This validates that the application of social strategies requires context-sensitive execution.
- **Simultaneous Improvement of S-IF and Social Goals**: Post-training models not only become better at achieving social goals but also conform better to social instructions (e.g., maintaining politeness, keeping secrets), indicating that these two aspects are not mutually exclusive.
- **Strategy Combinations Outperform Single Strategies**: The combination of multi-step reasoning strategies and direct strategies yields the best results; utilizing either alone exhibits clear limitations.
- **Effective Mitigation of Deadlock Issues**: Agents without strategy guidance often fall into "tug-of-war" negotiations after 5-8 rounds, while strategy injection significantly reduces the frequency of deadlocks.

## Highlights & Insights

- **Introducing Social Science Theory into AI**: Systematically extracting strategies from negotiation theory and formalizing them as computable modules represents a "theory-driven" approach that is more interpretable and controllable than pure data-driven methods. The key insight is that "humans have accumulated thousands of years of social wisdom, and AI should directly learn this knowledge rather than starting from scratch."
- **Strategies as Training Signals**: Beyond using strategies to guide data generation, the strategy reasoning process (chain-of-thought) is embedded in the training data, enabling small models to learn the "think about strategy before speaking" paradigm. This concept of training models on "metacognition" (thinking how to act beforehand) can be transferred to other strategic dialogue scenarios like customer service and education.
- **Introduction of the S-IF Concept**: Social Instruction Following extends traditional instruction-following concepts to the social domain, identifying the overlooked dimension of "social constraint compliance" and offering a new perspective for evaluating social agents.

## Limitations & Future Work

- Strategy sources are concentrated on negotiation theory, which may need to be expanded with other strategy frameworks for non-negotiation social scenarios (e.g., chitchat, emotional support, and teamwork).
- The evaluation of social goals relies primarily on LLM auto-evaluation, which may differ from human judgment.
- The training corpus is generated by GPT-4, which may inherit the stylistic preferences and limitations of GPT-4.
- While the two evaluation metrics for S-IF are insightful, their coverage is limited; a more comprehensive S-IF evaluation framework remains to be built.
- The framework is evaluated only in English social scenarios; the applicability of strategies in cross-cultural and cross-linguistic scenarios warrants further study.
- Future work could expand the strategy library (incorporating more psychological/sociological theories) or enable models to autonomously discover and learn new strategies.

## Related Work & Insights

- **vs SOTOPIA (Original Framework)**: SOTOPIA provides a social evaluation environment but lacks strategy-guided training methods. SOTOPIA-Ω fills the gap in "how to train better social agents."
- **vs Self-Play Methods**: Self-play allows two agents to converse freely to generate training data, but the lack of strategic guidance easily produces low-quality or repetitive dialogues. SOTOPIA-Ω's strategy injection ensures the diversity and quality of the generated corpus.
- **vs Chain-of-Thought Series**: While CoT focuses on reasoning tasks, SOTOPIA-Ω transfers a similar "think before you act" paradigm to the social domain, though it must handle higher levels of uncertainty and the unpredictability of the interlocutor's responses.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematically introducing negotiation theory into the training of social agents is an interesting interdisciplinary innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive with multiple models, detailed ablations, and variant experiments, though human evaluation is missing.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, and the concepts are well-defined.
- Value: ⭐⭐⭐⭐ Highly reusable framework; the S-IF concept holds promising future impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GA-S3: Comprehensive Social Network Simulation with Group Agents](ga-s3_comprehensive_social_network_simulation_with_group_agents.md)
- [\[ACL 2025\] Tag-Evol: Achieving Efficient Instruction Evolving via Tag Injection](tag-evol_achieving_efficient_instruction_evolving_via_tag_injection.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](../../NeurIPS2025/others/fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)
- [\[ICML 2025\] Revisiting the Predictability of Performative, Social Events](../../ICML2025/others/revisiting_the_predictability_of_performative_social_events.md)
- [\[ACL 2025\] Graphically Speaking: Unmasking Abuse in Social Media with Conversation Insights](graphically_speaking_unmasking_abuse_in_social_media_with_conversation_insights.md)

</div>

<!-- RELATED:END -->
