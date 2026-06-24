---
title: >-
  [Paper Note] Agentic Knowledgeable Self-Awareness
description: >-
  [ACL 2025][LLM Agent][agent planning] This paper proposes KnowSelf, a data-driven approach that labels special tokens on the agent's self-exploration trajectories to identify different thinking situations (fast thinking, slow thinking, knowledgeable thinking). Through a two-stage training process (SFT + RPO), the agent model learns to autonomously judge when to invoke external knowledge, achieving optimal planning performance with minimal knowledge consumption cost.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "agent planning"
  - "self-awareness"
  - "special tokens"
  - "knowledge utilization"
  - "two-stage training"
date: 2026-05-08
content_hash: 3ec935a1f6eaed85
---

# Agentic Knowledgeable Self-Awareness

**Conference**: ACL 2025  
**arXiv**: [2504.03553](https://arxiv.org/abs/2504.03553)  
**Code**: [https://github.com/zjunlp/KnowSelf](https://github.com/zjunlp/KnowSelf)  
**Area**: LLM Agent / Agent Planning  
**Keywords**: agent planning, self-awareness, special tokens, knowledge utilization, two-stage training

## TL;DR
This paper proposes KnowSelf, a data-driven approach that labels special tokens on the agent's self-exploration trajectories to identify different thinking situations (fast thinking, slow thinking, knowledgeable thinking). Through a two-stage training process (SFT + RPO), the agent model learns to autonomously judge when to invoke external knowledge, achieving optimal planning performance with minimal knowledge consumption cost.

## Background & Motivation

LLM-based agents have made significant progress in interactive planning tasks such as ALFWorld and WebShop. However, existing methods commonly adopt an "inundating" training strategy—indiscriminately injecting gold trajectories, external feedback, and domain knowledge into the model. This practice overlooks a fundamental cognitive principle in human decision-making: **situational self-awareness**—the ability to dynamically evaluate situational demands and strategically allocate cognitive resources during decision-making.

In other words, when making decisions, humans automatically determine whether a step is simple and can be completed quickly via intuition, is complex and requires pausing for careful reflection, or exceeds their current capability and requires searching for external information. Existing agent training methods lack this adaptive capability—either providing no knowledge at all (limiting capability) or injecting knowledge at every step (costly and introducing noise).

Key Challenge: **Granularity control of knowledge utilization**—how to make agent models, like humans, "know what they need and when they need it"?

Key Insight: Proposing the **agentic knowledgeable self-awareness** paradigm, which reformulates the problem into enabling the model to generate special tokens to identify its current thinking situation, thereby autonomously regulating its knowledge utilization strategy.

## Method

### Overall Architecture
The pipeline of KnowSelf consists of three stages:
1. **Knowledge System Construction**: Generates step-level trajectory pairs, and creates and integrates a domain knowledge base through comparative analysis.
2. **Training Data Construction**: Allows the model to generate trajectories through self-exploration, labels special tokens using heuristic rules, and constructs training data for the three thinking modes.
3. **Two-Stage Training**: Stage 1 uses SFT to learn the generation of special tokens; Stage 2 uses RPO to enforce preference alignment.

### Key Designs

1. **Three Thinking Situations and Special Tokens (Situation Classification)**:

    - Function: Classifies each decision step of the agent into three thinking modes.
    - Mechanism: Defines three situations and their corresponding special tokens:
        - **Fast Thinking** `[FAST]`: The current step is simple, allowing the model to make quick decisions directly via its existing capabilities without extra pausing or knowledge.
        - **Slow Thinking** `[SLOW]`: The current step is challenging, and the model needs to reflect carefully before taking action (similar to System 2 thinking) but does not require external knowledge.
        - **Knowledgeable Thinking** `[KNOW]`: The current step exceeds the model's capabilities, requiring the retrieval and utilization of information from an external knowledge base to assist in decision-making.
    - Design Motivation: This classification directly maps human dual-process cognitive theory (System 1/2) with an additional knowledge retrieval dimension. Leveraging special tokens achieves lightweight behavioral switching, bypassing complex routing networks or auxiliary classifiers.

2. **Heuristic Situation Judgement**:

    - Function: Automatically labels which thinking mode should be used for each step in the training data.
    - Mechanism: The model first autonomously explores tasks to generate trajectories, constructing "step-level trajectory pairs"—pairs of successful and failed actions under the same task state. The labeling logic is as follows:
        - If the model succeeds on its own → Labeled as `[FAST]`
        - If the model fails but corrects itself after reflection → Labeled as `[SLOW]`
        - If the model fails and cannot correct itself even after reflection, but succeeds after being provided with external knowledge → Labeled as `[KNOW]`
    - Design Motivation: Failure-driven labeling precisely captures the capability boundaries of the model—knowledge is only introduced where the model genuinely struggles, avoiding unnecessary knowledge injection.

3. **Knowledge System Construction**:

    - Function: Constructs a queryable domain knowledge base for the agent.
    - Mechanism: Drawing inspiration from the AutoManual approach, GPT-4o (2024-08-06) is used to generate knowledge entries through step-level trajectory comparative analysis, followed by knowledge integration and deduplication. The knowledge base is capped at $24$ entries for ALFWorld and $10$ entries for WebShop.
    - Design Motivation: A concise knowledge base ensures that every entry is highly valuable, avoiding retrieval noise caused by knowledge redundancy. Restricting knowledge entries also encourages the model to rely more on its own capabilities rather than external knowledge.

4. **Two-Stage Training**:

    - Function: Enables the model to learn special token generation and optimize its actions in a staged manner.
    - Mechanism:
        - **Stage 1 — SFT (Supervised Fine-Tuning)**: Standard causal language modeling training is conducted on mixed data labeled with special tokens. The data contains samples of all three thinking modes, prompting the model to learn to generate `[FAST]`, `[SLOW]`, or `[KNOW]` at appropriate times. Hyperparameters (using Llama-3.1-8B as an example): $\text{lr} = 2\text{e-}5$, $\text{batch\_size} = 8$, $3$ epochs, $\text{max\_seq\_len} = 3072$.
        - **Stage 2 — RPO (Rejection-sampling Policy Optimization)**: Collects failure trajectories of the Stage 1 model as negative samples and pairs them with corresponding gold trajectories as positive samples for preference optimization. $\beta = 0.5$, $\text{lr} = 5\text{e-}7$, $1$ epoch.
    - Design Motivation: Stage 1 teaches the model the "basic form" (when to generate what token), while Stage 2 further calibrates the model's decision quality through contrastive training, specifically reducing error rates on key steps.

### Inference Process
During inference, the model automatically generates a special token before making decisions at each step:
- Generating `[FAST]` $\rightarrow$ Outputs the action directly.
- Generating `[SLOW]` $\rightarrow$ Generates an internal reflection first, then outputs the action.
- Generating `[KNOW]` $\rightarrow$ Retrieves relevant knowledge from the knowledge base using DeepSeek-V3, injects it into the context, and then outputs the action.

## Key Experimental Results

### Main Results

| Task | Model | Method | Success Rate | Steps Using External Knowledge |
|------|------|------|--------|-----------------|
| ALFWorld | Llama-3.1-8B | ReAct | ~30% | 0 |
| ALFWorld | Llama-3.1-8B | Reflexion | ~45% | 0 |
| ALFWorld | Llama-3.1-8B | ETO | ~55% | 0 |
| ALFWorld | Llama-3.1-8B | KnowAgent | ~60% | All steps |
| ALFWorld | Llama-3.1-8B | WKM | ~62% | All steps |
| ALFWorld | Llama-3.1-8B | **KnowSelf** | **~70%** | **Few steps only** |
| WebShop | Llama-3.1-8B | ReAct | ~25% | 0 |
| WebShop | Llama-3.1-8B | ETO | ~40% | 0 |
| WebShop | Llama-3.1-8B | **KnowSelf** | **~50%** | **Few steps only** |

### Ablation Study

| Configuration | ALFWorld Success Rate | Note |
|------|--------------|------|
| KnowSelf (Full) | ~70% | Optimal |
| Remove [KNOW] (No Knowledge) | ~58% | Demonstrates the necessity of knowledge retrieval |
| Remove [SLOW] (No Reflection) | ~63% | Slow thinking contributes significantly |
| All-steps Knowledge Injection (No Autonomous Regulation) | ~62% | Excessive knowledge is actually harmful |
| Stage 1 Only (No RPO) | ~64% | RPO provides around a 6% boost |

### Key Findings
- **The knowledge utilization efficiency of KnowSelf is significantly higher than baselines**: using external knowledge in only about 20-30% of steps outperforms methods that inject knowledge at every step.
- Injecting knowledge at all steps yielded poorer results than selective utilization, verifying the drawbacks of the "indiscriminate inundation" strategy.
- RPO (Stage 2) primarily reduces errors at critical decision points, with minimal impact on simpler steps.
- KnowSelf is effective on both Llama-3.1-8B and Qwen-2.5-7B, demonstrating the generalizability of the approach.
- The generation distribution of special tokens is highly correlated with task difficulty: simpler tasks have more `[FAST]` tokens, while difficult tasks exhibit more `[KNOW]` tokens.

## Highlights & Insights
- **Special tokens as thinking mode switches**: An extremely lightweight design that implements complex adaptive behavioral switching using just three special tokens, avoiding the introduction of auxiliary modules.
- **Failure-driven labeling strategy**: Additional computation/knowledge is introduced only where the model genuinely fails, perfectly capturing capability boundaries.
- **Practice of dual-process theory**: The design of fast and slow thinking is highly aligned with System 1 and System 2 theories in cognitive science.
- **On-demand regulation of knowledge retrieval**: Instead of a binary choice of "whether to use knowledge," it offers fine-grained control over "when and how much knowledge to use."

## Limitations & Future Work
- The knowledge system relies on GPT-4o for construction, meaning its cost and quality are constrained by the capabilities of the auxiliary model.
- Evaluated only on two tasks (ALFWorld and WebShop); further verification on more real-world agent tasks is needed.
- Heuristic labeling criteria may not be optimal—a model's failure does not necessarily imply a need for knowledge (better reasoning strategies might be required instead).
- The sizes of the knowledge bases (24 and 10 entries) are manually set; adaptive knowledge base sizing might perform better.
- The tri-classification representation of special tokens might be too coarse-grained; a finer categorization of situations (e.g., "what type of knowledge is needed") could be beneficial.

## Related Work & Insights
- **vs KnowAgent (Zhu et al., 2024)**: KnowAgent utilizes knowledge graphs to assist planning at all steps without distinguishing situations; KnowSelf accesses knowledge on-demand, yielding higher efficiency and better performance.
- **vs WKM (World Knowledge Model)**: WKM also introduces external knowledge but similarly performs all-steps injection; KnowSelf demonstrates the superiority of selective injection.
- **vs ETO (Exploratory Training Optimization)**: ETO optimizes behavior through exploration without using external knowledge; KnowSelf builds upon ETO by integrating the dimensions of knowledge and situational awareness.
- **vs Self-RAG**: Self-RAG utilizes special tokens to control retrieval timing but focuses on QA tasks; KnowSelf extends a similar concept to agent planning while incorporating the slow thinking dimension.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces situational self-awareness from cognitive science to agent planning; the special token design is clean and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple baseline comparisons and ablation studies are provided, but evaluation on only two task scenarios is slightly insufficient.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and conceptual analogies are intuitive, though there are relatively few formulas.
- Value: ⭐⭐⭐⭐ Proposes a practical and lightweight knowledge regulation paradigm, with open-sourced code and data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)
- [\[ACL 2025\] Gödel Agent: A Self-Referential Agent Framework for Recursive Self-Improvement](gödel_agent_a_self-referential_agent_framework_for_recursive_self-improvement.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)
- [\[ACL 2025\] Agentic Reasoning: A Streamlined Framework for Enhancing LLM Reasoning with Agentic Tools](agentic_reasoning_tools.md)
- [\[ICLR 2026\] Agentic Context Engineering: Evolving Contexts for Self-Improving Language Models](../../ICLR2026/llm_agent/agentic_context_engineering_evolving_contexts_for_self-improving_language_models.md)

</div>

<!-- RELATED:END -->
