---
title: >-
  [Paper Note] The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement
description: >-
  [NeurIPS 2025][LLM Agent][Natural language feedback] This paper proposes CGI (Critique-Guided Improvement), a dual-role framework that trains a dedicated Critic model to provide structured natural language feedback (discrimination + correction suggestions) to an Actor Agent, and enables the Actor to learn to leverage such feedback through iterative action refinement. CGI achieves an average score of 74.20% across WebShop, ScienceWorld, and TextCraft…
tags:
  - "NeurIPS 2025"
  - "LLM Agent"
  - "Natural language feedback"
  - "Actor-Critic"
  - "Iterative refinement"
  - "Interactive environments"
  - "Agent training"
date: 2026-05-08
content_hash: 4570682a24eb8395
---

# The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement

**Conference**: NeurIPS 2025  
**arXiv**: [2503.16024](https://arxiv.org/abs/2503.16024)  
**Code**: [https://github.com/rhyang2021/CGI](https://github.com/rhyang2021/CGI)  
**Area**: Agent  
**Keywords**: Natural language feedback, Actor-Critic, Iterative refinement, Interactive environments, Agent training

## TL;DR
This paper proposes CGI (Critique-Guided Improvement), a dual-role framework that trains a dedicated Critic model to provide structured natural language feedback (discrimination + correction suggestions) to an Actor Agent, and enables the Actor to learn to leverage such feedback through iterative action refinement. CGI achieves an average score of 74.20% across WebShop, ScienceWorld, and TextCraft, surpassing GPT-4o (45.46%) and Iterative SFT (58.21%).

## Background & Motivation

**Background**: LLM agents in interactive environments require iterative exploration and improvement. Existing feedback mechanisms fall into two categories: numerical feedback (reward models/verifier scoring) and natural language feedback (self-correction/LLM-as-judge).

**Limitations of Prior Work**:
   - **Limited informativeness of numerical feedback**: Methods such as Best-of-N can only select the best candidate action but cannot provide contextual information on "why it is wrong" or "how to improve"
   - **Poor quality of self-generated feedback**: Self-refinement heavily depends on the model's intrinsic capability, is prone to hallucinations and low-quality feedback, and can even degrade performance on complex tasks
   - **Difficulty in utilizing language feedback**: Even when high-quality natural language feedback is available, agents frequently fail to correctly interpret and act upon the suggestions—a problem that is particularly pronounced after SFT fine-tuning

**Key Challenge**: Natural language feedback carries richer information than numerical signals, yet the two challenges of "generating high-quality feedback" and "effectively utilizing feedback" coexist and are mutually coupled

**Goal**: ① How to train a dedicated Critic to generate high-quality structured language feedback? ② How to enable the Actor to genuinely learn to leverage this feedback for behavioral improvement during iteration?

**Key Insight**: Decompose the problem into an Actor-Critic dual-role structure and train each separately—the Critic distills high-quality feedback from GPT-4, while the Actor learns to refine actions under feedback guidance via iterative SFT

**Core Idea**: Train a dedicated Critic to generate structured "discrimination + correction" feedback, then enable the Actor to learn to translate language feedback into improved actions through iterative refinement.

## Method

### Overall Architecture
CGI is a two-stage dual-role framework consisting of **Critique Generation** (Critic training) and **Action Refinement** (iterative Actor training). At inference time, at each step the Actor first generates $M$ candidate actions; the Critic provides structured evaluation and revision suggestions for each candidate; and the Actor generates the final refined action based on this feedback and executes it. The entire process is formalized under a POMDP framework.

### Key Designs

1. **Structured Critique Generation**:

    - Function: The Critic generates structured feedback comprising a "discrimination" part and a "correction" part for each candidate action
    - Mechanism: The discrimination part evaluates candidates along three dimensions—**contribution** (whether the action advances the task), **feasibility** (whether the action is valid), and **efficiency** (whether the action follows the optimal path). The correction part provides an overall rating (Excellent/Good/Neutral/Poor/Very Poor) and specific improvement suggestions
    - Design Motivation: Unstructured free-form feedback is too vague for agents to act upon; the three-dimensional evaluation covers the key questions of "what to do," "whether it can be done," and "whether it is worth doing," ensuring feedback comprehensiveness and actionability

2. **Critic Model Training**:

    - Function: GPT-4o is used as an expert Critic to generate high-quality feedback data, which is then distilled into a smaller model
    - Mechanism: Given expert trajectories $\tau^{exp}$ as reference, GPT-4o evaluates the alignment between candidate actions and optimal actions and generates structured critiques. Only feedback from successful trajectories ($\mathcal{R}(\tau')=1$) is collected, and the Critic model is trained with standard language modeling loss $\mathcal{L}_{critic}(\phi) = \mathbb{E}[\log \pi_\phi(c_t | \tau'_t, a_t, e)]$
    - Design Motivation: An 8B Critic model alone surpasses GPT-4o as a critic (average 61.44% vs. 32.28%), demonstrating that specialized training substantially outperforms general-purpose LLMs

3. **Iterative Action Refinement**:

    - Function: Enable the Actor to learn to utilize Critiques through multiple rounds of exploration and learning
    - Mechanism: In each iteration, the Actor interacts with the environment under Critic guidance, collecting two types of data—$\mathcal{D}_{correct}$ (correct trajectories, enhancing reasoning ability) and $\mathcal{D}_{refine}$ (critique-action pairs, enhancing feedback utilization ability). $\mathcal{D}_{general}$ (ShareGPT general-purpose data) is mixed in to prevent overfitting. Each round trains from the original base model rather than the previous round's model
    - Design Motivation: Models after direct SFT tend to be less capable of leveraging external feedback (the "policy misalignment" problem); iterative refinement keeps the Actor's policy distribution aligned with Critic feedback

### Loss & Training
The Actor loss function combines three data sources: $\mathcal{L}_{actor}(\theta) = \beta[\mathbb{E}_{\mathcal{D}_{train}}[\log \pi_\theta(\tau|x,e)] + \mathbb{E}_{\mathcal{D}_{refine}}[\log \pi_\theta(a'_t|\tau'_t,c_t,e)]] + (1-\beta)\mathbb{E}_{\mathcal{D}_{general}}[\log \pi_\theta(y|x)]$. The backbone is Llama-3-8B-Instruct; each round is trained from the base model to avoid overfitting. At inference time, the default number of candidate actions is $M=5$.

## Key Experimental Results

### Main Results

| Method | WebShop | ScienceWorld | TextCraft | Avg. |
|--------|---------|-------------|-----------|------|
| GPT-4o | 25.48 | 46.91 | 64.00 | 45.46 |
| Llama-3-70B-Instruct | 8.35 | 49.20 | 2.00 | 19.85 |
| AgentLM-70B | 49.50 | 10.68 | 4.00 | 21.39 |
| Iterative SFT (8B) | 78.21 | 41.42 | 55.00 | 58.21 |
| **CGI (8B, Ours)** | **76.17** | **78.43** | **68.00** | **74.20** |

CGI outperforms Iterative SFT by +15.99% and GPT-4o by +28.74%.

### Ablation Study

| Configuration | WebShop | ScienceWorld | TextCraft | Avg. |
|---------------|---------|-------------|-----------|------|
| CGI #Iter1 (Full) | 73.22 | 66.27 | 66.00 | 68.50 |
| w/o $\mathcal{D}_{refine}$ | 74.33 | 39.33 | 37.00 | 50.22 |
| w/o $\mathcal{D}_{correct}$ | 66.25 | 60.93 | 52.00 | 59.72 |
| w/o $\mathcal{D}_{general}$ | 67.88 | 67.23 | 62.00 | 65.70 |

### Key Findings
- **$\mathcal{D}_{refine}$ is the most critical component**: Removing critique-action pairs leads to an average drop of 18.28%, with the largest impact on long-horizon tasks (ScienceWorld −26.94%)
- **Language feedback substantially outperforms numerical signals**: An 8B Critic model guiding an 8B Actor achieves 61.44%, whereas the DGAP numerical discriminator achieves only 23.64%
- **8B Critic surpasses GPT-4**: Averaging 61.44% vs. GPT-4o's 32.28% across three environments, demonstrating that a specialized small model can far exceed a general-purpose large model
- **SFT-trained models are less capable of utilizing feedback**: The SFT-tuned Llama-3-8B achieves only 55.94% on ScienceWorld under Critic guidance, while the original model reaches 68.51%
- **CGI exhibits the greatest advantage on long-horizon tasks**: Hard tasks improve by +28.75% after 3 iterations, while simple tasks converge in the first round

## Highlights & Insights
- **"Feedback utilization" matters more than "feedback quality"**: This is a counterintuitive finding—even with high-quality feedback, performance remains limited if the agent cannot leverage it. CGI addresses this fundamental problem through iterative refinement
- **Retraining from the base model at each round**: This avoids distribution shift and overfitting commonly observed in iterative SFT; the technique is simple yet effective
- **Critiques are most effective in the early stages of a trajectory**: The Revision Ratio is highest at stage 1, indicating that effective feedback helps agents enter the correct exploration direction early, avoiding unproductive search

## Limitations & Future Work
- **Dependency on GPT-4 for Critic training data**: Data distillation is costly and is bounded by GPT-4's critique quality ceiling
- **High inference overhead**: Generating $M=5$ candidate actions per step plus Critic evaluation incurs an inference cost approximately 6× that of standard inference
- **Validated only in simulated environments**: WebShop, ScienceWorld, and TextCraft are relatively simple text-based interactive environments; validation in real-world settings such as software engineering and web navigation remains absent
- **Critic and Actor share the same backbone**: Optimal combinations of differently sized Critics and Actors have not been explored
- **The general data mixing ratio $\beta$ has not been analyzed in detail**

## Related Work & Insights
- **vs. Reflexion**: Reflexion uses self-generated summaries as feedback for subsequent rounds but is prone to local optima, yielding minimal improvements across the three environments. CGI employs an external Critic for more objective feedback
- **vs. Best-of-N / DGAP**: Numerical signals can only perform selection ("which is better") but not correction ("how to improve"), whereas CGI's language feedback carries substantially higher information density
- **vs. Self-Critique**: Self-critique achieves an average of only 10.19% across three environments with the 8B model, performing even worse than no-critique (12.65%), confirming the unreliability of self-feedback in small models

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of the dual-role framework and iterative refinement is novel; the observation that SFT-trained models struggle to utilize feedback is a valuable insight
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three environments, multiple baselines, detailed ablations, trajectory analysis, and analysis of the effect of candidate count are all comprehensive
- Writing Quality: ⭐⭐⭐⭐ The structure is clear and the organization around three Findings is effective, though some notation definitions are scattered
- Value: ⭐⭐⭐⭐ The Actor-Critic language feedback paradigm has practical value for agent training; the result that an 8B model surpasses GPT-4 is a practically significant conclusion

## Key Experimental Results

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LocAgent: Graph-Guided LLM Agents for Code Localization](../../ACL2025/llm_agent/locagent_graph-guided_llm_agents_for_code_localization.md)
- [\[ACL 2025\] Enhancing LLM Agent Safety via Causal Influence Prompting](../../ACL2025/llm_agent/enhancing_llm_agent_safety_via_causal_influence_prompting.md)
- [\[ACL 2025\] Enhancing Interpretable Image Classification Through LLM Agents and Conditional Concept Bottleneck Models](../../ACL2025/llm_agent/llm_agent_image_classification.md)
- [\[ICLR 2026\] DreamPhase: Offline Imagination and Uncertainty-Guided Planning for Large-Language-Model Agents](../../ICLR2026/llm_agent/dreamphase_offline_imagination_and_uncertainty-guided_planning_for_large-languag.md)
- [\[NeurIPS 2025\] DefenderBench: A Toolkit for Evaluating Language Agents in Cybersecurity Environments](defenderbench_a_toolkit_for_evaluating_language_agents_in_cybersecurity_environm.md)

</div>

<!-- RELATED:END -->
