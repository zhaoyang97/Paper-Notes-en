---
title: >-
  [Paper Note] OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers
description: >-
  [ACL 2025][LLM (Other)][Prompt strategy selection] This work proposes OPTS, the first explicit selection mechanism for prompt design strategies, modeling 11 strategies (such as CoT, role-playing, and emotional prompts) as arms of a multi-armed bandit. By using Thompson sampling to automatically select the most suitable strategy and integrating it into the EvoPrompt optimizer, OPTS achieves up to a 50% performance improvement using GPT-4o mini across 23 tasks of BIG-Bench Hard…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Prompt strategy selection"
  - "Thompson sampling"
  - "Multi-armed bandit"
  - "EvoPrompt"
  - "BIG-Bench Hard"
date: 2026-05-08
content_hash: 0c5986f36fd351c9
---

# OPTS: Bandit-Based Prompt Design Strategy Selection Improves Prompt Optimizers

**Conference**: ACL 2025  
**arXiv**: [2503.01163](https://arxiv.org/abs/2503.01163)  
**Code**: [GitHub](https://github.com/shiralab/OPTS)  
**Area**: LLM/NLP  
**Keywords**: Prompt strategy selection, Thompson sampling, Multi-armed bandit, EvoPrompt, BIG-Bench Hard

## TL;DR

This work proposes OPTS, the first explicit selection mechanism for prompt design strategies, modeling 11 strategies (such as CoT, role-playing, and emotional prompts) as arms of a multi-armed bandit. By using Thompson sampling to automatically select the most suitable strategy and integrating it into the EvoPrompt optimizer, OPTS achieves up to a 50% performance improvement using GPT-4o mini across 23 tasks of BIG-Bench Hard.

## Background & Motivation

**Prompt optimization automates the search for effective prompts, but the results often fall short of human-expert designs.** Methods like EvoPrompt use LLMs to simulate evolutionary algorithms to search the prompt space. Although they discover valid prompts, these prompts often lack design strategies commonly used by human experts (such as Chain-of-Thought reasoning, role-playing, step-by-step instructions, etc.), leaving a quality gap compared to well-crafted expert prompts.

**Prompt design strategies are not always beneficial.** CoT and role-playing prompts can actually degrade performance on certain LLM and task combinations. This implies that one cannot simply apply all strategies to every prompt. The APET method feeds all strategy descriptions together to the LLM for implicit selection, but the LLM's own optimization capability is limited, making implicit selection potentially suboptimal.

**Key Challenge: Strategies are valuable, but when to use which strategy remains unknown.** This is inherently an explore-exploit problem, where the classic multi-armed bandit framework is perfectly suited. **The Core Idea of OPTS is to treat each prompt design strategy as an 'arm' and use Thompson sampling to dynamically learn which strategy is most effective for the current task during the optimization process**, realizing the first explicit strategy selection mechanism.

## Method

### Overall Architecture

OPTS is inserted as a module into existing prompt optimizers (such as EvoPrompt) after the mutation/crossover steps: EvoPrompt generates candidate prompts $\rightarrow$ OPTS selects a strategy $\rightarrow$ LLM applies the strategy to the candidate prompt $\rightarrow$ evaluation performance is fed back to update the multi-armed bandit.

### Key Designs

1. **Multi-Armed Bandit Modeling**:
    - **Function**: Models $K=11$ prompt design strategies plus 1 "no strategy" inaction arm as $K+1=12$ arms.
    - **Mechanism**: Since the value of each strategy is uncertain and may change with the task, a bandit framework is used to learn the optimal strategy online. The inaction arm ensures that no strategy is forced—it is possible that all strategies are non-beneficial.
    - **Design Motivation**: Explicit selection is more controllable and optimizable than implicit selection.

2. **Thompson Sampling Selection (OPTS-TS)**:
    - **Function**: Utilizes Beta distribution priors and posterior updates to achieve an efficient exploration-exploitation balance.
    - **Mechanism**: Each arm maintains a $\text{Beta}(\alpha_k, \beta_k)$ distribution. In each iteration, parameters are sampled from each arm's distribution, and the arm with the highest sampled value is selected. The reward is defined as $r = \mathbf{1}[s > \max \tilde{S}]$, which indicates whether the new prompt's score exceeds the maximum score of the parent prompts.
    - **Design Motivation**: Thompson sampling provides theoretically guaranteed asymptotic optimality and achieves outstanding practical performance in bandit literature.

3. **Strategy Application Mechanism**:
    - **Function**: Passes the textual description of the selected strategy to the prompt-design LLM, instructing it to modify the candidate prompt.
    - **Mechanism**: When one of the first $K$ arms is selected, the corresponding strategy description and the prompt to be modified are fed together into the LLM. When the inaction arm is selected, no modification is made. The same meta-prompt format as APET is utilized.
    - **Design Motivation**: Strategies are passed as natural language descriptions, which the LLM naturally understands how to apply.

### 11 Prompt Design Strategies

These include ExpertPrompting (expert role-playing), CoT (Chain-of-Thought), Tree-of-Thought, Emotion Prompting, Re-Reading, Style Prompting, Rephrase and Respond, Avoiding Bias, Making Prompt Specific, Shortening, and Adding Necessary Information.

### Integration with EvoPrompt

OPTS can be integrated with both EvoPrompt(DE) and EvoPrompt(GA). In the DE variant, OPTS is inserted after crossover and mutation but before selection: first, $p'_i$ is generated using DE operations, then OPTS selects a strategy to modify it into $p''_i$, and finally, it is compared with the parent $p_i$ to retain the superior one.

## Key Experimental Results

### Main Results (GPT-4o mini for generation and problem-solving)

| Method | BIG-Bench Hard 23-Task Average Accuracy | vs EvoPrompt(DE) |
|------|-------------------------------|-----------------|
| Manual Prompt | 56.95 | - |
| APET | 57.93 | - |
| EvoPrompt(DE) | 60.11 | baseline |
| +OPTS(APET) | 62.36 | +2.25 |
| +OPTS(US) | 63.04 | +2.93 |
| +OPTS(TS) | **64.15** | **+4.04** |

### Llama-3-8B-Instruct for Evaluation

| Method | Average Accuracy | vs EvoPrompt(DE) |
|------|-----------|-----------------|
| EvoPrompt(DE) | 46.52 | baseline |
| +OPTS(TS) | **49.83** | **+3.31** |

### Ablation Study

| Configuration | Description | Results |
|------|------|------|
| OPTS(TS) vs OPTS(US) | TS vs Uniform Sampling | TS is superior in most tasks |
| OPTS(TS) vs OPTS(APET) | TS vs Implicit Selection | TS is consistently superior |
| EvoPrompt(GA)+OPTS(TS) | GA variant | Equally effective, independent of specific optimization algorithms |
| Inaction arm removal | Force-using strategies | Performance degrades — some tasks do not require strategies |

### Key Findings

- **Thompson sampling is consistently optimal**: It outperforms other selection mechanisms on both GPT-4o mini and Llama-3.
- **Up to 50% improvement on individual tasks**: On certain tasks, OPTS(TS) improves performance by 50% compared to EvoPrompt.
- **The inaction arm is critical**: Not all tasks benefit from strategies, making the "do not use" option essential.
- **Different tasks prefer different strategies**: Thompson sampling automatically learns the optimal strategy distribution for each task.

## Highlights & Insights

- **First to introduce multi-armed bandits to prompt strategy selection** — a simple concept, but applying a classic RL tool to the right problem.
- **The insight that "strategies can be harmful" is important** — highly relevant to all prompt engineering practitioners, advising against blindly stacking strategies.
- **Modular design**: OPTS is an independent module that can be plug-and-played into any prompt optimizer.
- **Explicit > Implicit**: The explicit selection of OPTS(TS) consistently outperforms the implicit selection of APET, showing that using dedicated optimization algorithms for optimization is superior to letting the LLM do it itself.

## Limitations & Future Work

- **Only evaluated on BIG-Bench Hard**: Performance on other types of tasks (such as generation, dialogue, coding, etc.) remains unknown.
- **Fixed strategy set**: The 11 predefined strategies cannot cover all possible effective strategies, and the automated discovery of new strategies remains unexplored.
- **Single strategy selection**: Only one strategy is applied at a time, and the effects of combined strategies have not been explored.
- **Context window limitations**: Strategy descriptions consume prompt space, which could compress critical information if descriptions are too long.

## Related Work & Insights

- **vs APET (Implicit Selection)**: Feeds all strategy descriptions to the LLM for implicit selection. In contrast, OPTS's explicit selection is more controllable and yields superior performance.
- **vs PromptWizard (Fixed Strategies)**: Always applies CoT and role-playing prompts. In contrast, OPTS selects strategies on-demand to avoid harmful strategies.
- **vs EvoPrompt (No Strategies)**: Lacks human design knowledge. OPTS injects best practices into the evolutionary optimization process.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First explicit strategy selection mechanism; a novel application of Thompson sampling in prompt optimization.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 23 tasks of BIG-Bench Hard $\times$ 2 models $\times$ 3 selection mechanisms $\times$ 2 optimization algorithm variants.
- **Writing Quality**: ⭐⭐⭐⭐ Clear problem definition, complete algorithmic descriptions, and intuitive illustrations.
- **Value**: ⭐⭐⭐⭐ Directly practical for prompt optimization practitioners, with a modular design that is easy to adopt.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Why Prompt Design Matters and Works: A Complexity Analysis of Prompt Search Space in LLMs](why_prompt_design_matters_and_works_a_complexity_analysis_of_prompt_search_space.md)
- [\[ACL 2025\] What Makes a Good Natural Language Prompt?](good_natural_language_prompt.md)
- [\[ACL 2025\] RiOT: Efficient Prompt Refinement with Residual Optimization Tree](riot_efficient_prompt_refinement_with_residual_optimization_tree.md)
- [\[ACL 2025\] Efficient Universal Goal Hijacking with Semantics-guided Prompt Organization](goal_hijacking_attack.md)
- [\[ACL 2025\] SEE: Strategic Exploration and Exploitation for Cohesive In-Context Prompt Optimization](see_strategic_exploration_exploitation_prompt_optimization.md)

</div>

<!-- RELATED:END -->
