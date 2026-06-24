---
title: >-
  [Paper Note] DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal
description: >-
  [ACL 2025][Code Intelligence][Coding agents] This paper proposes DARS (Dynamic Action Re-Sampling), an inference-time compute scaling method for coding agents. It dynamically branches and attempts alternative actions at key decision points where the agent makes suboptimal choices. Using Claude 3.5 Sonnet V2, DARS achieves a 55% pass@k and 47% pass@1 on SWE-Bench Lite, outperforming the open-source SOTA frameworks of the time.
tags:
  - "ACL 2025"
  - "Code Intelligence"
  - "Coding agents"
  - "inference-time compute scaling"
  - "tree search"
  - "dynamic re-sampling"
  - "SWE-Bench"
date: 2026-05-08
content_hash: 1e1ea083d8094b98
---

# DARS: Dynamic Action Re-Sampling to Enhance Coding Agent Performance by Adaptive Tree Traversal

**Conference**: ACL 2025  
**arXiv**: [2503.14269](https://arxiv.org/abs/2503.14269)  
**Code**: None  
**Area**: Code Intelligence  
**Keywords**: Coding agents, inference-time compute scaling, tree search, dynamic re-sampling, SWE-Bench

## TL;DR

This paper proposes DARS (Dynamic Action Re-Sampling), an inference-time compute scaling method for coding agents. It dynamically branches and attempts alternative actions at key decision points where the agent makes suboptimal choices. Using Claude 3.5 Sonnet V2, DARS achieves a 55% pass@k and 47% pass@1 on SWE-Bench Lite, outperforming the open-source SOTA frameworks of the time.

## Background & Motivation

**Background**: LLM-driven coding agents (such as SWE-Agent, Aider, OpenDevin) have demonstrated great potential in automating software engineering tasks, enabling automatic bug localization, patch generation, and the completion of complex development tasks. The common inference pattern is linear trajectory execution, where the agent sequentially generates and executes actions step by step.

**Limitations of Prior Work**: The core issue with linear trajectory approaches is that once the agent makes an incorrect decision at a certain step, all subsequent operations are built on this error, making recovery extremely difficult. Existing compute-scaling strategies (such as simple best-of-N random sampling) can improve success rates by generating multiple trajectories in parallel, but they are highly inefficient because multiple trajectories may make the same mistakes in different places, wasting substantial computational resources.

**Key Challenge**: The lack of fault tolerance in linear execution vs. the low efficiency of fully random sampling. There is a need for a targeted backtracking mechanism that precisely branches into alternative paths upon detecting errors, rather than starting over from scratch.

**Goal**: Design an efficient inference-time compute-scaling strategy that can utilize execution feedback to identify key decision points and intelligently branch into alternative action paths at these points.

**Key Insight**: The authors observe that programming tasks naturally provide execution feedback (e.g., test pass/fail, compilation errors), which can serve as signals to evaluate decision quality. By utilizing this feedback, the nodes requiring backtracking can be precisely localized.

**Core Idea**: In the execution trajectory of a coding agent, when a suboptimal decision is detected, instead of restarting, the agent backtracks to that decision point and utilizes the prior trajectory history along with execution feedback to generate an alternative action, thereby forming an adaptive decision tree.

## Method

### Overall Architecture

The execution flow of DARS is: (1) The agent begins linear execution of the task and records actions and feedback at each step; (2) When a step fails execution or triggers a preset branching condition, this step is marked as a "key decision point"; (3) The agent backtracks to this decision point and uses the prior trajectory history combined with the failure feedback of this attempt as context to prompt the LLM to generate an alternative action; (4) Execution resumes from the alternative action; (5) If the alternative path also fails, the agent continues to backtrack to an earlier decision point or attempts other alternative actions, forming an adaptive tree traversal.

### Key Designs

1. **Key Decision Point Identification**:

    - **Function**: Determine which steps in the trajectory are worth backtracking and retrying.
    - **Mechanism**: Utilize execution feedback (e.g., test results, error messages, compilation state) to judge whether an action led to a "suboptimal" outcome. When execution failure is detected or a code edit introduces new errors, this step is marked as a key decision point. This is far more precise than randomly selecting branching points.
    - **Design Motivation**: Not all steps are worth backtracking—only the "key turning points" that lead to subsequent failures need to be corrected. This significantly reduces the search space.

2. **Feedback-Aware Alternative Action Generation**:

    - **Function**: Generate informed alternative actions at key decision points.
    - **Mechanism**: Input the complete trajectory history (all actions and observations up to the decision point), along with previously attempted actions and their failure feedback, into the LLM as a prompt. Thus, the LLM not only knows "what to do" but also "what does not work," leading to the generation of truly different alternative options.
    - **Design Motivation**: Ordinary random sampling simply regenerates actions under the same context, which likely produces similar actions. By explicitly providing "failure experience," the LLM can be guided to explore fundamentally different resolution paths.

3. **Adaptive Tree Traversal Strategy**:

    - **Function**: Control the breadth and depth of the search to maximize the success rate within the computational budget.
    - **Mechanism**: Adopt a depth-first search strategy, prioritizing deep execution along the current branch, and only backtracking to the previous branching point when a path completely fails. The number of allowed alternative actions per decision point is capped to avoid infinite expansion. The overall computation budget is managed by limiting the total number of action steps.
    - **Design Motivation**: Compared to breadth-first search (expanding multiple paths simultaneously), a depth-first strategy is better suited for programming tasks—completing a full repair attempt before deciding to backtrack, rather than switching between multiple incomplete paths.

### Loss & Training

DARS is a pure inference-time method requiring no additional training. It directly leverages existing pre-trained LLMs (such as Claude 3.5 Sonnet V2) as policy models, guiding decisions solely through carefully designed prompt templates and execution feedback.

## Key Experimental Results

### Main Results

Results on the SWE-Bench Lite benchmark:

| Method | pass@1 | pass@k | Base Model |
|------|--------|--------|----------|
| SWE-Agent | 18.0% | — | GPT-4 |
| Aider | 26.3% | — | Claude 3 Opus |
| AutoCodeRover | 30.7% | — | GPT-4o |
| OpenDevin (best-of-N) | 38.0% | 45.0% | Claude 3.5 |
| **DARS** | **47.0%** | **55.0%** | Claude 3.5 Sonnet V2 |

### Ablation Study

| Configuration | pass@1 | Description |
|------|--------|------|
| Linear trajectory (no branching) | ~38% | Standard single-turn execution |
| Random re-sampling (best-of-N) | ~42% | Best-of-N independent runs |
| DARS (w/o feedback) | ~44% | Branching without utilizing failure feedback |
| **DARS (Full)** | **47%** | Adaptive branching utilizing feedback |

### Key Findings

- **Feedback-aware branching is key**: Compared to blind re-sampling, using execution feedback to guide alternative action generation yields the largest gain (approximately a 5 percentage point improvement).
- **Compute efficiency outperforms best-of-N**: With the same computational budget, DARS achieves a pass@k about 10 percentage points higher than the simple best-of-N strategy, as it avoids repeatedly wasting computation on already explored failed paths.
- **Most effective with Claude 3.5 Sonnet V2**: Powerful base models are better equipped to leverage feedback information to generate meaningful alternatives.

## Highlights & Insights

- **A new paradigm for inference-time scaling**: Unlike the traditional sample-and-verify paradigm (e.g., best-of-N), DARS proposes a method to dynamically branch during execution, closely resembling the debugging mindset of human programmers—"this path does not work, let's try another direction."
- **Elegant feedback design**: Providing information from failed trajectories as "negative examples" to the LLM is an approach that can be transferred to any agent task with explicit feedback signals (e.g., mathematical reasoning, robotic planning).
- **Highly practical**: Plug-and-play into existing coding agent frameworks without training, requiring only modifications to the inference strategy.

## Limitations & Future Work

- **Dependence on powerful base models**: The effectiveness of DARS highly depends on the LLM's capacity to utilize feedback, which may limit its performance with weaker models.
- **Insufficient automation of branching strategies**: Decisions such as when and how many times to branch currently rely on heuristic rules; ideally, a learned value function should guide this process.
- **Only evaluated on SWE-Bench Lite**: This benchmark is moderately difficult, and verification has not been performed on the harder SWE-Bench Full or other programming tasks.
- **Computational overhead scales exponentially with branching**: Though more efficient than best-of-N, deep recursive branching still consumes substantial computational resources.
- **No comparison with tree search methods (e.g., MCTS)**: Methods like MCTS can use value estimation to select branches more intelligently, and combining them with DARS could yield better performance.

## Related Work & Insights

- **vs SWE-Agent**: SWE-Agent employs linear trajectory execution with no backtracking capability. DARS adds adaptive branching on top of it, significantly improving the success rate.
- **vs best-of-N Strategy**: Multiple independent samplings lack information sharing, meaning different trajectories may make the exact same mistakes. DARS avoids duplicate errors by branching at failure points and passing failure information.
- **vs Tree-of-Thought / MCTS**: ToT is suited for searching in reasoning tasks, whereas DARS is an application of similar concepts within agent execution. The key difference is that DARS utilizes real code execution feedback (rather than model self-evaluation) to guide the search.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of adaptive tree traversal and feedback-aware re-sampling is a relatively new attempt in coding agents.
- Experimental Thoroughness: ⭐⭐⭐ Evaluated only on the SWE-Bench Lite benchmark, and ablation experiments are not exhaustive.
- Writing Quality: ⭐⭐⭐⭐ The method description is clear, and the motivation is reasonably argued.
- Value: ⭐⭐⭐⭐ Holds significant reference value for inference-time compute-scaling strategies of coding agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Tree-of-Code: A Tree-Structured Exploring Framework for End-to-End Code Generation](tree-of-code_a_tree-structured_exploring_framework_for_end-to-end_code_generatio.md)
- [\[ACL 2025\] SceneGenAgent: Precise Industrial Scene Generation with Coding Agent](scenegenagent_precise_industrial_scene_generation_with_coding_agent.md)
- [\[ACL 2025\] SHARE: An SLM-based Hierarchical Action CorREction Assistant for Text-to-SQL](share_text_to_sql_correction.md)
- [\[NeurIPS 2025\] A Self-Improving Coding Agent](../../NeurIPS2025/code_intelligence/a_selfimproving_coding_agent.md)
- [\[ACL 2025\] UTBoost: Rigorous Evaluation of Coding Agents on SWE-Bench](utboost_rigorous_evaluation_of_coding_agents_on_swe-bench.md)

</div>

<!-- RELATED:END -->
