---
title: >-
  [Paper Note] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning
description: >-
  [ICLR 2026][LLM Agent][Tool Planning] This paper proposes ToolTree, an MCTS-based tool planning framework for LLM agents that achieves look-ahead tool selection within a fixed computational budget through a dual-phase pre/post-execution evaluation mechanism and bidirectional pruning, yielding an average improvement of approximately 10% across 4 benchmarks.
tags:
  - ICLR 2026
  - LLM Agent
  - Tool Planning
  - MCTS
  - Search Planning
  - Pruning
date: 2026-05-08
content_hash: 0afbb79b4cfe25bb
---

# ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning

**Conference**: ICLR 2026
**arXiv**: [2603.12740](https://arxiv.org/abs/2603.12740)
**Code**: [https://github.com/SYang2000/ICLR_2026_ToolTree](https://github.com/SYang2000/ICLR_2026_ToolTree)
**Area**: Agent
**Keywords**: Tool Planning, MCTS, LLM Agent, Search Planning, Pruning

## TL;DR
This paper proposes ToolTree, an MCTS-based tool planning framework for LLM agents that achieves look-ahead tool selection within a fixed computational budget through a dual-phase pre/post-execution evaluation mechanism and bidirectional pruning, yielding an average improvement of approximately 10% across 4 benchmarks.

## Background & Motivation

**Background**: LLM agents addressing multi-step complex tasks must invoke chains of external tools (APIs, search engines, calculators, etc.), and the core challenge is tool planning — determining which tools to use, in what order, and with what parameters.

**Limitations of Prior Work**: (a) Greedy methods (ReAct, CoT) select the locally optimal tool at each step without look-ahead, causing early errors to propagate irreversibly; (b) Search methods (ToT, A*) expand multiple candidate branches, but the branching factor grows exponentially with the number of tools, incurring high computational cost while relying on hypothetical reasoning rather than actual execution results for evaluation.

**Key Challenge**: Search methods provide look-ahead but at high computational cost with ungrounded evaluation (evaluating hypothetical thoughts); greedy methods are efficient but lack error-correction capability. A method that combines look-ahead with feedback grounded in actual execution results is needed.

**Goal**: How can an agent perform look-ahead tool planning within a fixed computational budget while guaranteeing efficiency?

**Key Insight**: The standard MCTS selection–expansion–simulation–backpropagation loop is adapted into a framework suited for tool invocation, where an LLM rapidly pre-evaluates branches before execution and post-execution scores based on actual outputs are used to refine the policy.

**Core Idea**: Dual-phase evaluation (pre-execution anticipation + post-execution measurement) combined with bidirectional pruning (pruning low-scoring branches before execution and failed branches after execution), enabling MCTS to be both efficient and accurate in tool planning scenarios.

## Method

### Overall Architecture
ToolTree models tool planning as a sequential decision-making process: state $s$ encodes the dialogue context and intermediate results, and action $a$ corresponds to invoking a specific tool. Each root-to-leaf path in the search tree constitutes a candidate tool-call sequence. The MCTS loop iterates through: Selection → Pre-Evaluation → Expansion → Execution → Post-Evaluation → Backward Propagation, ultimately returning the highest-scoring trajectory to generate the final answer.

### Key Designs

1. **Prior-Augmented Selection**:

    - Function: Incorporates the pre-evaluation score $r_{\text{pre}}$ into the standard UCT formula as an exploration guide.
    - Mechanism: $\text{UCT}(s,a) = Q(s,a) + \lambda \cdot r_{\text{pre}}(s,a) \cdot \sqrt{\frac{\ln N(s)}{N(s,a)}}$, where $Q(s,a)$ drives exploitation and $r_{\text{pre}}$ steers exploration toward promising branches.
    - Design Motivation: The exploration term in standard MCTS depends solely on visit counts, which is insufficiently efficient for tool-calling scenarios; incorporating semantic priors biases the search toward reasonable tool combinations from the outset.

2. **Pre-Evaluation**:

    - Function: Before actually invoking a tool, the LLM assesses the applicability of that tool in the current context and outputs $r_{\text{pre}}(s,a) \in [0,1]$.
    - Mechanism: A lightweight score is computed based on the current dialogue context, tool cards (I/O patterns, descriptions, examples), and draft parameters; actions below threshold $\tau_{\text{pre}}$ are pruned without expansion.
    - Design Motivation: Avoids wasting API call budgets on clearly unsuitable tools, substantially reducing the branching factor.

3. **Post-Evaluation**:

    - Function: After tool execution, the LLM evaluates the quality of the actual output $r_{\text{post}}(s,a) \in [0,1]$.
    - Mechanism: $r_{\text{post}} = J(C_t, a, o_{t+1})$ assesses task consistency, correctness, constraint satisfaction, etc.; branches scoring below $\tau_{\text{post}}$ are marked as non-expandable.
    - Design Motivation: Scoring based on actual execution results is more reliable than hypothetical reasoning, enabling faithful credit assignment.

4. **Bidirectional Pruning**:

    - Function: Pre-pruning removes candidates with low $r_{\text{pre}}$ and retains only the top-$K$; post-pruning closes branches with low $r_{\text{post}}$.
    - Design Motivation: The two stages are complementary — the former reduces candidates based on anticipation, and the latter eliminates dead ends based on evidence, collectively concentrating the budget on the most promising trajectories.

5. **Deterministic Caching + Error Handling**:

    - Function: Within the same rollout, results from identical (tool, args) calls are reused; failed calls are appended with error tokens.
    - Design Motivation: Avoids redundant API calls that waste budget; explicit failure handling supports subsequent scoring and pruning decisions.

### Loss & Training
ToolTree is a **training-free** planning framework that requires no model fine-tuning. All evaluations are performed by an LLM judge, and search is conducted online at inference time.

## Key Experimental Results

### Main Results

**Closed-set: GTA (GPT-4o)**:

| Method | AVG F1 | vs Zero-shot |
|--------|--------|-------------|
| Zero-shot | 57.78 | baseline |
| ReAct | 58.46 | +0.7 |
| LATS (MCTS) | 64.78 | +7.0 |
| **ToolTree** | **66.95** | **+9.2** |

**Open-set: ToolBench (GPT-4o)**:

| Method | Pass Rate | Win Rate | AVG |
|--------|-----------|----------|-----|
| ReAct | 62.24 | 56.02 | 59.13 |
| LATS | 66.61 | 64.77 | 65.69 |
| **ToolTree** | **69.04** | **67.52** | **68.28** |

### Ablation Study

| Configuration | GTA F1 | ToolBench Pass |
|---------------|--------|---------------|
| Full ToolTree | 66.95 | 69.04 |
| w/o pre-evaluation | −2~3% | Decrease |
| w/o post-evaluation | −1~2% | Decrease |
| w/o bidirectional pruning | Efficiency degrades noticeably | Increased computational waste |

### Key Findings
- Efficiency analysis: In the 32–64 step budget range, ToolTree achieves the highest accuracy-per-second, demonstrating that bidirectional pruning effectively concentrates the computational budget.
- Pre-evaluation contributes most at low budgets (16 steps), where the effect of early pruning is most pronounced.
- Post-evaluation is more critical for long-chain tasks, as error correction based on actual execution results becomes necessary.
- ToolTree yields consistent improvement gains from GPT-4o-mini to GPT-4o, indicating that the method does not depend on model-specific capabilities.

## Highlights & Insights
- **Complementary dual-phase evaluation design**: Pre-evaluation provides a rapid "should this be done?" judgment, while post-evaluation provides a deliberate "was it done well?" assessment. This foresight–hindsight loop represents a general paradigm transferable to any search planning problem involving executable actions.
- **Training-free practicality**: No fine-tuning is required; the framework is plug-and-play and compatible with arbitrary LLM + tool library combinations at minimal deployment cost.
- **Efficiency–accuracy Pareto optimality**: Although slower than greedy methods, ToolTree achieves the highest accuracy-per-second, suggesting that "searching intelligently" matters more than "searching more."

## Limitations & Future Work
- The LLM judge itself incurs cost (two LLM calls per step), which may be prohibitive in API price-sensitive settings.
- Pre-evaluation quality depends on the LLM's understanding of tools; if tool descriptions are unclear, $r_{\text{pre}}$ may be unreliable.
- Learning-based evaluation functions (e.g., fine-tuning a small reward model to replace the LLM judge) have not been explored.
- Validation is limited to GPT-4o/4o-mini; open-source models have not been tested.

## Related Work & Insights
- **vs ReAct**: ReAct performs stepwise greedy selection, whereas ToolTree achieves global planning via tree search; however, ReAct's simplicity retains advantages on straightforward tasks.
- **vs LATS (Language Agent Tree Search)**: LATS employs standard MCTS; ToolTree augments it with a pre-evaluation prior and bidirectional pruning, consistently achieving better performance.
- **vs ToolChain* (A\*)**: A* requires accurate heuristic functions; ToolTree uses an LLM judge as an adaptive heuristic, offering greater flexibility.

## Rating
- Novelty: ⭐⭐⭐⭐ Combining MCTS with dual-phase evaluation and bidirectional pruning is a natural yet effective contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 4 benchmarks, 2 model scales, efficiency analysis, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich figures and tables.
- Value: ⭐⭐⭐⭐ The training-free approach offers strong practicality, though LLM judge costs may limit real-world deployment.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Prune4Web: DOM Tree Pruning Programming for Web Agent](../../AAAI2026/llm_agent/prune4web_dom_tree_pruning_programming_for_web_agent.md)
- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](../../AAAI2026/llm_agent/agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)
- [\[ICLR 2026\] Efficient Agent Training for Computer Use](efficient_agent_training_for_computer_use.md)
- [\[ICLR 2026\] LiveNewsBench: Evaluating LLM Web Search Capabilities with Freshly Curated News](livenewsbench_evaluating_llm_web_search_capabilities_with_fresh_news.md)
- [\[ICLR 2026\] MC-Search: Evaluating and Enhancing Multimodal Agentic Search with Structured Long Reasoning Chains](mc-search_evaluating_and_enhancing_multimodal_agentic_search_with_structured_lon.md)

<!-- RELATED:END -->
