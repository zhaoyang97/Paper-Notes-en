---
title: >-
  [Paper Note] Prune4Web: DOM Tree Pruning Programming for Web Agent
description: >-
  [AAAI 2026][LLM Agent][DOM tree pruning] This paper proposes Prune4Web, a programmatic DOM pruning approach that achieves 25–50× candidate element reduction via "LLM-generated scoring function parameters + fixed heuristi…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "DOM tree pruning"
  - "programmatic filtering"
  - "Web Agent"
  - "element grounding"
  - "scoring function generation"
date: 2026-05-08
content_hash: e0ba523d8df4846b
---

# Prune4Web: DOM Tree Pruning Programming for Web Agent

**Conference**: AAAI 2026
**arXiv**: [2511.21398](https://arxiv.org/abs/2511.21398)  
**Code**: N/A  
**Area**: LLM Agent / Web Agent / DOM Processing
**Keywords**: DOM tree pruning, programmatic filtering, Web Agent, element grounding, scoring function generation

## TL;DR
This paper proposes Prune4Web, a programmatic DOM pruning approach that achieves 25–50× candidate element reduction via "LLM-generated scoring function parameters + fixed heuristic template execution." The three-stage pipeline (Planner decomposes subtasks → Programmatic Filter generates scoring functions to prune DOM → Grounder executes actions) enables a 3B model to achieve 52.4% Step SR on Multimodal-Mind2Web, surpassing all baselines of the same parameter scale and even some 9.6B/32B models, while improving low-level grounding accuracy from 46.8% to 88.28%.

## Background & Motivation

**Background**: Web Agents must comprehend webpage DOM structures to execute actions, but modern webpages typically contain 1–100K tokens in the DOM, causing token truncation and attention dilution when fed directly into LLMs.

**Limitations of Prior Work**: Existing approaches either truncate the DOM directly (losing critical elements), employ standalone ranking/filtering models (high training cost, poor generalization), or rely on LLMs to perform Top-N selection (ineffective for small models).

**Key Challenge**: There is a fundamental tension between preserving key interactive elements and drastically reducing DOM size. LLMs excel at semantic understanding but struggle with large-scale structured data; heuristic rules are fast but lack semantic comprehension. How can the strengths of both be combined?

**Key Insight**: Rather than having the LLM process the DOM directly, the key insight is to have the LLM **generate the parameters of a DOM-processing program**—specifically, a keyword-weight dictionary consumed by a fixed scoring function template that performs multi-tier, multi-match-type element scoring.

**Core Idea**: LLM generates keyword parameters for the scoring function (controllable) + fixed template performs multi-tier weighted matching (efficient) = programmatic DOM pruning.

## Method

### Overall Architecture
A three-stage pipeline: Planner → Programmatic Element Filter → Action Grounder, trained jointly within a two-turn dialogue framework using Qwen2.5VL-3B.

### Key Designs

1. **Planner**:

    - Function: Decomposes high-level tasks into low-level subtask instructions.
    - Mechanism: Takes task $T$, screenshot $Sc_t$, and history $H_t$ as input, and outputs a subtask $S_t$ (e.g., "Find the destination field and Type NYC"). **Does not access HTML**; performs only strategic decomposition.
    - Design Motivation: Decouples planning from element grounding—the Planner only needs to understand visual content and task semantics, without processing the bulky DOM.

2. **Programmatic Element Filter (Core Innovation)**:

    - Function: Generates scoring function parameters, scores DOM elements via a fixed template, and prunes to Top-N.
    - Mechanism (three steps):
        - **Step 1 — Rule-based Pre-filtering**: Retains only interactive tags (`<a>`, `<button>`, `<input>`, etc.) or elements with a `role` attribute; text from non-interactive elements is appended to the nearest interactive element as context.
        - **Step 2 — Scoring Function Generation**: The LLM generates a `keyword_weights` dictionary (keyword → weight 1–50) based on subtask $S_t$, which is plugged into the fixed scoring template.
        - **Step 3 — Template Execution**: Three-tier attribute matching (Tier 1: visible text → Tier 2: aria-label/placeholder → Tier 3: class/id) and four match types (exact > phrase > word > fuzzy, using `rapidfuzz` + `nltk.stem.PorterStemmer`); elements are ranked by weighted sum and Top-20 are retained.
    - Design Motivation: The LLM **generates parameters rather than programs**—this ensures controllability (weights fixed in range 1–50, keywords grounded in subtask semantics), execution efficiency (the template is fixed Python code), and robustness (no reliance on the LLM to produce correct code).

3. **Action Grounder**:

    - Function: Selects the correct element from the pruned Top-N candidates and generates the corresponding action.
    - Mechanism: Takes subtask $S_t$ and candidate list $C_t$ (~20 elements rather than thousands) as input, and outputs CLICK/TYPE/SCROLL operations.

### Loss & Training
- **Unified Two-Turn Dialogue**: A single model operates over two turns—Turn 1 = Plan + Filter (generate subtask + keyword weights); Turn 2 = Ground (select element + execute action).
- **SFT + RFT (GRPO)**: SFT is followed by GRPO with a hierarchical reward: $R = R_{\text{format}} + R_{\text{filtering}} + R_{\text{grounding}}$ (all binary 0/1), encouraging both correct filtering and grounding simultaneously.
- Data: ~5,503 steps re-annotated from Multimodal-Mind2Web.

## Key Experimental Results

### Main Results (Multimodal-Mind2Web)

| Method | Params | Cross-Task Step SR | Cross-Website | Cross-Domain |
|--------|--------|-------------------|---------------|--------------|
| GPT-4 | — | 32.3 | 27.0 | 29.7 |
| SeeAct (GPT-4V) | — | 40.2 | 32.4 | 36.8 |
| MindAct Flan-T5XL | — | 52.0 | 38.9 | 39.6 |
| ScribeAgent | 32B | 35.6 | 32.5 | 37.3 |
| SeeClick | 9.6B | 23.7 | 18.8 | 20.2 |
| **Prune4Web (Unified)** | **3B** | **52.4** | **44.9** | **46.1** |

### Low-Level Grounding Accuracy

| Method | Recall@20 | Grounding Acc |
|--------|-----------|---------------|
| Qwen2.5VL-3B (no pruning) | — | 46.80% |
| Prune4Web + Qwen2.5VL-3B | **97.46%** | **88.28%** |
| Oracle + GPT-4o | — | 82.83% |
| End-to-End GPT-4o | 85.56% | 70.84% |

### Ablation Study

| Training Strategy | Framework | Step SR |
|-------------------|-----------|---------|
| SFT Only | Separate | 37.9% |
| SFT + RFT | Separate | 42.2% |
| SFT Only | Two-turn Dialogue | 46.5% |
| **SFT + RFT** | **Two-turn Dialogue** | **52.4%** |

### Key Findings
- **3B model achieves SOTA-level performance**: 52.4% Step SR surpasses GPT-4 (32.3%), SeeAct (40.2%), and ScribeAgent-32B (35.6%), approaching MindAct (52.0%) with a substantially smaller model.
- **Grounding accuracy improves from 46.8% to 88.28%**: After pruning, the candidate set shrinks from thousands to 20 elements, enabling accurate grounding even with small models.
- **Programmatic filtering substantially outperforms direct LLM selection**: For Qwen2.5VL-3B, LLM Top-N selection yields 0% online completion rate, while programmatic filtering yields 5.2%; gains are also significant for GPT-4o-mini (26.3% → 31.6%).
- **A 0.5B grounder approaches 3B performance**: Prune4Web Filter + 0.5B Grounder achieves 41.3% vs. 42.2% for 3B, demonstrating that the task has been sufficiently simplified by pruning.
- **Two-turn dialogue substantially outperforms separate models**: 52.4% vs. 42.2%, highlighting the importance of information sharing between the Planner and Filter/Grounder.
- **Plug-and-play compatibility with UI-tars**: Integrating Prune4Web with UI-tars improves performance from 53.6% to 54.9%.

## Highlights & Insights
- The **"LLM generates parameters; fixed template executes"** design philosophy is the most elegant contribution—rather than having the LLM write programs (uncontrollable) or select elements from a large DOM (inaccurate), it instructs the template on "which keywords to use and how important they are," while the template handles reliable execution. This is an excellent example of LLM–classical method collaboration.
- The **three-tier, four-match-type scoring template** reflects careful engineering—it accounts for attribute priority (visible text > aria-label > class) and match precision (exact > phrase > word > fuzzy), with NLP tools (stemming, fuzzy matching) enhancing robustness.
- **97.5% Recall@20** means the correct element is almost never excluded, while candidates are reduced from thousands to 20—the ideal outcome for a pruning approach.

## Limitations & Future Work
- Planning remains the primary bottleneck—when the Planner produces incorrect or stagnant plans, downstream stages cannot compensate.
- When webpages use non-standard HTML (e.g., `<div>` elements simulating buttons), the rule-based pre-filtering step may miss interactive elements.
- Purely icon-based elements (lacking text or aria-labels) cannot be captured by the keyword matching mechanism.
- Online evaluation covers only 30 tasks, limiting the scale of empirical validation.
- Training data comprises only ~5,500 steps, which may not adequately cover all webpage types.

## Related Work & Insights
- **vs. SeeAct**: SeeAct uses GPT-4V to process the full DOM and screenshot end-to-end; Prune4Web prunes first and then processes—a 3B model surpasses GPT-4V-level baselines.
- **vs. MindAct**: MindAct employs a separate ranking model to filter the DOM; Prune4Web uses a programmatic approach—Recall@20 is comparable (97.15% vs. 97.55%), but Prune4Web offers a more unified framework.
- **Implications for Web Agent development**: The paradigm of "generating parameters rather than programs" generalizes to any scenario requiring LLM interaction with structured data (e.g., database queries, API call parameter configuration).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ "Programmatic DOM pruning" is a highly elegant design; the paradigm of LLM-generated parameters + template execution is broadly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers offline + online evaluation, low-level grounding, Recall@k, ablations, plug-and-play verification, and small-model experiments.
- Writing Quality: ⭐⭐⭐⭐ Method description is clear with sufficient detail on the scoring template, though the multi-layered paper structure requires careful reading.
- Value: ⭐⭐⭐⭐⭐ A 3B model achieving SOTA-level performance with 97.5% Recall@20 represents a practical engineering paradigm with direct reference value for the Web Agent community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] WebClipper: Efficient Evolution of Web Agents with Graph-based Trajectory Pruning](../../ACL2026/llm_agent/webclipper_efficient_evolution_of_web_agents_with_graph-based_trajectory_pruning.md)
- [\[ICLR 2026\] ToolTree: Efficient LLM Agent Tool Planning via Dual-Feedback Monte Carlo Tree Search and Bidirectional Pruning](../../ICLR2026/llm_agent/tooltree_efficient_llm_agent_tool_planning_via_dual-feedback_monte_carlo_tree_se.md)
- [\[ICLR 2026\] WebOperator: Action-Aware Tree Search for Autonomous Agents in Web Environment](../../ICLR2026/llm_agent/weboperator_action-aware_tree_search_for_autonomous_agents_in_web_environment.md)
- [\[AAAI 2026\] Promoting Sustainable Web Agents: Benchmarking and Estimating Energy Consumption Through Empirical and Theoretical Analysis](promoting_sustainable_web_agents_benchmarking_and_estimating_energy_consumption_.md)
- [\[CVPR 2026\] Ego2Web: A Web Agent Benchmark Grounded in Egocentric Videos](../../CVPR2026/llm_agent/ego2web_a_web_agent_benchmark_grounded_in_egocentric_videos.md)

</div>

<!-- RELATED:END -->
