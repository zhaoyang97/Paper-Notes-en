---
title: >-
  [Paper Note] Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History
description: >-
  [ICML 2026][LLM Agent][Personalized web agent] This paper introduces Persona2Web, the first open-web benchmark for personalized web agents. It utilizes "implicit user history + three levels of ambiguous queries + reasoni…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Personalized web agent"
  - "user history"
  - "ambiguous query"
  - "clarify-to-personalize"
  - "reasoning-aware evaluation"
date: 2026-05-08
content_hash: ad6a60aaeebe6358
---

# Persona2Web: Benchmarking Personalized Web Agents for Contextual Reasoning with User History

**Conference**: ICML 2026  
**arXiv**: [2602.17003](https://arxiv.org/abs/2602.17003)  
**Code**: To be announced (Labeled [CODE] in the paper)  
**Area**: LLM Agent / Web Agent / Personalized Benchmark  
**Keywords**: Personalized web agent, user history, ambiguous query, clarify-to-personalize, reasoning-aware evaluation

## TL;DR
This paper introduces Persona2Web, the first open-web benchmark for personalized web agents. It utilizes "implicit user history + three levels of ambiguous queries + reasoning-aware scoring" to compel agents to infer user preferences from browsing records to disambiguate queries. Across five mainstream models including GPT-4.1 and o3, the success rate for Level-2 queries is only 13% even when history is provided, revealing a significant lack of true personalization capabilities in current web agents.

## Background & Motivation
**Background**: LLM-driven web agents (WebArena, Mind2Web, WebVoyager, AssistantBench, etc.) can execute multi-step tasks in browsers. However, most mainstream benchmarks assume users provide complete, explicit queries (e.g., "Search for female doctors near Southside Jacksonville on zocdoc.com").

**Limitations of Prior Work**: Real-world users rarely specify every detail, assuming the system knows their "frequently used websites" or "preferred doctor gender." Current agents facing such ambiguous instructions often fill in missing fields randomly or ask clarifying questions and give up, failing to achieve disambiguation. Existing personalization benchmarks cover only dialogue (Apollonion, LongMemEval) or abstract function calls (PersonalWAB), lacking real web interaction.

**Key Challenge**: Ambiguity is the fundamental prerequisite for personalization, yet existing web agent benchmarks eliminate it entirely, while personalized LLM benchmarks are detached from web environments. The intersection of these two domains is currently empty.

**Goal**: To construct a benchmark capable of evaluating personalization in real open-web environments, requiring (i) user history following realistic distributions, (ii) ambiguous queries that strictly require history for resolution, and (iii) fine-grained evaluation to distinguish "personalization failure" from "navigation failure."

**Key Insight**: The authors redefine personalization as **clarify-to-personalize**—the agent must rely on user history to clarify missing parts of the query rather than relying on the user to specify all constraints in the prompt. Thus, the effective usage of history is directly reflected in the ability to complete ambiguous fields, bypassing the illusion of "instruction following as personalization" created by explicit preference statements.

**Core Idea**: By using three levels of query ambiguity (Level 0/1/2) combined with implicit browsing history and reasoning-aware scoring, personalization capability is decoupled into three independently measurable sub-capabilities: "retrieval of history," "utilization of history," and "completion of navigation."

## Method

### Overall Architecture
The Persona2Web pipeline consists of two main tracks: **Data Construction** (User Profile → Event Seeds → Action Decomposition → History Aggregation → Ambiguous Query Derivation) and **Evaluation Framework** (a planner-retriever-generator personalization module atop AgentOccam/Browser-Use, paired with GPT-5-mini as a reasoning-aware judge). The final dataset contains 50 users, 102,568 history records, and 150 tasks (each with 3 ambiguity levels), covering 21 sub-domains and 105 real websites. Agents are evaluated through a cross-comparison of 5 backbones, 2 base architectures, and 3 history access schemes (no-history / on-demand / pre-execution).

### Key Designs

1.  **Implicit User History (Realistic User History via multi-stage generation)**:
    *   **Function**: Encodes user preferences into year-long, multi-domain implicit browsing logs containing over 2,000 records, forcing agents to induce preferences from behavioral patterns rather than reading declarations.
    *   **Mechanism**: A four-stage pipeline: first, GPT-4o generates demographics and selects $K$ domains with rationales; next, domain preferences are generated as $\mathrm{Pref}(u)=\{\mathcal{M}(d^{(k)},\pi^{(k)},\rho^{(k)}, \mathrm{Dem}(u))\}_{k=1}^K$; then, high/low-frequency event seeds $(\mathcal{E}_u^{\mathrm{HF}},\mathcal{E}_u^{\mathrm{LF}})$ are derived; each event is decomposed into $(a_{i,1},\ldots,a_{i,L_i})=\mathcal{M}(E_i)$ and scattered across a year-long timeline, with noise (cancellations/modifications) injected into ~10% of records. Each record contains only timestamp, type, object, and website. Crucially, preference values appear as exact strings in the history only **3.56%** of the time, forcing induction over matching.
    *   **Design Motivation**: Random event generation lacks preference consistency, while copying real logs lacks controllable "preference grounding." Event seeds provide anchors for traceability, while temporal scattering and noise injection prevent simple clustering rules from cracking the history, requiring agents to integrate patterns across long-term history.

2.  **Three Levels of Ambiguous Queries (Clarify-to-Personalize Query Set)**:
    *   **Function**: Derives three ambiguity levels for the same task to serve as a control group for determining if user history is truly necessary.
    *   **Mechanism**: Starting from a fully explicit Level 0 query (containing both website and preference constraints), fields are systematically masked. Level 1 masks the website keyword ("in my preferred website"), and Level 2 masks both the website and preference keywords ("in my usual area that match my preferred provider gender"). Level 2 is the primary target for evaluation—the agent must infer that zocdoc.com is the usual website and the user prefers female doctors.
    *   **Design Motivation**: Conventional benchmarks assume all queries are Level 0, making it impossible to distinguish between instruction following and history utilization. Through these three levels, the performance decay of an agent from Level 0 to Level 2 quantifies its reliance on explicit cues.

3.  **Reasoning-aware Evaluation**:
    *   **Function**: Uses GPT-5-mini as an LLM judge to read the full trajectory (actions + reasoning traces), splitting the score into $\mathcal{P}_{\text{web}}$, $\mathcal{P}_{\text{pref}}$, Intent, and SR (Success Rate) to distinguish "retrieval failure" from "utilization failure."
    *   **Mechanism**: Each personalization metric $\mathcal{P}_*$ is split into retrieval accuracy (whether correct history entries were accessed) and utilization accuracy (whether the extracted information was used in action planning). Intent satisfaction independently assesses task correctness, providing partial credit for correct plans interrupted by website errors. SR is counted as success only when both PS and Intent are perfect.
    *   **Design Motivation**: On the open web, multiple valid trajectories may exist for the same goal, causing action-wise scoring to penalize correct paths. Conversely, outcome-only scoring ignores intermediate reasoning. Meta-evaluation shows that reasoning-aware scoring achieves significantly higher Spearman correlation (0.72) and accuracy (0.88) on Preference metrics compared to action-wise or outcome-based methods.

## Key Experimental Results

### Main Results
Core results under the Browser-Use architecture for Level 2 queries:

| Backbone | No-history SR | On-demand $\mathcal{P}_{\text{avg}}$ | On-demand SR | Pre-exec $\mathcal{P}_{\text{avg}}$ | Pre-exec SR |
|----------|--------------|---------------------------------------|--------------|--------------------------------------|-------------|
| o3 | 0.00 | 0.655 | **0.13** | 0.671 | 0.10 |
| GPT-4.1 | 0.00 | **0.767** | **0.13** | 0.727 | **0.13** |
| Gemini-2.5-Flash | 0.00 | 0.597 | 0.02 | 0.659 | 0.03 |
| Qwen3-80B-Inst. | 0.00 | 0.674 | 0.03 | 0.720 | 0.03 |
| Llama-3.3-70B | 0.00 | 0.612 | 0.01 | 0.680 | 0.02 |

Without history, all models have a 0% success rate. Even with history, the strongest configuration (GPT-4.1 + Browser-Use) only reaches 13%, demonstrating that simply providing history to agents is insufficient.

### Ablation Study

| Experiment | Comparison | Key Finding |
|------|------|----------|
| Ambiguity Level (Level 0/1/2) | Avg SR: 23.8% → 16.3% → 7.8% | Even with history, agents fail as ambiguity increases across levels. |
| Implicit History vs. Explicit Profile | $\mathcal{P}_{\text{pref}}$ from 0.731 → 0.887, SR from 0.13 → 0.25 | Explicit statements inflate scores, proving implicit encoding tests induction. |
| Meta-evaluation vs. Human | Preference (Spearman): 0.72 (Ours) vs 0.40 (Action-wise) | Reasoning traces are necessary for reliable personalization scoring. |
| Cross-architecture | AgentOccam vs. Browser-Use | Browser-Use achieves higher $\mathcal{P}$ due to full DOM tree observation. |
| Repeated Execution | 3 runs, Browser-Use | Max std = 0.025, confirming benchmark stability on the open web. |

### Key Findings
*   **Task completion and personalization are decoupled**: Llama-3.3-70B with pre-execution achieved $\mathcal{P}_{\text{avg}}=0.680$ but an Intent score of 0.197 (correct personalization, failed navigation). Gemini-2.5-Flash with on-demand reached Intent=0.403 but $\mathcal{P}_{\text{avg}}=0.486$ (correct navigation, failed personalization).
*   **Backbones and schemes show specific alignments**: GPT-4.1 is stronger in on-demand settings (situational awareness), while Qwen3 and Llama-3.3 perform better in pre-execution (long-horizon planning).
*   **Clear queries still suffer personalization failure**: On Level 0, $\mathcal{P}_{\text{pref}}$ averaged only 0.92, failing due to partial constraint usage or not knowing when/where to input information, highlighting "utilization" as a distinct bottleneck.

## Highlights & Insights
*   **"Clarify-to-personalize" is a paradigm-level trick for benchmark design**: Instead of defining "good personalization," the authors create an environment where agents must personalize to succeed. This principle is transferable to other latent capability evaluations like Theory of Mind or long-context memory.
*   **Split rubrics for retrieval/utilization** provide an actionable attribution framework for agents, similar to the retriever vs. generator debate in the RAG community.
*   **Explicit vs. Implicit Contrast**: The significant score gap (~10-20 points) when the same information is merely presented differently proves that the benchmark's difficulty effectively stems from its encoding method.
*   **Temporal Drift Testing**: Systematic SR drops of ~10% over three months (due to website updates) address the inherent fragility of open-web benchmarks with honesty and transparency.

## Limitations & Future Work
*   User history is synthesized by GPT-4o, potentially inheriting LLM bias; alignment with real anonymous browsing datasets would increase credibility.
*   Domains and websites are biased toward English-language services; robustness across cultures and languages remains unknown.
*   The reliance on GPT-5-mini as a judge may introduce bias; future work could involve multi-judge voting or open-source judges.
*   The low 13% SR suggests a massive gap; the paper does not propose personalized training solutions (e.g., SFT on history), leaving a clear path for future research.

## Related Work & Insights
*   **vs. PersonalWAB**: While both use user history, PersonalWAB abstracts interactions into function calls. Persona2Web requires agents to navigate real DOMs, strictly distinguishing navigation failure from personalization failure.
*   **vs. WebVoyager/AssistantBench**: These benchmarks use explicit queries (equivalent to Level 0). Persona2Web introduces Level 1/2 and user history as an orthogonal dimension.
*   **vs. LongMemEval/Apollonion**: These focus on personalization in dialogue or text generation. Persona2Web extends these capabilities to stateful, real-world web environments.

## Rating
*   **Novelty**: ⭐⭐⭐⭐⭐ First personalized agent benchmark on the open web; "clarify-to-personalize" is a clear, transferable design principle.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Cross-evaluation of backbones, architectures, schemes, and ambiguity levels, supplemented by robust ablation and drift analysis.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear structure and compact tables; the methodology section is slightly dense with notations.
*   **Value**: ⭐⭐⭐⭐⭐ Reveals a massive performance gap (13% SR), likely driving research into personalized training and retrieval for web agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)
- [\[ICML 2026\] Scaling, Benchmarking, and Reasoning of Vision-Language Agents for Mobile GUI Navigation](scaling_benchmarking_and_reasoning_of_vision-language_agents_for_mobile_gui_navi.md)
- [\[ICLR 2026\] Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents](../../ICLR2026/llm_agent/web-cogreasoner_towards_knowledge-induced_cognitive_reasoning_for_web_agents.md)
- [\[ICML 2026\] Process Reward Agents for Steering Knowledge-Intensive Reasoning](process_reward_agents_for_steering_knowledge-intensive_reasoning.md)
- [\[ICML 2026\] MCP-Persona: Evaluating LLM Agent Capabilities in Real Personalized Applications via Environment Simulation](mcp-persona_benchmarking_llm_agents_on_real-world_personal_applications_via_envi.md)

</div>

<!-- RELATED:END -->
