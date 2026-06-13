---
title: >-
  [Paper Note] Generative Interfaces for Language Models
description: >-
  [ACL 2026][LLM/NLP][Generative Interfaces] This paper proposes **Generative Interfaces (GenUI)**, which enables LLMs to move beyond providing responses in a single chat box. Instead, it generates an on-demand…
tags:
  - "ACL 2026"
  - "LLM/NLP"
  - "Generative Interfaces"
  - "Adaptive UI"
  - "Finite State Machines"
  - "Iterative Refinement"
  - "Adaptive Reward"
date: 2026-05-08
content_hash: 3cc34ee4b4cf6ea9
---

# Generative Interfaces for Language Models

**Conference**: ACL 2026  
**arXiv**: [2508.19227](https://arxiv.org/abs/2508.19227)  
**Code**: https://github.com/SALT-NLP/GenUI  
**Area**: LLM Interaction / Human-Computer Interaction / UI Generation  
**Keywords**: Generative Interfaces, Adaptive UI, Finite State Machines, Iterative Refinement, Adaptive Reward

## TL;DR
This paper proposes **Generative Interfaces (GenUI)**, which enables LLMs to move beyond providing responses in a single chat box. Instead, it generates an on-demand, interactive web interface tailored to each query based on a structured intermediate representation—combining an "interaction flow graph + finite state machine"—and "adaptive reward-driven iterative refinement." On 100 UIX prompts, it achieved an **84%** overall preference win rate compared to the Claude 3.7 chat UI.

## Background & Motivation

**Background**: The current mainstream paradigm for LLM applications remains *Conversational UI* (ConvUI). Regardless of task complexity, outputs are crammed into a linear chat box as long blocks of text. While "canvas/artifact" tools like OpenAI Canvas and Claude Artifacts isolate code and documents into side windows, their UI components remain preset and fixed.

**Limitations of Prior Work**: In multi-turn, information-dense, and exploratory tasks (e.g., "I want to understand neural networks," "How to practice piano efficiently"), pure text output results in high cognitive load and lacks actionable interactions (e.g., illustrative animations, instant feedback, modular navigation). Users must repeatedly prompt to approximate the "tool" they actually need.

**Key Challenge**: While LLM capabilities allow for synthesizing entire web pages at once (e.g., Design2Code, Sketch2Code), the *interaction paradigm* between LLMs and users is still stuck in "textual dialogue." There is a paradigm-level asymmetry between strong generation capabilities and weak interaction forms.

**Goal**: Address two sub-problems: (I) How to generate a *functionally correct, interactive* interface for each query "on-demand" (infrastructure problem); (II) How to *systematically evaluate* whether generated interfaces truly improve user experience (evaluation protocol problem).

**Key Insight**: Directly generating HTML/JS code from LLMs results in an excessively large search space and poor controllability. The authors suggest introducing a *structured intermediate representation for interfaces*—where an interaction flow graph depicts "paths users take" and finite state machines (FSM) depict "how each component responds to events." LLMs populate a structured skeleton, which is then polished using task-adaptive reward functions through multiple refinement rounds.

**Core Idea**: Upgrade "generating answers" to "generating interfaces"—using "Interaction Flow + FSM" as the scaffolding and "query-specific rubrics" as the reward, performing a 5-round generate→evaluate→regenerate cycle to evolve the UI online for every query.

## Method

### Overall Architecture
Input: Natural language query $q$. Output: An HTML/CSS/JS interface $u^*$ renderable in a browser. The pipeline consists of five LLM-calling stages:

1. **Requirement specification**: Translates $q$ into a structured requirement specification (goals, features, UI components, interaction style, problem-solving strategy), acting as a bridge between natural language intent and formal design.
2. **Structured representation generation**: Based on the specification, generates an "interaction flow graph $\mathcal{G}=(\mathcal{V},\mathcal{T})$ + a set of FSMs $\mathcal{M}=(\mathcal{S},\mathcal{E},\delta,s_0)$."
3. **UI code synthesis**: Feeds the "query + specification + structured representation + preset component library (clocks, maps, calculators, video players, charts, etc.) + relevant UI examples retrieved via exa.ai" into the LLM to output and render executable HTML/CSS/JS.
4. **Adaptive reward function construction**: Uses an LLM to design a set of weighted, fine-grained evaluation dimensions (e.g., "Visual Structure / Explain Physics Concept / Clarity") specifically for the current query.
5. **Iterative refinement**: Samples multiple candidate UIs per round, scores them using the adaptive reward (0–100), selects the highest-scoring version as the seed for the next round, and includes the scoring rationale until the score $\geq 90$ or 5 rounds are reached.

The system is implemented on the OpenCanvas framework, using Claude 3.7 as the default backbone.

### Key Designs

1. **Structured Interface Representation (Interaction Flow + FSM)**:
    - **Function**: Serves as the "skeleton blueprint" before the LLM generates UI code, explicitly modeling "user paths" and "component behaviors" in layers.
    - **Mechanism**: The high-level *Interaction Flow* is a directed graph $\mathcal{G}=(\mathcal{V},\mathcal{T})$, where nodes $\mathcal{V}$ are UI views or sub-goals (e.g., Home View → Explore Tutorials → Run Simulation → Glossary Lookup) and edges $\mathcal{T}$ are transitions triggered by clicks or navigation. The low-level *FSM* models each UI component as $\mathcal{M}=(\mathcal{S},\mathcal{E},\delta,s_0)$, where $\mathcal{S}$ represents atomic states (e.g., `isModalOpen=true`), $\mathcal{E}$ represents user events, $\delta:\mathcal{S}\times\mathcal{E}\to\mathcal{S}$ is the transition function, and $s_0$ is the initial state.
    - **Design Motivation**: End-to-end generation of interactive interfaces by LLMs often leads to missing states or events due to the massive search space. By using FSMs to enforce constraints on "what state a component is in, what event it sees, and where it should transition," interaction correctness and interpretability are significantly improved. Ablations show that this structured representation improves the overall win rate from 13% to 17% compared to pure natural language descriptions (loss 78% vs 82%).

2. **Adaptive Reward**:
    - **Function**: Constructing a set of weighted rubrics on-the-fly for each query to provide a comprehensive score (0–100) for candidate UIs.
    - **Mechanism**: The LLM outputs evaluation dimensions, each containing four fields: `name / description / criteria / weight`. The final composite score is $R(u)=\sum_i w_i\cdot s_i(u)$. For example, for a "Understand Quantum Physics" query, it might automatically include "Interactive models effectively demonstrate phenomena like wave-particle duality" as an intent-sensitive criterion.
    - **Design Motivation**: Traditional general UI heuristics (usability, info organization, etc.) fail to distinguish the true requirements of different tasks. Query-specific rubrics provide "intent-aligned" reward signals. Ablations show that replacing adaptive with static rewards leads to a 17% drop in overall win rate and degradation across all evaluation dimensions.

3. **Iterative Refinement**:
    - **Function**: Polishing the UI from coarse to fine using rewards, similar to "LLM Best-of-N + feedback-based regeneration."
    - **Mechanism**: At round $t$, $K$ candidates $\{u^t_k\}$ are sampled. The candidate with the highest reward $u^t_*=\arg\max_k R(u^t_k)$ is selected. $u^t_*$ and its scoring rationale (weak points in specific dimensions) are fed into the prompt for round $t+1$, continuing until $R\geq 90$ or $t=5$.
    - **Design Motivation**: UI design is naturally an iterative process. One-shot generated code often suffers from crowded layouts, lack of onboarding, or unbalanced information density. Ablations show that iterative refinement consistently improves all 7 perceptual dimensions compared to one-shot generation, increasing the overall win rate by 14%.

### Loss & Training
The method itself is *training-free* and does not update any LLM weights. All "learning" occurs during the inference-time reward-guided self-refinement loop. Rewards are generated on-the-fly by an LLM, with stopping conditions set at $R\geq 90$ or 5 iterations. The backbone is Claude 3.7 (GPT-4o is also tested for ConvUI comparison).

## Key Experimental Results

### Main Results
Evaluation Protocol: 100 UIX prompts covering 10 domains × {Concise/Detailed} × {General Dialogue/Interaction Task}. 428 Prolific US native speakers performed pairwise comparisons (majority vote of three), Fleiss' $\kappa = 0.525$.

| Comparison | Dimension | GenUI Win | Tie | Opponent Win |
|------------|-----------|-----------|-----|--------------|
| GenUI vs ConvUI (Claude 3.7) | Overall | **84%** | 4% | 12% |
| GenUI vs ConvUI (Claude 3.7) | ASA (Aesthetics) | 89% | 8% | 3% |
| GenUI vs ConvUI (Claude 3.7) | IES (Satisfaction) | 87% | 7% | 6% |
| GenUI vs ConvUI (GPT-4o) | Overall | **69%** | 1% | 30% |
| GenUI vs IUI (Claude 3.7 + Artifact) | Overall | **75%** | 8% | 17% |

LLM-as-judge scores (0–100) indicate that GenUI improves Usability from 34.7 (ConvUI-Claude) to 87.0 (+151% relative gain) and Task Efficiency from 47.6 to 84.2 (+77%).

### Ablation Study

| Configuration | Representation | Generation | Reward | Overall Loss vs Full | Description |
|---------------|----------------|------------|--------|----------------------|-------------|
| Full GenUI | Structured | Iterative | Adaptive | — | Full version |
| w/o Adaptive Reward | Structured | Iterative | Static | 54% | Removed adaptive reward; win rate −17% |
| w/o Iterative | Structured | One-shot | Static | 78% | Single generation; win rate further −14% |
| w/o Structured | Natural Lang. | One-shot | Static | 82% | Fully degraded to natural language descriptions |

### Key Findings
- **Aesthetics and satisfaction are the strongest drivers**: Gains in ASA (+86%), IES (+112%), and Usability (+151%) were significantly higher than Learnability (+16%) and IC (+16%), suggesting GenUI wins primarily by "feeling like a tool" rather than just textual organization.
- **Significant domain variance**: GenUI leads by a large margin in structured information-heavy tasks like Data Analysis & Visualization (93.8%) and Business Strategy (87.5%), but only achieves 50% in formula-heavy Advanced AI/ML explanation scenarios, indicating pure text remains irreplaceable for some tasks.
- **Query type differences**: GenUI preference reached 80% for interactive queries, 80% for detailed queries, and 73% for concise queries; more complex queries benefit more.
- **Reward dimensions are more important than iteration count**: The cost of switching from adaptive to static rewards (−17%) was greater than switching from iterative to one-shot (−14%), indicating that "knowing what to score" is the bottleneck in reward-guided refinement.
- **Real-world user study replicates conclusions**: On 380 self-reported user queries, GenUI won 50.8% vs 41.1% lost; 30.3% of users strongly preferred GenUI (in $\geq 80$% of scenarios), compared to only 18.4% strongly preferring ConvUI.

## Highlights & Insights
- **Paradigm-level contribution**: Upgrading "generating responses" to "generating interfaces" is a significant redefinition of LLM application forms, opening new evaluation dimensions (Function/Interaction/Affect) for LLM × HCI research.
- **Value of structured intermediate representations**: Treating Interaction Flow as *interface-level CoT* and FSM as *component-level constraints* provides a "dual-layer skeleton + code LLM populating" paradigm that can be directly transferred to Agent UIs, educational courseware, and dashboards.
- **Adaptive rubrics over fixed metrics**: Constructing scoring items on-the-fly is essentially a lightweight implementation of *task-conditional reward modeling*, acting as a practical embodiment of RLAIF without training costs—serving as both an inference-time selector and a potential RL training signal.
- **"Professionalism" stems from presentation, not just content**: User surveys showed 86.5% chose GenUI as more "trustworthy and professional," despite many admitting the content was similar—providing empirical evidence that formatting significantly affects user trust even when content remains constant.

## Limitations & Future Work
- **Frontend only**: Supports only HTML/CSS/JS without backend logic; struggles with long-tail tasks requiring persistent data or complex flows. The two-tier representation might not be expressive enough for large applications.
- **Significant latency**: 5 rounds of iteration + multiple candidate sampling often take "minutes," making it unsuitable for real-time chat. The paper does not quantify latency distribution or token costs.
- **"UI generation for all queries"**: The authors admit some queries ("What is the weather in NY today?") do not need a GUI. A *router/classifier* is needed to determine when to trigger GenUI.
- **Risk of evaluator bias**: Agreement between LLM-as-judge and human evaluators was only 69%, with potential biases regarding length or visual complexity.
- **Accessibility and Ethics**: Highly graphical interfaces may be unfriendly to assistive technologies like screen readers. The "tool-like" feel might also lead to overtrust.

## Related Work & Insights
- **vs Claude Artifacts / OpenAI Canvas**: These provide "preset canvases" for code/doc outputs with fixed UI components; GenUI allows the *entire UI structure* to be dynamically generated by the LLM.
- **vs DynaVis / GenerativeGUI / ClarifyGPT**: These focus on "dynamic UI for specific scenarios" (chart editing, clarifying questions, code generation), whereas GenUI is a general-purpose query→UI framework.
- **vs Design2Code / Sketch2Code / WebSight**: Focus on visual-to-code conversion from screenshots/sketches; GenUI takes *natural language intent* as input without requiring UI sketches.
- **vs Graphologue**: Post-processes LLM responses into graphs for exploration; GenUI generates entire new interfaces end-to-end.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The paradigm of "answers as interfaces" + Flow/FSM representation is a remarkably complete approach.
- Experimental Thoroughness: ⭐⭐⭐⭐ 100-prompt controlled study + 380-query real user study, supported by ablations; lacks detailed latency/cost/failure analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear paradigm motivation, well-structured method and evaluation sections, with effective use of examples.
- Value: ⭐⭐⭐⭐⭐ Significant impact on both LLM product forms and HCI evaluation; the UIX benchmark and GenUI code are open-sourced for community reuse.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Generative Floor Plan Design with LLMs via Reinforcement Learning with Verifiable Rewards](generative_floor_plan_design_with_llms_via_reinforcement_learning_with_verifiabl.md)
- [\[ICLR 2026\] ConflictScope: Generative Value Conflicts Reveal LLM Priorities](../../ICLR2026/llm_nlp/quamo_quaternion_motions_for_vision-based_3d_human_kinematics_capture.md)
- [\[ACL 2026\] Repeated Sequences Reveal Gaps between Large Language Models and Natural Language](repeated_sequences_reveal_gaps_between_large_language_models_and_natural_languag.md)
- [\[ACL 2026\] Foresight Optimization for Strategic Reasoning in Large Language Models](foresight_optimization_for_strategic_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Adam's Law: Textual Frequency Law on Large Language Models](adam39s_law_textual_frequency_law_on_large_language_models.md)

</div>

<!-- RELATED:END -->
