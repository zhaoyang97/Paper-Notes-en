---
title: >-
  [Paper Note] OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation
description: >-
  [ACL 2026][LLM Agent][User behavior simulation] OPeRA is a user behavior dataset collected from real Amazon shopping sessions, aligning personas, web observations, fine-grained actions…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "User behavior simulation"
  - "online shopping"
  - "Persona"
  - "Rationale"
  - "Web Agent evaluation"
date: 2026-05-08
content_hash: b86274ddac8ac42a
---

# OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation

**Conference**: ACL 2026  
**arXiv**: [2506.05606](https://arxiv.org/abs/2506.05606)  
**Code**: No public code; Dataset: [HuggingFace](https://huggingface.co/datasets/NEU-HAI/OPeRA)  
**Area**: LLM Agent / User Behavior Simulation / Dataset  
**Keywords**: User behavior simulation, online shopping, Persona, Rationale, Web Agent evaluation

## TL;DR

OPeRA is a user behavior dataset collected from real Amazon shopping sessions, aligning personas, web observations, fine-grained actions, and real-time rationales on a single timeline to evaluate whether LLMs can truly simulate a specific user's next shopping action.

## Background & Motivation

**Background**: LLM agents are already capable of performing tasks such as searching, navigating, purchasing, and form-filling in web environments. They are also frequently used as "user proxies" in UI/UX testing, social science experiments, and recommender system evaluations.

**Limitations of Prior Work**: Many existing evaluations only measure task completion or aggregate outcomes like final surveys, purchases, or clicks. These metrics fail to address a more granular question: does the model observe the page, weigh preferences, explain rationales, and take actions at each step as a specific real user would?

**Key Challenge**: Task-oriented web agents pursue the shortest path and high success rates, whereas real user shopping behavior often involves non-linear paths, such as repeated comparisons, reading reviews, modifying search terms, or abandoning purchases. Evaluating "human-likeness" requires recording not just actions, but also the page context, long-term user preferences, and immediate rationales when actions occur.

**Goal**: Construct a public dataset and benchmark allowing researchers to evaluate the ability of LLMs to simulate a specific user's next action on the same set of real trajectories.

**Key Insight**: The authors chose online shopping as the starting point because shopping inherently involves individual preferences, budget constraints, trade-offs between brand/price/reviews, multi-step page interactions, and clear session outcomes, making it an ideal testbed for personalized behavior simulation.

**Core Idea**: Use a browser extension to synchronously collect Observation, Persona, Rationale, and Action during natural shopping by real users, allowing "what the user did" and "why they did it" to enter the behavior simulation evaluation of LLM agents together.

## Method

### Overall Architecture

The overall pipeline of OPeRA is divided into three layers.

The first layer is user-side collection. Participants first complete a persona questionnaire and can opt for semi-structured interviews. Subsequently, they install the ShoppingFlow Chrome extension and shop on Amazon as usual for four weeks.

The second layer is trajectory-side collection. The extension records actions, timestamps, target elements, web HTML, screenshots, and page metadata while users browse, search, click, input, scroll, and switch pages. It also triggers pop-up questions with a certain probability to collect real-time rationales for the current action.

The third layer is benchmark construction. The authors perform privacy cleaning, session segmentation, action filtering, and action space abstraction on the raw continuous logs to obtain OPeRA-full and the more evaluation-friendly OPeRA-filtered. OPeRA-test is then sampled from the filtered version for next-action prediction.

Ultimately, each shopping session is treated as a time series: given user persona, historical actions, historical rationales, and current web observation, the model must predict the user's next action.

### Key Designs

1.  **OPeRA Quadruple Data Structure**:
    *   **Function**: Unified the four types of signals required for behavior simulation into a single user trajectory.
    *   **Mechanism**: Each user has a persona including demographics, shopping habits, consumer style, Big Five, and MBTI; each session has a chronologically ordered action trace; some actions include user-provided rationales; each step is paired with web observations including full HTML, simplified HTML, screenshots, and product metadata.
    *   **Design Motivation**: Previous datasets often only recorded clicks or purchase outcomes, lacking "what was seen on the page" and "why the user did this." The quadruple design allows the model to see both the external environment and the user's long-term preferences and local decision rationales.

2.  **ShoppingFlow Browser Extension Collection**:
    *   **Function**: Recorded fine-grained actions and context with low interference during real shopping.
    *   **Mechanism**: Content Scripts captured click, input, and scroll interactions within Amazon pages, recording timestamps, target elements, and HTML. Background Scripts handled page-level events, navigation, and uploads. Rationale pop-ups were triggered with approximately 8% probability after key interactions, asking users why they performed the action.
    *   **Design Motivation**: Asking annotators for rationales after the fact leads to recall bias, while asking users to explain every step disrupts the natural shopping process. Randomized real-time pop-ups represent a trade-off between data density and natural behavior.

3.  **Privacy Scrubbing, Session Segmentation, and Action Abstraction**:
    *   **Function**: Trnasformed raw browsing logs into publicly shareable, modelable, and evaluable data.
    *   **Mechanism**: The system does not record sensitive pages (login, account, checkout) and uses rule-based scripts to mask usernames, zip codes, addresses, workplaces, and payment info. Continuous behavior streams were first segmented by time intervals and then further divided into sessions by purchase intent events. Actions with fewer than 5 steps, non-interactive clicks, rare pages, and Amazon Rufus-related actions were filtered. For evaluation, the action space was compressed into input, click, and terminate, with clicks further categorized into semantic types like review, search, product_option, product_link, and purchase.
    *   **Design Motivation**: Raw web logs have high noise and privacy risks; direct release is neither feasible nor evaluable. Post-processing converts data from "raw browsing records" into a "reproducible behavior simulation benchmark."

### Loss & Training

This paper does not train new models but establishes zero-shot prompt-based evaluations.

The next-action prediction task can be written as $a_t = F_{action}(a_{1...t-1}, r_{1...t-1}, o_{1...t}, P_i)$, where $P_i$ is the user persona, $o$ is the web observation, $r$ is the sparse rationale, and $a$ represents historical actions.

Model outputs must follow a strict JSON format for the next action. For clicks, the target must be provided; for inputs, the field and text; and for terminate, an indication that the user decided to end the session.

Evaluation uses exact match and multi-level F1 scores: exact match for full action generation, macro/weighted F1 for high-level action types, weighted F1 for click sub-types, and session outcome evaluation for predicting purchase or termination.

## Key Experimental Results

### Main Results

OPeRA-full contains 692 shopping sessions, 28,904 action-observation pairs, and 604 human rationales contributed by 51 real users.

After cleaning and action abstraction, OPeRA-filtered contains 527 sessions, 5,856 action-observation pairs, and 207 rationales.

Experiments sampled 15 users and 90 sessions from OPeRA-filtered to construct OPeRA-test, evaluating GPT-4.1, DeepSeek-R1, Claude-3.7-Sonnet, and Llama-3.3-70B-Instruct.

| Model | Next Action Acc. | Action Type Macro F1 | Click Type Weighted F1 | Outcome Weighted F1 |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4.1 | 21.51 | 48.78 | 44.47 | 47.54 |
| DeepSeek-R1 | 14.75 | 27.37 | 35.12 | 46.36 |
| Claude-3.7-Sonnet | 10.75 | 31.58 | 27.27 | 43.52 |
| Llama-3.3-70B-Instruct | 8.31 | 24.29 | 19.99 | 36.64 |

The main results indicate that even for the strongest model, GPT-4.1, the exact match for the complete next action is only about 21.5%. This is not a task where switching models nearly solves it; rather, it exposes the difficulty of real user simulation at the level of fine-grained interaction.

### Ablation Study

The authors primarily examined the impact of persona and historical rationales on model behavior.

| Model / Configuration | Next Action Acc. | Action Type Macro F1 | Click Type Weighted F1 | Outcome Weighted F1 |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4.1 full | 21.51 | 48.78 | 44.47 | 47.54 |
| GPT-4.1 w/o persona | 22.06 | 45.55 | 43.45 | 58.47 |
| GPT-4.1 w/o rationale | 21.28 | 34.93 | 42.63 | 51.17 |
| DeepSeek-R1 full | 14.75 | 27.37 | 35.12 | 46.36 |
| DeepSeek-R1 w/o rationale | 15.74 | 27.16 | 32.65 | 47.92 |
| Claude-3.7 full | 10.75 | 31.58 | 27.27 | 43.52 |
| Claude-3.7 w/o rationale | 10.08 | 26.06 | 20.29 | 43.10 |
| Llama-3.3 full | 8.31 | 24.29 | 19.99 | 36.64 |
| Llama-3.3 w/o rationale | 8.76 | 23.60 | 19.22 | 34.19 |

The effect of persona is not always reflected in exact match. It acts more as a prior for "how this user usually shops," helping more with action type and click type classification; however, if the model cannot integrate the persona with the current page state, the extra information can become noise.

The role of rationale is more stable. Removing historical rationales caused most models to drop in action type, click type, or outcome performance, suggesting that intermediate explanations of "why" indeed help the model align with the intent of the current session.

### Key Findings

*   The largest source of error is incorrect click targets: models usually know a user is likely to click but struggle to select the specific button, product, filter, or review entry a real user would choose on a complex page.
*   Termination is rarely predicted by models; specifically, Claude and Llama almost never output termination actions in Table 7. This suggests current LLM agents may have an optimization bias toward "completing the shopping task," whereas real users often leave due to dissatisfaction, hesitation, or insufficient information.
*   Inputs are also difficult, especially the exact text of search queries. Search terms in shopping are not merely semantic paraphrases but behaviors resulting from user goals, brand preferences, budgets, and page feedback.
*   The value of OPeRA lies not in providing a high-score leaderboard, but in proving that existing LLMs remain weak at step-level personalized simulation of real users.

## Highlights & Insights

*   **Rationale collection shifted to the time of action**: This is closer to the user's true mental state than post-hoc annotation and allows models to learn the "reason behind the action" rather than just fitting click sequences.
*   **The quadruple design is highly suitable as an agent evaluation foundation**: Observation corresponds to the environment, Persona to long-term user state, Rationale to local intent, and Action to verifiable output; the four exactly cover the major variables of behavior simulation.
*   **Error analysis reveals the divergence between web agents and human simulation**: Being able to complete a task is not equivalent to being able to simulate a person. Real users browse aimlessly, hesitate, read reviews, change search terms, and exit early—behaviors frequently ignored by task-oriented agents.
*   **Dataset connects recommendation, UX, and agent research**: The same trajectory can be used for next-action prediction, personalized recommendation, web design evaluation, synthetic user trajectory generation, and digital twin modeling.

## Limitations & Future Work

*   The data domain is concentrated on Amazon online shopping, and users are primarily from English-speaking populations meeting recruitment criteria. Cross-culture, cross-platform, and mobile generalization still need verification.
*   To enable evaluation, OPeRA-filtered omits raw actions like scroll, navigate, and tab activate, reducing action space complexity at the cost of some continuity in real web browsing.
*   Although screenshots were collected, the experiments did not utilize visual information; for e-commerce pages, visual layout, images, price positioning, and review presentation can all influence click decisions.
*   Rationales are sparsely sampled, and the pop-ups themselves may slightly interfere with user behavior; future work could explore more natural ways to collect reasons, such as short post-shopping playback interviews or multimodal behavior replays.
*   There are slight inconsistencies in the number of test actions and instance counts in table captions between the main text and appendix in the current version.

## Related Work & Insights

*   **vs Amazon Review / Amazon-M2 / Taobao datasets**: These datasets are larger and suitable for recommendation/purchase prediction but lack step-by-step web observations, real-time rationales, and fine-grained personas. OPeRA is much smaller but has higher behavioral semantic density.
*   **vs Mind2Web / WebArena / WebShop benchmarks**: These emphasize task completion or instruction following, with trajectories often produced by annotators or synthetic tasks. OPeRA emphasizes natural real-user behavior, targeting "simulating what this person will do next."
*   **vs Role-playing agents / Social simulation**: Many works use persona prompts to generate plausible group behaviors but lack step-by-step real trajectory validation. OPeRA provides supervision signals aligned with specific users and page states.
*   **Insight**: To train more human-like shopping agents, merely providing a persona prompt is insufficient. It may be necessary to include rationales as intermediate supervision, incorporate termination and hesitation into rewards, and model behavior using both web structure and visual information.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ Systematically integrates Observation, Persona, Rationale, and Action from real shopping into a public benchmark for the first time; problem definition is highly valuable.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Covers four strong models, persona/rationale ablation, and error analysis, though tasks are limited to zero-shot evaluation without exploring trained methods or multimodal inputs.
*   Writing Quality: ⭐⭐⭐⭐ Data construction pipeline is clear and tables are helpful, despite minor inconsistencies in instance counts.
*   Value: ⭐⭐⭐⭐⭐ Directly relevant to LLM agents, personalized recommendation, automated UX evaluation, and user digital twins.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] StructMem: Structured Memory for Long-Horizon Behavior in LLMs](structmem_structured_memory_for_long-horizon_behavior_in_llms.md)
- [\[ICML 2026\] MCP-Persona: Evaluating LLM Agent Capabilities in Real Personalized Applications via Environment Simulation](../../ICML2026/llm_agent/mcp-persona_benchmarking_llm_agents_on_real-world_personal_applications_via_envi.md)
- [\[ACL 2026\] CodeStruct: Code Agents over Structured Action Spaces](codestruct_code_agents_over_structured_action_spaces.md)
- [\[ACL 2026\] HAG: Hierarchical Demographic Tree-based Agent Generation for Topic-Adaptive Simulation](hag_hierarchical_demographic_tree-based_agent_generation_for_topic-adaptive_simu.md)
- [\[ACL 2026\] YIELD: A Large-Scale Dataset and Evaluation Framework for Information Elicitation Agents](yield_a_large-scale_dataset_and_evaluation_framework_for_information_elicitation.md)

</div>

<!-- RELATED:END -->
