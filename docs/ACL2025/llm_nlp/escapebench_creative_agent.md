---
title: >-
  [Paper Note] EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents
description: >-
  [ACL 2025 (Long Paper)][LLM (Other)][creative intelligence] This paper introduces EscapeBench, a benchmark for evaluating the creative intelligence of LLM Agents based on escape room games (36 scenarios, 3 difficulty levels). It reveals severe deficiencies in current models regarding creative tool use and implicit goal inference, and proposes EscapeAgent (incorporating Foresight + Reflection) to reduce prompt dependency by nearly 50%.
tags:
  - "ACL 2025 (Long Paper)"
  - "LLM (Other)"
  - "creative intelligence"
  - "escape room"
  - "agent benchmark"
  - "tool use"
  - "reasoning"
date: 2026-05-08
content_hash: 4ae6ab5f3a847af7
---

# EscapeBench: Towards Advancing Creative Intelligence of Language Model Agents

**Conference**: ACL 2025 (Long Paper)  
**arXiv**: [2412.13549](https://arxiv.org/abs/2412.13549)  
**Code**: [https://github.com/qiancheng0/EscapeBench](https://github.com/qiancheng0/EscapeBench)  
**Area**: LLM/NLP  
**Keywords**: creative intelligence, escape room, agent benchmark, tool use, reasoning

## TL;DR

This paper introduces EscapeBench, a benchmark for evaluating the creative intelligence of LLM Agents based on escape room games (36 scenarios, 3 difficulty levels). It reveals severe deficiencies in current models regarding creative tool use and implicit goal inference, and proposes EscapeAgent (incorporating Foresight + Reflection) to reduce prompt dependency by nearly 50%.

## Background & Motivation

**Background**: LLM Agents have made significant progress in long-horizon planning and reasoning, with numerous evaluation benchmarks emerging—ranging from web operations to scientific research, and from text-based games to Minecraft sandboxes. While these benchmarks primarily evaluate analytical intelligence (reasoning ability) and practical intelligence (knowledge application ability), the evaluation of creative intelligence is severely missing.

**Limitations of Prior Work**: The training of current Agents focuses on memorizing standard associations between tools and tasks, heavily neglecting the deep exploration of tool affordances and the ability to adapt in unstructured scenarios. When faced with scenarios requiring "thinking outside the box" (e.g., creatively using a wooden stick as a crowbar), models are often at a loss.

**Key Challenge**: Sternberg's Triarchic Theory of Intelligence categorizes intelligence into practical, analytical, and creative types. Existing benchmarks cover the first two, but creative intelligence—the ability to think innovatively and solve problems adaptively in novel environments—remains an evaluation blind spot, lacking a dedicated testing environment.

**Goal**: 1) Address the lack of dedicated benchmark environments to evaluate the creative intelligence of LLM Agents; 2) Quantify the performance of current models in creative tool usage and implicit goal discovery; 3) Find ways to enhance the creative reasoning capabilities of Agents.

**Key Insight**: Escape room games naturally demand creative tool use, implicit goal inference, and ultra-long reasoning chains, making them highly suitable as an evaluation scenario for creative intelligence. A single game may take a human about an hour to complete, and even an oracle Agent requires at least 100+ action steps and 40+ key steps.

**Core Idea**: Construct a creative intelligence evaluation benchmark using escape room games as the vehicle, and enhance Agent creativity through Foresight (prospective tool utilization hypotheses) and Reflection (dynamic task list management).

## Method

### Overall Architecture

EscapeBench is based on a custom game engine containing three core components: Scenes (connected in a graph structure), Items (interactive objects), and Tools (which can be collected, used, or crafted). Agents interact with the environment via five types of actions. EscapeAgent integrates Foresight and Reflection modules on top of a BaseAgent (which uses working memory + CoT reasoning).

### Key Designs

1. **Game Engine and Evaluation Scenario Design**:

    - **Function**: Simulate escape room environments that require creative reasoning.
    - **Mechanism**: Scenes are connected via a graph structure representing physical spatial connectivity; Items require tool application (`Apply`), input (`Input`), or clicking (`Click`) to trigger state changes or effects; Tools can be collected and then `Apply`-ed to Items or merged via `Craft` with other Tools to form new tools. Five action types are supported: `Move(Scene)`, `Click(Item)`, `Apply(Tool, Item)`, `Input(str, Item)`, and `Craft(Tool, Tool)`. Among these, `Apply` and `Craft` most heavily test creativity, requiring the Agent to use or synthesize tools innovatively.
    - **Design Motivation**: 36 hand-annotated, high-quality scenarios (3 difficulty levels $\times$ 12 scenarios per difficulty, with 3 pedagogical description clarity versions per scenario) ensure diversity and a controllable difficulty gradient. The Oracle optimal solution for each game averages 107.83 steps.

2. **Foresight Module (Prospective Reasoning)**:

    - **Function**: Enhance creative tool usage capability.
    - **Mechanism**: Activated under two conditions: when discovering a new task, the Agent hypothesizes potential tool usage strategies based on existing tools and evaluates their feasibility; when collecting a new tool, the Agent evaluates the applicability of this tool in solving existing tasks or the potential to synthesize it with other tools. If an executable hypothesis is proposed, it enters the "Try Action" status to try it directly; otherwise, it remains in the "Free Explore" mode to explore freely.
    - **Design Motivation**: Prompt the Agent to explicitly reason about unconventional uses of tools before acting, mimicking the human "hypothesize-then-verify" creative decision-making process, thus avoiding aimless trial-and-error.

3. **Reflection Module (Reflection Management)**:

    - **Function**: Manage implicit goals, prevent repetitive failures, and improve action efficiency.
    - **Mechanism**: Maintain a structured task list supporting three operations: `New` (add newly discovered unresolved tasks), `Update` (record failed attempts to avoid repetition), and `Delete` (remove completed tasks). Each record includes the task name, target item, and a list of failed actions. Updates are triggered based on environmental feedback after each non-`Move` action.
    - **Design Motivation**: In escape rooms, goals are implicit and discovered progressively. Agents need to actively manage a list of known and unknown subgoals to prevent wasting steps repeatedly on the same errors.

### Loss & Training

EscapeAgent is a training-free framework that relies on prompting and runtime inference. Core settings: BaseAgent uses CoT reasoning and a working memory of length 10; hints are automatically provided by the system after 50 consecutive steps without progress to ensure the game is eventually completable; sampling temperature $T=0$, $n=1$.

## Key Experimental Results

### Main Results

| Model | Hints Used ↓ | Total Steps ↓ | Early Exit ↑ | Key Steps Hints % ↓ |
|------|-------------|---------------|--------------|---------------------|
| GPT-4o | 10.30 | 723.61 | 24.75% | 24.27% |
| Claude-3.5-Sonnet | 8.97 | 690.31 | 28.95% | 22.44% |
| Llama-3.1-70B | 14.53 | 982.42 | 19.00% | 33.29% |
| Qwen2.5-72B | 16.50 | 1102.50 | 12.46% | 32.02% |
| Llama-3.1-8B | 25.86 | 1543.30 | 10.10% | 56.00% |
| Qwen2.5-7B | 32.20 | 1950.42 | 6.52% | 54.43% |
| **Human Average** | **4.33** | **257.83** | **59.65%** | **12.28%** |

### EscapeAgent Performance

| Model + Agent | Hints Used ↓ | Steps ↓ | Hints Decrease | Steps Decrease |
|-------------|-------------|---------|-----------|-----------|
| GPT-4o BaseAgent | 10.30 | 723.61 | — | — |
| GPT-4o **EscapeAgent** | **5.03** | **452.75** | **-5.27** | **-270.86** |
| Llama-70B BaseAgent | 14.53 | 982.42 | — | — |
| Llama-70B **EscapeAgent** | **7.92** | **645.19** | **-6.61** | **-337.23** |
| Qwen-72B BaseAgent | 16.50 | 1102.50 | — | — |
| Qwen-72B **EscapeAgent** | **9.72** | **746.61** | **-6.78** | **-355.89** |

### Key Findings

- **Creative intelligence is severely lacking**: Even the best model Claude-3.5-Sonnet requires around 9 hints to complete a game (humans require only about 4), takes 6-7 times more action steps than the Oracle optimal solution, and achieves an Early Exit of only 29% (humans achieve 60%).
- **`Apply` and `Craft` are the biggest bottlenecks**: Analysis of hint demand distribution reveals that key steps involving `Apply` (creative tool use) and `Craft` (tool synthesis) have the highest dependency on hints, highlighting them as the models' weakest capability dimensions.
- **EscapeAgent is comprehensively effective**: Foresight + Reflection significantly reduces hints and steps across all evaluated models. GPT-4o drops from 10.30 to 5.03 hints ($\downarrow$ 51%), and increases Early Exit Progress from 24.75% to 47.03%.
- Model scale is positively correlated with creative capability, but the performance gap is much wider than in analytical reasoning tasks—even the largest closed-source models remain far behind human-level performance.
- Some smaller models (e.g., DeepSeek-67B, Yi-34B) show a slight increase in Tool Hints when using EscapeAgent, indicating that Foresight may introduce noisy hypotheses in weaker models.

## Highlights & Insights

- **First Agent benchmark focused on creative intelligence**: Fills the evaluation gap for the creative dimension in Sternberg's Triarchic Theory of Intelligence. Escape room games are a very clever vehicle—naturally encompassing the three core challenges of creative tool use, implicit goal discovery, and long-horizon reasoning chains.
- **Foresight's "hypothesize-before-acting" paradigm**: Externalizes creative thinking into a verifiable hypothesis-action loop, essentially acting as hypothesis-driven reasoning. This methodology can be transferred to open-world game AI, creative tool use in robotic manipulation, etc.
- **Granular evaluation metric system**: Multi-dimensional metrics such as hints used, total steps, early exit progress, and tool hints / key steps hints allow for fine-grained characterization of performance across different dimensions of creative reasoning.

## Limitations & Future Work

- Although the 36 scenarios are of high quality due to manual annotation, the overall scale is limited and may not cover all dimensions of creative intelligence.
- Currently, the interaction is purely text-based. Introducing visual information (such as item appearance, materials, and textures) could further benefit reasoning about creative tool usage.
- The Foresight module of EscapeAgent still relies on the LLM's intrinsic reasoning capabilities; it may still fail in highly creative scenarios that lie completely outside the training distribution.
- While escape room games are engaging, they remain game scenarios; readability and transferability to real-world creative problem-solving remain to be validated.
- The models' extreme vulnerability in `Apply` and `Craft` actions suggests a need to fundamentally shift the training paradigm of tool usage; prompting alone may not suffice.

## Related Work & Insights

- **vs TextWorld/Zork**: Traditional text games evaluate language understanding and planning capabilities but feature relatively simple scenarios with explicit goals; EscapeBench's implicit goals and creative tool usage demand higher-order cognitive capabilities.
- **vs Minecraft Benchmarks**: Minecraft tests spatial reasoning and planning, but tool usage usually follows standard crafting recipes; EscapeBench requires unconventional tool use (e.g., using a stick as a crowbar).
- **vs AUT/TTCT Creativity Psychometric Tests**: Psychological creativity tests evaluate divergent thinking but are detached from task execution; EscapeBench evaluates creativity within real interaction tasks, thereby demonstrating higher ecological validity.
- The core question inspired by this work: Is the current bottleneck of Agent "creativity" caused by model capability or framework design? The significant improvements of EscapeAgent suggest that framework design is crucial, but the huge gap with humans indicates that model capabilities themselves also require fundamental improvements.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Pioneering work proposing the evaluation of creative intelligence with a complete environment, evaluation paradigm, and method suite.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive experiments including a comparison of 12 models, a human baseline, full ablation of EscapeAgent across 8 models, and error analyses.
- **Writing Quality**: ⭐⭐⭐⭐ Deeply argued motivations; references to the Triarchic Theory of Intelligence are precise and persuasive.
- **Value**: ⭐⭐⭐⭐⭐ Opens up a new direction for the evaluation of Agent creative intelligence, with both the benchmark and agent framework holding long-term influence.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] AgentGym: Evolving Large Language Model-based Agents across Diverse Environments](agentgym_evaluating_and_training_large_language_model-based_agents_across_divers.md)
- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [\[NeurIPS 2025\] Writing in Symbiosis: Mapping Human Creative Agency in the AI Era](../../NeurIPS2025/llm_nlp/writing_in_symbiosis_mapping_human_creative_agency_in_the_ai_era.md)
- [\[ICML 2026\] Creative Collision: Directorial Persona Steering and Competition in Large Language Models](../../ICML2026/llm_nlp/creative_collision_directorial_persona_steering_and_competition_in_large_languag.md)
- [\[ACL 2025\] A Survey of LLM-based Agents in Medicine: How Far Are We from Baymax?](a_survey_of_llm-based_agents_in_medicine_how_far_are_we_from_baymax.md)

</div>

<!-- RELATED:END -->
