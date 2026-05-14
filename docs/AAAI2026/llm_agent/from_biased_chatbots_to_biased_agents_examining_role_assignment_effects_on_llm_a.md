---
title: >-
  [Paper Note] From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness
description: >-
  [AAAI 2026][LLM Agent][implicit bias] The first systematic case study demonstrating that demographically grounded persona assignment causes up to 26.2% performance degradation in LLM agent task execution across 5 operati…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "implicit bias"
  - "persona"
  - "demographic bias"
  - "robustness"
date: 2026-05-08
content_hash: 3880087512442fd0
---

# From Biased Chatbots to Biased Agents: Examining Role Assignment Effects on LLM Agent Robustness

**Conference**: AAAI 2026
**arXiv**: [2602.12285](https://arxiv.org/abs/2602.12285)
**Code**: None
**Area**: LLM Agent / AI Safety
**Keywords**: LLM Agent, implicit bias, persona, demographic bias, robustness

## TL;DR
The first systematic case study demonstrating that demographically grounded persona assignment causes up to 26.2% performance degradation in LLM agent task execution across 5 operational domains, establishing that persona-induced bias extends beyond text generation into action decision-making.

## Background & Motivation

**Background**: LLMs are transitioning from chatbots to autonomous agents capable of executing real-world operations (code deployment, OS automation, medical decision-making, etc.). Personas are widely employed to shape agent behavior and role identity.

**Limitations of Prior Work**: Persona-induced bias in pure text generation has been extensively studied—covering both explicit bias (triggering harmful outputs) and implicit bias (differential outputs for identical tasks under different personas). However, the effect of persona on **actual actions and decisions** in agent task execution remains almost entirely uninvestigated.

**Key Challenge**: Agent behavior directly affects the real world (executing commands, manipulating databases, trading decisions). If persona can alter agent decision quality, the harm is more immediate than textual bias—an agent assigned a particular demographic identity may perform worse on entirely unrelated tasks.

**Goal**: Systematically verify whether demographically grounded personas (gender, race/region, religion, occupation) affect LLM agent performance in multi-step task execution.

**Key Insight**: 23 personas spanning 4 demographic dimensions are used to evaluate 3 mainstream LLMs across 5 operational benchmark domains, quantifying the impact of persona on agent performance.

**Core Idea**: Persona cues entirely irrelevant to the task can cause performance drops of up to 26.2%, and this effect is pervasive across different models and task types.

## Method

### Overall Architecture
This is an empirical case study that proposes no new method; instead, it designs a rigorous experimental framework to measure the effect of persona on agent performance. The overall pipeline is: (1) select persona set → (2) inject persona via fixed prefix → (3) evaluate across 5 benchmarks → (4) compare performance with and without persona.

### Key Designs

1. **Persona Selection and Injection**:

    - **Function**: Covers 23 personas across 4 demographic dimensions, injected via a two-turn conversational prefix.
    - **Mechanism**: A standardized prompt template `"From now on, you are a [ROLE]."` followed by a role-confirmation reply is prepended before all downstream instructions. All personas use a consistent phrasing structure, differing only in demographic content. The baseline omits the prefix entirely.
    - **Design Motivation**: Ensures that persona content is the sole variable, eliminating interference from prompt format differences. The 4 selected dimensions (gender, race/region, religion, occupation) all have prior evidence of bias in text generation.

2. **Evaluation Benchmarks Covering 5 Operational Domains**:

    - **ALFWorld**: Household task planning (navigation, object localization, multi-step interaction in a simulated environment), measured by task success rate.
    - **WebShop**: E-commerce decision-making (search, filter, purchase), measured by reward score.
    - **Card Game**: Strategic reasoning (competitive card game), measured by win rate and score.
    - **OS Interaction**: System-level operations (parsing instructions to execute shell commands), measured by command accuracy.
    - **Database**: SQL generation (ranging from simple filtering to multi-table joins), measured by query correctness.

3. **Model Selection**:

    - GPT-4o-mini (commercial), DeepSeek-V3 (open-source), Qwen3-235B (MoE model).
    - All models use deterministic decoding with default configurations, reflecting real-world deployment conditions.

### Loss & Training
This is a purely evaluative study with no training involved. All models use zero-temperature deterministic decoding.

## Key Experimental Results

### Main Results

| Model | Task | Baseline | Worst Persona | Max Drop |
|-------|------|----------|---------------|---------|
| DeepSeek V3 | Card Game | 71.2% | from Africa: 45.0% | **-26.2%** |
| Qwen3 235B | Card Game | 61.7% | White: 37.9% | **-23.8%** |
| GPT-4o-mini | Card Game | 78.2% | from America: 59.1% | **-19.1%** |
| GPT-4o-mini | ALFWorld | 52.0% | Asian: 46.0% | -6.0% |
| DeepSeek V3 | ALFWorld | 86.0% | Baseline lowest | +6.0% (persona improves performance) |

Technical tasks (OS, Database) remain relatively stable with 2–5% fluctuation; tasks requiring multi-step reasoning and planning (Card Game, ALFWorld) are most affected.

### Ablation Study (Analysis by Demographic Dimension)

| Dimension | Key Findings |
|-----------|-------------|
| Race/Region | Most significant impact: DeepSeek V3 Card Game drops 26.2% under "from Africa" and 23.4% under "Asian" |
| Gender | In GPT-4o-mini, Male persona drops to 88% on household tasks while Female rises to 108%, reflecting stereotypes |
| Occupation | GPT-4o-mini performs worst under Laborer in ALFWorld; DeepSeek V3 shows significant improvement under Doctor |
| Religion | In DeepSeek V3, Christian drops from 71.2% to 48.5%; conversely, Christian improves performance in GPT-4o-mini |

### Key Findings
- **High-level reasoning is most vulnerable**: Card Game (strategic reasoning) is most affected by persona; OS/Database (execution tasks) are relatively stable.
- **Bias direction varies by model**: The same religious persona can have opposite effects across different models, indicating that bias originates from differences in training data and alignment.
- **Gender bias reflects social stereotypes**: Male personas perform worse on household tasks, suggesting models have internalized the stereotype that "males are less suited to domestic tasks."
- **Occupational bias is hierarchical**: High-status occupations (Doctor, CEO) generally improve performance, while low-status occupations (Laborer, Farmer) reduce it.

## Highlights & Insights
- **From textual to behavioral bias**: This work is the first to systematically extend persona bias research from text generation to agent task execution, revealing more direct operational risks—a perspective of critical importance for safe agent deployment.
- **The insidious nature of bias**: The persona is entirely irrelevant to the task (assigning an "Asian" identity to an agent should not affect its card game ability), yet the model exhibits systematic performance differences, demonstrating that bias is deeply embedded in the model's reasoning process.
- **Concise yet powerful experimental design**: A mere two-line prompt prefix is sufficient to induce performance fluctuations of up to 26.2%, yielding simple but compelling findings.

## Limitations & Future Work
- Only 3 models and 5 benchmarks are evaluated, limiting coverage.
- Mechanistic analysis is absent—why does persona affect reasoning, and at which stage of information processing does interference occur?
- No mitigation strategies (debiasing) are proposed—the work identifies the problem without resolving it.
- Deterministic decoding (temperature=0) may not fully reflect behavior under stochastic sampling.
- The interaction effects of persona in multi-agent collaborative settings are not considered.

## Related Work & Insights
- **vs. Text Bias Research (Bai et al. 2025, Gupta et al. 2024)**: Prior work demonstrates that persona influences text generation quality and reasoning; this paper extends that thread to agent behavior. The key distinction is that textual bias affects output quality, whereas agent bias affects actual decisions and actions.
- **vs. Agent Safety (DoomArena, etc.)**: Agent safety research typically focuses on adversarial attacks and jailbreaking; this paper reveals a more covert vulnerability—entirely normal persona assignments can degrade performance.
- **Implications**: Persona robustness testing should be incorporated into agent development to ensure stable performance across diverse identity assignments.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of persona effects on agent task execution; a genuinely fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 models × 5 benchmarks × 23 personas; comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Results are clearly presented with systematic dimension-level analysis.
- Value: ⭐⭐⭐⭐ Important warning for safe agent deployment, though no solutions are offered.

## Additional Notes
- The finding that role assignment amplifies bias has direct implications for multi-agent system design: socially stereotyped role descriptions should be avoided, or debiasing instructions should be incorporated into role prompts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)
- [\[ACL 2026\] Towards Scalable Lightweight GUI Agents via Multi-role Orchestration](../../ACL2026/llm_agent/towards_scalable_lightweight_gui_agents_via_multi-role_orchestration.md)
- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)
- [\[AAAI 2026\] DEPO: Dual-Efficiency Preference Optimization for LLM Agents](depo_dual-efficiency_preference_optimization_for_llm_agents.md)
- [\[AAAI 2026\] SoMe: A Realistic Benchmark for LLM-based Social Media Agents](some_a_realistic_benchmark_for_llm-based_social_media_agents.md)

</div>

<!-- RELATED:END -->
