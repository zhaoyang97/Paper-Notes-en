---
title: >-
  [Paper Note] Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives
description: >-
  [ACL2026][Multi-Agent][Multi-agent systems] This paper proves that representative agents in LLM multi-agent systems are not only limited by their own reasoning capabilities but are also significantly influenced by "social dynamics"—such as the number of peers, peer capabilities, argument length, and rhetorical style—leading to incorrect decisions on tasks with objective answers.
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Multi-agent systems"
  - "social conformity"
  - "adversarial peers"
  - "representative agents"
  - "decision robustness"
date: 2026-05-08
content_hash: 269be0bcb40fe071
---

# Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives

**Conference**: ACL2026  
**arXiv**: [2604.06091](https://arxiv.org/abs/2604.06091)  
**Code**: No public code  
**Area**: LLM Agent / Multi-agent Decision Making / AI Safety Evaluation  
**Keywords**: Multi-agent systems, social conformity, adversarial peers, representative agents, decision robustness

## TL;DR
This paper proves that representative agents in LLM multi-agent systems are not only limited by their own reasoning capabilities but are also significantly influenced by "social dynamics"—such as the number of peers, peer capabilities, argument length, and rhetorical style—leading to incorrect decisions on tasks with objective answers.

## Background & Motivation
**Background**: LLM agents are increasingly designed as collaborative systems: multiple peer agents provide viewpoints, and a representative agent aggregates information to make a final judgment for the user. This structure is often seen as a means to improve performance and reduce single-model errors in reasoning, coding, fact-checking, and tool selection.

**Limitations of Prior Work**: Once peer opinions are introduced, the representative agent is no longer an isolated reasoner but part of an information network. Many previous studies focused on how multi-agent debates form group consensus, but few have investigated whether a representative agent, who might have answered correctly on their own, can be misled by a group of erroneous peers.

**Key Challenge**: The benefit of multi-agent collaboration comes from adopting external perspectives, but the risk arises from over-adopting them. Human groups exhibit socio-psychological phenomena such as conformity, authority influence, verbosity effects, and rhetorical persuasion. If LLM agents exhibit similar biases, then "discussing among multiple agents" is not necessarily more reliable.

**Goal**: The authors aim to systematically manipulate the social pressure of peer networks in tasks with objective ground-truth answers to observe how the accuracy of the representative agent changes, interpreting these shifts as safety vulnerabilities of multi-agent systems.

**Key Insight**: The paper adopts a representative-centric framework: fixing one representative agent and five peer agents, where a subset of peers is set as misleading characters providing incorrect answers and justifications. The representative agent outputs a final answer after reviewing all five opinions.

**Core Idea**: Operationalize social psychology concepts—conformity, perceived expertise, the dominant speaker effect, and rhetorical persuasion—into controllable multi-agent experimental variables, measuring the representative agent's vulnerability to social pressure through the decline in accuracy.

## Method

### Overall Architecture
Each trial consists of a multiple-choice question with an objective answer, five peer agents, and one representative agent. Benign peers solve the problem normally; misleading peers are instructed to support a specific incorrect option and provide plausible-sounding reasons. The representative agent receives the original question, candidate options, and the five peer opinions to independently output a final answer. The system uses regex to match answer options and calculate final accuracy.

The paper is structured around four research questions: RQ1 manipulates the number of misleading peers to simulate social conformity; RQ2 manipulates the model capability of misleading peers to simulate perceived expertise; RQ3 manipulates the length of misleading justifications to simulate the dominant speaker effect; RQ4 changes the justification style into Ethos, Logos, and Pathos to simulate rhetorical persuasion.

### Key Designs

**1. Representative-Centric Experimental Structure: Decoupling "Group Discussion Failure" from "Representative Deviation"**

Previous multi-agent research often examined whether a whole group could converge on a correct answer through discussion. However, this confuses two things: whether the discussion mechanism itself failed, or whether the final decision-maker was persuaded by incorrect peers. This paper deliberately keeps the five peers from debating each other, ensuring they provide independent answers and reasons. The representative agent then performs a single-round aggregation. Any change in accuracy can thus only be attributed to how the representative handles peer opinions, rather than the mixed effects of multi-round discussions. This perspective is chosen because, in real-world products, users often only see the conclusion of a primary agent, which may be fed suggestions by multiple sub-agents; thus, the robustness of the final representative is closer to actual operational risk than average group accuracy.

**2. Four Categories of Social Dynamics Variables: Translating Socio-Psychological Concepts into Controllable Variables**

Concepts like social conformity, perceived expertise, dominant speakers, and rhetorical persuasion are inherently abstract. The key contribution here is operationalizing each into a precisely controlled variable. Conformity is dialed by the number of misleading peers (from 0 to 5); perceived expertise is dialed by replacing misleading peers with larger or same-family models; the dominant speaker effect is dialed by lengthening misleading justifications from one sentence to three paragraphs; and rhetorical persuasion is dialed by appending Ethos (credibility), Logos (logic), or Pathos (emotion) styles to the misleading prompts. The logic is straightforward: a representative relying solely on objective evidence should be insensitive to these fact-irrelevant variables. If accuracy drops systematically as a knob is turned, it indicates a "social influence channel" bypassing the facts, and the magnitude of the drop quantifies the width of that channel.

**3. Cross-Task and Cross-Model Verification: Ensuring Vulnerability is Not Benchmark-Specific**

If these phenomena appeared only on one model or dataset, they could be dismissed as artifacts of prompt engineering. To counter this, the authors apply the same manipulations across three types of tasks: social bias scenarios (BBQ), knowledge reasoning (MMLU-Pro), and tool decision-making (MetaTool). Models tested includes Qwen2.5 7B/14B, Gemma3 12B, GPT-4o mini, GPT-4o, and Claude 3.5 Haiku. Representative agents use a temperature of 0 to ensure stable, reproducible output, while peer temperature is set to 1 to ensure diverse misleading justifications. The consistency of vulnerability across these domains and models suggests a systemic weakness in aggregative multi-agent architectures rather than a configuration fluke.

### Loss & Training
The paper does not train new models but performs a systematic evaluation: all tasks are zero-shot. Peers provide answers and reasons; the representative agent aggregates these to select a final answer, which is then parsed via regex to calculate accuracy. The four research questions correspond to four sets of manipulations—in RQ3, peer justification lengths increase from 1 sentence, 3 sentences, 5 sentences, to 1 paragraph and 3 paragraphs; in RQ4, Ethos, Logos, or Pathos instructions are appended to misleading peer prompts.

## Key Experimental Results

### Main Results
The results for RQ1 are the most intuitive: accuracy for many models begins to drop significantly when misleading peers reach a majority of 3; when all 5 peers are misleading, some models collapse entirely.

| Task / Representative Model | 0 Adv. Peers | 1 | 2 | 3 | 4 | 5 | Observation |
|-----------------------------|--------------|---|---|---|---|---|-------------|
| BBQ Gender ambig., Qwen2.5 7B | 99.89 | 97.92 | 91.43 | 78.14 | 61.21 | 30.39 | Steady decline from the 1st misleader |
| BBQ Gender ambig., Qwen2.5 14B | 99.44 | 99.44 | 99.01 | 93.69 | 56.52 | 7.40 | Robust in minority; sharp collapse in majority |
| BBQ Gender ambig., Gemma3 12B | 95.63 | 95.59 | 95.17 | 81.03 | 35.33 | 0.00 | Complete failure with 5 misleading peers |
| MMLU-Pro STEM, GPT-4o | 53.83 | 51.00 | 47.83 | 41.17 | 29.33 | 19.50 | Knowledge tasks also affected by pressure |
| MetaTool Awareness, Qwen2.5 14B | 56.83 | 55.96 | 40.87 | 8.75 | 2.31 | 1.06 | Tool judgment highly sensitive to peers |

RQ2 shows that "stronger" misleading peers are more persuasive, and being in the same model family amplifies this effect. When Qwen2.5 7B serves as the representative, replacing misleading peers with stronger models further lowers accuracy.

| Setting | # Adv. | Original Same-Family Adv. | Partially Replaced w/ Stronger | Fully Replaced w/ Stronger | Observation |
|---------|--------|---------------------------|--------------------------------|----------------------------|-------------|
| BBQ Gender disambig., Rep Qwen2.5 7B | 3 | 65.20 | 57.02 | 54.23 | Stronger adv. consistently lower accuracy |
| BBQ Race disambig., Rep Qwen2.5 7B | 3 | 76.19 | 68.20 | 65.15 | Clear facts cannot fully resist "experts" |
| BBQ Gender ambig., Rep Qwen2.5 14B | 2 | 99.01 | Maintained high | Dep. on config | Stronger reps more stable in ambig. scenarios |
| MMLU/MetaTool Overall Trend | 1-3 | Small effect for weak peers | Larger effect for strong peers | Same-family strong models most persuasive | Ability and style work together |

### Ablation Study
RQ3 places only one misleading peer but increases its verbosity. The results indicate that a single verbose peer can exert an influence similar to expanding the misleading group.

| Task / Model | 1 Sen. | 3 Sen. | 5 Sen. | 1 Para. | 3 Para. | Conclusion |
|--------------|--------|--------|--------|---------|---------|------------|
| BBQ Gender ambig., Qwen2.5 7B | 97.92 | 97.71 | 96.65 | 95.24 | 93.97 | Steady decline in ambiguous scenarios |
| BBQ Gender disambig., Qwen2.5 14B | 81.35 | 79.83 | 77.47 | 77.22 | 71.90 | Clear facts weakened by long arguments |
| BBQ Race disambig., Qwen2.5 14B | 91.42 | 89.10 | 86.48 | 86.02 | 81.22 | 3-para reasons cause ~10% drop |
| MetaTool Selection, Qwen2.5 14B | 69.25 | 69.05 | 69.15 | 68.74 | 68.14 | Smaller drop but consistent trend |

In RQ4, the effect of rhetorical strategies depends on the representative model's capability and task context. Qwen2.5 7B is less sensitive to complex rhetoric, sometimes treating it as noise. Qwen2.5 14B is more susceptible to Ethos and Logos. In the BBQ ambiguous scenario for Qwen2.5 14B, the three rhetorical types lead to a maximum accuracy drop of approximately 7 percentage points; in MMLU-Pro, Ethos and Logos consistently reduce accuracy across multiple categories.

### Key Findings
- The majority threshold is critical: strong models often resist 1-2 misleaders, but accuracy drops significantly once 3 misleaders form a majority.
- Perceived expertise is not just about model size; it relates to model family alignment. Strong models from the same family are more likely to persuade the representative.
- Long justifications are misinterpreted as more substantial evidence; even in disambiguated BBQ contexts where facts are clear, clear context cannot fully offset verbose misinformation.
- Stronger representative agents are not always safer. While they reason better, they may also be more "attuned" to complex rhetorical signals, making them more sensitive to Ethos/Logos.

## Highlights & Insights
- The most interesting aspect is the expansion of multi-agent safety from "single malicious input" to "how social structures alter final judgment." This is closer to real-world agentic workflows than traditional prompt attacks.
- The representative-centric design is highly intuitive, as many actual systems aggregate sub-agent outputs through a primary agent for the end user.
- The results serve as a reminder that aggregation mechanisms in multi-agent systems should not rely simply on "majority opinion" or "plausible-sounding reasons." Instead, they must explicitly model source credibility, evidence independence, and fact-checking.
- Ours also suggests a counter-intuitive finding: increasing model capability might increase sensitivity to complex social signals. Therefore, robustness training must specifically target peer influence rather than solely pursuing individual benchmark scores.

## Limitations & Future Work
- Misleading peers in the experiment are explicitly configured; in real systems, erroneous peers might stem from retrieval errors, tool failures, bias, or model hallucinations, presenting more complex forms.
- The representative agent only performs single-round aggregation, lacking the chance to follow up, request evidence, or call external verification tools. The results thus reflect vulnerabilities under weak aggregation mechanisms.
- The paper primarily uses accuracy to measure final results without deeply analyzing how representative agents weigh peer evidence internally or comparing different aggregation algorithms.
- Rhetorical strategies in RQ4 are coarse-grained; real persuasion might mix credibility, logic, emotion, and formatting controls.
- Future work could investigate aggregation with evidence citations, peer independence detection, anti-conformity calibration, peer-weight learning for representative agents, and outlier isolation mechanisms in multi-agent systems.

## Related Work & Insights
- **vs. Multi-agent Debate**: Multi-agent debate often looks at whether a group can reach a correct answer via discussion; this paper examines how an erroneous peer group affects a single representative's final judgment.
- **vs. Subjective Opinion Conformity**: Previous work often focused on opinion formation or subjective preferences; this paper uses objective tasks like BBQ, MMLU-Pro, and MetaTool, showing that conformity undermines objective decision-making.
- **vs. Adversarial Prompt Attack**: The attack surface here is not a single input prompt but the social context composed of peer response distribution, peer capabilities, and speaking styles.
- **Insight**: When designing agent systems, the primary agent should not simply concatenate peer opinions into the context. It needs to check if peers are independent, whether they cite evidence, if the majority shares the same origin, and if long reasons are merely redundant persuasion.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically mapping social psychology variables to LLM collectives is highly inspired; the representative-centric perspective is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models, tasks, and four types of variables with detailed tables; however, it lacks defense methods and more realistic agent workflows.
- Writing Quality: ⭐⭐⭐⭐☆ Research questions are clearly organized and results are explained with insight; some charts require the appendix for full numerical detail.
- Value: ⭐⭐⭐⭐⭐ Directly warns designers of multi-agent products, AI agent safety, and collaborative reasoning systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](../../ACL2025/multi_agent/voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[ACL 2026\] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)
- [\[ICML 2026\] EduMirror: Modeling Educational Social Dynamics with Value-driven Multi-agent Simulation](../../ICML2026/multi_agent/edumirror_modeling_educational_social_dynamics_with_value-driven_multi-agent_sim.md)
- [\[ACL 2026\] PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows](protea_offline_evaluation_and_iterative_refinement_for_multi-agent_llm_workflows.md)
- [\[ACL 2026\] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)

</div>

<!-- RELATED:END -->
