---
title: >-
  [Paper Note] Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives
description: >-
  [ACL 2026][Multi-Agent][Paper Note] This paper demonstrates that representative agents in LLM multi-agent systems are limited not only by their own reasoning capabilities but also significantly influenced by "social dynamics"—such as the number of peers, peer capabilities, argument length, and rhetorical style—leading to incorrect decisions in tasks with
tags:
  - ACL 2026
  - Multi-Agent
date: 2026-05-08
content_hash: 85c4da20458010a9
---
# Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives

**Conference**: ACL2026  
**arXiv**: [2604.06091](https://arxiv.org/abs/2604.06091)  
**Code**: No public code  
**Area**: LLM Agent / Multi-agent Decision Making / AI Safety Evaluation  
**Keywords**: Multi-agent Systems, Social Conformity, Adversarial Peers, Representative Agent, Decision Robustness

## TL;DR
This paper demonstrates that representative agents in LLM multi-agent systems are limited not only by their own reasoning capabilities but also significantly influenced by "social dynamics"—such as the number of peers, peer capabilities, argument length, and rhetorical style—leading to incorrect decisions in tasks with objective answers.

## Background & Motivation
**Background**: LLM agents are increasingly designed as collaborative systems where multiple peer agents provide opinions, and a representative agent aggregates information to make a final judgment for the user. This structure is often seen as a means to improve performance and reduce individual model errors in reasoning, coding, fact-checking, and tool selection.

**Limitations of Prior Work**: Once peer opinions are introduced, the representative agent is no longer an isolated reasoner but part of an information network. Many previous studies focused on how multi-agent debates form group consensus, but few have investigated whether a representative agent, who might otherwise answer correctly, can be misled by a group of incorrect peers.

**Key Challenge**: The benefit of multi-agent collaboration comes from adopting external perspectives, but the risk arises from over-adopting them. Human groups exhibit socio-psychological phenomena such as conformity, authority influence, verbosity effects, and rhetorical persuasion. If LLM agents exhibit similar biases, "letting multiple agents discuss" is not necessarily more reliable.

**Goal**: The authors aim to systematically manipulate the social pressure of the peer network in tasks with objective ground truths to observe how the representative agent's accuracy changes, interpreting these changes as safety vulnerabilities of multi-agent systems.

**Key Insight**: The paper adopts a representative-centric framework: fixing one representative agent and five peer agents, where a subset of peers are set as misleading roles providing a specific wrong answer and justification. The representative agent outputs a final answer after viewing the five opinions.

**Core Idea**: Operationalize socio-psychological concepts—conformity, perceived expertise, the dominant speaker effect, and rhetorical persuasion—into controllable multi-agent experimental variables. The decline in accuracy is used to quantify the representative agent's vulnerability to social pressure.

## Method

### Overall Architecture
Each trial consists of a multiple-choice question with an objective answer, five peer agents, and one representative agent. Benign peers solve the problem normally; misleading peers are set to support a specific incorrect option with plausible-sounding reasons. The representative agent receives the original question, candidate answers, and the five peer opinions, then independently outputs a final answer. The system uses regex to match answer options and calculate final accuracy.

The paper addresses four research questions: RQ1 varies the number of misleading peers to simulate social conformity; RQ2 varies the model capability of misleading peers to simulate perceived expertise; RQ3 varies the length of misleading justifications to simulate the dominant speaker effect; RQ4 varies the justification style into Ethos, Logos, and Pathos to simulate rhetorical persuasion.

### Key Designs

**1. Representative-Centric Experimental Structure: Decoupling "Group Discussion Failure" from "Individual Representative Misguidance"**

Previous multi-agent research often examined whether an entire group could converge on the correct answer through discussion. However, this conflates two issues: whether the discussion mechanism itself failed or whether the final decision-maker was persuaded by incorrect peers. This paper deliberately ensures the five peers **do not debate each other** and provide their answers and reasons independently. The representative agent then outputs the final choice in a single round of aggregation. Any change in accuracy can thus only be attributed to how the representative handles peer opinions, rather than the mixed effects of multi-round debates. This perspective is chosen because, in real products, users often only see the conclusion of a primary agent, which may be fed suggestions by multiple sub-agents; thus, the robustness of the final representative is closer to actual operational risk than the group's average accuracy.

**2. Four Categories of Social Dynamics Variables: Translating Socio-Psychological Concepts into Experimental Knobs**

Concepts like social conformity, perceived expertise, dominant speakers, and rhetorical persuasion are inherently abstract. The key contribution here is operationalizing each into a precisely controlled variable. Conformity is adjusted by varying the number of misleading peers (0 to 5); perceived expertise is adjusted by replacing misleading peers with larger or same-family models; the dominant speaker effect is simulated by extending the length of misleading justifications from one sentence to three paragraphs; and rhetorical persuasion is simulated by appending Ethos (credibility), Logos (logic), or Pathos (emotion) styles to misleading prompts. The logic is straightforward: a representative relying solely on objective evidence should be insensitive to these fact-irrelevant variables. If accuracy drops systematically with a specific "knob," it indicates a "social influence channel" that bypasses facts, with the magnitude of the drop quantifying the width of that channel.

**3. Cross-Task and Cross-Model Validation: Ensuring Vulnerability Is Not Specific to Benchmarks or Models**

If these phenomena appeared only on one model or dataset, they could be dismissed as artifacts of prompt engineering. To counter this, the paper applies the same manipulations across three task types: social bias scenarios (BBQ), knowledge reasoning (MMLU-Pro), and tool decision-making (MetaTool). Models include Qwen2.5 7B/14B, Gemma3 12B, GPT-4o mini, GPT-4o, and Claude 3.5 Haiku. The representative agent's temperature is fixed at 0 to ensure stable and reproducible output, while peer temperature is set to 1 to ensure diverse misleading justifications. Since vulnerability holds across so many domains and models, it suggests a systemic weakness in aggregated multi-agent architectures rather than an accidental configuration.

### Experimental setup & Protocol
The paper does not train new models but performs a systematic evaluation: all tasks are zero-shot, peers provide answers and reasons, the representative agent selects a final answer after aggregation, and regex is used for option matching. The four RQs correspond to four sets of manipulations—in RQ3, peer justification lengths increase from 1 sentence to 3 paragraphs; in RQ4, Ethos, Logos, or Pathos style instructions are appended to misleading peer prompts.

## Key Experimental Results

### Main Results
The results for RQ1 are the most intuitive: accuracy begins to drop significantly when misleading peers reach a majority (3); with 5 misleading peers, some models almost completely collapse.

| Task/Rep Model | 0 Adv. | 1 Adv. | 2 Adv. | 3 Adv. | 4 Adv. | 5 Adv. | Observation |
|----------------|-------------|------|------|------|------|------|------|
| BBQ Gender ambig., Qwen2.5 7B | 99.89 | 97.92 | 91.43 | 78.14 | 61.21 | 30.39 | Steady decline from the first misleader |
| BBQ Gender ambig., Qwen2.5 14B | 99.44 | 99.44 | 99.01 | 93.69 | 56.52 | 7.40 | Robust at minority, collapses at majority |
| BBQ Gender ambig., Gemma3 12B | 95.63 | 95.59 | 95.17 | 81.03 | 35.33 | 0.00 | Total failure under 5 misleading peers |
| MMLU-Pro STEM, GPT-4o | 53.83 | 51.00 | 47.83 | 41.17 | 29.33 | 19.50 | Knowledge tasks also affected by pressure |
| MetaTool Awareness, Qwen2.5 14B | 56.83 | 55.96 | 40.87 | 8.75 | 2.31 | 1.06 | Tool judgment extremely sensitive to peers |

RQ2 shows that the "stronger" the misleading peer, the more persuasive it is, and model family alignment amplifies this effect. Using Qwen2.5 7B as the representative, replacing misleading peers with stronger models further reduces accuracy.

| Setup | # Adv. | Original Same-Family | Partial Stronger Peer | All Stronger Peers | Observation |
|------|--------|----------------|--------------------|--------------------|----------|
| BBQ Gender disambig., Rep Qwen2.5 7B | 3 | 65.20 | 57.02 | 54.23 | Stronger peers continuously lower accuracy |
| BBQ Race disambig., Rep Qwen2.5 7B | 3 | 76.19 | 68.20 | 65.15 | Clear facts cannot fully resist "experts" |
| BBQ Gender ambig., Rep Qwen2.5 14B | 2 | 99.01 | Stays high | Depends on config | High-capability reps more stable in ambiguity |
| MMLU/MetaTool Overall | 1-3 | Small impact (weak) | Increased impact | Strongest (same family) | Capability and style work together |

### Ablation Study
RQ3 uses only one misleading peer but increases its verbosity. The results indicate that a single verbose peer can induce accuracy drops nearing those caused by expanding the misleading group.

| Task/Model | 1 Sent | 3 Sent | 5 Sent | 1 Para | 3 Para | Conclusion |
|-----------|-----|-----|-----|-----|-----|------|
| BBQ Gender ambig., Qwen2.5 7B | 97.92 | 97.71 | 96.65 | 95.24 | 93.97 | Steady decline in ambiguous scenarios |
| BBQ Gender disambig., Qwen2.5 14B | 81.35 | 79.83 | 77.47 | 77.22 | 71.90 | Clear fact scenarios also weakened by length |
| BBQ Race disambig., Qwen2.5 14B | 91.42 | 89.10 | 86.48 | 86.02 | 81.22 | ~10pt drop with 3 paragraphs |
| MetaTool Selection, Qwen2.5 14B | 69.25 | 69.05 | 69.15 | 68.74 | 68.14 | Smaller drop but consistent trend |

In RQ4, the effectiveness of rhetorical strategies depends on the representative's capability and task context. Qwen2.5 7B is less sensitive to complex rhetoric, sometimes treating it as noise; Qwen2.5 14B is more susceptible to Ethos and Logos. In the BBQ ambiguous scenario for Qwen2.5 14B, the three rhetorical types can lead to a maximum accuracy drop of about 7 percentage points; in MMLU-Pro, Ethos and Logos consistently lower accuracy across multiple categories.

### Key Findings
- The majority threshold is critical: strong models often resist 1-2 misleaders, but accuracy drops significantly once 3 misleaders form a majority.
- Perceived expertise is not just about model size; it relates to model family alignment. Stronger models within the same family provide more persuasive reasons to the representative.
- Representative agents mistake long justifications for more substantial evidence; even in disambiguated BBQ, clear context cannot fully offset verbose misinformation.
- Stronger representative agents are not always safer. While they reason better, they may also "understand" complex rhetoric more deeply, making them more sensitive to Ethos/Logos.

## Highlights & Insights
- The most interesting aspect is the expansion of multi-agent safety from "single malicious input" to "how social structure alters final judgment." This is far more relevant to agentic workflows than traditional prompt attacks.
- The representative-centric design is very clear, as many practical systems indeed aggregate multiple sub-agents into a primary agent for user output.
- The findings serve as a reminder that aggregation mechanisms in multi-agent systems should not rely solely on "majority opinion" or "plausibility of reasoning"; they should explicitly model source credibility, evidence independence, and fact-checking.
- A counter-intuitive insight is provided: increasing model capability might increase sensitivity to complex social signals, so robustness training must specifically target peer influence rather than just chasing single-model benchmark scores.

## Limitations & Future Work
- Misleading peers in the experiments are explicitly configured; in real systems, incorrect peers may stem from retrieval errors, tool failures, bias, or model hallucinations, making their forms more complex.
- The representative agent only performs single-round aggregation without opportunities to follow up, request evidence, or call external verification tools, thus reflecting vulnerabilities under weak aggregation mechanisms.
- The paper primarily uses accuracy to measure final results without deeply analyzing how the representative agent weights peer evidence internally or systematically comparing different aggregation algorithms.
- Rhetorical strategies in RQ4 are relatively coarse; real-world persuasion might mix credibility, logic, emotion, and format control.
- Future work could investigate aggregation with evidence citation, peer independence detection, anti-conformity calibration, peer-weight learning for representative agents, and outlier opinion isolation mechanisms in multi-agent systems.

## Related Work & Insights
- **vs. Multi-agent Debate**: Multi-agent debate often looks at whether a group reaches a correct answer through discussion; this paper looks at how an incorrect peer group affects a single representative's final judgment.
- **vs. Subjective Opinion Conformity**: Much prior work focused on opinion formation or subjective preferences; this paper uses tasks with ground truths (BBQ, MMLU-Pro, MetaTool), showing that conformity harms objective decision-making.
- **vs. Adversarial Prompt Attack**: The attack surface here is not a single input prompt but the social context composed of the peer response distribution, peer capability, and speaking style.
- **Insights**: When designing agent systems, the primary agent should not simply concatenate peer opinions into the context; it needs to check if peers are independent, if they cite evidence, if the majority is from the same source, and if long reasons are merely redundant persuasion.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Systematically mapping socio-psychological variables to LLM collectives is highly inspired; the representative-centric perspective is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers multiple models, tasks, and four categories of variables with detailed appendix tables; however, it lacks defense methods and more realistic agent workflows.
- Writing Quality: ⭐⭐⭐⭐☆ Research questions are clearly organized, and results are interpreted with insight; some charts require the appendix for full numerical detail.
- Value: ⭐⭐⭐⭐⭐ Direct warning significance for multi-agent products, AI agent safety, and collaborative reasoning system design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

## Related Papers

- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](../../ACL2025/multi_agent/voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[ACL 2026\] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)
- [\[ACL 2026\] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/multi_agent/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[ACL 2026\] PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows](protea_offline_evaluation_and_iterative_refinement_for_multi-agent_llm_workflows.md)

</div>

<!-- RELATED:END -->
