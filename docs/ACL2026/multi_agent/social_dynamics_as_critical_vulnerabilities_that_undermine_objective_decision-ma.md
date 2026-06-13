---
title: >-
  [Paper Note] Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives
description: >-
  [ACL2026][Multi-Agent][Multi-agent systems] This paper demonstrates that representative agents in LLM multi-agent systems are not only limited by their own reasoning capabilities but are also significantly influenced by…
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Multi-agent systems"
  - "Social conformity"
  - "Adversarial peers"
  - "Representative agents"
  - "Decision robustness"
date: 2026-05-08
content_hash: 0bc57f075d64f669
---

# Social Dynamics as Critical Vulnerabilities that Undermine Objective Decision-Making in LLM Collectives

**Conference**: ACL2026  
**arXiv**: [2604.06091](https://arxiv.org/abs/2604.06091)  
**Code**: No public code  
**Area**: LLM Agent / Multi-Agent Decision Making / AI Safety Evaluation  
**Keywords**: Multi-agent systems, Social conformity, Adversarial peers, Representative agents, Decision robustness

## TL;DR
This paper demonstrates that representative agents in LLM multi-agent systems are not only limited by their own reasoning capabilities but are also significantly influenced by "social dynamics"—such as the number of peers, peer capabilities, argument length, and rhetorical style—leading to erroneous decisions in tasks with objective answers.

## Background & Motivation
**Background**: LLM agents are increasingly designed as collaborative systems: multiple peer agents provide perspectives, and a representative agent aggregates information to make a final judgment for the user. Such architectures are often seen as a means to improve performance and reduce single-model errors in reasoning, coding, fact-checking, and tool selection.

**Limitations of Prior Work**: Once peer opinions are introduced, the representative agent is no longer an isolated reasoner but resides within an information network. Previous research has largely focused on how multi-agent debates form consensus, with less focus on whether a representative agent, who might otherwise be correct, can be misled by a group of erroneous peers.

**Key Challenge**: The benefits of multi-agent collaboration stem from adopting external perspectives, but the risks arise from the over-adoption of these perspectives. Human groups exhibit socio-psychological phenomena such as conformity, authority influence, speech length bias, and rhetorical persuasion. If LLM agents exhibit similar biases, then "multi-agent discussion" is not necessarily more reliable.

**Goal**: The authors aim to systematically manipulate social pressure in peer networks during tasks with objective ground truths to observe changes in representative agent accuracy, interpreting these shifts as safety vulnerabilities of multi-agent systems.

**Key Insight**: The paper adopts a representative-centric framework: fixing one representative agent and five peer agents, where a portion of the peers are assigned misleading roles to provide a specific incorrect answer and justification; the representative agent then outputs the final answer after reviewing all five opinions.

**Core Idea**: Operationalize socio-psychological concepts like conformity, perceived expertise, the dominant speaker effect, and rhetorical persuasion into controllable multi-agent experimental variables, using the drop in accuracy to measure the representative agent's vulnerability to social pressure.

## Method

### Overall Architecture
Each trial consists of an objective multiple-choice question, five peer agents, and one representative agent. Benign peers solve the problem normally; misleading peers are set to support a specific incorrect option with plausible-sounding justifications. The representative agent receives the original question, candidate answers, and five peer opinions, then independently outputs a final answer. The system uses regular expressions for answer matching to calculate final accuracy.

The paper is structured around four research questions: RQ1 varies the number of misleading peers to simulate social conformity; RQ2 varies the misleading peers' model capability to simulate perceived expertise; RQ3 varies the length of misleading justifications to simulate the dominant speaker effect; RQ4 varies the justification style to Ethos, Logos, and Pathos to simulate rhetorical persuasion.

### Key Designs
1.  **Representative-Centric Experimental Structure**:
    - **Function**: Distinguishes between "group discussion failure" and "individual representative failure under group influence."
    - **Mechanism**: Peers do not engage in multi-round debate; they each provide a single answer and justification. The representative agent acts as the final decision-maker aggregating these opinions. This allows direct observation of the peer network's impact on individual judgment.
    - **Design Motivation**: In real products, users often see only the final conclusion of one agent, which may be supported by multiple sub-agents; thus, the robustness of the final representative is more critical than the group's average accuracy.

2.  **Four Categories of Social Dynamic Variables**:
    - **Function**: Translates abstract socio-psychological concepts into controllable experimental conditions.
    - **Mechanism**: Conformity is controlled via 0 to 5 misleading peers; perceived expertise is controlled through model size and family; the dominant speaker effect is controlled by justification length (one sentence to multiple paragraphs); rhetorical persuasion is controlled via credibility, logic, and emotional appeals.
    - **Design Motivation**: If representative agents relied solely on objective evidence, they should be relatively insensitive to these social variables. Changes in accuracy indicate the existence of non-factual influence channels.

3.  **Cross-Task and Cross-Model Verification**:
    - **Function**: Prevents conclusions from being tied to a single model or dataset.
    - **Mechanism**: Tasks cover social bias scenarios (BBQ), knowledge reasoning (MMLU-Pro), and tool decision-making (MetaTool). Models include Qwen2.5 7B/14B, Gemma3 12B, GPT-4o mini, GPT-4o, and Claude 3.5 Haiku. Representative agents use temperature 0 for stability, while peers use temperature 1 for diversity.
    - **Design Motivation**: If multi-agent vulnerabilities exist across domains and models, they represent a systemic issue rather than an artifact of a specific benchmark or prompt.

### Loss & Training
The paper does not train new models but performs a systematic evaluation. All tasks are zero-shot. In RQ3, peer justification lengths vary from 1 sentence to 3 paragraphs. In RQ4, stylistic instructions (Ethos, Logos, Pathos) are appended to the misleading peers' prompts.

## Key Experimental Results

### Main Results
The results for RQ1 are the most salient: when misleading peers reach a majority (3), accuracy for many models drops significantly; with 5 misleading peers, some models collapse entirely.

| Task/Representative Model | 0 Adv. Peers | 1 Adv. | 2 Adv. | 3 Adv. | 4 Adv. | 5 Adv. | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BBQ Gender ambig., Qwen2.5 7B | 99.89 | 97.92 | 91.43 | 78.14 | 61.21 | 30.39 | Steady decline from the first misleader |
| BBQ Gender ambig., Qwen2.5 14B | 99.44 | 99.44 | 99.01 | 93.69 | 56.52 | 7.40 | Robust at minority, sharp collapse at majority |
| BBQ Gender ambig., Gemma3 12B | 95.63 | 95.59 | 95.17 | 81.03 | 35.33 | 0.00 | Complete failure with five misleaders |
| MMLU-Pro STEM, GPT-4o | 53.83 | 51.00 | 47.83 | 41.17 | 29.33 | 19.50 | Knowledge tasks also affected by pressure |
| MetaTool Awareness, Qwen2.5 14B | 56.83 | 55.96 | 40.87 | 8.75 | 2.31 | 1.06 | Tool judgment highly sensitive to peers |

RQ2 shows that "stronger" misleading peers are more persuasive, especially when they belong to the same model family as the representative.

| Setup | # Adv. | Org. Family Misleader | Partial Replace w/ Stronger | Full Replace w/ Stronger | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BBQ Gender disambig., Rep Qwen2.5 7B | 3 | 65.20 | 57.02 | 54.23 | Stronger misleaders drive accuracy lower |
| BBQ Race disambig., Rep Qwen2.5 7B | 3 | 76.19 | 68.20 | 65.15 | Clear facts cannot resist "expert" influence |
| BBQ Gender ambig., Rep Qwen2.5 14B | 2 | 99.01 | Remains High | Varies by config | Strong representatives are stable in short-term |
| MMLU/MetaTool Trends | 1-3 | Weak peers have less impact | Strong peers have more impact | Same-family strong models most persuasive | Interaction of capability and style |

### Ablation Study
RQ3 uses only one misleading peer but varies its verbosity. The results suggest that a single verbose peer can exert influence comparable to a larger misleading group.

| Task/Model | 1 Sent | 3 Sent | 5 Sent | 1 Para | 3 Para | Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| BBQ Gender ambig., Qwen2.5 7B | 97.92 | 97.71 | 96.65 | 95.24 | 93.97 | Steady decline in ambiguous scenarios |
| BBQ Gender disambig., Qwen2.5 14B | 81.35 | 79.83 | 77.47 | 77.22 | 71.90 | Clear facts weakened by long arguments |
| BBQ Race disambig., Qwen2.5 14B | 91.42 | 89.10 | 86.48 | 86.02 | 81.22 | ~10 point drop for 3 paragraphs |
| MetaTool Selection, Qwen2.5 14B | 69.25 | 69.05 | 69.15 | 68.74 | 68.14 | Consistent trend, though smaller drop |

In RQ4, the effectiveness of rhetorical strategies depends on the representative model's capability and context. Qwen2.5 7B is less sensitive to complex rhetoric (sometimes treating it as noise), while Qwen2.5 14B is more susceptible to Ethos and Logos.

### Key Findings
- The majority threshold is critical: strong models often resist 1-2 misleaders but show significant accuracy drops once a 3-peer majority is formed.
- Perceived expertise is not just about model size but also family alignment; justifications from stronger same-family models are more persuasive to the representative.
- Long justifications are often mistaken by representative agents for more substantial evidence; even in unambiguous context (disambiguous BBQ), verbose misinformation can erode accuracy.
- Stronger representative agents are not always safer. While they reason better, they may "understand" complex rhetoric more deeply, making them more sensitive to Ethos and Logos.

## Highlights & Insights
- The most compelling aspect of the paper is the extension of multi-agent safety from "single malicious inputs" to how "social structures alter final judgment." This is closer to actual agentic workflows than traditional prompt attacks.
- The representative-centric design is highly practical, reflecting systems where multiple sub-agents aggregate to a primary agent.
- The results serve as a reminder that aggregation mechanisms in multi-agent systems cannot rely on "majority rule" or "reasoning plausibility" alone; they must explicitly model source credibility, evidence independence, and fact-checking.
- The paper presents a counter-intuitive insight: improving model capability may increase sensitivity to complex social signals, thus robustness training needs to target peer influence specifically rather than just chasing base benchmark scores.

## Limitations & Future Work
- Misleading peers are explicitly set in the experiment; in real systems, erroneous peers might arise from retrieval errors, tool failures, bias, or hallucinations, resulting in more complex patterns.
- Representative agents only perform single-round aggregation without opportunities to follow up, ask for evidence, or invoke external verification tools.
- The paper primarily uses accuracy as a metric and does not deeply analyze the internal trade-offs made by the representative agent or compare different aggregation algorithms.
- Future work could investigate aggregation with evidence citations, peer independence detection, anti-conformity calibration, and mechanisms for isolating anomalous opinions.

## Related Work & Insights
- **vs. Multi-agent Debate**: While debate research typically examines if groups reach a correct answer, this work looks at how an erroneous peer group affects a single representative's final judgment.
- **vs. Subjective Conformity**: Unlike previous work focused on opinion formation or preferences, this study uses benchmarks with ground truths (BBQ, MMLU-Pro, MetaTool), showing that conformity harms objective decision-making.
- **vs. Adversarial Prompt Attack**: The attack surface here is the social context—peer response distribution, peer capability, and speaking style—rather than a single input prompt.
- **Insights**: When designing agent systems, primary agents should not simply concatenate peer opinions into context; they must check for independence and whether long justifications are merely redundant persuasion.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Systematically mapping socio-psychological variables to LLM collectives is highly insightful.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers multiple models, tasks, and four variables; extensive data in appendices.
- **Writing Quality**: ⭐⭐⭐⭐☆ Research questions are clearly organized with insightful explanations.
- **Value**: ⭐⭐⭐⭐⭐ Directly warns of risks in multi-agent agentic products and collaborative reasoning system designs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)
- [\[NeurIPS 2025\] MetaMind: Modeling Human Social Thoughts with Metacognitive Multi-Agent Systems](../../NeurIPS2025/multi_agent/metamind_modeling_human_social_thoughts_with_metacognitive_multi-agent_systems.md)
- [\[ICLR 2026\] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](../../ICLR2026/multi_agent/mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)
- [\[ACL 2026\] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)
- [\[ACL 2026\] PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows](protea_offline_evaluation_and_iterative_refinement_for_multi-agent_llm_workflows.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2025\] Voting or Consensus? Decision-Making in Multi-Agent Debate](../../ACL2025/multi_agent/voting_or_consensus_decision-making_in_multi-agent_debate.md)
- [\[ACL 2026\] OxyGent: Making Multi-Agent Systems Modular, Observable, and Evolvable via Oxy Abstraction](oxygent_making_multi-agent_systems_modular_observable_and_evolvable_via_oxy_abst.md)
- [\[ACL 2026\] Seeing the Whole Elephant: A Benchmark for Failure Attribution in LLM-based Multi-Agent Systems](seeing_the_whole_elephant_a_benchmark_for_failure_attribution_in_llm-based_multi.md)
- [\[ACL 2026\] PROTEA: Offline Evaluation and Iterative Refinement for Multi-Agent LLM Workflows](protea_offline_evaluation_and_iterative_refinement_for_multi-agent_llm_workflows.md)
- [\[ACL 2026\] Memory-Augmented LLM-based Multi-Agent System for Automated Feature Generation on Tabular Data](memory-augmented_llm-based_multi-agent_system_for_automated_feature_generation_o.md)

</div>

<!-- RELATED:END -->
