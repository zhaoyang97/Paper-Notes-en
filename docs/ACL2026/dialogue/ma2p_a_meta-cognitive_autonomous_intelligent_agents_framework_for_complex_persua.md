---
title: >-
  [Paper Note] MA$^2$P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion
description: >-
  [ACL2026][Dialogue Systems][Complex Persuasion] MA$^2$P decomposes complex persuasive dialogue into a closed loop of "meta-strategy selection - task-level multi-agent persuasion - post-hoc knowledge updating." Without tr…
tags:
  - "ACL2026"
  - "Dialogue Systems"
  - "Complex Persuasion"
  - "Meta-cognition"
  - "Multi-agent"
  - "Theory of Mind Modeling"
  - "Strategy Knowledge Base"
date: 2026-05-08
content_hash: a9bc8ab63a101702
---

# MA$^2$P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion

**Conference**: ACL2026 Findings  
**arXiv**: [2605.18572](https://arxiv.org/abs/2605.18572)  
**Code**: The paper states that prompts, code, and knowledge bases will be released, but no public repository link is provided in the cache.  
**Area**: Dialogue Systems / LLM Agent / Persuasive Dialogue  
**Keywords**: Complex Persuasion, Meta-cognition, Multi-agent, Theory of Mind Modeling, Strategy Knowledge Base

## TL;DR
MA$^2$P decomposes complex persuasive dialogue into a closed loop of "meta-strategy selection - task-level multi-agent persuasion - post-hoc knowledge updating." Without training the backbone LLM, it transforms the persuadee's beliefs, desires, and concerns into specific strategic actions, significantly improving the persuasion success rate of various LLMs on CToMPersu.

## Background & Motivation
**Background**: Persuasive dialogue has evolved from early single-domain donation and negotiation tasks to more diverse domains and fine-grained user state modeling. Datasets like CToMPersu provide not only dialogue context but also expose the mental states of the persuadee, such as beliefs and desires. Consequently, models must go beyond generating fluent responses and engage in continuous planning based on latent concerns.

**Limitations of Prior Work**: Current LLM persuaders often function as single next-turn generators. While they can identify explicit obstacles like "lack of money" or "lack of time," they frequently revert to generic suggestions—such as emphasizing the importance of psychotherapy without translating obstacles into actionable items like insurance reimbursement, online sessions, or low-cost trials. Another issue is unstable cross-domain performance; motivation experiments show that the success rate of gpt-5-mini on CToMPersu varies across domains from 88.24% to 16.67%, a span of 71.57 percentage points.

**Key Challenge**: Complex persuasion is not a single-turn language generation task but a partially observable, multi-turn, goal-oriented interaction. The model must select strategies based on the counterpart's hidden states while maintaining stable generalization across domains. Monolithic LLMs lack explicit planning states and strategic memory, making them prone to producing polished but non-actionable reactive outputs.

**Goal**: The authors aim to construct a plug-and-play, training-free external framework that enables any backbone LLM to complete complex persuasion more stably. Specific sub-problems include: extracting mental states from dialogue history, grounding high-level psychological strategies into specific next-turn tactics, reducing cross-domain volatility using historical success cases, and writing successful experiences back into the system after dialogue.

**Key Insight**: The paper draws on the structure of perception, world model, actor, memory, and cost/evaluator from LeCun's autonomous agents while introducing planning, monitoring, and evaluation from meta-cognition. The core observation is that a persuasion system needs to first decide "what high-level strategy should be used for this scenario" before task-level agents generate the next action, rather than letting the LLM improvise every turn.

**Core Idea**: A meta-cognitive configurator first selects domain-related meta-strategies from a structured knowledge base. Then, perception, world model, persuader, memory, and evaluator agents execute and update within a closed loop, converting mental state clues into strategy-consistent, executable persuasive actions.

## Method

### Overall Architecture
MA$^2$P models a persuasive dialogue as a three-stage cycle. The input is a scenario $S$, including domain, goal, and background; the output consists of multi-turn persuasive dialogue and an updated knowledge base. The first stage is **Meta-level Judging**: the Configurator retrieves candidate meta-strategies from the knowledge base based on the scenario domain, selects the historically most effective strategy, and constructs evaluation rules for the current session. The second stage is **Task-level Persuading**: multiple autonomous agents collaborate to generate each response, including perceiving mental clues, inferring specific strategies, generating utterances, and maintaining short-term memory. The third stage is **Knowledge Updating**: the Evaluator determines the success of the session, and successful cases are written back to the knowledge base to provide evidence for future strategy selection in similar domains.

This framework does not retrain the LLM; instead, it serves as an external orchestration layer connected to the backbone model. In experiments, the same MA$^2$P framework was applied to gpt-4o-mini, gpt-4o, gpt-5-mini, gemini-2.5-flash, and deepseek-v3, demonstrating its plug-and-play design.

### Key Designs
1.  **Meta-cognitive Configurator and Meta-strategy Selection**:
    - **Function**: Determines the high-level strategy for the current session before the dialogue begins and generates success criteria for the subsequent Evaluator.
    - **Mechanism**: The knowledge base is organized into three layers: meta-strategy, domain, and case. The Configurator first retrieves the set of candidate strategies $M(S)$ matching the current domain $D(S)$, then scores them using the historical success counts of the domain-strategy combinations in the Case Layer, selecting $M=\arg\max_{m \in M(S)} score(m,S)$. This step ensures the system has a global intent of "how to persuade in this scenario" rather than making spontaneous judgments during turn-by-turn generation.
    - **Design Motivation**: One root cause of cross-domain volatility is the uneven generalization of LLM knowledge and strategies across different fields. Using historical success counts for meta-strategy selection explicitly records "which strategy works best in which domain," reducing the probability of the model performing blindly in weak domains.

2.  **Task-level Autonomous Agent Decomposition**:
    - **Function**: Grounding abstract meta-strategies into specific, context-aware persuasive utterances for each turn.
    - **Mechanism**: Perception extracts explicit signals and latent mental clues $P_t=f_{perc}(H_t)$ from the history $H_t$, such as beliefs, desires, and latent concerns; the World Model infers the specific strategy for the next turn $W_t=f_{wm}(M,\Sigma_t)$ by combining the meta-strategy $M$ and short-term memory $\Sigma_t$; the Persuader Agent then converts $W_t$ and the dialogue history into a natural language response $U_t=f_{pers}(W_t,H_t)$; Short-term Memory stores the history, perception results, and past strategies $\Sigma_t=\{H_t,P_t,W_{1:t-1}\}$.
    - **Design Motivation**: This approach more closely mimics human persuasion—understanding why the other party resists, deciding on a strategy, and then organizing language—compared to a single LLM directly generating the next sentence. In scenarios where resistance is implicit or dynamic, explicit memory prevents strategy drift and repetitive generic persuasion.

3.  **Evaluator and Knowledge Base Write-back**:
    - **Function**: Converting the outcome of an interaction into reusable experience.
    - **Mechanism**: The Evaluator uses the rules $E$ generated in the first stage and the final short-term memory $\Sigma_T$ to judge success, yielding $R=f_{eval}(E,\Sigma_T)$. If $R=1$, the system increments the success count of the selected meta-strategy in the current domain: $K_{case}(M,D(S)) \leftarrow K_{case}(M,D(S))+1$, and generates an updated knowledge base via $K'=update(K,M,S,R)$.
    - **Design Motivation**: The effectiveness of persuasive strategies is highly dependent on domain and population. Writing back successful cases accumulates interactions into retrievable evidence, allowing the framework to evolve from a cold-start rule-based agent into a meta-cognitive system with experience.

### Loss & Training
MA$^2$P does not train the backbone model and lacks traditional supervised or RL losses. It employs a prompt-based, multi-agent inference-time strategy. The main experiments utilize 525 instances from the CToMPersu official test set, with a maximum dialogue turn $T_{max}=4$. gpt-4o-mini is fixed as the persuadee simulator and LLM judge. The knowledge base size is studied as a warm-up hyperparameter: it achieves a 0.66 success rate at $K=0$ and reaches 0.79 at $K=500$, which is used as the main experimental setting.

## Key Experimental Results

### Main Results
The paper compares five backbone LLMs and their MA$^2$P-enhanced versions on CToMPersu. Metrics include Success (rate), Persuasive, Logic, Helpful, cross-domain Range/SD, and the average number of turns to success (Avg_Turn). Success rates and turn data are presented below:

| Backbone Model | Success Baseline | Success + MA$^2$P | Gain | Avg_Turn Baseline | Avg_Turn + MA$^2$P |
| :--- | :--- | :--- | :--- | :--- | :--- |
| gpt-4o-mini | 0.45 | 0.79 | +0.34 | 2.94 | 1.86 |
| gpt-4o | 0.46 | 0.75 | +0.29 | 3.03 | 2.00 |
| gpt-5-mini | 0.51 | 0.72 | +0.21 | 2.66 | 1.60 |
| gemini-2.5-flash | 0.46 | 0.66 | +0.20 | 3.27 | 2.08 |
| deepseek-v3 | 0.53 | 0.80 | +0.27 | 3.05 | 1.82 |

Quality metrics generally improved: for instance, gpt-5-mini’s Persuasive score increased from 6.40 to 7.15, Logic from 7.81 to 8.28, and Helpful from 7.55 to 8.27. deepseek-v3 saw Persuasive rise from 6.98 to 7.58 and Helpful from 7.84 to 8.42. One exception was gemini-2.5-flash, which saw slight drops in Logic and Helpful, though its Success rate still improved by 0.20.

### Ablation Study
The authors compared the base LLM, an autonomous agent system without meta-cognitive enhancement (+Auto), and the complete MA$^2$P. Results indicate that while multi-agent decomposition increases the success rate, the meta-cognitive configurator further reduces cross-domain volatility.

| Model | Config | Success | Range | SD | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 4o-mini | Base | 0.45 | 0.450 | 0.104 | Monolithic persuader |
| 4o-mini | + Auto | 0.66 | 0.530 | 0.118 | Success rate rises, but domain volatility increases |
| 4o-mini | + MA$^2$P | 0.79 | 0.400 | 0.107 | Highest success rate, Range decreases |
| 4o | Base | 0.46 | 0.500 | 0.114 | Monolithic persuader |
| 4o | + Auto | 0.68 | 0.458 | 0.120 | Success rate rises, SD increases slightly |
| 4o | + MA$^2$P | 0.75 | 0.488 | 0.109 | Success rate continues to rise, SD decreases |

### Knowledge Base Scale & Human Preference
| Setting | Key Results | Implications |
| :--- | :--- | :--- |
| K=0 | Success 0.66, Range 0.53, SD 0.118 | Effective even with cold start, but behaves more like +Auto |
| K=100 | Success 0.73, Range 0.44, SD 0.107 | Noticeable improvement with minimal warm-up |
| K=500 | Success 0.79, Range 0.40, SD 0.107 | Used in main experiments; best overall performance |
| Human Preference | 400 samples, 2 CS Master students; LLM-human weighted Cohen's $\kappa_w=0.549$ | Moderate agreement between LLM and human; both favor MA$^2$P |

### Key Findings
- MA$^2$P improves Success for all five backbone models, indicating that gains are not due to an accidental prompt trick in a specific API model.
- +Auto improves the average success rate but sometimes exacerbates domain gaps; the value of the full MA$^2$P lies in combining "multi-agent execution" with "domain-level strategy selection."
- Warm-up requirements are modest: K=100 yields an improvement from 0.66 to 0.73, though K=500 is most stable in Success and Range.
- Human preferences correlate moderately with the LLM judge, both showing a trend toward MA$^2$P.

## Highlights & Insights
- **Reframing Persuasion from a Generation Problem to a Closed-Loop Control Problem**: Rather than refining prompts, the paper models persuasion as a cycle of perception, world modeling, action, memory, and evaluation. This adds an interpretable strategic layer between "understanding concerns" and "generating utterances."
- **Meta-strategy Selection Addresses Cross-Domain Stability, Not Just Mean Performance**: While +Auto increases success rates, MA$^2$P emphasizes Range/SD. This is crucial as real-world persuasion systems must avoid failure in weak domains rather than just excelling in strong ones.
- **Lightweight Knowledge Base Design**: The Case Layer merely records domain-strategy success counts yet provides an interpretable meta-strategy prior. This "lightweight empirical statistics + prompt scheduling" approach may be more practical for LLM agent systems than complex training.
- **Training-Free but Not Deployment-Free**: The framework shifts training costs to inference-time multi-agent calls and warm-up interaction costs. This trade-off is suitable for high-value, low-throughput tasks like counseling, tutoring, or negotiation assistance but may not be ideal for high-concurrency chatbots.

## Limitations & Future Work
- Automated metrics rely primarily on a gpt-4o-mini judge, while the quality of open-ended persuasion remains subjective. Human preference verification used a limited scale (2 annotators, 400 samples).
- Persuadee simulation remains simplified, conditioning only on beliefs and desires without systematically modeling personality, long-term preferences, values, or trust. Real-world "humans" are far more complex.
- New domains require a warm-up phase to accumulate knowledge base cases; while cold start is functional, optimal results depend on mid-scale experience (e.g., K=500).
- The paper focuses on scenarios with clear user goals (education, counseling), but persuasion technology carries inherent risks of misuse. Deployment for real users requires stronger consent mechanisms, domain restrictions, manipulation risk assessments, and auditable logs.
- Multi-agent scheduling increases inference costs and system complexity; the paper lacks a detailed analysis of latency, token costs, or error propagation.

## Related Work & Insights
- **vs. Monolithic LLM Persuaders**: Monolithic methods generate next turns directly from history, offering simplicity and low cost. MA$^2$P adds explicit mental state extraction, strategy selection, and memory updates, providing better interpretability and stability at the cost of longer inference chains.
- **vs. User State-Aware Persuasion**: Existing methods emphasize identifying user states or selecting psychological strategies; this work further places strategy selection into an updatable knowledge base instantiated turn-by-turn by a World Model.
- **vs. ReAct / Reflexion Agents**: ReAct is more general, emphasizing thought-action-observation. MA$^2$P is task-specific for persuasion, centering on meta-strategies, persuasion principles, and domain-case success counts.
- **Inspiration**: Dialogue agent "memory" need not be full-text history; it can be task-related structured statistics. For scenarios like customer retention or healthcare compliance, MA$^2$P’s domain-strategy success counts could be extended into more rigorous causal or bandit strategy selection mechanisms.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Solid combination of autonomous agent blueprints and meta-cognitive strategy selection for complex persuasion, though core modules rely heavily on prompt scheduling and success counts.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers 5 backbone models, ablations, warm-up, human preference, and case studies; lacks real-world user experiments.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear motivation; methods and algorithms are easy to follow. Some formulas serve more as formal wrappers; lacks system cost analysis.
- **Value**: ⭐⭐⭐⭐☆ Highly valuable for research into training-free, interpretable persuasive agents, especially for those studying complex dialogue planning and cross-domain robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[AAAI 2026\] Emergent Persuasion: Will LLMs Persuade Without Being Prompted?](../../AAAI2026/dialogue/emergent_persuasion_will_llms_persuade_without_being_prompted.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)
- [\[ACL 2026\] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems](stride-ed_a_strategy-grounded_stepwise_reasoning_framework_for_empathetic_dialog.md)

</div>

<!-- RELATED:END -->
