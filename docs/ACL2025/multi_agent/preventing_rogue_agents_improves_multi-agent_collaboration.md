---
title: >-
  [Paper Note] Preventing Rogue Agents Improves Multi-Agent Collaboration
description: >-
  [ACL 2025][Multi-Agent][Multi-Agent Collaboration] A framework is proposed to detect "rogue agents" by monitoring agent uncertainty in real-time and to intervene accordingly. This framework achieves performance improvements of up to 17.4%, 2.5%, and 20% on the self-built WhoDunitEnv multi-agent collaboration environment, code generation tasks, and resource sustainability tasks, respectively.
tags:
  - "ACL 2025"
  - "Multi-Agent"
  - "Multi-Agent Collaboration"
  - "Anomaly Detection"
  - "Uncertainty Estimation"
  - "Intervention Mechanism"
  - "WhoDunitEnv"
date: 2026-05-08
content_hash: 5ebfc73ac35d1fbe
---

# Preventing Rogue Agents Improves Multi-Agent Collaboration

**Conference**: ACL 2025  
**arXiv**: [2502.05986](https://arxiv.org/abs/2502.05986)  
**Code**: [Available](https://github.com/Ohav/rogue-agents)  
**Area**: Others  
**Keywords**: Multi-Agent Collaboration, Anomaly Detection, Uncertainty Estimation, Intervention Mechanism, WhoDunitEnv

## TL;DR

A framework is proposed to detect "rogue agents" by monitoring agent uncertainty in real-time and to intervene accordingly. This framework achieves performance improvements of up to 17.4%, 2.5%, and 20% on the self-built WhoDunitEnv multi-agent collaboration environment, code generation tasks, and resource sustainability tasks, respectively.

## Background & Motivation

Multi-agent systems (MAS) enable multiple specialized agents to collaborate on shared tasks, demonstrating great potential in areas such as reasoning enhancement, software development simulation, and human behavior simulation. However, they also possess a core risk: **a single rogue agent can cause the entire system to fail**.

Typical issues currently faced by LLM agents in collaboration include:

- **Ignoring critical information**: Agents may overlook important messages in communication.
- **Distraction by irrelevant information**: Generating hallucinations or introducing noise.
- **Error propagation**: Errors from a single agent propagate through communication channels and are amplified over multi-round interactions.
- **Premature actions**: Agents may make irreversible, terminal decisions when information is insufficient.

Although existing improvement methods (such as modifying communication protocols, incorporating belief systems, or adding reasoning modules) are helpful, none can stop a "rogue agent" from dragging the entire system down into failure.

Inspired by industrial monitoring systems (intrusion detection, manufacturing quality control) and biological immune systems, this paper proposes to **detect potential failures and proactively intervene before agents act**, rather than performing post-hoc fixes.

## Method

### Overall Architecture

The framework consists of two core components:

1. **Monitor**: Predicts the probability of system failure based on agent uncertainty signals.
2. **Intervention**: Triggers environmental intervention to prevent error propagation when the failure probability exceeds a threshold $\tau$.

### Key Designs

1. **Uncertainty-Based Monitoring**:

    - **Feature Extraction**: When an agent generates an action, the maximum values of three statistics—**entropy**, **varentropy**, and **kurtosis**—are extracted from the output probability distribution $\mathbf{p}_i$ at critical positions (action selection and prior thought). Combining these with the current round count yields a total of $m \le 4$ features.
    - **Classifier**: A simple polynomial ridge regression classifier is used to fit the success probability $f: \mathbb{R}^m \to [0,1]$, trained using Boolean labels for each round on the training data.
    - **Trigger Condition**: An intervention is triggered when $P(\text{success}) < \tau$.
    - **Design Motivation**: An agent being "confused" (high entropy) during action selection implies that it may introduce noise, leading to system failure. A simple classifier is selected because the feature dimension is extremely low ($m \le 4$), and experiments demonstrate it is sufficiently effective.

2. **Intervention Strategy**:

    - Distinguishes between **reversible actions** (information sharing) and **irreversible actions** (accusing a suspect, submitting code, consuming resources).
    - During an intervention, the system rolls back to the state immediately after the latest irreversible action—i.e., revoking all reversible operations to give agents a fresh chance to collaborate.
    - In WhoDunitEnv: Information sharing is reversible, accusing is irreversible, and intervention corresponds to resetting the entire communication.
    - In CodeGen: Code generation is irreversible, and intervention corresponds to the Judge and Tester rewriting feedback.
    - In GovSim: Resource consumption is irreversible, and intervention corresponds to resetting the discussion of the previous round.

3. **WhoDunitEnv Environment Design**:

    - Inspired by the board game *Guess Who*, agents act as detectives collaborating to identify a culprit.
    - **Asymmetric Variant (Asym)**: Comprises two agents—the Accuser (knows the culprit description, can ask questions and accuse) and the Intel (knows all suspect descriptions, can reply).
    - **Symmetric Variant (Sym)**: All agents are equal, each holding partial clues (3 clues) about the culprit, and they can share, accuse, or skip.
    - **Adjustable Complexity**: Crucial parameters include the number of suspects (6/10/14/20), the number of attributes, and the maximum number of rounds.
    - Uses ReAct prompts to encourage agents to "think-before-acting".

### Loss & Training

- The monitor uses polynomial ridge regression, trained on the intermediate states of each round in the training set (positive/negative labels correspond to final success/failure).
- Grid search is conducted on the validation set to obtain the optimal feature combination, polynomial degree $d \in [1,5]$, and threshold $\tau \in [0,1]$.
- The number of interventions is capped (1–2 times in WhoDunitEnv, 1 time in CodeGen/GovSim).

## Key Experimental Results

### WhoDunitEnv-Asym Main Results (Success Rate)

| Model | No Intervention | + Monitor & Intervention (1 reset) | + Monitor & Intervention (2 resets) |
|------|--------|---------------------|---------------------|
| GPT-4o | ~62% | +6.1% | +11.8% (final) |
| Llama-3.1-70B | ~60% | +10.6% | — |
| Qwen-2.5-72B | ~60% | +10.3% | — |

### CodeGen Experiments (Pass@1, Llama-3.1-70B)

| Method | HumanEval | LiveCodeBench |
|------|-----------|---------------|
| Zero-shot | 80.5% | 18.2% |
| Multi-agent (no monitor) | 81.6% | 19.3% |
| **Multi-agent + Monitor** | **83.5%** | **21.8%** |

### GovSim Experiments

| Model | Method | Survival Rate | Efficiency |
|------|------|---------------|------------|
| Qwen-1.5-110B | No Intervention | 35.0% | 49.4% |
| Qwen-1.5-110B | **+ Monitor** | **55.0%** | 48.8% |
| GPT-4o | No Intervention | 100% | 69.1% |
| GPT-4o | **+ Monitor** | 100% | **76.0%** |

### Ablation Study (Qwen-2.5-72B, WhoDunitEnv-Asym)

| Variant | Success Rate |
|------|-------------|
| No Intervention | 59.8% |
| Best Baseline (random reset) | 62.5% |
| Worst Monitor | 62.0% |
| Action = resample agent (instead of resetting communication) | 61.3% |
| Second-best Monitor | 69.3% |
| **Best Monitor (single reset)** | **70.1%** |
| **Best Monitor (double reset)** | **72.2%** |

### Key Findings

1. **Monitoring + Intervention is consistently effective**: Significant improvements are observed across all environments and models, up to 17.4% in WhoDunitEnv and 20% in GovSim.
2. **Monitor quality is crucial**: In the ablation study, the worst monitor only achieves a 2% improvement (close to the 2.6% improvement of the random baseline), whereas the best monitor yields a 10.3% gain, indicating that accurate failure prediction is key.
3. **The intervention scheme is equally important**: Resampling a single agent (without resetting communication) only improves performance by 1.5%, which is far inferior to resetting communication channels. This suggests that the root cause lies in polluted communication rather than a single agent.
4. **Monitors generalize well**: The monitor trained on HumanEval also performs successfully on LiveCodeBench (+2.5%); monitors trained with a fixed set of 10 suspects remain consistently effective on environments with 6 or 14 suspects.
5. **Hallucination is the most common trigger cause**: Qualitative analysis of 50 trigger incidents reveals that 48% are due to hallucinations, 16% to agent breakdown (repeating the same action), 8% to loss of role, and 4% to failure in information recall, making up 76% of cases falling into these four error categories.

## Highlights & Insights

- **Extremely Simple yet Effective Monitor**: Using only $\le 4$ features combined with a polynomial ridge regression is highly effective in predicting multi-agent system failures, indicating that LLM uncertainty signals (e.g., entropy) serve as strong indicators of failure.
- **Elegant "Immune System" Analogy**: Porting the real-time detection-intervention paradigm from industrial monitoring and biological immune systems to LLM multi-agent systems is conceptually novel and highly practical.
- **Well-designed WhoDunitEnv**: The modular symmetric/asymmetric variants, adjustable complexity, and structured action space provide an excellent testbed for researching multi-agent communications.
- **Detailed Analysis of Trigger Causes**: The comprehensive qualitative analysis (Figure 7) not only validates the effectiveness of the method, but also provides a taxonomy of LLM collaboration failures (hallucination > breakdown > loss of role > recall failure).
- **Honest Cost Analysis**: Explicitly reports the extra inference cost introduced by interventions (averaging 1.6–1.9 times more rounds).

## Limitations & Future Work

- It requires sufficient successful cases in the training data to train the monitor—rendering the approach inapplicable to scenarios like Llama-3-70B on GovSim where the baseline success rate is near zero.
- The intervention mechanism is relatively simplistic (global reset), without exploring more fine-grained local recovery strategies (e.g., only rolling back messages from the problematic agent).
- The evaluation environments remain relatively simple (such as the Guess Who game and single-file code generation); the efficacy in more complex open-ended collaborative scenarios (e.g., full software development) remains to be verified.
- For closed-source models, monitoring precision may be limited since one can only obtain top-k token probabilities as approximations.
- The monitor is currently trained offline; exploring online learning to reduce the sample demands of the monitor itself is a viable future direction.

## Related Work & Insights

This work bridges three main directions: (1) multi-agent communication protocol design (Li et al. 2023; Hong et al. 2024), (2) language-model uncertainty estimation (Kadavath et al. 2022; Yona et al. 2024), and (3) multi-generation aggregation (Wang et al. 2023; Du et al. 2024). Distinct from the post-hoc aggregation strategies of the latter, this paper focuses on "detection and intervention before irreversible actions," which is better suited for agent environments where actions yield consequences. The key insight is that for any multi-agent system, integrating a lightweight "immune layer" to monitor communication quality and agent states could serve as a universal strategy to enhance robustness.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The approach of real-time monitoring + intervention is systematically proposed and validated for the first time in multi-agent LLM systems, and WhoDunitEnv is also a valuable contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Very comprehensive, covering 3 environments, 3+ models, multiple complexity levels, ablation studies, monitor quality analysis, qualitative analysis, and generalization tests.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear concepts, intuitive illustrations, precise formal definitions, and detailed environment descriptions.
- **Value**: ⭐⭐⭐⭐ — Provides a plug-and-play enhancement strategy for multi-agent collaboration, which has direct practical value for increasingly popular agent systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-Agent Collaboration via Cross-Team Orchestration](multi-agent_collaboration_via_cross-team_orchestration.md)
- [\[ACL 2025\] Beyond Frameworks: Unpacking Collaboration Strategies in Multi-Agent Systems](beyond_frameworks_multi_agent_collaboration.md)
- [\[ICLR 2026\] Graph-of-Agents: A Graph-based Framework for Multi-Agent LLM Collaboration](../../ICLR2026/multi_agent/graph-of-agents_a_graph-based_framework_for_multi-agent_llm_collaboration.md)
- [\[ACL 2026\] Multi-Agent Reasoning Improves Compute Efficiency: Pareto-Optimal Test-Time Scaling](../../ACL2026/multi_agent/multi-agent_reasoning_improves_compute_efficiency_pareto-optimal_test-time_scali.md)
- [\[NeurIPS 2025\] Multi-Agent Collaboration via Evolving Orchestration](../../NeurIPS2025/multi_agent/multi-agent_collaboration_via_evolving_orchestration.md)

</div>

<!-- RELATED:END -->
