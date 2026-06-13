---
title: >-
  [Paper Note] Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards
description: >-
  [ACL2026][LLM Alignment][Persuasive Dialogue] This paper focuses on hotel price reduction negotiation scenarios for Online Travel Agency (OTA) platforms. It proposes REPO…
tags:
  - "ACL2026"
  - "LLM Alignment"
  - "Persuasive Dialogue"
  - "Multi-source Rewards"
  - "PPO"
  - "Business Negotiation"
  - "Rule Constraints"
date: 2026-05-08
content_hash: d45f855a51b5dfdb
---

# Teaching LLM to be Persuasive: Reward-Enhanced Policy Optimization for Alignment from Heterogeneous Rewards

**Conference**: ACL2026  
**arXiv**: [2510.04214](https://arxiv.org/abs/2510.04214)  
**Code**: None  
**Area**: Task-oriented Dialogue / Alignment RLHF / Industrial Agents  
**Keywords**: Persuasive Dialogue, Multi-source Rewards, PPO, Business Negotiation, Rule Constraints

## TL;DR
This paper focuses on hotel price reduction negotiation scenarios for Online Travel Agency (OTA) platforms. It proposes REPO, which utilizes three types of rewards—Preference Reward Model (RM), LLM Judge (RJ), and Rule Functions (RF)—to jointly train Qwen3-32B. The model demonstrates simultaneous improvements in persuasiveness, SOP compliance, and error correction quality across expert evaluations and 9,653 real-world A/B dialogues.

## Background & Motivation
**Background**: Task-oriented dialogue systems have evolved from traditional intent recognition and slot filling toward LLM-driven multi-turn agents. For business sectors such as customer service, booking, and operational outbound calls, the natural language capabilities of LLMs significantly improve fluency and generalization. However, real-world deployment still requires adherence to SOPs, boundary rules, and business metrics.

**Limitations of Prior Work**: The scenario in this paper involves price pursuit outbound calls for an OTA platform. BD agents need to persuade hotel managers to lower prices without making exaggerated promises, leaking internal ticket terminology, or confusing selling prices with net prices, all while advancing through a multi-stage SOP. Simple SFT tends to learn rigid scripts; DPO is limited by the coverage of preference data; and standard PPO/GRPO may suffer from reward hacking or breach of compliance if the reward design is insufficient.

**Key Challenge**: Persuasive business dialogue involves both verifiable constraints and soft skills that are difficult to formalize. Numbers, formats, and forbidden words can be checked via rules, but "soothing emotions," "smooth transitions," and "linking price reductions to exposure/conversion" are difficult to express via regular expressions. Conversely, relying solely on an LLM judge or preference RM makes it difficult to stably handle price values and internal terminology leaks.

**Goal**: The authors aim to construct an RL post-training framework that allows for rapid online iteration, where rewards retain the main direction of human preferences while quickly injecting new issues discovered by business experts via prompt rubrics and rules.

**Key Insight**: This work treats rewards as originating from three complementary sources: RM provides dense human preference signals, RJ uses LLM-as-a-judge to evaluate emotional value, SOP, and negotiation strategies, and RF uses rule functions to check values, formats, and guardrails. The key to REPO is not simple addition, but rather using RJ/RF as bounded enhancement terms to modulate the RM.

**Core Idea**: Using the preference reward model as the primary signal, LLM judge and rule functions are aggregated and clipped as bounded multipliers to perform sign-sensitive amplification or weakening of the RM reward. This injects business constraints and persuasive behavioral preferences into stable RL training.

## Method
REPO targets a specific but representative industrial dialogue task: a platform BD proactively contacts a hotel to negotiate price reductions for one or more chase tickets. The model must judge the current SOP stage based on the counterparty's response, distinguish between selling price and net price when quoting, stop persuasion when an acceptable price is offered, avoid misinterpreting silence as consent, and prevent the leakage of internal terms like "work-order" or "target price."

### Overall Architecture
Given a dialogue history and task information, the policy model generates the next BD response. This response is simultaneously sent to three reward components. The Reward Model provides an alignment score based on preference data. The Reward Judge evaluates emotional value, dialogue style, SOP compliance, negotiation progress, and out-of-bounds behavior according to a task rubric. The Reward Function uses regular expressions or deterministic rules to check formats, mixed languages, internal leaks, forbidden words, repetitive scripts, output length, and price constraints.

The signals from these three sources are integrated into a total reward for RL training. The process follows a PPO-style actor-critic workflow: the policy model generates output, the value model estimates state value, and rewards form the advantage via GAE to update the policy. The authors emphasize that all RL baselines use the same LoRA configuration, training budget, and hyperparameters, ensuring that gains primarily stem from the reward design.

### Key Designs
1.  **Functional Division of Triple-Source Rewards**:
    - **Function**: Assigns different business requirements to different reward types, preventing a single reward source from being burdened with both stylistic understanding and hard rule checking.
    - **Mechanism**: The RM is trained on 6,632 high-quality preference samples, covering online cases, linguistic expert annotations, task expert annotations, and SOP path samples, responsible for learning natural human BD expressions. The RJ is an LLM evaluator using editable rubrics for scoring, covering complex behaviors like appeasement, continued pursuit, and misjudgment of no-response. The RF consists of rule functions that quickly capture deterministic issues like internal terms, format breakage, and CoT leakage.
    - **Design Motivation**: In hotel price negotiation, soft skills and hard constraints are intertwined. Entrusting everything to the RM is limited by data coverage, while relying solely on rules fails to learn persuasion techniques. The division of labor allows the system to learn to "speak like an excellent BD" while providing quick patches for deployment incidents.

2.  **Bounded Reward Enhancement**:
    - **Function**: Enables RJ/RF to correct the RM instead of being linearly added, which could cause reward scale instability.
    - **Mechanism**: REPO first aggregates the judge scores and rule scores into $E_{judge} + E_{func}$, then clips them to $[-n, n]$ to obtain $E_{enh}$. The total reward is defined as $R_{total} = R_{model}(1 \pm E_{enh}/n)$, with $n=100$ in experiments. If the RM is positive and the auxiliary signal is also positive, the reward is amplified; if RM is positive but the auxiliary signal is negative, the reward is weakened; if RM is negative and the auxiliary signal is negative, the penalty is amplified; and if RM is negative but the auxiliary signal is positive, the penalty is mitigated.
    - **Design Motivation**: The RM identifies the primary direction, while RJ/RF serve as stability supplements. This design prevents a specific rule score from overwhelming the preference model and allows business patches to influence the policy via "modulation," reducing RL training oscillations and reward hacking.

3.  **Deployment-Driven Rapid Iterative Loop**:
    - **Function**: Allows issues observed online to quickly enter training signals without re-collecting large-scale preference data.
    - **Mechanism**: If it is discovered online that the model only handles one room type in a multi-ticket dialogue or leaks internal words like "work-order," the team can update the scoring rules in the RJ prompt or add a new RF regular expression, then continue RL training. The RM provides the baseline human preference, while RJ/RF handle patches for new error types.
    - **Design Motivation**: Constraints in industrial dialogue change with business processes, and annotated data usually cannot keep up. REPO’s iterative mechanism transforms reward engineering into an auditable and quickly modifiable engineering interface.

### Loss & Training
The base model is Qwen3-32B-Instruct, with a maximum response length of 512 and a batch size of 128. The LoRA rank is 64, LoRA alpha is 64, the learning rate is $10^{-6}$, and the warmup steps are 2. SFT and DPO are trained for 10 epochs, while PPO, GRPO, and REPO are trained for 2 epochs, with the best checkpoints reported.

The training data consists of 6,632 preference samples: 252 from production collected by business experts, 3,178 annotated by linguistic experts, 1,211 annotated by task experts (human BDs), and 1,991 SFT samples covering full SOP paths later expanded into preference data. Evaluation includes 30 full online dialogues (~240 turns) and 45 bad cases curated by business experts (~360 turns).

## Key Experimental Results

### Main Results
An expert consensus evaluation was conducted by task experts, linguistics experts, and computer science experts. The online test set focused on overall dialogue ratings and the presence of at least one excellent response; the bad case set focused on recovery rate and clean recovery ratio.

| Method | Online Dialogue Score | Excellent Response Ratio | Bad Case Recovery Rate | Clean Recovery Ratio | Main Conclusion |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Base | 3.43 | 13.33% | N/A | N/A | Weak persuasion capability in the original model |
| SFT | N/A | N/A | 93.33% | 31.11% | Covers many bad cases but repair quality is coarse |
| DPO | 3.80 | 33.33% | 93.33% | 40.00% | More stable than SFT, but low excellent response ratio |
| PPO | < Baseline | < Baseline | 86.66% | 33.33% | Vanilla PPO degrades online metrics |
| GRPO | 4.30 | 43.33% | 71.10% | 44.44% | Decent dialogue quality but insufficient bad case coverage |
| REPO | 4.63 | 66.67% | 93.33% | 75.56% | Best performance in score, excellent responses, and clean repairs |

Production A/B tests compared REPO only with the existing intent-driven dialogue system, as other end-to-end post-trained models failed to meet stability and compliance requirements for deployment.

| Online Metric | Production Intent System | REPO | Gain | Significance | Explanation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Sample Size | 9653 Real Dialogues | 9653 Real Dialogues | - | - | Real customer/hotel traffic |
| Response Rate | 46.72% | 58.86% | +12.14 pp | $p < 0.001$ | Reflects naturalness and acceptance after proxy's first sentence |
| Task Success Rate | 19.32% | 25.26% | +5.94 pp | $p < 0.001$ | Successful price negotiation per ticket; reflects persuasion |
| Human BD Comparability | N/A | ~25% Success Rate | - | - | Success rate approaching human BD levels |

### Ablation Study
The paper does not provide a traditional ablation table removing RM/RJ/RF one by one; however, the benefits of REPO are demonstrated through bad case distribution, training progress, and fine-grained skill assessments.

| Analysis Item | Result | Explanation |
| :--- | :--- | :--- |
| Serious Unresolved Bad Cases | REPO/PPO/SFT: 0; GRPO: 4.44%; DPO: 2.22% | REPO maintains high recovery while avoiding severe residual errors |
| Major Problem Repair | REPO 4.44%, GRPO 13.33%, DPO/PPO 31.11%, SFT 42.22% | REPO repairs are cleaner, not just replacing one error with another |
| Negotiation Ability in Training | Evaluated by DeepSeek-R1; late checkpoints increased by ~+14 points (~30%) | Multi-source rewards pushed negotiation strategies beyond training "gold" |
| Fine-grained Skills | REPO leads in negotiation effectiveness across online and bad case sets | Rewards improve business goal attainment, not just fluency |

### Key Findings
- The standout advantage of REPO is not merely "repairing more bad cases" (since SFT and DPO also reached 93.33% recovery); the true difference lies in the **clean recovery rate**, where REPO reached 75.56%, indicating it introduces fewer new problems.
- PPO experienced degradation in this task, suggesting that reinforcement learning is not a "guaranteed button" for gains. Without a stable, controllable reward combination, RL can damage dialogue quality.
- GRPO had high online scores but only a 71.10% bad case recovery rate, indicating that group rewards or relative optimization may not suit scenarios with strong business constraints, especially when error distributions differ from online distributions.
- The online A/B test is the most persuasive part of the paper: REPO did not just score higher on LLM judges but improved response and success rates across 9,653 real-world dialogues.

## Highlights & Insights
- The greatest value of this paper lies in its industrialized approach to reward engineering. While many RLHF papers only compare offline benchmarks, this work clearly demonstrates how "human preferences, LLM rubrics, and rule patches" function together in real business.
- REPO’s multiplicative enhancement is more stable than simple weighted summation. It maintains the dominance of the RM while allowing RJ/RF to modify rewards sensitivity based on signs, fitting scenarios with different multi-source reward scales and update frequencies.
- While the task definition is specific, it reveals universal difficulties for dialogue agents: being persuasive while following rules, and handling soft emotions without making hard errors in numbers or processes.
- The prompts and reward mappings in the appendix are highly practical. They translate rules such as "do not count no-response as consent" and "accept immediately if quote is below target" into checkable items, providing a template for other enterprise agents.

## Limitations & Future Work
- Experiments are focused on OTA hotel price negotiation. While realistic, the task is narrow; whether REPO generalizes to after-sales, finance, medical consulting, or cross-linguistic business negotiation needs broader validation.
- The paper lacks a full numerical ablation study removing RM/RJ/RF sequentially, so the precise contribution of each source is not fully quantified.
- The RJ relies on LLM-as-a-judge, which may have biases or be affected by prompt phrasing. While online A/B tests support the final effect, the stability of the reward judge requires long-term monitoring.
- Online data and original business prompts are desensitized, making it difficult for external researchers to replicate experiments. Future work could build public constrained negotiation benchmarks to lower the validation threshold.

## Related Work & Insights
- **vs SFT**: SFT is stable but only mimics existing scripts and tends to learn fixed patterns. REPO, guided by rewards, generates richer emotional value and competitive argumentation compared to gold labels.
- **vs DPO**: DPO depends on preference pairs and is computationally efficient, but its upper bound is restricted by data coverage. REPO converts expert rules and deployment incidents directly into rewards for faster iteration.
- **vs PPO / GRPO**: While PPO and GRPO are optimization algorithms, REPO's contribution is in reward composition. Experiments show that in constrained dialogue, reward design is more critical than the choice of RL algorithm.
- **vs Traditional Intent-Driven Systems**: Intent systems are stable and controllable but struggle with natural persuasion and long-range adaptation. REPO retains the flexible language capability of LLMs while using rules and judges for compliance control.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The components of the triple-source rewards are not entirely new, but the bounded enhancement mechanism and deployment-driven iteration loop highly address industrial pain points.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes expert evaluation, bad case sets, and real A/B tests; lacks precise ablation by removing individual reward sources.
- Writing Quality: ⭐⭐⭐⭐☆ Task background and results are clear, with rich prompt appendices; however, some chart values are not fully tabulated.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for enterprise task-oriented dialogue, sales agents, and constrained RLHF, particularly regarding reward design workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Topology-Enhanced Alignment for Large Language Models: Trajectory Topology Loss and Topological Preference Optimization](topology-enhanced_alignment_for_large_language_models_trajectory_topology_loss_a.md)
- [\[ICLR 2026\] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](../../ICLR2026/llm_alignment/safedpo_preference_optimization_safety.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](../../ICLR2026/llm_alignment/mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ACL 2026\] MDP-GRPO: Stabilized Group Relative Policy Optimization for Multi-Constraint Instruction Following](mdp-grpo_stabilized_group_relative_policy_optimization_for_multi-constraint_inst.md)
- [\[ICLR 2026\] Is On-Policy Data always the Best Choice for Direct Preference Optimization-based LM Alignment?](../../ICLR2026/llm_alignment/is_on-policy_data_always_the_best_choice_for_direct_preference_optimization-base.md)

</div>

<!-- RELATED:END -->
