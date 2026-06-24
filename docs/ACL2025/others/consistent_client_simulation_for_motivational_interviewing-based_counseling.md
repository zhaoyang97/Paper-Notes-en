---
title: >-
  [Paper Note] Consistent Client Simulation for Motivational Interviewing-based Counseling
description: >-
  [ACL 2025][Client Simulation] This paper proposes a consistent client simulation framework for motivational interviewing (MI) psychological counseling. Through four modules—state transition, action selection, information selection, and response generation—it ensures that the behavior of simulated clients aligns with their predefined profiles (motivations, beliefs, change plans, receptivity), outperforming baseline methods in both automatic and expert evaluations.
tags:
  - "ACL 2025"
  - "Client Simulation"
  - "Motivational Interviewing"
  - "Consistency"
  - "State Tracking"
  - "Action Selection"
  - "LLM Agent"
date: 2026-05-08
content_hash: cebd3b7019d98285
---

# Consistent Client Simulation for Motivational Interviewing-based Counseling

**Conference**: ACL 2025  
**arXiv**: [2502.02802](https://arxiv.org/abs/2502.02802)  
**Code**: None  
**Authors**: Yizhe Yang, Palakorn Achananuparp, Heyan Huang, Jing Jiang, John Pinto, Jenny Giam, Kit Phey Leng, Nicholas Gabriel Lim, Cameron Tan Shi Ern, Ee-peng Lim  
**Affiliations**: Beijing Institute of Technology, Singapore Management University, Australian National University, etc.  
**Area**: Other  
**Keywords**: Client Simulation, Motivational Interviewing, Consistency, State Tracking, Action Selection, LLM Agent

## TL;DR

This paper proposes a consistent client simulation framework for motivational interviewing (MI) psychological counseling. Through four modules—state transition, action selection, information selection, and response generation—it ensures that the behavior of simulated clients aligns with their predefined profiles (motivations, beliefs, change plans, receptivity), outperforming baseline methods in both automatic and expert evaluations.

## Background & Motivation

**Background**: The training of psychological counselors traditionally requires the participation of human clients. To reduce costs, researchers have begun using LLM-simulated client agents. However, existing methods primarily focus on evaluating counselor agents, neglecting the consistency and quality of the client simulation.

**Limitations of Prior Work**:
   - Simple profile prompting (Yosef et al., 2024; Wang et al., 2024a) or conversational example prompting (Chiu et al., 2024) leads to four types of inconsistency:
     - **(a) Motivational Inconsistency**: The client's reasons for agreeing to change do not align with their predefined motivations.
     - **(b) Belief Inconsistency**: The client fails to adhere to their predefined beliefs.
     - **(c) Plan Inconsistency**: The client accepts a change plan that does not exist in their profile.
     - **(d) Receptivity Inconsistency**: The client's level of cooperation with the counselor does not match their predefined settings.
   - LLMs (such as ChatGPT) possess an inherent tendency to generate highly compliant responses (a side effect of safety/alignment), leading to overly-compliant simulated clients with monotonous behaviors.

**Design Motivation**: There is a need for fine-grained control over the state transitions and action selections of simulated clients across different stages of counseling, ensuring their profiles and behaviors remain highly consistent with real clients.

## Method

### Overall Architecture

The framework comprises four core modules that are executed sequentially in each dialogue turn:

1. **State Transition Module** -> Determines the client's next state.
2. **Action Selection Module** -> Selects the type of client action.
3. **Information Selection Module** -> Selects the profile information to disclose.
4. **Response Generation Module** -> Generates the client's utterance.

### Client Profile Definition

The client profile consists of the following elements:

| Element | Description |
|------|------|
| Behavioral Issue | The behavioral problem the client needs to address |
| Initial/Terminal State | Psychological state before and after counseling |
| Persona | Client background information |
| Motivation | Specific reasons driving the change |
| Beliefs | Beliefs that hinder the change |
| Plans | Specific behavioral changes the client may agree to |
| Receptivity | The level of cooperation with the counselor, rated 1-5 |

### Key Design 1: State Transition

Based on the Transtheoretical Model, three core states plus termination are defined:

- **Precontemplation** -> Denying the issue. Transition to Contemplation occurs if the counselor mentions the motivations in the client profile.
- **Contemplation** -> Recognizing the issue but hesitating. Transition to Preparation occurs if the belief barriers in the profile are adequately addressed.
- **Preparation** -> Beginning to plan changes. Transition to Termination occurs if the preferred change plan has been discussed with the counselor.
- **Termination** -> Ending the session.

State transitions strictly depend on whether the counselor addresses key content in the client profile, thereby ensuring consistency.

### Key Design 2: Action Selection

A fusion strategy combining two action distributions is introduced:

1. **Context-aware distribution**: Infers an appropriate action distribution based on the current dialogue context using the LLM.
2. **(State, Receptivity)-aware distribution**: Learns the action distribution for each (state, receptivity) combination from the real AnnoMI dataset.

The final action distribution is the average of the two, from which sampling is performed among candidate actions relevant to the current state.

### Key Design 3: Information Selection

Actions are categorized into two types:
- **Type-1** (e.g., Deny, Engage, Accept): Generated without requiring profile information.
- **Type-2** (e.g., Inform, Blame, Hesitate, Plan): Requires selecting relevant information from the profile.

The information selection module prevents the client from disclosing too much profile information at once, avoiding unrealistically shortened counseling sessions.

### Data Annotation

Based on the AnnoMI dataset (consisting of real MI counseling dialogues), 86 clients and sessions were selected:
- GPT-4 was leveraged to annotate the four profile elements, states, actions, and receptivity.
- Quality verification of annotations: 87.31% accuracy for states, 85.20% for actions, and 80.32% for receptivity.
- All profile items were manually verified for factual accuracy.

## Key Experimental Results

### Experimental Setup

- LLM backbone model: gpt-3.5-turbo-0125
- Counselor agent: LLM agent based on MI knowledge prompting
- Facilitator agent: Monitors dialogue termination conditions
- 3 sessions generated per client profile -> 258 generated sessions in total

### Baseline Methods

1. **Base**: Simple prompt containing only the behavioral issue.
2. **Example-based** (Chiu et al., 2024): Prompts containing real session examples.
3. **Profile-based** (Yosef et al., 2024): Prompts containing client profiles.
4. **Pro+Act-based** (Zhang et al., 2024): Profiles + action descriptions.

### Profile Consistency Evaluation (Automatic Evaluation, GPT-4 Entailment Decision)

| Method | Persona↑ | Motivation↑ | Belief↑ | Plan↑ | Receptivity↑ |
|------|-------|-------|-------|-------|---------|
| Base | 9.01 | 16.17 | 12.15 | 9.30 | −0.31 |
| Example-based | 53.68 | 45.73 | 45.55 | 33.53 | 0.25 |
| Profile-based | 61.97 | 53.44 | 67.17 | 54.67 | 0.31 |
| Pro+Act-based | 67.09 | 55.33 | 68.60 | 57.17 | 0.33 |
| **Ours** | **70.57** | **73.37** | **71.70** | **68.51** | **0.58** |

Ours achieves the best results across all five consistency dimensions.

### Dialogue Behavior Analysis

| Method | Mean Receptivity | Motivation Rate (20 steps) | Mean Motivation Turn | Action KL↓ |
|------|-----------|-----------|-------------|---------|
| Base | 4.42±0.47 | 1.00 | 6.60 | 0.39 |
| Profile-based | 4.12±0.64 | 0.96 | 9.76 | 0.15 |
| Pro+Act-based | 3.86±1.01 | 0.94 | 9.93 | 0.13 |
| **Ours** | **3.32±1.15** | **0.69** | **18.60** | **0.06** |
| Real Client | 3.27±1.12 | 0.48 | 27.56 | 0.00 |

Ours demonstrates behavioral distributions (receptivity, motivation rate, action KL divergence) closest to real clients.

### Expert Evaluation (1-5 scale, 6 clients × 4 profile dimensions × 3 annotators)

| Method | Persona | Belief | Motivation | Plan | Realism |
|------|------|------|------|------|--------|
| Profile-based | 2.61 | 2.00 | 2.61 | 1.56 | 2.38 |
| Pro+Act-based | 2.65 | 2.22 | 2.78 | 1.56 | 2.50 |
| **Ours** | **3.33** | **2.89** | **3.00** | **2.27** | **3.16** |
| Real Client | 4.72 | 4.67 | 4.56 | 4.61 | 4.72 |

### Receptivity Control Evaluation

| Method | Receptivity=1 | Receptivity=3 | Receptivity=5 | Correlation Coeff. |
|------|---------|---------|---------|---------|
| Pro+Act-based | 5.0 | 4.3 | 5.0 | 0.00 |
| **Ours** | **1.3** | **3.0** | **4.3** | **0.86** |

Ours effectively controls client behavior across different receptivity levels (Spearman correlation 0.86, p=0.0003), whereas baseline methods almost completely fail to differentiate receptivity settings.

## Highlights & Insights

1. **First to systematically define** consistency dimensions (motivation, beliefs, plans, receptivity) for psychology client simulation.
2. **Exquisite state transition design**: Embedding the Transtheoretical Model of MI into a state machine, directly binding state transitions to profile content.
3. **Controllable receptivity** is an important practical feature—training counselors requires interacting with clients of varying difficulty levels.
4. **Action distribution fusion** (context-aware + data-driven) balances dialogue fluency and profile consistency.
5. **Identification of the over-compliance issue** in baseline methods, revealing the side effects of LLM alignment in role-playing scenarios.

## Limitations & Future Work

1. **Error accumulation in multi-step prompting**: The framework relies on sequential LLM prompts, exhibiting high sensitivity to prompt design.
2. **Exclusively focused on client simulation**, without optimizing or evaluating counselor agents.
3. **Evaluated only on the AnnoMI dataset**, restricted to the single counseling methodology of MI.
4. **Significant gap remains between simulated and real clients** (expert evaluation score of 3.16 vs 4.72).
5. **Relatively short conversation length**: Baseline methods terminate rapidly due to over-compliance, whereas the proposed method can cause the counselor to give up early due to rigid state control.

## Related Work

- **Motivational Interviewing**: Miller & Rollnick (2012), Transtheoretical Model (Prochaska & Velicer, 1997)
- **Client Agent Simulation**: PATIENT-Ψ (Wang et al., 2024b), State-Aware Patient Simulator (Liao et al., 2024)
- **LLM Dialogue**: Chiu et al. (2024) computational framework evaluating LLM counselors

## Rating

⭐⭐⭐⭐ (4/5)

This work presents an in-depth study in the niche area of mental health client simulation. The multi-dimensional definition of consistency is highly valuable, and controllable receptivity is a notable highlight. While the evaluation is thorough (automatic + expert), the experiments are limited to a single dataset and counseling methodology, and the gap between simulated and real clients has yet to be bridged.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Hanging in the Balance: Pivotal Moments in Crisis Counseling Conversations](hanging_in_the_balance_pivotal_moments_in_crisis_counseling_conversations.md)
- [\[ACL 2025\] GA-S3: Comprehensive Social Network Simulation with Group Agents](ga-s3_comprehensive_social_network_simulation_with_group_agents.md)
- [\[ICLR 2026\] Consistent Low-Rank Approximation](../../ICLR2026/others/consistent_low-rank_approximation.md)
- [\[ACL 2025\] Are Any-to-Any Models More Consistent Across Modality Transfers Than Specialists?](are_any-to-any_models_more_consistent_across_modality_transfers_than_specialists.md)
- [\[ACL 2025\] CORAL: Learning Consistent Representations across Multi-step Training with Lighter Speculative Drafter](coral_speculative_drafting.md)

</div>

<!-- RELATED:END -->
