---
title: >-
  [Paper Note] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation
description: >-
  [ACL 2026][Medical Imaging][Emotional Support Conversation] This paper proposes CoPoLLM, a framework that constructs CogBiasESC — the first emotional support conversation dataset annotated with cognitive distortions — an…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Emotional Support Conversation"
  - "Cognitive Distortion"
  - "Cognitive Behavioral Therapy"
  - "Reinforcement Learning Policy"
  - "Safe Intervention"
date: 2026-05-08
content_hash: b6ec90c14b7a1632
---

# Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation

**Conference**: ACL 2026
**arXiv**: [2604.17178](https://arxiv.org/abs/2604.17178)
**Code**: [https://github.com/Chips98/CoPoLLM-for-ACL-2026](https://github.com/Chips98/CoPoLLM-for-ACL-2026)
**Area**: Medical Imaging
**Keywords**: Emotional Support Conversation, Cognitive Distortion, Cognitive Behavioral Therapy, Reinforcement Learning Policy, Safe Intervention

## TL;DR

This paper proposes CoPoLLM, a framework that constructs CogBiasESC — the first emotional support conversation dataset annotated with cognitive distortions — and integrates a Cognitive Policy Reinforcement Learning (CPRL) engine with Dual-Stream Conditional Optimization (DSCO) to enable LLMs to diagnose eight types of cognitive distortions and generate strategy-aware intervention responses, achieving state-of-the-art performance over 15 baselines.

## Background & Motivation

**Background**: LLMs have demonstrated promising empathetic capabilities in Emotional Support Conversation (ESC) tasks. Methods such as SoulChat and ChatCounselor have made progress in fluency and empathy via SFT or DPO. However, professional psychological counseling requires not only emotional comfort but also cognitive intervention grounded in Cognitive Behavioral Therapy (CBT).

**Limitations of Prior Work**: Existing ESC methods overlook cognitive distortions implicitly expressed by help-seekers, such as catastrophizing or all-or-nothing thinking. Existing datasets (e.g., D4, CPsyCounD) contain counselor responses that largely disregard cognitive distortions, resulting in models trained on such data that can only provide surface-level comfort rather than deeper cognitive-level support.

**Key Challenge**: At the data level, there is a lack of ESC datasets annotated with cognitive distortions. At the algorithmic level, effective CBT requires precise selection of intervention strategies based on distortion type, severity, and risk level — a granularity that existing methods fail to achieve.

**Goal**: To construct a cognitive-distortion-annotated dataset and design an LLM framework capable of diagnosing cognitive distortions and selecting optimal intervention strategies.

**Key Insight**: Modeling psychological counseling as a multi-agent reinforcement learning environment, where a counselor agent learns optimal intervention policies via DQN.

**Core Idea**: Use RL to learn CBT strategy selection, then distill the acquired strategy knowledge into an LLM via dual-stream optimization, ensuring both accurate diagnosis and effective intervention.

## Method

### Overall Architecture

CoPoLLM consists of two core components: (1) the **CPRL Engine**, which learns an optimal mapping from diagnostic states to intervention strategies via DQN in a multi-agent simulation environment; and (2) the **DSCO Algorithm**, which offline-distills the learned strategy knowledge into an LLM, enabling unified cognitive distortion diagnosis and strategy-aligned intervention generation.

### Key Designs

1. **CogBiasESC Dataset Construction**:

    - **Function**: Provides a data foundation for cognitive distortion diagnosis and intervention.
    - **Mechanism**: Based on CBT theory, eight types of cognitive distortions are defined (emotional reasoning, catastrophizing, all-or-nothing thinking, etc.). Dialogues containing cognitive distortions are selected from three public ESC datasets and independently annotated by three experts for distortion type, severity (mild/moderate/severe), and risk level (low/medium/high). The resulting dataset contains 2,499 multi-turn dialogues, 82,293 utterances, and 15,092 distortion labels, with an average of 3.2 labels per dialogue. Fleiss' Kappa ranges from 0.73 to 0.85.
    - **Design Motivation**: To fill the gap in the ESC field caused by the absence of cognitive-distortion-annotated resources, and to provide a standardized benchmark for training and evaluating cognitive intervention models.

2. **Cognitive Policy Reinforcement Learning Engine (CPRL)**:

    - **Function**: Learns to map cognitive diagnostic states to optimal CBT intervention strategies.
    - **Mechanism**: A three-agent simulation environment is constructed — a counselor agent $\mathcal{A}_{coun}$ (strategy selection), a help-seeker agent $\mathcal{A}_{seek}$ (generating distorted expressions), and an evaluator agent $\mathcal{A}_{eval}$ (computing rewards). States are encoded as continuous vectors of utterances and distortion labels; the action space comprises $K$ CBT strategies. A DQN is used to approximate the value function. The hybrid reward is defined as $R_t = \omega_1 R_{imp} + \omega_2 R_{match} + \omega_3 R_{safe}$, where $R_{safe}$ and $R_{match}$ are rule-based rewards enforcing safety constraints and CBT compliance, and $R_{imp}$ is an LLM-evaluated symptom improvement reward.
    - **Design Motivation**: Value-based RL is more suitable than PPO/DPO for explicit safety constraints, as it can directly penalize unsafe strategies in high-risk states.

3. **Dual-Stream Conditional Optimization (DSCO)**:

    - **Function**: Injects CPRL-learned strategy knowledge into the LLM, enabling joint optimization of diagnosis and intervention.
    - **Mechanism**: The trained policy $\pi_{\theta^*}$ is first used to infer the optimal intervention strategy for each dialogue. GPT-4o then generates augmented responses under strategy guidance, which are manually reviewed to construct CogBiasESC-PRO. The LLM is subsequently trained via a target-masking mechanism that decouples the diagnosis and intervention training streams: $\mathcal{L}_{total} = \mathcal{L}_\tau(\phi; X, \mathcal{C}_t) + \mathcal{L}_\tau(\phi; X, y^*)$.
    - **Design Motivation**: To prevent generative targets (intervention responses) from overshadowing diagnostic learning (cognitive labels), ensuring the model simultaneously learns accurate diagnosis and strategy-aligned intervention.

### Loss & Training

CPRL minimizes the TD error $\mathcal{L}_{DQN}(\theta) = \mathbb{E}[(y_t - Q(s_t, a_t; \theta))^2]$, employing Double DQN to decouple action selection from evaluation. DSCO applies conditional masked cross-entropy loss, computed separately for the diagnosis stream and the intervention stream.

## Key Experimental Results

### Main Results

CoPoLLM vs. 15 SOTA baselines (including SoulChat, ChatCounselor, PsycoLLM, etc.):

| Metric | CoPoLLM | Best Baseline | Gain |
|--------|---------|---------------|------|
| Cognitive Distortion Diagnosis F1 | Best | — | Significant |
| High-Risk Miss Detection Rate (HRMDR) ↓ | Lowest | — | Substantial safety improvement |
| Intervention Strategy Effectiveness | Best | — | Consistent across GPT and human evaluation |
| Clinical Compliance | Best | — | Confirmed by professional counselors |

### Ablation Study

| Configuration | Key Findings |
|---------------|-------------|
| w/o CPRL | Strategy selection degrades to random/imitation; intervention effectiveness drops significantly |
| w/o DSCO | LLM fails to effectively leverage strategy knowledge |
| w/o Safety Reward $R_{safe}$ | High-risk miss detection rate increases significantly |
| w/o Diagnosis Stream | Intervention responses lack specificity |

### Key Findings

- Conventional ESC methods perform poorly on cognitive distortion diagnosis, validating fundamental deficiencies in existing data and models.
- The hard-penalty design of $R_{safe}$ is critical for reducing high-risk misses, ensuring the model activates safety mechanisms immediately upon detecting self-harm or suicidal ideation.
- Emotional reasoning dominates CogBiasESC (36.9%), exhibiting a severe long-tail distribution that poses challenges for model training.
- Dual-stream decoupled training outperforms joint training, as diagnosis and intervention occupy different optimization landscapes.

## Highlights & Insights

- Modeling psychological counseling as an RL decision problem is particularly well-motivated: CBT is inherently a sequential decision-making process — selecting strategies based on current symptoms, observing responses, and adjusting accordingly — which maps naturally onto the RL framework.
- The three-agent simulation environment (counselor–help-seeker–evaluator) constitutes a self-consistent training loop, enabling exploration of the strategy space without requiring large volumes of real counseling data.
- The safety mechanism design is noteworthy: rule-based hard penalties (rather than soft regularization) ensure safety in high-risk scenarios, a design principle applicable to other safety-critical applications.

## Limitations & Future Work

- CogBiasESC is primarily derived from Chinese psychological counseling datasets; cross-lingual and cross-cultural generalizability remains to be validated.
- Although the eight cognitive distortion types cover core CBT constructs, real-world counseling involves more numerous and ambiguous distortion categories.
- The fidelity of the multi-agent simulation environment depends on the role-playing capability of LLMs, which may introduce systematic biases.
- The discrete action space of DQN limits strategic flexibility; a continuous strategy space may be more appropriate for complex scenarios.

## Related Work & Insights

- **vs. SoulChat/ChatCounselor**: These methods focus on empathy and fluency but lack cognitive intervention capabilities; CoPoLLM outperforms them comprehensively on both diagnosis and intervention dimensions.
- **vs. PsycoLLM**: Introduces ethical checking mechanisms but employs coarse strategy selection; CoPoLLM learns more fine-grained strategy mappings through RL.
- **vs. CSO (Zhao et al., 2025)**: Uses MCTS for strategy search but lacks a cognitive framework; CoPoLLM deeply integrates CBT theory into the RL design.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First ESC dataset annotated with cognitive distortions + RL-based strategy learning framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparison with 15 baselines + multi-dimensional evaluation + human evaluation
- Writing Quality: ⭐⭐⭐⭐ Clear framework design with well-motivated CBT rationale
- Value: ⭐⭐⭐⭐⭐ Advances ESC from surface-level comfort toward professional cognitive intervention

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] A Principle-Driven Adaptive Policy for Group Cognitive Stimulation Dialogue for Elderly with Cognitive Impairment](../../AAAI2026/medical_imaging/a_principle-driven_adaptive_policy_for_group_cognitive_stimu.md)
- [\[ACL 2026\] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection](multi-view_attention_multiple-instance_learning_enhanced_by_llm_reasoning_for_co.md)
- [\[CVPR 2026\] MedKCO: Medical Vision-Language Pretraining via Knowledge-Driven Cognitive Orchestration](../../CVPR2026/medical_imaging/medkco_medical_vision-language_pretraining_via_knowledge-driven_cognitive_orches.md)
- [\[ACL 2026\] Measuring What Matters!! Assessing Therapeutic Principles in Mental-Health Conversation](measuring_what_matters_assessing_therapeutic_principles_in_mental-health_convers.md)
- [\[ACL 2026\] Stable On-Policy Distillation through Adaptive Target Reformulation](stable_on-policy_distillation_through_adaptive_target_reformulation.md)

</div>

<!-- RELATED:END -->
