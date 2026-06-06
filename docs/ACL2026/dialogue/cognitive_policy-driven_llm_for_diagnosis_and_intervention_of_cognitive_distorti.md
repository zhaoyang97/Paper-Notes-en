---
title: >-
  [Paper Note] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation
description: >-
  [ACL 2026][Dialogue Systems][Emotional Support Conversation] Ours proposes the CoPoLLM framework, which constructs the first Emotional Support Conversation (ESC) dataset with cognitive distortion annotations…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Emotional Support Conversation"
  - "Cognitive Distortion"
  - "Cognitive Behavioral Therapy"
  - "Reinforcement Learning Policy"
  - "Safety Intervention"
date: 2026-05-08
content_hash: e796860070a2be35
---

# Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation

**Conference**: ACL 2026  
**arXiv**: [2604.17178](https://arxiv.org/abs/2604.17178)  
**Code**: [https://github.com/Chips98/CoPoLLM-for-ACL-2026](https://github.com/Chips98/CoPoLLM-for-ACL-2026)  
**Area**: Medical Image  
**Keywords**: Emotional Support Conversation, Cognitive Distortion, Cognitive Behavioral Therapy, Reinforcement Learning Policy, Safety Intervention

## TL;DR

Ours proposes the CoPoLLM framework, which constructs the first Emotional Support Conversation (ESC) dataset with cognitive distortion annotations, CogBiasESC. By combining a Cognitive Policy Reinforcement Learning (CPRL) engine with Dual-Stream Conditional Optimization (DSCO), the LLM can diagnose 8 types of cognitive distortions and generate policy-aware intervention responses, outperforming 15 SOTA baselines comprehensively.

## Background & Motivation

**Background**: LLMs have demonstrated strong empathy in ESC tasks, with methods like SoulChat and ChatCounselor advancing fluency and empathy through SFT or DPO. However, professional psychological counseling requires not just emotional comfort but also cognitive intervention based on Cognitive Behavioral Therapy (CBT).

**Limitations of Prior Work**: Existing ESC methods ignore the implicit cognitive distortions (e.g., catastrophizing, all-or-nothing thinking) in seeking help. In existing datasets (D4, CPsyCounD, etc.), the original responses from counselors often fail to fully consider these distortions, causing models trained on this data to provide superficial comfort rather than deep cognitive assistance.

**Key Challenge**: There is a lack of ESC datasets with cognitive distortion annotations at the data level. At the algorithmic level, effective CBT requires precise strategy selection based on distortion type, intensity, and risk level, whereas current strategy selection mechanisms are too coarse.

**Goal**: Construct a dataset with cognitive distortion annotations and design an LLM framework capable of diagnosing cognitive distortions and selecting optimal intervention strategies.

**Key Insight**: Model psychological counseling as a multi-agent interaction environment for reinforcement learning, allowing the counselor agent to learn optimal intervention policies via DQN.

**Core Idea**: Use RL to learn CBT strategy selection policies, and then distill this policy knowledge into the LLM through dual-stream optimization to ensure both accurate diagnosis and effective intervention.

## Method

### Overall Architecture

CoPoLLM consists of two core components: (1) The CPRL engine, which learns the optimal mapping from diagnostic states to intervention strategies via DQN in a multi-agent simulation environment; (2) The DSCO algorithm, which distills the learned policy knowledge offline into the LLM to achieve unified distortion diagnosis and policy-aligned intervention generation.

### Key Designs

1.  **CogBiasESC Dataset Construction**:
    - **Function**: Provide a data foundation for cognitive distortion diagnosis and intervention.
    - **Mechanism**: Define 8 categories of cognitive distortions based on CBT theory (Emotional Reasoning, Catastrophizing, All-or-Nothing, etc.). Conversations containing cognitive distortions were filtered from three public ESC datasets. Three experts independently annotated the distortion types, intensity (Low/Medium/High), and risk level (Low/Medium/High). The final dataset contains 2,499 multi-turn conversations, 82,293 utterances, and 15,092 distortion labels, averaging 3.2 labels per conversation. Fleiss' Kappa reached 0.73-0.85.
    - **Design Motivation**: Fill the gap in the ESC field regarding the lack of cognitive distortion annotations and provide standardized resources for training and evaluating cognitive intervention models.

2.  **Cognitive Policy Reinforcement Learning Engine (CPRL)**:
    - **Function**: Learn to map cognitive diagnostic states to optimal CBT intervention strategies.
    - **Mechanism**: Construct a three-agent simulation environment: a counselor agent $\mathcal{A}_{coun}$ (selects strategies), a seeker agent $\mathcal{A}_{seek}$ (generates distorted expressions), and an evaluation agent $\mathcal{A}_{eval}$ (calculates rewards). States are encoded as continuous vectors of utterances + distortion labels, and the action space consists of K types of CBT strategies. DQN is used to approximate the value function. A mixed reward is used: $R_t = \omega_1 R_{imp} + \omega_2 R_{match} + \omega_3 R_{safe}$, where $R_{safe}$ and $R_{match}$ are rule-based rewards (enforcing safety constraints and CBT standards), and $R_{imp}$ is the symptom improvement reward evaluated by an LLM.
    - **Design Motivation**: Value-based methods are better suited for explicit safety constraints than PPO/DPO, as they can directly penalize unsafe policies in high-risk states.

3.  **Dual-Stream Conditional Optimization (DSCO)**:
    - **Function**: Inject policy knowledge learned by CPRL into the LLM to achieve joint optimization of diagnosis and intervention.
    - **Mechanism**: First, the trained policy $\pi_{\theta^*}$ is used to infer the optimal intervention strategy for each conversation. GPT-4o then generates enhanced responses under policy guidance, which are manually reviewed to construct CogBiasESC-PRO. Finally, a target masking mechanism decouples the training streams for diagnosis and intervention: $\mathcal{L}_{total} = \mathcal{L}_\tau(\phi; X, \mathcal{C}_t) + \mathcal{L}_\tau(\phi; X, y^*)$.
    - **Design Motivation**: Prevent the generation target (intervention response) from overshadowing the diagnostic learning (cognitive labels), ensuring the model simultaneously learns accurate diagnosis and policy-aligned intervention.

### Loss & Training

CPRL uses the TD error $\mathcal{L}_{DQN}(\theta) = \mathbb{E}[(y_t - Q(s_t, a_t; \theta))^2]$, employing Double DQN to decouple action selection and evaluation. DSCO uses a conditional masked cross-entropy loss, calculated separately for the diagnostic stream and the intervention stream.

## Key Experimental Results

### Main Results

CoPoLLM vs. 15 SOTA baselines (including SoulChat, ChatCounselor, PsycoLLM, etc.):

| Metric | CoPoLLM | Best Baseline | Gain |
| :--- | :--- | :--- | :--- |
| Cognitive Distortion Diagnosis F1 | Optimal | - | Significant |
| High-Risk Missed Detection Rate (HRMDR) ↓ | Lowest | - | Substantial safety improvement |
| Intervention Strategy Effectiveness | Optimal | - | Consistent with GPT and human evaluation |
| Clinical Standard Compliance | Optimal | - | Confirmed by professional counselors |

### Ablation Study

| Configuration | Key Findings |
| :--- | :--- |
| w/o CPRL | Strategy selection degrades to random/imitation; intervention effectiveness drops significantly. |
| w/o DSCO | The LLM cannot effectively utilize policy knowledge. |
| w/o Safety Reward $R_{safe}$ | High-risk missed detection rate increases significantly. |
| w/o Diagnostic Stream | Intervention responses lack specificity. |

### Key Findings

-   Traditional ESC methods perform extremely poorly in cognitive distortion diagnosis, verifying the fundamental flaws in existing data and models.
-   The hard penalty design of the safety reward $R_{safe}$ is crucial for reducing high-risk missed detections—ensuring the model immediately activates safety mechanisms when self-harm/suicidal tendencies are detected.
-   Emotional Reasoning (36.9%) dominates in CogBiasESC, presenting a severe long-tail distribution that challenges model training.
-   Dual-stream decoupled training is more effective than joint training—diagnosis and intervention have different optimization landscapes.

## Highlights & Insights

-   Modeling psychological counseling as an RL decision problem is ingenious: CBT itself is a sequential decision process—selecting strategies based on current symptoms, observing reactions, and adjusting strategies—which naturally fits the RL framework.
-   The three-agent simulation environment (counselor-seeker-evaluator) forms a self-consistent training loop that allows exploration of the strategy space without requiring massive amounts of real counseling data.
-   The design of the safety mechanism is noteworthy: ensuring safety in high-risk scenarios through rule-based hard penalties (rather than soft regularization) is applicable to other safety-critical applications.

## Limitations & Future Work

-   CogBiasESC is primarily based on Chinese psychological counseling datasets; cross-lingual and cross-cultural generalizability remains to be verified.
-   While the 8 types of cognitive distortions cover the core of CBT, more types and ambiguous distortions exist in real counseling.
-   The fidelity of the multi-agent simulation environment depends on the role-playing capabilities of the LLM, which may introduce systematic bias.
-   The discrete action space of DQN limits policy flexibility; continuous policy spaces might be better suited for complex scenarios.

## Related Work & Insights

-   **vs SoulChat/ChatCounselor**: Focuses on empathy and fluency but lacks cognitive intervention capabilities; CoPoLLM surpasses them in both diagnosis and intervention dimensions.
-   **vs PsycoLLM**: Introduces ethical check mechanisms but has coarse strategy selection; CoPoLLM learns finer strategy mapping through RL.
-   **vs CSO (Zhao et al., 2025)**: Uses MCTS for policy search but lacks a cognitive framework; CoPoLLM deeply integrates CBT theory into the RL design.

## Rating
-   Novelty: ⭐⭐⭐⭐⭐ First ESC dataset with cognitive distortion labels + RL policy learning framework.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comparison with 15 baselines + multi-dimensional evaluation + human evaluation.
-   Writing Quality: ⭐⭐⭐⭐ Clear framework design with strong CBT motivation.
-   Value: ⭐⭐⭐⭐⭐ Drives ESC from superficial comfort toward professional cognitive intervention.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers](stress-testing_emotional_support_models_moving_from_homogeneous_to_diverse_help_.md)
- [\[ACL 2026\] MA$^2$P: A Meta-Cognitive Autonomous Intelligent Agents Framework for Complex Persuasion](ma2p_a_meta-cognitive_autonomous_intelligent_agents_framework_for_complex_persua.md)
- [\[AAAI 2026\] Chatsparent: An Interactive System for Detecting and Mitigating Cognitive Fatigue in LLMs](../../AAAI2026/dialogue/chatsparent_an_interactive_system_for_detecting_and_mitigating_cognitive_fatigue.md)
- [\[ACL 2026\] Dual Hierarchical Dialogue Policy Learning for Legal Inquisitive Conversational Agents](dual_hierarchical_dialogue_policy_learning_for_legal_inquisitive_conversational_.md)
- [\[ACL 2026\] Towards Proactive Information Probing: Customer Service Chatbots Harvesting Value from Conversation](towards_proactive_information_probing_customer_service_chatbots_harvesting_value.md)

</div>

<!-- RELATED:END -->
