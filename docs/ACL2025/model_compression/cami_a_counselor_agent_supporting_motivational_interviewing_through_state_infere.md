---
title: >-
  [Paper Note] CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration
description: >-
  [ACL 2025][Model Compression][Motivational Interviewing] This paper proposes CAMI (Counselor Agent for Motivational Interviewing), a counseling agent based on the principles of Motivational Interviewing (MI). It utilizes the STAR framework (State inference, Topic exploration, Action & Response generation) to guide clients to generate change talk, outperforming existing methods in both automated and human evaluations.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Motivational Interviewing"
  - "Counseling Agent"
  - "State Inference"
  - "Topic Exploration"
  - "Behavior Change"
date: 2026-05-08
content_hash: b29cb0cd12ad18e5
---

# CAMI: A Counselor Agent Supporting Motivational Interviewing through State Inference and Topic Exploration

**Conference**: ACL 2025  
**arXiv**: [2502.02807](https://arxiv.org/abs/2502.02807)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: Motivational Interviewing, Counseling Agent, State Inference, Topic Exploration, Behavior Change

## TL;DR

This paper proposes CAMI (Counselor Agent for Motivational Interviewing), a counseling agent based on the principles of Motivational Interviewing (MI). It utilizes the STAR framework (State inference, Topic exploration, Action & Response generation) to guide clients to generate change talk, outperforming existing methods in both automated and human evaluations.

## Background & Motivation

**Background**: With the rapid growth in demand for mental health services, automated dialogue counseling agents have become an important means to extend the accessibility of mental health support. Existing mental health counseling dialogue systems are mostly based on simple dialogue strategies and lack support from systematic counseling theories.

**Limitations of Prior Work**: (1) Existing Law-driven (LLM-driven) counseling agents lack the ability to dynamically track the client's psychological state, often responding only to the current utterance rather than planning based on the overall state; (2) Most methods do not follow mature counseling theoretical frameworks, leading to dialogues that lack direction and therapeutic efficacy; (3) In motivational interviewing, guiding a client from "ambivalence" to "change talk" requires delicate strategies, which simple LLM generation fails to handle adequately.

**Key Challenge**: Motivational interviewing requires the counselor to simultaneously perform three cognitive tasks—understanding the client's current stage of readiness for change, selecting appropriate topics for exploration, and generating responses that align with MI principles. Conflating these three tasks into a single end-to-end generation problem leads to poor performance on each subtask.

**Goal**: Design a structured counseling agent that follows MI principles and decouples the counseling process into three explicit modules: state inference, topic exploration, and response generation.

**Key Insight**: The "stages of change" model and "evoking change talk" in MI theory provide explicit theoretical guidance for the agent's decision-making.

**Core Idea**: Tracking the client's readiness for change through an explicit state inference module and utilizing a topic exploration module to search for topics capable of evoking change talk, thereby enabling theoretically-guided, purposeful counseling behavior in the agent.

## Method

### Overall Architecture

The STAR framework of CAMI consists of three modules: **S**tate inference $\rightarrow$ **T**opic exploration $\rightarrow$ **A**ction & **R**esponse generation. The input is the multi-turn dialogue history, and the output is a counselor response aligned with MI principles.

### Key Designs

1. **Client State Inference Module**:

    - **Function**: Infer the client's current stage of change readiness based on the dialogue history.
    - **Mechanism**: According to the Transtheoretical Model (Stages of Change) in MI theory, a client's process of change is divided into precontemplation, contemplation, preparation, action, and maintenance. This module uses an LLM to analyze linguistic cues in the dialogue (e.g., sustain talk vs. change talk) to infer which stage the client is currently in. The result of the state inference directly influences the subsequent topic selection strategy—for clients in the precontemplation stage, open questions should be used more, while for those in the preparation stage, action plans can be discussed more directly.
    - **Design Motivation**: Counseling agents without state inference cannot adjust their strategies based on the client's readiness, making them prone to "premature intervention" or "overly conservative" issues.

2. **Topic Exploration Module**:

    - **Function**: Select the topic most likely to evoke change talk based on the client's current state.
    - **Mechanism**: Maintain a candidate topic set, including various themes mentioned by the client (e.g., health, family, work, etc.). Based on the currently inferred state and dialogue history, an LLM is used to evaluate the "change potential" of each candidate topic—i.e., whether continuing to explore this topic helps to evoke change talk from the client. The topic with the highest score is selected for in-depth exploration. This implements the "selective reinforcing" strategy in MI.
    - **Design Motivation**: The core of MI is "evoking" rather than "directing"—the counselor helps clients realize the need for change themselves by selecting the right topics, rather than directly telling them what to do.

3. **Response Generation Module**:

    - **Function**: Generate MI-compliant counseling responses based on the inferred state and selected topic.
    - **Mechanism**: Treating the state inference result and topic selection as conditional information, combined with a library of MI response strategies (e.g., open questions, reflective listening, affirmation, summarizing), an LLM is used to generate the final counselor response. The type of MI strategy in the response is also explicitly planned—reflective listening is used more for resistant clients, while affirmations and guiding are used more for change-oriented clients.
    - **Design Motivation**: Providing structured conditional information (state + topic + strategy type) to the LLM during response generation yields responses that conform much better to MI principles than direct generation.

### Loss & Training

Using simulated clients to conduct large-scale dialogue generation and evaluation. Simulated clients are played by an LLM, given specific problem scenarios and resistance levels, and engage in multi-turn dialogues with CAMI.

## Key Experimental Results

### Main Results

| Evaluation Metric | CAMI | ChatGPT | MI-specific baseline | Human Counselor |
|----------|------|---------|---------------------|-----------|
| MI Compatibility Score | Best | Low | Moderate | Reference Standard |
| Change Talk Evocation Rate | Significantly outperforms baselines | Low | Moderate | - |
| Response Diversity | High | High | Low | High |
| Avg. Dialogue Turns to Change Talk | Fewer | More | Moderate | - |
| State Inference Accuracy | ~75% | - | - | - |

### Ablation Study

| Configuration | MI Compatibility | Change Talk Evocation Rate | Description |
|------|---------|-------------|------|
| Full CAMI (STAR) | Best | Best | Full model |
| w/o State Inference | Significant decline | Substantial decline | Cannot adjust strategy based on client state |
| w/o Topic Exploration | Moderate decline | Significant decline | Topic selection becomes random |
| w/o Strategy Planning | Slight decline | Slight decline | Response style is affected |
| Direct End-to-End Generation | Worst | Worst | No structured guidance |

### Key Findings
- The state inference module is the most critical component of CAMI—MI compatibility and the change talk evocation rate both decline substantially without it.
- The contribution of the topic exploration module is more pronounced in long dialogues—it has little impact on short dialogues (<5 turns), but selecting the correct topic is crucial in dialogues of 10+ turns.
- Responses generated by CAMI received high scores in expert evaluations of MI skills, particularly in reflective listening and open questions.
- Evaluation with simulated clients aligns well with a small number of human evaluations, validating the reliability of simulated evaluations.

## Highlights & Insights
- **Explicitly incorporating psychotherapy theory (the Transtheoretical Model of MI) into the Agent architecture** is the biggest highlight of this paper—instead of treating the LLM as a black box, psychological theory guides the module design. This "theory-driven Agent design" mindset can be transferred to other dialogue scenarios requiring specialized expertise.
- **The cascaded decision-making of State Inference $\rightarrow$ Topic Selection $\rightarrow$ Response Generation** is more interpretable than end-to-end generation, facilitating review and intervention by human counselors.
- Using simulated clients for large-scale evaluation is a pragmatic solution—human counseling evaluation is extremely costly and involves ethical concerns.

## Limitations & Future Work
- Although simulated clients align with a small number of human evaluations, they cannot fully replace real clinical scenarios.
- The accuracy of state inference is about 75%, and misjudgments can lead to inappropriate counseling strategies.
- It only focuses on one counseling school (MI) and does not explore other mainstream methods such as Cognitive Behavioral Therapy (CBT).
- Safety considerations are insufficient—how the Agent should safely refer the client to a human counselor when facing clients with self-harm risk.

## Related Work & Insights
- **vs SMILE**: SMILE is a mental health counseling dialogue system based on CBT that uses predefined dialogue strategies; CAMI's state inference mechanism is more flexible.
- **vs AugESC**: AugESC trains emotional support dialogue models through data augmentation but lacks explicit counseling strategy planning.
- **vs ChatCounselor**: ChatCounselor primarily focuses on emotion recognition and empathetic response, without involving behavior change guidance.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The STAR framework elegantly combines MI theory with LLM Agents, and the state inference module is cleverly designed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dimensional evaluation (MI compatibility, change talk rate, ablation), but lacks large-scale human evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ Clearly explains psychological counseling theory while maintaining technical depth.
- **Value**: ⭐⭐⭐⭐ Provides a theory-driven methodology for the field of mental health AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] IAM: Efficient Inference through Attention Mapping between Different-scale LLMs](iam_efficient_inference_through_attention_mapping_between_different-scale_llms.md)
- [\[ACL 2025\] State-offset Tuning: State-based Parameter-Efficient Fine-Tuning for State Space Models](state_offset_tuning_ssm_peft.md)
- [\[ECCV 2024\] ELSE: Efficient Deep Neural Network Inference through Line-based Sparsity Exploration](../../ECCV2024/model_compression/else_efficient_deep_neural_network_inference_through_line-based_sparsity_explora.md)
- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking](deepsolution_boosting_complex_engineering_solution_design_via_tree-based_explora.md)

</div>

<!-- RELATED:END -->
