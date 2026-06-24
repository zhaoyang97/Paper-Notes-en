---
title: >-
  [Paper Note] Can You Share Your Story? Modeling Clients' Metacognition and Openness for LLM Therapist Evaluation
description: >-
  [ACL 2025 Findings][LLM (Other)][LLM therapist evaluation] This paper proposes the MindVoyager framework, which evaluates the exploratory capabilities of LLM therapists by constructing client simulators with dynamic metacognition and openness, addressing the issue of overly "cooperative" client simulators in existing evaluation methods.
tags:
  - "ACL 2025 Findings"
  - "LLM (Other)"
  - "LLM therapist evaluation"
  - "client simulator"
  - "metacognition"
  - "openness"
  - "Cognitive Behavioral Therapy (CBT)"
date: 2026-05-08
content_hash: 2caf526d2f74884b
---

# Can You Share Your Story? Modeling Clients' Metacognition and Openness for LLM Therapist Evaluation

**Conference**: ACL 2025 Findings  
**arXiv**: [2507.19643](https://arxiv.org/abs/2507.19643)  
**Code**: None  
**Area**: NLP Applications / LLM Evaluation  
**Keywords**: LLM therapist evaluation, client simulator, metacognition, openness, Cognitive Behavioral Therapy (CBT)

## TL;DR

This paper proposes the MindVoyager framework, which evaluates the exploratory capabilities of LLM therapists by constructing client simulators with dynamic metacognition and openness, addressing the issue of overly "cooperative" client simulators in existing evaluation methods.

## Background & Motivation

**Background**: Large Language Models (LLMs) are increasingly being applied in the mental health domain, particularly in playing the role of therapists to converse with clients. Existing studies evaluate the quality of LLM therapists through LLM-to-LLM interactions, using an LLM to simulate a client interacting with the LLM therapist under evaluation.

**Limitations of Prior Work**: Current client simulators possess two key limitations: (1) Lack of metacognitive abilities—real clients often lack a clear understanding of their own cognitive issues, whereas LLM-simulated clients can explicitly articulate their internal states; (2) Lack of dynamic openness—real clients do not instantly open up to a therapist, as trust-building takes time, but LLM simulators often reveal everything right from the start.

**Key Challenge**: If a client simulator directly discloses all internal states in the conversation, it becomes impossible to distinguish between an "excellent therapist good at proactive exploration and trust-building" and a "mediocre therapist who only passively receives information." This leads existing evaluation methods to overestimate the actual capabilities of LLM therapists.

**Goal**: To design a more realistic and challenging evaluation framework for LLM therapists, enabling the client simulator to exhibit dynamically changing openness and limited metacognitive abilities like real clients.

**Key Insight**: Starting from the Cognitive Conceptualization Diagram (CCD) in Cognitive Behavioral Therapy (CBT), which describes the relationships among a client's core beliefs, intermediate beliefs, and automatic thoughts. By "masking" different parts of the CCD, the client's limited metacognition can be simulated, and the level of masking can be dynamically adjusted based on interaction quality to simulate changes in openness.

**Core Idea**: To simulate the client's metacognition and openness using a controllable cognitive diagram masking mechanism, bringing the client simulator closer to real-world scenarios and thereby more accurately evaluating the exploratory capabilities of LLM therapists.

## Method

### Overall Architecture

The MindVoyager framework consists of three phases: (1) Cognitive diagram initialization—constructing the CCD based on the client case and simulating limited metacognition by masking certain nodes; (2) Therapeutic dialogue—the client simulator dialogues with the LLM therapist over multiple turns, where the accessible parts of the CCD are dynamically adjusted based on conversation quality (rapport building quality and therapist utterance quality); (3) Evaluation—scoring based on the number of cognitive diagram nodes successfully uncovered by the therapist and whether the client was guided to proactively express these nodes.

### Key Designs

1. **Cognitive Diagram Masking Mechanism**:

    - **Function**: Simulating the client's limited metacognition.
    - **Mechanism**: Constructing the client's complete cognitive diagram (containing nodes like core beliefs, intermediate beliefs, automatic thoughts, etc.) based on CBT's CCD, and then randomly masking a portion of the nodes. The masked nodes represent cognitive aspects that the client is not yet aware of, and the client simulator can only access the unmasked parts when responding. This prevents the client from describing their psychological issues with "omniscient" knowledge.
    - **Design Motivation**: To address the unrealistic phenomenon of "clients knowing themselves better than the therapist do" in existing simulators, making the client's cognitive abilities closer to real-world scenarios.

2. **Dynamic Openness Adjustment**:

    - **Function**: Dynamically adjusting the client's trust level and level of disclosure based on the dialogue progress.
    - **Mechanism**: During the conversation, the system evaluates the current dialogue quality based on two dimensions: the establishment of therapeutic rapport and the quality of the therapist's current utterance. If the therapist demonstrates empathy and appropriate questioning skills, the client's openness increases, and more CCD nodes are "unlocked"; conversely, if the therapist performs poorly, openness may remain unchanged or decrease.
    - **Design Motivation**: In real counseling, the client's trust is built progressively, and a good therapist can help clients open up faster. This mechanism enables the evaluation to differentiate the strength of the therapist's exploratory capabilities.

3. **Exploratory Capability Evaluation Indicators**:

    - **Function**: Quantitatively evaluating the exploratory capabilities of LLM therapists.
    - **Mechanism**: Evaluation is conducted from two perspectives: (1) Reveal rate—how many originally masked cognitive diagram nodes the therapist successfully got the client to reveal during the conversation; (2) Expressive guidance rate—whether the therapist successfully guided the client to actively articulate these revealed cognitive contents in language (rather than just detecting their existence).
    - **Design Motivation**: To evaluate not only how much information the therapist can "discover" but also their ability to guide clients in self-exploration and expression, which is the core of high-quality therapy.

### Loss & Training

Ours does not involve model training; instead, it constructs an evaluation framework. Both the client simulator and the evaluation module are implemented based on Prompt Engineering, leveraging the instruction-following capabilities of pre-trained LLMs.

## Key Experimental Results

### Main Results

| LLM Therapist | Reveal Rate (%) | Expressive Guidance Rate (%) | Overall Score |
|-----------|-----------|--------------|---------|
| GPT-4 | ~45% | ~38% | ~42% |
| GPT-3.5-turbo | ~35% | ~28% | ~32% |
| Llama-3-70B | ~38% | ~30% | ~34% |
| Llama-3-8B | ~25% | ~18% | ~22% |
| Human Therapist Reference | ~65% | ~58% | ~62% |

### Ablation Study

| Configuration | Reveal Rate | Expressive Guidance Rate | Description |
|------|-------|-----------|------|
| Full (Metacognition + Openness) | ~45% | ~38% | Full MindVoyager framework |
| w/o Metacognitive Masking | ~62% | ~55% | Without masking, the client is overly cooperative, leading to inflated scores |
| w/o Dynamic Openness | ~50% | ~42% | Openness does not change with dialogue, failing to differentiate therapist quality |
| Static Client Simulator | ~70% | ~65% | Traditional method, severely overestimating therapist capabilities |

### Key Findings

- Traditional static client simulators severely overestimate the capabilities of LLM therapists, since clients proactively disclose all information.
- After introducing metacognitive masking and dynamic openness, the performance gaps between different LLM therapists become more pronounced, showing that the framework provides better discriminative power.
- GPT-4 significantly outperforms other models in exploratory capabilities, but remains well below the performance level of human therapists.
- Smaller models (such as 8B parameters) perform particularly poorly in scenarios that require trust-building and proactive exploration.

## Highlights & Insights

- **Formalizing key concepts in psychotherapy**: Translating psychological concepts like "metacognition" and "openness" into computable mechanisms (masking and dynamic adjustment). This interdisciplinary methodology is highly referable, and similar approaches can be applied to other evaluation scenarios requiring the simulation of human cognitive traits.
- **Shift in evaluation focus**: Shifting from "what the therapist said" to "what the therapist can discover," focusing on exploratory capability rather than mere response quality. This evaluation philosophy can be transferred to other dialogue AI evaluation tasks.
- **Controllable simulator design**: Achieving precise control over the client's cognitive state via CCD masking, making experiments reproducible and comparable.

## Limitations & Future Work

- The framework is based on the cognitive conceptualization diagram of CBT, which may not apply to other psychotherapy schools (e.g., psychodynamic, humanistic, etc.).
- The adjustment of dynamic openness relies on the LLM's judgment of therapeutic rapport quality, which itself may not be fully accurate.
- The entire framework is highly dependent on the LLM's role-playing capacity, and the authenticity of client simulation cannot be fully guaranteed.
- Future work can extend to multi-session treatment evaluation and validation combined with real clinical data.

## Related Work & Insights

- **vs Chiu et al. (2024)**: This work proposes a behavioral evaluation framework for LLM therapists but uses a static client simulator, failing to evaluate the therapist's proactive exploration capabilities. The current work addresses this issue through dynamic openness.
- **vs Wang et al. (2024)**: Patient-$\psi$ uses LLMs to simulate patients for training mental health professionals but similarly lacks modeling of the client's limited metacognition. The masking mechanism in this paper is closer to real-world scenarios.
- **vs CACTUS (Lee et al., 2024)**: CACTUS constructs counseling dialogues based on CBT theory but focuses on dialogue generation rather than therapist evaluation. The two are complementary in their application of the CBT theoretical framework.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to introduce metacognition and dynamic openness into LLM therapist evaluation, providing a novel interdisciplinary perspective.
- **Experimental Thoroughness**: ⭐⭐⭐ Evaluated multiple models and conducted ablation studies, but lacks comparison and validation against real clinical scenarios.
- **Writing Quality**: ⭐⭐⭐⭐ The problem motivation is clear, the framework description is detailed, and the introduction of psychological concepts is natural and logical.
- **Value**: ⭐⭐⭐⭐ Holds significant reference value for the safety evaluation of LLM mental health applications, though the application scenario remains relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] ConsistencyChecker: Tree-based Evaluation of LLM Generalization Capabilities](consistencychecker_tree_evaluation.md)
- [\[ACL 2025\] Mind Your Tone: Investigating How Prompt Politeness Affects LLM Accuracy](mind_your_tone_investigating_how_prompt_politeness_affects_llm_accuracy_short_pa.md)
- [\[ICML 2026\] Express Your Doubts: Probabilistic World Modeling Should Not Be Based on Token logprobs](../../ICML2026/llm_nlp/express_your_doubts_--_probabilistic_world_modeling_should_not_be_based_on_token.md)
- [\[ACL 2025\] MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](membench_towards_more_comprehensive_evaluation_on_the_memory_of_llm-based_agents.md)
- [\[ACL 2025\] Can LLMs Reason About Program Semantics? A Comprehensive Evaluation of LLMs on Formal Specification Inference](can_llms_reason_about_program_semantics_a_comprehensive_evaluation_of_llms_on_fo.md)

</div>

<!-- RELATED:END -->
