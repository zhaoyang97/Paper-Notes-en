---
title: >-
  [Paper Note] Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning
description: >-
  [ACL 2025][Hallucination Detection][Knowledge Misalignment] To address the hallucination issue in LLMs caused by knowledge misalignment (inconsistency between model parametric knowledge and reality), this paper proposes a Selective Abstention Learning method. This approach enables the model to actively refuse to answer when encountering questions outside its knowledge boundary instead of fabricating content, thereby reducing hallucinations.
tags:
  - "ACL 2025"
  - "Hallucination Detection"
  - "Knowledge Misalignment"
  - "Selective Abstention"
  - "Hallucination Mitigation"
  - "Knowledge Boundary Awareness"
  - "Rejection Mechanism"
date: 2026-05-08
content_hash: feff1789cbc93198
---

# Alleviating Hallucinations from Knowledge Misalignment in Large Language Models via Selective Abstention Learning

**Conference**: ACL 2025  
**Code**: None  
**Area**: Hallucination Detection  
**Keywords**: Knowledge Misalignment, Selective Abstention, Hallucination Mitigation, Knowledge Boundary Awareness, Rejection Mechanism

## TL;DR
To address the hallucination issue in LLMs caused by knowledge misalignment (inconsistency between model parametric knowledge and reality), this paper proposes a Selective Abstention Learning method. This approach enables the model to actively refuse to answer when encountering questions outside its knowledge boundary instead of fabricating content, thereby reducing hallucinations.

## Background & Motivation

**Background**: Large Language Models suffer from serious "hallucination" issues—generating seemingly fluent but factually incorrect content. The sources of hallucinations are diverse, including noise in the training data, incomplete and outdated model knowledge, and randomness during the generation process.

**Limitations of Prior Work**: Existing hallucination mitigation methods mainly include: (1) Retrieval-Augmented Generation (RAG)—supplementing the model's knowledge by retrieving external information, which increases system complexity and latency; (2) Post-processing verification—checking factuality after generation, which cannot prevent the initial occurrence of hallucinations; (3) Alignment training—reducing hallucinations through methods like RLHF, which has limited efficacy and may affect fluency. None of these methods fundamentally solve the problem of "the model not knowing what it does not know."

**Key Challenge**: A "knowledge misalignment" exists between the LLM's parametric knowledge and real-world knowledge—the model memorizes certain knowledge inaccurately or has not learned it at all, yet the autoregressive generation mechanism forces the model to produce answers for any input, leading to hallucinations in unknown knowledge domains.

**Goal**: To equip LLMs with the ability to perceive their own knowledge boundaries, allowing them to answer normally when knowledge is sufficient and choose to abstain when seeking to answer questions where knowledge is lacking, rather than forcing fabrication.

**Key Insight**: Reformulate the problem as "LLM knowledge boundary detection"—determining whether a question falls within the model's knowledge coverage. By comparing the model's actual knowledge coverage with the requirements of the question, the model is trained to learn when to say "I don't know."

**Core Idea**: Utilizing a selective abstention learning framework, the model first probes its knowledge boundaries (identifying which questions it can answer correctly and which it answers incorrectly), and then learns to generate abstention signals for incorrect questions while maintaining normal answering capabilities for correct ones.

## Method

### Overall Architecture
The method consists of three phases: (1) Knowledge boundary probing—evaluating the model's accuracy on various questions through multiple samplings to distinguish "known" and "unknown" questions; (2) Abstention data construction—constructing training samples containing abstention tokens for "unknown" questions; (3) Selective abstention training—fine-tuning the model on mixed data to learn to abstain at appropriate times.

### Key Designs

1. **Knowledge Boundary Probing**:

    - **Function**: Quantitatively evaluate the model's level of knowledge mastery over each knowledge point.
    - **Mechanism**: For each question in the training set, multiple random samplings (e.g., 10 times) are conducted using the model, and the proportion of correct answers is calculated as "knowledge confidence". Thresholds are set to classify questions into three categories: high confidence (>0.8, genuinely known by the model), medium confidence (0.3-0.8, partially known), and low confidence (<0.3, unknown). Low confidence questions are high-risk areas for hallucinations.
    - **Design Motivation**: The hidden states of autoregressive LLMs do not contain explicit "uncertainty signals"; thus, multiple samplings can indirectly estimate the model's level of knowledge mastery.

2. **Abstention Data Construction**:

    - **Function**: Provide the model with training signals regarding "when to abstain."
    - **Mechanism**: For low-confidence questions, training samples with abstention responses are constructed. The abstention response is not a simple "I don't know" but a structured template: "According to my knowledge, I cannot determine the exact information for [specific knowledge domain], and recommend consulting [relevant resources]." Meanwhile, correct answers for high-confidence questions are kept as positive samples to ensure the model does not over-abstain.
    - **Design Motivation**: Directly training the model to reject all difficult questions would make it excessively conservative. A careful balance between abstention and answering must be maintained.

3. **Selective Abstention Loss Function**:

    - **Function**: Jointly optimize answer quality and abstention timing.
    - **Mechanism**: Design a dual-objective loss: $\mathcal{L} = \mathcal{L}_{answer} + \beta \cdot \mathcal{L}_{abstain}$, where $\mathcal{L}_{answer}$ is the standard language modeling loss on high-confidence samples, and $\mathcal{L}_{abstain}$ is the prediction loss of the abstention response on low-confidence samples. The hyperparameter $\beta$ controls the strictness of the abstention.
    - **Design Motivation**: A single loss results either in over-abstention (impairing helpfulness) or under-abstention (hallucinations persist); the dual-objective loss offers flexible adjustment.

### Loss & Training
The model is trained on mixed data using the LoRA parameter-efficient fine-tuning strategy. The training data consists of correct answers for high-confidence questions and abstention responses for low-confidence questions, with the proportion adjusted by $\beta$.

## Key Experimental Results

### Main Results

| Method | TruthfulQA Acc↑ | SelfAware Acc↑ | Abstention Rate | Correct Answer Quality |
|------|-----------------|----------------|--------|------------|
| Vanilla LLM | 42.3 | 53.2 | 0% | High |
| + RLHF Alignment | 48.7 | 58.4 | 2% | High |
| + RAG | 52.1 | 61.3 | 0% | Relatively High |
| + Selective Abstention (Ours) | **58.6** | **67.8** | 18% | High |

### Ablation Study

| Configuration | TruthfulQA Acc | Abstention Rate | Description |
|---|---|---|---|
| Full model | 58.6 | 18% | Full model |
| Without knowledge boundary probing | 51.2 | 25% | Imprecise abstention, over-rejection |
| Fixed threshold abstention | 54.3 | 15% | Lack of flexibility |
| $\beta=0$ (No abstention training) | 42.3 | 0% | Degenerates to the vanilla model |
| $\beta=2.0$ (High abstention weight) | 55.1 | 35% | Excessive abstention affecting usability |

### Key Findings
- Selective abstention improves TruthfulQA accuracy from 42.3% to 58.6% (+16.3%), outperforming RAG (+9.8%) and RLHF (+6.4%).
- An abstention rate of approximately 18% is the optimal balance point—lower rates lead to insufficient hallucination mitigation, while higher rates degrade model usability.
- The precision of knowledge boundary probing is critical—inaccurate probing causes the model to incorrectly abstain (from questions it should answer) or incorrectly answer (questions it should abstain from).
- The structured information provided by the model during abstention (e.g., "suggest consulting materials in domain XX") is more popular with users than a simple "I don't know".

## Highlights & Insights
- Moving from "preventing the model from making mistakes" to "letting the model know when to keep silent" is a very insightful paradigm shift. Analogous to humans, admitting ignorance is itself a form of knowledge.
- The knowledge boundary probing concept can be transferred to other scenarios: (1) Dynamic RAG—retrieving only when the model is uncertain; (2) Active Learning—prioritizing learning in the model's knowledge gaps.
- The trade-off between selective abstention and over-rejection is a worthy research problem, affecting the reliability and usability of AI systems.

## Limitations & Future Work
- Knowledge boundary probing requires multiple samplings, incurring high computational costs.
- The model's knowledge boundaries change over time (knowledge obsolescence), requiring periodic re-probing.
- For tasks requiring creative responses (e.g., writing, brainstorming), abstention might not be an appropriate strategy.
- Integrating abstention signals with RAG has not been explored—such as automatically triggering retrieval upon abstention.

## Related Work & Insights
- **vs R-Tuning**: R-Tuning also investigates teaching LLMs to say "I don't know" but uses a simple second-class classification. In contrast, the multi-sampling knowledge probing proposed in this paper is more fine-grained.
- **vs Know-No**: Know-No focuses on the uncertainty estimation of LLMs, whereas this paper further translates uncertainty into abstention behavior.
- **vs Self-RAG**: Self-RAG allows the model to decide whether retrieval is necessary, which is similar in concept to the abstention decision in this paper but has a different objective.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The idea of selective abstention is inspiring, though the specific techniques for knowledge probing and abstention training are relatively conventional.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated on multiple hallucination evaluation benchmarks, with ablation studies covering key design choices.
- **Writing Quality**: ⭐⭐⭐ Unable to fully evaluate (full paper not reviewed).
- **Value**: ⭐⭐⭐⭐ Provides a new paradigm for hallucination mitigation; the abstention mechanism has broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Aligning Large Language Models to Follow Instructions and Hallucinate Less via Effective Data Filtering](aligning_large_language_models_to_follow_instructions_and_hallucinate_less_via_e.md)
- [\[AAAI 2026\] Hallucinate Less by Thinking More: Aspect-Based Causal Abstention for Large Language Models](../../AAAI2026/hallucination/hallucinate_less_by_thinking_more_aspect-based_causal_absten.md)
- [\[ACL 2025\] Activation Steering Decoding: Mitigating Hallucination in Large Vision-Language Models through Bidirectional Hidden State Intervention](activation_steering_decoding_mitigating_hallucination_in_large_vision-language_m.md)
- [\[ACL 2025\] On-Policy Self-Alignment with Fine-grained Knowledge Feedback for Hallucination Mitigation](on-policy_self-alignment_with_fine-grained_knowledge_feedback_for_hallucination_.md)
- [\[ACL 2025\] DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination](drag_distilling_rag_slm.md)

</div>

<!-- RELATED:END -->
