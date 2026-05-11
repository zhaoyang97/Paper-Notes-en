---
title: >-
  [Paper Note] Evaluating, Synthesizing, and Enhancing for Customer Support Conversation
description: >-
  [AAAI 2026][LLM Reasoning][Customer support conversation] This paper defines five dialogue phases and twelve support strategies based on the COPC industry standard, generates 11…
tags:
  - "AAAI 2026"
  - "LLM Reasoning"
  - "Customer support conversation"
  - "COPC standard"
  - "role-playing"
  - "data synthesis"
  - "strategy alignment"
date: 2026-05-08
content_hash: f1c93beada601dd0
---

# Evaluating, Synthesizing, and Enhancing for Customer Support Conversation

**Conference**: AAAI 2026
**arXiv**: [2508.04423](https://arxiv.org/abs/2508.04423)
**Code**: [https://github.com/aliyun/qwen-dianjin](https://github.com/aliyun/qwen-dianjin)
**Area**: LLM Reasoning / Dialogue Systems
**Keywords**: Customer support conversation, COPC standard, role-playing, data synthesis, strategy alignment

## TL;DR
This paper defines five dialogue phases and twelve support strategies based on the COPC industry standard, generates 11,232 strategy-rich synthetic dialogues (RoleCS) via five-agent role-playing, and constructs a 1,855-sample evaluation set (CSConv) by rewriting real conversations. Fine-tuning on these resources substantially improves strategy-aligned response quality and issue resolution rates.

## Background & Motivation
**Background**: Customer support dialogue AI has primarily focused on task-oriented dialogue systems (e.g., MultiWOZ, NATCS), emphasizing task completion. Emotional support conversation research (e.g., ESConv) addresses emotional guidance but is not tailored to customer service contexts.

**Limitations of Prior Work**: (a) Most customer support datasets derive from asynchronous interactions (e.g., Twitter), which differ in immediacy from real-time support; (b) task-oriented dialogue datasets lack deliberate use of support strategies (e.g., emotion management, empathetic care), focusing solely on information transfer; (c) real customer support data is difficult to obtain and sensitive, making annotation challenging.

**Key Challenge**: High-quality customer support requires not only problem resolution but also structured, empathetic communication — yet no systematic framework grounded in industry standards exists to define and evaluate this capability.

**Goal**: To systematically define a strategy framework for customer support dialogues and generate sufficiently diverse training data to enhance LLMs' strategic customer support capabilities.

**Key Insight**: Drawing on the COPC (Customer Operations Performance Center) international standard and emotional support dialogue research, the authors collaborate with domain experts to define a five-phase, twelve-strategy framework.

**Core Idea**: A unified approach combining a COPC-aligned framework definition, multi-agent role-playing for synthetic training data, and a real-dialogue rewriting evaluation set to holistically improve LLM customer support capabilities.

## Method

### Overall Architecture
Given a customer service scenario description and customer profile as input, the system produces strategy-aligned, high-quality support dialogues. The pipeline comprises three components: (1) the CSC framework defining five phases and twelve strategies; (2) RoleCS generating training data via role-playing; (3) CSConv constructing an evaluation set by rewriting real dialogues.

### Key Designs

1. **CSC Framework (5 Phases × 12 Strategies)**:

    - **Function**: Provides structured strategic guidance for customer support dialogues.
    - **Mechanism**: Five dialogue phases: Connecting (relationship establishment) → Identifying (problem and emotion identification) → Exploring (solution exploration) → Resolving (solution implementation) → Maintaining (relationship maintenance). Twelve strategies: Greeting (GT), Identity Verification (IV), Emotion Management (EM), Restatement/Paraphrasing (RP), Problem Refinement (PR), Problem-Solving Suggestion (PS), Information Delivery (ID), Resolution Implementation (RI), Feedback Request (FR), Appreciation and Closure (AC), Relationship Continuation (RC), and Others.
    - **Design Motivation**: The framework is defined in collaboration with domain experts based on COPC operational guidelines and the emotional support literature, ensuring both industry authority and operational executability.

2. **RoleCS Role-Playing Synthesis (5-Agent Framework)**:

    - **Function**: Generates large-scale, strategy-rich, and diverse training dialogues.
    - **Mechanism**: Five agents with distinct roles — the Planner selects topics and customer profiles → generates scenarios and communication goals $(g, e') = \mathcal{M}(o, e)$; the Supporter Assistant recommends strategies $t = \mathcal{M}(h_s, G, e')$; the Supporter generates responses $r_s = \mathcal{M}(h_s, t, e')$; the Customer Assistant plans dialogue direction $d = \mathcal{M}(h_c, g, e')$; the Customer generates user utterances $r_c = \mathcal{M}(h_c, d, o, e')$. All agents are implemented using DeepSeek-R1.
    - **Design Motivation**: (a) The Assistant agents ensure strategy consistency and persona fidelity, offering greater controllability than single-LLM generation; (b) a pool of 1,948 deduplicated customer profiles (cosine similarity threshold > 0.85) promotes dialogue diversity.

3. **CSConv Evaluation Set Construction**:

    - **Function**: Constructs a high-quality evaluation set from real customer support dialogues.
    - **Mechanism**: 690K real Chinese customer support dialogues → rule-based filtering (length, balance, quality) → sampling 500 dialogues per topic → LLM-based rewriting to align with the CSC framework (preserving original semantics and intent) → secondary filtering → manual review by COPC-certified experts → final 1,855 samples.
    - **Design Motivation**: Directly annotating strategies in raw dialogues is infeasible due to inconsistent strategy usage (original dialogue strategy utilization rate: 55.28%); after rewriting, the strategy utilization rate reaches 97.82%.

### Loss & Training
- The CSC task is decomposed into two sub-tasks: (1) strategy prediction — predicting the next strategy $T_k \in G$ given dialogue history; (2) response generation — generating response $U_k$ conditioned on the predicted strategy and dialogue history.
- DeepSeek-R1 is used for rewriting (dialogues generated by GPT-4o were found to be shorter and less emotionally expressive).

## Key Experimental Results

### Main Results (Response Generation on CSConv)

| Model | Ref-Context B-4 | Ref-Context BS | Strategy ACC | Gen-Context B-4 | Gen-Context ACC |
|-------|----------------|----------------|-------------|----------------|----------------|
| GPT-4o | 2.97 | 64.26 | 42.58% | 2.07 | 36.29% |
| DeepSeek-R1 671B | 5.09 | — | — | — | — |
| Qwen2.5-72B + RoleCS Fine-tuning | Best | Best | Best | Best | Best |

### Ablation Study

| Configuration | Observation |
|---------------|-------------|
| With strategy prediction + response generation | Best — explicit strategy prediction substantially improves strategy alignment of responses. |
| Response generation only (no strategy) | Strategy ACC decreases, confirming the necessity of explicit strategy prediction. |
| RoleCS training vs. CSConv training | Fine-tuning on RoleCS synthetic data outperforms fine-tuning on the smaller real-data set. |

### Key Findings
- Strategy utilization rate rises sharply from 55.28% to 97.82% after rewriting, demonstrating that LLM-based rewriting effectively aligns dialogues with the CSC framework.
- Information Delivery (14.9%), Emotion Management (11.9%), and Problem-Solving Suggestion (10.0%) are the three most frequently used strategies, reflecting the dual demands of customer support: information transfer and emotion regulation.
- Customer utterances consistently exhibit higher lexical overlap with their assigned profiles than with random profiles, validating persona fidelity in role-playing.
- Human evaluation confirms that strategy-aligned responses yield significantly higher issue resolution rates.

## Highlights & Insights
- **First integration of COPC industry standards into customer support dialogue AI research**: This bridges the gap between academic research and industry practice, providing an authoritative benchmark for evaluating customer service AI.
- **The 5-agent role-playing framework** is elegantly designed: each agent has a clearly defined role, and the Assistant agents function as "strategy advisors" to ensure the primary agents adhere to the framework. This design is transferable to other structured communication scenarios (e.g., psychological counseling, educational tutoring).
- **Synthetic data outperforms real data** in fine-tuning effectiveness, demonstrating the substantial value of high-quality structured synthetic data in data-scarce domains.

## Limitations & Future Work
- CSConv covers only eight topics in Chinese customer support; generalization to multiple languages and broader domains remains to be validated.
- LLM-based rewriting may introduce rewriting bias — rewritten dialogues may be overly "idealized" and less reflective of the messiness of real customer interactions.
- The optimality of the 12-strategy granularity is unclear: finer granularity may increase prediction difficulty, while coarser granularity may reduce guidance effectiveness.
- Role-playing-generated dialogues lack real system interactions (e.g., order lookup, API calls), leaving a gap toward fully automated customer support agents.

## Related Work & Insights
- **vs. ESConv**: ESConv addresses emotional support but is constructed via Wizard-of-Oz crowdsourcing, with dialogue quality constrained by annotator proficiency; this paper ensures quality through LLM-based rewriting of real dialogues combined with expert review.
- **vs. NATCS**: NATCS provides naturalistic customer support dialogues but lacks strategy annotation; this paper defines and annotates strategies through the CSC framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Both the COPC-aligned framework and the 5-agent role-playing design are novel contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual datasets (synthetic and real) plus human evaluation.
- Writing Quality: ⭐⭐⭐⭐ Framework definitions are clear and experimental presentation is complete.
- Value: ⭐⭐⭐⭐ Directly applicable to customer support AI systems; production feasibility is reinforced by the Alibaba Cloud team's involvement.

## Additional Notes
- The methodology and experimental design of this work offer valuable reference points for related research areas.
- Future work may validate the generalizability and scalability of the approach across more scenarios and at larger scale.
- Integration with recent related work (e.g., intersections with RL/MCTS/multimodal methods) presents potential research directions.
- Deployment feasibility and computational efficiency should be assessed against practical application requirements.
- The choice of datasets and evaluation metrics may affect the generality of conclusions; cross-validation on additional benchmarks is recommended.

## Additional Notes
- The methodology and experimental design of this work offer valuable reference points for related research areas.
- Future work may validate the generalizability and scalability of the approach across more scenarios and at larger scale.
- Integration with recent related work (e.g., intersections with RL/MCTS/multimodal methods) presents potential research directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] CMMCoT: Enhancing Complex Multi-Image Comprehension via Multi-Modal Chain-of-Thought and Memory Augmentation](cmmcot_enhancing_complex_multi-image_comprehension_via_multi.md)
- [\[AAAI 2026\] Jupiter: Enhancing LLM Data Analysis Capabilities via Notebook and Inference-Time Value-Guided Search](jupiter_enhancing_llm_data_analysis_capabilities_via_notebook_and_inference-time.md)
- [\[ACL 2026\] MARCH: Evaluating the Intersection of Ambiguity Interpretation and Multi-hop Inference](../../ACL2026/llm_reasoning/march_evaluating_the_intersection_of_ambiguity_interpretation_and_multi-hop_infe.md)
- [\[NeurIPS 2025\] Note 8: PolyMath — Evaluating Mathematical Reasoning in a Multilingual Context](../../NeurIPS2025/llm_reasoning/self-evaluating_llms_for_multi-step_tasks_stepwise_confidence_estimation_for_fai.md)
- [\[NeurIPS 2025\] The Hawthorne Effect in Reasoning Models: Evaluating and Steering Test Awareness](../../NeurIPS2025/llm_reasoning/the_hawthorne_effect_in_reasoning_models_evaluating_and_steering_test_awareness.md)

</div>

<!-- RELATED:END -->
