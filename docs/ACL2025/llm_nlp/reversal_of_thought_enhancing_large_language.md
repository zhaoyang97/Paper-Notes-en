---
title: >-
  [Paper Note] Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up
description: >-
  [ACL 2025][LLM (Other)][reverse reasoning] Proposes Reversal of Thought (RoT), a plug-and-play reasoning framework. Through a preference-guided reverse reasoning warm-up strategy, RoT enables LLMs to back-generate "LLM-flavored" optimal prompts from examples, and utilizes a Cognitive Preference Manager to automatically distinguish between known and unknown tasks, outperforming baselines like CoT, ToT, and GoT on multiple reasoning tasks.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "reverse reasoning"
  - "cognitive preference"
  - "meta-cognition"
  - "prompt optimization"
  - "knowledge boundary"
date: 2026-05-08
content_hash: 366b97e1c74f4655
---

# Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up

**Conference**: ACL 2025  
**arXiv**: [2410.12323](https://arxiv.org/abs/2410.12323)  
**Code**: None  
**Area**: LLM/NLP  
**Keywords**: reverse reasoning, cognitive preference, meta-cognition, prompt optimization, knowledge boundary

## TL;DR
Proposes Reversal of Thought (RoT), a plug-and-play reasoning framework. Through a preference-guided reverse reasoning warm-up strategy, RoT enables LLMs to back-generate "LLM-flavored" optimal prompts from examples, and utilizes a Cognitive Preference Manager to automatically distinguish between known and unknown tasks, outperforming baselines like CoT, ToT, and GoT on multiple reasoning tasks.

## Background & Motivation

**Background**: CoT and its variants (ToT, GoT, BoT) improve LLM logical capabilities through multi-step reasoning, but face challenges such as high reasoning costs, cascading errors, and poor flexibility.

**Limitations of Prior Work**: (1) Multi-step CoT increases reasoning cost without necessarily improving logical accuracy. (2) Existing methods are prone to generating cascading errors (unfaithful reasoning). (3) BoT retrieves thought templates via RAG but relies on manual pre-definition, lacking flexibility.

**Key Challenge**: How to enhance the logical reasoning ability of LLMs without increasing reasoning costs? How to utilize the cognitive preferences formed in LLMs through RLHF training?

**Goal**: (1) How to enable LLMs to output prompts that suit their own cognitive preferences? (2) How to automatically expand the knowledge boundary of LLMs for unknown tasks?

**Key Insight**: Inspired by metacognition theory and cognitive preferences—LLMs develop specific design/cognitive paradigm preferences after pre-training and RLHF. Reverse reasoning can activate these preferences and generate prompts better suited to the model.

**Core Idea**: Given examples (input-output pairs), the LLM is prompted to perform reverse reasoning to determine "what kind of prompt can generate these results", and the optimal prompt is selected through preference ranking.

## Method

### Overall Architecture
Two stages: (1) Preference-Guided Reverse Reasoning (PGRR) $\to$ generate multiple candidate prompts $\to$ pairwise preference evaluation $\to$ select the optimal "LLM-flavored" prompt. (2) Cognitive Preference Manager (CPM) $\to$ determine whether the task lies within the knowledge boundary $\to$ aggregate problem-solving logic for known tasks $\to$ transfer cognitive style templates for unknown tasks.

### Key Designs

1. **Preference-Guided Reverse Reasoning (PGRR)**:

    - Function: Given input-output examples $D$, query the LLM multiple times using a reverse prompt $P_r$ to generate candidate prompts $R = \{R_1, ..., R_{warm}\}$.
    - Mechanism: (1) Reverse reasoning warm-up—generating $warm$ candidates; (2) Pairwise preference evaluation—allowing the LLM to compare adjacent candidates' preferences $P_{pre}(R_{i+1} \succ R_i)$ and constructing a preference matrix using preference transitivity; (3) Preference ranking—integrating the average generation probability and preference score to select the optimal prompt $P_{opt}$.
    - Design Motivation: Leveraging preferences formed during RLHF to select prompts that align with the model's cognitive habits is more effective than random selection or manual design.

2. **Metacognitive Logic Reinforcement**:

    - Function: Incorporate logical pseudocode (algorithmic structures, mathematical symbols) into reverse reasoning to enhance reasoning comprehension.
    - Mechanism: Introduce logical operators, quantifiers, inequalities, and conditional statements to guide the LLM to think in a semi-formal manner.
    - Design Motivation: Pure natural language reasoning is prone to ambiguity, whereas pseudocode constraints make reasoning more precise.

3. **Cognitive Preference Manager (CPM)**:

    - Function: Determine whether reverse reasoning is within the knowledge boundary of the LLM, and then handle different cases separately.
    - Mechanism: Use an offline LLM embedding model to calculate the similarity between the original task definition $P_{task}$ and the LLM's cognitive task definition $P^*_{task}$. If $\text{sim} \geq \delta$ (known), aggregate problem-solving logic to optimize the prompt; if $\text{sim} < \delta$ (unknown), transfer and expand cognitive style templates.
    - Design Motivation: Known and unknown tasks require different strategies—solving logic for known tasks, and style transfer for unknown tasks. The threshold is set to $\delta \in [0.6, 0.8]$.

### Loss & Training
RoT works entirely during the inference phase and requires no additional training. It is used during the "warm-up" phase prior to batch inference.

## Key Experimental Results

### Main Results

| Method | GSM8K | AQUA | LogiQA | Average | API Calls |
|------|:---:|:---:|:---:|:---:|:---:|
| CoT | Baseline | Baseline | Baseline | Baseline | 1x |
| ToT | Gain | Gain | Gain | Higher | Multiple times |
| GoT | Gain | Gain | Gain | Higher | Multiple times |
| BoT | Gain | Gain | Gain | High | 2-3x |
| **RoT** | **Optimal** | **Optimal** | **Optimal** | **Optimal** | **warm+1** |

### Ablation Study

| Configuration | Effect | Description |
|------|------|------|
| w/o Reverse Reasoning | Significant Drop | Core component |
| w/o Preference Evaluation | Drop | Random candidate selection is inferior to preference ranking |
| w/o CPM | Drop | Unable to distinguish between known/unknown tasks |
| w/o Pseudocode | Slight Drop | Metacognitive logic provides auxiliary assistance |

### Key Findings
- **RoT outperforms baselines in both accuracy and efficiency**: It requires far fewer API calls than ToT/GoT while achieving higher accuracy.
- **Preference evaluation is crucial**: Leveraging the LLM's own preferences to select prompts shows a significant improvement over random selection.
- **Knowledge boundary detection is effective**: CPM accurately distinguishes between known and unknown tasks, validating the effectiveness of the strategy (style transfer) for unknown tasks.

## Highlights & Insights
- **The philosophy of "participation over teaching"**: Instead of telling the LLM how to think (CoT), it allows the LLM to discover the most suitable way of thinking for itself.
- **Preference transitivity reduces evaluation cost**: A complete preference matrix can be constructed with only $O(warm)$ pairwise comparisons instead of $O(warm^2)$.
- **Plug-and-play design**: It does not modify models or require training; it only requires a warm-up phase before inference, offering strong utility.

## Limitations & Future Work
- The warm-up phase still requires multiple LLM calls, which may not be suitable for API cost-sensitive scenarios.
- The preference transitivity assumption $P(A>B) \times P(B>C) = P(A>C)$ does not necessarily hold true.
- The knowledge boundary threshold $\delta$ requires manual tuning, lacking an adaptive mechanism.

## Related Work & Insights
- **vs BoT**: BoT relies on manually predefined thought templates, whereas RoT automatically generates "LLM-flavored" prompts, offering greater flexibility.
- **vs Auto-Prompt**: Automatic prompt optimization typically requires annotated data or reinforcement learning; RoT generates prompts through reverse reasoning requiring only examples.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of reverse reasoning and preference ranking is creative
- Experimental Thoroughness: ⭐⭐⭐ Covers multiple tasks but some experimental details lack depth
- Writing Quality: ⭐⭐⭐ Interesting concepts but the organization is slightly complex
- Value: ⭐⭐⭐⭐ A plug-and-play reasoning enhancement solution with good utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Reason from Future: Reverse Thought Chain Enhances LLM Reasoning](reason_from_future_reverse_thought_chain_enhances_llm_reasoning.md)
- [\[ACL 2025\] Clue Guided Re-Assessment to Improve Reasoning in Large Language Models](clue_guided_re-assessment_to_improve_reasoning_in_large_language_models.md)
- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[ACL 2025\] Assessing and Enhancing the Causal Reasoning Abilities of Language Models via Faithful Textual Interpretation](assessing_and_enhancing_the_causal_reasoning_abilities_of_language_models_via_fai.md)
- [\[ACL 2025\] The Role of Deductive and Inductive Reasoning in Large Language Models](the_role_of_deductive_and_inductive_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
