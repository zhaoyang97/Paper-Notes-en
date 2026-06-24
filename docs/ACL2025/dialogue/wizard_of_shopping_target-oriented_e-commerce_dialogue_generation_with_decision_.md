---
title: >-
  [Paper Note] Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching
description: >-
  [ACL 2025][Dialogue Systems][Conversational Product Search] This paper proposes TRACER, a method that leverages decision tree models to plan dialogue paths, guiding two LLM agents (customer and seller) to generate natural and target-oriented e-commerce shopping dialogues. It also releases the Wizard of Shopping (WoS) dataset containing 3,600 dialogues, validating the effectiveness of the dataset on two downstream tasks: conversational query generation and product ranking.
tags:
  - "ACL 2025"
  - "Dialogue Systems"
  - "Conversational Product Search"
  - "Decision Tree Planning"
  - "LLM Dialogue Generation"
  - "E-commerce Dialogue Dataset"
  - "Dialogue State Tracking"
date: 2026-05-08
content_hash: 4132c8230cb6ca37
---

# Wizard of Shopping: Target-Oriented E-commerce Dialogue Generation with Decision Tree Branching

**Conference**: ACL 2025  
**arXiv**: [2502.00969](https://arxiv.org/abs/2502.00969)  
**Code**: None  
**Area**: Text Generation / Dialogue Systems  
**Keywords**: Conversational Product Search, Decision Tree Planning, LLM Dialogue Generation, E-commerce Dialogue Dataset, Dialogue State Tracking

## TL;DR

This paper proposes TRACER, a method that leverages decision tree models to plan dialogue paths, guiding two LLM agents (customer and seller) to generate natural and target-oriented e-commerce shopping dialogues. It also releases the Wizard of Shopping (WoS) dataset containing 3,600 dialogues, validating the effectiveness of the dataset on two downstream tasks: conversational query generation and product ranking.

## Background & Motivation

1. **Background**: Conversational Product Search (CPS) aims to understand user shopping intents, ask clarifying questions, and locate relevant products through interactive dialogues. Existing methods mainly rely on templated synthetic datasets for training, or use small-scale human-annotated data.

2. **Limitations of Prior Work**: (1) Lack of large-scale, high-quality CPS datasets—existing datasets are either templated (unnatural), extremely small in scale (e.g., MG-ShopDial contains only 64 dialogues), or private and irreproducible; (2) Directly generating dialogues using LLMs faces challenges such as hallucination, prompt vulnerability, and lack of control and planning capabilities; (3) Prior CPS systems were evaluated using templated clarifying questions, which fail to reflect real dialogue scenarios.

3. **Key Challenge**: High-quality CPS data is scarce and expensive to construct (human annotation is expensive, while direct LLM generation is uncontrollable), which severely hinders the development of intelligent shopping assistants.

4. **Goal**: How to controllably generate natural and target-oriented e-commerce shopping dialogues at scale?

5. **Key Insight**: Use a decision tree to plan the sequence of product attributes to be discussed in the dialogue (maximizing the efficiency of search space partitioning), and then use LLMs to "verbalize" the structured dialogue plan into natural conversation.

6. **Core Idea**: Decision tree planning guarantees efficiency and controllability, while LLM verbalization ensures naturalness. The two complement each other to generate high-quality e-commerce dialogues.

## Method

### Overall Architecture

TRACER generates dialogues in three steps: (1) Customer preference sampling—sampling products from the product catalog and randomly labeling their attributes as wanted/unwanted/optional; (2) Dialogue planning—using a decision tree model to decide which product attributes should be discussed in sequence (selecting the attribute with the highest information gain to narrow down the search space fastest); (3) Verbalization—inputting structured preferences and plans into the LLM to generate natural language dialogues.

### Key Designs

1. **Customer Preference Sampling**:

    - **Function**: Simulating a reasonable customer shopping demand for each dialogue.
    - **Mechanism**: Randomly select product $p$ from the catalog and randomly divide its attributes into three groups: wanted (directly use the product's attribute value), unwanted (replaced with other values of the same attribute, e.g., color=blue → color=red, indicating the customer does not want blue), and optional (indifferent). This constructs a preference vector $preference = [PC, (A_1, V_1, I_1), ..., (A_m, V_m, I_m)]$, where $I_i \in \{wanted, unwanted, optional\}$.
    - **Design Motivation**: Simulating real-world scenarios where customers have clear preferences, negative constraints, and indifferent attributes, which is more natural than setting all attributes to "wanted".

2. **Decision Tree-Based Dialogue Planning**:

    - **Function**: Selecting the optimal product attribute to query in each round of dialogue to minimize user search effort.
    - **Mechanism**: In each round, retrofitted product sets $P_o$ are retrieved based on the currently known preferences (RevPref). A decision tree is fit on a temporary training set constructed with attributes from $P_o$ (where features are attributes and labels are concatenated attribute value strings). The root node of the decision tree selects the attribute with the maximum information gain. Traversal of the decision tree yields the sequence of attributes to be discussed in the current turn. This process is iteratively executed until all products in $P_o$ meet the preferences or no more attributes are left to discuss.
    - **Design Motivation**: Drawing inspiration from search evaluation studies where user satisfaction is negatively correlated with search effort, the decision tree ensures that customers find the target product within the fewest conversational turns. Refitting the decision tree query-by-query is necessary because the search space changes dynamically.

3. **Verbalization**:

    - **Function**: Converting structured plans and preferences into natural language shopping dialogues.
    - **Mechanism**: Two generation styles are provided—(1) Interactive generation: The customer and seller LLMs take turns speaking, generating one utterance at a time based on the dialogue history, with a dialogue state tracker monitoring discussed/to-be-discussed attributes; (2) Single-pass generation: Inputting all preferences, plans, and product information at once into the LLM to generate the entire dialogue. Experiments show that single-pass generation yields higher quality with fewer contextual conflicts.
    - **Design Motivation**: Interactive generation is more realistic but error-prone (agents cannot anticipate future turns), whereas single-pass generation exhibits better global consistency.

### Extra Strategies to Improve Naturalness

- **Providing Value Suggestions**: The seller provides at most 3 most common attribute values as options (e.g., SSD capacity: 256GB/512GB/1TB) to prevent customers from being unaware of possible choices.
- **LLM Knowledge Offloading**: Allowing the LLM to rearrange the sequence of attributes in the plan and explain technical terms to make the dialogue more aligned with common sense.
- **Dialogue Ending**: Once the search converges, the seller recommends a product and continues for at most 3 turns to conclude the dialogue.

### Loss & Training

This paper does not involve end-to-end training; the core is dataset construction. In downstream tasks, LED (Longformer Encoder-Decoder) is used for seq2seq fine-tuning to perform conversational query generation.

## Key Experimental Results

### Human Evaluation (Dialogue Quality)

Dialogue-level evaluation (5-point Likert scale):

| Model | Strategy | Authenticity | Conciseness | Coherence | Naturalness |
|------|------|--------|--------|--------|--------|
| GPT-4 | Single-pass | **4.3** | **4.8** | **4.7** | **4.2** |
| GPT-4 | Interactive | 3.3 | 2.9 | 3.4 | 2.9 |
| LLaMA-2 | Single-pass | 4.1 | 4.7 | 4.1 | 3.9 |
| LLaMA-2 | Interactive | 2.5 | 3.0 | 3.3 | 2.5 |

Utterance-level evaluation (Number of bad utterances ↓):

| Model | Strategy | Unrealistic Utterances | Failure to Follow Script |
|------|------|-----------|-----------|
| GPT-4 | Single-pass | **0.8** | **0.1** |
| GPT-4 | Interactive | 2.0 | 0.4 |
| LLaMA-2 | Single-pass | 1.2 | 1.7 |
| LLaMA-2 | Interactive | 2.9 | 1.1 |

### Downstream Task: Conversational Query Generation (CQG)

| Method | F1 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|------|-----|---------|---------|---------|
| Baseline (Original Dialogues) | 0.008 | 0.137 | 0.047 | 0.087 |
| D2Q (GPT-4 few-shot) | 0.553 | 0.793 | 0.628 | 0.734 |
| D2Q (LED fine-tuned on WoS) | **0.834** | **0.899** | **0.822** | **0.873** |

### Ablation Study

| Dimension | Findings |
|---------|------|
| Single-pass vs Interactive | Single-pass ratings consistently outperform interactive ones. In the interactive approach, sellers tend to cram too many attributes into a single utterance. |
| GPT-4 vs LLaMA-2 | GPT-4 consistently outperforms LLaMA-2, which aligns with the overall capability of the language models. |
| WoS vs MG-ShopDial | WoS displays more "eliciting preference" intents, whereas MG-ShopDial displays more "recommendation" intents (due to its smaller product catalog leading to premature recommendations). |

### Key Findings

- Single-pass generation consistently outperforms interactive generation—global planning ensures consistency, whereas local interaction is prone to context conflicts.
- The naturalness of dialogues generated by GPT-4 in a single-pass manner reaches 4.2/5 (where 5 indicates indistinguishability from human-to-human dialogues), which is very close to human performance.
- Small models (LED) fine-tuned on WoS significantly outperform GPT-4 few-shot on the CQG task (F1: 0.834 vs 0.553), demonstrating the effectiveness of the dataset for downstream task training.
- WoS dialogues are notably more concise and efficient (average of 19.7 turns vs 34.3 turns for MG-ShopDial) because the decision tree guarantees the shortest path.

## Highlights & Insights

- **Clever complementary design of Decision Trees + LLMs**: Decision trees resolve planning and controllability issues that LLMs struggle with, while LLMs resolve the lack of naturalness in templated generation, leveraging the strengths of both.
- **Counter-intuitive finding of Single-pass > Interactive**: Although interactive dialogue aligns better with human dialogue intuition, single-pass generation is more stable due to global visibility, which provides valuable insights for dialogue data generation.
- **Three-way preference setting (wanted/unwanted/optional)**: This is more realistic than prior work that only considered "wanted". The introduction of "unwanted" attributes makes the dialogue replicate real shopping scenarios more accurately.
- **Strong scalability**: Dialogue data for new domains can be seamlessly generated as long as a product catalog of the target domain is provided.

## Limitations & Future Work

- Product attribute quality depends on the original catalog; some attributes (e.g., ASIN, Date First Available) are not intuitive and occasionally slip through despite data cleaning.
- LLMs sometimes fail to follow instructions, fabricating attributes that are not in the catalog.
- In interactive generation, the seller does not offer the customer a second chance to choose (causing incoherence).
- It only covers 3 domains (Home & Kitchen, Electronics, Beauty & Personal Care), which calls for expansion.
- Complex shopping scenarios, such as multi-round comparisons and product returns, are not yet considered.

## Related Work & Insights

- **vs MG-ShopDial (Bernard & Balog 2023)**: The only publicly available e-commerce dialogue dataset, containing only 64 dialogues and a very small product catalog (approximately 14 items per category). WoS contains 3,600 dialogues utilizing a real product catalog of 236k items.
- **vs Templated CPS (Zhang et al. 2018; Bi et al. 2019)**: Previous works simulated dialogues using slot-filling templates, which are unnatural. This paper verbalizes decision tree planning using LLMs, balancing both naturalness and controllability.
- **Insight**: The "planning + verbalization" paradigm of dialogue data generation can be transferred to other task-oriented dialogue tasks (e.g., medical inquiries, technical support).

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of decision tree planning and LLM verbalization is an innovative contribution to e-commerce dialogue generation, though the overall approach is somewhat engineering-oriented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Human evaluation, downstream task verification, error analysis, and comparison with MG-ShopDial provide a relatively comprehensive assessment.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, and the algorithm pseudocode and illustrative diagrams are intuitive, though some chapters are slightly verbose.
- Value: ⭐⭐⭐⭐ Releases the first large-scale target-oriented e-commerce dialogue dataset, which effectively advances the CPS field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Goal-oriented Proactive Dialogue Systems via Consistency Reflection and Correction](enhancing_goal-oriented_proactive_dialogue_systems_via_consistency_reflection_an.md)
- [\[ACL 2025\] Know Your Mistakes: Towards Preventing Overreliance on Task-Oriented Conversational AI Through Accountability Modeling](know_your_mistakes_towards_preventing_overreliance_on_task-oriented_conversation.md)
- [\[ACL 2025\] Exploring Persona Sentiment Sensitivity in Personalized Dialogue Generation](persona_sentiment_dialogue.md)
- [\[ACL 2025\] When Harry Meets Superman: The Role of The Interlocutor in Persona-Based Dialogue Generation](when_harry_meets_superman_the_role_of_the_interlocutor_in_persona-based_dialogue.md)
- [\[ACL 2026\] CoDial: Interpretable Task-Oriented Dialogue Systems Through Dialogue Flow Alignment](../../ACL2026/dialogue/codial_interpretable_task-oriented_dialogue_systems_through_dialogue_flow_alignm.md)

</div>

<!-- RELATED:END -->
