---
title: >-
  [Paper Note] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents
description: >-
  [ACL 2026][Recommender Systems][Decision Support] This paper proposes DECISIVE, an interactive decision-making framework that extracts an objective option scoring matrix from unstructured documents and combines it with Bayesian preference inference to adaptively select pairwise comparison questions, efficiently learning users' latent preference vectors. The system minimizes user interaction burden while delivering transparent, personalized recommendations, achieving up to 20% higher decision accuracy over strong baselines.
tags:
  - ACL 2026
  - Recommender Systems
  - Decision Support
  - Preference Elicitation
  - Bayesian Inference
  - Document Grounding
  - Interactive Systems
date: 2026-05-08
content_hash: 9a8e9f2af6429653
---

# Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents

**Conference**: ACL 2026
**arXiv**: [2604.18122](https://arxiv.org/abs/2604.18122)
**Code**: None
**Area**: Recommender Systems
**Keywords**: Decision Support, Preference Elicitation, Bayesian Inference, Document Grounding, Interactive Systems

## TL;DR

This paper proposes DECISIVE, an interactive decision-making framework that extracts an objective option scoring matrix from unstructured documents and combines it with Bayesian preference inference to adaptively select pairwise comparison questions, efficiently learning users' latent preference vectors. The system minimizes user interaction burden while delivering transparent, personalized recommendations, achieving up to 20% higher decision accuracy over strong baselines.

## Background & Motivation

**State of the Field**: Decision-making is a cognitively intensive task—users must synthesize information from multiple unstructured sources, weigh competing factors, and incorporate personal subjective preferences. Typical scenarios include choosing products, schools, or medical treatments. Existing decision-assistance tools include direct LLM-generated recommendations and traditional decision support systems.

**Limitations of Prior Work**: When LLMs directly answer decision-making queries, they either produce information overload (listing all pros and cons without a clear recommendation) or are overly prescriptive (providing recommendations that are opaque and indifferent to individual preferences). Traditional decision support systems require structured input and explicit preference weights, yet users often cannot accurately articulate their own preferences—they "know what they want" but cannot specify concrete weight assignments.

**Root Cause**: Effective decision support must simultaneously address two challenges: (1) objectively extracting multi-dimensional scores for options from unstructured information, and (2) efficiently eliciting the user's subjective preferences. Existing methods either neglect grounding in objective information (relying entirely on LLM subjective judgment) or ignore the efficiency of preference elicitation (requiring users to complete lengthy questionnaires).

**Paper Goals**: To build an interactive decision-making framework that objectively extracts option information from documents, efficiently learns user preferences through minimal interaction, and ultimately delivers transparent, personalized recommendations.

**Starting Point**: The authors decompose the decision problem into an "objective dimension" and a "subjective dimension"—the former is addressed via a document-grounded scoring matrix, and the latter via Bayesian preference inference. Adaptively selected pairwise comparison questions serve as the bridge between the two.

**Core Idea**: A document-grounded option scoring matrix provides the objective foundation; adaptively selected pairwise comparison questions, chosen to maximize information gain, efficiently learn the user's latent preference vector. Combining both components enables transparent, efficient, and personalized decision recommendations.

## Method

### Overall Architecture

DECISIVE takes as input a set of unstructured documents relevant to a decision (e.g., product reviews, school descriptions) and a decision query, and outputs a personalized ranking of options with a recommendation. The pipeline proceeds in four steps: (1) extract options and evaluation dimensions from the documents to construct an objective scoring matrix; (2) present an initial pairwise comparison question to the user; (3) update the posterior distribution over preferences based on the user's response and adaptively select the next question; (4) output the final recommendation when preferences converge or a maximum number of interactions is reached.

### Key Designs

1. **Document-Grounded Option Scoring Matrix**:

    - Function: Extracts structured multi-dimensional scores from unstructured documents to provide an objective foundation for decision-making.
    - Mechanism: An LLM identifies options and evaluation dimensions (e.g., price, quality, convenience) from source documents, then scores each option on each dimension based on document content, constructing an $m \times n$ scoring matrix ($m$ options, $n$ dimensions). The scoring process requires the LLM to cite documentary evidence, ensuring traceability.
    - Design Motivation: Anchoring scores to documentary facts rather than the LLM's prior knowledge prevents hallucinated or training-biased scores, and makes the recommendation process transparent—users can inspect the documentary basis for each score.

2. **Bayesian Preference Inference**:

    - Function: Infers the user's latent preference weight vector from responses to pairwise comparison questions.
    - Mechanism: The user is assumed to possess a latent preference vector $\mathbf{w} \in \mathbb{R}^n$ representing the importance weights assigned to each evaluation dimension. This is initialized as a uniform prior. Each time the user answers a pairwise question ("Do you prioritize A or B more?"), the posterior distribution over preferences is updated via Bayes' rule. The final recommendation is derived from the product of the scoring matrix and the posterior mean, $S \cdot E[\mathbf{w}]$, yielding an aggregate score for each option.
    - Design Motivation: Users need not directly specify numerical preference weights (which is unnatural); they only answer intuitive questions about relative importance. The Bayesian framework naturally handles uncertainty, and preference estimates grow increasingly precise as more responses are collected.

3. **Adaptive Elicitation via Information Gain Maximization**:

    - Function: Selects the pairwise comparison question with the highest information gain at each interaction turn, minimizing user burden.
    - Mechanism: Among all possible pairwise dimension comparisons, the system selects the pair that maximizes the information gain about the final decision. Formally, the selected question is $q^* = \arg\max_q I(D; A_q | \mathcal{H})$, where $D$ is the final decision, $A_q$ is the user's answer to question $q$, and $\mathcal{H}$ is the history of prior responses. Intuitively, the system prioritizes dimension comparisons that most strongly affect the final recommendation ranking.
    - Design Motivation: Random question selection is inefficient—many comparisons have no bearing on the final decision (e.g., user preferences on two irrelevant dimensions may not alter the outcome). Adaptive selection converges to a reliable recommendation with the fewest possible questions.

## Key Experimental Results

### Main Results

| Method | Decision Accuracy | User Satisfaction | Interaction Turns |
|--------|------------------|-------------------|-------------------|
| DECISIVE | Best | Highest | Converges in 5–8 turns |
| GPT-4 Direct Recommendation | −20% | Lower | 0 (but non-personalized) |
| Traditional MCDM | −15% | Moderate | Requires full weight input |
| Random Question Selection | −12% | Moderate | Requires more turns |

### Ablation Study

| Configuration | Decision Accuracy | Note |
|---------------|------------------|-------|
| Full DECISIVE | Best | Document grounding + Bayesian inference + adaptive selection |
| w/o Document Grounding (free LLM scoring) | Significant drop | LLM scores are inconsistent and untraceable |
| w/o Adaptive Selection (random questions) | Slower convergence | Requires 2–3× more interaction turns |
| w/o Bayesian Inference (direct weight estimation) | Slight drop | Uncertainty modeling contributes to robustness |

### Key Findings

- Document grounding is the most critical component—removing it introduces significant training bias and inconsistency in LLM-generated scores.
- Adaptive question selection typically achieves reliable recommendations within 5–8 interaction turns, whereas random selection requires 15+ turns.
- The framework generalizes well across domains, performing strongly in product selection, school selection, travel planning, and other settings.
- The uncertainty estimates from the Bayesian framework can be used to determine "when a recommendation is sufficiently reliable"—elicitation automatically stops when the posterior variance falls below a threshold.

## Highlights & Insights

- The framework design that **elegantly decomposes the decision problem into objective scoring and subjective preference** is notably clean. This decomposition allows each component to be independently optimized and evaluated.
- **Adaptive pairwise comparison** as a preference elicitation interface is more natural than traditional weight sliders or Likert scales—users make intuitive judgments rather than precise quantifications.
- The framework is transferable to any scenario requiring personalized recommendations, particularly information-intensive decisions such as choosing insurance plans or investment strategies.

## Limitations & Future Work

- The quality of the scoring matrix depends on the completeness of the source documents—if critical information is absent, scores will be biased.
- The framework assumes user preferences can be represented by a linear weighted model, whereas real-world preferences may be non-linear (e.g., options falling below a threshold on a given dimension are immediately eliminated).
- The quality of natural language generation for pairwise comparison questions may affect user comprehension and response accuracy.
- Future work could explore multi-turn conversational preference elicitation (beyond multiple-choice questions) and dynamic updating of the scoring matrix.

## Related Work & Insights

- **vs. Direct LLM Recommendation**: LLM recommendations are opaque and non-personalized; DECISIVE addresses both issues through explicit preference elicitation and document grounding.
- **vs. Traditional MCDM (Multi-Criteria Decision Making)**: Traditional MCDM methods (e.g., AHP, TOPSIS) require users to provide complete preference weights upfront; DECISIVE reduces user burden through adaptive learning.
- **vs. Conversational Recommendation**: Conversational recommenders elicit preferences via free-text interaction, but this is inefficient and difficult to converge; DECISIVE's structured pairwise comparisons are more effective.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of document grounding, Bayesian preference inference, and adaptive selection is innovative in the decision support domain.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-domain evaluation and detailed ablation study, though a large-scale user study is lacking.
- Writing Quality: ⭐⭐⭐⭐ Framework description is clear and motivation is convincingly articulated.
- Value: ⭐⭐⭐⭐ Provides a principled framework for LLM-assisted decision-making with broad application prospects.

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[ACL 2026\] What Makes LLMs Effective Sequential Recommenders? A Study on Preference Intensity and Temporal Context](what_makes_llms_effective_sequential_recommenders_a_study_on_preference_intensit.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[AAAI 2026\] Preference is More Than Comparisons: Rethinking Dueling Bandits with Augmented Human Feedback](../../AAAI2026/recommender/preference_is_more_than_comparisons_rethinking_dueling_bandits_with_augmented_hu.md)

<!-- RELATED:END -->
