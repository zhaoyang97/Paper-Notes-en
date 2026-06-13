---
title: >-
  [Paper Note] Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents
description: >-
  [ACL 2026][Recommender Systems][Decision support] The DECISIVE interactive decision framework is proposed. By extracting objective option scoring matrices from unstructured documents and combining them with Bayesian pref…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "Decision support"
  - "preference elicitation"
  - "Bayesian inference"
  - "document grounding"
  - "interactive system"
date: 2026-05-08
content_hash: 346a97cb5c34f369
---

# Decisive: Guiding User Decisions with Optimal Preference Elicitation from Unstructured Documents

**Conference**: ACL 2026  
**arXiv**: [2604.18122](https://arxiv.org/abs/2604.18122)  
**Code**: None  
**Area**: Recommender Systems  
**Keywords**: Decision support, preference elicitation, Bayesian inference, document grounding, interactive system

## TL;DR

The DECISIVE interactive decision framework is proposed. By extracting objective option scoring matrices from unstructured documents and combining them with Bayesian preference inference, the system adaptively selects pairwise comparison questions to efficiently learn user latent preference vectors. This achieves transparent personalized recommendations while minimizing user interaction burden, improving decision accuracy by up to 20% over strong baselines.

## Background & Motivation

**Background**: Decision-making is a cognitively intensive task where users must synthesize information from multiple unstructured sources, weigh competing factors, and incorporate individual subjective preferences. Typical scenarios include choosing products, schools, or medical plans. Existing decision support tools include direct suggestion generation by LLMs and traditional decision support systems.

**Limitations of Prior Work**: When LLMs directly answer decision-making questions, they either provide information overload (listing all pros and cons without a clear recommendation) or are overly arbitrary (giving recommendations without transparency or consideration of personal preferences). Traditional decision support systems require structured input and explicit preference weights, but users often cannot accurately express their preferences—they "know what they want" but cannot articulate specific weight allocations.

**Key Challenge**: Effective decision support needs to simultaneously solve two problems: (1) objectively extracting multi-dimensional scores for options from unstructured information; (2) efficiently eliciting subjective user preferences. Existing methods either ignore the grounding of objective information (relying purely on LLM subjective judgment) or ignore the efficiency of preference elicitation (requiring users to fill out extensive questionnaires).

**Goal**: Construct an interactive decision framework that can objectively extract option information from documents while efficiently learning user preferences through minimal interaction, ultimately providing transparent and personalized recommendations.

**Key Insight**: The authors decompose the decision problem into "objective dimensions" and "subjective dimensions"—objective dimensions are addressed through document-grounded scoring matrices, and subjective dimensions are addressed via Bayesian preference inference. The bridge between the two is adaptively selected pairwise comparison questions.

**Core Idea**: Use a document-grounded option scoring matrix to provide an objective foundation, and efficiently learn the user's latent preference vector through adaptive pairwise comparison questions that maximize information gain. Combining both enables transparent, efficient, and personalized decision recommendations.

## Method

### Overall Architecture

The input to DECISIVE is a set of unstructured documents related to the decision (e.g., product reviews, school profiles) and a decision problem. The output is a personalized ranking and recommendation of options. The process consists of four steps: (1) extracting options and evaluation dimensions from documents to construct an objective scoring matrix; (2) designing initial pairwise comparison questions for the user; (3) updating the posterior distribution of preferences based on user answers and adaptively selecting the next question; (4) outputting the final recommendation when preferences converge or the interaction limit is reached.

### Key Designs

1.  **Document-Grounded Option Scoring Matrix**:
    *   **Function**: Extracts structured multi-dimensional scores from unstructured documents to provide an objective basis for decision-making.
    *   **Mechanism**: Uses an LLM to identify options and evaluation dimensions (e.g., price, quality, convenience) from source documents, then scores each option across each dimension based on document content, constructing an $m \times n$ scoring matrix ($m$ options, $n$ dimensions). The scoring process requires the LLM to cite document evidence to ensure traceability.
    *   **Design Motivation**: Grounding scores in document facts rather than the LLM's prior knowledge avoids fabricated scores or training bias. This also makes the recommendation process transparent, as users can view the document evidence for each score.

2.  **Bayesian Preference Inference**:
    *   **Function**: Infers the user's latent preference weight vector through their answers to pairwise comparison questions.
    *   **Mechanism**: Assumes the user has a latent preference vector $\mathbf{w} \in \mathbb{R}^n$ representing the importance weights of the evaluation dimensions, initialized with a uniform prior distribution. After each user response to a pairwise comparison like "Do you value A more than B?", the posterior distribution of preferences is updated via Bayesian updates. Final recommendations are based on the comprehensive score for each option, calculated as the product of the scoring matrix $S$ and the posterior mean $E[\mathbf{w}]$, i.e., $S \cdot E[\mathbf{w}]$.
    *   **Design Motivation**: Users do not need to provide numerical preference weights directly (which is unnatural); they only need to answer intuitive questions about what they value more. The Bayesian framework naturally handles uncertainty and gradually improves preference estimation with more responses.

3.  **Adaptive Elicitation via Information Gain Maximization**:
    *   **Function**: Selects the pairwise comparison question with the maximum information gain in each interaction round to minimize the user interaction burden.
    *   **Mechanism**: Among all possible pairwise dimension comparisons, the pair that maximizes the information gain of the final decision is selected. Formally: select question $q^*$ such that $q^* = \arg\max_q I(D; A_q | \mathcal{H})$, where $D$ is the final decision, $A_q$ is the user's answer to question $q$, and $\mathcal{H}$ is the interaction history. Intuitively, priority is given to dimension comparisons that impact the final recommendation ranking the most.
    *   **Design Motivation**: Random question selection is inefficient because many comparisons do not affect the final decision (e.g., a user's preference between two irrelevant dimensions doesn't change the final choice). Adaptive selection allows convergence to a reliable recommendation with fewer questions.

## Key Experimental Results

### Main Results

| Method | Decision Accuracy | User Satisfaction | Interaction Rounds |
| :--- | :--- | :--- | :--- |
| DECISIVE (Ours) | Optimal | Highest | 5-8 rounds to converge |
| GPT-4 Direct Recommendation | -20% | Lower | 0 (not personalized) |
| Traditional MCDM Methods | -15% | Medium | Full weight input required |
| Random Question Selection | -12% | Medium | More rounds required |

### Ablation Study

| Config | Decision Accuracy | Explanation |
| :--- | :--- | :--- |
| Full DECISIVE | Optimal | Document grounding + Bayesian inference + Adaptive selection |
| w/o Document Grounding (Free LLM scoring) | Significant Decline | LLM scoring is inconsistent and untraceable |
| w/o Adaptive Selection (Random questions) | Slow Convergence | Requires 2-3x more interaction rounds |
| w/o Bayesian Inference (Direct weight estimation) | Slight Decline | Uncertainty modeling contributes to robustness |

### Key Findings

*   Document grounding is the most critical component—without it, LLM scores exhibit significant training bias and inconsistency.
*   Adaptive question selection makes 5-8 interaction rounds sufficient for reliable recommendations, whereas random selection requires 15+ rounds.
*   Strong cross-domain generalizability: Performs excellently across different fields such as product choice, school selection, and travel planning.
*   Uncertainty estimation in the Bayesian framework can determine "when the recommendation is reliable enough"—automatically stopping questions when the posterior variance falls below a threshold.

## Highlights & Insights

*   The framework design, which **elegantly decomposes the decision problem into objective scoring + subjective preference**, is very clear. This decomposition allows each part to be independently optimized and evaluated.
*   **Adaptive pairwise comparison** is more natural as a preference elicitation interface than traditional weight sliders or Likert scales—users only need to make intuitive judgments rather than precise quantifications.
*   The framework can be transferred to any scenario requiring personalized recommendations, especially information-intensive decisions (e.g., choosing insurance plans, investment strategies).

## Limitations & Future Work

*   The quality of the scoring matrix depends on the completeness of the source documents; if critical information is missing, scores will be biased.
*   Assumes user preferences can be represented by a linear weighting model, though in reality, preferences may be non-linear (e.g., direct exclusion if a dimension falls below a threshold).
*   The linguistic quality of generated pairwise comparison questions may affect user understanding and response accuracy.
*   Future work could explore multi-round conversational preference elicitation (rather than just multiple-choice) and dynamic updates to the scoring matrix.

## Related Work & Insights

*   **vs LLM Direct Recommendation**: LLM recommendations are opaque and non-personalized; DECISIVE addresses these through explicit preference elicitation and document grounding.
*   **vs Traditional MCDM (Multi-Criteria Decision Making)**: Traditional MCDM (e.g., AHP, TOPSIS) requires users to provide full preference weights upfront; DECISIVE reduces user burden through adaptive learning.
*   **vs Conversational Recommendation**: Conversational recommendation elicits preferences through free-text interaction but is inefficient and difficult to converge; DECISIVE's structured pairwise comparison is more efficient.

## Rating

*   Novelty: ⭐⭐⭐⭐ The combination of document grounding, Bayesian preference inference, and adaptive selection is innovative in decision support.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation across multiple domains with detailed ablations, though lacks large-scale user studies.
*   Writing Quality: ⭐⭐⭐⭐ Clear framework description and persuasive motivation.
*   Value: ⭐⭐⭐⭐ Provides a principled framework for LLM-assisted decision-making with broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mirroring Users: Towards Building Preference-aligned User Simulator with User Feedback in Recommendation](mirroring_users_towards_building_preference-aligned_user_simulator_with_user_fee.md)
- [\[ACL 2026\] Learning to Retrieve User History and Generate User Profiles for Personalized Persuasiveness Prediction](learning_to_retrieve_user_history_and_generate_user_profiles_for_personalized_pe.md)
- [\[ACL 2026\] HORIZON: A Benchmark for in-the-wild User Behaviour Modeling](horizon_a_benchmark_for_in-the-wild_user_behaviour_modeling.md)
- [\[ACL 2026\] HARPO: Hierarchical Agentic Reasoning for User-Aligned Conversational Recommendation](harpo_hierarchical_agentic_reasoning_for_user-aligned_conversational_recommendat.md)
- [\[ACL 2026\] SenseJudge: Human-Centric Preference-Driven Judgment Framework](sensejudge_human-centric_preference-driven_judgment_framework.md)

</div>

<!-- RELATED:END -->
