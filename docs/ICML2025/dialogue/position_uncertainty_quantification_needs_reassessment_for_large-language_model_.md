---
title: >-
  [Paper Note] Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents
description: >-
  [ICML 2025][Dialogue Systems][uncertainty quantification] This position paper argues that the traditional dichotomy between aleatoric and epistemic uncertainty fundamentally fails in interactive LLM scenarios by reviewing the conflicting definitions in literature. It proposes three new research directions: underspecification uncertainty (task/context under-specification), interactive learning (reducing uncertainty through follow-up questions)…
tags:
  - "ICML 2025"
  - "Dialogue Systems"
  - "uncertainty quantification"
  - "aleatoric"
  - "epistemic"
  - "LLM agents"
  - "position paper"
date: 2026-05-08
content_hash: 2168d4b69c834e68
---

# Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents

**Conference**: ICML 2025  
**Authors**: Michael Kirchhof, Gjergji Kasneci, Enkelejda Kasneci  
**arXiv**: [2505.22655](https://arxiv.org/abs/2505.22655)  
**Code**: None  
**Area**: Dialogue Systems  
**Keywords**: uncertainty quantification, aleatoric, epistemic, LLM agents, position paper  

## TL;DR

This position paper argues that the traditional dichotomy between aleatoric and epistemic uncertainty fundamentally fails in interactive LLM scenarios by reviewing the conflicting definitions in literature. It proposes three new research directions: underspecification uncertainty (task/context under-specification), interactive learning (reducing uncertainty through follow-up questions), and output uncertainty (expressing uncertainty via natural language rather than scalar values).

## Background & Motivation

**Background**: LLMs and chatbots inevitably produce hallucinations, and recent research theoretically proves that this issue cannot be entirely eliminated. Consequently, uncertainty quantification (UQ) is crucial—either by outputting a total uncertainty score or separating it into aleatoric (irreducible) and epistemic (reducible) numerical components.

**Limitations of Prior Work**: Traditional UQ frameworks exhibit fundamental flaws in multi-turn interactive dialogues with LLM agents: (1) conflicting definitions of aleatoric and epistemic uncertainty persist within the community; (2) the estimated uncertainties are highly correlated in practice (correlation coefficients of 0.8–0.999), making true decoupling impossible; (3) in multi-turn interactions, the category of uncertainty can repeatedly shift between aleatoric and epistemic—what is "irreducible" can become "reducible" as users provide more information, and "reducible" can revert to "irreducible" if the agent decides to stop asking.

**Key Challenge**: Traditional UQ assumes fixed input formats (e.g., an image or a feature vector), whereas LLM agents operate in open environments where the task itself may be ambiguous, input information may be incomplete, and the output need not be a probabilistic scalar but rather explanatory language.

**Key Insight**: Instead of disputing which aleatoric/epistemic definition is "correct", this paper directly argues that this dichotomy is inapplicable to LLM agents. Instead, it proposes a classification and management of uncertainty better suited for interactive settings.

**Core Idea**: The uncertainty of an LLM agent should not be compressed into two scalars (aleatoric and epistemic), but should instead center on a three-stage pipeline: "detecting under-specification — interactive reduction — rich-text expression."

## Method

### Overall Architecture

Rather than proposing a specific algorithm, this paper constructs a three-tier argument:
1. **Deconstruction** (Sec 2): Systematically reviews conflicting definitions of aleatoric/epistemic uncertainty, proving their inapplicability to LLM agents.
2. **Construction** (Sec 3): Proposes three new research directions.
3. **Counter-argument** (Sec 4): Fairly discusses opposing views where traditional UQ remains valuable.

### Key Designs

1. **Conflict in the Definition of Epistemic Uncertainty**:

    - Core Argument: Consider a Bernoulli classification problem where the learner believes only $\theta=0$ or $\theta=1$ are possible. (a) **Disagreement School** (Houlsby et al., Gal et al.) defines it using mutual information $I(y;\theta)$—with maximum disagreement between the two beliefs, epistemic uncertainty is **maximized**. (b) **Plausible Model Count School** (Wimmer et al.) uses an axiomatic definition—only two possible models remain, so epistemic uncertainty is **near its minimum**. (c) **Density School** (Mukhoti et al.) defines it by training data density—depending on the distance of $x$ from the training data, the **answer is uncertain**. Three theoretically grounded definitions yield three contradictory conclusions.
    - Significance: Demonstrates that epistemic uncertainty is not a universally agreed-upon concept, and different theoretical frameworks lead to entirely opposite conclusions.

2. **The "Reducible Irreducibility" of Aleatoric Uncertainty**:

    - Core Argument: When the class of models is linear but the data generation process is non-linear, the optimal linear model still has residual risk (model bias). The **Bayesian Optimality School** (Schweighofer et al.) considers this irreducible $\to$ aleatoric. The **Data Uncertainty School** (Lahlou et al.) argues that switching to a stronger model class can eliminate it $\to$ not aleatoric. Crucially, if epistemic = total - aleatoric, then the boundary of aleatoric directly dictates the value of epistemic—and this boundary is a subjective choice.
    - Significance: The so-called "irreducible" uncertainty depends heavily on where the boundaries of the considered model class are set.

3. **Underspecification Uncertainty**:

    - Function: Addresses the unique issue of "unclear tasks/contexts" in LLM agents.
    - Mechanism: Categorized into **task underspecification** (unclear user intent, $P(y|x) = \int_{t \in \mathcal{T}} P(y|t) P(t|x) dt$, where an unknown task $t$ introduces extra uncertainty) and **context underspecification** (lacking key information, such as "When was the Harry Potter movie released?" lacking a country specification—56% of questions in Natural Questions contain such ambiguities).
    - Design Motivation: This type of uncertainty is neither traditional aleatoric (it can be resolved through follow-up questions) nor traditional epistemic (it is not caused by a lack of model training data). Instead, it is caused by incomplete user inputs at inference time—making it irreducible even with infinite training data and a perfect model.

4. **Interactive Learning**:

    - Function: Reduces underspecification uncertainty via follow-up questions.
    - Mechanism: LLM agents can proactively ask questions to obtain missing information, resembling active learning but with two key differences: (a) the goal is to resolve the current query rather than improve a global model; (b) the information source is the user rather than an unlabeled database, involving human-computer interaction research. A balance must be found between "asking too much and annoying the user" and "not asking and giving vague answers."
    - Current Gap: Even with GPT-3.5-Turbo-16k, the accuracy in detecting ambiguous questions is only 57% (where 50% is random), and human evaluators found only 53% of follow-up questions helpful.

5. **Output Uncertainty**:

    - Function: Goes beyond scalar probabilities by using rich text to express uncertainty.
    - Mechanism: LLMs should not merely output "confidence 0.7", but should list potential answers, explain the reasons for uncertainty, and state what information would resolve it. This is analogous to extending conformal prediction from a "prediction set" to a "naturally described space of possibilities." Linguistic devices ("most likely", "perhaps") and even acoustic features (hesitations in tone) can convey uncertainty.
    - Design Motivation: Faced with numerical probabilities, users tend to exhibit blind trust in incorrect high-confidence outputs ("blind trust" behavior), whereas verbalized explanations provide richer criteria for decision-making.

### Loss & Training

As a position paper, this work does not introduce specific training methods. Equation (1) presents the information-theoretic decomposition $\mathbb{H}(y) = \mathbb{E}_\theta[\mathbb{H}(y|\theta)] + \mathbb{I}(y;\theta)$ as a formal framework for the literature review, but the paper questions the practical utility of this decomposition (Mucsányi et al. 2024 found rank correlations between the two components as high as 0.8–0.999).

## Key Experimental Results

### Quantitative Evidence from Literature Review

| Finding | Source | Data |
|------|------|------|
| Highly correlated aleatoric/epistemic estimates | Mucsányi et al. 2024 | Deep ensembles on ImageNet-1k: rank correlation 0.8–0.999 |
| Aleatoric estimators can be used for OOD detection (traditionally considered an epistemic task) | Mucsányi et al. 2024 | Performance comparable to epistemic estimators |
| LLMs are highly deficient in detecting ambiguous queries | Zhang et al. 2024c | GPT-3.5-Turbo-16k accuracy is only 57% (random 50%) |
| Poor follow-up question quality | Zhang et al. 2024c | Human evaluators deem only 53% of follow-ups helpful for disambiguation |
| Usage of "aleatoric" and "epistemic" on arXiv | Authors' calculation | Approx. 1 preprint per day containing these terms in 2024 |

### Summary of Counter-arguments

| Counter-argument | Authors' Response |
|---------|---------|
| Aleatoric/epistemic still valuable | Agree—still useful in training and active learning, but must be clearly defined |
| Interactive learning = standard next-token prediction | Partially agree—viable in standardized interactions, but still requires verification on whether follow-ups reflect true internal knowledge |
| Uncertainty must be numerical | Numerical values are indeed needed when LLMs communicate with automated systems, but verbalized expressions are better for human-computer interaction |

### Key Findings

- The traditional aleatoric/epistemic dichotomy of UQ has at least 6 conflicting definitions in the community (Table 1), yielding contradictory conclusions even in the simplest Bernoulli case.
- In multi-turn interactions with LLM agents, the "reducible/irreducible" nature of uncertainty dynamically shifts—consistent with Der Kiureghian & Ditlevsen (2009)'s conclusion that such labels are ultimately subjective modeling choices.
- Existing LLMs have extremely weak introspective capabilities regarding uncertainty, highlighting that this is a critical direction in urgent need of research.

## Highlights & Insights

- Uses an elegant, minimalist Bernoulli example to expose conflicts in both epistemic and aleatoric definitions simultaneously—this argument is concise, powerful, and difficult to refute.
- The concept of "reducible irreducibility" accurately captures the impact of model class selection on uncertainty categorization—targeting a vulnerability frequently overlooked in practice.
- The three proposed directions (underspecification $\to$ interactive $\to$ output) form a comprehensive pipeline for handling inference-time uncertainty, rather than offering fragmented recommendations.

## Limitations & Future Work

- As a position paper, it does not provide algorithmic implementations or experimental validation—the three research directions remain conceptual proposals.
- The paper argues primarily from theoretical and philosophical perspectives, without delving deeply into the technical implementation details of interactive learning or output uncertainty.
- The discussion of counter-arguments (Sec 4) is relatively brief, and responses to some opposing points lack depth.
- It does not address the propagation of uncertainty through chain-of-thought reasoning.
- Subsequent works such as SelfReflect (Kirchhof et al., 2025) have emerged since publication, meaning the paper itself offers limited advancements in concrete pipelines.

## Related Work & Insights

- **vs Baan et al. (2023)**: They also argued that the aleatoric/epistemic dichotomy is insufficient for NLP. This paper generalizes their observations to the broader context of LLM agent interactions.
- **vs Der Kiureghian & Ditlevsen (2009)**: A classic discussion from engineering on the subjectivity of aleatoric/epistemic uncertainty; this work brings these insights into the machine learning community.
- **vs Mucsányi et al. (2024, NeurIPS)**: Provides crucial empirical evidence showing a high correlation between the two components of traditional decomposition methods—this paper leverages this as a core argument.
- **Insight**: For developing LLM agents, one should not chase the separation of aleatoric and epistemic uncertainty, but rather focus on: "What can I do about this uncertainty right now?"—ask follow-up questions if possible, and explain clearly if not.

## Rating

| Dimension | Score | Reason |
|------|------|------|
| Novelty | ⭐⭐⭐⭐ | The proposed three directions and unified framework for LLM agent UQ are highly insightful. |
| Technical Depth | ⭐⭐⭐ | A comprehensive review but lacks formalized new theoretical contributions. |
| Experimental Thoroughness | ⭐⭐ | Position papers do not require experiments, but the cited evidence relies heavily on prior work. |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear argumentative logic, concise examples, and a fair discussion of both sides. |
| Value | ⭐⭐⭐⭐ | Provides a valuable roadmap for LLM agent UQ research. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dropping Just a Handful of Preferences Can Change Top Large Language Model Rankings](../../ICLR2026/dialogue/dropping_just_a_handful_of_preferences_can_change_top_large_language_model_ranki.md)
- [\[ACL 2025\] UniConv: Unifying Retrieval and Response Generation for Large Language Models in Conversations](../../ACL2025/dialogue/uniconv_retrieval_response_gen.md)
- [\[ACL 2025\] Sparse Rewards Can Self-Train Dialogue Agents](../../ACL2025/dialogue/sparse_rewards_can_self-train_dialogue_agents.md)
- [\[NeurIPS 2025\] Less is More: Local Intrinsic Dimensions of Contextual Language Models](../../NeurIPS2025/dialogue/less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)
- [\[ICLR 2026\] Non-Collaborative User Simulators for Tool Agents](../../ICLR2026/dialogue/non-collaborative_user_simulators_for_tool_agents.md)

</div>

<!-- RELATED:END -->
