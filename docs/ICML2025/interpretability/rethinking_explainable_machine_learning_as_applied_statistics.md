---
title: >-
  [Paper Note] Rethinking Explainable Machine Learning as Applied Statistics
description: >-
  [ICML2025][Interpretability][Explainable ML] This position paper proposes that explainable machine learning (XAI) should be viewed as "applied statistics of high-dimensional functions." Explanation algorithms are fundamentally statistical functionals of functions (functionals), and research should focus on their **interpretation**—similar to traditional statistics (such as p-values or confidence intervals)—rather than merely studying their mathematical properties. The most si…
tags:
  - "ICML2025"
  - "Interpretability"
  - "Explainable ML"
  - "Applied Statistics"
  - "Post-hoc Explanations"
  - "SHAP"
  - "Interpretation"
date: 2026-05-08
content_hash: c797f2fd80a4f33d
---

# Rethinking Explainable Machine Learning as Applied Statistics

**Conference**: ICML2025  
**arXiv**: [2402.02870](https://arxiv.org/abs/2402.02870)  
**Code**: None (Position Paper, no code implementation)  
**Area**: Explainability  
**Keywords**: Explainable ML, Applied Statistics, Post-hoc Explanations, SHAP, Interpretation

## TL;DR

This position paper proposes that explainable machine learning (XAI) should be viewed as "applied statistics of high-dimensional functions." Explanation algorithms are fundamentally statistical functionals of functions (functionals), and research should focus on their **interpretation**—similar to traditional statistics (such as p-values or confidence intervals)—rather than merely studying their mathematical properties. The most significant deficiency in the current literature is the neglect of the core issue: "What intuitive question does the output of an explanation algorithm actually answer?"

## Background & Motivation

**Background**: Explainable Artificial Intelligence (XAI) has grown rapidly over the past decade, yielding numerous post-hoc explanation algorithms (e.g., SHAP, LIME, Grad-CAM, LRP) alongside mathematical analyses, computational optimization, and user studies. However, the field remains in a "pre-paradigmatic" stage, lacking consensus on fundamental concepts such as "what is an explanation" and "what constitutes explainability."

**Limitations of Prior Work**:
   - Numerous papers focus solely on the **mathematical properties** of explanation algorithms (e.g., the axioms of Shapley values, sensitivity of LIME, computational complexity) while rarely discussing what real-world questions these explanations answer.
   - Even relatively simple estimators in classical statistics (e.g., p-values) are frequently misunderstood and misused in practice; more complex explanation outputs in XAI are prone to even more severe misinterpretations.
   - The relationships between explanation algorithms and other model evaluation metrics, such as fairness and robustness, remain poorly understood.

**Key Challenge**: The XAI community devotes intensive effort to the form of explanation algorithms—namely, mathematical definitions and computational methods—while neglecting the **interpretation of explanations**, i.e., how these mathematical objects map to human intuitive questions about the real world.

**Goal**
   - Provide a unified conceptual framework for XAI, positioning it as "applied statistics of high-dimensional functions."
   - Clearly distinguish between the two levels of description: "mathematical form" and "interpretative meaning" of explanation algorithms.
   - Propose concrete recommendations to improve research practices.

**Key Insight**: The authors observe that classical applied statistics summarizes high-dimensional information of probability distributions and datasets, whereas XAI summarizes the high-dimensional information of learned functions. The two are structurally analogous. Applied statistics is a mature field whose lessons can be leveraged to refine the research paradigm of XAI.

**Core Idea**: Explanation algorithms are statistics of functions; foundational issues in XAI can be clarified and resolved through a direct analogy with applied statistics.

## Method

### Overall Architecture

This paper does not propose a new algorithm but rather constructs a unified conceptual framework. The overall line of argumentation is as follows:

**Input**: Conflicting concepts and research practices in the current XAI field $\rightarrow$ **Core Mapping**: Formalizing explanation algorithms as "statistics of functions" $\rightarrow$ **Analogy Bridge**: Establishing a systematic analogy between XAI and applied statistics $\rightarrow$ **Output**: In-depth discussion of interpretation + recommendations for improving research practices.

The argument is structured into three main steps:
1. Formally define "statistics of functions" to describe post-hoc explanation algorithms in a unified manner (Section 2).
2. Establish a systematic analogy between XAI and applied statistics (Section 3).
3. Thoroughly discuss the "interpretation of statistics" and its implications for XAI (Section 4+).

### Key Designs

#### 1. **Three-Tier Definition Hierarchy of Statistics**

The authors construct three parallel formal definitions to demonstrate the structural isomorphism between XAI and classical statistics:

- **Statistics of Probability Distributions** (Definition 2.1): $F: \mathcal{P}(\mathbb{R}^d) \to \mathbb{R}^k$, which maps a high-dimensional probability distribution to a low-dimensional vector. Examples include the mean, median, and moments of distribution.
- **Statistics of Datasets** (Definition 2.2): $F: \mathbb{R}^{d \times m} \to \mathbb{R}^k$, which maps $m$ data points of dimension $d$ to a low-dimensional vector. Examples include p-values, F-statistics, and visualization plots.
- **Statistics of Functions** (Definition 2.3): $F: \mathcal{F}(\mathbb{R}^d) \times \mathcal{P}(\mathbb{R}^d) \times \mathbb{R}^d \to \mathbb{R}^k$, which maps a high-dimensional function $f$ (along with a distribution $\mathcal{D}$ and a specific data point $x$) to a low-dimensional vector. This provides the formalization of explanation algorithms.

Design Motivation: Placing the three definitions alongside each other in the same mathematical framework clearly demonstrates that XAI is not an entirely new field but a natural extension of statistics—where the object being summarized is simply shifted from a distribution/dataset to a function.

#### 2. **Unified Coverage of Mainstream Explanation Algorithms**

The authors show that Definition 2.3 can dynamically represent almost all post-hoc explanation algorithms, including SHAP, LIME, Grad-CAM, LRP, counterfactual explanations, and perturbation-based explanations. Furthermore, metrics like generalization error and fairness indicators are also "statistics of functions." This implies that explanations, fairness measures, and robustness metrics are fundamentally similar mathematical objects and should not be artificially segmented.

#### 3. **Two Key Insights**

- **Insight 1**: Post-hoc explanation algorithms are statistics of functions **regardless of** whether they are useful to end-users, satisfy a specific goal, or are intuitively understandable. Being a statistic is an objective mathematical fact, whereas utility requires separate justification. This resolves some of the debate in XAI over "what counts as an explanation."

- **Insight 2**: A significant portion of the XAI literature studies only the mathematical or computational properties of statistics (e.g., sensitivity of SHAP, efficient approximation of Shapley values) without discussing what real-world questions these statistics answer. This is akin to studying the asymptotic properties of an estimator in statistics without discussing its relevance to scientific inquiries.

#### 4. **Philosophical Analysis of Interpretation**

The authors make a sharp distinction between two levels:
- **(a) Mathematical Form and Properties**: The formal definition, axiomatic properties, and calculation methods of the explanation algorithm.
- **(b) Interpretative Meaning**: How the statistic maps to human intuitive questions.

Human intuitive questions about models include: How does the model make predictions? Is it trustworthy? Which features are important? Does it rely on spurious correlations? How does genetic variation affect disease risk? What is the impact of carbon pricing on economic growth? Borrowing a framework from the philosophy of science on "intuitive concepts" (Justus, 2012), the authors argue that mapping an intuitive question to a mathematical formalization is non-trivial and requires rigorous justification.

### Key Recommendations

Specific recommendations proposed by the paper to improve research practice include:
- Explanation algorithms should be designed to **answer specific questions** (e.g., Schut et al., 2023; Arditi et al., 2024).
- Acknowledge that the use of explanation tools requires **a certain level of expertise**; one should not assume that end-users can interpret them intuitively.
- Re-evaluate the role of **benchmark datasets** in XAI evaluation (Section 6).
- Recognize the connection between explanations and metrics of **fairness and robustness**, as all are statistics of functions.

## Key Experimental Results

### Conceptual Comparison

Because this is a position paper, it contains no classical experiments. The core comparative analysis of the paper is summarized below:

| Dimension | Applied Statistics | Explainable Machine Learning |
|------|-----------|--------------|
| Subject Summarized | Probability distributions / Datasets | Learned functions $f$ |
| Form of Statistic | Mean, p-value, confidence interval, etc. | SHAP, LIME, Grad-CAM, etc. |
| Theoretical Foundation | Highly mature, built over a century | Immature, in a pre-paradigmatic stage |
| Interpretation / Misuse | p-values are widely misused (ASA statement) | Explanation outputs are frequently misunderstood |
| Research Focus | Mathematical properties + Interpretation + Applications | Biased towards mathematical properties, with insufficient discussion on interpretation |
| Recognition of Expertise | Training is recognized as necessary | Often assumes end-users can directly understand |
| Benchmark Evaluation | Mature evaluation frameworks | The utility of benchmark datasets is questionable |

### Categorization of XAI Literature Research Types

| Research Type | Example | Discusses Interpretative Meaning? | Estimated Proportion |
|----------|------|----------------|----------|
| Analysis of Mathematical Properties | Sensitivity of SHAP, behavior of LIME on specific function classes | Typically No | High |
| Computational Optimization | Efficient approximation algorithms for Shapley values | Typically No | High |
| Axiomatic Characterization | Uniqueness of Shapley values, efficiency axiom | Partially | Medium |
| User Studies | Usability testing for end-users | Implicitly | Low |
| Problem-driven Design | Designing explanations tailored to specific inquiries (Schut et al., 2023) | Directly | Extremely Low |

### Key Findings

- **Neglect of the Core Question**: The most critical defect in the current literature is that a vast majority of papers fail to discuss the "interpretative meaning" of their explanation algorithms—i.e., what specific question the algorithm's output answers, and why.
- **The Lesson of p-values**: Even the seemingly simple p-value is widely misinterpreted in practice (as outlined in the 2016 ASA statement), indicating that even mathematically precise statistics face interpretation hurdles. More complex statistics in XAI present a significantly higher risk of misinterpretation.
- **Value of a Unified Perspective**: Framing explanations, fairness, and robustness uniformly as "statistics of functions" dismantles artificial barriers between these subfields, fostering cross-disciplinary synergy.
- **Necessity of Specialized Training**: Utilizing XAI tools requires proper training, just as statistical tools demand statistical literacy.

## Highlights & Insights

- **Power of Analogy**: Re-centering XAI as a branch of applied statistics is a compelling analogy. This holds not only in mathematical form (stat of functions vs. stat of distributions) but also offers practical lessons—centuries of warnings in applied statistics (e.g., p-hacking, misinterpretations of Simpson's paradox) can be directly imported as cautionary tales for XAI.

- **Separation of Form and Interpretation**: Distinctly separating (a) mathematical form/properties from (b) interpretative meaning constitutes the primary contribution of this work. Though deceptively simple, this distinction targets the root of various XAI debates—many of which conflate the mathematical properties of the statistic itself with whether the statistic actually addresses a meaningful question.

- **Transferable Meta-Research Perspective**: This methodology of evaluating an entire research field through the lens of philosophy of science can be mapped to other burgeoning ML subfields, such as LLM alignment (What is alignment? Is the formalization accurate?) and AI Safety (Do safety metrics truly measure the safety attributes of interest?).

- **Introduction of Intuitive Concepts**: Utilizing Justus' (2012) philosophical framework to model the mapping from intuitive questions to formalizations provides a more rigorous philosophical foundation for the conceptual bases of XAI.

## Limitations & Future Work

- **Lack of Concrete Methodology**: Although the paper highlights the necessity of discussing "interpretative meaning," it falls short of providing a systematic methodological guide. It remains unclear how exactly researchers should argue for the correspondence between a specific explanation algorithm and an intuitive user query, yielding no actionable review checklists or design pipelines.

- **Inherent Limitations of Position Papers**: As a position paper, it lacks experimental validation. The core claim (recognizing the analogy $\rightarrow$ improving research practices) is not yet supported by empirical evidence regarding its actual impact on the quality of XAI research.

- **Insufficient Discussion of Mechanistic Interpretability**: The authors acknowledge that their focus is primarily on post-hoc explanation algorithms, only briefly discussing mechanistic interpretability in Section 7. The latter is evolving rapidly and likely possesses distinct epistemological characteristics.

- **Neglect of Computational Constraints**: In practical scenarios, the selection of explanation algorithms is tightly constrained by computational costs and usability (e.g., the overhead of computing exact SHAP values for LLMs). The theoretical discussion remains somewhat idealized, forgoing deep consideration of engineering and deployment constraints.

- **Potential Oversimplification of the Unified Framework**: While classifying all explanation algorithms under a singular formalism is elegant, it risks obscuring fundamental differences between distinct paradigms of explanations (e.g., global vs. local explanations, model-agnostic vs. model-specific methods).

## Related Work & Insights

- **vs SHAP (Lundberg & Lee, 2017)**: SHAP is the canonical example of "defining mathematical form first, then searching for an interpretative meaning." The paper critiques such works for focusing excessively on axiomatic properties (e.g., efficiency, symmetry) rather than addressing the causal or statistical questions shape values actually answer.

- **vs LIME (Ribeiro et al., 2016)**: LIME is similarly contextualized within the "statistics of functions" framework. Through this lens, the primary issue of LIME is not approximation error, but rather the lack of clarity regarding the "interpretative meaning" of its linear approximation coefficients—namely, which question is being answered when users intercept feature importance scores.

- **vs Molnar et al. (2020)**: While Molnar et al. demonstrated that explanation outputs are frequently misinterpreted, this paper advances this line of thought by explaining *why* such misinterpretations occur from a statistical viewpoint—attributing it to researchers' failure to formally discuss interpretative meaning.

- **vs Miller (2019) from Social Science**: Previous XAI literature often drew from social sciences/psychology, framing explanations as answers to "why-questions." This work offers a broader yet more mathematically concrete perspective: an explanation is a statistical answer to *any* intuitive query about a function.

- **Relevance to AI Safety/Alignment**: The methodology presented here (scrutinizing whether the core concepts of a field are adequately formalized and interpreted) is directly transferable to AI alignment. For instance, when designing alignment metrics, is it clear which intuitive safety questions those metrics are designed to answer?

## Rating

- Novelty: ⭐⭐⭐⭐ Re-centering XAI as applied statistics is not radically jointless (e.g., Fisher et al., 2019 made similar observations), but the systematic formulation and explicit "form vs. interpretation" split holds highly original value.
- Experimental Thoroughness: ⭐⭐⭐ Unnecessary for a position paper, though it would have been more compelling had it included case studies (e.g., executing a demonstration of the "interpretation of explanation" debate on a specific algorithm).
- Writing Quality: ⭐⭐⭐⭐⭐ The arguments are coherent, highly structured, and the mathematical definitions are elegant, while the philosophical discussions are profound yet accessible.
- Value: ⭐⭐⭐⭐ Highly influential as directional guidance for the XAI community; the call to discuss "what exact question does your explanation algorithm answer" directly addresses a long-standing core oversight.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Ideas Should be the Center of Machine Learning Research](../../ICML2026/interpretability/position_ideas_should_be_the_center_of_machine_learning_research.md)
- [\[ICLR 2026\] When Machine Learning Gets Personal: Evaluating Prediction and Explanation](../../ICLR2026/interpretability/when_machine_learning_gets_personal_evaluating_prediction_and_explanation.md)
- [\[ICLR 2026\] Explainable Mixture Models through Differentiable Rule Learning](../../ICLR2026/interpretability/explainable_mixture_models_through_differentiable_rule_learning.md)
- [\[AAAI 2026\] Explainable Melanoma Diagnosis with Contrastive Learning and LLM-based Report Generation](../../AAAI2026/interpretability/explainable_melanoma_diagnosis_with_contrastive_learning_and_llm-based_report_ge.md)
- [\[NeurIPS 2025\] SpEx: A Spectral Approach to Explainable Clustering](../../NeurIPS2025/interpretability/spex_a_spectral_approach_to_explainable_clustering.md)

</div>

<!-- RELATED:END -->
