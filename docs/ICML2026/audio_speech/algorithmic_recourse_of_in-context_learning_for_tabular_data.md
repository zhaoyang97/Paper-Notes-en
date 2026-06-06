---
title: >-
  [Paper Note] Algorithmic Recourse of In-Context Learning for Tabular Data
description: >-
  [ICML 2026][Audio & Speech][Tabular ICL] This paper presents the first systematic study of algorithmic recourse in the context of in-context learning (ICL) for tabular data. It proves that recourse induced by the dynamic…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Tabular ICL"
  - "Algorithmic Recourse"
  - "Black-box Optimization"
  - "Counterfactual Explanations"
  - "Actionability"
date: 2026-05-08
content_hash: 477dd0c270f9270b
---

# Algorithmic Recourse of In-Context Learning for Tabular Data

**Conference**: ICML 2026  
**arXiv**: [2605.31272](https://arxiv.org/abs/2605.31272)  
**Code**: No public code  
**Area**: Explainability / Tabular Data / Algorithmic Recourse  
**Keywords**: Tabular ICL, Algorithmic Recourse, Black-box Optimization, Counterfactual Explanations, Actionability  

## TL;DR
This paper presents the first systematic study of algorithmic recourse in the context of in-context learning (ICL) for tabular data. It proves that recourse induced by the dynamic decision rules of ICL remains definable and bounded, and proposes ASR-ICL, which utilizes adaptive subspace zero-order optimization to generate low-cost, sparse, and actionable counterfactual modifications on black-box ICL models.

## Background & Motivation
**Background**: Algorithmic recourse is typically applied in high-stakes tabular decision systems such as loans, justice, and healthcare. The goal is not just to explain why a model rejected a sample, but to inform users how to change actionable features to obtain a favorable prediction. Traditional methods assume a fixed classifier post-training, allowing for gradient optimization, integer programming, or graph search around a stable decision boundary.

**Limitations of Prior Work**: Tabular prediction is being reshaped by TabPFN, TabICL, and in-context learning via general LLMs. ICL does not explicitly train a fixed model; instead, it induces a temporary predictor during inference based on context examples. The actual decision rules may vary for the same user sample when presented with different demonstration sets. Consequently, traditional recourse assumptions—such as "fixed models," "stable boundaries," and "accessible gradients"—no longer hold.

**Key Challenge**: High-stakes tabular tasks require stable, actionable, and low-cost modification suggestions, yet ICL decision rules are context-conditional black-box functions. The paper does not merely apply existing counterfactual methods to LLMs; instead, it first determines whether such dynamic predictors still possess definable recourse and then designs practical algorithms that operate within finite query budgets.

**Goal**: The authors decompose the problem into two levels: theoretically, proving the feasibility, cost upper bounds, and convergence behavior as context scale grows within analyzable linear self-attention ICL settings; and practically, generating effective recourse for tabular tasks characterized by black-box queries, mixed continuous/discrete features, and actionability constraints.

**Key Insight**: Theoretical analysis reveals that as the number of test-time context examples increases, the recourse induced by ICL approaches that of classic linear models, though the cost upper bound remains influenced by feature dimensionality. This observation directly inspires the method design: since pre-trained context lengths cannot be changed, the search should be compressed into a few key features during inference.

**Core Idea**: Replace full-dimensional search with "adaptive selection of small subspaces + zero-order black-box optimization," ensuring that recourse for ICL tabular models satisfies validity, low query volumes, and sparse interpretability.

## Method
The methodology is divided into theoretical and algorithmic components. The theoretical part defines the ICL predictor as a function $f_{\mathcal{C}}$ dependent on a context set $\mathcal{C}$, analyzing the recourse solution for linear self-attention in regression tasks. The algorithmic part implements this insight as Adaptive Subspace Recourse for In-Context Learning (ASR-ICL).

### Overall Architecture
The input consists of a tabular sample $x$ that received an unfavorable prediction, a set of ICL demonstrations, a target favorable class $y^+$, feature actionability constraints $\Omega(x)$, and a black-box ICL model providing only predicted labels. ASR-ICL maintains an importance distribution over mutable features, samples a small subspace each round, and utilizes a zero-order optimizer to find a candidate $x'$ within these features. Candidates are projected back to the feasible set and queried against the model. Subspaces yielding lower target values reward their constituent features by increasing their importance scores. The best feasible recourse is finally outputted.

Theoretical analysis justifies this approach. In linear ICL settings, the optimal perturbation can be expressed in terms of the empirical covariance $S$ of the context and the effective pre-training covariance $\Gamma$. As training context length $N$ and test demonstration count $M$ increase, $S$ approximates the true covariance, and ICL recourse converges to that of a classic linear model. Meanwhile, the perturbation upper bound contains a $\sqrt{\ln(2d/\delta)/M}$ term, implying that high-dimensional tabular spaces require more context for stability, thus making active dimensionality reduction reasonable.

### Key Designs
1. **Theoretical Definition of Contextual Recourse**:
    - **Function**: Redefines traditional fixed-model recourse as a context-conditional problem for $f_{\mathcal{C}}$, clarifying "whether a user can obtain the target prediction under the same context."
    - **Mechanism**: The authors define the validity of a candidate $x'$ on the ICL predictor and analyze the optimal perturbation $\delta^*_{\mathrm{ICL}}$ using linear self-attention. This perturbation is determined by the current sample score, context empirical covariance, and a regularization term; thus, it is not a simple replication of traditional fixed-weight models.
    - **Design Motivation**: Designs are only empirically heuristic if recourse is not first proven meaningful under ICL. The theory provides feasibility conditions and high-probability cost upper bounds, explaining how context scale affects recourse stability.

2. **Adaptive Subspace Selection**:
    - **Function**: Dynamically identifies a small number of mutable features in high-dimensional tabular spaces likely to produce effective recourse.
    - **Mechanism**: Each feature is assigned an importance score $I_j$, with sampling probabilities $p_{\mathrm{sel}}(j) \propto \exp(I_j)$. After sampling a subspace of size $k$ and completing a local search, $I_j$ is updated as $I_j \leftarrow (1-\alpha)I_j + \alpha r_t / |S_t|$, using the negative target value as a reward.
    - **Design Motivation**: Tabular recourse typically favors changing only a few actionable attributes. Compared to full-dimensional zero-order optimization, adaptive subspaces reduce query counts and produce modifications that are easier for humans to understand and implement.

3. **Black-box Zero-order Optimization and Feasibility Projection**:
    - **Function**: Optimizes mixed continuous/discrete tabular features without access to gradients, logits, or internal model states.
    - **Mechanism**: The objective is formulated as $L_{\mathrm{pr}}(x,x')=(1-\mathbb{I}[\hat{y}(x')=y^+])+\lambda c(x,x')$, purely dependent on label queries and recourse costs. Inner optimization uses RACOS for continuous ranges and discrete grids. Candidates are passed through a projection $\Pi_{\Omega}$ to fix immutable features and adhere to monotonic constraints before evaluation.
    - **Design Motivation**: Real-world ICL services often expose only prediction results, and tabular data contains immutable attributes (e.g., age, gender). This design aligns the algorithm with deployment constraints rather than idealized white-box access.

### Loss & Training
ASR-ICL does not train the prediction model but performs a black-box search during inference. The reward function consists of two parts: a reward for the candidate $x'$ being predicted as the target class by the ICL model, and a penalty $c(x,x')$ for normalized changes in continuous features and modifications to discrete features. The default subspace size is $\min(5,\lceil\sqrt{d}\rceil)$, with a smoothing coefficient $\alpha=0.5$, a total query budget of 150, and RACOS as the inner zero-order optimizer. Recourse is generated towards the favorable label in binary tasks and the designated optimal class in multi-class tasks.

## Key Experimental Results

### Main Results
The paper compares ASR-ICL with classic trained-model recourse methods across three binary tabular tasks: Australian Credit, COMPAS, and Diabetes. ICL results use a 32-shot context; lower costs are preferred.

| Model / Method | Australian Credit Validity / Cost | COMPAS Validity / Cost | Diabetes Validity / Cost | Conclusion |
|----------------|----------------------------------|-----------------------|-------------------------|------------|
| MLP + DiCE | 1.00 / 8.96 | 1.00 / 7.51 | 1.00 / 5.02 | High validity, but significantly high modification costs |
| Linear + AR | 0.82 / 1.76 | 0.94 / 3.44 | 0.72 / 1.52 | Low cost, but often fails to find valid recourse |
| Linear + FACE | 1.00 / 6.18 | 1.00 / 6.51 | 1.00 / 5.13 | Stable validity, but costs remain high |
| TabPFN-2.5 + ASR-ICL | 1.00 / 3.83 | 1.00 / 2.76 | 1.00 / 2.78 | Maintains full validity with lower costs under black-box ICL |
| TabICL + ASR-ICL | 1.00 / 4.47 | 0.81 / 3.44 | 1.00 / 2.80 | Dedicated tabular ICL is stable overall; COMPAS is harder |
| Qwen3-8B + ASR-ICL | 0.87 / 2.94 | 0.99 / 2.55 | 0.84 / 1.50 | General LLMs can also generate low-cost recourse |
| LLaMA-3.2-3B + ASR-ICL | 1.00 / 2.99 | 1.00 / 2.43 | 0.98 / 1.67 | Costs around 2 to 3 across multiple open-source LLMs |
| GPT-4o + ASR-ICL | 0.99 / 4.75 | 0.78 / 3.62 | 0.71 / 4.31 | Lower validity for closed-source models suggests boundary noise affects recourse |

### Ablation Study
The paper uses Full ZO as a non-adaptive full-space zero-order optimization baseline to contrast ICL query efficiency and cost. Representative results from Australian Credit and Corporate Rating are shown.

| Configuration | Australian Credit Validity / Cost / Queries | Corporate Rating Validity / Cost / Queries | Description |
|---------------|------------------------------------------|------------------------------------------|-------------|
| Full ZO + TabPFN-2.5 | 1.00 / 12.88 / 144.28 | 1.00 / 15.44 / 149.45 | Full-dimensional search nears budget limit with high costs |
| ASR-ICL + TabPFN-2.5 | 1.00 / 3.83 / 27.01 | 0.98 / 4.79 / 111.71 | Adaptive subspace significantly reduces binary cost and queries |
| Full ZO + Qwen3-4B | 1.00 / 12.03 / 150.00 | 0.95 / 17.42 / 150.00 | Full-dimensional search almost always exhausts budget on LLMs |
| ASR-ICL + Qwen3-4B | 0.79 / 3.01 / 56.01 | 0.94 / 3.54 / 117.32 | Cost drops significantly; validity limited by model prediction quality |
| Full ZO + GPT-4o | 0.97 / 11.82 / 150.00 | 0.77 / 14.18 / 150.00 | High query costs do not guarantee better validity |
| ASR-ICL + GPT-4o | 0.99 / 4.75 / 49.04 | 0.72 / 4.87 / 40.29 | Achieves similar validity and lower cost with far fewer queries |

### Key Findings
- As context examples increase, recourse validity becomes more stable and average costs decrease, consistent with the theoretical trend of ICL recourse converging to classic linear recourse.
- In multi-class tasks, TabPFN and TabICL achieve near-full validity on Corporate Rating and Student Performance; general LLMs are significantly more unstable on Student Performance, indicating sensitivity to complex class boundaries.
- Adaptive subspace selection is the primary contribution: it not only reduces queries but also concentrates search on fewer features, making recourse sparser. In Diabetes cases, ASR-ICL modifies only Glucose and BMI, whereas DiCE impacts immutable attributes.

## Highlights & Insights
- Instead of directly porting counterfactual explanations to LLMs, the paper addresses the foundational question: "Do dynamic models in ICL still possess recourse?" This clean setup creates a genuine link between theory and algorithm.
- The design of ASR-ICL is pragmatic: it acknowledges that black-box ICL can only be queried for labels and exploits the natural sparsity of tabular recourse to transform full-dimensional optimization into adaptive small-subspace search.
- The dimensionality term in the theoretical upper bound provides a useful heuristic: when ICL models are fixed and context budgets are limited, reducing the effective search dimension is more critical than blindly increasing zero-order iterations. This approach is transferable to black-box LLM tool-calling, medical tabular decisions, and risk control strategy explanations.

## Limitations & Future Work
- The theoretical part is based on linear regression and single-layer linear self-attention, which remains distant from the mechanisms of actual TabPFN, TabICL, or general LLMs; it serves as an idealized model to explain trends rather than a complete characterization.
- Experiments primarily evaluate the ability to flip model predictions and modification costs, lacking user studies on human executability, long-term stability, or causal validity. For high-stakes scenarios, satisfying feature constraints is not synonymous with real-world actionability.
- ASR-ICL still relies on a significant number of black-box queries. Although fewer than Full ZO, dozens to hundreds of queries can still be expensive or trigger risk controls for paid APIs or audited systems.
- Future work could integrate adaptive subspaces with causal constraints, domain rules, and confidence estimation, ensuring recourse not only flips labels but also improves robustness across contexts and models.

## Related Work & Insights
- **vs DiCE**: DiCE generates diverse counterfactuals via optimization, suitable for fixed models with rich feedback; ASR-ICL targets context-conditional black-box ICL and relies only on label queries, making it better suited for models served without gradient access.
- **vs Actionable Recourse**: AR provides low-cost, even certifiable actions under white-box linear models, but validity is limited by model assumptions; ours extends "low-cost actions" to ICL predictors, albeit at the loss of global optimality certification.
- **vs FACE**: FACE searches along data manifolds emphasizing feasible paths; ASR-ICL emphasizes sparse subspace search and query efficiency. These could be combined in the future using manifold graphs to constrain ASR-ICL candidate regions.
- **vs TabPFN / TabICL**: While TabPFN and TabICL focus on tabular ICL prediction accuracy, this work focuses on providing actionable modifications to rejected individuals after deployment, extending from performance to accountability mechanisms.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic integration of algorithmic recourse within dynamic decision rules of tabular ICL, providing a theoretical and algorithmic closed loop.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various datasets, specialized tabular models, and general LLMs, including multi-class, query efficiency, and robustness analysis, though lacks real-world user/causal validation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and methodology; theoretical-to-algorithmic connection is natural. Some experimental tables are dense and require careful interpretation.
- Value: ⭐⭐⭐⭐⭐ Highly relevant for high-stakes tabular ICL deployment, providing a reusable framework for recourse research in black-box LLM decision systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multimodal In-Context Learning for ASR of Low-Resource Languages](../../ACL2026/audio_speech/multimodal_in-context_learning_for_asr_of_low-resource_languages.md)
- [\[ICML 2026\] A Semantically Consistent Dataset for Data-Efficient Query-Based Universal Sound Separation](a_semantically_consistent_dataset_for_data-efficient_query-based_universal_sound.md)
- [\[ICML 2026\] Multiple Choice Learning of Low-Rank Adapters for Language Modeling](multiple_choice_learning_of_low-rank_adapters_for_language_modeling.md)
- [\[ICML 2026\] Group Cognition Learning: Making Everything Better Through Governed Two-Stage Agents Collaboration](group_cognition_learning_making_everything_better_through_governed_two-stage_age.md)
- [\[ACL 2026\] DRInQ: Evaluating Conversational Implicature with Controlled Context Variation](../../ACL2026/audio_speech/drinq_evaluating_conversational_implicature_with_controlled_context_variation.md)

</div>

<!-- RELATED:END -->
