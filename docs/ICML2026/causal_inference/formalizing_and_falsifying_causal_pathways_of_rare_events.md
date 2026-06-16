---
title: >-
  [Paper Note] Formalizing and Falsifying Causal Pathways of Rare Events
description: >-
  [ICML 2026][Causal Inference][Paper Note] This paper formalizes "verbal causal explanations" of rare events into a **causal pathway**—a subgraph composed of binarized events—and defines a **pathway explanation score** to quantify the explanatory power of the "root cause + intermediate pathway" relative to the target event, establishing a falsifiable framework
tags:
  - ICML 2026
  - Causal Inference
date: 2026-05-08
content_hash: 5a1c9ce4ae9b5093
---
# Formalizing and Falsifying Causal Pathways of Rare Events

**Conference**: ICML 2026  
**arXiv**: [2605.31254](https://arxiv.org/abs/2605.31254)  
**Code**: None  
**Area**: Causal Inference / Explainable AI / Rare Event Root Cause Analysis  
**Keywords**: Causal Pathways, Rare Events, Explanation Score, Causal Abstraction, Falsifiability  

## TL;DR
This paper formalizes "verbal causal explanations" of rare events into a **causal pathway**—a subgraph composed of binarized events—and defines a **pathway explanation score** to quantify the explanatory power of the "root cause + intermediate pathway" relative to the target event, establishing a falsifiable framework for evaluating causal explanations.

## Background & Motivation

**Background**: Causal analysis of unusual or rare events (natural disasters, stock market crashes, technical system failures, abnormal gene expressions, etc.) primarily follows the **root cause analysis (RCA)** route—identifying a small set of "root cause nodes" in a Structural Causal Model (SCM) that significantly increase the counterfactual probability of the target event.

**Limitations of Prior Work**: RCA only answers "what changed" but not "how it affected the target." Real explanations typically involve (i) multiple interacting intermediate mechanisms and (ii) context variables that control propagation. Providing only a list of root causes is uninterpretable to humans and cannot be falsified by experiments in AI systems.

**Key Challenge**: Two categories of related work are incomplete:
1. **Extreme Value Statistical Causal Models** (Engelke 2025, Klüppelberg 2026): Rely on asymptotic or heavy-tailed parameter assumptions and cannot handle "not extreme but statistically rare" events (e.g., values very close to 0, severely imbalanced binary events).
2. **Path-specific Effects Analysis**: Decomposes average causal effects to find "the share of path A," but does not answer "which part of the graph provides a good explanation for this specific observation."
3. **Causal Abstraction** (Rubenstein 2017, Beckers 2020): Operates on the entire model hierarchy and lacks **event-level** local abstract concepts.

**Goal**: Establish an **event-level** formal system for causal explanations that: (a) does not rely on asymptotic extreme value assumptions; (b) applies to any value space (continuous, discrete, text embeddings); (c) is falsifiable via data or consistency tests; and (d) can be automatically derived from finer-grained SCMs via abstraction.

**Key Insight**: Redefine an "explanation" as a **subgraph + set of binarized events** rather than just a list of root causes. Specifically, given a target event $B_t=1$ and a set of root causes $\mathbf{B}_R$, one measures how closely the log-likelihood of other events on the pathway $\mathbf{B}_{K\setminus R}=\mathbf{1}$ occurring simultaneously after $do(\mathbf{B}_R=\mathbf{1})$ matches the target's log-rarity. This explicitly incorporates the requirement that "intermediate events must also look reasonable" into the scoring function.

**Core Idea**: Use the **log-likelihood ratio** $\mathcal{E}^K_{R\to t} := 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(B_t=1)}$ as a falsifiable measure of explanation quality. Apply **feature monotonicity** to unify variables from arbitrary spaces into binary events, allowing the theory to generalize smoothly from binary SCMs to continuous, discrete, or text variables.

## Method

### Overall Architecture
The framework objective is to transform verbal causal chains like "Because A, therefore B, therefore C, therefore Target" into a score falsifiable by data or probability beliefs. The "explanation" is re-organized as a pathway of binary events. A $[0,1]$ explanation score is defined on binary SCMs, which is then extended to arbitrary spaces via "feature monotonicity + binarization." Finally, "event-level causal abstraction" quantifies the information loss during coarsening. Inputs are (SCM, observed sample, target event); outputs are (pathway subgraph $\mathcal{P}$, root cause set $R$, explanation score $\mathcal{E}^K_{R\to t}$, abstraction accuracy $r$).

### Key Designs

**1. Cluster / Pathway explanation score: Making a causal chain falsifiable by scoring**

Standard RCA outputs root causes without auditing whether "intermediate events are reasonable," making it impossible to overturn "plausible-sounding" explanations. This paper adopts the cluster score $\mathcal{E}_{R\to K} = 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(\mathbf{B}=\mathbf{1})}$ from Oesterle 2025, which measures how much the root causes $R$ increase the likelihood of the entire cluster. However, the cluster score only considers if the likelihood is raised, not if the intermediates look normal.

The pathway score replaces the denominator with the target event's log-rarity $\log P(B_t=1)$: $\mathcal{E}^K_{R\to t} = 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(B_t=1)}$. When the target is rarer than the cluster, this score is stricter—any "intermediate event that is inherently strange" is penalized by the log-likelihood term in the numerator. The two satisfy an affine relationship $1-\mathcal{E}^K_{R\to t} = (1-\mathcal{E}_{R\to K}) \cdot \frac{\log P(\mathbf{B}=\mathbf{1})}{\log P(B_t=1)}$, meaning the contribution of each node on the pathway remains additive (Eq. 11), allowing for greedy root cause selection. Lemma 3.7 further provides a log-likelihood gap $\Delta_i := [\log P(B_{\mathrm{Pa}(i)}=\mathbf{1}) - \log P(B_i=1)]_+$ for each edge to control the upper bound of the score; a larger gap indicates a rarer mechanism that drags down the whole chain.

**2. Feature monotonicity + Binarization: Extending theory to arbitrary spaces**

Variables in real systems (continuous, categorical, tokens) are not naturally binary. The authors assign a feature function $\tau_j:\mathcal{X}_j\to\mathbb{R}$ to each $X_j$, mapping variables to binary events $B_j := \{\tau_j(X_j) \geq \tau_j(x_j)\}$ (i.e., "the feature of $X_j$ is at least as large as the observation"). The mechanism $P(X_j\mid \mathbf{X}_{\mathrm{Pa}(j)})$ is required to be monotonic with respect to $(\tau_j, \tau_{\mathrm{Pa}(j)})$: larger parent features stochastically result in larger child features.

Under this condition, Lemma 4.2 provides a critical tail probability guarantee: "For $x_j$ sampled from $P(X_j\mid \mathbf{x}_{\mathrm{Pa}(j)})$, the probability that the conditional likelihood $\leq \alpha$ is at most $\alpha$." Theorem 4.3 generalizes this to the entire DAG: for any $\mathbf{x}_R$, the probability that the negative log-likelihood $L \geq c$ after generating other variables from $do(\mathbf{x}_R)$ is bounded by $\sum_{i=0}^{n-|R|-1}\frac{c^i}{i!}e^{-c}$ (a Poisson tail with degree-of-freedom correction). This serves as the **p-value** when the explanation score deviates from 1. Choosing "$X\geq x$ explains $Y\geq y$" over "$X=x$ explains $Y=y$" is more robust and aligns with human language. Even if the true distribution is not strictly monotonic, this p-value acts as a diagnostic threshold.

**3. Pathway abstraction + natural micro-realization: Quantifying coarsening loss**

To automatically derive a coarse-grained pathway explanation from a fine-grained SCM $(\mathcal{G}, P_\mathbf{X})$, one must measure the information lost via binarization. The abstraction accuracy $r$ is defined as $r := 1 - \max_{S, \mathbf{b}_S} \frac{D_{KL}[P_\mathbf{X}(\mathbf{B}\mid do(\mathbf{B}_S=\mathbf{b}_S))\,\|\,P_\mathbf{B}(\mathbf{B}\mid do(\mathbf{B}_S=\mathbf{b}_S))]}{-\log P_\mathbf{X}(B_t=1)}$, normalizing the KL divergence between interventional distributions by the target rarity. Since $do(\mathbf{B}_j=b_j)$ is ill-defined in the original model (multiple $X_j$ map to one $B_j$), **natural micro-realization** defines it as: "sample underlying variables $X_i$ from $P_\mathbf{X}(X_i\mid B_i=b_i)$ and then perform $do$ on the original model."

Crucially, the explanation score can be rewritten in the same KL form, placing it on the same scale as accuracy $r$. Thus, design choices like "whether to include a context variable in the pathway" become a calculable trade-off between accuracy and explanation score.

### Loss & Training
This is a theoretical framework and does not involve training. Probabilities are either estimated from observed samples (data consistency test) or estimated from LLMs/experts via QA (internal consistency test). Selecting the root cause set $R$ is performed via a greedy algorithm: $R\gets R\cup\{\arg\max_i \mathcal{E}^K_{\{i\}\cup R\to t}\}$, with optimality guaranteed by the additivity in Eq. (11).

## Key Experimental Results

### Main Results: Explanation Scores vs. Event Rarity

| Case | Setting | Explanation Score | Meaning |
|------|---------|-------------------|---------|
| Gaussian Causal Pair (Ex 4.6) | $Y=\rho X+N$, $\rho=0.5$, $x\geq 3$, $y\approx \rho x$ | $\geq 0.8$ | "$X\geq x$ explains $Y\geq y$ at least 80%," matching acceptable causal statements. |
| Triadic Chain (Ex 3.6) | $P(b_1^1)=10^{-3}, P(b_3^1\mid b_2^1)=10^{-3}, P(b_4^1\mid b_3^1)=10^{-2}$ | $R=\{1,3\}: 3/4$; $R=\{1,3,4\}: 1$ | Adding nodes where the "mechanism itself is rare" completes the pathway. |
| Contextual Confounding (Ex 4.7) | $B_1$ prob 0.5, $P(B_2=1\mid B_1=1)=\delta$, $B_3=B_1\wedge B_2$ | Triadic: $\to 1$; Binary $B_2\to B_3$: accuracy $\to 1/2$ ($\delta\to 0$) | Ignoring context $B_1$ causes the do-posterior to deviate by 50%. |

### LLM Demo: Pathways for Homelessness

The authors had an LLM generate a causal chain for a fictional case (35yo male with schizophrenia $A$ $\to$ fired $B$ $\to$ evicted $C$ $\to$ family estrangement $D$ $\to$ chronic homelessness $E$) and then separately estimate conditional probabilities for each mechanism:

| Edge | Conditional Probability |
|------|-------------------------|
| $P(B\mid A)$ | 0.55 |
| $P(C\mid B)$ | 0.80 |
| $P(D\mid C)$ | **0.05** |
| $P(E\mid D)$ | 0.20 |
| $P(E)$ Prior | 0.0005 |

With $R=\{A\}$, the pathway explanation score $\mathcal{E}^K_{R\to t} \approx 0.29$, **clearly identifying the weak link at $C\to D$**—eviction alone does not explain family estrangement. Adding $A\to D$ or rewriting $C$ is suggested. This exemplifies the **falsification power** of the framework.

### Key Findings
- **The rarest mechanism on the chain dominates the score**: Larger log-likelihood gaps tighten the upper bound of the score. This provides a tool for diagnosing LLM causal narratives: low scores with small conditional probabilities indicate suspicious edges.
- **"Non-rare context" must be explicitly modeled**: Examples show that excluding context events that are not rare (e.g., "$|X|$ is normal") can drop abstraction accuracy to 0.5. This contrasts with traditional RCA which only looks at "outlier root causes."
- **Necessity is implicit**: Although the score does not explicitly include counterfactual necessity (Rung 3), high interventional likelihood for a rare target event implies a high Probability of Necessity (Appendix B).

## Highlights & Insights
- **Translating verbal chains into falsifiable scores**: It captures the implicit promise in human language that "each step in the chain is not too strange" and converts it into log-likelihood terms.
- **Poisson Tail p-values**: A concise way to extend SCM theory into arbitrary spaces while providing p-value corrections for multiple testing, avoiding asymptotic assumptions of extreme value theory.
- **Unified Scale for Score and Accuracy**: Both are normalized to $-\log P(B_t=1)$, allowing design choices like adding context nodes to be evaluated as numerical trade-offs.
- **Transferable to LLM Self-Check**: The homelessness example demonstrates a pipeline for using LLMs to generate chains, estimate weights, and then self-falsify via the framework.

## Limitations & Future Work
- **Truth vs. Consistency**: The framework only measures if an explanation is consistent with data/beliefs, not if the underlying causal graph is objectively true.
- **Monotonicity Assumption**: Real systems with threshold responses may violate this; p-values are used to tolerate deviations, but robust studies are needed.
- **Markov Property in Abstraction**: Binarization may destroy independence; high-dimensional estimation of $r$ is not discussed.
- **Automated Discovery**: The paper focuses on evaluating given pathways rather than searching for them from scratch.

## Related Work & Insights
- **vs. RCA**: RCA focuses on root cause sets and Shapley values (context-dependent), while this requires intermediate paths to be explanatory and uses context-free log-likelihood additivity.
- **vs. Extreme Value Theory**: This framework handles "non-asymptotic / arbitrary rarity," including imbalanced binary variables and near-zero values.
- **vs. Mediation Analysis**: Mediation decomposes average effects; this scores the explanatory power of a single specific occurrence.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->
<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)
- [\[ICML 2026\] The (Marginal) Value of a Search Ad: An Online Causal Framework for Repeated Second-price Auctions](the_marginal_value_of_a_search_ad_an_online_causal_framework_for_repeated_second.md)

</div>

<!-- RELATED:END -->
