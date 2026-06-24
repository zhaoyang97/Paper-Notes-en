---
title: >-
  [Paper Note] Formalizing and Falsifying Causal Pathways of Rare Events
description: >-
  [ICML 2026][Causal Inference][Causal Pathways] This paper formalizes "verbal causal explanations" of rare events as **causal pathways**—subgraphs composed of binarized events. By defining a **pathway explanation score** to quantify the explanatory power of "root causes + mediation pathways" relative to the target event, the authors establish a falsifiable evaluation framework for causal explanations.
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Causal Pathways"
  - "Rare Events"
  - "Explanation Score"
  - "Causal Abstraction"
  - "Falsifiability"
date: 2026-05-08
content_hash: 3b1676c9a35e02a4
---

# Formalizing and Falsifying Causal Pathways of Rare Events

**Conference**: ICML 2026  
**arXiv**: [2605.31254](https://arxiv.org/abs/2605.31254)  
**Code**: None  
**Area**: Causal Inference / Explainable AI / Root Cause Analysis of Rare Events  
**Keywords**: Causal Pathways, Rare Events, Explanation Score, Causal Abstraction, Falsifiability  

## TL;DR
This paper formalizes "verbal causal explanations" of rare events as **causal pathways**—subgraphs composed of binarized events. By defining a **pathway explanation score** to quantify the explanatory power of "root causes + mediation pathways" relative to the target event, the authors establish a falsifiable evaluation framework for causal explanations.

## Background & Motivation

**Background**: Causal analysis of anomalies or rare events (e.g., natural disasters, stock market crashes, technical system failures, abnormal gene expression) currently follows the **root cause analysis (RCA)** paradigm. In Structural Causal Models (SCMs), RCA seeks a small set of "root cause nodes" that significantly increases the counterfactual probability of the target event.

**Limitations of Prior Work**: RCA only identifies "which variables moved" but fails to explain "how they affected the target." A realistic explanation typically includes (i) multiple interacting mediation mechanisms and (ii) context variables that control propagation. Simply providing a list of root causes is neither interpretable for humans nor falsifiable for AI systems.

**Key Challenge**: Two categories of related work are incomplete:
1. **Extreme Value Causal Models** (Engelke 2025, Klüppelberg 2026): These rely on asymptotic or heavy-tail parametric assumptions and cannot handle "non-extreme but statistically rare" events (e.g., values very close to 0, highly imbalanced binary events).
2. **Path-specific Mediation Analysis**: These decompose average causal effects to answer "what share did path A take," but do not address "which part of the graph provides a good explanation for this specific observation."
3. **Causal Abstraction** (Rubenstein 2017, Beckers 2020): These operate at the entire model level and lack **event-level** local abstraction concepts.

**Goal**: Establish an **event-level** formal system for causal explanation that satisfies: (a) independence from asymptotic extreme value assumptions; (b) applicability to arbitrary value spaces (continuous/discrete/text embeddings); (c) falsifiability of the explanation via data or consistency tests; and (d) automated derivation from fine-grained SCMs through abstraction.

**Key Insight**: Redefining an "explanation" as a **subgraph + a set of binary events** rather than a root cause list. Specifically, given a target event $B_t=1$ and a set of root causes $\mathbf{B}_R$, one evaluates how closely the log-likelihood of other events $\mathbf{B}_{K\setminus R}=\mathbf{1}$ occurring simultaneously on the pathway after $do(\mathbf{B}_R=\mathbf{1})$ matches the log-rarity of the target. This explicitly incorporates the requirement that "mediation events must also look plausible" into the scoring function.

**Core Idea**: Utilizing the **log-likelihood ratio** $\mathcal{E}^K_{R\to t} := 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(B_t=1)}$ as a falsifiable measure of explanation quality. Through **feature monotonicity**, variables in arbitrary spaces are unified via binarization, allowing the theory to generalize smoothly from binary SCMs to continuous, discrete, or text variables.

## Method

### Overall Architecture
The paper addresses how to transform "verbal causal chains" (e.g., "Because A, therefore B, therefore C, hence the target event") into a score falsifiable by data or probabilistic beliefs. The approach reorganizes the entire explanation into a subgraph of binary events (pathway). An explanation score in $[0,1]$ is first defined for binary SCMs. Then, "feature monotonicity + binarization" is used to extend this to any space (real-valued/discrete/tokens). Finally, "event-level causal abstraction" quantifies the information loss during coarse-graining from fine-grained models. Inputs are (SCM, observed sample, target event); outputs are (pathway subgraph $\mathcal{P}$, root cause set $R$, explanation score $\mathcal{E}^K_{R\to t}$, abstraction accuracy $r$).

### Key Designs

**1. Cluster / Pathway explanation score: Making causal chains falsifiable via scoring**

Naive RCA only outputs root causes without auditing whether the "mediation events themselves are reasonable," making it difficult to overturn "plausible-sounding" explanations. This paper adopts the cluster score $\mathcal{E}_{R\to K} = 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(\mathbf{B}=\mathbf{1})}$ from Oesterle 2025, which measures how much $do(\mathbf{B}_R=\mathbf{1})$ increases the likelihood of the entire cluster. However, the cluster score only checks if root causes raise the likelihood without requiring the mediation to "look normal."

The pathway score replaces the denominator with the log-rarity of the target event itself: $\mathcal{E}^K_{R\to t} = 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(B_t=1)}$. When the target is rarer than the whole cluster, this score is more stringent—any "inherently strange mediation events" are penalized by the log-likelihood term in the numerator. The two satisfy an affine relationship $1-\mathcal{E}^K_{R\to t} = (1-\mathcal{E}_{R\to K}) \cdot \frac{\log P(\mathbf{B}=\mathbf{1})}{\log P(B_t=1)}$, allowing the contribution of each node on the pathway to remain additive (Eq. 11), facilitating greedy selection of root causes. Lemma 3.7 provides the log-likelihood gap for each edge $\Delta_i := [\log P(B_{\mathrm{Pa}(i)}=\mathbf{1}) - \log P(B_i=1)]_+$ to bound the score; a larger gap indicates a rarer mechanism that drags down the entire chain. This definition formally encodes the implicit promise in verbal causal chains that "each step should not be too unusual."

**2. Feature monotonicity + Binarization: Extending theory to arbitrary spaces**

Variables in real systems are often real-valued, categorical, or embeddings. The paper assigns a feature function $\tau_j:\mathcal{X}_j\to\mathbb{R}$ to each $X_j$, mapping variables to binary events $B_j := \{\tau_j(X_j) \geq \tau_j(x_j)\}$ (i.e., "the feature of $X_j$ is at least as large as what was observed"). It requires the mechanism $P(X_j\mid \mathbf{X}_{\mathrm{Pa}(j)})$ to be monotonic with respect to $(\tau_j, \tau_{\mathrm{Pa}(j)})$, meaning larger parent features stochastically result in larger child features.

Under this condition, Lemma 4.2 provides a critical tail probability guarantee: "For $x_j$ sampled from $P(X_j\mid \mathbf{x}_{\mathrm{Pa}(j)})$, the probability that its conditional likelihood is $\leq \alpha$ does not exceed $\alpha$." Theorem 4.3 generalizes this to the whole DAG: after generating variables from $do(\mathbf{x}_R)$, the probability that the negative log-likelihood $L\geq c$ is at most $\sum_{i=0}^{n-|R|-1}\frac{c^i}{i!}e^{-c}$ (a Poisson tail with corrected degrees of freedom). This provides a **p-value** for diagnosing when explanation scores deviate from 1. Choosing "$X\geq x$ explains $Y\geq y$" over "$X=x$ explains $Y=y$" is more robust to exact values, aligns with human language, and is more falsifiable. By setting $\tau_X(x):=-|x|$, one can express events like "value $x$ is normal" (Example 4.8), covering scenarios that asymptotic EVT cannot reach.

**3. Pathway abstraction + natural micro-realization: Quantifying coarse-graining loss**

To automatically derive a coarse-grained pathway explanation $(\mathcal{C}, \mathcal{P}, P_\mathbf{B})$ from a fine-grained SCM $(\mathcal{G}, P_\mathbf{X})$, one must measure the information lost during binarization. The paper defines abstraction accuracy $r := 1 - \max_{S, \mathbf{b}_S} \frac{D_{KL}[P_\mathbf{X}(\mathbf{B}\mid do(\mathbf{B}_S=\mathbf{b}_S))\,\|\,P_\mathbf{B}(\mathbf{B}\mid do(\mathbf{B}_S=\mathbf{b}_S))]}{-\log P_\mathbf{X}(B_t=1)}$. The challenge is that $do(\mathbf{B}_j=b_j)$ is ill-defined in the original model if multiple $X_j$ map to the same $B_j$. **Natural micro-realization** defines it as: sample underlying variables from $\prod_{i\in S}P_\mathbf{X}(X_i\mid B_i=b_i)$ independently, then apply $do$ in the original model.

The key benefit is that the explanation score can also be rewritten in terms of KL divergence $\mathcal{E}^K_{R\to t} = 1 - \frac{D_{KL}(\delta_\mathbf{1}\|P_\mathbf{B}(\mathbf{B}\mid do(\mathbf{B}_R=\mathbf{1})))}{-\log P_\mathbf{B}(B_t=1)}$, placing it on the same scale as accuracy $r$. Design choices, such as whether to include a context variable in the pathway, become a trade-off between accuracy and explanation score. In Example 4.8, a three-node pathway retaining context $B_1$ ("$|X|$ is normal") outperforms a binary simplification $B_2\to B_3$ in both accuracy and score; excluding the context causes the negative effect of a confounding path to be absorbed into the explanation, distorting conditional probabilities.

### Mechanism Mechanism
This framework is purely theoretical. In practice, probabilities are either estimated from observed samples (data consistency tests) or via Q&A with LLMs/experts (internal consistency tests). The root cause set $R$ is selected from the cluster using a greedy algorithm: $R\gets R\cup\{\arg\max_i \mathcal{E}^K_{\{i\}\cup R\to t}\}$. Optimality is guaranteed by the additivity in Eq. 11, with complexity $O(|K|\cdot|R|)$.

## Key Experimental Results

The paper primarily features **theoretical and conceptual examples** rather than standard benchmarks. Three types of examples demonstrate the framework's behavior:

### Main Results: Explanation Score vs. Event Rarity

| Example | Setting | Explanation Score | Insight |
|------|------|---------|------|
| Gaussian Causal Pair (Ex 4.6) | $Y=\rho X+N$, $\rho=0.5$, $x\geq 3$, $y\approx \rho x$ | $\geq 0.8$ | "$X\geq x$ explains $Y\geq y$ by at least 80%", matching acceptable causal statements in human language. |
| Ternary Chain (Ex 3.6) | $P(b_1^1)=10^{-3}, P(b_3^1\mid b_2^1)=10^{-3}, P(b_4^1\mid b_3^1)=10^{-2}$ | $R=\{1,3\}: 3/4$; $R=\{1,3,4\}: 1$ | Adding nodes where the "mechanism itself is rare" is required to complete the pathway. |
| Contextual Confounding (Ex 4.7) | $B_1$ with 0.5 prob, $P(B_2=1\mid B_1=1)=\delta$, $B_3=B_1\wedge B_2$ | Ternary pathway: $\to 1$; Binary $B_2\to B_3$ accuracy $\to 1/2$ ($\delta\to 0$) | Ignoring context $B_1$ causes the do-posterior to deviate by 50%. |

### Real LLM Demo: Causal Pathway of Homelessness

The authors prompted an LLM to generate a causal chain for a fictional case (35-year-old male with schizophrenia $A$ $\to$ fired $B$ $\to$ evicted $C$ $\to$ family estrangement $D$ $\to$ chronic homelessness $E$). The same LLM was used to estimate conditional probabilities:

| Edge | Conditional Probability |
|----|---------|
| $P(B\mid A)$ | 0.55 |
| $P(C\mid B)$ | 0.80 |
| $P(D\mid C)$ | **0.05** |
| $P(E\mid D)$ | 0.20 |
| $P(E)$ Prior | 0.0005 |

For $R=\{A\}$, the pathway explanation score $\mathcal{E}^K_{R\to t} = 1 - \frac{\log(0.55\cdot 0.8\cdot 0.05\cdot 0.2)}{\log 0.0005} \approx 0.29$. This **explicitly identifies the weak link at $C\to D$**—eviction itself does not explain family estrangement. Suggested fixes include adding a direct edge $A\to D$ or rewriting $C$. This exemplifies the **falsifiability** of the framework.

### Key Findings
- **The rarest mechanism on the chain dominates the score**: A larger log-likelihood gap (Lemma 3.7) tightens the upper bound of the pathway score. This serves as a diagnostic tool for LLM causal narratives: a low score with a significantly small conditional probability on an edge implies that edge is suspicious.
- **"Non-rare context" must be explicitly modeled**: Examples 4.7/4.8 show that events that control propagation but are not rare (e.g., "$|X|$ is normal") can drop binary abstraction accuracy to 0.5 if removed. This contrasts with traditional RCA, which focuses only on "deviant root causes."
- **Necessity is implicit**: Although the score does not explicitly include "counterfactual necessity" (Rung 3), the rarity of the target event implies that high likelihood after intervention leads to a high probability of necessity (Appendix B), eliminating the need for counterfactual axioms.

## Highlights & Insights
- **Translating "verbal causal chains" into falsifiable scores**: When humans/LLMs say "A caused B caused C caused D," there is an implicit promise that each mediation event's occurrence is "not too unusual." This paper explicitly encodes this promise as log-likelihood terms, allowing "plausible-sounding" explanations to be falsified by data or the agent's own probabilistic beliefs.
- **Feature monotonicity + Poisson tail p-values**: A concise differential condition extends the theory from binary SCMs to any space while providing p-value corrections for multiple testing. This reframes statistical extreme value theory on a "logarithmic scale," avoiding asymptotic assumptions and handling discrete objects like tokens.
- **Unifying explanation score and abstraction accuracy**: Both are normalized by $-\log P(B_t=1)$, turning design choices (like whether to include context $B_1$ in a pathway) into numerical trade-offs. This KL-based perspective complements path-specific effects in mediation analysis—the former evaluates "overall explanatory power for a specific observation," while the latter splits "proportion of the average effect."
- **Transferability to LLM self-checking**: The homelessness case in Section 5 demonstrates a minimal viable demo where one LLM generates a chain, estimates weights, and is then graded by the framework—a pipeline that can be integrated into GenAI fact-checking and causal self-audit workflows.

## Limitations & Future Work
- **The authors admit**: The framework only answers whether an explanation is "consistent with data/beliefs," not whether the underlying causal graph is "true." A high score does not prove correctness.
- **Feature monotonicity is a strong assumption**: Real-world systems (with sine waves, multi-modality, or threshold responses) may not satisfy it. The authors suggest using p-values to tolerate deviations, but robust studies on real datasets are missing.
- **Binarization breaks Markov properties**: Even if $X_1\to X_2\to X_3$ satisfies $X_1\perp X_3\mid X_2$, they may not be independent after binarization. Thus, the KL distance between $P_\mathbf{B}$ and $P_\mathbf{X}$ must be actively controlled; the cost of estimating $r$ in high dimensions was not discussed.
- **Automated pathway discovery algorithm is missing**: The paper provides a tool for "evaluating a given pathway." Discovering pathways still requires human or LLM proposals. Future work could extend greedy $R$ selection to "greedy selection of pathway subgraphs + context nodes."
- **Small LLM demo scale**: There is only one manual example without large-scale statistical metrics. Systematic experiments on Medical NLI or legal causal reasoning datasets are needed.

## Related Work & Insights
- **vs. RCA (Budhathoki 2022, Lin 2018/2024, Li 2022, Gnecco 2021)**: RCA outputs root cause sets and counterfactual contributions. This paper adds the requirement that mediation paths be interpretable and scored, shifting "contribution attribution" from Shapley-value modes (context-dependent) to additive log-likelihood decomposition (context-free, Eq. 6).
- **vs. Extreme Value Causal Models (Engelke 2025, Klüppelberg 2026)**: While they focus on asymptotic/heavy-tail behavior, this work addresses "non-asymptotic/arbitrary rarity," covering non-tail scenarios like highly imbalanced binary variables (echoing Ebtekar 2025).
- **vs. Path-specific Effect / Mediation (Robins, Singal 2024)**: Mediation decomposes **average** effects into paths; this framework scores the explanatory power of a **single observation**. They are complementary.
- **vs. Causal Abstraction (Rubenstein 2017, Beckers 2020)**: Traditional abstraction operates on the whole model. This paper performs **event-level local abstraction**, where accuracy is normalized by target rarity, making it more suitable for constructing small-graph explanations for specific rare events.
- **Insights**: (i) "Log-likelihood gaps" can be used as edge-weight diagnostics for LLM reasoning chains; (ii) Natural micro-realization serves as a semantic anchoring method for "doing do-calculus on LLM natural language events," potentially combining with counterfactual prompting for falsifiable reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifying RCA, causal abstraction, extreme value statistics, and LLM self-checking into a log-likelihood scoring framework is a rare "consistent conceptual synthesis."
- Experimental Thoroughness: ⭐⭐ Primarily conceptual examples and a single LLM demo. Lacks comparison on real-world datasets/benchmarks; theoretical contribution is primary.
- Writing Quality: ⭐⭐⭐⭐ Compact flow between definitions, lemmas, and examples with complete appendices. Some notation like $\mathcal{E}^K_{R\to t}$ versus $\mathcal{E}_{R\to t}$ is notation-heavy.
- Value: ⭐⭐⭐⭐ Provides a clear, deployable formal tool for GenAI causal self-audit and explanation reliability evaluation. Likely to be adopted by the Causal NLP and LLM Safety communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Causal Modeling of Selection in Evolution](causal_modeling_of_selection_in_evolution.md)
- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICLR 2026\] Topological Causal Effects](../../ICLR2026/causal_inference/topological_causal_effects.md)

</div>

<!-- RELATED:END -->
