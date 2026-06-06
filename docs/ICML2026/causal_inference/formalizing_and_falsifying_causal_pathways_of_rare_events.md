---
title: >-
  [Paper Note] Formalizing and Falsifying Causal Pathways of Rare Events
description: >-
  [ICML 2026][Causal Inference][Causal Pathways] This paper formalizes "verbal causal explanations" of rare events as a **causal pathway**—a subgraph composed of binarized events—and defines a **pathway explanation score**…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Causal Pathways"
  - "Rare Events"
  - "Explanation Score"
  - "Causal Abstraction"
  - "Falsifiability"
date: 2026-05-08
content_hash: 7e792057b80be965
---

# Formalizing and Falsifying Causal Pathways of Rare Events

**Conference**: ICML 2026  
**arXiv**: [2605.31254](https://arxiv.org/abs/2605.31254)  
**Code**: None  
**Area**: Causal Inference / Explainable AI / Root Cause Analysis for Rare Events  
**Keywords**: Causal Pathways, Rare Events, Explanation Score, Causal Abstraction, Falsifiability  

## TL;DR
This paper formalizes "verbal causal explanations" of rare events as a **causal pathway**—a subgraph composed of binarized events—and defines a **pathway explanation score** to quantify the explanatory power of "root causes + intermediary pathways" on a target event, resulting in a falsifiable evaluation framework for causal explanations.

## Background & Motivation

**Background**: Causal analysis of anomalies/rare events (natural disasters, stock market crashes, technical system failures, abnormal gene expressions, etc.) currently follows the **root cause analysis (RCA)** approach—identifying a small subset of "root cause nodes" in a Structural Causal Model (SCM) that significantly increase the counterfactual probability of the target event.

**Limitations of Prior Work**: RCA only answers "who moved," not "how it moved to the target." A realistic explanation often includes (i) multiple interacting intermediary mechanisms and (ii) context variables controlling propagation. Providing only a list of root causes is uninterpretable to humans and cannot be experimentally falsified by AI systems.

**Key Challenge**: Two categories of related existing works are each incomplete:
1. **Extreme Value Statistical Causal Models** (Engelke 2025, Klüppelberg 2026): Rely on asymptotic/heavy-tail parametric assumptions and cannot handle events that are "non-extreme but statistically rare" (e.g., values very close to 0, severely imbalanced binary events).
2. **Path-specific Mediation Analysis** (path-specific effects): Decomposes average causal effects, answering "what share path A occupies," but not "which part of the graph provides a good explanation for this specific observation."
3. **Causal Abstraction** (Rubenstein 2017, Beckers 2020): Operates at the total model level, lacking a local abstraction concept at the **event-level**.

**Goal**: Establish an **event-level** formal system for causal explanations that: (a) does not rely on asymptotic extreme value assumptions; (b) applies to arbitrary value spaces (continuous/discrete/text embeddings); (c) is itself falsifiable via data or consistency tests; and (d) can be automatically derived from fine-grained SCMs through abstraction.

**Key Insight**: Redefine an "explanation" as a **subgraph + set of binarized events** rather than just a list of root causes. Specifically, given a target event $B_t=1$ and a set of root causes $\mathbf{B}_R$, evaluate how close the log-likelihood of other events on the pathway $\mathbf{B}_{K\setminus R}=\mathbf{1}$ occurring simultaneously after $do(\mathbf{B}_R=\mathbf{1})$ is to the log-rarity of the target. This explicitly writes the requirement that "intermediary events must also look reasonable" into the scoring function.

**Core Idea**: Use the **log-likelihood ratio** $\mathcal{E}^K_{R\to t} := 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(B_t=1)}$ as a falsifiable measure of explanation quality; and unify the binarization of variables in arbitrary spaces through **feature monotonicity**, allowing the theory to generalize smoothly from binary SCMs to continuous/discrete/text variables.

## Method

### Overall Architecture
The framework consists of four layers:

1. **Cluster Layer**: Given a binary variable DAG $\mathcal{C}$ and joint distribution $P_\mathbf{B}$, "perturbations" are modeled as soft/hard interventions on the mechanisms $P(B_i\mid \mathbf{B}_{\mathrm{Pa}(i)})$ of certain nodes; the set of replaced indices $R$ constitutes the **root causes**.
2. **Pathway Layer**: A target node $B_t$ is singled out within the cluster, and the "subgraph relevant to this explanation" is extracted as pathway $\mathcal{P}$. $\mathcal{P}$ differs from $\mathcal{C}$ only on edges pointing to $R$, as $do(\mathbf{B}_R=\mathbf{1})$ severs those incoming edges.
3. **Abstraction Layer**: For real systems (where variables can be real-valued/categorical/tokens), each original variable is mapped to a binary event $B_j := \chi_{[\tau(\mathbf{x}_{I_j}),\infty)}(\tau(\mathbf{X}_{I_j}))$ via a **feature function $\tau_j$ + threshold**, entering the pathway layer.
4. **Evaluation Layer**: A pathway is falsified using three types of consistency tests (data consistency / internal qualitative QA / internal quantitative QA).

Input is (SCM, observed sample, target event); output is (pathway subgraph $\mathcal{P}$, root cause set $R$, explanation score $\mathcal{E}^K_{R\to t}$, and abstraction accuracy $r$).

### Key Designs

1. **Cluster / Pathway explanation score (Two-level definition)**:
    - **Function**: Maps "a set of root causes explaining a target event" to a score in the $[0,1]$ interval; a score closer to 1 indicates a stronger explanation.
    - **Mechanism**: The cluster score $\mathcal{E}_{R\to K} = 1 - \frac{\log P(\mathbf{B}=\mathbf{1}\mid do(\mathbf{B}_R=\mathbf{1}))}{\log P(\mathbf{B}=\mathbf{1})}$ measures "how much root causes pull the entire cluster of events toward probability"; the pathway score $\mathcal{E}^K_{R\to t}$ replaces the denominator with $\log P(B_t=1)$. Therefore, **when target rarity is greater than cluster rarity**, the pathway score is strictly more stringent than the cluster score. The two satisfy an affine relationship $1-\mathcal{E}^K_{R\to t} = (1-\mathcal{E}_{R\to K}) \cdot \frac{\log P(\mathbf{B}=\mathbf{1})}{\log P(B_t=1)}$, ensuring contributions of each node on the pathway remain additive (Eq. 11), allowing greedy selection of $R$. In the chain example (Example 3.6), with $P(b_1^1)=10^{-3}, P(b_3^1\mid b_2^1)=10^{-3}, P(b_4^1\mid b_3^1)=10^{-2}$, taking $R=\{1,3\}$ yields $\mathcal{E}^K_{R\to t}=3/4$; adding $B_4$ reaches 1, intuitively corresponding to "each mechanism along the chain being reasonable."
    - **Design Motivation**: The cluster score is inherited from Oesterle 2025 but only checks if root causes raise cluster likelihood, without requiring intermediaries to "look normal." The pathway score explicitly penalizes cases where "intermediary events are themselves very strange" (controlled by the log-likelihood gap $\Delta_i := [\log P(B_{\mathrm{Pa}(i)}=\mathbf{1}) - \log P(B_j=1)]_+$ in Lemma 3.7), forcing the explanation to scrutinize every intermediary. This is the key to formalizing "verbal causal chains" into falsifiable metrics.

2. **Feature monotonicity + Binarization (Bridge from arbitrary space to binary pathways)**:
    - **Function**: Enables the pathway theory to be used on real-valued/discrete/token/embedding variables of arbitrary distributions and provides probabilistic guarantees for the confidence at which intermediary events are supported by explanation scores.
    - **Mechanism**: Each $X_j$ is paired with a feature function $\tau_j:\mathcal{X}_j\to\mathbb{R}$, defining event $B_j := \{\tau_j(X_j) \geq \tau_j(x_j)\}$. Mechanism $P(X_j\mid \mathbf{X}_{\mathrm{Pa}(j)})$ is considered monotonic relative to $(\tau_j, \tau_{\mathrm{Pa}(j)})$ if larger parent features $\Rightarrow$ stochastically larger child feature distributions. Under this condition, Lemma 4.2 gives "for $x_j$ sampled from $P(X_j\mid \mathbf{x}_{\mathrm{Pa}(j)})$, the probability that conditional likelihood $\leq \alpha$ is $\leq \alpha$." Theorem 4.3 generalizes this to a DAG: for any $\mathbf{x}_R$, generating remaining variables from $do(\mathbf{x}_R)$, the probability that negative log-likelihood $L\geq c$ does not exceed $\sum_{i=0}^{n-|R|-1}\frac{c^i}{i!}e^{-c}$ (Poisson tail with DOF correction), serving as the **p-value** for pathway score deviations from 1.
    - **Design Motivation**: (i) Explaining "$X\geq x \to Y\geq y$" is more falsifiable (robust to specific values) than "$X=x \to Y=y$", matching human linguistic habits; (ii) even if true distributions do not strictly satisfy feature monotonicity, the p-value serves as a diagnostic threshold for "allowed deviation" without breaking the framework; (iii) feature functions like $\tau_X(x):=-|x|$ can express events like "$x$ is a normal value" (Example 4.8), covering scenarios unreachable by asymptotic extreme value theory.

3. **Pathway abstraction + natural micro-realization (Event-level causal abstraction)**:
    - **Function**: Automatically generates a coarse-grained pathway explanation $(\mathcal{C}, \mathcal{P}, P_\mathbf{B})$ from a fine-grained SCM $(\mathcal{G}, P_\mathbf{X})$ and quantifies coarse-graining loss.
    - **Mechanism**: Abstraction accuracy is defined as $r := 1 - \max_{S, \mathbf{b}_S} \frac{D_{KL}[P_\mathbf{X}(\mathbf{B}\mid do(\mathbf{B}_S=\mathbf{b}_S))\,\|\,P_\mathbf{B}(\mathbf{B}\mid do(\mathbf{B}_S=\mathbf{b}_S))]}{-\log P_\mathbf{X}(B_t=1)}$, normalizing the KL distance between post-intervention distributions of the abstract and true models by target rarity. Interventions $do(\mathbf{B}_j=b_j)$ are ill-posed in the original model (multiple $X_j$ map to the same $B_j$); this is resolved using **natural micro-realization**: interpreting $do(\mathbf{B}_S=\mathbf{b}_S)$ as "independently sampling from $\prod_{i\in S}P_\mathbf{X}(X_i\mid B_i=b_i)$ and then applying the original model $do$," ensuring consistent probabilistic operators. The explanation score can similarly be rewritten in KL form $\mathcal{E}^K_{R\to t} = 1 - \frac{D_{KL}(\delta_\mathbf{1}\|P_\mathbf{B}(\mathbf{B}\mid do(\mathbf{B}_R=\mathbf{1})))}{-\log P_\mathbf{B}(B_t=1)}$, placing it on the same scale as accuracy $r$ for unified trade-offs.
    - **Design Motivation**: Transforms design choices, such as "whether to include a certain context variable in the pathway," into computable trade-offs between accuracy vs. explanation score. In Example 4.8, a ternary pathway retaining context node $B_1$ ("$|X|\leq x$") significantly outperforms a binary simplification $B_2\to B_3$ in both accuracy and explanation score, as ignoring context forces negative effects of confounding paths into the explanation, distorting conditional probabilities.

### Loss & Training
This is a purely theoretical framework; no learning is involved. During evaluation, all probabilistic assumptions can be estimated from observed samples (data consistency tests) or estimated from LLMs/experts via QA (internal consistency tests). The practical selection of root cause set $R$ from a cluster uses a greedy algorithm: $R\gets R\cup\{\arg\max_i \mathcal{E}^K_{\{i\}\cup R\to t}\}$, with optimality guaranteed by the additivity in (11), complexity $O(|K|\cdot|R|)$.

## Key Experimental Results

The paper focuses on **theory + conceptual examples** and lacks standard benchmark comparisons. Three typical examples demonstrate the framework's behavior:

### Main Results: Score Shape vs. Event Rarity

| Example | Setting | Explanation Score | Implication |
| :--- | :--- | :--- | :--- |
| Gaussian Causal Pair (Ex 4.6) | $Y=\rho X+N$, $\rho=0.5$, $x\geq 3$, $y\approx \rho x$ | $\geq 0.8$ | "$X\geq x$ explains $Y\geq y$ by at least 80%", matching acceptable causal statements in human language. |
| Ternary Chain (Ex 3.6) | $P(b_1^1)=10^{-3}, P(b_3^1\mid b_2^1)=10^{-3}, P(b_4^1\mid b_3^1)=10^{-2}$ | $R=\{1,3\}: 3/4$; $R=\{1,3,4\}: 1$ | Nodes with "rare mechanisms themselves" must be added to complete the pathway. |
| Context Confounding (Ex 4.7) | $B_1$ prob 0.5, $P(B_2=1\mid B_1=1)=\delta$, $B_3=B_1\wedge B_2$ | Ternary: $\to 1$; Binary $B_2\to B_3$ accuracy $\to 1/2$ ($\delta\to 0$) | Ignoring context $B_1$ causes do-posterior to deviate by 50%. |

### Real LLM Demo: Homelessness Causal Pathway

The authors asked an LLM to generate a causal chain for a fictional case (35-year-old male with schizophrenia $A \to$ fired $B \to$ evicted $C \to$ family estrangement $D \to$ chronic homelessness $E$) and then used the same LLM to estimate conditional probabilities for each mechanism separately:

| Edge | Conditional Probability |
| :--- | :--- |
| $P(B\mid A)$ | 0.55 |
| $P(C\mid B)$ | 0.80 |
| $P(D\mid C)$ | **0.05** |
| $P(E\mid D)$ | 0.20 |
| $P(E)$ Prior | 0.0005 |

Taking $R=\{A\}$, the pathway explanation score $\mathcal{E}^K_{R\to t} = 1 - \frac{\log(0.55\cdot 0.8\cdot 0.05\cdot 0.2)}{\log 0.0005} \approx 0.29$, **clearly identifying the weak link at $C\to D$**—"being evicted" by itself does not explain "family estrangement." Suggested fix: add a direct edge $A\to D$ or rewrite $C$. This exemplifies the **falsification capability** of the framework.

### Key Findings
- **The "rarest mechanism" on the chain dominates the score**: The larger the log-likelihood gap (Lemma 3.7), the tighter the upper bound on the pathway score. This provides a tool for diagnosing LLM causal narratives: low scores with significantly small conditional probabilities on an edge $\Rightarrow$ that edge is suspicious.
- **"Non-rare context" must be explicitly modeled**: Examples 4.7 / 4.8 show that events that control propagation but are not rare themselves (e.g., "$|X|$ is normal") can drop binary abstraction accuracy to 0.5 if removed. This contrasts sharply with traditional RCA, which only looks at "outlier root causes."
- **Necessity is implicit**: Although the explanation score doesn't explicitly include "counterfactual necessity" (rung 3), the rarity of the target event means high post-intervention likelihood $\Rightarrow$ high probability of necessity (Appendix B), thus eliminating the need for counterfactual axioms.

## Highlights & Insights
- **Translating "Verbal Causal Chains" into Falsifiable Scoring**: When humans/LLMs say "because A so B so C so D," every edge implies a commitment that "the occurrence of intermediary events is not too strange"; this paper explicitly writes that implicit commitment as log-likelihood terms in the score, allowing "correct-sounding" explanations to be falsified by data or the agent's own probabilistic beliefs.
- **Feature Monotonicity + Poisson Tail p-values**: Uses a concise differentiable condition to push the theory from binary SCMs to arbitrary spaces while providing p-value calibration for multiple testing. This adapts ideas from statistical extreme value theory into a "log-scale" framework, avoiding asymptotic assumptions and handling discrete objects like tokens/embeddings.
- **Explanation Score and Abstraction Accuracy on the Same Scale**: Both are normalized by $-\log P(B_t=1)$, turning design choices like "should I include context $B_1$ in the pathway" into numerical trade-offs. This KL perspective clearly complements path-specific effects in mediation analysis—the former evaluates "overall explanatory power for a single observation," the latter splits "the share of the average effect."
- **Transferable to LLM Self-Check**: The homelessness example in Section 5 is a minimum viable demo showing how to "use the same LLM to generate a chain, estimate edge weights, and score with the framework"—this pipeline can be directly integrated into any generative AI's factuality/causality self-check process.

## Limitations & Future Work
- **Authors' Admission**: The framework only answers "whether the explanation is consistent with data/beliefs," not "whether the latent causal graph is true"; high scores do not prove an explanation is correct.
- **Feature Monotonicity is a Strong Assumption**: Real systems (with sin functions / multimodality / threshold responses) may not satisfy it. The authors suggest using p-values to tolerate deviations, but robustness studies on real datasets are lacking.
- **Binarization Breaks Markov Property**: Even if $X_1\to X_2\to X_3$ satisfies $X_1\perp X_3\mid X_2$, they may no longer be independent after binarization. Thus, the KL distance between abstract $P_\mathbf{B}$ and true $P_\mathbf{X}$ needs active control; the cost of estimating $r$ in high dimensions is not discussed.
- **No Algorithm for Automated Pathway Selection**: The paper provides tools for "evaluating a given pathway"; discovering pathways still requires human/LLM proposals. Extending greedy selection of $R$ to "greedy selection of pathway subgraphs + context nodes" is a next step.
- **Small-scale LLM Demo**: Only one manual example. Systemic experiments on statistical metrics for LLM causal narratives in datasets like medical NLI or legal causal argumentation are expected in the future.

## Related Work & Insights
- **vs RCA (Budhathoki 2022, Lin 2018/2024, Li 2022, Gnecco 2021)**: RCA only outputs root cause sets + counterfactual contributions. This paper additionally requires intermediary pathways to be interpretable and scored, and swaps "contribution assignment" from Shapley-value mode (dependent on context choice) to log-likelihood additive decomposition (context-free, Eq. 6).
- **vs Extreme Value Statistical Causal Models (Engelke 2025, Klüppelberg 2026)**: They focus on asymptotics/heavy tails; this paper focuses on "non-asymptotic / arbitrary rarity," specifically covering non-tail scenarios like "extremely imbalanced binary variables / near-zero values" (echoing Ebtekar 2025).
- **vs Path-specific Effect / Mediation (Robins, Singal 2024)**: Mediation splits **average** effects into paths; this paper scores the explanatory power for a **single observation**—different goals, used complementarily.
- **vs Causal Abstraction (Rubenstein 2017, Beckers 2020)**: Traditional abstraction acts on the whole model; this paper performs **event-level local abstraction**, where accuracy is normalized by target rarity, making it more suitable for "constructing small-graph explanations for a single rare event."
- **Insights**: (i) Treat "$\log$-likelihood gap" as a diagnostic edge-weight metric for LLM reasoning chains; (ii) use natural micro-realization as a semantic anchoring method for "performing do-calculus on LLM natural language events," potentially combining with counterfactual prompting for falsifiable counterfactual reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Integrating RCA + Causal Abstraction + Extreme Value Statistics + LLM Self-check into a unified log-likelihood scoring framework is a rare example of "conceptually stitched yet self-consistent" work.
- Experimental Thoroughness: ⭐⭐ Primarily conceptual examples and a single LLM demo, lacking real-world datasets/benchmark comparisons; theoretical contribution is primary.
- Writing Quality: ⭐⭐⭐⭐ Definitions, lemmas, and examples are tightly paced with a complete appendix; some notation like $\mathcal{E}^K_{R\to t}$ vs $\mathcal{E}_{R\to t}$ is visually cluttered, making initial reading difficult.
- Value: ⭐⭐⭐⭐ Provides a clear, actionable formal tool for GenAI causality self-checks and explanation credibility assessment; expected to be adopted by the causal NLP / LLM safety communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ICML 2026\] Evaluating Bivariate Causal Statements Based on Mutual Compatibility](evaluating_bivariate_causal_statements_based_on_mutual_compatibility.md)
- [\[ICML 2026\] Causal Fine-Tuning under Latent Confounded Shift](causal_fine-tuning_under_latent_confounded_shift.md)
- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)

</div>

<!-- RELATED:END -->
