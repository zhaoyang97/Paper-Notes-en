---
title: >-
  [Paper Note] Evaluating Bivariate Causal Statements Based on Mutual Compatibility
description: >-
  [ICML 2026][Causal Inference][Bivariate Causal Statements] This paper addresses scenarios where only bivariate causal statements exist without ground truth. It proposes two faithfulness-free compatibility scores (`comp`…
tags:
  - "ICML 2026"
  - "Causal Inference"
  - "Bivariate Causal Statements"
  - "Compatibility Scores"
  - "Confounding Backdoor"
  - "LLM Causal Evaluation"
date: 2026-05-08
content_hash: 6ca802f5501d14e7
---

# Evaluating Bivariate Causal Statements Based on Mutual Compatibility

**Conference**: ICML 2026  
**arXiv**: [2606.00278](https://arxiv.org/abs/2606.00278)  
**Code**: https://github.com/ejahn17/compatibility-scores  
**Area**: Causal Inference / Causal Discovery Evaluation / LLM Causal Reasoning  
**Keywords**: Bivariate Causal Statements, Compatibility Scores, Confounding Backdoor, LLM Causal Evaluation  

## TL;DR
This paper addresses scenarios where only bivariate causal statements exist without ground truth. It proposes two faithfulness-free compatibility scores (`comp` for linear cases and `incomp` for graph structures). By determining whether a multivariate model assembled from these pairwise statements requires "anomalous extra confounding" to explain the observed covariance, the method identifies incorrect causal claims and uses this to score LLM causal outputs.

## Background & Motivation
**Background**: Causal inference in fields like medicine and economics has long relied on expert-provided pairwise cause-effect claims; recently, many works have pivoted to letting LLMs directly produce such pairwise causal relationships. Both paths lack a mechanism where "errors can be detected"—traditional causal discovery algorithms have consistency guarantees under strong assumptions, whereas LLMs have neither training objectives nor theoretical constraints for this.

**Limitations of Prior Work**: In real-world scenarios lacking ground truth (GT), it is nearly impossible to verify whether a list of "$X_i \to X_j, \alpha_{ij}$" is correct. Existing "GT-free evaluation" works (Textor 2016, Eulig 2025, Sheth 2026) mostly measure the "consistency between the estimated graph and data," but only for full DAG estimations. Faller 2024 uses whether "multiple subset graphs can be stitched into the same large graph" for compatibility, but this requires subset graphs as input and is not directly applicable to "pairwise claims," the most common form of human/LLM output.

**Key Challenge**: For linear, acyclic pairwise claims $\{\alpha_{ij}\}$, the authors first prove (Lemma 2.3) that any such list **uniquely** induces a multivariate linear SEM that perfectly fits the observed covariance. Therefore, they do not conflict in the sense of Faller's "hard compatibility"—there are no exploitable constraints. Evaluation must be approached from a "soft" perspective (plausibility).

**Goal**: (1) Design a continuous compatibility score for linear pairwise claims that does not depend on faithfulness; (2) Design an incompatibility count for graph structural claims that only specify the presence/absence of causality/confounding; (3) Validate on synthetic data and LLM outputs that both can distinguish correct from incorrect claims.

**Key Insight**: The authors propose a **Confounding Postulate**—a "general" multivariate causal model should not have **more** unobserved confounding than its pairwise marginal models. Intuitively, after marginalizing a multivariate model into pairs, originally observed backdoor paths (e.g., $X_i \leftarrow X_k \to X_j$) become unobserved confounding, so the amount of confounding in the marginal model should only increase. Unless the induced multivariate model deliberately "tunes observed causal paths to exactly cancel out" its own confounding, such fine-tuning rarely occurs in random models (Theorem 2.9 provides a guarantee in the sense of expectation).

**Core Idea**: Use the difference between the "confounding of the induced multivariate SEM" and the "confounding of the pairwise marginal SEM" as the compatibility score; if the multivariate model instead "requires less" confounding to explain the covariance, it indicates that the claims require an counter-intuitive, precise cancellation and can be judged as untrustworthy.

## Method

### Overall Architecture
Input: Observed covariance matrix $\Sigma$ of $n$ variables (or sample estimate $\hat{\Sigma}$) + a list of $\binom{n}{2}$ pairwise causal claims.

- **Linear Case**: Each claim provides a direction $i\to j$ and a coefficient $\alpha_{ij}$. These are placed into a unit lower triangular matrix $A$ ($A_{ji}=\alpha_{ij}, i<j$), and by letting $\Gamma = I - A^{-1}$, the unique multivariate SEM $X=\Gamma X + N$ is obtained (Lemma 2.3). Then, the "squared covariance not explained by observed causal paths" is compared between "multivariate vs. pairwise" granularities, and the sum yields the `comp` score.
- **Graph Structural Case**: Each claim provides a two-node ADMG (with/without $\to$ edges, with/without $\leftrightarrow$ bidirected edges). All two-node ADMGs are merged into a statement graph $G$. `incomp(G)` is defined as the minimum number of edge additions/deletions (Hamming distance) required to transform $G$ into a graph that "can be stitched into a large ADMG."

Output: `comp` score (linear) or `incomp` score (graph). `comp < 0` → violation of the confounding postulate → falsify the claim list; `incomp > 0` → there must be a claim error, and the value reflects the magnitude of the errors.

### Key Designs

1. **Linear comp score = Marginal Confounding − Multivariate Confounding**:
    - **Function**: Measures how anomalous the multivariate model induced by this set of pairwise claims is using a scalar.
    - **Mechanism**: For each pair $i<j$, two types of confounding are defined. In the pairwise SEM $X_j = \alpha_{ij}X_i + \tilde N_{ij}$, the squared covariance not explained by the causal effect is $C^{biv}_{ij} = (\Sigma_{ij} - \alpha_{ij}\Sigma_{ii})^2$. In the multivariate SEM, the contribution of all "observed directed paths + complete backdoor paths" is subtracted using Wright’s path tracing to obtain $C^{mult}_{ij}(\Sigma,\Gamma) = (\Sigma_{ij} - \sum_{k\le i}\Sigma_{kk}\sum_{P_1: k\rightsquigarrow i, P_2: k\rightsquigarrow j} \Gamma_{P_1}\Gamma_{P_2})^2$. Finally, $\text{comp}(\Sigma,A) = \sum_{i<j} C^{biv}_{ij}(\Sigma,A_{ji}) - C^{mult}_{ij}(\Sigma, I-A^{-1})$. Intuition: When marginalizing a multivariate model to pairs, "observed backdoors" are swallowed into noise and become confounding, so $C^{biv} \ge C^{mult}$ is a property a "plausible" model should have; `comp < 0` is equivalent to violating Assumption 2.4 (confounding postulate).
    - **Design Motivation**: After proving Lemma 2.3, the authors found that "hard compatibility" was unusable, so they sought a soft constraint that does not rely on faithfulness but "holds in the vast majority of cases" for random models. The design of `comp` simultaneously satisfies: (a) It reduces to the classic bivariate confounding measure (Janzing-Schölkopf) in the two-variable case; (b) Theorem 2.9 proves that under a random SEM with "zero-mean coefficients + mechanism independence + non-degeneracy," the expected `comp > 0` for true claims; (c) Theorem 2.10 gives a sample complexity of $N = O(n^4(1+a+b)^4 V^4 / \varepsilon^2 \cdot \log(n/\delta))$ to keep $|\text{comp}(\hat\Sigma,A) - \text{comp}(\Sigma,A)|$ within $\varepsilon$, with 100 samples being sufficient for 90% sign accuracy in experiments.

2. **Graph incomp score = Minimum edit distance to a "stitchable" graph**:
    - **Function**: Counts the number of violations of global consistency when claims are qualitative (presence/absence of causality/confounding).
    - **Mechanism**: The union of all $\binom{n}{2}$ two-node ADMGs forms the statement graph $G$. Lemma 3.5 provides the necessary and sufficient conditions for "$G$ to be marginalizable from some large ADMG": (i) the directed part is acyclic; (ii) the directed part is **transitively closed**—if there is a path $X_i \rightsquigarrow X_j$ in $G$, there must be a direct edge $X_i \to X_j$ (as marginalization preserves reachability); (iii) if a confounding path with "dual arrowheads at endpoints + no arrowheads in the middle" exists between any two nodes, there must be a $X_i \leftrightarrow X_j$ bidirected edge (confounding paths necessarily become bidirected edges after marginalization). Define $\text{incomp}(G) = \min_{G^*} d(G, G^*)$, where $d$ is the mixed graph Hamming distance and $G^*$ ranges over all graphs satisfying (i)-(iii).
    - **Design Motivation**: Extends the hard compatibility of Faller 2024 to a "violation metric"—hard compatibility only answers "yes/no" and cannot distinguish between "almost correct, only one error" and "entirely wrong"; quantifying it with Hamming distance allows the score to directly reflect the lower bound of erroneous claims and naturally inherits global constraints from faithfulness + acyclicity assumptions.

3. **Finite-sample estimation and LLM evaluation protocol**:
    - **Function**: Applies `comp` to real data and defines a workflow for "evaluating LLM causal reasoning capabilities."
    - **Mechanism**: In practice, we obtain the sample covariance $\hat\Sigma = \frac{1}{N}\sum_r X^{(r)} X^{(r)\top}$. Before scoring, variables are standardized to unit variance, which is equivalent to replacing $\Sigma$ with the correlation matrix; claim coefficients $A$ are transformed by the same scaling. For LLM evaluation (using 7 country-level indicators from gapminder), variable descriptions + empirical correlation matrices are fed as prompts to different LLMs, asking them to output causal ordering + pairwise linear coefficients. The results of 15 independent runs are averaged and compared against a random baseline where coefficients are sampled from a zero-mean Gaussian with variance matching the LLM output.
    - **Design Motivation**: Moves the method beyond synthetic experiments to provide a **reproducible ranking of LLM causal capabilities**—the practical landing point of the paper. On a real-world problem without ground truth (gapminder), more capable LLMs tend to have higher `comp` scores, while many LLMs still receive negative scores (i.e., they are falsified by this method), proving the score can indeed pick out incorrect outputs.

### Loss & Training
This method is **pure evaluation, no training**. `comp` is a closed-form function of the covariance and coefficient matrices; `incomp` is a combinatorial optimization problem (solvable via enumeration/heuristics for small $n$).

## Key Experimental Results

### Main Results

**Ability of `comp` to distinguish correct from incorrect claims on synthetic data** (Figure 2): Ground truth is sampled from random linear Gaussian SEMs, and Gaussian noise $\sigma$ with increasing variance is added to the true coefficients to simulate "claim lists of decreasing quality." Results are averaged over 50 models × 20 noise levels per point.

| Setting | Proportion of positive scores at $\sigma=0$ (GT) | Trend as $\sigma$ increases | Key Observation |
|:---|:---|:---|:---|
| $n=10, p=0.5$, varying $m$ (latent variables 0/1/3/5) | ≈ 95%+ | Monotonic decrease | Robust to number of latents; curves nearly overlap |
| $n=10, m=3$, varying $p$ (sparsity 0.3/0.5/0.7/0.9) | ≈ 95%+ | Monotonic decrease | Larger $p$ (denser graphs) leads to stronger discriminative power |
| $p=0.5, m=3$, varying $n$ (variables 5/10/15/20) | ≈ 95%+ | Monotonic decrease | Larger $n$ leads to stronger discriminative power |

**LLM Causal Claim Evaluation** (Figure 4, gapminder 7 variables):

| Model Group | `comp` score (Avg. of 15 runs, relative to random baseline) | Conclusion |
|:---|:---|:---|
| High-capability LLMs | Significantly above 0 and higher than random baseline | Pass falsification |
| Medium LLMs | Close to random baseline | Inconclusive |
| Low-capability LLMs | **Negative scores** | Falsified by `comp`; evidence shows causal claims are untrustworthy |

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|:---|:---|:---|
| `comp` with true $\Sigma$ vs. $\hat\Sigma$ (Figure 3, varying $N$) | Sign consistency > 90% when $N \ge 100$; relative error drops fast to < 5% | Empirical convergence is much faster than the worst-case bound in Theorem 2.10 |
| Number of latent variables $m \in \{0,1,3,5\}$ | Positive score proportion curves nearly overlap | The method is robust to unobserved confounding (which is the reason for its existence) |
| Sparsity $p$ increased from 0.3 to 0.9 | Positive proportion of GT stays same; positive proportion of incorrect values drops sharply | Denser graphs → more backdoor paths → larger difference between marginal vs. multivariate confounding → easier to catch wrong claims |
| Random baseline LLM | Slightly above 0 | Even random guessing can correctly identify that "very wrong models should be negative" |

### Key Findings
- **Core Conclusion**: On synthetic data, `comp` almost never misses the ground truth (≥ 95% positive scores) and can stably distinguish incorrect values after adding noise; sample complexity in practice is much better than theoretical bounds.
- **Dense Graphs are Advantageous**: When $n$ or $p$ is large, observed backdoor paths in the multivariate SEM increase, and more information is lost during marginalization. The discriminative power of `comp` increases accordingly—this is perfectly self-consistent with the mechanism (the postulate comes from "marginalization swallowing backdoors").
- **Insights for LLM Applications**: High-capability LLMs achieve higher `comp` scores on gapminder, while several low-capability LLMs achieve **negative scores** and perform worse than the random baseline, indicating that this method can be directly used as a "GT-free health check for LLM causal capabilities."

## Highlights & Insights
- **Bypassing faithfulness is the real selling point**: Traditional causal discovery consistency guarantees are built on faithfulness. Theorem 2.9, under a weaker randomness assumption of "zero-mean coefficients + mechanism independence + non-degeneracy," ensures that true claims have an expected `comp > 0`—very attractive for real-world scenarios (medicine, socio-economics) where the validity of faithfulness is questioned.
- **"Marginalization inevitably increases confounding" as a prior**: Using this physical intuition to replace faithfulness and using "induced multivariate models must fine-tune to cancel out" as counter-evidence is an elegant way to translate causal identifiability into a "rare event in parameter space" problem. This can be transferred to any evaluation task where "local estimates are stitched into global models" (e.g., distributed federated causal discovery, cross-modal alignment verification).
- **Actionable template for evaluating LLM causal outputs**: Many LLM-causal works can only compare on artificial datasets with GT. This paper provides a zero-annotation workflow of "feeding correlation matrix + extracting coefficients + calculating comp"—a concrete and usable metric for the LLM evaluation community, especially when evaluating scientific discovery or medical diagnosis tasks.

## Limitations & Future Work
- **The authors acknowledge**: (1) A positive score can still be wrong (i.e., a wrong claim just happens to induce a "plausible" model), so `comp` can only **falsify**, not **verify**; (2) Linear + acyclic + Gaussian assumptions are strong; (3) `comp` values depend on variable scaling, requiring prior standardization.
- **Observed limitations**: (a) Theorem 2.9 assumes zero-mean coefficients, which does not hold in many fields (e.g., "treatment → improvement" coefficients in medicine are usually positive); if the real distribution is biased, the expected guarantee weakens; (b) Experiments only go up to $n=20$; for large graphs with $n=100+$ in real data, whether the computation of path tracing sums is manageable is not demonstrated; (c) The minimization in `incomp` is a combinatorial problem, and no specific approximation algorithm for large $n$ is provided; (d) LLM experiments only used 7 variables from gapminder; effectiveness on more complex medical/biological causal systems is unknown.
- **Possible improvements**: Generalizing `comp` to non-linear SEMs (replacing linear residuals with conditional means/kernel covariances); introducing "score decay" to distinguish "minor errors" from "serious errors" rather than just looking at the sign; combining with existing LLM causal methods (agent self-consistency, multi-sample voting) to use `comp` as a reward signal for RLHF or filtering LLM outputs.

## Related Work & Insights
- **vs. Faller et al. (2024)**: They require "multiple subset ADMGs" as input and perform hard compatibility checks; this paper only requires "$\binom{n}{2}$ pairwise claims" and provides **soft** scores, covering the most natural output form of human experts/LLMs, and extends hard compatibility (Lemma 3.5) into a continuous measure of "violation count."
- **vs. Janzing & Schölkopf (2018) Bivariate Confounding Measure**: They only define squared unexplained covariance in the two-variable case; this paper uses Wright’s path tracing to generalize the same idea to multiple variables and uses the difference between "bivariate vs. multivariate" as the evaluation signal—essentially turning an originally standalone metric into a "self-contrastive" metric.
- **vs. Textor 2016 / Eulig 2025 etc. "GT-free Causal Evaluation"**: These are based on "consistency between estimated full graphs and data," requiring a full DAG as input; this paper targets the sparser and more realistic input form of "only pairwise," being one of the first works in this sub-direction.
- **vs. LLM-causal evaluation works (Kiciman 2024, Sheth 2025)**: Most of those works compare LLM accuracy on synthetic benchmarks with GT; this paper provides a method to rank LLMs on **GT-free** real data, which is closer to actual deployment conditions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using "marginalization inevitably increases confounding" as soft compatibility evidence and bypassing faithfulness to evaluate pairwise causality is a novel idea with theoretical backing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments include ablations across multiple parameter dimensions; LLM experiments have baselines, but are limited to the 7-variable gapminder dataset; cross-domain validation is lacking.
- Writing Quality: ⭐⭐⭐⭐ The chain of definitions-lemmas-theorems-experiments is clear; the counter-examples in Figure 1 are intuitive; the appendix holds the bulk of the work, but the main text is self-consistent.
- Value: ⭐⭐⭐⭐ Provides one of the first deployable, zero-annotation tools for the growing problem of "evaluating causal statements from LLM outputs," useful for both Causal Inference and LLM communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Evaluating Counterfactual Strategic Reasoning in Large Language Models](../../ACL2026/causal_inference/evaluating_counterfactual_strategic_reasoning_in_large_language_models.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ICML 2026\] Formalizing and Falsifying Causal Pathways of Rare Events](formalizing_and_falsifying_causal_pathways_of_rare_events.md)
- [\[ICML 2026\] Towards a Holistic Understanding of Selection Bias for Causal Effect Identification](towards_a_holistic_understanding_of_selection_bias_for_causal_effect_identificat.md)
- [\[ICML 2026\] Tailoring Strictly Proper Scoring Rules for Downstream Tasks: An Application to Causal Inference](tailoring_strictly_proper_scoring_rules_for_downstream_tasks_an_application_to_c.md)

</div>

<!-- RELATED:END -->
