---
title: >-
  [Paper Note] Evaluating Bivariate Causal Statements Based on Mutual Compatibility
description: >-
  [ICML 2026][Causal Inference][Paper Note] This paper addresses scenarios where "only pairwise (bivariate) causal statements are available without ground truth." It proposes two compatibility scores that do not rely on faithfulness: `comp` for linear cases and `incomp` for graph structures. By determining whether the multivariate model formed by stitching these
tags:
  - ICML 2026
  - Causal Inference
date: 2026-05-08
content_hash: b848ce0a75c73da1
---
# Evaluating Bivariate Causal Statements Based on Mutual Compatibility

**Conference**: ICML 2026  
**arXiv**: [2606.00278](https://arxiv.org/abs/2606.00278)  
**Code**: https://github.com/ejahn17/compatibility-scores  
**Area**: Causal Inference / Causal Discovery Evaluation / LLM Causal Reasoning  
**Keywords**: Bivariate causal statements, compatibility scores, confounding backdoors, LLM causal evaluation  

## TL;DR
This paper addresses scenarios where "only pairwise (bivariate) causal statements are available without ground truth." It proposes two compatibility scores that do not rely on faithfulness: `comp` for linear cases and `incomp` for graph structures. By determining whether the multivariate model formed by stitching these pairwise statements requires "anomalous extra confounding" to explain the observed covariance, the method identifies incorrect causal claims and uses it to score LLM causal outputs.

## Background & Motivation
**Background**: Causal inference in fields such as medicine and economics has long relied on pairwise cause-effect statements provided by experts; recently, many works have turned to letting LLMs directly generate such pairwise causal relationships. Both paths lack a mechanism to detect errors—traditional causal discovery algorithms have consistency guarantees under strong assumptions, while LLMs possess neither training objectives nor theoretical constraints.

**Limitations of Prior Work**: In real-world scenarios lacking ground truth, it is virtually impossible to verify the accuracy of a list of "$X_i \to X_j, \alpha_{ij}$". Existing "no-GT evaluation" works (Textor 2016, Eulig 2025, Sheth 2026) mostly measure the "consistency between the estimated graph and the data," but only for full DAG estimation. Faller 2024 uses "whether multiple subset graphs can be stitched into the same large graph" for compatibility, but this requires inputs as subgraphs and is not directly applicable to "pairwise statements," the most common output format for humans and LLMs.

**Key Challenge**: For linear, acyclic pairwise statements $\{\alpha_{ij}\}$, the authors first prove (Lemma 2.3) that any such list **uniquely** induces a multivariate linear SEM that perfectly fits the observed covariance. Therefore, in the sense of Faller's "hard compatibility," there is no conflict—meaning no exploitable constraints. To perform evaluation, one must approach from a "soft" perspective (plausibility).

**Goal**: (1) Design a continuous compatibility score for linear pairwise statements that does not depend on faithfulness; (2) design an incompatibility count for graph-structured statements that only specify the presence or absence of causality/confounding; (3) verify that both can distinguish correct from incorrect claims on synthetic data and LLM outputs.

**Key Insight**: The authors propose a **Confounding Postulate**—a "generic" multivariate causal model should not have **more** unobserved confounding than its pairwise marginal models. Intuitively, when marginalizing a multivariate model to pairs, previously observed backdoor paths (e.g., $X_i \leftarrow X_k \to X_j$) become unobserved confounding; thus, the confounding in the marginal model should only increase. Unless the induced multivariate model deliberately "tunes observed causal paths to exactly cancel" its own confounding, this fine-tuning is extremely rare in random models (Theorem 2.9 provides a guarantee in expectation).

**Core Idea**: Use the difference between the "confounding of the induced multivariate SEM" and the "confounding of the pairwise marginal SEM" as a compatibility score. If the multivariate model instead "requires less" confounding to explain covariance, it implies the statements require a counter-intuitive fine-tuned cancellation and can be judged as untrustworthy.

## Method

### Overall Architecture
The method addresses how to judge the truth of pairwise causal statements in the absence of ground truth. The core approach is to stitch these pairwise statements into a multivariate causal model and use a physical intuition—that marginalization only swallows observed backdoors and thus increases confounding—to test whether the statements "anomalously require less confounding." In linear cases, the input consists of the covariance matrix $\Sigma$ (or an empirical estimate $\hat{\Sigma}$) and the coefficients $\alpha_{ij}$ for each pair, outputting a continuous `comp` score; in graph-structured cases, the input consists of qualitative ADMGs for each pair, outputting a count of consistency violations `incomp`. Statements are judged untrustworthy if `comp < 0` or `incomp > 0`.

### Key Designs

**1. Linear comp score: Measuring how anomalous a statement is via "Marginal Confounding − Multivariate Confounding"**

The limitation for linear statements is proved in Lemma 2.3: "any set of pairwise coefficients uniquely induces a multivariate SEM that perfectly fits $\Sigma$," so hard compatibility provides no conflict to exploit. The authors arrange pairwise coefficients into a unit lower triangular matrix $A$ ($A_{ji} = \alpha_{ij}, i < j$), define $\Gamma = I - A^{-1}$ to obtain the unique multivariate SEM $X = \Gamma X + N$, and then compare the "squared covariance not explained by causal effects" at two levels of granularity. The pairwise SEM $X_j = \alpha_{ij}X_i + \tilde N_{ij}$ gives $C^{biv}_{ij} = (\Sigma_{ij} - \alpha_{ij}\Sigma_{ii})^2$. The multivariate SEM, following Wright's path tracing, subtracts all observed directed paths and full backdoor paths, yielding:

$$C^{mult}_{ij}(\Sigma,\Gamma) = \Big(\Sigma_{ij} - \sum_{k\le i}\Sigma_{kk}\sum_{P_1: k\rightsquigarrow i,\,P_2: k\rightsquigarrow j} \Gamma_{P_1}\Gamma_{P_2}\Big)^2,$$

The final score is the sum of the differences: $\text{comp}(\Sigma,A) = \sum_{i<j} C^{biv}_{ij} - C^{mult}_{ij}$. The intuition is that marginalizing a multivariate model turns originally observed backdoor paths into noise (confounding); thus, a "plausible" model should satisfy $C^{biv} \ge C^{mult}$. A negative score `comp < 0` is equivalent to violating the confounding postulate (Assumption 2.4). This design inherits several properties: it reduces to the classic Janzing-Schölkopf confounding measure in the bivariate case; Theorem 2.9 proves that for random SEMs under "zero-mean coefficients + independent mechanisms + non-degeneracy," true statements expect `comp > 0` (weaker than faithfulness); and Theorem 2.10 shows that the estimation error can be controlled with $N = O(1/\varepsilon^2 \cdot \log(n/\delta))$ samples.

**2. Graph structural incomp score: Minimum edit distance to "stitchable" graphs**

When statements are purely qualitative (presence of causality or confounding), no coefficients are available. Instead, the authors score based on the "number of violations of global consistency," upgrading hard compatibility (yes/no) to a continuous metric. They take the union of all $\binom{n}{2}$ two-node ADMGs to form a statement graph $G$. Lemma 3.5 provides necessary and sufficient conditions for $G$ to be marginalized from some large ADMG: (i) the directed part is acyclic; (ii) the directed part is transitively closed—if there is a path $X_i \rightsquigarrow X_j$ in $G$, there must be a direct edge $X_i \to X_j$; (iii) any path with "double arrowhead endpoints and no double arrowheads in the middle" (confounding path) must result in a bidirected edge $X_i \leftrightarrow X_j$. Thus, $\text{incomp}(G) = \min_{G^*} d(G, G^*)$, where $d$ is the mixed graph Hamming distance and $G^*$ ranges over all graphs satisfying (i)-(iii).

**3. Finite-sample estimation and LLM evaluation protocol: Applying scores to real data**

To make the method practical, the authors propose a complete workflow from samples to LLM evaluation. Since `comp` values depend on variable scaling, variables are standardized to unit variance (equivalent to using the correlation matrix), and the statement coefficients $A$ are transformed accordingly. In LLM evaluation (using 7 country-level indicators from Gapminder), variable descriptions and the empirical correlation matrix are provided as prompts. LLMs are asked to output a causal ordering and pairwise linear coefficients. Results are averaged over 15 runs and compared against a random baseline where coefficients are sampled from a zero-mean Gaussian matched to the LLM output's variance. This protocol identifies that high-capability LLMs tend to have higher `comp` scores, while many weak LLMs receive negative scores, falsifying their outputs.

### Loss & Training
The method is **purely for evaluation and involves no training**. `comp` is a closed-form function of the covariance and coefficient matrices; `incomp` is a combinatorial optimization problem (solvable via enumeration or heuristics for small $n$).

## Key Experimental Results

### Main Results

**`comp` capability to distinguish truth from error on synthetic data** (Figure 2): True values were sampled from random linear Gaussian SEMs, and Gaussian noise $\sigma$ was added to the true coefficients to simulate "decreasing quality of statements."

| Setting | Proportion of positive scores for truth ($\sigma=0$) | Trend as $\sigma$ increases | Key Observation |
|------|--------|--------|------|
| $n=10, p=0.5$, varying $m$ (latents: 0/1/3/5) | ≈ 95%+ | Monotonic decrease | Robust to number of latent variables; curves almost overlap. |
| $n=10, m=3$, varying $p$ (sparsity: 0.3/0.5/0.7/0.9) | ≈ 95%+ | Monotonic decrease | Higher $p$ (denser graphs) leads to stronger discriminative power. |
| $p=0.5, m=3$, varying $n$ (variables: 5/10/15/20) | ≈ 95%+ | Monotonic decrease | Higher $n$ leads to stronger discriminative power. |

**LLM Causal Statement Evaluation** (Figure 4, Gapminder 7 variables):

| Model Group | `comp` Score (Mean of 15, relative to random baseline) | Conclusion |
|------|------|------|
| High-capability LLMs | Significantly > 0 and higher than random baseline | Passed falsification |
| Medium LLMs | Close to random baseline | Indeterminate |
| Low-capability LLMs | **Negative scores** | Falsified by `comp`; causal statements are untrustworthy |

### Ablation Study

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| `comp` using true $\Sigma$ vs $\hat\Sigma$ (Figure 3) | Sign consistency > 90% for $N \ge 100$ | Empirical convergence is much faster than the worst-case theoretical bound. |
| Latent variables $m \in \{0,1,3,5\}$ | Positive score curves nearly overlap | Method is robust to unobserved confounding. |
| Sparsity $p$ from 0.3 to 0.9 | True positive rate constant; false positive rate drops sharply | Denser graphs $\to$ more backdoor paths $\to$ larger difference between marginal vs multivariate confounding $\to$ easier to catch errors. |
| Random baseline LLM | Slightly above 0 | Even random guessing can sometimes avoid being "more wrong than negative." |

### Key Findings
- **Core Conclusion**: On synthetic data, `comp` almost never misses true statements (≥ 95% positive scores) and stably distinguishes noisy false statements; sample complexity in practice is far better than theoretical bounds.
- **Dense graphs are more favorable**: In cases with large $n$ or large $p$, there are more observed backdoor paths in the multivariate SEM. More information is lost during marginalization, increasing the discriminative power of `comp`—this is consistent with the mechanism of the confounding postulate.
- **LLM Application Insight**: Higher-capability LLMs achieve higher `comp` on Gapminder, while several lower-capability LLMs receive **negative scores** and perform worse than the random baseline, indicating this method can serve as a "causal capability health check" for LLMs without ground truth.

## Highlights & Insights
- **Bypassing faithfulness is the primary merit**: Most consistency guarantees for traditional causal discovery rely on faithfulness. Theorem 2.9 in this paper guarantees that true statements expect `comp > 0` under weaker "randomness" assumptions, which is highly attractive for real-world scenarios (medicine, social sciences) where faithfulness is questionable.
- **"Marginalization necessarily increases confounding" as a prior**: Using this physical intuition instead of faithfulness, and treating "models requiring fine-tuned cancellation" as anomalies, elegantly translates causal identifiability into a parameter-space rarity problem. This can be transferred to other tasks like federated causal discovery or cross-modal alignment verification.
- **Operational template for evaluating LLM causal outputs**: While many LLM-causal studies are limited to synthetic benchmarks with GT, this paper provides a zero-annotation workflow (correlation matrix + coefficients + comp), which is a concrete metric for the LLM evaluation community, especially for scientific discovery or medical diagnostic tasks.

## Limitations & Future Work
- **Authors' Acknowledgments**: (1) A positive score can still be wrong (a false statement might happen to induce a "plausible" model), so `comp` can only **falsify**, not **verify**; (2) Linear, acyclic, and Gaussian assumptions are restrictive; (3) `comp` values depend on variable scaling and require standardization.
- **Observed Limitations**: (a) Theorem 2.9 assumes zero-mean coefficients, which may not hold in fields with biased distributions (e.g., in medicine, "treatment $\to$ recovery" is usually positive); (b) experiments were limited to $n=20$, and the computational cost of path tracing for $n=100+$ was not discussed; (c) the minimization in `incomp` is a combinatorial problem with no heuristic provided for large $n$.
- **Future Directions**: Generalizing `comp` to non-linear SEMs (using conditional means or kernel covariance); introducing "score decay" to distinguish slight from severe errors; and combining `comp` with existing LLM methods (e.g., agent self-consistency) as a reward signal for training or filtering.

## Related Work & Insights
- **vs Faller et al. (2024)**: They require "multiple subset ADMGs" and perform hard compatibility checks; this paper requires only "$\binom{n}{2}$ pairwise statements" and provides a **soft** score, covering the most natural output format and extending hard compatibility to a continuous violation count.
- **vs Janzing & Schölkopf (2018) Bivariate Confounding Measure**: They define unexplained squared covariance only for the two-variable case; this paper generalizes this using Wright's path tracing to multivariate cases, using the "bivariate vs multivariate" difference as the evaluation signal.
- **vs Textor 2016 / Eulig 2025 "No-GT Causal Evaluation"**: These rely on "consistency between the estimated full graph and data," requiring a complete DAG; this paper focuses on the more sparse and realistic "pairwise only" input format.
- **vs LLM-causal Evaluation (Kiciman 2024, Sheth 2025)**: Most of those works compare LLM accuracy on synthetic benchmarks with GT; this paper provides a way to rank LLMs on real data without GT, which is closer to actual deployment conditions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using "marginalization necessarily increases confounding" as soft compatibility evidence and bypassing faithfulness to evaluate pairwise causality is an innovative approach with theoretical backing.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic experiments are ablated across multiple dimensions. LLM experiments have baselines but are limited to the 7-variable Gapminder dataset; cross-domain validation is needed.
- Writing Quality: ⭐⭐⭐⭐ Clarity in the chain of definitions-lemmas-theorems-experiments; Figure 1 provides an intuitive counter-example.
- Value: ⭐⭐⭐⭐ Provides one of the first actionable, zero-annotation tools for the rising problem of "evaluating LLM causal claims," beneficial to both the causal inference and LLM communities.

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
