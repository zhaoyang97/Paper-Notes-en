---
title: >-
  [Paper Note] Faithful Group Shapley Value
description: >-
  [NeurIPS 2025][Data Shapley] This paper proposes the Faithful Group Shapley Value (FGSV), the unique group-level data valuation method satisfying five axioms including "faithfulness…
tags:
  - "NeurIPS 2025"
  - "Data Shapley"
  - "Group-Level Data Valuation"
  - "Faithfulness Axiom"
  - "Shell Company Attack"
  - "Copyright Attribution"
  - "Explainable AI"
date: 2026-05-08
content_hash: 399590c97137377d
---

# Faithful Group Shapley Value

**Conference**: NeurIPS 2025
**arXiv**: [2505.19013](https://arxiv.org/abs/2505.19013)  
**Code**: [KiljaeL/Faithful_GSV](https://github.com/KiljaeL/Faithful_GSV)  
**Area**: Others (Data Valuation / Cooperative Game Theory)
**Keywords**: Data Shapley, Group-Level Data Valuation, Faithfulness Axiom, Shell Company Attack, Copyright Attribution, Explainable AI

## TL;DR

This paper proposes the Faithful Group Shapley Value (FGSV), the unique group-level data valuation method satisfying five axioms including "faithfulness," which effectively defends against the "shell company attack" (artificially inflating valuation by splitting subgroups), and introduces an efficient approximation algorithm with $O(n \cdot \text{Poly}(\log n))$ complexity.

## Background & Motivation

### Importance of Data Valuation

In modern machine learning, quantifying data value is critical for fair compensation and data market design. The Shapley value, a classical concept from cooperative game theory, is the unique valuation method satisfying four axioms—null player, symmetry, linearity, and efficiency—and is widely used in Data Shapley and feature attribution (SHAP).

### Demand for Group-Level Valuation

Many practical scenarios require valuing **collections of data** rather than individual data points:

- **Data markets**: Data owners contribute data as entire datasets
- **Copyright compensation**: Generative AI model training requires group-level compensation for copyright holders
- **Explainable AI**: Group-level feature importance is more robust and interpretable than individual-level
- **Computational efficiency**: Computing individual Shapley values on large datasets is prohibitively expensive; group-level valuation is faster

### Fatal Flaw of Existing Methods — Shell Company Attack

Existing Group Shapley Value (GSV) methods adopt the "Group-as-Individual" (GaI) strategy, treating each group as an atomic unit and applying the standard Shapley formula. However, this approach contains a serious fairness vulnerability:

**Shell Company Attack**: A malicious participant can split their data into multiple subgroups, thereby unfairly **inflating their own valuation** and **compressing others' valuations** without changing the actual data content.

The paper theoretically proves (Proposition 1) that when the utility function satisfies the "prudence condition" (i.e., the third-order difference $\Delta_s^3 \bar{U}(s) > 0$, corresponding to risk aversion in economics and diminishing marginal returns of data in ML), GSV necessarily satisfies:

$$\mathbb{E}[\text{GSV}(S_k)] < \mathbb{E}[\text{GSV}(S_k')] + \mathbb{E}[\text{GSV}(S_k'')]$$

meaning that splitting a group strictly increases its total valuation.

## Method

### Overall Architecture: Five Axioms and Uniqueness Theorem

The authors propose five axioms that **faithful group-level data valuation** should satisfy (Definition 2):

| Axiom | Meaning |
|-------|---------|
| Null Player | If every subset of group $S$ makes no marginal contribution to any coalition, then $\nu(S)=0$ |
| Symmetry | If members of two groups correspond one-to-one with equivalent contributions, their valuations are equal |
| Linearity | Linearity is preserved under linear combinations of utility functions |
| Efficiency | The sum of all group valuations equals the total utility $U([n])$ |
| **Faithfulness** | A group's valuation depends only on its own members' contributions, independent of how other groups are partitioned |

**Theorem 1 (Uniqueness)**: The unique group-level valuation method satisfying the above five axioms is:

$$\text{FGSV}(S_0) := \sum_{i \in S_0} \text{SV}(i)$$

where $\text{SV}(i)$ is the individual Shapley value. This means FGSV is simply the sum of individual Shapley values of group members.

### Key Designs: Efficient Approximation Algorithm

Directly computing FGSV requires combinatorially computing all individual Shapley values, which is extremely costly. The paper designs an efficient approximation algorithm (Algorithm 1) through a series of mathematical insights:

**Key Observation 1**: In the expansion of FGSV, the coefficient of the $U(S)$ term depends only on the triple $(s_1, s, s_0)$, where $s_0 = |S_0|$, $s = |S|$, $s_1 = |S_0 \cap S|$. Terms with identical coefficients can be aggregated.

**Key Observation 2**: The hypergeometric distribution $\mathbb{P}(\boldsymbol{s_1} = s_1)$ decays exponentially in $|s_1 - s \cdot s_0/n|$, meaning $\mathcal{T}(s)$ is dominated by values near the mean of $\boldsymbol{s_1}$.

**Key Observation 3** (Core): After approximate expansion, $\mathcal{T}(s) \approx s^{-1}(s_0/n)(1-s_0/n) \mu'(s_0/n)$, requiring only a derivative evaluation at a single point for efficient estimation.

**Theorem 2** formalizes these intuitions:

$$\mathcal{T}(s) = \frac{n}{n-1} \alpha_0 (1-\alpha_0) \left\{ \Delta\mu(s_1^*/s; s, s_0, n) + O(s^{-(1+\upsilon)}) \right\}$$

**Threshold-Based Strategy** (Algorithm 1):
- When $s < \bar{s}$ (small sets): directly estimate $\mathcal{T}(s)$ via Monte Carlo
- When $s \geq \bar{s}$ (large sets): apply the approximation formula from Theorem 2, estimating only $\Delta\mu$

**Paired Monte Carlo Variance Reduction**: Rather than independently estimating two $\mu$ values and subtracting, paired samples are used to directly estimate the difference (Equation 11), substantially reducing variance.

### Computational Complexity (Theorem 3)

For $O(1/s)$-deletion stable utility functions, Algorithm 1 achieves an $(\epsilon, \delta)$-approximation using only

$$O\left(n \cdot \max\left\{1, (\alpha_0(1-\alpha_0))^2 (\log n)^3\right\}\right)$$

utility function evaluations, where $\alpha_0 = s_0/n$. This is **one order of $n$ fewer** than the SOTA approach of computing all individual Shapley values and summing them ($O(\alpha_0 n^2 \text{Poly}(\log n))$).

### Handling Small-Coalition Utility Functions

For cases such as large models where utility is poorly defined on small datasets, the paper proposes injecting non-informative data points: when $|S| < B$, $B - |S|$ noise data points are added to bring the input set to threshold $B$, ensuring the utility function is well-defined for any subset size.

## Key Experimental Results

### Main Results: Approximation Accuracy in Sum-of-Unanimity Games

| Method | AUCC (↓ better) | ARE (↓ better) | Runtime per Iteration |
|--------|:---:|:---:|:---:|
| **FGSV (Ours)** | **Lowest** | **Lowest** | **Among fastest** |
| Permutation | Higher | Higher | Fast |
| Group Testing | High | High | Slow (with internal optimization) |
| Complementary Contribution | Medium | Medium | Medium |
| One-for-All | Medium | Medium | Fast |
| KernelSHAP | High | High | Slow |
| Unbiased KernelSHAP | High | High | Slow |
| LeverageSHAP | Medium | Medium | Medium |

Under $n \in \{64, 128, 256\}$ with a fixed budget of 20,000 utility evaluations, FGSV achieves the best AUCC and ARE across all problem scales.

### Ablation Study / Application Experiments

**Copyright Attribution (Stable Diffusion LoRA fine-tuning + FlickrLogo-27)**:

| Scenario | SRS (GSV-based) | FSRS (FGSV-based) |
|----------|:---:|:---:|
| No attack (30 images per brand, single group) | Reasonable allocation | Reasonable allocation |
| Shell Company Attack (Google/Sprite split into 20/10 subgroups) | Google/Sprite valuations inflate, others compressed | **Valuations remain stable, unaffected by attack** |

Key finding: Under the prompt "A logo by Vodafone," the total share of Google/Sprite under the attacked SRS surpasses that of Vodafone itself; FSRS consistently produces reasonable copyright attributions.

**Explainable AI (Diabetes dataset + Ridge Regression)**:

| Method | Stability Across Grouping Schemes |
|--------|:---:|
| GSV | Category rankings inconsistent across grouping strategies (e.g., "young"/"middle-aged"/"elderly" each ranks first under at least one scheme) |
| **FGSV** | **Rankings consistent across all grouping schemes** ("young" group consistently ranks highest) |

### Key Findings

1. FGSV achieves faster convergence and higher final accuracy than all 7 baseline methods under a fixed computational budget
2. FGSV effectively defends against the shell company attack, whereas GSV is completely compromised
3. FGSV provides consistent and interpretable group-level valuations across different grouping schemes

## Highlights & Insights

1. **Value of problem identification**: The shell company attack vulnerability in GSV is identified as a novel, practically significant security threat in group-level data valuation
2. **Elegant theoretical result**: The uniqueness theorem (Theorem 1) is concise and powerful—the unique method satisfying five axioms is exactly the sum of individual Shapley values, requiring no additional complex construction
3. **Mathematically driven algorithm design**: By exploiting the concentration property of the hypergeometric distribution and smoothness of the $\mu$ function, complexity is reduced from $O(n^2)$ to $O(n \cdot \text{Poly}(\log n))$, tightly coupling theory and practice
4. **Paired Monte Carlo variance reduction**: Directly estimating the difference rather than subtracting independent estimates is seemingly simple but yields significant precision gains
5. **Strong practical relevance**: Both application scenarios—copyright attribution and explainable AI—are realistic and important

## Limitations & Future Work

1. **Defends only against shell company attacks**: No direct defense against copier attacks (duplicating high-value data); the paper only discusses ad-hoc solutions such as preprocessing deduplication
2. **Utility function assumptions**: Assumption 2 (second-order algorithmic stability) is verified under SGD and influence functions, but its validity for more complex models (LLMs, diffusion models, etc.) remains unclear
3. **Small-coalition utility handling**: While the noise injection strategy has principled motivation, the sensitivity of results to the choice of threshold $B$ and noise distribution is insufficiently discussed
4. **Limited experimental scale**: The SOU experiments reach at most $n=256$, and the copyright attribution experiment involves only 4 brands with 30 images each, making it difficult to assess performance in real large-scale scenarios
5. **Single-group estimation**: Each invocation of Algorithm 1 estimates FGSV for only one group; while still superior to baselines for multi-group estimation, this requires the number of groups to be $o(n)$

## Related Work & Insights

- **Relationship to Data Shapley line of work**: FGSV builds on Data Shapley (Ghorbani & Zou, 2019), Beta Shapley (Kwon & Zou, 2022), and related works, but is the first to systematically study faithfulness in the group-level setting
- **Connection to KernelSHAP**: KernelSHAP and its variants (Unbiased KernelSHAP, LeverageSHAP) participate as individual-level estimation baselines; FGSV avoids error accumulation by directly targeting the group-level objective
- **Copyright AI direction**: The FSRS improvement over the Shapley Royalty Share (Wang et al., 2024) can be directly applied to copyright compensation in generative AI
- **Methodological insight**: The paradigm of "first prove uniqueness, then design efficient algorithms" is worth emulating; the discovery of the shell company attack also alerts data market designers to game-theoretic robustness concerns

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|:---:|-------|
| Novelty | ⭐⭐⭐⭐ | New vulnerability + new axioms + new algorithm, organically integrated |
| Theoretical Depth | ⭐⭐⭐⭐⭐ | Uniqueness theorem, complexity analysis, stability verification—mathematically rigorous |
| Experimental Thoroughness | ⭐⭐⭐ | Application scenarios are convincing but limited in scale |
| Practical Value | ⭐⭐⭐⭐ | Direct applicability to data markets and copyright attribution |
| Writing Quality | ⭐⭐⭐⭐ | Clear structure, rigorous mathematical derivations, intuitive illustrations |
| **Overall** | **⭐⭐⭐⭐** | Excellent theory-driven work with a coherent arc from problem definition to solution |

## Related Work & Insights (Detailed Comparison)

| Method | Valuation Level | Faithfulness | Shell Company Attack Defense | Complexity | Core Limitation |
|--------|:---:|:---:|:---:|:---:|------|
| Data Shapley (Ghorbani & Zou, 2019) | Individual | N/A | N/A | $O(n \epsilon^{-2} \log n)$ | Cannot be directly applied to group-level settings |
| Group Shapley / GaI (Jullum et al., 2021; Wang et al., 2024) | Group (atomic) | ✗ | ✗ | $O(K!)$ ($K$ = number of groups) | Sensitive to grouping strategy; vulnerable to attack |
| KernelSHAP / Unbiased KernelSHAP | Individual → Sum | ✓ (indirect) | ✓ (indirect) | High (large internal optimization overhead) | Error accumulates when used for group estimation |
| Beta Shapley (Kwon & Zou, 2022) | Individual | N/A | N/A | Similar to Data Shapley | Downweights small-coalition terms; does not satisfy Efficiency axiom |
| **FGSV (Ours)** | **Group (direct)** | **✓** | **✓** | $O(n \cdot \text{Poly}(\log n))$ | Defends only shell company attack, not copier attack |

- **Fundamental distinction between GSV and FGSV**: GSV treats groups as atomic units and applies the individual Shapley formula, ignoring within-group member information; FGSV is defined as the sum of individual Shapley values of group members, preserving traceability of individual contributions. This makes FGSV invariant to arbitrary repartitioning of other groups (Faithfulness axiom).
- **Relationship to the KernelSHAP family**: KernelSHAP and its variants are fundamentally approximation algorithms for individual-level Shapley values. Using them to estimate each member's SV and summing to obtain FGSV theoretically requires $O(\alpha_0 n^2 \text{Poly}(\log n))$ utility evaluations (with errors amplified by $\sqrt{s_0}$), which is one order of $n$ slower than Algorithm 1.
- **Relationship to Shapley Royalty Share (SRS)**: SRS is the copyright compensation share proposed by Wang et al. (2024) based on GSV. This paper improves it to FSRS (replacing GSV with FGSV), defending against the shell company attack while maintaining a normalized allocation.

## Further Insights and Connections

- **"Prove uniqueness first, then design algorithms" research paradigm**: This paper first axiomatically proves FGSV is the unique solution, then leverages the mathematical structure to design efficient approximation algorithms. This paradigm offers methodological inspiration for subsequent work in data valuation and mechanism design.
- **Warning to data markets about shell company attacks**: As AI data markets become operational, the GSV vulnerability may be practically exploited—data vendors could register multiple subsidiaries to inflate valuations. FGSV provides a theoretically complete defense.
- **Complementary challenge of copier attacks**: While shell company attacks involve "splitting oneself" for gain, copier attacks involve "duplicating others." The paper notes that FGSV does not directly defend against the latter, though deduplication preprocessing or do-utility designs may help. A unified defense against both attack types remains an open problem.
- **Application of hypergeometric concentration in valuation**: Key Observation 2 exploits the exponential concentration of the hypergeometric distribution to reduce the number of terms requiring estimation—a technique generalizable to other combinatorial estimation problems involving subset sampling.
- **Small-coalition utility padding strategy**: For models such as LLMs that cannot be effectively trained on small data, the strategy of injecting non-informative data (inspired by machine unlearning) is worth borrowing in other cooperative game settings requiring handling of "small coalitions."
- **Possible extensions**: (1) FGSV updates in dynamic grouping scenarios; (2) introducing the faithfulness concept into contribution evaluation in federated learning; (3) group-level copyright attribution for training data of large-scale generative models such as LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] FACE: Faithful Automatic Concept Extraction](face_faithful_automatic_concept_extraction.md)
- [\[NeurIPS 2025\] A Unified Framework for Provably Efficient Algorithms to Estimate Shapley Values](a_unified_framework_for_provably_efficient_algorithms_to_estimate_shapley_values.md)
- [\[ICML 2026\] On the Coordination of Value-Maximizing Bidders](../../ICML2026/others/on_the_coordination_of_value-maximizing_bidders.md)
- [\[AAAI 2026\] HyperSHAP: Shapley Values and Interactions for Explaining Hyperparameter Optimization](../../AAAI2026/others/hypershap_shapley_values_and_interactions_for_explaining_hyperparameter_optimiza.md)
- [\[ICML 2026\] Sequential Group Composition: A Window into the Mechanics of Deep Learning](../../ICML2026/others/sequential_group_composition_a_window_into_the_mechanics_of_deep_learning.md)

</div>

<!-- RELATED:END -->
