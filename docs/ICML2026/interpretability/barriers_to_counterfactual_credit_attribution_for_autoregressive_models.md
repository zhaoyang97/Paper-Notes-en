---
title: >-
  [Paper Note] Barriers to Counterfactual Credit Attribution for Autoregressive Models
description: >-
  [ICML 2026][Interpretability][Counterfactual Credit Attribution] This paper formalizes the problem of "Counterfactual Credit Attribution (CCA)" for generative models in RAG/in-context deployment…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Counterfactual Credit Attribution"
  - "Differential Privacy"
  - "Autoregressive Models"
  - "RAG"
  - "Infeasibility Lower Bound"
date: 2026-05-08
content_hash: 107c5c504a4963c4
---

# Barriers to Counterfactual Credit Attribution for Autoregressive Models

**Conference**: ICML 2026  
**arXiv**: [2605.01425](https://arxiv.org/abs/2605.01425)  
**Code**: Not yet released (theoretical paper)  
**Area**: AI Safety / Theory / Generative Model Copyright  
**Keywords**: Counterfactual Credit Attribution, Differential Privacy, Autoregressive Models, RAG, Infeasibility Lower Bound

## TL;DR
This paper formalizes the problem of "Counterfactual Credit Attribution (CCA)" for generative models in RAG/in-context deployment, and proves two surprising negative results: (1) Even if the underlying next-token predictor is (0,0)-CCA, the autoregressive rollout is not CCA—CCA does not naturally compose under autoregression as DP does; (2) Black-box "CCA retrofitting" of a deployed non-attributing model requires at least exponential (in output length $\ell$) number of queries.

## Background & Motivation

**Background**: Generative AI has fundamentally disrupted the principle that "creators must credit prior works"—RAG/in-context learning obscures the causal chain between final outputs and specific sources behind a model black box. In academic writing, this is academic misconduct; in law (copyright), it concerns licensing and commercial boundaries. Livni-Moran-Nissim-Pabbaraju (2024) proposed **Counterfactual Credit Attribution (CCA)** as a relaxation of differential privacy (DP): the algorithm $\tilde A(S)$ outputs both the main result $y$ and a credited subset $C\subseteq S$; the requirement is that the distribution of "uncredited input" is close to the distribution where that input is entirely removed, i.e., $\tilde A^{-i}(S)\approx_{\varepsilon,\delta}\tilde A_{-i}(S)$. The original work only studied CCA in the PAC learning setting, leaving generative models as an open problem.

**Limitations of Prior Work**: Directly applying CCA to LLMs, there are two natural engineering paths—(a) design a CCA next-token predictor and run it autoregressively, hoping to "automatically synthesize" CCA for the whole output; (b) take an existing non-attributing LLM and wrap it to automatically supplement the credit set—are these feasible? No systematic answer exists. This contrasts sharply with DP-LLM: DP at the token level composes almost for free (Majmudar 2022, Amin 2024).

**Key Challenge**: DP enjoys a clean composition theorem under sequential composition ($k$-fold composition at worst $k\varepsilon$). CCA resembles DP, but its core is the **conditional distribution** of "uncredited input," which is amplified by the autoregressive product chain; CCA appears similar to DP but does not compose. The retrofitting path faces the same issue: once the model decides the output, backtracking "who to credit" is equivalent to estimating how much mass sensitive data contributed on each trajectory, which is exponentially hard for a black-box model.

**Goal**: To deploy CCA on deployment-time data (RAG database, in-context examples), systematically test the above engineering intuitions, and provide rigorous (in)feasibility proofs.

**Key Insight**: The authors use a "counterexample first, lower bound second" strategy—first constructing a 2-token, single-document toy counterexample (extremely compact and easy to reproduce), then generalizing it into a parameterized lower bound theorem; for retrofitting, they construct a family of "almost identical, only slightly 1-bit biased" hard models, transferring the difficulty of finding the hidden identifier $\mathbf{z}$ to the retrofitter.

**Core Idea**: Strictly distinguish CCA from "DP relaxation"—prove that it is **infeasible** on both the most natural engineering paths: autoregressive composition and black-box retrofitting, thus clarifying the boundaries for future CCA-LLM research.

## Method

### Overall Architecture
The paper has two main threads. **First (§4)**: studies the industrial path of "imposing CCA on next-token predictor $\tilde M$ + autoregressive rollout," providing a counterexample (Thm 4.2) and a general lower bound (Thm 4.3). **Second (§5)**: studies the retrofitting path of "black-box modification of a non-attributing $M$," constructing a family of hard models $\{M_\mathbf{z}\}_{\mathbf{z}\in\{0,1\}^\ell}$, and proving that any $\alpha<1/2$ approximate retrofit requires at least $\widetilde\Omega(2^\ell)$ oracle queries in the worst case (Thm 5.5). All proofs are based on rigorous LP characterization and information-theoretic query lower bounds.

### Key Designs

1. **Counterexample and General Lower Bound for Non-Compositionality of CCA (Thm 4.2 & 4.3)**:

    - Function: Disproves the intuition that "$\tilde M$ is $(0,0)$-CCA $\Rightarrow$ rollout $G^{\tilde M}$ is $(\varepsilon,\delta)$-CCA".
    - Mechanism: Take dataset $\mathcal S=\{s_1\}$, vocabulary $\mathcal X=\{\mathtt a,\mathtt b\}$, define a $(0,0)$-CCA next-token predictor: under empty prefix, $\tilde M(\{s_1\},\lambda)=\tilde M(\emptyset,\lambda)$ (outputs $\mathtt a$ with $p$, $\mathtt b$ with $1-p$, never credits); under prefix $\mathtt a$, only for $S=\{s_1\}$, credits with probability $1/2$, otherwise outputs $\mathtt b$ without credit (the conditional non-credit distribution matches $S{=}\emptyset$, so token-level $(0,0)$-CCA holds); under prefix $\mathtt b$, always credits and outputs $\mathtt a$ (the condition triggers the 1.0 term, bypassing CCA constraint). However, after rollout: under the non-credit condition, $G^{\tilde M}(S^{-1},\lambda)$ must output $(\mathtt{ab},\emptyset)$; the counterfactual $G^{\tilde M}(S_{-1},\lambda)$ outputs $(\mathtt{ab},\emptyset)$ only with probability $p$. When $p<e^{-\varepsilon}(1-\delta)$, $(\varepsilon,\delta)$-CCA is violated. The general lower bound (Thm 4.3) abstracts this as $\varepsilon'\geq\ln\big(\prod_j\Pr[E_j\mid\cdots]/\Pr[s_i\notin C]\big)-|\mathbf x^{-i}|\cdot\varepsilon$, where $E_j$ is the event of not crediting at step $j$.
    - Design Motivation: DP composes because the "conditional distribution unaffected by $s_i$" is consistent at both token and sequence levels; the key CCA quantity $\Pr[s_i\notin C]$ shrinks multiplicatively along the chain, amplifying the conditional distribution ratio—this is the counterintuitive source of "$\varepsilon\to 0$ makes the lower bound larger" (note $\varepsilon$ is endogenous and model-coupled). The minimality of the counterexample ensures it blocks almost all natural CCA designs.

2. **Hard Model Family $\mathcal M_\ell$ for CCA Retrofitting (Hard Instance for Thm 5.5)**:

    - Function: Construct a family $\{M_\mathbf{z}\}$ such that "identifying $\mathbf z$" is an implicit subproblem of retrofitting and is exponentially hard for the original model oracle.
    - Mechanism: $\mathcal X=\{0,1,\bot\}$, $\mathcal S=\{s_1\}$, $\ell\geq 1$, $\gamma\in(0,1)$, $\varepsilon\geq 0$. $M_\mathbf z(S,\mathbf x)$ is almost everywhere $\mathsf{Bern}(1/2)$ for $|\mathbf x|\leq\ell$, **except** when $S\neq\emptyset$ and the prefix is exactly $\mathbf z$, where the probability of 1 is biased to $\tfrac12+\tfrac{1-e^{-\varepsilon}(1-\gamma)}{2}$; output length is always $\ell+1$. Under the oracle model, $M_\mathbf z$ and $M_\emptyset$ have TV distance $\leq 2^{-\ell}$ on every prompt (Remark 5.6.1); finding $\mathbf z$ is equivalent to a needle-in-haystack search over $\{0,1\}^\ell$ $\Rightarrow$ Lemma 5.8: $\Omega(2^\ell)$ query lower bound.
    - Design Motivation: To lower bound the "computational power" of retrofitting, information must be hidden in oracle-inaccessible locations; this model family encodes the statistical difficulty of "distinguishing two nearly identical distributions" as a secret $\ell$-bit string, with tunable bias to ensure that under $\varepsilon$-CCA, there remains an irreducible $\gamma$ probability of credit.

3. **LP Characterization of Optimal CCA Augmentation (Lemma 5.6) + Reduction from Finding $\mathbf z$ to Retrofitting (Lemma 5.9)**:

    - Function: Any "oracle for approximately optimal CCA augmentation" can be inverted into an algorithm that finds $\mathbf z$ in $O(\ell\log\ell)$ queries, implying that the retrofitter must make at least $\widetilde\Omega(2^\ell/\ell\log\ell)$ queries to the original model.
    - Mechanism: Formalize "optimal CCA augmentation" as an LP: variables are probabilities over $(S,\mathbf x,y,C)$, constraints are augmentation (marginals match $G^M$) + CCA (non-credit conditional distribution $\varepsilon$-close to counterfactual); objective is to minimize $\mathbb{E}[f(C)]$. The LP solution shows: only when the prefix $\mathbf x\sqsubseteq\mathbf z$ does $\Pr[\tilde G^*_\mathbf z(S^*,\mathbf x)\text{ credits }s_1]=\gamma$, otherwise 0. This "crediting probability is constant $\gamma$ on the prefix tree of $\mathbf z$, zero elsewhere" structure is equivalent to "binary search along the prefix tree to locate $\mathbf z$"—requiring only $O(\ell\log\ell/(\gamma-2\alpha)^2)$ retrofit queries to solve for $\mathbf z$.
    - Design Motivation: This treats the "structure of the optimal solution" as an attack surface: if the retrofit must be optimal (even $\alpha$-approximate), its crediting probability at each prefix "leaks" $\mathbf z$. In other words, **the better the retrofit, the easier it is to invert, allowing an attacker to extract secrets inaccessible to the original model oracle**—an elegant "computational hardness via solution-structure leakage".

### Loss & Training
This is a purely theoretical paper with no training or loss functions. The main contribution is a series of rigorous (in)feasibility proofs.

## Key Experimental Results

### Main Results
The paper contains no empirical experiments; the following "theoretical results" substitute for the main results table.

| Proposition | Setting | Conclusion |
|-------------|---------|------------|
| Thm 4.2 | $\forall\varepsilon\geq 0,\delta<1$ | There exists a $(0,0)$-CCA next-token predictor whose rollout is not $(\varepsilon,\delta)$-CCA |
| Thm 4.3 | $\tilde M$ $\varepsilon$-CCA, rollout $\varepsilon'$-CCA | $\varepsilon'\geq\max\big[\ln(\prod_j\Pr[E_j]/\Pr[s_i\notin C])-\|\mathbf{x}^{-i}\|\cdot\varepsilon\big]$ |
| Thm 5.5 | $\alpha<1/2$, $\delta=0$ | $\alpha$-approximate retrofit requires at least $\Omega(2^\ell/\ell\log\ell)$ oracle queries in the worst case $M\in\mathcal M_\ell$ |
| Remark 5.6.1 | $\mathcal M_\ell$ | Data affects model TV by $\leq 2^{-\ell}$, yet is still required to be credited with probability $\gamma>0$ |

### Ablation Study
The paper uses "case analysis tightening" in place of ablation:

| Condition | Result | Explanation |
|-----------|--------|-------------|
| $\varepsilon\to 0$ | $\varepsilon'$ lower bound **increases** | Because $\varepsilon$ is endogenous to the model, making $\tilde M$ more "perfect CCA" actually makes the non-compositionality of rollout more apparent |
| Strict CCA $\delta=0$ | Retrofit requires $\Omega(2^\ell)$ | Strict CCA completely blocks efficient black-box solutions |
| Relaxed $\delta>0$, approximate augmentation | Authors conjecture lower bound still holds | Remark 5.5.1 left as open |
| Credit set always equals $S$ | Trivially CCA | But attribution is uninformative; bounding $|C|$ is key to meaningful CCA |

### Key Findings
- **CCA is not a natural extension of DP to sequences**: DP enjoys sequential composition because the "marginal distribution unaffected by $s_i$" is of the same type along the token chain; CCA's "non-credit condition" shrinks multiplicatively, so the lower bound $\varepsilon'$ does not converge.
- **Better retrofits are more vulnerable to inversion attacks**: The LP optimal solution explicitly exposes the prefix structure of $\mathbf z$, revealing a fundamental conflict between "perfect credit optimization" and "information hiding."
- **Vanishing-impact still requires credit**: Remark 5.6.1 notes that in $\mathcal M_\ell$, the data's effect on output vanishes at rate $2^{-\ell}$, but CCA still requires constant probability $\gamma$ credit—forcing a reconsideration of whether CCA is appropriate at the "almost no impact" boundary.
- **Token-level (0,0)-CCA is essentially useless**: Because it falls directly into the strongest counterexample of Thm 4.3, "perfect token-level attribution" is a seemingly robust but actually illusory goal.

## Highlights & Insights
- The counterintuitive result that "$\varepsilon\to 0$ actually increases the lower bound" is highly informative: it shows that "stronger token-level CCA" $\neq$ "stronger sequence-level CCA," directly refuting the engineering intuition of pursuing local perfection.
- The LP-based structural result for optimal attribution is a beautiful "computational vs information" dichotomy: information-wise, retrofitting is well-defined, but the structure of the solution leaks information, making it computationally exponential—such "well-defined but hard to compute" results are valuable references in ML security literature.
- The authors candidly include a critique in Remark 5.6.1: vanishing-impact still being mandatorily credited exposes that CCA conflates "any dependence" with "substantial dependence," suggesting that future work needs "impact-aware" CCA variants.

## Limitations & Future Work
- All lower bounds are under deployment-time CCA + black-box model + strict $\delta=0$; whether similar lower bounds hold for approximate augmentation (matching the original model only within TV distance $\leq 2d$) or $\delta>0$ is left as a conjecture.
- The counterexample family $\mathcal M_\ell$ is a highly adversarial artificial construction, not directly reflecting the geometry/semantics of real LLMs; whether "average-case" retrofitting is easier on real models is an open question.
- The paper only considers "binary credit," not Shapley-style continuous contribution measures; for copyright use cases, binary credit may be appropriate, but for scoring platforms, it may be insufficient.
- End-to-end CCA (joint training + deployment) is explicitly left as future work, which is the biggest gap for practical adoption.

## Related Work & Insights
- **vs Livni et al. 2024**: The original paper proved the existence of CCA algorithms in PAC learning (learnable within VC dimension + log factor); this paper shifts to the sequential generative model setting, with completely opposite conclusions—strong infeasibility, clarifying the gap between the two settings.
- **vs DP-LLM (Majmudar 2022 / Amin 2024)**: DP is almost a free lunch under the next-token + composition paradigm; this paper proves that the same paradigm does not work for CCA, suggesting that researchers must redesign sequence-aware CCA training objectives.
- **vs Vyas et al. 2023 (near access-freeness)**: That line uses black-box wrapping to achieve a different copyright relaxation; this paper proves that CCA does **not** admit a similar black-box solution, thoroughly separating the engineering friendliness of the two relaxations.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Extends CCA rigorously from PAC learning to generative models, providing two independent strong infeasibility results
- Experimental Thoroughness: N/A Theoretical paper, complete proofs, elegant counterexample constructions
- Writing Quality: ⭐⭐⭐⭐ Counterexample → general lower bound → LP characterization → reduction → ultimate lower bound, with a very clear reasoning chain
- Value: ⭐⭐⭐⭐ "Clears the minefield" for the emerging CCA-LLM direction, helping future work avoid two dead ends, and inspiring "impact-aware" CCA variants

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Cognitive Fatigue in Autoregressive Transformers: Formalization and Measurement](cognitive_fatigue_in_autoregressive_transformers_formalization_and_measurement.md)
- [\[ICML 2026\] Manifold-Aligned Guided Integrated Gradients for Reliable Feature Attribution](manifold-aligned_guided_integrated_gradients_for_reliable_feature_attribution.md)
- [\[ICML 2026\] Towards Long-Horizon Interpretability: Efficient and Faithful Multi-Token Attribution for Reasoning LLMs](towards_long-horizon_interpretability_efficient_and_faithful_multi-token_attribu.md)
- [\[AAAI 2026\] Distribution-Based Feature Attribution for Explaining the Predictions of Any Classifier](../../AAAI2026/interpretability/distribution-based_feature_attribution_for_explaining_the_predictions_of_any_cla.md)
- [\[CVPR 2026\] Feature Attribution Stability Suite: How Stable Are Post-Hoc Attributions?](../../CVPR2026/interpretability/feature_attribution_stability_suite_how_stable_are_post-hoc_attributions.md)

</div>

<!-- RELATED:END -->
