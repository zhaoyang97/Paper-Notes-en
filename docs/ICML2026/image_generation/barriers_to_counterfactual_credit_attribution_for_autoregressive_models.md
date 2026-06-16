---
title: >-
  [Paper Note] Barriers to Counterfactual Credit Attribution for Autoregressive Models
description: >-
  [ICML 2026][Image Generation][RAG] This paper formally investigates the problem of "Counterfactual Credit Attribution (CCA)" for generative models in RAG/in-context deployment. It proves two surprising negative results: (1) An autoregressive rollout is not necessarily CCA even if the underlying next-token predictor is $(0,0)$-CCA—CCA does not compose na
tags:
  - ICML 2026
  - Image Generation
  - RAG
date: 2026-05-08
content_hash: 350c3b4edd5dc497
---
# Barriers to Counterfactual Credit Attribution for Autoregressive Models

**Conference**: ICML 2026  
**arXiv**: [2605.01425](https://arxiv.org/abs/2605.01425)  
**Code**: Not yet public (Theoretical paper)  
**Area**: AI Safety / Theory / Generative Model Copyright  
**Keywords**: Counterfactual Credit Attribution, Differential Privacy, Autoregressive Models, RAG, Infeasibility Lower Bounds

## TL;DR
This paper formally investigates the problem of "Counterfactual Credit Attribution (CCA)" for generative models in RAG/in-context deployment. It proves two surprising negative results: (1) An autoregressive rollout is not necessarily CCA even if the underlying next-token predictor is $(0,0)$-CCA—CCA does not compose naturally under autoregression like DP does; (2) Black-box "CCA retrofitting" for a deployed non-attributing model requires an exponential number of queries relative to the output length $\ell$.

## Background & Motivation

**Background**: Generative AI has disrupted the principle that "creators should cite predecessors"—RAG/in-context learning obscures the causal chain between final outputs and specific sources within the model black box. This constitutes academic misconduct in scholarly writing and involves licensing and commercial boundaries at the legal (copyright) level. Livni-Moran-Nissim-Pabbaraju (2024) proposed **Counterfactual Credit Attribution (CCA)** as a relaxation of Differential Privacy (DP): an algorithm $\tilde A(S)$ simultaneously outputs a result $y$ and a credited subset $C\subseteq S$. It requires the distribution of "uncredited inputs" to be close to the distribution as if those inputs were completely removed, i.e., $\tilde A^{-i}(S)\approx_{\varepsilon,\delta}\tilde A_{-i}(S)$. The original paper only studied CCA in PAC learning scenarios, leaving generative models as an open problem.

**Limitations of Prior Work**: Directly applying CCA to LLMs suggests two natural engineering paths: (a) designing a CCA next-token predictor and running it autoregressively to achieve whole-sequence CCA through "automatic composition"; (b) taking a non-attributing off-the-shelf LLM and adding a "wrapper" to retroactively provide a credit set. The feasibility of these paths remained systematically unanswered. This contrasts sharply with the development of DP-LLMs, where token-level composition is essentially a free lunch (Majmudar 2022, Amin 2024).

**Key Challenge**: DP has clean composition theorems under sequential composition (worst-case $k\varepsilon$ for $k$ steps). CCA resembles DP but centers on the **conditional distribution** of being "uncredited," which is amplified by the autoregressive multiplicative chain. CCA seems like DP but does not compose. Retrofitting faces similar issues: once a model decides on an output, reverse-engineering "who gets credit" is equivalent to estimating how much "sensitive data contributed to the mass" of each trajectory, which is exponentially difficult for a black-box model.

**Goal**: To deploy CCA for deployment-time data (RAG databases, in-context examples) and systematically verify the two aforementioned engineering intuitions with rigorous (in)feasibility proofs.

**Key Insight**: The authors adopt a "counterexample first, lower bound second" strategy. They first construct a 2-token, single-document toy counterexample (compact and reproducible) and then generalize it into a parameterized lower bound theorem. For retrofitting, they construct a family of "nearly identical models with a slight 1-bit bias" to transfer the difficulty of finding a hidden identifier $\mathbf{z}$ to the retrofitter.

**Core Idea**: Clearly distinguish CCA from "looking like a DP relaxation." The paper proves that CCA is **infeasible** in both of the most natural engineering paths—autoregressive composition and black-box retrofitting—thereby defining the boundaries for subsequent CCA-LLM research.

## Method

### Overall Architecture
This is a purely theoretical work addressing whether two natural engineering paths for deploying CCA on RAG/in-context data are viable. The first path (§4) involves adding CCA to a next-token predictor $\tilde M$ and performing an autoregressive rollout, expecting the full output to "automatically" inherit attribution properties. The second path (§5) involves using an existing non-attributing model $M$ and adding a black-box wrapper to provide credit. The paper uses a strategy of "simplistic counterexamples followed by parameterized lower bounds" to provide rigorous infeasibility proofs for both paths, utilizing LP characterization and information-theoretic query lower bounds.

### Key Designs

**1. CCA Does Not Compose Autoregressively: Counterexample and General Lower Bound (Thm 4.2 & 4.3): Why intuition fails**

DP is useful because it has clean sequential composition theorems; thus, it was naturally expected that CCA would follow: "$\tilde M$ is $(0,0)$-CCA $\Rightarrow$ rollout $G^{\tilde M}$ is $(\varepsilon,\delta)$-CCA." This paper falsifies this with a 2-token counterexample. Given dataset $\mathcal S=\{s_1\}$ and vocabulary $\mathcal X=\{\mathtt a,\mathtt b\}$, a token-level strictly $(0,0)$-CCA predictor is constructed. Under an empty prefix, $\tilde M(\{s_1\},\lambda) = \tilde M(\emptyset,\lambda)$ are identical (outputting $\mathtt a$ with probability $p$ and $\mathtt b$ with $1-p$, never crediting). When the prefix is $\mathtt a$, it credits with probability $1/2$ under $S=\{s_1\}$, and with probability $1/2$ it does not credit and outputs $\mathtt b$, where this "uncredited conditional distribution" is identical to the $S{=}\emptyset$ case (satisfying token-level $(0,0)$-CCA). When the prefix is $\mathtt b$, it always credits and outputs $\mathtt a$ (satisfying CCA constraints trivially due to $1.0$ trigger probability). However, during rollout, under the "uncredited" condition, $G^{\tilde M}(S^{-1},\lambda)$ necessarily yields $(\mathtt{ab},\emptyset)$, whereas the counterfactual $G^{\tilde M}(S_{-1},\lambda)$ with $s_1$ removed yields $(\mathtt{ab},\emptyset)$ only with probability $p$. If $p < e^{-\varepsilon}(1-\delta)$, $(\varepsilon,\delta)$-CCA is violated. Thm 4.3 abstracts this into a general lower bound $\varepsilon'\geq\ln\big(\prod_j\Pr[E_j\mid\cdots]/\Pr[s_i\notin C]\big)-|\mathbf x^{-i}|\cdot\varepsilon$ (where $E_j$ is the uncredited event at step $j$). The essence is that DP composes because the "unaffected conditional distribution" is the same type of quantity at token and sequence levels, whereas the core CCA element $\Pr[s_i\notin C]$ **shrinks** along the multiplicative chain, magnifying the ratio of conditional distributions. The simplicity of the counterexample nearly precludes all natural CCA designs.

**2. The Hard Model Family $\mathcal M_\ell$ for Retrofitting (Thm 5.5): Hiding difficulty in a secret string**

The second path proves that "black-box modification" is computationally infeasible. The key is constructing a model family $\{M_\mathbf{z}\}_{\mathbf z\in\{0,1\}^\ell}$ where "identifying the hidden identifier $\mathbf z$" is both an unavoidable sub-problem for retrofitting and exponentially difficult for the original model oracle. Let $\mathcal X=\{0,1,\bot\}$, $\mathcal S=\{s_1\}$, $\ell\geq 1$, $\gamma\in(0,1)$, and $\varepsilon\geq 0$. $M_\mathbf z(S,\mathbf x)$ behaves like $\mathsf{Bern}(1/2)$ almost everywhere when $|\mathbf x|\leq\ell$, **except** when $S\neq\emptyset$ and the prefix exactly matches $\mathbf z$, at which point it biases the probability of outputting 1 to $\tfrac12+\tfrac{1-e^{-\varepsilon}(1-\gamma)}{2}$. The output length is fixed at $\ell+1$. Information is hidden where the oracle is unlikely to sample: the TV distance between $M_\mathbf z$ and $M_\emptyset$ for any prompt is $\leq 2^{-\ell}$ (Remark 5.6.1). Finding $\mathbf z$ is equivalent to a needle-in-haystack search over $\{0,1\}^\ell$, yielding an $\Omega(2^\ell)$ query lower bound (Lemma 5.8).

**3. LP Characterization and Reduction of Optimal CCA Augmentation (Lemma 5.6 & 5.9): Better retrofits are easier to reverse-engineer**

To derive the final $\widetilde\Omega(2^\ell/\ell\log\ell)$ lower bound, "finding $\mathbf z$" must be reduced to retrofitting. The "optimal CCA augmentation" is formalized as an LP where variables are probabilities over $(S,\mathbf x,y,C)$, constraints include augmentation (matching $G^M$ marginals) and CCA constraints, and the goal is to minimize $\mathbb{E}[f(C)]$. The analytical solution reveals a sharp structure: $\Pr[\tilde G^*_\mathbf z(S^*,\mathbf x)\text{ credits }s_1] = \gamma$ if and only if the prefix $\mathbf x\sqsubseteq\mathbf z$, and 0 otherwise. Thus, the crediting probability is a constant $\gamma$ on the prefix tree of $\mathbf z$ and 0 elsewhere. This allows for identifying $\mathbf z$ via binary search on the prefix tree with $O(\ell\log\ell/(\gamma-2\alpha)^2)$ retrofit queries. This "computational hardness via solution-structure leakage" is fatal: the closer a retrofit is to optimal (even $\alpha$-approximate), the more its attribution probability at each prefix "leaks" $\mathbf z$.

## Key Experimental Results

### Main Results
The paper contains no empirical experiments; theoretical results are summarized below.

| Proposition | Setting | Conclusion |
|------|------|------|
| Thm 4.2 | $\forall\varepsilon\geq 0,\delta<1$ | Exist $(0,0)$-CCA next-token predictor whose rollout is not $(\varepsilon,\delta)$-CCA |
| Thm 4.3 | $\tilde M$ $\varepsilon$-CCA, rollout $\varepsilon'$-CCA | $\varepsilon'\geq\max\big[\ln(\prod_j\Pr[E_j]/\Pr[s_i\notin C])-\|\mathbf{x}^{-i}\|\cdot\varepsilon\big]$ |
| Thm 5.5 | $\alpha<1/2$, $\delta=0$ | $\alpha$-approximate retrofit on worst-case $M\in\mathcal M_\ell$ needs $\Omega(2^\ell/\ell\log\ell)$ queries |
| Remark 5.6.1 | $\mathcal M_\ell$ | Data TV impact on model is $\leq 2^{-\ell}$, yet $\gamma>0$ credit probability is required |

### Ablation Study
The paper uses "tightening cases" instead of traditional ablations:

| Condition | Result | Explanation |
|------|------|------|
| $\varepsilon\to 0$ | $\varepsilon'$ lower bound **increases** | As $\varepsilon$ is endogenous, a "perfectly CCA" $\tilde M$ makes rollout non-composability more prominent |
| $\delta=0$ Strict CCA | Retrofit requires $\Omega(2^\ell)$ | Strict CCA completely blocks efficient black-box solutions |
| $\delta>0$ Relaxation | Lower bound conjectured to hold | Remark 5.5.1 leaves this as an open problem |
| Credit set always equals $S$ | Trivially CCA | Attribution becomes uninformative; limiting $|C|$ is key to CCA meaningfulness |

### Key Findings
- **CCA is not a natural DP extension for sequences**: DP enjoys sequential composition because "unaffected marginal distributions" are of the same type; in CCA, the "uncredited condition" shrinks multiplicatively, causing the lower bound $\varepsilon'$ to diverge.
- **Superior retrofitting facilitates reverse-engineering attacks**: LP optimal solutions explicitly expose the prefix structure of $\mathbf z$, revealing an inherent conflict between "perfect credit optimization" and "information hiding."
- **Vanishing-impact still requires credit**: Remark 5.6.1 notes that in $\mathcal M_\ell$, the data's impact on output vanishes at a rate of $2^{-\ell}$, yet CCA still mandates a constant credit probability $\gamma$. This necessitates a re-evaluation of CCA as an appropriate definition in "near-zero impact" regimes.
- **Token-level (0,0)-CCA is practically useless**: It falls directly into the strongest counterexample zone of Thm 4.3, implying that "perfect token-level attribution" is a safe-looking but illusory goal.

## Highlights & Insights
- The counter-intuitive conclusion that "$\varepsilon\to 0$ increases the lower bound" is highly informative: it proves that "stronger token-level CCA" $\neq$ "stronger sequence-level CCA," falsifying engineering intuitions that seek local perfection.
- The structural result of LP-based optimal attribution represents an elegant "computational vs information" split: while the retrofit is well-defined informationally, its structural leakage makes it exponentially difficult computationally.
- The authors honestly critique the CCA definition in Remark 5.6.1: the requirement to credit vanishing-impact inputs suggests CCA conflates "any dependence" with "substantial dependence," prompting future "impact-aware" CCA variants.

## Limitations & Future Work
- All lower bounds are established under deployment-time CCA + black-box models + strict $\delta=0$ settings; the strength of bounds under $\delta>0$ remains conjectured.
- The model family $\mathcal M_\ell$ is a highly adversarial synthetic construction and does not directly reflect the geometric/semantic structure of real LLMs; "average-case" retrofitting on real models remains an open question.
- The paper only considers binary credit, not continuous Shapley-style contribution measures. Binary credit may suit copyright use cases but might be insufficient for scoring-based platforms.
- End-to-end CCA (joint training + deployment) is explicitly left for future work as the primary gap for practical application.

## Related Work & Insights
- **vs Livni et al. 2024**: While the original work proved the existence of CCA algorithms in PAC learning, this work moves to the sequential generative setting and reaches the opposite conclusion—strong infeasibility.
- **vs DP-LLM (Majmudar 2022 / Amin 2024)**: DP is a "free lunch" in next-token composition; this paper proves the paradigm does not work for CCA, suggesting a need for sequence-aware CCA training objectives.
- **vs Vyas et al. 2023 (near access-freeness)**: That line achieves a different copyright relaxation via black-box wrapping; this paper proves CCA does **not** have a similar black-box solution, separating the engineering feasibility of the two approaches.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Rigorous extension of CCA to generative models with two independent strong infeasibility results.
- Experimental Thoroughness: N/A Theoretical paper with complete proofs and sophisticated counterexamples.
- Writing Quality: ⭐⭐⭐⭐ Clear progression from counterexamples to general bounds and LP reductions.
- Value: ⭐⭐⭐⭐ Protects researchers from pursuing two "dead ends" in CCA-LLM and inspires "impact-aware" CCA variants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICLR 2026\] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models](../../ICLR2026/image_generation/pcpo_proportionate_credit_policy_optimization_for_aligning_image_generation_mode.md)
- [\[ICML 2026\] Visual Implicit Autoregressive Modeling](visual_implicit_autoregressive_modeling.md)
- [\[ICLR 2026\] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting](../../ICLR2026/image_generation/doflow_flow-based_generative_models_for_interventional_and_counterfactual_foreca.md)
- [\[CVPR 2026\] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution](../../CVPR2026/image_generation/attribution_as_retrieval_modelagnostic_aigenerated.md)

</div>

<!-- RELATED:END -->
