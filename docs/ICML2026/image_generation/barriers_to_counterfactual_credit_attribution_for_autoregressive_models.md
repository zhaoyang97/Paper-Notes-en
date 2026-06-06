---
title: >-
  [Paper Note] Barriers to Counterfactual Credit Attribution for Autoregressive Models
description: >-
  [ICML 2026][Image Generation][Counterfactual Credit Attribution] This paper formally studies the "Counterfactual Credit Attribution (CCA)" problem for generative models in RAG/in-context deployment settings. It proves tw…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Counterfactual Credit Attribution"
  - "Differential Privacy"
  - "Autoregressive Models"
  - "RAG"
  - "Impossibility Lower Bounds"
date: 2026-05-08
content_hash: cf9a209817849dcd
---

# Barriers to Counterfactual Credit Attribution for Autoregressive Models

**Conference**: ICML 2026  
**arXiv**: [2605.01425](https://arxiv.org/abs/2605.01425)  
**Code**: Not yet public (Theoretical paper)  
**Area**: AI Safety / Theory / Generative Model Copyright  
**Keywords**: Counterfactual Credit Attribution, Differential Privacy, Autoregressive Models, RAG, Impossibility Lower Bounds

## TL;DR
This paper formally studies the "Counterfactual Credit Attribution (CCA)" problem for generative models in RAG/in-context deployment settings. It proves two surprising negative results: (1) Autoregressive rollout does not satisfy CCA even if the underlying next-token predictor is (0,0)-CCA—CCA does not compose naturally like DP; (2) Performing black-box "CCA retrofitting" on a deployed non-attribution model requires an exponential number of queries relative to the output length $\ell$.

## Background & Motivation

**Background**: Generative AI disrupts the convention of "creators crediting predecessors"—RAG and in-context learning create a black-box model that obscures the causal chain between the final output and specific sources. In academic writing, this is misconduct; legally (copyright), it involves licensing and commercial boundaries. Livni-Moran-Nissim-Pabbaraju (2024) proposed **Counterfactual Credit Attribution (CCA)** as a relaxation of Differential Privacy (DP): an algorithm $\tilde A(S)$ outputs both a primary result $y$ and a credited subset $C\subseteq S$; it requires that "uncredited inputs" be distributionally close to the scenario where those inputs are entirely removed, i.e., $\tilde A^{-i}(S)\approx_{\varepsilon,\delta}\tilde A_{-i}(S)$. The original paper only studied CCA in PAC learning scenarios, leaving generative models as an open problem.

**Limitations of Prior Work**: There are two natural engineering paths to apply CCA to LLMs: (a) designing a CCA next-token predictor and running it autoregressively to obtain the whole output's CCA through "automatic composition"; (b) taking a non-attribution off-the-shelf LLM and adding a "wrapper" to retroactively provide credit sets. Whether these are feasible remains unanswered. This contrasts sharply with the development of DP-LLMs, where token-level composition is nearly a free lunch (Majmudar 2022, Amin 2024).

**Key Challenge**: DP has clean sequential composition theorems (worst case $k\varepsilon$ for $k$ compositions). While CCA resembles DP, its core lies in the **conditional distribution** of being "uncredited." This conditionality is amplified by the autoregressive chain; thus, CCA looks like DP but fails to compose. Similarly for retrofitting: once a model decides on an output, back-calculating "who to credit" is equivalent to estimating how much mass sensitive data contributed to each trajectory, which is exponentially difficult for a black-box model.

**Goal**: Systematically examine the two engineering intuitions for deployment-time data (RAG databases, in-context examples) and provide rigorous (im)possibility proofs.

**Key Insight**: The authors use a "counterexample first, lower bound second" strategy. They first construct a counterexample using a small 2-token, single-document toy model (extremely compact and reproducible), then generalize it into a parameterized lower bound theorem. For retrofitting, they construct a family of "nearly identical but slightly 1-bit biased" hard models, transferring the difficulty of finding a hidden identifier $\mathbf{z}$ to the retrofitter.

**Core Idea**: Strictly distinguish CCA from "DP relaxation." Prove it is **infeasible** through both the autoregressive composition and black-box retrofitting paths, setting clear boundaries for future CCA-LLM research.

## Method

### Overall Architecture
The paper follows two main lines. **First Line (§4)**: Investigates the industrial path of "imposing CCA on a next-token predictor $\tilde M$ + autoregressive rollout," providing a counterexample (Thm 4.2) and a general lower bound (Thm 4.3). **Second Line (§5)**: Investigates the retrofitting path of "modifying a non-attribution $M$ as a black box," constructing a hard model family $\{M_\mathbf{z}\}_{\mathbf{z}\in\{0,1\}^\ell}$ and proving that any $\alpha < 1/2$ approximate retrofit requires $\widetilde\Omega(2^\ell)$ oracle queries in the worst case (Thm 5.5). All proofs are built on rigorous LP characterizations and information-theoretic query lower bounds.

### Key Designs

1.  **Counterexample and General Lower Bound for CCA Autoregressive Composition (Thm 4.2 & 4.3)**:
    - **Function**: Disproves the intuition that "$\tilde M$ is $(0,0)$-CCA $\Rightarrow$ rollout $G^{\tilde M}$ is $(\varepsilon,\delta)$-CCA."
    - **Mechanism**: Take dataset $\mathcal S=\{s_1\}$ and vocabulary $\mathcal X=\{\mathtt a, \mathtt b\}$. Define a $(0,0)$-CCA next-token predictor: under an empty prefix, $\tilde M(\{s_1\},\lambda)$ and $\tilde M(\emptyset, \lambda)$ are identical (outputs $\mathtt a$ with $p$ and $\mathtt b$ with $1-p$, never credits). For prefix $\mathtt a$, it credits $s_1$ with 1/2 probability and outputs $\mathtt b$ without credit otherwise (the distribution conditioned on not crediting is identical to $S=\emptyset$). For prefix $\mathtt b$, it always credits and outputs $\mathtt a$ (the 1.0 term bypasses CCA constraints). However, after rollout: conditioned on not crediting, $G^{\tilde M}(S^{-1},\lambda)$ must output $(\mathtt{ab},\emptyset)$, whereas the counterfactual $G^{\tilde M}(S_{-1},\lambda)$ only produces $(\mathtt{ab},\emptyset)$ with probability $p$. CCA is violated when $p < e^{-\varepsilon}(1-\delta)$. Thm 4.3 generalizes this as $\varepsilon'\geq\ln\big(\prod_j\Pr[E_j\mid\dots]/\Pr[s_i\notin C]\big)-|\mathbf x^{-i}|\cdot\varepsilon$, where $E_j$ is the event of not crediting at step $j$.
    - **Design Motivation**: DP composes because the "distribution unaffected by $s_i$" is a consistent quantity at both token and sequence levels. In CCA, the core element $\Pr[s_i\notin C]$ **contracts** along the chain, causing the ratio of conditional distributions to amplify. This explains the counter-intuitive result where the bound increases as $\varepsilon \to 0$.

2.  **Hard Model Family $\mathcal M_\ell$ for CCA Retrofitting (Thm 5.5)**:
    - **Function**: Construct a family $\{M_\mathbf{z}\}$ such that "identifying $\mathbf z$" is an implicit subproblem of retrofitting yet exponentially difficult for an oracle of the original model.
    - **Mechanism**: $\mathcal X=\{0,1,\bot\}$, $\mathcal S=\{s_1\}$, $\ell\geq 1$, $\gamma\in(0,1)$, $\varepsilon\geq 0$. $M_\mathbf z(S, \mathbf x)$ is almost everywhere $\mathsf{Bern}(1/2)$ for $|\mathbf x| \leq \ell$, **except** when $S \neq \emptyset$ and the prefix exactly matches $\mathbf z$, where it biases the probability of 1 to $\tfrac12+\tfrac{1-e^{-\varepsilon}(1-\gamma)}{2}$; output length is constant $\ell+1$. Under the oracle model, the TV distance between $M_\mathbf z$ and $M_\emptyset$ is $\leq 2^{-\ell}$ (Remark 5.6.1). Finding $\mathbf z$ is equivalent to a needle-in-a-haystack search on $\{0,1\}^\ell \Rightarrow$ Lemma 5.8: $\Omega(2^\ell)$ query lower bound.
    - **Design Motivation**: To set a computational lower bound for retrofitting, information must be hidden in positions extremely difficult for an oracle to sample. This family encodes the statistical difficulty of distinguishing two nearly identical distributions into an $\ell$-bit secret string.

3.  **LP Characterization of Optimal CCA Augmentation (Lemma 5.6) + Reducibility (Lemma 5.9)**:
    - **Function**: Transforms any "oracle for near-optimal CCA augmentation" into an algorithm that finds $\mathbf z$ in $O(\ell \log \ell)$ queries, proving that a retrofitter must pay at least $\widetilde\Omega(2^\ell / \ell \log \ell)$ oracle queries.
    - **Mechanism**: Formalizes "optimal CCA augmentation" as an LP where variables are probabilities over $(S, \mathbf x, y, C)$. Constraints include augmentation (marginal consistency with $G^M$) and CCA (closeness of conditional distributions). The target is minimizing $\mathbb{E}[C]$. The analytical solution shows: $\Pr[\tilde G^*_\mathbf z(S^*, \mathbf x) \text{ credits } s_1] = \gamma$ if and only if prefix $\mathbf x \sqsubseteq \mathbf z$, otherwise 0. This sharp structure allows binary searching $\mathbf z$ along the prefix tree.
    - **Design Motivation**: This uses the "structure of the optimal solution" as an attack surface: if a retrofit must be optimal ($\alpha$-approximate), its attribution probability leaks $\mathbf z$. A better retrofit is easier to invert, allowing an attacker to extract secrets the model oracle could not find.

### Loss & Training
This is a purely theoretical paper; no training or loss functions are used. The primary contribution is a series of rigorous (im)possibility proofs.

## Key Experimental Results

### Main Results
The paper contains no empirical experiments; theoretical results substitute for empirical data.

| Proposition | Setting | Conclusion |
| :--- | :--- | :--- |
| Thm 4.2 | $\forall\varepsilon\geq 0, \delta<1$ | Exist $(0,0)$-CCA next-token predictors whose rollout is not $(\varepsilon,\delta)$-CCA |
| Thm 4.3 | $\tilde M$ $\varepsilon$-CCA, rollout $\varepsilon'$-CCA | $\varepsilon'\geq\max\big[\ln(\prod_j\Pr[E_j]/\Pr[s_i\notin C])-\|\mathbf{x}^{-i}\|\cdot\varepsilon\big]$ |
| Thm 5.5 | $\alpha<1/2, \delta=0$ | $\alpha$-approximate retrofit on worst-case $M\in\mathcal M_\ell$ needs $\Omega(2^\ell/\ell \log \ell)$ queries |
| Remark 5.6.1 | $\mathcal M_\ell$ | Data affect on model TV is $\leq 2^{-\ell}$, yet $\gamma>0$ credit probability is required |

### Ablation Study
The paper uses "case-by-case tightening" instead of traditional ablations:

| Condition | Result | Explanation |
| :--- | :--- | :--- |
| $\varepsilon \to 0$ | $\varepsilon'$ bound **increases** | As $\varepsilon$ is endogenous, a "perfectly CCA" $\tilde M$ makes rollout non-compositionality more apparent |
| $\delta=0$ (Strict) | Retrofit needs $\Omega(2^\ell)$ | Strict CCA completely blocks efficient black-box solutions |
| $\delta>0$ (Relaxed) | Lower bound likely holds | Mentioned in Remark 5.5.1 as an open conjecture |
| Credit set always equals $S$ | Trivially CCA | Attribution becomes uninformative; small $|C|$ is key to CCA's meaningfulness |

### Key Findings
- **CCA is not a natural extension of DP for sequences**: DP enjoys sequential composition because the "marginal distribution unaffected by $s_i$" is consistent; CCA's "uncredited condition" contracts multiplicatively, causing $\varepsilon'$ to diverge.
- **Superior retrofitting is more vulnerable to inversion**: The LP optimal solution explicitly exposes the prefix structure of $\mathbf z$, revealing a fundamental conflict between "perfect credit optimization" and "information hiding."
- **Vanishing impact still requires credit**: Remark 5.6.1 notes that even if data impact vanishes at $2^{-\ell}$, CCA requires credit with constant probability $\gamma$. This suggests CCA may conflate any dependency with "substantial" dependency.
- **Token-level (0,0)-CCA is practically useless**: It falls directly into the strongest counterexample of Thm 4.3, meaning "perfect token-level attribution" is a deceptive goal.

## Highlights & Insights
- The counter-intuitive conclusion that "$\varepsilon \to 0$ increases the bound" is highly informative: it shows that stronger token-level CCA does not imply stronger sequence-level CCA.
- The LP-based structural result for optimal attribution represents a beautiful split between "computation" and "information": while a retrofit is well-defined informationally, it is exponentially hard computationally.
- The authors honestly critique the CCA definition in Remark 5.6.1, highlighting that mandatory credit for vanishing impacts may limit its applicability, prompting future "impact-aware" CCA variants.

## Limitations & Future Work
- Lower bounds are restricted to deployment-time CCA, black-box models, and strict $\delta=0$ settings. Whether bounds hold for $\delta > 0$ remains a conjecture.
- The model family $\mathcal M_\ell$ is a highly adversarial construction and may not reflect the geometry of real LLMs; "average-case" retrofitting remains an open question.
- Only "binary credit" is considered, without touching Shapley-style continuous contribution measures.
- End-to-end CCA (joint training and deployment) is explicitly left for future work.

## Related Work & Insights
- **vs Livni et al. 2024**: While the original work proved existence in PAC learning, this work shifts to the sequential generative setting and reaches the opposite conclusion—strong infeasibility.
- **vs DP-LLM**: DP is a free lunch under composition; CCA is not. This forces researchers to rethink sequence-aware CCA training objectives.
- **vs Vyas et al. 2023**: While other copyright relaxations allow black-box wrappers, this paper proves CCA does not, distinguishing the "engineering friendliness" of different definitions.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: N/A (Theoretical)
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ Creates clear "no-go" zones for CCA-LLM research and inspires impact-aware variants.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] GUDA: Counterfactual Group-wise Training Data Attribution for Diffusion Models via Unlearning](guda_counterfactual_group-wise_training_data_attribution_for_diffusion_models_vi.md)
- [\[ICML 2026\] Visual Implicit Autoregressive Modeling](visual_implicit_autoregressive_modeling.md)
- [\[ICLR 2026\] PCPO: Proportionate Credit Policy Optimization for Aligning Image Generation Models](../../ICLR2026/image_generation/pcpo_proportionate_credit_policy_optimization_for_aligning_image_generation_mode.md)
- [\[CVPR 2026\] Attribution as Retrieval: Model-Agnostic AI-Generated Image Attribution](../../CVPR2026/image_generation/attribution_as_retrieval_modelagnostic_aigenerated.md)
- [\[ICLR 2026\] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting](../../ICLR2026/image_generation/doflow_flow-based_generative_models_for_interventional_and_counterfactual_foreca.md)

</div>

<!-- RELATED:END -->
