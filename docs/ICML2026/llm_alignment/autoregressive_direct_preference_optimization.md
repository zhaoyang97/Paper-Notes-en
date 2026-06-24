---
title: >-
  [Paper Note] Autoregressive Direct Preference Optimization
description: >-
  [ICML2026][LLM Alignment][DPO] The authors observe that DPO's derivation sequence is flawed: it constructs a Bradley-Terry (BT) preference model based on the entire answer first and imposes the autoregressive assumption on the model only afterwards. ADPO **advances** the autoregressive assumption to before the BT model construction by defining energy functions on the prefix closure of the output space. This yields a minimalist new loss that moves the summation sign from **ins…
tags:
  - "ICML2026"
  - "LLM Alignment"
  - "DPO"
  - "Autoregressive"
  - "Bradley-Terry"
  - "Prefix Closure"
  - "Granularity"
date: 2026-05-08
content_hash: 898cdff7da00cd27
---

# Autoregressive Direct Preference Optimization

**Conference**: ICML2026  
**arXiv**: [2602.09533](https://arxiv.org/abs/2602.09533)  
**Code**: [Project Page](https://stjohn2007.github.io/ADPO_project/)  
**Area**: Alignment RLHF / Preference Optimization  
**Keywords**: DPO, Autoregressive, Bradley-Terry, Prefix Closure, Granularity  

## TL;DR
The authors observe that DPO's derivation sequence is flawed: it constructs a Bradley-Terry (BT) preference model based on the entire answer first and imposes the autoregressive assumption on the model only afterwards. ADPO **advances** the autoregressive assumption to before the BT model construction by defining energy functions on the prefix closure of the output space. This yields a minimalist new loss that moves the summation sign from **inside** the log-sigmoid to the **outside**. Consequently, it distinguishes two independent length measures for the first time—token length $\mu$ and feedback length $\mu'$-unifying training at any granularity from full answers to individual tokens.

## Background & Motivation
**Background**: DPO has become the mainstream method for aligning LLMs. By bypassing explicit reward models and optimizing policies directly on preference pairs, it is both efficient and scalable, spawning numerous variants such as SimPO, TDPO, TGDPO, and cDPO.

**Limitations of Prior Work**: These variants (including those claiming to be token-level) fundamentally **depend on answer-level BT models**. The BT assumption defines the reward $r(x,y)$ on the complete answer $y\in\mathcal{Y}$. However, LLMs generate tokens autoregressively, creating a structural mismatch between "answer-level modeling" and "autoregressive generation."

**Key Challenge**: Why is answer-level the default? Reward models usually learn to evaluate complete $(x,y)$ pairs, as it is unrealistic for humans to rate an incomplete prefix $(x,y_{\le i})$. However, DPO is unique because **it does not require an explicit reward model**. This provides an opportunity to introduce an implicit reward function that better fits the autoregressive structure without being constrained by the "human evaluation" bottleneck.

**Goal**: Can we define a set of energy functions such that the Boltzmann distribution $p_2$ in the DPO derivation is **explicitly** an autoregressive distribution, rather than forcing the autoregressive assumption post-hoc?

**Key Insight**: A gap exists in the DPO derivation—while the learnable model $\pi_\theta$ is autoregressive, the distribution $p_2$ defined in Eq. (4) is not formulated autoregressively; autoregressivity is "assumed only after deriving the objective."

**Core Idea**: Expand the domain of the energy function from the output space $\mathcal{Y}$ to its **prefix closure $\mathcal{Y}^*$** (the set of all incomplete prefixes). By building the BT model on prefixes, the autoregressive assumption becomes intrinsic to the derivation, leading to a loss where summation is moved outside the log-sigmoid.

## Method

### Overall Architecture
ADPO is a "re-foundation" of the DPO derivation. It does not change the architecture or add modules; its core lies in three foundational shifts: ① defining two energy functions (prefix likelihood energy $E_1^*$ and prefix posterior energy $E_2^*$) on the **prefix closure $\mathcal{Y}^*$** rather than complete answers, explicitly assuming the reference model is autoregressive within $E_2^*$; ② constructing a **prefix-level BT model** using these energies (multiplying BT preference probabilities across each prefix length); ③ minimizing the negative log-likelihood and reparameterizing to obtain the ADPO loss. The final loss is nearly identical to DPO, differing in only one position: DPO uses $-\log\sigma\big(\beta\sum_i(\cdots)\big)$ (summation inside), while ADPO uses $-\sum_i\log\sigma\big(\beta(\cdots)\big)$ (summation outside). Theoretical analysis further reveals two independent length measures, unifying DPO and token-level methods into a "granularity family." As this is a purely theoretical work focused on loss derivation, no framework diagram is provided.

### Key Designs

**1. Defining Energy and Prefix-level BT Models on Prefix Closures to Deriving "Summation Outside Log-sigmoid"**

DPO energies, $E_1(x,y)=-r(x,y)$ and $E_2(x,y)=-\frac1\beta r(x,y)-\log\pi_{\text{ref}}(y|x)$, are defined on complete answers, making $p_2$ a distribution over full sequences where autoregressivity is forced post-hoc. ADPO changes the domain to the prefix closure $\mathcal{Y}^*=\bigcup_{y}\{y_{\le i}:0\le i\le T'\}$, defining prefix energies $E_1^*(x,y_{\le i})=-r^*(x,y_{\le i})$ and $E_2^*(x,y_{\le i})=-\frac1\beta r^*(x,y_{\le i})-\log\pi_{\text{ref}}(y_i|y_{<i},x)$. Here, the reference term in $E_2^*$ is the **per-token conditional probability** $\pi_{\text{ref}}(y_i|y_{<i},x)$, embedding autoregressivity into the energy itself. Thus, $p_2$ naturally decomposes into an autoregressive form $p_2(y|x)=\prod_i p_2(y_i|y_{<i},x)$, eliminating the mismatch with $\pi_\theta$. The prefix-level BT model expresses preference probability as a product across all prefix lengths $p_1(y^w\succ y^l|x)=\prod_{i=1}^{T'}\frac{\exp(-E_1^*(x,y^w_{\le i}))}{\sum_{y_{\le i}\in Y_i}\exp(-E_1^*(x,y_{\le i}))}$. Minimizing $-\log p_1$ and reparameterizing $p_2=\pi_\theta$ yields:

$$\mathcal{L}_{\text{ADPO}}=-\mathbb{E}_{(x,Y)\sim\mathcal{D}}\Big[\sum_{i=1}^{T'}\log\sigma\big(\beta\log\tfrac{\pi_\theta(y^w_i|y^w_{<i},x)}{\pi_{\text{ref}}(y^w_i|y^w_{<i},x)}-\beta\log\tfrac{\pi_\theta(y^l_i|y^l_{<i},x)}{\pi_{\text{ref}}(y^l_i|y^l_{<i},x)}\big)\Big].$$.

Compared to DPO, the summation is moved from **inside** to **outside** the log-sigmoid. Intuitively, DPO "sums the log-ratios of the entire answer first, then passes it through a sigmoid to determine win/loss," whereas ADPO "determines win/loss at each prefix position and then sums them up." The latter allows preference signals to take effect at every token step, providing finer granularity without violating DPO's theoretical foundations; the difference arises solely from the domain of the energy function.

**2. Autoregressive Reparameterization Completeness (Theorem 1): Any Reward can be Represented by an Autoregressive Model**

A beautiful loss is insufficient; it must be proven that this prefix-level reward is not arbitrary but consistent with classical rewards. The authors first prove **Proposition 1 (Prefix-level Reparameterization Completeness)**: For any prefix reward $r^*$, there exists a unique representative $r^*_\circ$ in its reward-shift equivalence class $[r^*]$ such that $r^*_\circ(x,y_{\le i})\equiv\beta\log\frac{\pi(y_i|y_{<i},x)}{\pi_{\text{ref}}(y_i|y_{<i},x)}$. Using **Definition 3 (Additive Decomposition)**, a standard reward $r(x,y)=\sum_i r^*(x,y_{\le i})$ is decomposed into a sum of prefix rewards, with **Lemma 1** ensuring every reward has such a decomposition. Combining these yields **Theorem 1**: All reward classes consistent with prefix-level BT models can be written as $r(x,y)=\beta\log\frac{\pi(y|x)}{\pi_{\text{ref}}(y|x)}$, where $\pi$ is an autoregressive model. This advances DPO theory—DPO states that some reward reparameterization exists; ADPO explicitly proves it can be implemented with an **autoregressive** model, truly aligning DPO theory with the autoregressive LLM paradigm.

**3. Distinguishing Two Independent Measures: Token Length $\mu$ and Feedback Length $\mu'$**

A byproduct of the theoretical analysis is an anti-intuitive but profound insight. **Corollary 1**: ADPO degenerates back to DPO when $\mu'(y)\equiv1$. However, the authors emphasize this does not mean the output space is limited to a single token, but rather that **the original DPO implicitly assumes a feedback length measure $\mu'$ that assigns a length of 1 to every sequence**. This means DPO treats the entire answer as an indivisible feedback unit. This implicit constraint is why previous DPO variants kept the summation inside the log-sigmoid. The authors decouple the two measures: $\mu$ is the **token length** from LLM tokenization, and $\mu'$ is the **feedback length** from the evaluation scenario (mapping the answer through an evaluation measure $\nu:\mathcal{Y}\to\mathbb{R}$ to one dimension, hence $\mu'=1$). Since they originate differently (one from tokenization, one from prefix closure), they are **theoretically independent**: DPO is $\mu'\equiv1$ (full answer), token-level ADPO is $\mu'=\mu$ (per-token), and choosing these measures independently allows training at any granularity.

**4. Static/Adaptive Granularity Families: Interpolating Between DPO and Token-level**

After freeing $\mu'$, ADPO allows intermediate granularities $1\le\mu'(y)\le\mu(y)$: segmenting sequence $y$ into sub-sequences $\{z_i\}$, where each prefix $z_{\le i}$ acts as a unit for implicit feedback. The loss is written as $\mathcal{L}_{\text{ADPO}}=-\mathbb{E}\big[\sum_i\log\sigma(\beta S^w_\theta(i)-\beta S^l_\theta(i))\big]$, where $S_\theta(i)$ is the accumulated per-token log-ratio within the sub-sequence. The segmentation is determined by a "strong composition" $\xi$, yielding two families: the **Static Family** uses a fixed window $k$, segmenting each sequence into $\lceil T/k\rceil$ segments ($k=1$ is token-level, larger $k$ is coarser); the **Adaptive Family** uses a fixed number of segments $m$, segmenting as evenly as possible ($m=1$ is DPO, $m>1$ refines granularity). This positions DPO and various token-level methods at two ends of a continuous spectrum.

### Loss & Training
The training process of ADPO is identical to DPO—only the loss function is replaced. No explicit reward model is added, and the optimal solution under KL constraints is maintained (Appendix B). The paper also overlays ADPO onto cDPO (key token weighting) to create cADPO, verifying that this foundation can be orthogonally combined with existing variants.

## Key Experimental Results

### Main Results
Four base models (Llama-3-8B / Gemma-3-12B / Qwen-3-8B / DeepSeek-Math-7B) across two math reasoning benchmarks (GSM8K / MATH):

| Method | Llama-3-8B GSM/MATH | Gemma-3-12B GSM/MATH | Qwen-3-8B GSM/MATH | DS-Math-7B GSM/MATH |
|------|------|------|------|------|
| DPO | 64.37 / 18.00 | 77.03 / 39.80 | 86.96 / 53.80 | 67.78 / 32.00 |
| **ADPO (Ours)** | **68.08 / 21.00** | **78.32 / 41.20** | **88.10 / 55.40** | **69.98 / 33.40** |
| cDPO | 67.90 / 16.80 | 77.18 / 38.60 | 90.98 / 56.80 | 72.90 / 33.40 |
| **cADPO (Ours)** | **68.76 / 20.20** | **78.85 / 40.40** | **91.74 / 57.20** | **73.54 / 35.40** |

ADPO outperforms the corresponding DPO in all 4×2=8 settings. Furthermore, cADPO (layering ADPO onto cDPO) consistently outperforms cDPO, indicating that the prefix-level foundation is orthogonal and additive to existing token-weighting techniques.

### Ablation Study (Static Family)

| Method | $\mu'(y)$ | Composition | Granularity | Llama GSM/MATH | Gemma GSM/MATH | Qwen GSM |
|------|-----------|------|------|------|------|------|
| ADPO | $\mu(y)/8$ | $\xi_{\text{static}}(k{=}8)$ | 8-token | 64.97 / 18.20 | 76.88 / 40.00 | 87.79 |
| ADPO | $\mu(y)/4$ | $\xi_{\text{static}}(k{=}4)$ | 4-token | 66.26 / 17.40 | 77.41 / 40.80 | 88.48 |

Different windows $k$ provide various granularities. Results vary by base model and dataset, confirming that "granularity is adjustable and impacts learning behavior."

### Key Findings
- **Moving the summation brings stable gains**: Simply moving the summation outside the log-sigmoid consistently outperforms DPO across 8 settings while maintaining the same theoretical optimal solution.
- **Orthogonal to existing variants**: cADPO generally outperforms cDPO, showing that ADPO serves as a superior foundation rather than just a replacement.
- **Adjustable granularity is a real degree of freedom**: The independence of $\mu'$ and $\mu$, along with $k$/$m$, provides a continuous spectrum from DPO to token-level, leaving room for future design.

## Highlights & Insights
- **The "Summation Inside vs. Outside" difference stems from the foundation**: The most elegant aspect of this work is tracing a seemingly engineering-level loss tweak back to a fundamental sequence problem: whether the autoregressive assumption should occur before or after building the BT model. It is theoretically consistent and elegant.
- **Revealing DPO's Implicit Constraint $\mu'\equiv1$**: Identifying that DPO defaults to treating the entire answer as a single feedback unit explains why previous variants were forced to keep the summation inside the sigmoid. This framing is highly explanatory.
- **Decoupled Length Measures are Transferable**: The abstraction that token length and feedback length are independent provides a unified language for "any-granularity preference optimization," potentially guiding the design of new granularity-aware losses.

## Limitations & Future Work
- Experiments were only validated on mathematical reasoning (GSM8K/MATH). Evidence for generalization to other alignment scenarios like dialogue, safety, or long-form text is limited.
- The optimal granularity ($k$, $m$) for the static/adaptive families must be tuned per task; the paper lacks principled guidelines for selection, leaving practical costs to be evaluated.
- Improvements are mostly in the range of 1–3 percentage points. While stable, they are not massive. Whether "theoretical alignment" translates to larger gains in massive-scale or harder tasks remains to be verified.
- It is unclear if prefix-level BT implicit rewards truly correspond to meaningful "process-level preferences" or are just mathematically equivalent rewrites. Qualitative process analysis is missing.

## Related Work & Insights
- **vs. DPO**: Shares the same optimal solution and lack of an explicit reward model, but ADPO advances the autoregressive assumption to before building the BT model, moves the summation out of the log-sigmoid, and reveals DPO as a special case where $\mu'\equiv1$.
- **vs. TDPO / TGDPO / cDPO etc.**: These methods add KL constraints, reward guidance, or token weighting at specific positions but still **fundamentally rely on an answer-level BT model**. ADPO changes the modeling foundation to prefix-level BT, allowing it to be orthogonally combined (e.g., cADPO).
- **vs. Rafailov et al. (2024)**: That work provides a token-level soft-Q interpretation of the original answer-level BT formula **without changing the BT model itself**. ADPO expands the energy domain to the prefix closure *before* applying BT, resulting in a **different** objective function.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Fixes the sequential gap in DPO derivation, introducing prefix-level BT and two independent length measures. Very novel theoretical perspective.
- Experimental Thoroughness: ⭐⭐⭐ Solid across 4 models and 2 benchmarks plus granularity ablations, but limited to math reasoning with moderate gains.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear derivation from observation of the gap to energy redefinition, theorems, and granularity families. Table 1 comparison is excellent.
- Value: ⭐⭐⭐⭐ Provides a theoretically grounded Foundation for preference optimization that better fits autoregressive LLMs, acting as a framework for other methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Boosting Direct Preference Optimization with Penalization](boosting_direct_preference_optimization_with_penalization.md)
- [\[ICLR 2026\] SafeDPO: A Simple Approach to Direct Preference Optimization with Enhanced Safety](../../ICLR2026/llm_alignment/safedpo_preference_optimization_safety.md)
- [\[ICLR 2026\] ActiveDPO: Active Direct Preference Optimization for Sample-Efficient Alignment](../../ICLR2026/llm_alignment/activedpo_active_direct_preference_optimization_for_sample-efficient_alignment.md)
- [\[CVPR 2026\] Uncertainty-Aware Exploratory Direct Preference Optimization for Multimodal Large Language Models](../../CVPR2026/llm_alignment/uncertainty-aware_exploratory_direct_preference_optimization_for_multimodal_larg.md)
- [\[ICLR 2026\] Sharpness-Aware Minimization in Logit Space Efficiently Enhances Direct Preference Optimization](../../ICLR2026/llm_alignment/sharpness-aware_minimization_in_logit_space_efficiently_enhances_direct_preferen.md)

</div>

<!-- RELATED:END -->
