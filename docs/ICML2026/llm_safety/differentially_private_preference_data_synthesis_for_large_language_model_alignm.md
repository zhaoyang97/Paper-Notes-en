---
title: >-
  [Paper Note] Differentially Private Preference Data Synthesis for Large Language Model Alignment
description: >-
  [ICML 2026][LLM Safety][Differential Privacy] DPPrefSyn replaces "DP fine-tuning on private preference data" with "learning a DP preference reward model distribution and then synthesizing DP preference data using public…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Differential Privacy"
  - "Preference Data Synthesis"
  - "Bradley-Terry"
  - "DP-PCA"
  - "DPO/RLHF"
date: 2026-05-08
content_hash: b4c68c46169a3835
---

# Differentially Private Preference Data Synthesis for Large Language Model Alignment

**Conference**: ICML 2026  
**arXiv**: [2605.30808](https://arxiv.org/abs/2605.30808)  
**Code**: https://github.com/gfengyu/Differentially-Private-Preference-Data-Synthesis  
**Area**: LLM Security / Differential Privacy / Preference Alignment  
**Keywords**: Differential Privacy, Preference Data Synthesis, Bradley-Terry, DP-PCA, DPO/RLHF

## TL;DR
DPPrefSyn replaces "DP fine-tuning on private preference data" with "learning a DP preference reward model distribution and then synthesizing DP preference data using public prompts." By leveraging the geometric structure of Bradley-Terry linear rewards + DP-PCA + DP-KMeans clustering to capture user preference heterogeneity, it achieves a 56.5% GPT-4o win-rate on Anthropic-HH at $\varepsilon=2$, outperforming both non-private fine-tuning (55.95%) and DP-FT (37.0%).

## Background & Motivation

**Background**: LLM preference alignment (RLHF / DPO) relies on triplet data consisting of a prompt, a pair of responses, and human preference labels. These datasets (e.g., Anthropic-HH, OpenAssistant, TL;DR) often contain sensitive information such as health, identity, and political orientation in prompts, and annotations themselves may leak annotator preferences.

**Limitations of Prior Work**: Existing DP alignment works fall into three categories: (1) Label-DP (Chowdhury 2024, Zhang 2025), which only protects labels while prompts remain exposed; (2) Private fine-tuning for specific algorithms like DP-PPO (Wu 2023a), which is incompatible with DPO; (3) DP synthesis for instructions (Yu 2024) that does not target preference pairs. All three offer "partial protection" or are "algorithm-specific" and suffer from limited private data volume—human preference annotation is extremely expensive.

**Key Challenge**: Preference data exhibits strong heterogeneity (different users value different aspects: accuracy, politeness, creativity), but DP-SGD has extremely low sample efficiency in high-dimensional embedding spaces. Furthermore, it is desirable for the DP-protected output to be reusable for DPO / RLHF / various downstream LLMs without further budget consumption.

**Goal**: (1) Protect all private signals including prompt + response + label; (2) Maintain compatibility with any alignment algorithm like DPO or RLHF; (3) Surpass the utility baseline of DP fine-tuning on private data.

**Key Insight**: Transform the task from "privately fine-tuning an alignment model" to "using DP to learn a preference reward model distribution → and using it to construct synthetic preference pairs on public prompts." Public prompts do not consume budget; all budget is used to build preference models. Synthetic data can be reused arbitrarily through the DP post-processing property.

**Core Idea**: Bradley-Terry + Linear Reward $\rightarrow$ preference = sign of $\langle \theta, \phi(x, a^+) - \phi(x, a^-) \rangle$; cluster based on $\phi$ difference vectors to capture heterogeneous preferences; use DP-PCA for dimension reduction to save samples, DP-KMeans for clustering, and DP-SGD to learn linear rewards for each cluster; finally, sample across the cluster distribution on public prompts and use the corresponding reward models to select best/worst pairs.

## Method

### Overall Architecture

DPPrefSyn consists of three steps:
1.  **Preference Representation + Clustering**: For each $(x_i, a_i^+, a_i^-)$, calculate $d_i = \psi(x_i, a_i^+) - \psi(x_i, a_i^-)$; apply DP-PCA to reduce to $p=20$ dimensions (consumes $\varepsilon_0$); use DP-KMeans to partition into $K=5$ clusters (consumes $\varepsilon_1$).
2.  **DP Reward Model Training**: Learn a linear $\theta_k \in \mathbb{R}^p$ for each cluster using DP-SGD; the parallel composition theorem (disjoint clusters) ensures the total budget is $\varepsilon - \varepsilon_0 - \varepsilon_1$.
3.  **Synthetic Generation**: Compute DP histogram $\bm p \leftarrow \bm h / |\mathcal{D}_{\text{priv}}|$. For each public prompt $\tilde x_j$, the LLM generates $L=5$ candidates; sample cluster $k \sim \bm p$, calculate rewards using $\theta_k$, and select the max/min as $(\tilde a^+, \tilde a^-)$; pairs with reward differences too small (< 0.5) are discarded to ensure quality.

A PRV accountant is used for tight composition of DP-SGD steps; the post-processing property allows the synthetic data to be reused indefinitely.

### Key Designs

1.  **Bradley-Terry Linear Reward $\rightarrow$ Geometric Clustering**:
    - **Function**: Capture heterogeneous user preferences in a low-dimensional space, replacing a single global reward with a family of cluster rewards.
    - **Mechanism**: Under the BT model, $\mathbb{P}[a^+ \succ a^-] = \sigma(\langle \theta, \phi(x,a^+) - \phi(x,a^-)\rangle)$. Users with similar preferences have aligned $\phi$ difference vectors, allowing clustering by difference vectors to approximate cluster-specific $\theta_k$. Discovered clusters are interpretable (e.g., "focus on factuality" vs. "focus on politeness").
    - **Design Motivation**: A single reward cannot represent heterogeneous preferences; clustering avoids high-dimensional multi-model issues; the linear structure balances expressivity and DP-friendliness (DP-SGD sample efficiency is much higher for linear models than deep models).

2.  **Budget Configuration for DP-PCA + DP-KMeans + DP-SGD**:
    - **Function**: Improve DP-SGD sample efficiency through dimension reduction and phased budget consumption.
    - **Mechanism**: Original embeddings are 1024D; training a reward model directly via DP-SGD at this dimension requires massive samples. DP-PCA projects difference vectors to $p=20$ to retain primary signals. Budget is allocated as $\varepsilon_0$ (PCA) + $\varepsilon_1$ (KMeans) + remaining for DP-SGD. Disjoint clusters $\rightarrow$ DP parallel composition, where budget is only affected by the smallest cluster.
    - **Design Motivation**: Dimension reduction is a standard technique for high-dimensional DP training. PCA is more targeted than random projection; KMeans ensures intra-cluster preference homogeneity so linear models suffice.

3.  **Public Prompts to Save Budget + Candidate Scoring for Pair Construction**:
    - **Function**: Spend all DP budget on "building preferences" rather than "synthesizing prompts."
    - **Mechanism**: Use public prompt sets (Alpaca / SafeRLHF / XSum); for each prompt, an LLM generates 5 candidates with high temperature. Draw cluster $k$ per the DP histogram, calculate rewards with $\theta_k$, and form preference pairs from the highest/lowest. Discard pairs with reward diffs < 0.5 to avoid noisy pairs.
    - **Design Motivation**: Synthesizing prompts consumes budget and yields poor quality. Using public prompts saves this budget; distribution shifts are mitigated by $\theta_k$ capturing preference invariance (as preferences typically do not change with prompts).

## Key Experimental Results

### Main Results: GPT-4o Win-rate (Pythia-2.8B + SFT+DPO)

| Task | $\varepsilon=0$ (base) | DP-FT $\varepsilon=2$ | **DPPrefSyn $\varepsilon=2$** | DP-FT $\varepsilon=\infty$ (Non-private) |
| :--- | :--- | :--- | :--- | :--- |
| OpenAssistant | 2.11 | 6.18 | **11.04** | 8.20 |
| Anthropic-HH | 12.14 | 37.02 | **56.48** | 38.72 |
| TL;DR | 11.64 | 35.2 | **53.8** | 39.5 |

Under strong privacy ($\varepsilon = 2$), DPPrefSyn significantly outperforms DP-FT and even exceeds fully non-private DP-FT ($\varepsilon = \infty$)—**DP is no longer just a utility cost, but acts as a regularizer**.

### Privacy vs. Performance (Anthropic-HH)

| $\varepsilon$ | DP-FT win-rate | DPPrefSyn win-rate |
| :--- | :--- | :--- |
| 0.5 | 35.00 | **55.08** |
| 1 | 36.27 | **55.96** |
| 2 | 37.02 | **56.48** |
| 4 | 36.74 | **56.51** |
| 8 | 36.94 | **56.86** |
| ∞ | 38.72 | 57.53 |

DPPrefSyn remains stable at 55%+ across almost all $\varepsilon$, while DP-FT is stuck at 35-37%. DPPrefSyn is insensitive to the budget because it is only used on low-dimensional linear rewards, which is much more efficient than training an entire LLM.

### Ablation Study (OpenAssistant, $\varepsilon = 2$)

| Configuration | win-rate |
| :--- | :--- |
| Full DPPrefSyn | 11.04 |
| w/o DP-PCA (Direct 1024D DP-SGD) | 6.32 |
| w/o KMeans (Single global reward) | 8.41 |
| DP synthesized prompts instead of public | 7.95 |
| GPT-2 fine-tuned reward instead of linear | 11.21 |

DP-PCA contributes the most (-4.7 points), followed by clustering for heterogeneity (-2.6 points). Linear rewards perform similarly to full GPT-2, proving the linear structure is sufficient.

### Key Findings
- **DP Synthetic Data > Direct DP Fine-tuning**: DPPrefSyn beats DP-FT at all $\varepsilon$, challenging the common belief that synthetic data loses information.
- **Dimension Reduction is Vital for High-D DP**: Dropping DP-PCA causes a 4.7-point drop, indicating DP-SGD learns almost nothing in 1024D.
- **Heterogeneous Preference Modeling works**: Clustering provides a 2.6-point boost, confirming human preferences are multimodal.
- **Post-processing Reusability**: Once the synthetic dataset is generated, it can be used for different models/algorithms (SFT, DPO, RLHF) with zero additional budget.

## Highlights & Insights
- **Elegant "Trio" of DP-PCA + Linear Reward + Clustering**: Each component solves a specific pain point of high-dimensional DP training (sample efficiency / expressivity / heterogeneity). The combination crosses the utility-privacy boundary.
- **Maximizing Post-processing Property**: Synthesis detaches the data from DP control, allowing for arbitrary reuse—a fundamental advantage of DP synthesis over DP fine-tuning that this paper fully exploits.
- **"DP as Regularization" Phenomenon**: DPPrefSyn at $\varepsilon=2$ outperforms the non-private baseline, suggesting DP noise acts as a regularizer on heterogeneous data, mitigating overfitting to specific annotator biases.
- **Insight on Public vs. Private Prompts**: The authors argue that "user preference decouples from prompt distribution," allowing public prompts to carry private preferences—a logic applicable to other preference tasks (recommendation, advertising).

## Limitations & Future Work
- The linear reward assumption might be too strong; it lacks expressivity for non-linear preferences (e.g., compositional or long-range dependency judgments).
- The choice of $K=5$ clusters lacks a principled method and relies on heuristics; too many or too few clusters hurt performance.
- If public prompt distributions deviate severely from private ones, coverage may be insufficient; there is a lack of quantitative analysis on distribution shift.
- Validation is limited to Pythia-2.8B; biases from DP-PCA dimension reduction might be amplified in larger models (e.g., 13B+).

## Related Work & Insights
- **vs DP-FT / DP-PPO / DP-RLHF**: These fine-tune the alignment model directly; budgets must be re-spent for every algorithm or model change. DPPrefSyn: DP once, reuse many times.
- **vs label-DP (Chowdhury / Zhang)**: Protections are label-only; prompts are still leaked. DPPrefSyn protects everything.
- **vs DP synthetic instructions (Yu 2024)**: General instruction synthesis isn't optimized for preference. DPPrefSyn synthesizes preference pairs directly for better alignment.
- **vs Aug-PE (Xie 2024)**: Generic text synthesis via LLM API iterations. DPPrefSyn is more specialized for preference pairs using the BT geometric structure.
- **Insight**: Replacing "DP direct training" with "DP learn abstraction $\rightarrow$ Synthetic data $\rightarrow$ Reuse" can be generalized to many supervised tasks requiring label protection (medical, legal, recommendation).

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First dedicated DP preference pair synthesis; systematic strategy combining BT geometry + DP-PCA + clustering.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 tasks × 5 $\varepsilon$ values × multiple models × detailed ablation; comprehensive head-to-head with DP-FT.
- Writing Quality: ⭐⭐⭐⭐ Clear three-step algorithm; budget allocation well-explained; Fig 1 is intuitive; geometric arguments for BT-clustering could be deeper.
- Value: ⭐⭐⭐⭐⭐ DP alignment is a compliance necessity for enterprise LLM deployment; providing a pipeline ready for industry adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Privacy Amplification in Differentially Private Zeroth-Order Optimization with Hidden States](privacy_amplification_in_differentially_private_zeroth-order_optimization_with_h.md)
- [\[ICML 2026\] ACTG-ARL: Differentially Private Conditional Text Generation with RL-Boosted Control](actg-arl_differentially_private_conditional_text_generation_with_rl-boosted_cont.md)
- [\[ICML 2026\] Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences](federated_variational_preference_alignment_with_gumbel-softmax_prior_for_persona.md)
- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ACL 2026\] Differentially Private Synthetic Text Generation for Retrieval-Augmented Generation (RAG)](../../ACL2026/llm_safety/differentially_private_synthetic_text_generation_for_retrieval-augmented_generat.md)

</div>

<!-- RELATED:END -->
