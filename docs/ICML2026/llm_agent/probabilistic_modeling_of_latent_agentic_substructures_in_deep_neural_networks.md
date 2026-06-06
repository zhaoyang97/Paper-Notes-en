---
title: >-
  [Paper Note] Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks
description: >-
  [ICML 2026][LLM Agent][Sub-agent structures] The authors formalize neural networks (especially LLMs) as composite agents synthesized through the log-weighted pooling of multiple implicit sub-agents (each represented as a…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Sub-agent structures"
  - "Logarithmic pooling"
  - "Epistemic utility"
  - "Waluigi effect"
  - "Alignment theory"
date: 2026-05-08
content_hash: 199f10c84bd01b8c
---

# Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2509.06701](https://arxiv.org/abs/2509.06701)  
**Code**: None (Theoretical paper)  
**Area**: LLM Agent / AI Alignment / Probabilistic Modeling  
**Keywords**: Sub-agent structures, Logarithmic pooling, Epistemic utility, Waluigi effect, Alignment theory

## TL;DR
The authors formalize neural networks (especially LLMs) as composite agents synthesized through the log-weighted pooling of multiple implicit sub-agents (each represented as a probability distribution over outcomes). Within the framework of epistemic utility $W_i(o)=\log P_i(o)$, they prove that "strict unanimity" (where all sub-agents benefit) is impossible under linear pooling or binary outcomes but feasible when $|\mathcal O|\ge 3$. This leads to the alignment principle that "explicitly manifesting the Waluigi persona before suppression" is strictly superior to "only reinforcing the Luigi persona."

## Background & Motivation

**Background**: Viewing LLMs as a collection of competing personae or priors is a common empirical observation in alignment. Humans possess dual pro-social and anti-social drives, and LLMs exhibit phenomena like self-preservation and the Waluigi effect (where reinforcing a "good" persona inadvertently triggers an adversarial one). However, these observations remain descriptive without a rigorous mathematical framework. Meanwhile, machine learning has long utilized additive combinations like ensembles, Product of Experts (PoE), and multi-head attention, where the final softmax output is essentially a weighted sum of logits.

**Limitations of Prior Work**: (1) While economics and game theory have mature theories for aggregating preferences via utility functions (social welfare aggregation, unanimity), these have not been applied to neural network analysis. (2) Current LLM alignment discussions regarding the Waluigi effect rely on intuition and cannot specify conditions under which sub-agents stably synthesize or form inconsistent composites. (3) There is a lack of mathematical proof comparing the effectiveness of "manifest then suppress" versus "direct reinforcement" in RLHF.

**Key Challenge**: Since the training objective of neural networks is $-\log P(\cdot)$ (cross-entropy), the utility should be defined as $\log P(\cdot)$. However, "strict unanimity" under logarithmic utility is a more stringent condition than under linear utility, being highly sensitive to the outcome space structure ($|\mathcal O|=2$ vs $\ge 3$) and the pooling form (linear vs logarithmic). Deriving these conclusions requires a rigorous definition of composite agents and aggregation forms compatible with neural network architectures.

**Goal**: (i) Propose a formal framework where sub-agents are probability distributions and composite agents are formed via log-pooling. (ii) Prove clear boundaries for the existence of strict unanimity (linear vs log-pooling, $|\mathcal O|=2$ vs $\ge 3$). (iii) Establish properties such as cloning invariance, continuity, and openness for recursion and stability. (iv) Reinterpret the Waluigi phenomenon and derive mathematically guaranteed alignment principles.

**Key Insight**: The authors note that the final layer of modern LLMs consists of linear logits followed by a softmax. Consequently, any additive decomposition on the logits (ensemble, PoE, multi-head) is **mathematically equivalent** to the logarithmic pooling of the corresponding distributions. This naturally embeds the "persona synthesis" problem of alignment into the economic and information-theoretic framework of *logarithmic pooling + epistemic utility*.

**Core Idea**: Treat the LLM as a log-pooled composite agent. Use the welfare gap $\Delta_{P_i}(P)=H(P_i)-H(P)-\mathrm{KL}(P\|P_i)$ to quantify the benefit to each sub-agent, and use this formula to derive the first-order relationship between Waluigi and Luigi personae.

## Method

### Overall Architecture
Each sub-agent $i$ is assigned a pair (belief $P_i$, welfare $W_i$). The belief of the composite agent is given by logarithmic pooling: $P(o)=\frac1Z\prod_j P_j(o)^{\beta_j}$, where weights $\beta_j\ge 0$ and $\sum_j\beta_j=1$. When utility is defined as $W_i(o)=\log P_i(o)$ (epistemic utility), the condition for "sub-agent $i$ benefiting from synthesis" is the welfare gap $\Delta_{P_i}(P)=\mathbb E_P[\log P_i]-\mathbb E_{P_i}[\log P_i]\ge 0$, which can be expressed in information-theoretic terms as $\Delta_i=H(P_i)-H(P)-\mathrm{KL}(P\|P_i)$. A "strictly unanimous group" $\mathcal U_{\text{strict}}$ requires all $i$ to satisfy $\Delta_i > 0$. Based on this core definition, the authors derive possibility boundaries, recursion stability, and alignment analysis for Luigi-Waluigi.

### Key Designs

1.  **Log-pooling as a "Natural" Aggregation Rule for Neural Networks**:
    *   **Function**: Unifies additive logit decompositions (ensembles, PoE, multi-head attention) as sub-agent synthesis under log-pooling, embedding persona aggregation into classical economic aggregation theory.
    *   **Mechanism**: Log-pooling $P(o)=\frac1Z\prod_j P_j(o)^{\beta_j}$ is equivalent to a weighted sum of logits $\log P_i$ followed by a softmax, matching the final layer structure of modern Transformers. Epistemic utility $W_i(o)=\log P_i(o)$ maps to the cross-entropy training objective. Since gradients propagate through the $\log P(\cdot)$ term, $\log P$ is the implicit utility the network optimizes.
    *   **Design Motivation**: To apply economic welfare aggregation to neural networks, the aggregation form must match (i) the network architecture (linear + softmax) and (ii) the training objective (log-likelihood). Log-pooling is the unique form satisfying both.

2.  **Possibility Boundaries of Strict Unanimity (Theorem 8/9/10)**:
    *   **Function**: Characterizes conditions under which all sub-agents can benefit simultaneously.
    *   **Mechanism**: (i) **Binary Outcome Impossibility Theorem**: For $|\mathcal O|=2$, no non-trivial weights can satisfy $\Delta_i\ge 0$ for all agents with at least one being strict; log-pooling in binary space is a zero-sum struggle. (ii) **$|\mathcal O|\ge 3$ Existence Theorem**: It is possible to explicitly construct $\{P_i\}_{i=1}^n, \beta_i$ such that $\mathbb E_P[\log P_i]>\mathbb E_{P_i}[\log P_i]$ holds strictly for all $i$. (iii) **Linear Pooling Impossibility Theorem**: Under $P_C^{\text{lin}}(o)=\sum\beta_i P_i(o)$ and $W_i(o)=\log P_i(o)$, strictly unanimous synthesis never exists.
    *   **Design Motivation**: These theorems indicate: (a) multi-persona LLMs must utilize log-pooling rather than simple weighted averages; (b) toy problems with tiny outcome spaces ($|\mathcal O|=2$) are unsuitable for this theory; (c) at least 3 outcomes are needed for non-trivial synthesis. This implies that binary "safe/unsafe" alignment labels are fundamentally degenerate.

3.  **Recursion, Stability, and Luigi-Waluigi Alignment Principles**:
    *   **Function**: Provides structural guarantees for the "synthesis-decomposition-resynthesis" process and proves the superiority of "manifest-then-suppress" for Waluigi.
    *   **Mechanism**: (a) **Lemma 13 (Invariance under Compatible Splitting)**: Replacing $P_i$ with $m$ weight-compatible sub-agents does not change the global pool. (b) **Theorem 14 (Parent Benefit does not Inherit to Children)**: A parent agent may satisfy $\Delta_{P_i}(P)>0$ while its split child satisfies $\Delta_{P_{i,j}}(P)<0$, implying alignment cannot be judged solely at the top level. (c) **Theorem 17 (Openness)**: If $P\in\mathcal U_{\text{strict}}$, a neighborhood of $P$ also lies in $\mathcal U_{\text{strict}}$. (d) **Waluigi Analysis**: Using a Hilbert space construction, the authors prove that when RLHF reinforces a Luigi persona under a KL budget, an adversarial Waluigi sub-persona is inevitably activated. A "manifest-then-suppress" strategy yields a strictly larger first-order alignment error reduction than pure Luigi reinforcement.
    *   **Design Motivation**: Stability results (Openness) ensure that small parameter updates in RLHF do not abruptly destroy the persona structure. Theorem 14 explains why top-level RLHF metrics may be high while sub-persona issues persist.

### Loss & Training
This is a theoretical paper and does not involve a specific training pipeline. However, all conclusions are based on the standard setting of additive logit structures, cross-entropy training, and KL-regularized RLHF (e.g., DPO/PPO).

## Key Experimental Results

### Main Results
The "experiments" consist of theorems and closed-form constructions.

| Pooling Form | Utility | $|\mathcal O|$ | Strict Unanimity Reachable? | Source |
| :--- | :--- | :--- | :--- | :--- |
| Linear | Epistemic $\log P_i$ | Any | **Impossible** | Theorem 10 |
| Logarithmic | Epistemic $\log P_i$ | $2$ | **Impossible** | Theorem 8 |
| Logarithmic | Epistemic $\log P_i$ | $\ge 3$ | **Reachable** (via construction) | Theorem 9 |
| Logarithmic | General $W_i$ | $n\ge 2$ | Possible for some $\{P_i,W_i,\beta_i\}$ | Theorem 6 |

**Alignment Strategy Comparison**:

| Strategy | First-order Alignment Error Reduction | Strictness |
| :--- | :--- | :--- |
| Only reinforce Luigi | $\Delta_{\text{Luigi}}^{(1)}$ | Baseline |
| Manifest-then-suppress Waluigi | Strictly greater than $\Delta_{\text{Luigi}}^{(1)}$ | Proven under KL budget |

### Ablation Study

| Configuration | Key Conclusion | Explanation |
| :--- | :--- | :--- |
| Utility = $\log P_i$ vs. General $W_i$ | Unanimity is easier under general $W_i$ | Epistemic utility is more constrained. |
| Compatible Split (Lemma 13) vs. Non-compatible | Former keeps pool invariant; latter shows parent benefit $\neq$ child benefit | Top-level alignment is insufficient. |
| Strict vs. Non-strict Unanimity | Non-strict allows trivial duplication; Strict is stable in neighborhoods | Distinguishes meaningful synthesis from replication. |

### Key Findings
*   The outcome space dimensionality $|\mathcal O| \ge 3$ is a hard threshold for strict unanimity. Simplifying alignment to "safe vs. unsafe" binaries places the system in the impossibility region of Theorem 8.
*   The phenomenon where "top-level model performance is good but sub-personae are damaged" is a mathematical theorem (Theorem 14); RLHF cannot rely solely on reward model scores.
*   The "manifest-then-suppress" principle for Waluigi is formally proven to be superior to "direct Luigi reinforcement," providing guidance for negative sampling and jailbreak defense.

## Highlights & Insights
*   The "three-in-one" connection between log-pooling, logit-additive architecture, and cross-entropy training is elegant—it allows alignment theory to leverage economic aggregation tools directly.
*   Theorem 14 explains why high reward model scores can coexist with deceptive behavior: rewards track the top-level composite, while sub-personae may still suffer.
*   The "Openness" result provides geometric intuition for KL-budget training: if the base model is in a strictly unanimous region, small updates stay within that region, explaining the theoretical stability of KL-regularized RLHF.
*   The "manifest-then-suppress" principle suggests that for jailbreaks or deceptive behaviors, one should first elicit the behavior to construct negative samples, then use them for contrastive training.

## Limitations & Future Work
*   The assumption of a finite outcome space with $P_i > 0$ needs extension to the massive vocabularies and long sequences of actual LLMs.
*   While epistemic utility $W_i = \log P_i$ is natural for training, real-world RLHF involves complex implicit utilities from reward models; extending to non-epistemic utilities is needed.
*   The Luigi-Waluigi conclusions are first-order (local expansions for small KL budgets); large-scale alignment fine-tuning requires higher-order analysis.
*   Lack of empirical experiments to quantify the actual performance gap of the proposed principles in DPO/RLHF settings.

## Related Work & Insights
*   **vs. Opinion Pooling**: Bridges decades of economic research on pooling axioms to neural network analysis.
*   **vs. Active Inference**: Shares the "agent as generative model" view but focuses on the mathematical structure of sub-agent synthesis.
*   **vs. Waluigi Effect empiricals**: Provides the first formal proof under KL budgets for observations previously limited to blogs/discussions.
*   **vs. PoE / MoE**: PoE is essentially log-pooling; this work proves reachability conditions for such combinations.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Introduces log-pooling and epistemic utility to alignment theory with formal proofs for empirical heuristics.
*   Experimental Thoroughness: ⭐⭐ Purely theoretical with no LLM-based verification.
*   Writing Quality: ⭐⭐⭐⭐ Clear motivation regarding the natural emergence of log-pooling from network architecture.
*   Value: ⭐⭐⭐⭐ Provides a rigorous framework for vague concepts like "persona synthesis" and the "Waluigi effect."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](../../AAAI2026/llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)
- [\[ICML 2026\] Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models](hunt_instead_of_wait_evaluating_deep_data_research_on_large_language_models.md)
- [\[ICLR 2026\] A Benchmark for Deep Information Synthesis (DeepSynth)](../../ICLR2026/llm_agent/a_benchmark_for_deep_information_synthesis.md)
- [\[ICML 2026\] AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](agentxray_white-boxing_agentic_systems_via_workflow_reconstruction.md)
- [\[ICML 2026\] HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation](hawkesllm_semantic_uncertainty_propagation_in_agentic_text_simulation.md)

</div>

<!-- RELATED:END -->
