---
title: >-
  [Paper Note] Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks
description: >-
  [ICML 2026][Interpretability][Sub-agent structure] The authors formalize neural networks (especially LLMs) as composite agents synthesized from multiple implicit sub-agents (each a probability distribution over outcomes)…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Sub-agent structure"
  - "log pooling"
  - "cognitive utility"
  - "Waluigi effect"
  - "alignment theory"
date: 2026-05-08
content_hash: 24e12295c7efc727
---

# Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2509.06701](https://arxiv.org/abs/2509.06701)  
**Code**: None (theoretical paper)  
**Area**: LLM Agent / AI Alignment / Probabilistic Modeling  
**Keywords**: Sub-agent structure, log pooling, cognitive utility, Waluigi effect, alignment theory

## TL;DR
The authors formalize neural networks (especially LLMs) as composite agents synthesized from multiple implicit sub-agents (each a probability distribution over outcomes) via log-weighted pooling. Within the cognitive utility framework $W_i(o)=\log P_i(o)$, they prove that "strict unanimity benefit" is impossible under linear pooling or binary outcomes, but feasible when $|\mathcal O|\ge 3$. This leads to the alignment principle that "explicitly manifesting Waluigi before suppression" is strictly superior to "only reinforcing Luigi".

## Background & Motivation

**Background**: Viewing LLMs as collections of internally competing personas/priors is a common empirical observation in alignment—humans have both prosocial and antisocial drives, and LLMs exhibit self-preservation, Waluigi (reinforcing the benevolent persona triggers adversarial personas), etc. However, these are descriptive, lacking a rigorous mathematical framework. In machine learning, ensemble, Product of Experts, and multi-head architectures are additive combinations, with the final softmax output essentially a weighted sum on the logit level.

**Limitations of Prior Work**: (1) Economics/game theory has mature theory for aggregating multi-agent preferences via utility functions (social welfare aggregation, unanimity), but this has not been applied to neural network analysis; (2) Current LLM alignment discussions of phenomena like Waluigi are intuitive, unable to answer key questions such as "under what conditions can sub-agents stably compose" or "which sub-agent combinations cannot form a composite with unanimous benefit"; (3) Empirical observations of "manifest then suppress" versus "directly reinforce the benevolent" RLHF strategies lack mathematical proof.

**Key Challenge**: Neural network training targets $-\log P(\cdot)$ (cross-entropy), so utility should be $\log P(\cdot)$. However, "strict unanimity benefit" under log utility is a sharper condition than under linear utility, highly sensitive to the outcome space structure ($|\mathcal O|=2$ vs $\ge 3$) and pooling form (linear vs log). Deriving these sharp results requires a rigorous definition of composite agents and aggregation forms compatible with neural network structure.

**Goal**: (i) Propose a formal framework where sub-agents are probability distributions and composite agents are formed via log pooling; (ii) Prove clear boundaries for the existence/non-existence of strict unanimity benefit (linear vs log pooling, $|\mathcal O|=2$ vs $\ge 3$); (iii) Establish properties such as cloning invariance, continuity, and openness; (iv) Reinterpret the Waluigi phenomenon in LLMs using this framework and derive mathematically guaranteed alignment principles.

**Key Insight**: The authors observe that the final layer of modern LLMs is linear logit plus softmax, so any additive decomposition on logits (ensemble, PoE, multi-head) is mathematically equivalent to log pooling of the corresponding distributions. This naturally embeds the alignment problem of "persona synthesis" into the economic/information-theoretic framework of log pooling and log utility.

**Core Idea**: Treat the LLM as a composite agent via log pooling, quantify each sub-agent's benefit using the welfare gap $\Delta_{P_i}(P)=H(P_i)-H(P)-\mathrm{KL}(P\|P_i)$, and use this formula to derive the first-order relationship between Waluigi and Luigi.

## Method

### Overall Architecture
Each sub-agent $i$ is associated with a pair (belief $P_i$, welfare $W_i$). The composite agent's belief is given by log pooling $P(o)=\frac1Z\prod_j P_j(o)^{\beta_j}$, with weights $\beta_j\ge 0,\sum_j\beta_j=1$. When utility is $W_i(o)=\log P_i(o)$ (cognitive utility), the condition for "sub-agent $i$ benefiting from the composition" is equivalently the welfare gap $\Delta_{P_i}(P)=\mathbb E_P[\log P_i]-\mathbb E_{P_i}[\log P_i]\ge 0$, which can be rewritten in information-theoretic form as $\Delta_i=H(P_i)-H(P)-\mathrm{KL}(P\|P_i)$. The "strict unanimity group" $\mathcal U_{\text{strict}}$ requires all $i$ to have strictly $>0$. Around this core definition, the authors derive possibility boundaries, recursion and stability properties, and ultimately apply the analysis to Luigi-Waluigi alignment.

### Key Designs

1. **Log Pooling as the "Natural" Sub-agent Aggregation Rule in Neural Networks**:

    - **Function**: Unifies additive decompositions of neural network final-layer logits (ensemble, Product of Experts, multi-head attention) as sub-agent synthesis under log pooling, embedding the "persona aggregation" problem into established economic aggregation theory.
    - **Mechanism**: Log pooling $P(o)=\frac1Z\prod_j P_j(o)^{\beta_j}$ is equivalent to weighted sum on logits $\log P_i$ followed by softmax, matching the form of modern transformer final layers. Cognitive utility $W_i(o)=\log P_i(o)$ aligns with the cross-entropy training objective—gradients propagate through the $\log P(\cdot)$ term, so $\log P$ is the network's implicit utility. Composite agents are defined by $\mathbb E_P[W_i]\ge \mathbb E_{P_i}[W_i]$ and strictly unanimous groups.
    - **Design Motivation**: To apply economic welfare aggregation to neural networks, the aggregation form must match both (i) network structure (final linear + softmax) and (ii) training objective (log-likelihood). Log pooling uniquely satisfies both, dictated by network structure and training objective.

2. **Possibility Boundaries for Strict Unanimity Benefit (Theorem 8/9/10)**:

    - **Function**: Clearly characterizes under what conditions "all sub-agents benefit simultaneously" is achievable; the most technical part of the paper.
    - **Mechanism**: (i) **Impossibility Theorem for Binary Outcomes**: For $|\mathcal O|=2$, with any nontrivial weights, it is impossible for both agents to have $\Delta_i\ge 0$ with at least one strict—intuitively, in binary space, log pooling is zero-sum, one agent's gain is another's loss. (ii) **Existence Theorem for $|\mathcal O|\ge 3$**: Explicit construction of $\{P_i\}_{i=1}^n,\beta_i$ such that $\mathbb E_P[\log P_i]>\mathbb E_{P_i}[\log P_i]$ holds strictly for all $i$. (iii) **Impossibility Theorem for Linear Pooling**: For $P_C^{\text{lin}}(o)=\sum\beta_i P_i(o)$ and $W_i(o)=\log P_i(o)$, strict unanimity benefit is never achievable—intuitively, linear pooling is equivalent to random dictatorship, and adversarial agents as dictators severely harm others.
    - **Design Motivation**: These theorems provide concrete guidance: (a) Multi-persona LLMs must use log pooling, not simple weighted averaging (matching neural network structure); (b) Toy problems with binary outcome spaces are unsuitable for this theory; (c) At least 3 outcomes are needed for nontrivial synthesis. This also means binary "safe/unsafe" alignment settings are inherently degenerate.

3. **Recursion, Stability, and Luigi-Waluigi Alignment Principle**:

    - **Function**: Establishes structural guarantees (cloning invariance, continuity, openness) for recursive "composition–decomposition–recomposition" processes, and rigorously proves that "manifest-then-suppress Waluigi" is superior to "only reinforce Luigi".
    - **Mechanism**: (a) **Lemma 13 (Pooling Invariance under Compatible Splitting)**: Replacing $P_i$ with $m$ weight-compatible sub-agents does not change the global pool. (b) **Theorem 14 (Parent Benefit Does Not Transfer to Children)**: Examples can be constructed where the parent agent $\Delta_{P_1}(P)>0$ but the split child $\Delta_{P_{1,1}}(P)<0$, indicating alignment cannot be judged solely at the top level. (c) **Theorem 17 (Openness)**: If $P\in\mathcal U_{\text{strict}}$, then some neighborhood of $P$ is also in $\mathcal U_{\text{strict}}$—unanimity benefit is locally stable. (d) Section 5 uses $L=\log P$, $l_i=\log P_i$, and $P$-centered profile $v_i(o)=l_i(o)-\mathbb E_P[l_i]$ to construct a Hilbert space analysis, proving that when RLHF applies a $\mathrm{KL}$ budget constraint to the parent agent $P$ and reinforces the Luigi persona, the adversarial Waluigi sub-agent is necessarily activated in the opposite direction; whereas "manifesting Waluigi before unified suppression" achieves strictly greater first-order alignment error reduction than simply reinforcing Luigi.
    - **Design Motivation**: Stability results (especially openness) underpin the Luigi-Waluigi analysis—only if strict unanimity benefit is an open set can RLHF's small parameter updates avoid suddenly destroying persona structure. Theorem 14 explains why RLHF may pass top-level metrics while sub-persona issues persist. The Luigi-Waluigi result provides the first formal proof for the "manifest-then-suppress" heuristic.

### Loss & Training
As a theoretical paper, no training procedures are involved, but all results are based on the standard setup of "final layer logit additive structure + cross-entropy training + KL budget RLHF constraint", corresponding to KL-regularized DPO/PPO fine-tuning in practical RLHF.

## Key Experimental Results

### Main Results
The paper does not contain traditional empirical experiments; the main "experiments" are theorems and closed-form constructions. The results can be summarized in two "theory-assertion" tables:

| Pooling Form | Utility | $|\mathcal O|$ | Is Strict Unanimity Benefit Achievable? | Source |
|--------------|---------|----------------|-----------------------------------------|--------|
| Linear       | Cognitive utility $\log P_i$ | Any   | **Impossible**                        | Theorem 10 |
| Log          | Cognitive utility $\log P_i$ | $2$   | **Impossible**                        | Theorem 8  |
| Log          | Cognitive utility $\log P_i$ | $\ge 3$ | **Achievable**, explicit construction | Theorem 9  |
| Log          | General welfare $W_i$        | Any $n\ge 2$ | Exists for some $\{P_i,W_i,\beta_i\}$ | Theorem 6  |

Core theorem for alignment applications (Section 5):

| Strategy                       | First-order Alignment Error Reduction | Strictness |
|---------------------------------|--------------------------------------|-----------|
| Only reinforce Luigi (increase $\beta_{\text{Luigi}}$) | $\Delta_{\text{Luigi}}^{(1)}$ | baseline  |
| Manifest-then-suppress Waluigi | Strictly greater than $\Delta_{\text{Luigi}}^{(1)}$ | Proven under KL budget constraint |

### Ablation Study

| Configuration | Key Conclusion | Notes |
|---------------|---------------|-------|
| Welfare = $\log P_i$ vs general $W_i$ | General $W_i$ more easily achieves unanimity benefit (Theorem 6) | Cognitive utility is "narrower" and more restrictive |
| Pool-compatible splitting (Lemma 13) vs incompatible splitting (Theorem 14) | The former preserves the pool, the latter does not transfer parent benefit to children | Top-level alignment does not guarantee sub-level alignment |
| Strict vs non-strict unanimity | Non-strict allows duplication (Lemma 48), strict is locally stable (Theorem 17) but prohibits trivial duplication (Theorem 55) | Distinguishes meaningful synthesis from degenerate replication |

### Key Findings
- "Outcome space dimension $\ge 3$" is a hard threshold for the possibility of strict unanimity benefit. Any alignment setting reduced to binary "safe vs unsafe" falls into the impossibility region of Theorem 8.
- "Good top-level model performance but damaged sub-personas" is not an illusion but a mathematical theorem (Theorem 14)—RLHF cannot rely solely on reward model scores.
- "Manifesting Waluigi before suppression" is strictly superior to "directly reinforcing Luigi", now with formal proof. This directly informs negative sampling strategies and jailbreak defense training in RLHF/DPO.

## Highlights & Insights
- The unification of log pooling, neural network final-layer structure, and cross-entropy training is an elegant observation—it means alignment theory can leverage the entire toolkit of economic aggregation, without reinventing the wheel.
- Theorem 14 (parent benefit does not transfer to children) explains the recurring empirical phenomenon in alignment where "reward model scores are high but deceptive behavior persists": reward is measured at the top level, while sub-personas may still be strictly harmed.
- The "openness" result provides geometric intuition for KL-budgeted RLHF training: as long as the base model is in the strict unanimity benefit region, small updates remain within the region, so KL-budgeted training is theoretically stable.
- "Manifest-then-suppress" is a concrete RLHF operational principle—eliciting jailbreak or deceptive behaviors to construct negative samples, then using them for contrastive training, is strictly superior to only amplifying positive samples.

## Limitations & Future Work
- The entire analysis assumes a finite outcome space and all $P_i>0$; rigorous extension is needed for real LLMs with large vocabularies and long-sequence conditionals (the paper absorbs state into $\mathcal O$ to sidestep this).
- Log utility $W_i=\log P_i$ is "naturally given by the training objective", but practical RLHF involves reward models, preference learning, and more complex implicit utilities; extending to these non-epistemic utilities is needed for direct industrial RLHF application.
- The Luigi-Waluigi result in Section 5 is first-order (i.e., local expansion under small KL budgets); higher-order analysis is needed for large-scale alignment fine-tuning.
- The paper lacks empirical experiments to verify the actual numerical gap between "manifest-then-suppress" and pure Luigi reinforcement; future work should conduct controlled experiments in RLHF/DPO.

## Related Work & Insights
- **vs Opinion Pooling Classical Theory**: Economics has studied axiomatic properties of log vs linear pooling for decades; this paper imports these results into neural network analysis.
- **vs Active Inference (Friston et al.)**: Active inference posits "agent = generative model + minimize free energy"; this work borrows the probabilistic agent perspective but focuses more on the mathematical structure of sub-agent synthesis.
- **vs Waluigi Effect Empirical Discussion**: Previously limited to blog/twitter observations, this paper provides the first strict first-order proof under KL budget, and derives the manifest-then-suppress principle.
- **vs Product of Experts / Mixture of Experts**: PoE is essentially log pooling; this paper proves the stability/achievability conditions for PoE-style combinations—MoE gating can be viewed as dynamic $\beta_i$ selection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces log pooling + cognitive utility framework to alignment theory, and provides the first formal proof that "manifest-then-suppress" is strictly superior to direct reinforcement in the alignment literature.
- Experimental Thoroughness: ⭐⭐ Purely theoretical, no numerical experiments or LLM validation; all results are theorem-based.
- Writing Quality: ⭐⭐⭐⭐ The motivation for why neural network final-layer structure + cross-entropy training naturally leads to log pooling is well explained; the arrangement of possibility and recursion theorems is clear.
- Value: ⭐⭐⭐⭐ Methodologically valuable for alignment researchers and RLHF practitioners—provides the first rigorous framework for concepts like "persona synthesis" and the "Waluigi effect".

## Related Papers

- [\[AAAI 2026\] BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](../../AAAI2026/llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)
- [\[ICLR 2026\] A Benchmark for Deep Information Synthesis (DeepSynth)](../../ICLR2026/llm_agent/a_benchmark_for_deep_information_synthesis.md)
- [\[NeurIPS 2025\] Deep Video Discovery: Agentic Search with Tool Use for Long-form Video Understanding](../../NeurIPS2025/llm_agent/deep_video_discovery_agentic_search_with_tool_use_for_longfo.md)
- [\[AAAI 2026\] Structured Personalization: Modeling Constraints as Matroids for Data-Minimal LLM Agents](../../AAAI2026/llm_agent/structured_personalization_modeling_constraints_as_matroids_for_data-minimal_llm.md)
- [\[ICML 2026\] Position: Agentic AI Orchestration Should Be Bayes-Consistent](position_agentic_ai_orchestration_should_be_bayes-consistent.md)

</div>

<!-- RELATED:END -->
