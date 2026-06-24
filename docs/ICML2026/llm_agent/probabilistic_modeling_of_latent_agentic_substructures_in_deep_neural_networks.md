---
title: >-
  [Paper Note] Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks
description: >-
  [ICML 2026][LLM Agent][Sub-agent structures] The authors formalize neural networks (specifically LLMs) as composite agents synthesized through the log-weighted pooling of multiple implicit sub-agents (each defined as a probability distribution over outcomes). Under the framework of epistemic utility $W_i(o)=\log P_i(o)$, it is proven that "strict unanimity" is impossible under linear pooling or binary outcomes, but feasible when $|\mathcal O|\ge 3$. Consequently…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Sub-agent structures"
  - "Log-pooling"
  - "Epistemic utility"
  - "Waluigi effect"
  - "Alignment theory"
date: 2026-05-08
content_hash: 756a39ff93780d01
---

# Probabilistic Modeling of Latent Agentic Substructures in Deep Neural Networks

**Conference**: ICML 2026  
**arXiv**: [2509.06701](https://arxiv.org/abs/2509.06701)  
**Code**: None (Theoretical paper)  
**Area**: LLM Agent / AI Alignment / Probabilistic Modeling  
**Keywords**: Sub-agent structures, Log-pooling, Epistemic utility, Waluigi effect, Alignment theory

## TL;DR
The authors formalize neural networks (specifically LLMs) as composite agents synthesized through the log-weighted pooling of multiple implicit sub-agents (each defined as a probability distribution over outcomes). Under the framework of epistemic utility $W_i(o)=\log P_i(o)$, it is proven that "strict unanimity" is impossible under linear pooling or binary outcomes, but feasible when $|\mathcal O|\ge 3$. Consequently, an alignment principle is derived: "explicitly manifesting Waluigi before suppression" is strictly superior to "only reinforcing Luigi."

## Background & Motivation

**Background**: Viewing LLMs as a collection of competing internal personas or priors is a common empirical observation in the alignment field—humans possess dual pro-social and anti-social drives, and LLMs exhibit self-preservation and the Waluigi effect (where reinforcing a kind persona inadvertently triggers an adversarial one). However, these observations remain descriptive and lack a rigorous mathematical framework. Meanwhile, additive combinations like ensembles, Product of Experts, and multi-head structures have long existed in machine learning, where their final softmax outputs are essentially weighted sums of logits.

**Limitations of Prior Work**: (1) While aggregating multi-agent preferences using utility functions (social welfare aggregation, unanimity) is a mature theory in economics and game theory, it has not been applied to neural network analysis; (2) Current LLM alignment discussions regarding phenomena like the Waluigi effect rely on intuition and cannot answer critical questions such as the conditions under which sub-agents can stably synthesize or which sub-agent combinations cannot form a unanimously beneficial composite; (3) Between "manifesting then suppressing" and "directly reinforcing the benevolent persona" in RLHF, a mathematical proof of superiority is missing.

**Key Challenge**: The training objective of neural networks is $-\log P(\cdot)$ (cross-entropy), suggesting utility should be $\log P(\cdot)$. However, "strict unanimity" under logarithmic utility is a sharper condition than under linear utility, being highly sensitive to the structure of the outcome space ($|\mathcal O|=2$ vs $\ge 3$) and the pooling form (linear vs logarithmic). Deriving these sharp conclusions requires a rigorous definition of composite agents and aggregation forms compatible with neural network architectures.

**Goal**: (i) Propose a formal framework where "sub-agents are probability distributions and composite agents are log-pooled"; (ii) Prove clear boundaries for the existence or non-existence of strict unanimity (linear vs log-pooling, $|\mathcal O|=2$ vs $\ge 3$); (iii) Establish recursive and stability properties including cloning invariance, continuity, and openness; (iv) Reinterpret the Waluigi effect in LLMs and derive alignment principles with mathematical guarantees.

**Key Insight**: The authors observe that the final layer of modern LLMs consists of linear logits followed by a softmax. Thus, any additive decomposition on the logits (ensemble, PoE, multi-head) is mathematically **exactly equivalent** to the log-pooling of the corresponding distributions. This naturally embeds the alignment problem of "persona synthesis" into the economic and information-theoretic framework of *logarithmic pooling + epistemic utility*.

**Core Idea**: By treating the LLM as a log-pooled composite agent, the welfare gap $\Delta_{P_i}(P)=H(P_i)-H(P)-\mathrm{KL}(P\|P_i)$ is used to quantify the benefit to each sub-agent. This formula is then used to derive the first-order relationship between Waluigi and Luigi.

## Method

### Overall Architecture
Every sub-agent $i$ is paired with a (belief $P_i$, welfare $W_i$). The belief of the composite agent is given by log-pooling: $P(o)=\frac1Z \prod_j P_j(o)^{\beta_j}$, with weights $\beta_j\ge 0, \sum_j \beta_j=1$. When utility is defined as $W_i(o)=\log P_i(o)$ (epistemic utility), the equivalent condition for "sub-agent $i$ benefiting from synthesis" is the welfare gap $\Delta_{P_i}(P)=\mathbb E_P[\log P_i]-\mathbb E_{P_i}[\log P_i]\ge 0$, which simplifies to the information-theoretic form $\Delta_i=H(P_i)-H(P)-\mathrm{KL}(P\|P_i)$. A "strictly unanimous group" $\mathcal U_{\text{strict}}$ requires all $i$ to be strictly $>0$. Around this core definition, the authors derive feasibility boundaries, recursive stability, and finally apply the results to Luigi-Waluigi alignment analysis.

### Key Designs

**1. Log-pooling: An Aggregation Rule Dictated by Network Architecture and Training Objectives**

To apply economic welfare aggregation to neural networks, an aggregation form must match both the network structure (final linear layer + softmax) and the training objective (log-likelihood). Log-pooling is the unique form satisfying both. Its form $P(o)=\frac1Z \prod_j P_j(o)^{\beta_j}$ is equivalent to a weighted sum of logits $\log P_i$ followed by a softmax, which is exactly how modern Transformers operate. Consequently, additive decompositions on logits—such as ensembles, Product of Experts, and multi-head attention—are mathematically equivalent to the log-pooling synthesis of sub-agents. Correspondingly, epistemic utility $W_i(o)=\log P_i(o)$ is not an arbitrary choice: the gradient of cross-entropy training is propagated through the $\log P(\cdot)$ term, making $\log P$ the implicit utility minimized by the network. Under this mapping, the authors define "sub-agent $i$ benefiting from synthesis" as $\mathbb E_P[W_i]\ge \mathbb E_{P_i}[W_i]$.

**2. Possibility Boundaries of Strict Unanimity: Three Theorems Defining Feasibility**

This is the most technical portion of the paper, clarifying the conditions for simultaneous benefit. Three theorems provide boundaries: the Impossibility Theorem for binary outcomes states that when $|\mathcal O|=2$, no non-trivial weights can satisfy $\Delta_i\ge 0$ for two agents with at least one being strict—intuitively, log-pooling in binary space is a zero-sum struggle (Theorem 8). The Existence Theorem for $|\mathcal O|\ge 3$ allows for the explicit construction of a set $\{P_i\}_{i=1}^n, \beta_i$ such that $\mathbb E_P[\log P_i]>\mathbb E_{P_i}[\log P_i]$ holds strictly for all $i$ (Theorem 9). The Impossibility Theorem for Linear Pooling proves that under $P_C^{\text{lin}}(o)=\sum\beta_i P_i(o)$ paired with epistemic utility, strict unanimity is unattainable because linear pooling is equivalent to random dictatorship; an anti-aligned agent selected as dictator would severely harm others (Theorem 10). In practice, this suggests that multi-persona LLMs must use log-pooling, and binary "safe vs unsafe" framing is fundamentally ill-suited for this theory.

**3. Recursive Stability and the Luigi-Waluigi Alignment Principle**

This set of results provides structural guarantees for the "synthesis-decomposition-resynthesis" process and strictly proves that "manifesting then suppressing Waluigi" is superior. Stability rests on three pillars: Lemma 13 states that replacing one $P_i$ with $m$ weight-compatible sub-agents does not change the global pool. Theorem 14 constructs a case where a parent agent satisfies $\Delta_{P_1}(P)>0$ but its split sub-agents satisfy $\Delta_{P_{1,1}}(P)<0$, indicating that parent-level alignment does not guarantee sub-agent alignment. Theorem 17 (Openness) proves that if $P\in\mathcal U_{\text{strict}}$, an entire neighborhood of $P$ resides in $\mathcal U_{\text{strict}}$, meaning strict unanimity is a locally stable open set. This openness justifies why small RLHF parameter updates do not suddenly shatter persona structures. Based on this, Section 5 utilizes Hilbert space analysis with $L=\log P$ and $P$-centered profiles $v_i(o)=l_i(o)-\mathbb E_P[l_i]$: when RLHF imposes a KL budget and reinforces the Luigi persona, the adversarial Waluigi sub-agent is inevitably activated in opposition. However, "manifesting Waluigi first and then suppressing it" yields a strictly larger first-order alignment error reduction compared to purely reinforcing Luigi.

## Key Experimental Results

### Main Results
The paper lacks traditional empirical experiments; its "experiments" consist of theorems and closed-form constructions, summarized below:

| Pooling Form | Utility | $|\mathcal O|$ | Is Strict Unanimous Reachable? | Source |
| :--- | :--- | :--- | :--- | :--- |
| Linear | Epistemic $\log P_i$ | Any | **Impossible** | Theorem 10 |
| Log | Epistemic $\log P_i$ | $2$ | **Impossible** | Theorem 8 |
| Log | Epistemic $\log P_i$ | $\ge 3$ | **Reachable** (Explicitly constructible) | Theorem 9 |
| Log | General welfare $W_i$ | Any $n\ge 2$ | Exists $\{P_i, W_i, \beta_i\}$ where it holds | Theorem 6 |

Core Theorem for Alignment Application:

| Strategy | First-order Alignment Error Reduction | Strictness |
| :--- | :--- | :--- |
| Only Reinforce Luigi (Increase $\beta_{\text{Luigi}}$) | $\Delta_{\text{Luigi}}^{(1)}$ | Baseline |
| Manifest-then-suppress Waluigi | Strictly Greater than $\Delta_{\text{Luigi}}^{(1)}$ | Proven under KL budget constraint |

### Ablation Study

| Configuration | Key Conclusion | Description |
| :--- | :--- | :--- |
| Welfare = $\log P_i$ vs General $W_i$ | Easier to reach unanimity under general $W_i$ (Theorem 6) | Epistemic utility is "narrower" and more constrained |
| Split Compatibility (L13) vs Non-comp. (T14) | Former preserves pool; latter shows benefit doesn't transfer | Top-level alignment does not ensure sub-level alignment |
| Strict vs Non-strict Unanimity | Non-strict allows duplicates (L48); Strict is stable in neighborhood (T17) | Distinguishes meaningful synthesis from trivial copying |

### Key Findings
- An outcome space dimension of $|\mathcal O| \ge 3$ is a hard threshold for the possibility of strict unanimity. Any setting reducing alignment to a binary "safe vs unsafe" choice falls into the impossibility zone of Theorem 8.
- "High top-level performance but damaged sub-personas" is a mathematical theorem (Theorem 14), not just an intuition—RLHF success cannot be judged solely by reward model scores.
- "Manifesting then suppressing Waluigi" is strictly superior to "direct Luigi reinforcement," providing the first formal proof for this empirical heuristic.

## Highlights & Insights
- The "trinity" linking log-pooling, neural network logit structures, and cross-entropy training is an elegant observation, allowing alignment theory to leverage economic aggregation tools.
- Theorem 14 explains the recurring empirical phenomenon of "high reward model scores alongside deceptive behavior": rewards monitor the top level, while sub-personas may still be strictly harmed.
- The "Openness" result provides geometric intuition for the KL budget in RLHF: as long as the base model is in the strict unanimity zone, small updates keep it there, explaining the stability of KL-regularized training.
- "Manifest-then-suppress" is a concrete operational principle—eliciting jailbreaks or deceptive behaviors to construct negative samples for contrastive training is strictly more effective than only amplifying positive samples.

## Limitations & Future Work
- The paper assumes a finite outcome space with $P_i > 0$, requiring extensions for actual LLM vocabularies and long-sequence conditionals.
- While $W_i = \log P_i$ is naturally derived, real-world RLHF involves more complex implicit utilities like preference learning; generalizing to non-epistemic utilities is necessary.
- The Luigi-Waluigi conclusion is first-order (local expansion under small KL budget); large-scale alignment fine-tuning requires higher-order analysis.
- There is a lack of empirical experiments to quantify the actual performance gap between "manifest-then-suppress" and pure reinforcement.

## Related Work & Insights
- **vs. Classical Opinion Pooling**: This work ports decades of economic research on logarithmic vs. linear pooling to neural network analysis.
- **vs. Active Inference**: Similar to the "agent = generative model" view, but focuses specifically on the mathematical structure of sub-agent synthesis.
- **vs. Empirical Waluigi Discussions**: Provides the first rigorous first-order proof under KL budgets for previous informal observations.
- **vs. PoE / MoE**: Product of Experts is essentially log-pooling; this work proves the stability conditions for such combinations. MoE gating can be viewed as dynamic $\beta_i$ selection.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces the log-pooling framework to alignment theory with formal proofs for empirical heuristics.
- Experimental Thoroughness: ⭐⭐ Theoretical paper with no numerical experiments or LLM validation.
- Writing Quality: ⭐⭐⭐⭐ Clearly explains the transition from network architecture to log-pooling and organizes theorems logically.
- Value: ⭐⭐⭐⭐ Provides a rigorous framework for vague concepts like "persona synthesis" and the "Waluigi effect."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] BayesAgent: Bayesian Agentic Reasoning Under Uncertainty via Verbalized Probabilistic Graphical Modeling](../../AAAI2026/llm_agent/bayesagent_bayesian_agentic_reasoning_under_uncertainty_via_.md)
- [\[ICML 2026\] Hunt Instead of Wait: Evaluating Deep Data Research on Large Language Models](hunt_instead_of_wait_evaluating_deep_data_research_on_large_language_models.md)
- [\[ICLR 2026\] MemGen: Weaving Generative Latent Memory for Self-Evolving Agents](../../ICLR2026/llm_agent/memgen_weaving_generative_latent_memory_for_self-evolving_agents.md)
- [\[ICML 2026\] AgentXRay: White-Boxing Agentic Systems via Workflow Reconstruction](agentxray_white-boxing_agentic_systems_via_workflow_reconstruction.md)
- [\[ICML 2026\] HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation](hawkesllm_semantic_uncertainty_propagation_in_agentic_text_simulation.md)

</div>

<!-- RELATED:END -->
