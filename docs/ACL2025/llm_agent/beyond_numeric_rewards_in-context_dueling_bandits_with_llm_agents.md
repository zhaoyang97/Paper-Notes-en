---
title: >-
  [Paper Note] Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents
description: >-
  [ACL 2025][LLM Agent][In-Context Reinforcement Learning] This work systematically evaluates the zero-shot in-context decision-making capabilities of LLMs in Dueling Bandits (preference-feedback reinforcement learning). It reveals that GPT-4 Turbo excels in weak regret but displays a gap in sstrong regret. Consequently, the LEAD (LLM with Enhanced Algorithmic Dueling) framework is proposed, which achieves both theoretical guarantees and robustness by adaptively and fine-graine…
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "In-Context Reinforcement Learning"
  - "Dueling Bandits"
  - "LLM Decision Making"
  - "LEAD Framework"
  - "Regret Bounds"
date: 2026-05-08
content_hash: f17b381d7d3ace24
---

# Beyond Numeric Rewards: In-Context Dueling Bandits with LLM Agents

**Conference**: ACL 2025  
**arXiv**: [2407.01887](https://arxiv.org/abs/2407.01887)  
**Code**: To be confirmed  
**Area**: LLM Agent  
**Keywords**: In-Context Reinforcement Learning, Dueling Bandits, LLM Decision Making, LEAD Framework, Regret Bounds

## TL;DR

This work systematically evaluates the zero-shot in-context decision-making capabilities of LLMs in Dueling Bandits (preference-feedback reinforcement learning). It reveals that GPT-4 Turbo excels in weak regret but displays a gap in sstrong regret. Consequently, the LEAD (LLM with Enhanced Algorithmic Dueling) framework is proposed, which achieves both theoretical guarantees and robustness by adaptively and fine-grainedly integrating classical DB algorithms with LLM agents.

## Background & Motivation

**The Rise of In-Context Reinforcement Learning (ICRL)**: Transformers trained on pre-existing interaction datasets can infer RL tasks in-context and make effective decisions without parameter updates. However, the zero-shot ICRL capabilities of LLMs remain largely unexplored.

**Difficulties of LLMs in Handling Numeric Rewards**: Prior studies indicate that LLMs perform poorly in traditional Multi-Armed Bandit (MAB) settings, are vulnerable to adversarial loss functions, and exhibit insufficient exploration capabilities (e.g., failing to correctly compare $13.11 > 13.8$).

**Natural Advantages of Preference Feedback**: Dueling Bandits employ binary preference feedback (A wins or B wins) instead of numeric rewards, which aligns better with comparative reasoning in natural language and may be more suited to the capability profile of LLMs.

**Theoretical Connection with RLHF**: The DB problem shares conceptual similarities with preference learning in RLHF. Understanding LLM performance in DB can deepen the comprehension of LLM preference learning dynamics.

**Robustness Requirements**: In real-world applications, prompts might be noisy or adversarial. There is a need for a framework that provides theoretical guarantees even under suboptimal prompts.

**Necessity of Algorithmic Enhancement**: Relying solely on LLMs for decision-making lacks convergence guarantees, necessitating a non-trivial integration with classical algorithms.

## Method

### Overall Architecture

The paper is divided into two parts: (1) a systematic evaluation of the zero-shot capabilities of LLMs in DB problems (Section 3); and (2) the proposed LEAD algorithmic enhancement framework (Section 4), which fuses Explore-then-Exploit DB algorithms (such as IF2) with LLM agents.

### Key Designs 1: LLM Zero-shot DB Evaluation System

A systematic evaluation protocol is designed, including:
- **Prompt Design**: Comprises query description $P$ ($K$ arms, horizon $T$, task objectives), externally aggregated interaction history $H_t$ (including duel outcomes and empirical probabilities), and zero-shot CoT reasoning $R$.
- **Performance Metrics**: Strong Regret (SR, the sum of preference gaps between the two chosen arms and the optimal arm) and Weak Regret (WR, the preference gap between the better of the chosen arms and the optimal arm).
- **Test Environments**: Transitive-Easy, Transitive-Hard, Intransitive-Easy, Intransitive-Hard, with $K=5$ and $K=10$ settings.

### Key Designs 2: Analysis of LLM Capabilities (Success and Failure Modes)

Detailed behavioral analysis reveals three major issues of LLMs:
- **Exploration Vulnerability**: GPT-4 Turbo tends to rapidly narrow focus down to a small subset of arms and compare them repeatedly, easily getting trapped in local optima due to initial biases.
- **Exploitation Inability**: Even with explicit convergence instructions, LLMs fail to stably select the same optimal arm for "self-duels" because of a pre-training bias indicating "an arm cannot duel against itself."
- **Pre-training Bias**: GPT-4 Turbo and o1-preview exhibit systematic misunderstandings of the DB problem, which in-context instructions cannot fully override.

### Key Designs 3: The LEAD Framework

LEAD adopts a two-phase adaptive switching strategy:
- **Phase 1 (LLM Phase)**: Selects two arms recommended by the LLM, finds the winner $b_{LLM}$ among them, and then sequentially compares it with other arms in the candidate set $B$. This is controlled by a `TrustLLM` flag—if $b_{LLM}$ is defeated, the flag is set to `False`.
- **Phase 2 (DB Phase)**: When the LLM recommendation is untrusted, the system falls back to a round of the classical IF2 algorithm, utilizing the incumbent arm $b_{IF2}$ selected from the estimated preference matrix.
- **Adaptive Switching**: Returns to Phase 1 after Phase 2 ends, repeating until the candidate set contains only the optimal arm.

### Key Designs 4: Theoretical Guarantees

- **Theorem 4.1 (Vulnerability)**: Proves that any standalone LLM agent will suffer a regret of $\Omega(\min\{\Phi(T), T/K\})$ under an attacker budget of $\Phi(T)$.
- **Theorem 4.2 (LEAD Regret Bounds)**: The strong regret of LEAD is $\le \tilde{O}((K \log T)/\varepsilon_{1,2})$, and the weak regret is $\le \min\{\tilde{O}(T_{LLM} + K \log K / \varepsilon_{1,2}), \tilde{O}(K \log T / \varepsilon_{1,2})\}$.
- **Theorem 4.3 (Converse Theorem)**: Proves that the extra term $(K \log K)/\varepsilon_{1,2}$ in LEAD is nearly tight.

### Loss & Training

The optimization objective is to minimize cumulative strong regret and weak regret. LEAD inherits the regret bounds of IF2 while reducing average regret by leveraging the short-term exploration advantages of LLMs.

## Key Experimental Results

### Main Results: Performance on Transitive-Easy Instances

| Method | Strong Regret ($T=2000$) | Weak Regret ($T=2000$) | Convergence |
|------|-----------------|-----------------|--------|
| GPT-3.5 Turbo | Very High | High | ✗ |
| GPT-4 | Very High | High | ✗ |
| GPT-4 Turbo | Medium-High | **Lowest** | ✗ |
| o1-preview | Medium | Medium-High | Partial |
| Self-Sparring | Medium | Medium | ✓ |
| DTS | Medium | Medium | ✓ |
| IF2 | Medium-Low | Medium-Low | ✓ |
| **LEAD ($\delta=0.4$)** | **Lowest** | **Lowest** | **✓** |

### Robustness Experiments: Strong Regret under Different Prompt Conditions

| Prompt Condition | GPT-4 Turbo | LEAD ($\delta=1/TK^2$) | IF2 |
|----------|-------------|-----------------|-----|
| Original Prompt | Continuously Growing | Convergent (<2000 steps) | Convergent |
| Biased History (Noisy Prompt) | Trapped in Local Optima | Convergent | Convergent |
| Adversarial Prompt (Target Reversal) | Seriously Divergent | Near-Optimal Convergence | Convergent |

### Key Findings

1. GPT-4 Turbo displays an emergent zero-shot capability in weak regret, significantly outperforming classical DB algorithms by rapidly identifying the best arm and incorporating it into duels.
2. However, LLMs suffer from a systematic gap in strong regret, failing to converge to the optimal strategy (persistently selecting the optimal arm to duel against itself).
3. LEAD at $\delta=0.4$ concurrently achieves the lowest strong and weak regret, and maintains robustness across all prompt conditions (original, noisy, and adversarial).
4. GPT-4 Turbo exhibits the lowest performance variance among all evaluated LLMs, indicating a relatively stable decision-making process.
5. Scalability is limited: when scaling from $K=5$ to $K=10$, the long-term performance of the LLMs deteriorates significantly; in intransitive preference structures, the long-term weak regret advantage of LLMs vanishes.

## Highlights & Insights

1. **Pioneering Research Problem**: This is the first study to systematically investigate the in-context decision-making capability of LLMs in preference-feedback (non-numeric reward) RL, revealing the emergent capabilities of LLMs in relative comparative reasoning.
2. **Best-of-Both-Worlds**: LEAD elegantly combines the short-term exploration advantage of LLMs with the convergence guarantees of classical algorithms, reconciling theory with practice.
3. **Comprehensive Theoretical Framework of Three Theorems**: Vulnerability lower bound $\to$ regret upper bound $\to$ converse theorem for tightness, presenting a complete and elegant theoretical system.
4. **In-depth Behavioral Analysis**: The visualization of LLM decision trajectories and analysis of failure modes (exploration vulnerability, exploitation inability, and pre-training bias) provide valuable insights.
5. **Robustness Verification**: Systematic evaluations under noisy and adversarial prompts demonstrate the practical reliability of the LEAD framework.

## Limitations & Future Work

1. **Focus on Multi-Armed (Non-Contextual) DB Only**: Does not cover more complex scenarios such as Contextual Dueling Bandits, Multi-Dueling, or adversarial DB settings.
2. **Single Definition of Winner**: Only considers the Condorcet winner definition, leaving LLM performance under alternative definitions like Borda winner or von Neumann winner unexplored.
3. **Exploration Limited to Explore-then-Exploit Algorithms**: Integration with other online regret minimization algorithms has not been considered.
4. **Limited Selection of LLMs**: Primarily relies on the GPT series of models, leaving the performance of open-source models (such as LLaMA, Mistral) unknown.
5. **Mapping to Practical Scenarios**: The paper focuses on theoretical analysis, lacking case studies that apply the DB framework to practical applications like recommendation systems and information retrieval.

## Related Work & Insights

- **Krishnamurthy et al.**: Demonstrated exploration failures of LLMs in MAB problems, whereas this paper reveals a different performance pattern in DB—excellent weak regret but poor strong regret.
- **Nie et al. (EvoLVe)**: Utilized inference-time algorithmic guidance to optimize LLM performance in MAB, but adopted naive algorithmic guidance. LEAD proposes a fine-grained adaptive fusion, representing a fundamental improvement.
- **RLHF Connections**: DB shares conceptual similarities with preference learning in RLHF, yet irreducibility results exist, which warrants further exploration.
- **Insight**: This work implies that the "LLM + Classical Algorithm" hybrid paradigm may serve as a general design pattern for LLM decision systems—leveraging the linguistic priors of LLMs for fast exploration, while utilizing classical algorithms to guarantee convergence and robustness.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to study in-context decision making of LLMs in preference RL + complete theoretical framework
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive evaluation across multiple models, environments, and prompt conditions
- Writing Quality: ⭐⭐⭐⭐⭐ — Close integration of theory and experiments, with deep and thorough behavioral analysis
- Value: ⭐⭐⭐⭐ — Significant contribution to the understanding of LLM decision-making capabilities, with the generalizable concept of the LEAD framework

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CORE: Full-Path Evaluation of LLM Agents Beyond Final State](../../NeurIPS2025/llm_agent/core_full-path_evaluation_of_llm_agents_beyond_final_state.md)
- [\[ACL 2025\] Self-Taught Agentic Long-Context Understanding](self_taught_agentic_long_ctx.md)
- [\[NeurIPS 2025\] ContextAgent: Context-Aware Proactive LLM Agents with Open-World Sensory Perceptions](../../NeurIPS2025/llm_agent/contextagent_context-aware_proactive_llm_agents_with_open-world_sensory_percepti.md)
- [\[ACL 2025\] LLM Agents Making Agent Tools](llm_agents_making_agent_tools.md)
- [\[ICML 2026\] ACON: Optimizing Context Compression for Long-horizon LLM Agents](../../ICML2026/llm_agent/acon_optimizing_context_compression_for_long-horizon_llm_agents.md)

</div>

<!-- RELATED:END -->
