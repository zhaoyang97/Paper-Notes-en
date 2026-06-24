---
title: >-
  [Paper Note] Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis
description: >-
  [AAAI 2026 Oral][LLM Alignment][AI alignment] This paper formalizes AI alignment as an $\langle M,N,\varepsilon,\delta\rangle$-agreement multi-objective optimization problem, proves information-theoretic lower bounds on alignment (encoding "all human values" is fundamentally intractable) from a communication complexity perspective, and derives explicit achievable algorithms and tight upper bounds for unbounded/bounded rational agents, revealing the theoretical basis for the g…
tags:
  - "AAAI 2026 Oral"
  - "LLM Alignment"
  - "AI alignment"
  - "communication complexity"
  - "agreement framework"
  - "No-Free-Lunch"
  - "reward hacking"
date: 2026-05-08
content_hash: c169bb4de1772f28
---

# Intrinsic Barriers and Practical Pathways for Human-AI Alignment: An Agreement-Based Complexity Analysis

**Conference**: AAAI 2026 Oral  
**arXiv**: [2502.05934](https://arxiv.org/abs/2502.05934)  
**Code**: None  
**Area**: Other
**Keywords**: AI alignment, communication complexity, agreement framework, No-Free-Lunch, reward hacking

## TL;DR

This paper formalizes AI alignment as an $\langle M,N,\varepsilon,\delta\rangle$-agreement multi-objective optimization problem, proves information-theoretic lower bounds on alignment (encoding "all human values" is fundamentally intractable) from a communication complexity perspective, and derives explicit achievable algorithms and tight upper bounds for unbounded/bounded rational agents, revealing the theoretical basis for the global inevitability of reward hacking in large state spaces.

## Background & Motivation

**Background**: Current AI alignment research is practically oriented — methods such as RLHF, DPO, and Constitutional AI have achieved notable success in preventing LLM jailbreaks and related issues. On the theoretical side, AI Safety via Debate uses interactive proofs to isolate misalignment through zero-sum debate games; CIRL formalizes alignment as a cooperative partial-information game and reduces it to a POMDP.

**Limitations of Prior Work**: These approaches rely on strong assumptions. Debate requires precise, unbiased human verifiers; CIRL implicitly assumes a Common Prior Assumption (CPA) and Markov property, limiting the ability to exploit richer historical context. More fundamentally, all existing approaches lack general theoretical guarantees about the **intrinsic complexity** of alignment.

**Key Challenge**: Existing alignment frameworks focus on the feasibility of specific methodologies, yet no unified framework addresses: what **method-agnostic, inherent barriers** does alignment itself possess? When the number of objectives $M$ or agents $N$ is sufficiently large, do information-theoretic barriers exist that no protocol can overcome?

**Goal**: (a) Formalize an alignment framework with minimal assumptions; (b) prove information-theoretic lower bounds — a "No-Free-Lunch" theorem for alignment; (c) provide explicit achievable algorithms as upper-bound certificates; (d) analyze the exponential cost under bounded rationality.

**Key Insight**: The authors observe that prior methods implicitly rely on foundational concepts such as iterated reasoning, mutual updating, and common knowledge, and build their analytical framework upon Aumann's agreement theorem and Aaronson's $\langle\varepsilon,\delta\rangle$-agreement extension.

**Core Idea**: Alignment is modeled as a communication game in which $N$ agents without a common prior must reach approximate consensus on $M$ objectives. The paper proves that the intrinsic complexity necessarily grows with the number of objectives, agents, and state-space size, and that bounded rationality leads to exponential deterioration.

## Method

### Overall Architecture

This paper presents a **theoretical analysis framework** rather than a system-level method. The overall structure is:

**Input**: $M$ tasks $\{f_j\}$, $N$ agents (humans and AI), each agent holding its own prior distribution $\mathbb{P}_j^i$ over each task (no common prior required).

**Goal**: All agents reach $\varepsilon$-approximate consensus on the expected value of the objective function for all tasks, with probability $\geq 1-\delta$.

**Output**: Tight upper and lower bounds on communication complexity, along with explicit algorithms.

The framework proceeds in three progressive layers:
(1) Information-theoretic lower bounds (§4) → (2) Upper bounds and algorithms for unbounded rationality (§5.1) → (3) Exponential deterioration under bounded rationality (§5.2).

### Key Designs

1. **Formalization of $\langle M,N,\varepsilon,\delta\rangle$-Agreement**

    - **Function**: Defines the mathematical framework for the alignment problem.
    - **Mechanism**: There are $M$ tasks; task $j$ has state space $S_j$ of size $D_j$ and objective function $f_j: S_j \to [0,1]$. The $N$ agents each hold a prior $\mathbb{P}_j^i$ and iteratively refine knowledge partitions $\Pi_j^{i,t}$ by broadcasting messages. Agreement is reached when the conditional expectation difference between all agent pairs is $\leq \varepsilon_j$ with probability $> 1-\delta_j$. The prior distance $\nu_j$ measures the minimum $L^1$ distance between two priors.
    - **Design Motivation**: The framework makes **no assumption of a common prior** (CPA) — this is the most critical relaxation, as humans and AI cannot realistically share priors over all tasks. The framework also supports multiple objectives, multiple agents, noisy communication, non-Markovian history, computational bounds, and asymmetric costs, making it the only framework satisfying all these conditions simultaneously (Table 1).

2. **Information-Theoretic Lower Bounds (Propositions 1–3)**

    - **Function**: Proves the incompressibility of alignment communication.
    - **Mechanism**:
        - **General lower bound** (Prop. 1): $\Omega(MN^2 \log(1/\varepsilon))$ bits — derived by constructing instances where each agent pair holds independent uniform inputs and applying a counting argument to show the inevitability of information transfer.
        - **Smooth protocol lower bound** (Prop. 2): $\Omega(MN^2(\nu + \log(1/\varepsilon)))$ bits — requires that posteriors cannot deviate too far from priors, introducing the prior distance term $\nu$.
        - **BBF protocol lower bound** (Prop. 3): $\Omega(MN^2[D\nu + \log(1/\varepsilon)])$ bits — restricts message updates to bounded Bayesian factor updates, introducing the state-space size multiplier $D$, yielding a tighter match with the upper bound.
    - **Design Motivation**: These lower bounds establish the theorem that "encoding all human values is fundamentally intractable" — when $M$ or $N$ is large, alignment overhead cannot be avoided regardless of computational power. This is the rigorous formulation of the **No-Free-Lunch principle**.

3. **Explicit Achievable Algorithm (Algorithm 1 + Theorem 1)**

    - **Function**: Proves that $\langle M,N,\varepsilon,\delta\rangle$-agreement is indeed achievable.
    - **Mechanism**: The algorithm operates in two phases: (a) constructing a common prior $\mathbb{CP}_j$ via message passing and partition refinement (Lemma 2: $O(N^2 D_j)$ rounds); (b) running Aaronson's $\langle\varepsilon,\delta\rangle$-agreement protocol on the common prior (Lemma 3: $O(N^7/(\delta\varepsilon)^2)$ rounds). The total upper bound is $T = O(MN^2D + M^3N^7/(\varepsilon^2\delta^2))$.
    - **Design Motivation**: The upper bound tightly matches the lower bound up to polynomial factors within natural protocol classes, demonstrating the precision of the analysis. A discretization extension (Prop. 4) shows that restricting to discrete messages (e.g., LLM tokens) does not increase complexity.

4. **Bounded Rationality Analysis (Theorem 2 + Corollary 1)**

    - **Function**: Quantifies the alignment cost imposed by computational bounds and noise.
    - **Mechanism**: Human (evaluation cost $T_H$) and AI (evaluation cost $T_{AI}$) agents are distinguished. Bounded agents must approximate expectations via sampling; the "Total Bayesian Wannabe" definition ensures that bounded agents are statistically indistinguishable from ideal Bayesian agents. The result is an exponential blowup in time complexity: even with $N=2$, $M=1$, and the most relaxed parameters ($\varepsilon=\delta=1/2$), the required number of subroutine calls is approximately $O(10^{10^{13.28}})$, far exceeding the number of atoms in the observable universe.
    - **Design Motivation**: This directly explains why reward hacking is inevitable in practice — finite sampling systematically underestimates rare high-loss states. Scalable oversight must target safety-critical slices rather than pursuing uniform coverage.

### Theoretical Analysis

The central contribution of this paper is rigorous theoretical analysis, yielding four major findings:

1. **No-Free-Lunch**: Too many objectives → alignment cost is necessarily high → values must be compressed/prioritized.
2. **Reward hacking is globally inevitable**: Large state space + finite sampling → rare events are necessarily missed.
3. **Bounded rationality incurs enormous cost**: Transitioning from unbounded to bounded rationality → exponential deterioration.
4. **Tight lower bounds provide governance thresholds**: Matching upper and lower bounds → risk thresholds can be established accordingly.

## Key Experimental Results

This is a purely theoretical work with no traditional experiments, but it includes quantitative complexity analysis and framework comparisons.

### Framework Capability Comparison (Table 1)

| Framework | No CPA | Approx. | Multi-Obj. | Multi-Agent | Non-Markov | Bounded Rat. | Asymmetric | Noise | Upper Bd. | Lower Bd. |
|-----------|--------|---------|------------|-------------|------------|--------------|------------|-------|-----------|-----------|
| Aumann (1976) | ✗ | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ | ✗ | ✗ | ✗ |
| Aaronson (2005) | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| CIRL (2016) | ✗ | ✓ | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ |
| Debate (2018) | ✓ | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✓ | ✗ |
| **Ours** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

### Complexity Upper and Lower Bound Summary

| Setting | Lower Bound | Upper Bound | Tightness |
|---------|-------------|-------------|-----------|
| General protocol (unbounded) | $\Omega(MN^2 \log(1/\varepsilon))$ bits | $O(MN^2D + M^3N^7/(\varepsilon\delta)^2)$ msgs | Polynomial gap |
| Smooth protocol | $\Omega(MN^2(\nu + \log(1/\varepsilon)))$ bits | Same as above | Prior distance included |
| BBF protocol | $\Omega(MN^2[D\nu + \log(1/\varepsilon)])$ bits | Same as above | Polynomial gap |
| Discretized (unbounded) | Same as above | $O(MN^2D + M^3N^7/(\varepsilon\delta)^2)$ msgs | Same as continuous |
| Bounded rationality | $\Omega(MT_{N,q} e^D)$ | $O(MT_{N,q} B^{\text{poly}(M,N,D,1/\varepsilon,1/\delta)})$ | Both exponential |

### Key Findings

- **Most striking numerical result**: With only 2 bounded agents, 1 task, and the most permissive parameters ($\varepsilon=\delta=\rho=1/2$), the required computation is approximately $10^{10^{13.28}}$, far exceeding the estimated $\sim 4.8 \times 10^{79}$ atoms in the observable universe. This demonstrates that alignment under bounded rationality is extremely hard in the worst case.
- **BBF protocol lower bound introduces the $D$ factor**: This shows that larger state spaces make alignment harder in a structurally fundamental way.
- **Discretization does not increase complexity**: Prop. 4 proves that discrete messages (e.g., LLM tokens) achieve the same upper bound as continuous-valued messages, which is encouraging for practical systems.

## Highlights & Insights

- **"No-Free-Lunch for Alignment" principle**: This is the first rigorous impossibility theorem in alignment research — not that a particular method fails, but that **any method** must fail when the number of objectives is too large. This elevates alignment from an engineering challenge to a mathematical inevitability.
- **Reward hacking elevated from empirical observation to theorem**: The needle-in-a-haystack construction in Proposition 5 rigorously proves that sampling-based protocols must miss rare dangerous states in large state spaces. This provides theoretical guidance for scalable oversight: focus on safety-critical slices rather than uniform coverage.
- **Unifying power of the framework**: The $\langle M,N,\varepsilon,\delta\rangle$-agreement formulation is the first framework simultaneously satisfying all conditions — no CPA, approximation, multiple objectives, multiple agents, noise robustness, and bounded rationality — and can serve as a unified foundation for future theoretical research.
- **Transferable perspectives**: The communication complexity viewpoint is transferable to privacy-efficiency tradeoff analysis in federated learning, communication overhead modeling in multi-agent negotiation, and the design of information-theoretically optimal human feedback sampling strategies in RLHF.

## Limitations & Future Work

- **Lack of empirical validation**: This is a purely theoretical work; all results are worst-case analyses. Whether structural favorable cases in practice (e.g., correlations among alignment objectives) could significantly reduce complexity is not addressed.
- **Gap between upper and lower bounds**: In the general setting, a polynomial gap remains (upper bound has $M^3N^7$ terms, lower bound only $MN^2$); tightness is achieved only within the BBF protocol class.
- **Bounded rationality analysis may be overly pessimistic**: The exponential upper bound in Corollary 1 may be too loose — practical AI systems may exploit task structure (e.g., symmetry, low-rank structure) to substantially reduce the cost.
- **Assumption limitations**: The assumption that knowledge partitions are common knowledge may not hold in distributed AI systems; all agents are assumed to participate honestly, without consideration of adversarial agents.
- **Vague recommendation to "compress the objective set"**: The paper recommends compressing the value set but provides no theoretical guidance on how to do so (a follow-up work, nayebi2025coresafety, partially addresses this).

## Related Work & Insights

- **vs. Aumann Agreement (1976)**: Aumann requires a common prior and exact consensus; this paper relaxes both conditions to no CPA, approximate agreement, and multiple objectives, representing a well-motivated modernization.
- **vs. Aaronson $\langle\varepsilon,\delta\rangle$-agreement (2005)**: Aaronson addresses the single-objective, CPA setting; the present paper's no-CPA multi-objective extension introduces a new complexity term $O(N^2D)$ for the common-prior construction phase (Lemma 2).
- **vs. CIRL (Hadfield-Menell 2016)**: CIRL is a single-agent inverse reinforcement learning approach to hidden reward recovery, assuming a common prior and Markov property. The present framework is more general but also more abstract — CIRL provides concrete algorithms, while this paper provides complexity lower bounds.
- **vs. AI Safety via Debate (Irving 2018)**: Debate relies on a correct, unbiased human judge; the present paper does not. However, Debate provides a concrete game-theoretic protocol, while this paper's contribution is more foundational in nature.
- **Relationship to the RLHF pipeline** (Figure 1): The proposed framework can be mapped onto existing RLHF/DPO/Constitutional AI pipelines, providing a theoretical perspective on these practical methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First method-agnostic information-theoretic impossibility theorem for AI alignment; outstanding theoretical contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ No experiments as a purely theoretical work, but quantitative analysis and numerical examples are reasonably compelling.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and rigorous mathematics, though the high technical density poses readability challenges for non-theorist readers.
- **Value**: ⭐⭐⭐⭐⭐ Establishes an information-theoretic foundation for alignment research; the No-Free-Lunch principle and reward hacking inevitability theorem will have lasting impact on the field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Operationalising the Superficial Alignment Hypothesis via Task Complexity](../../ICML2026/llm_alignment/operationalising_the_superficial_alignment_hypothesis_via_task_complexity.md)
- [\[AAAI 2026\] EASE: Practical and Efficient Safety Alignment for Small Language Models](ease_practical_and_efficient_safety_alignment_for_small_language_models.md)
- [\[ICLR 2026\] Skywork-Reward-V2: Scaling Preference Data Curation via Human-AI Synergy](../../ICLR2026/llm_alignment/skywork-reward-v2_scaling_preference_data_curation_via_human-ai_synergy.md)
- [\[ICLR 2026\] Enforcing Axioms for AI Alignment under Loss-Based Rules](../../ICLR2026/llm_alignment/enforcing_axioms_for_ai_alignment_under_loss-based_rules.md)
- [\[ICLR 2026\] Aligner, Diagnose Thyself: A Meta-Learning Paradigm for Fusing Intrinsic Feedback in Preference Alignment](../../ICLR2026/llm_alignment/aligner_diagnose_thyself_a_meta-learning_paradigm_for_fusing_intrinsic_feedback_.md)

</div>

<!-- RELATED:END -->
