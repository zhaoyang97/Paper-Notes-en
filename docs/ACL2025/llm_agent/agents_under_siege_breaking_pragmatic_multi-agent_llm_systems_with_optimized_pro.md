---
title: >-
  [Paper Note] Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks
description: >-
  [ACL 2025][LLM Agent][Multi-agent security] This paper is the first to systematically study adversarial attacks in realistic multi-agent LLM systems featuring bandwidth constraints, latency, and security mechanisms. It proposes attack methods based on Minimum-Cost Maximum-Flow (MCMF) topological optimization and Permutation-Invariant Evasion Loss (PIEL), achieving up to a 7-fold increase in success rate compared to traditional attacks across multiple LLM architectures.
tags:
  - "ACL 2025"
  - "LLM Agent"
  - "Multi-agent security"
  - "jailbreak attacks"
  - "permutation invariance"
  - "minimum-cost maximum-flow"
  - "adversarial prompt propagation"
date: 2026-05-08
content_hash: a365cc86d3d02fbe
---

# Agents Under Siege: Breaking Pragmatic Multi-Agent LLM Systems with Optimized Prompt Attacks

**Conference**: ACL 2025  
**arXiv**: [2504.00218](https://arxiv.org/abs/2504.00218)  
**Code**: None  
**Area**: LLM Agent / LLM Security  
**Keywords**: Multi-agent security, jailbreak attacks, permutation invariance, minimum-cost maximum-flow, adversarial prompt propagation

## TL;DR
This paper is the first to systematically study adversarial attacks in realistic multi-agent LLM systems featuring bandwidth constraints, latency, and security mechanisms. It proposes attack methods based on Minimum-Cost Maximum-Flow (MCMF) topological optimization and Permutation-Invariant Evasion Loss (PIEL), achieving up to a 7-fold increase in success rate compared to traditional attacks across multiple LLM architectures.

## Background & Motivation

**Background**: Multi-agent LLM systems enhance task performance through distributed reasoning and collective intelligence, and are increasingly applied in automation systems and AI governance. Research on LLM security primarily focuses on jailbreak attacks and defenses in single-agent settings.

**Limitations of Prior Work**: Multi-agent systems introduce a brand-new dimension of security risks—communication between agents can be exploited as an attack vector. Existing multi-agent attack studies (such as Evil Geniuses and Prompt Infection) assume that attackers can send unrestricted messages to agents, ignoring actual system communication constraints: token bandwidth limits, message latency, and security filtering mechanisms.

**Key Challenge**: In constrained multi-agent systems, the attacker faces three major constraints: (1) each communication edge has a token bandwidth limit, preventing a complete adversarial prompt from being transmitted over a single edge; (2) the arrival order of messages from different paths is uncertain; (3) security detection mechanisms are deployed on some communication edges. How to successfully execute attacks under these constraints remains an unstudied open problem.

**Goal**: To design optimal adversarial prompt propagation strategies in realistic multi-agent systems with token bandwidth limits, asynchronous message arrivals, and security mechanisms.

**Key Insight**: To model attack path optimization as a Minimum-Cost Maximum-Flow (MCMF) problem from graph theory, while designing a permutation-invariant adversarial loss to ensure the attack remains effective under any chunk arrival sequence.

**Core Idea**: To split the adversarial prompt into multiple chunks, find the optimal propagation paths using an MCMF algorithm (maximizing token flow while minimizing detection risk), and optimize the token content of each chunk using a permutation-invariant evasion loss to ensure they trigger the jailbreak under any permutation.

## Method

### Overall Architecture
The method consists of two decoupled modules: (1) Topological Optimization—given the communication topology, bandwidth constraints, and security mechanism deployment of the multi-agent system, it solves for the optimal attack paths and chunk allocation; (2) Permutation-Invariant Evasion Loss—under a fixed chunking scheme, it optimizes the token content of each chunk so that the attack is successful under all possible arrival orders.

### Key Designs

1. **Minimum-Cost Maximum-Flow Topological Optimization**:

    - **Function**: Finds the optimal adversarial prompt propagation paths in a constrained communication network.
    - **Mechanism**: Defines a flow function $f: \mathcal{E} \to \mathbb{R}_{\geq 0}$ representing the number of adversarial tokens transmitted over each edge. The optimization goal is to minimize the total risk $\min \sum_{(u,v)} G(u,v) \cdot f(u,v)$, subject to token capacity constraints $0 \leq f(u,v) \leq F(u,v)$, flow conservation, and source-sink constraints. A standard MCMF algorithm implemented via NetworkX is used to solve this, outputting the chunk length that should be transmitted on each edge.
    - **Design Motivation**: Intuitively, the attacker needs to "distribute" a complete adversarial prompt across multiple paths to deliver it to the target agent, while avoiding "dangerous" paths monitored by security detection. This is fundamentally a network flow problem.

2. **Permutation-Invariant Evasion Loss (PIEL)**:

    - **Function**: Ensures that chunked adversarial prompts can trigger the jailbreak regardless of their arrival order.
    - **Mechanism**: Divides the adversarial prompt into $K$ chunks $\mathcal{C} = \{C_1, ..., C_K\}$. The loss function is defined as the average negative log-likelihood across all possible permutations: $\mathcal{L}(\mathcal{C}) = \frac{1}{K!} \sum_{\pi \in S_K} -\log p(x^*_{n+1:n+L} | \phi)$, where $\phi$ is the input compiled by concatenating chunks according to permutation $\pi$, and $x^*$ is the target harmful output. The Greedy Coordinate Gradient (GCG) method is adopted to iteratively optimize the tokens in each chunk.
    - **Design Motivation**: Due to network latency, the arrival order of chunks transmitted along different paths to the target agent is non-deterministic. If the adversarial prompt is only effective under a specific order, the attack will frequently fail due to arrival sequence randomness.

3. **Stochastic PIEL (S-PIEL)**:

    - **Function**: Reduces the computational complexity of PIEL.
    - **Mechanism**: When the number of chunks $K$ is large, iterating over all $K!$ permutations is computationally infeasible. S-PIEL randomly samples $M$ permutations to approximate the full expectation: $\tilde{\mathcal{L}}(\mathcal{C}) = \frac{1}{|\tilde{S}_K|} \sum_{\pi \in \tilde{S}_K} -\log p(x^*|  \phi)$.
    - **Design Motivation**: For $K=5$, there are 120 permutations, making full traversal computationally expensive. Experiments show that sampling approximately 50% of the permutations yields high effectiveness.

### Loss & Training
Token selection is optimized using the GCG method: the PIEL gradient is calculated for each token, and the position with the largest gradient is selected for replacement. In each iteration, losses across all permutations are computed, gradients are aggregated, and token replacements are executed. The optimization runs for 500 GCG iterations with a search width of 64.

## Key Experimental Results

### Main Results

| Target Model | Method | JailbreakBench ASR | AdversarialBench ASR | In-the-wild ASR |
|---------|------|-------------------|---------------------|-----------------|
| Llama-2-7B | Vanilla | 0.000 | 0.000 | 0.144 |
| Llama-2-7B | GCG | 0.017 | 0.160 | 0.201 |
| Llama-2-7B | **Ours** | **0.726** | **0.533** | **0.561** |
| Mistral-7B | Vanilla | 0.000 | 0.000 | 0.215 |
| Mistral-7B | GCG | 0.324 | 0.212 | 0.203 |
| Mistral-7B | **Ours** | **0.812** | **0.543** | **0.627** |

### Ablation Study

| Factor | ASR Change | Description |
|------|---------|------|
| Chain Topology | ~60% | Worst connectivity, hardest to attack |
| Complete Topology | ~78% | Fully connected, most vulnerable |
| S-PIEL M=8 | 0% (does not converge) | Too few samples to approximate |
| S-PIEL M=64 | 56% | Good performance at around 50% of $K!$ |
| PIEL (M=K!) | 72.6% | Full permutation is optimal |

### Key Findings
- The proposed method improves the ASR on Llama-2-7B from 1.7% (with GCG) to 72.6% (JailbreakBench), representing an approximate 43x improvement.
- Existing security mechanisms (various versions of Llama-Guard, PromptGuard) struggle to detect the proposed attack, experiencing a drop in detection F1-score of about 30%, which indicates that chunk-based adversarial prompts are harder to flag.
- The attack demonstrates high cross-model transferability—attacks optimized on Llama-2 are also effective against Mistral and Gemma (61-71% ASR).
- Network topology plays an important role: Complete Graphs are the most vulnerable (78% ASR), whereas Chain topologies are the most secure (60% ASR).

## Highlights & Insights
- Modeling multi-agent attacks as a minimum-cost maximum-flow problem is an elegant formulation that perfectly captures the trade-off between bandwidth constraints and detection risks. This modeling approach can also be used in reverse for defense—minimizing the maximum flow of adversarial tokens.
- The permutation-invariant design of the PIEL loss directly addresses the core challenge of asynchronous communication, and this concept can be transferred to other scenarios requiring order-robust inputs (such as distributed inference).
- This paper reveals the counter-intuitive phenomenon that "increasing connectivity in multi-agent systems actually increases their vulnerability," providing important insights for secure system design.

## Limitations & Future Work
- Evaluation was limited to open-source models, without security testing on proprietary commercial models like GPT-4.
- Assuming that the attacker knows the exact network topology and security placement might be too strong of a knowledge assumption in some practical scenarios.
- The assumptions of static security detection and fixed bandwidth limits do not apply to dynamic, adaptive defensive systems.
- Only text-based interactions were considered; the attack surface of multimodal agent systems is broader and remains to be explored.

## Related Work & Insights
- **vs Evil Geniuses**: Evil Geniuses studies role-based adversarial prompt designs, while this work focuses on attacks at the communication topology level, offering a more systemic perspective.
- **vs Agent Smith**: Agent Smith focuses on exponential self-propagation but assumes unconstrained communication, whereas this work explicitly models real-world bandwidth and latency constraints.
- **vs Prompt Infection**: Prompt Infection studies self-replicating prompt injections, whereas this study focuses on optimized prompt distribution.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic study of attacks in constrained multi-agent systems, with highly innovative MCMF modeling and PIEL loss.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive quantitative evaluation across five models, three benchmarks, and multiple topologies.
- Writing Quality: ⭐⭐⭐⭐ Detailed technical descriptions, though the elaboration is somewhat long.
- Value: ⭐⭐⭐⭐ Highly critical findings that serve as a strong warning for multi-agent security design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Multi-Agent Framework for Mitigating Dialect Biases in Privacy Policy Question-Answering Systems](multi_agent_dialect_bias_privacy_qa.md)
- [\[ACL 2025\] iAgent: LLM Agent as a Shield between User and Recommender Systems](iagent_llm_agent_as_a_shield_between_user_and_recommender_systems.md)
- [\[ACL 2025\] AndroidGen: Building an Android Language Agent under Data Scarcity](androidgen_agent_data_scarcity.md)
- [\[NeurIPS 2025\] EU-Agent-Bench: Measuring Illegal Behavior of LLM Agents Under EU Law](../../NeurIPS2025/llm_agent/eu-agent-bench_measuring_illegal_behavior_of_llm_agents_under_eu_law.md)
- [\[ACL 2025\] REPRO-Bench: Can Agentic AI Systems Assess the Reproducibility of Social Science Research?](repro-bench_can_agentic_ai_systems_assess_the_reproducibility_of_research_claims.md)

</div>

<!-- RELATED:END -->
