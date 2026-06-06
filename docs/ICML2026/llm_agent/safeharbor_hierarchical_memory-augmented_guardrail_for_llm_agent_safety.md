---
title: >-
  [Paper Note] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety
description: >-
  [ICML 2026][LLM Agent][Guardrail] SafeHarbor upgrades LLM Agent safety defense from a "static coarse-grained classifier" to a "dynamic hierarchical memory tree + dual-score gating." Through adversarial rule generation an…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Guardrail"
  - "Agent Safety"
  - "Hierarchical Memory"
  - "Contrastive Learning"
  - "Over-Refusal"
date: 2026-05-08
content_hash: 53797f426a936dbc
---

# SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety

**Conference**: ICML 2026  
**arXiv**: [2605.05704](https://arxiv.org/abs/2605.05704)  
**Code**: [ljj-cyber/SafeHarbor](https://github.com/ljj-cyber/SafeHarbor)  
**Area**: AI Safety / LLM Agent  
**Keywords**: Guardrail, Agent Safety, Hierarchical Memory, Contrastive Learning, Over-Refusal

## TL;DR
SafeHarbor upgrades LLM Agent safety defense from a "static coarse-grained classifier" to a "dynamic hierarchical memory tree + dual-score gating." Through adversarial rule generation and information entropy-driven self-evolution, it enables GPT-4o to maintain a 93%+ refusal rate while increasing the benign tool invocation success rate to 63.6%, significantly alleviating the over-refusal problem.

## Background & Motivation
**Background**: LLM Agents can invoke tools and perform real-world actions (writing files, sending emails, calling APIs), which expands the attack surface from "outputting harmful text" to "executing harmful actions." Current defenses either (i) use auxiliary LLMs to monitor runtime (GuardAgent, ShieldAgent), (ii) fine-tune safety models (AgentAlign, Llama-Guard-3), or (iii) rely on static rule matching.

**Limitations of Prior Work**: Existing solutions treat safety boundaries as "globally fixed linear partitions." Attempting to strictly prevent malicious prompts often leads to the rejection of similar but legitimate complex benign workflows, resulting in severe over-refusal. Furthermore, auxiliary agents introduce prohibitive latency (e.g., ShieldAgent requires real-time code generation).

**Key Challenge**: There is an acute trade-off between safety strictness and utility on benign tasks. Stricter models are prone to over-refusal, while more permissive ones are easily bypassed. The root cause is that the boundary itself does not dynamically adjust to context.

**Goal**: To equip LLM Agents with a defense layer capable of "dynamically reconstructing safety boundaries for each query" without retraining the base model or adding heavy agent proxies, while maintaining acceptable latency.

**Key Insight**: Safety rules are viewed as "locally semantic clustered boundaries" rather than global thresholds. By using retrieval-based dynamic rule injection and training a lightweight Safety Projector to geometrizing the semantic space, the boundary is determined by the query's position.

**Core Idea**: A self-organized "Hierarchical Memory Tree" stores adversarially generated prohibition-exception pairs. A dual-center MLP Projector, trained with contrastive loss, provides both harmful and benign scores. Finally, a "fast path + fuzzy-zone LLM judge" gating mechanism determines whether to trigger full safety verification.

## Method

### Overall Architecture
SafeHarbor processes query $x$ in three stages: (I) **Adversarial Rule Generation** — Offline mutation of seed harmful trajectories into diverse variants, followed by an LLM rule generator producing contrastive rule pairs $\Pi_i=\{R_{\text{harm}},E_{\text{benign}}\}$; (II) **Dual Knowledge Storage** — Organizing rules into a two-layer memory tree $\mathcal{M}$ (upper layer for routing pivots, lower leaf layer for fine-grained rule pairs) while training a Safety Projector $f_\theta:\mathcal{X}\to\mathbb{R}^d$ with two learnable prototypes $\mathbf{w}_B,\mathbf{w}_H$; (III) **Online Retrieval and Scoring** — Utilizing dual-score gating, where most benign queries take the fast path for immediate release, while fuzzy or high-risk queries trigger rule retrieval and an LLM judge. The formalized trajectory goal is $\tau^*\in\mathcal{T}_{\text{refuse}}$ if $x\in\mathcal{T}_{\text{harm}}$, otherwise $\tau^*\in\mathcal{T}_{\text{exec}}$.

### Key Designs

1. **Adversarial Rule Generation + Entropy-driven Memory Evolution**:
    - **Function**: Automatically expands sparse harmful examples into a rule library covering three social engineering paradigms, determining whether to "create a new cluster," "add a leaf," or "merge and refine" based on information gain.
    - **Mechanism**: For each seed trajectory $\tau_h$, the generator uses Goal Decomposition, Privilege Escalation, and Contextual Reframing mutations to generate variants. The cosine distance between $z_h=f_\theta(\tau_h)$ and existing cluster centers is calculated. The logic employs Shannon entropy: $p_i=\exp(\text{Sim}(z_i,c)/\gamma)/\sum_j\exp(\text{Sim}(z_j,c)/\gamma)$, $H(C)=-\sum p_i\log_2 p_i$. Information gain is defined as $\Delta I(z_h,C^*)=H(C^*\cup\{z_h\})-H(C^*)$. A new cluster is created if the similarity to the nearest cluster is $<\tau_{\text{sim}}$; a new leaf is created if $\Delta I>\tau_{\text{gain}}$; otherwise, the nearest leaf's rule pair is refined.
    - **Design Motivation**: Traditional static rule libraries cannot keep pace with adversarial evolution, and relying on a single similarity threshold leads to structure explosion or redundant merging. Information entropy gain provides a statistical standard to determine if a sample brings a new distribution.

2. **Geometry-aware Dual-center Contrastive Safety Projector**:
    - **Function**: Maps queries into a geometric space anchored by two learnable prototypes $\mathbf{w}_B$ and $\mathbf{w}_H$, where distance directly reflects the degree of harmfulness.
    - **Mechanism**: The projector is a 2-layer MLP outputting $z'=\text{MLP}(z)$. Distances $d_B=\|z'-\mathbf{w}_B\|_2$ and $d_H=\|z'-\mathbf{w}_H\|_2$ are calculated, and the risk score is $s(x)=\exp(-d_H)/[\exp(-d_H)+\exp(-d_B)]$. Training optimizes a binary cross-entropy loss $\mathcal{L}_{cls}$ and a margin-based center-wise contrastive loss $\mathcal{L}_{con}=\frac{1}{|\mathcal{B}|}\sum_z \max(0,\Delta+\|z'-\mathbf{w}_y\|_2-\|z'-\mathbf{w}_{\neg y}\|_2)$, where $\mathcal{L}_{\text{total}}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{con}$.
    - **Design Motivation**: Pure cross-entropy leads to polarized scores (0 or 1), masking differences in ambiguous samples. Margin contrastive loss forces the latent space to be both separable and compact, allowing distance to measure the "semantic risk level."

3. **Dual-score Gating: Fast Path + LLM Judge**:
    - **Function**: Releases over 90% of common traffic from heavy LLM inference while maintaining precision.
    - **Mechanism**: For each query, (i) the projector's harmful probability $S_{\text{harm}}$ and (ii) the similarity to the nearest sample in the global benign DB $S_{\text{benign}}=1-\|\mathbf{z}_q-\mathbf{b}_{ret}\|_2^2/2$ are calculated. If $S_{\text{harm}}<\tau_{\text{low}}$ and $S_{\text{benign}}>\tau_{\text{high}}$, it proceeds via the fast path. Otherwise, top-$k$ clusters are retrieved to find the most similar leaf, and its rule pairs $\{R_{\text{harm}}, E_{\text{benign}}\}$ are provided to the base LLM for in-context judgment.
    - **Design Motivation**: Most agent requests are benign; performing rule retrieval and LLM judging for all queries is wasteful. The fast path only allows queries where dual evidence confirms they are benign.

### Loss & Training
Only the projector is trained: $\mathcal{L}_{\text{total}}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{con}$, while the base LLM remains fully frozen. The memory tree is constructed offline and evolves online without additional training. The system is plug-and-play for any frozen LLM agent.

## Key Experimental Results

### Main Results
Evaluated on GPT-4o and multiple base LLMs using benign and harmful requests, measuring Score, Full Pass, Refusal, and Non-Refusal.

| Model | Method | Harmful Refusal ↑ | Benign Score ↑ | Evaluation |
|-------|--------|------------------|---------------|-----|
| GPT-4o | No Defense | 58.0% | 44.2% | Over-permissive |
| GPT-4o | Rule Traverse | 100.0% | 12.1% | Severe over-refusal |
| GPT-4o | **SafeHarbor** | **93%+** | **63.6%** | Best trade-off |

SafeHarbor is the only solution in the table that achieves both "harmful refusal > 93%" and "benign utility > 60%."

### Ablation Study

| Configuration | Phenomenon | Explanation |
|------|------|------|
| Full SafeHarbor | 93%+ refusal / 63.6% benign | Main result |
| w/o $\mathcal{L}_{con}$ | Benign score decreases | Margin contrast is key for geometric separation |
| w/o fast path | Latency increases significantly | Fast path is core to latency optimization |
| Disable evolution | Attack success rate rises over time | Entropy-driven splitting is necessary |
| Single score ($S_{\text{harm}}$ only) | Over-refusal returns | Benign similarity is key to reducing false positives |
| Naive MoE/Linear Cls | Ambiguous query misclassification | Dual-center space provides stronger semantic structure |

### Key Findings
- Rotating through three social engineering paradigms for adversarial rule generation ensures the library covers structural (multi-step decomposition), authoritative (privilege escalation), and semantic (contextual framing) attacks.
- The information entropy gate $\Delta I$ distinguishes "new threats" from "variants" more effectively than fixed thresholds, preventing rule explosion while capturing new attack surfaces.
- The dual-prototype geometric space forces ambiguous query scores into the 0.3~0.7 range, providing informative continuous metrics for gating.

## Highlights & Insights
- Implementing "per-query reconstruction of safety boundaries" as a lightweight, deployable structure (projector + memory tree) creates a system that can be attached to closed-source LLMs like GPT-4o training-free.
- Contrastive rule pairs $\{R_{\text{harm}},E_{\text{benign}}\}$ are a sophisticated design to mitigate over-refusal, forcing the LLM judge to recognize explicit exception boundaries.
- The entropy-driven memory evolution mechanism is transferable to any retrieval-augmented system that needs to incorporate new patterns without index explosion (e.g., RAG, ToolBench).
- The fast path concept (using inexpensive dual scores to block the majority of traffic from heavy verification) should be standard for all LLM-as-a-Judge guardrails.

## Limitations & Future Work
- The "harmful score" evaluation depends on an LLM-based judge $\mathcal{M}_{\text{eval}}$, which possesses its own biases and limits.
- The three mutation paradigms are fixed; future work is needed to measure coverage against unknown attacks like multi-modal injections or long-horizon planning attacks.
- Potential "drift" or "forgetting" in the memory tree over long-term evolution has not been fully discussed—could adversarial prompts eventually pollute the rules?
- Thresholds $\tau_{\text{low}},\tau_{\text{high}}$ for the fast path are empirically set; adaptive strategies and recalibration costs for different domains are not yet quantified.
- Dependency on a large, clean benign query database might be a barrier for niche domains.

## Related Work & Insights
- **vs AgentAlign**: AgentAlign incorporates safety via SFT, which requires retraining; SafeHarbor is training-free and compatible with frozen LLMs.
- **vs Llama-Guard-3**: The latter is a static classifier unaware of agent tool execution context; SafeHarbor defines safety at the trajectory level.
- **vs GuardAgent / ShieldAgent**: These require online code generation and execution, leading to high latency; SafeHarbor circumvents this using a lightweight projector and retrieval.
- **vs A-Mem**: While A-Mem focuses on temporal knowledge networks, this work proposes "constraint-driven" safety memory.
- **Insight**: The "prototype anchored embedding" idea in dual-center geometry can be migrated to safety filtering in RAG retrieval and multi-modal content moderation.

## Rating
- Novelty: ⭐⭐⭐⭐ (Combining entropy evolution with adversarial rule pairs for agent guardrails)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Multiple LLMs and attack paradigms, though lacking long-horizon/multi-modal coverage)
- Writing Quality: ⭐⭐⭐⭐ (Clear three-stage framework, standard Algorithm 1)
- Value: ⭐⭐⭐⭐⭐ (Training-free, highly deployable for closed-source models)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SE-GA: Memory-Augmented Self-Evolution for GUI Agents](se-ga_memory-augmented_self-evolution_for_gui_agents.md)
- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ICML 2026\] Think Twice Before You Act: Enhancing Agent Behavioral Safety with Thought Correction](think_twice_before_you_act_enhancing_agent_behavioral_safety_with_thought_correc.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](../../ACL2026/llm_agent/hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ACL 2026\] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks](../../ACL2026/llm_agent/shopping_companion_a_memory-augmented_llm_agent_for_real-world_e-commerce_tasks.md)

</div>

<!-- RELATED:END -->
