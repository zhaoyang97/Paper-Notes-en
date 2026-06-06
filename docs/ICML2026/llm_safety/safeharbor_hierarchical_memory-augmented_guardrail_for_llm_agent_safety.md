---
title: >-
  [Paper Note] SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety
description: >-
  [ICML 2026][LLM Safety][Guardrail] SafeHarbor upgrades LLM Agent safety from "static coarse-grained classifiers" to "dynamic hierarchical memory tree + dual-score gating." Through adversarial rule generation and entropy-…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Guardrail"
  - "Agent Safety"
  - "Hierarchical Memory"
  - "Contrastive Learning"
  - "Over-Refusal"
date: 2026-05-08
content_hash: 770de1e4220623d9
---

# SafeHarbor: Defining Precise Decision Boundaries via Hierarchical Memory-Augmented Guardrail for LLM Agent Safety

**Conference**: ICML 2026  
**arXiv**: [2605.05704](https://arxiv.org/abs/2605.05704)  
**Code**: [ljj-cyber/SafeHarbor](https://github.com/ljj-cyber/SafeHarbor)  
**Area**: AI Safety / LLM Agent  
**Keywords**: Guardrail, Agent Safety, Hierarchical Memory, Contrastive Learning, Over-Refusal

## TL;DR
SafeHarbor upgrades LLM Agent safety from "static coarse-grained classifiers" to "dynamic hierarchical memory tree + dual-score gating." Through adversarial rule generation and entropy-driven self-evolution, GPT-4o maintains a 93%+ refusal rate while raising benign tool invocation success to 63.6%, significantly alleviating the over-refusal problem.

## Background & Motivation
**Background**: LLM Agents can invoke tools and perform real-world actions (write files, send emails, call APIs), expanding the attack surface from "outputting harmful text" to "executing harmful actions." Mainstream defenses either (i) use auxiliary LLMs for runtime monitoring (GuardAgent, ShieldAgent), (ii) fine-tune safety models (AgentAlign, Llama-Guard-3), or (iii) rely on static rule matching.

**Limitations of Prior Work**: All these approaches treat the safety boundary as a "globally fixed linear split"—strictly blocking malicious prompts also blocks similar but legitimate benign workflows, causing severe over-refusal. Introducing auxiliary agents brings prohibitive latency (e.g., ShieldAgent requires real-time code generation).

**Key Challenge**: There is a sharp trade-off between safety strictness and utility on benign tasks; stricter boundaries increase over-refusal, looser ones are easier to bypass—the root cause is that "boundaries do not dynamically adjust with context."

**Goal**: Without retraining the base model or adding heavyweight agent proxies, equip LLM Agents with a defense layer that can dynamically reconstruct safety boundaries per query, while keeping latency within acceptable limits.

**Key Insight**: Treat safety rules as "locally clustered semantic boundaries" rather than global thresholds; inject rules dynamically via retrieval and train a lightweight Safety Projector to geometrize the semantic space, letting the boundary be determined by the query's own position.

**Core Idea**: Use a self-organizing "hierarchical memory tree" to store adversarially generated forbidden and exemption pairs, combined with a dual-center MLP Projector trained with contrastive loss to provide harmful/benign dual scores. A "fast path + fuzzy zone LLM judge" gating mechanism decides whether to trigger full safety verification.

## Method

### Overall Architecture
SafeHarbor processes query $x$ in three stages: (I) **Adversarial Rule Generation**—offline, seed harmful trajectories are mutated to generate diverse adversarial variants, then an LLM rule generator produces contrastive rule pairs $\Pi_i=\{R_{\text{harm}},E_{\text{benign}}\}$; (II) **Dual Knowledge Storage**—rules are structurally organized into a two-level memory tree $\mathcal{M}$ (upper layer: routing pivots; lower leaf: fine-grained rule pairs), while training a Safety Projector $f_\theta:\mathcal{X}\to\mathbb{R}^d$ with two learnable prototypes $\mathbf{w}_B,\mathbf{w}_H$; (III) **Online Retrieval & Scoring**—dual-score gating allows most benign queries to take the fast path, while ambiguous/high-risk queries retrieve relevant rules and invoke the LLM judge. The formal trajectory objective is $\tau^*\in\mathcal{T}_{\text{refuse}}$ if $x\in\mathcal{T}_{\text{harm}}$, otherwise $\tau^*\in\mathcal{T}_{\text{exec}}$.

### Key Designs

1. **Adversarial Rule Generation + Entropy-Driven Memory Tree Evolution**:

    - **Function**: Automatically expands scattered harmful samples into a rule base covering three social engineering paradigms, using information gain to decide "new cluster / new leaf / merge & refine."
    - **Mechanism**: For each seed trajectory $\tau_h$, the generator rotates through Goal Decomposition (decomposing malicious intent), Privilege Escalation (masquerading as high-priority debug requests), and Contextual Reframing (wrapping in educational/hypothetical scenarios) to generate diverse variants. Then, $z_h=f_\theta(\tau_h)$ computes cosine distance to existing cluster centers. The decision logic uses Shannon entropy: $p_i=\exp(\text{Sim}(z_i,c)/\gamma)/\sum_j\exp(\text{Sim}(z_j,c)/\gamma)$, $H(C)=-\sum p_i\log_2 p_i$, information gain $\Delta I(z_h,C^*)=H(C^*\cup\{z_h\})-H(C^*)$. If similarity to the nearest cluster $<\tau_{\text{sim}}$, create a new cluster; if $\Delta I>\tau_{\text{gain}}$, create a new leaf under the original cluster; otherwise, merge and refine the nearest leaf's rule pair.
    - **Design Motivation**: Traditional static rule bases cannot keep up with adversarial evolution, and relying on a single similarity threshold leads to tree explosion or redundant merging. Information entropy gain provides a statistical criterion for whether a sample introduces a new distribution, preventing rule bloat and missed detections.

2. **Geometric Dual-Center Contrastive Safety Projector**:

    - **Function**: Maps queries into a geometric space anchored by two learnable prototypes $\mathbf{w}_B,\mathbf{w}_H$, with distances directly reflecting harmfulness.
    - **Mechanism**: The projector is a 2-layer MLP, outputting $z'=\text{MLP}(z)$, computing $d_B=\|z'-\mathbf{w}_B\|_2, d_H=\|z'-\mathbf{w}_H\|_2$, and final risk score $s(x)=\exp(-d_H)/[\exp(-d_H)+\exp(-d_B)]$. Training jointly optimizes binary CE $\mathcal{L}_{cls}$ and margin-based center-wise contrastive loss $\mathcal{L}_{con}=\frac{1}{|\mathcal{B}|}\sum_z \max(0,\Delta+\|z'-\mathbf{w}_y\|_2-\|z'-\mathbf{w}_{\neg y}\|_2)$, with total loss $\mathcal{L}_{\text{total}}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{con}$.
    - **Design Motivation**: Pure CE leads to extreme scores (either 0 or 1), masking differences among ambiguous samples. Introducing margin contrastive loss forces the latent space to be both separable and compact, making distances truly measure "semantic risk level," supporting subsequent dual-score gating.

3. **Dual-Score Gating: Fast Path + LLM Judge**:

    - **Function**: Releases over 90% of ordinary traffic from heavyweight LLM inference while maintaining accuracy.
    - **Mechanism**: For each query, compute (i) harmful probability $S_{\text{harm}}$ from the projector, and (ii) similarity to the nearest neighbor $\mathbf{b}_{ret}$ in the global benign DB, $S_{\text{benign}}=1-\|\mathbf{z}_q-\mathbf{b}_{ret}\|_2^2/2$. If $S_{\text{harm}}<\tau_{\text{low}}$ and $S_{\text{benign}}>\tau_{\text{high}}$, take the fast path and allow directly; otherwise, enter "centralized rule retrieval"—select top-$k$ clusters, then find the most similar leaf within the cluster, and concatenate the leaf's forbidden $R_{\text{harm}}$ and exemption $E_{\text{benign}}$ as a prompt for in-context base LLM judgment.
    - **Design Motivation**: In practice, the vast majority of agent requests are plainly benign; performing rule retrieval and LLM judgment for all queries is wasteful. The fast path only allows queries with "dual evidence of benignity," while ambiguous cases pay the cost of complex verification, focusing latency where it matters most.

### Loss & Training
Only the projector is trained: $\mathcal{L}_{\text{total}}=\mathcal{L}_{cls}+\lambda\mathcal{L}_{con}$, with the base LLM fully frozen. The memory tree is constructed offline and evolves online without training. The entire system is plug-and-play and can be mounted in front of any frozen LLM agent.

## Key Experimental Results

### Main Results
Based on GPT-4o and multiple base LLMs, both benign and harmful requests are evaluated for "Score / Full pass / Refusal / Non-Refusal."

| Model | Method | Harmful Refusal ↑ | Benign Score ↑ | Evaluation |
|-------|--------|------------------|---------------|------------|
| GPT-4o | No Defense | 58.0% | 44.2% | over-permissive |
| GPT-4o | Rule Traverse | 100.0% | 12.1% | severe over-refusal |
| GPT-4o | **SafeHarbor** | **93%+** | **63.6%** | best trade-off |

SafeHarbor is the only method in the table achieving both "harmful refusal > 93%" and "benign utility > 60%."

### Ablation Study

| Configuration | Phenomenon | Explanation |
|---------------|------------|-------------|
| Full SafeHarbor | 93%+ refusal / 63.6% benign | Main result |
| Remove $\mathcal{L}_{con}$ contrastive loss | benign score drops | margin contrast is key for geometric separation |
| Remove fast path | latency rises significantly | fast path is core to latency optimization |
| Disable memory self-evolution (fixed rule base) | long-term attack pass rate increases | entropy-driven merge/split is necessary |
| Use single score ($S_{\text{harm}}$ only) | over-refusal returns | benign similarity is key to reducing false positives |
| Naive MoE/linear classifier | ambiguous samples misclassified | dual-center geometric space provides stronger semantic structure |

### Key Findings
- Adversarial rule generation rotates through three social engineering paradigms, ensuring the rule base covers structural (multi-step decomposition), authority (privilege escalation), and semantic (scenario wrapping) attacks—rules generated by a single paradigm are easily bypassed by similar adversarial prompts.
- Entropy gate $\Delta I$ distinguishes "new threats" from "similar variants" better than fixed similarity thresholds—avoiding both rule explosion and missing genuinely new attack surfaces.
- The dual-prototype geometric space places ambiguous queries' scores in the 0.3–0.7 range, providing informative continuous metrics for fast path / LLM judge gating.

## Highlights & Insights
- "Per-query dynamic safety boundary reconstruction" is implemented as a lightweight, deployable structure (projector + memory tree), and the system is training-free, mountable on closed-source LLMs like GPT-4o.
- Contrastive rule pairs $\{R_{\text{harm}},E_{\text{benign}}\}$ are a clever design to mitigate over-refusal—the same leaf specifies not only "what is forbidden" but also "what is the legitimate adjacent case," forcing the LLM judge to clarify exemption boundaries rather than blanket refusal.
- The entropy-driven memory evolution mechanism is transferable to any retrieval-augmented system that "needs to continually incorporate new patterns without index explosion" (e.g., RAG knowledge bases, ToolBench).
- The fast path concept (using cheap dual scores to keep most traffic out of heavyweight validation) should be standard for all LLM-as-a-Judge guardrails.

## Limitations & Future Work
- The "harmful score" evaluation depends on the LLM-based judge $\mathcal{M}_{\text{eval}}$, which has its own biases and upper limits.
- The three mutation paradigms (Goal Decomp/Privilege/Contextual Reframing) are fixed; coverage against unknown attack types (e.g., multimodal injection, long-horizon planning attacks) requires further assessment.
- "Drift" and "forgetting" in the memory tree's long-term evolution are not fully discussed—will adversarial prompts pollute the rule base after months of operation?
- The two fast path thresholds $\tau_{\text{low}},\tau_{\text{high}}$ are empirically set, with no adaptive strategy provided; the cost of recalibration across domains is unquantified.
- The benign DB requires a large, clean, pre-prepared legitimate query set, which may not be available for niche scenarios.

## Related Work & Insights
- **vs AgentAlign**: AgentAlign burns safety constraints into the model via SFT, requiring retraining and incurring retrain cost; SafeHarbor is training-free and compatible with any frozen LLM.
- **vs Llama-Guard-3**: The latter is a static content classifier, unaware of agent tool execution context; SafeHarbor directly defines trajectory-level safety.
- **vs GuardAgent / ShieldAgent**: The former generates and executes code online each time, resulting in high latency and fragile maintenance; SafeHarbor uses a lightweight projector + memory retrieval to bypass code-gen, with much lower end-to-end latency.
- **vs A-Mem and other memory mechanisms**: A-Mem focuses on temporally-aware knowledge networks, while this work proposes "time-independent, constraint-driven" safety memory, relevant to misevolution issues (Shao et al. 2025).
- **Insights**: The "prototype anchored embedding" concept of dual-center geometry + margin contrast can be transferred to RAG retrieval safety filtering, multimodal content moderation, etc.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces entropy-driven memory evolution + adversarial rule pairs to LLM agent guardrails
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple LLMs + multiple attack paradigms, but lacks coverage of long-horizon and multimodal attacks
- Writing Quality: ⭐⭐⭐⭐ Three-stage framework diagram is clear, Algorithm 1 is well-written
- Value: ⭐⭐⭐⭐⭐ Training-free, can be directly mounted on closed LLMs like GPT-4o, highly practical for engineering deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MemoPhishAgent: Memory-Augmented Multi-Modal LLM Agent for Phishing URL Detection](../../ACL2026/llm_safety/memophishagent_memory-augmented_multi-modal_llm_agent_for_phishing_url_detection.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)
- [\[ACL 2026\] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](../../ACL2026/llm_safety/into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)
- [\[ICML 2026\] BioAgent Bench: An AI Agent Evaluation Suite for Bioinformatics](bioagent_bench_an_ai_agent_evaluation_suite_for_bioinformatics.md)
- [\[ICML 2026\] From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG](from_volume_to_value_preference-aligned_memory_construction_for_on-device_rag.md)

</div>

<!-- RELATED:END -->
