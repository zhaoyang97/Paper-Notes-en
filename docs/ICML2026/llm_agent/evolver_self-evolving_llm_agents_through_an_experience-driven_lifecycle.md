---
title: >-
  [Paper Note] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle
description: >-
  [ICML 2026][LLM Agent][Experience Lifecycle] EvolveR establishes a closed-loop lifecycle for LLM agents: "online interaction → offline self-distillation into a principle library → GRPO policy evolution." Instead of disca…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Experience Lifecycle"
  - "Self-Distilled Principle Library"
  - "Dynamic Scoring"
  - "GRPO"
  - "Multi-hop QA"
date: 2026-05-08
content_hash: f8718455682b4a34
---

# EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle

**Conference**: ICML 2026  
**arXiv**: [2510.16079](https://arxiv.org/abs/2510.16079)  
**Code**: https://github.com/Edaizi/EvolveR (available)  
**Area**: LLM Agent / Continual Learning / Reinforcement Learning  
**Keywords**: Experience Lifecycle, Self-Distilled Principle Library, Dynamic Scoring, GRPO, Multi-hop QA

## TL;DR
EvolveR establishes a closed-loop lifecycle for LLM agents: "online interaction → offline self-distillation into a principle library → GRPO policy evolution." Instead of discarding past trajectories, the agent abstracts its own successes and failures into a retrievable set of "policy principles," then uses RL to learn **how to leverage its own principles** to solve new problems. On seven multi-hop QA benchmarks, it significantly outperforms RL agent baselines such as Search-R1.

## Background & Motivation

**Background**: LLM agents (ReAct, Reflexion, ExpeL, Search-R1, etc.) have achieved tool usage, but most are "stateless": each task is independent, past experience is either discarded or temporarily injected as hints distilled by external LLM teachers.

**Limitations of Prior Work**: (1) Reflexion-like methods treat reflection as a "one-off hint," without updating the agent's internal policy; (2) Retrieving raw trajectories (case-based) risks overfitting or copying answers on new tasks, rather than abstracting strategies; (3) Using strong external teachers for distillation may cause cognitive misalignment, especially for smaller models; (4) RL agents like Search-R1/O2-Searcher excel at learning strategies for external search, but do not address "learning from one's own experience."

**Key Challenge**: Human experts grow through a continuous cycle of "interaction–reflection–abstraction." Existing agent frameworks either short-circuit reflection (stateless), abstraction (raw case), or internalization (prompt-only, no policy update).

**Goal**: Construct a complete closed loop—agents generate their own trajectories, distill reusable policy principles, and use RL to learn to apply these principles, all without relying on external teachers.

**Key Insight**: Treat the "principle library" as an explicitly retrievable tool (on par with the search engine); enable GRPO to learn not only "how to solve problems" but also "how to use experience."

**Core Idea**: Self-distilled principles + dynamic scoring maintenance + experimental experience as actions—tightly integrating the experience lifecycle with RL policy evolution.

## Method

### Overall Architecture
EvolveR alternates between two phases in its main loop. **Online phase**: The agent, in a Think-Act-Observe loop, can take three types of actions—`<search_experience>` to query its own experience library $\mathcal E$, `<search_knowledge>` to query external search, and `<answer>` to provide the final answer. Trajectories $\tau_{\text{new}}$ are collected for subsequent training. **Offline phase**: With parameters frozen, the agent uses its own policy $\pi_\theta$ to act as an "expert," reviewing recent trajectories and distilling them into "success principles / failure principles"—each principle consists of a natural language description and several structured knowledge triples. Newly distilled principles are deduplicated, merged by similarity, dynamically scored, and then written into $\mathcal E$. Finally, GRPO is used to update $\pi_\theta$ on $\tau$, closing the loop.

Cold start uses ~700 NQ/HotpotQA CoT trajectories for LoRA SFT to stabilize early RL, before entering the lifecycle iteration.

### Key Designs

1. **Self-distillation + Two-level Deduplication for Experience Library $\mathcal E$**:

    - **Function**: Converts the agent's own successful/failed trajectories into a retrievable, non-redundant, and incrementally updatable set of policy principles.
    - **Mechanism**: (a) Each trajectory $\tau$ is used by $\pi_\theta$ itself (not an external teacher) to extract a candidate principle $p_{\text{cand}}$ via prompt; (b) First-level deduplication—semantically equivalent principles from multiple GRPO samples on the same problem are merged; (c) Second-level merging—embedding retrieval and binary semantic classification are performed across the entire library $\mathcal E$. If $\max_{p\in\mathcal E}\text{sim}(p_{\text{cand}},p)<\theta_{\text{sim}}$, $p_{\text{cand}}$ is added as a new entry; otherwise, $\tau_{\text{src}}$ is merged into the most similar entry $p^*$, enriching its evidence without increasing redundancy.
    - **Design Motivation**: Self-distillation avoids capability mismatch from external teachers; two-level integration prevents raw case library explosion and dilution of retrieval quality by duplicates.

2. **Dynamic Scoring + Threshold Pruning for Quality Control**:

    - **Function**: Enables the experience library to self-select, ensuring that genuinely reusable principles are prioritized in retrieval.
    - **Mechanism**: Each principle $p$ tracks usage count $c_{\text{use}}(p)$ and success count $c_{\text{succ}}(p)$, with Laplace smoothing: $s(p)=\frac{c_{\text{succ}}(p)+1}{c_{\text{use}}(p)+2}$. Entries below threshold $\theta_{\text{prune}}$ are periodically pruned to prevent the library from becoming cluttered.
    - **Design Motivation**: Laplace smoothing gives new, low-usage principles a reasonable default score, while high-usage scores approach true success rates; pruning is key to the library's longevity.

3. **Experience as RL Action + GRPO Closed-loop Training**:

    - **Function**: Enables the agent not only to read experience, but also to learn at the policy level "when to query experience and which experience is most useful."
    - **Mechanism**: The reward is a weighted sum of outcome reward (EM vs. ground truth) and format reward (encouraging at least one occurrence each of think, search, and answer, with both search_experience and search_knowledge invoked): $R(\tau)=w_o R_{\text{outcome}}+w_f R_{\text{format}}$. Policy is optimized with GRPO: $\mathcal J_{\text{GRPO}}(\theta)=\mathbb E_\tau[\sum_t \min(\rho_t \hat A_t, \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t) - \beta D_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]]$, with $G=8$ trajectories per prompt per batch for relative advantage estimation.
    - **Design Motivation**: GRPO requires no critic and is stable to train; relative advantage comparison among experience-guided trajectories strengthens the causal link between "successful principle → successful outcome"—the core of EvolveR's closed loop.

### Loss & Training
Cold start: LoRA SFT (LLama_Factory) on 700 CoT samples; RL phase: GRPO + Verl framework, batch size 128 prompts, $G=8$, Adam lr $1\times 10^{-6}$, warmup 20, mini-batch 128, 8 A100 GPUs. In the reward, $R_{\text{format}}=\mathbb I(\tau_{\text{complete}})\cdot (R_{\text{think}}+R_{\text{search}})/2$—rewarding both reasonable think steps and diverse search actions.

## Key Experimental Results

### Main Results
Seven QA benchmarks, divided into In-domain (NQ, HotpotQA) and OOD (TriviaQA, PopQA, 2Wiki, Musique, Bamboogle). EM is the main metric; full comparison for Qwen2.5-3B and 7B.

| Model | Method | NQ | HotpotQA | TriviaQA | PopQA | 2Wiki | Musique | Bamboogle | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| 3B | Direct | .106 | .149 | .288 | .108 | .244 | .020 | .024 | .134 |
| 3B | RAG | .348 | .255 | .544 | .387 | .226 | .047 | .080 | .270 |
| 3B | Search-R1-instruct | .341 | .324 | .545 | .378 | .319 | .103 | .264 | .325 |
| 3B | **EvolveR** | **.434** | **.373** | .584 | .434 | **.381** | **.137** | **.328** | **.382** |
| 7B | RAG | .349 | .299 | .585 | — | — | — | — | — |
| 7B | **EvolveR** | — | — | — | — | — | — | — | **.417** |

(More 7B rows are provided in Table 1 of the original paper; on 3B, EvolveR outperforms the strongest baseline Search-R1-instruct by an average of +5.7 EM.)

### Ablation Study

| Configuration | Avg EM Change | Notes |
|---|---|---|
| Full EvolveR | 0.382 (3B) | Complete experience lifecycle |
| Remove self-distillation, use strong external teacher | 3B drops, 7B unchanged | Shows cognitive alignment is more important for small models |
| Remove deduplication + scoring | Library bloats, performance drops | Curation is key |
| Remove `<search_experience>` action | Degrades to Search-R1 | RL cannot learn to "use own experience" |
| Prompt + raw case retrieval only | Significant drop | Abstract principles ≫ raw trajectory |

### Key Findings
- **Self-distillation outperforms strong external teacher distillation on 3B models**—counterintuitive but reasonable: teacher-generated principles may exceed the agent's execution capability and thus be unusable.
- The synergy between experience action and RL is crucial: simply using the principle library as RAG without policy updates yields much less improvement than EvolveR.
- Gains are more pronounced on OOD datasets (e.g., Bamboogle, adversarial multi-hop), indicating that distilled "policy principles" generalize better than memorizing specific facts.

## Highlights & Insights
- **"Experience as a learnable action" design**: Treats `<search_experience>` and `<search_knowledge>` as first-class actions, allowing GRPO to directly optimize "whether to query experience"—the fundamental distinction between EvolveR and all prompt-only memory frameworks.
- **Cognitive alignment**: Self-distillation aligns experience with the agent's own capability distribution, offering an open-ended insight—suggesting that "stronger" teachers are not always better for self-improving systems.
- **Closed-loop and maintainable**: Dynamic scoring and pruning keep the library sustainable, avoiding the common collapse of "experience pollution" seen in ExpeL-like systems.

## Limitations & Future Work
- On 7B models, self-distillation and external distillation perform similarly, suggesting that as base models grow stronger, the "cognitive alignment" benefit diminishes—unclear if this holds for 30B/70B.
- Experiments focus on multi-hop QA; validation on long-horizon agentic tasks (web navigation, code agents) is lacking—these are the scenarios where the lifecycle should shine.
- As the principle library grows, issues like retrieval latency and embedding drift in long-term deployment are not fully addressed.
- The format reward engineering (forcing all action types to appear) may introduce spurious calls; whether this harms truly optimal strategies requires further investigation.

## Related Work & Insights
- **vs Reflexion / ExpeL**: They store reflections/experience but do not update policy; EvolveR uses experience to drive both retrieval and RL.
- **vs Search-R1 / O2-Searcher**: These use RL to learn external knowledge retrieval; EvolveR goes further by making the agent's own experience a learnable target.
- **vs Mem0 / G-Memory**: Principle structure is inspired by them (natural language + knowledge triples), but EvolveR embeds this structure into a full RL lifecycle, not just retrieval.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Integrates experience lifecycle, self-distillation, and GRPO into a closed loop—a rare end-to-end solution for continual agent learning.
- Experimental Thoroughness: ⭐⭐⭐⭐ Seven benchmarks, multi-scale ablation, cognitive alignment comparison—broad coverage.
- Writing Quality: ⭐⭐⭐⭐ Clear component descriptions, Figure 1 intuitively contrasts four paradigms.
- Value: ⭐⭐⭐⭐ Provides a reproducible engineering paradigm for "agent self-evolution," contributing methodologically to long-horizon agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](../../ACL2026/llm_agent/expseek_self-triggered_experience_seeking_for_web_agents.md)
- [\[ICLR 2026\] Your Agent May Misevolve: Emergent Risks in Self-evolving LLM Agents](../../ICLR2026/llm_agent/your_agent_may_misevolve_emergent_risks_in_self-evolving_llm_agents.md)
- [\[ICML 2026\] Internalizing Agency from Reflective Experience](internalizing_agency_from_reflective_experience.md)
- [\[ICLR 2026\] InfiAgent: Self-Evolving Pyramid Agent Framework for Infinite Scenarios](../../ICLR2026/llm_agent/infiagent_self-evolving_pyramid_agent_framework_for_infinite_scenarios.md)

</div>

<!-- RELATED:END -->
