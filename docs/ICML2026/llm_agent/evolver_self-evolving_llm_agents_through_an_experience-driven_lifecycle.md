---
title: >-
  [Paper Note] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle
description: >-
  [ICML 2026][LLM Agent][Experience Lifecycle] EvolveR wraps LLM agents in a closed-loop lifecycle: "Online Interaction $\rightarrow$ Offline Self-Distillation into Principle Libraries $\rightarrow$ GRPO Policy Evolution."…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "Experience Lifecycle"
  - "Self-Distilled Principle Library"
  - "Dynamic Scoring"
  - "GRPO"
  - "Multi-hop QA"
date: 2026-05-08
content_hash: 5331af7835d4e705
---

# EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle

**Conference**: ICML 2026  
**arXiv**: [2510.16079](https://arxiv.org/abs/2510.16079)  
**Code**: https://github.com/Edaizi/EvolveR (Available)  
**Area**: LLM Agent / Continual Learning / Reinforcement Learning  
**Keywords**: Experience Lifecycle, Self-Distilled Principle Library, Dynamic Scoring, GRPO, Multi-hop QA

## TL;DR
EvolveR wraps LLM agents in a closed-loop lifecycle: "Online Interaction $\rightarrow$ Offline Self-Distillation into Principle Libraries $\rightarrow$ GRPO Policy Evolution." Instead of discarding past trajectories, the agent abstracts successes and failures into retrievable "strategic principles" and uses RL to learn **how to apply its own principles** to solve new problems. It significantly outperforms RL agent baselines like Search-R1 on seven multi-hop QA benchmarks.

## Background & Motivation

**Background**: LLM agents (e.g., ReAct, Reflexion, ExpeL, Search-R1) have demonstrated viability in tool invocation but remain largely "stateless." Each task is handled independently, and past experiences are either discarded or temporarily injected as hints distilled by external LLM teachers.

**Limitations of Prior Work**: (1) Reflexion-style methods treat reflection as a "one-off hint" without updating the agent's internal policy; (2) Case-based retrieval using raw trajectories tends to overfit or copy answers directly on new tasks rather than using abstracted strategies; (3) Distilling experience from strong external teachers may lead to "cognitive misalignment" with the agent's own capability distribution, especially for smaller models; (4) RL agents like Search-R1/O2-Searcher optimize external search strategies well but fail to address learning from internal experience.

**Key Challenge**: Human experts grow through a continuous cycle of "Interaction—Reflection—Abstraction." Existing agent frameworks either short-circuit reflection (stateless), short-circuit abstraction (raw cases), or short-circuit internalization (prompt-only without policy updates).

**Goal**: Construct a complete closed loop where the agent generates its own trajectories, distills reusable strategic principles, and uses RL to internalize these principles, all without relying on external teachers.

**Key Insight**: Treat the "Principle Library" as an explicit, retrievable tool for the agent (equivalent to a search engine) and utilize GRPO to learn not just "how to solve problems" but also "how to utilize experience."

**Core Idea**: Integrate self-distilled principles, dynamic scoring maintenance, and "experience as an action" into a unified lifecycle coupled with RL policy evolution.

## Method

### Overall Architecture
The EvolveR main loop alternates between two phases. **Online phase**: The agent performs a Think-Act-Observe cycle with three types of actions—`<search_experience>` to query its own experience library $\mathcal{E}$, `<search_knowledge>` for external search, and `<answer>` for the final result. Trajectories $\tau_{\text{new}}$ are collected for training. **Offline phase**: With parameters frozen, the agent uses its policy $\pi_\theta$ as an "expert" to review recent trajectories and distills them into "Success Principles / Failure Principles"—each consisting of a natural language description plus structured knowledge triplets. New principles are written to $\mathcal{E}$ after deduplication, similarity merging, and dynamic scoring. Finally, $\pi_\theta$ is updated via GRPO on $\tau$ to close the loop.

Cold-start is managed via LoRA SFT on ~700 NQ/HotpotQA CoT trajectories to stabilize early RL before entering the lifecycle iterations.

### Key Designs

1.  **Experience Library $\mathcal{E}$ with Self-Distillation + Two-layer Deduplication**:
    - **Function**: Transforms the agent's successes/failures into a retrievable, non-redundant, and evolving set of strategic principles.
    - **Mechanism**: (a) Each trajectory $\tau$ is processed by $\pi_\theta$ itself (not an external teacher) to extract a candidate principle $p_{\text{cand}}$; (b) First-layer deduplication—merging semantically equivalent principles derived from multiple GRPO samples for the same question; (c) Second-layer merging—performing embedding retrieval within the global library $\mathcal{E}$ and binary semantic classification. If $\max_{p\in\mathcal{E}}\text{sim}(p_{\text{cand}},p)<\theta_{\text{sim}}$, it is added as a new entry $\mathcal{E}\leftarrow\mathcal{E}\cup\{p_{\text{cand}}\}$; otherwise, $\tau_{\text{src}}$ is merged into the most similar entry $p^*$, enriching its evidence without redundancy.
    - **Design Motivation**: Self-distillation avoids capability distribution misalignment from external teachers; two-layer integration prevents raw case library explosion and maintains retrieval quality.

2.  **Quality Control via Dynamic Scoring + Threshold Pruning**:
    - **Function**: Enables "survival of the fittest" within the experience library, ensuring high-value principles are prioritized in retrieval.
    - **Mechanism**: Each principle $p$ tracks usage count $c_{\text{use}}(p)$ and success count $c_{\text{succ}}(p)$, with a score calculated via Laplace smoothing: $s(p)=\frac{c_{\text{succ}}(p)+1}{c_{\text{use}}(p)+2}$. Principles falling below a threshold $\theta_{\text{prune}}$ are periodically pruned.
    - **Design Motivation**: Laplace smoothing provides reasonable default scores for new principles while allowing scores to converge to true success rates with high usage. Pruning is essential for library longevity.

3.  **Experience as RL Action + GRPO Closed-loop Training**:
    - **Function**: Teaches the agent not just to read experience, but to learn the policy of "when to retrieve experience and which experience is most useful."
    - **Mechanism**: The reward is a weighted sum of outcome reward (EM vs. ground truth) and format reward (encouraging presence of think, search, and answer steps, and calls to both search types): $R(\tau)=w_o R_{\text{outcome}}+w_f R_{\text{format}}$. The policy is optimized using GRPO: $\mathcal{J}_{\text{GRPO}}(\theta)=\mathbb{E}_\tau[\sum_t \min(\rho_t \hat A_t, \text{clip}(\rho_t,1-\epsilon,1+\epsilon)\hat A_t) - \beta D_{\text{KL}}[\pi_\theta\|\pi_{\text{ref}}]]$, with $G=8$ trajectories sampled per prompt for relative advantage estimation.
    - **Design Motivation**: GRPO is stable and critic-free. Comparing experience-guided trajectories via relative advantage strengthens the causal link between "Success Principle $\rightarrow$ Successful Outcome," which is critical for the EvolveR loop.

### Loss & Training
Cold-start: LoRA SFT (via Llama_Factory) on 700 CoT samples. RL stage: GRPO via the Verl framework, batch size 128 prompts, $G=8$, Adam lr $1\times 10^{-6}$, 20 warmup steps, mini-batch 128, on 8x A100. The format reward $R_{\text{format}}=\mathbb{I}(\tau_{\text{complete}})\cdot (R_{\text{think}}+R_{\text{search}})/2$ encourages both a reasonable number of thinking steps and diverse search invocations.

## Key Experimental Results

### Main Results
Evaluation conducted on 7 QA benchmarks, including In-domain (NQ, HotpotQA) and OOD (TriviaQA, PopQA, 2Wiki, Musique, Bamboogle). EM is the primary metric, comparing Qwen2.5-3B and 7B.

| Model | Method | NQ | HotpotQA | TriviaQA | PopQA | 2Wiki | Musique | Bamboogle | **Avg** |
|---|---|---|---|---|---|---|---|---|---|
| 3B | Direct | .106 | .149 | .288 | .108 | .244 | .020 | .024 | .134 |
| 3B | RAG | .348 | .255 | .544 | .387 | .226 | .047 | .080 | .270 |
| 3B | Search-R1-instruct | .341 | .324 | .545 | .378 | .319 | .103 | .264 | .325 |
| 3B | **Ours** | **.434** | **.373** | .584 | .434 | **.381** | **.137** | **.328** | **.382** |
| 7B | RAG | .349 | .299 | .585 | — | — | — | — | — |
| 7B | **Ours** | — | — | — | — | — | — | — | **.417** |

(EvolveR achieves a +5.7 EM gain over the strongest baseline Search-R1-instruct on the 3B model.)

### Ablation Study

| Configuration | Avg EM Change | Description |
|---|---|---|
| Full EvolveR | 0.382 (3B) | Complete experience lifecycle |
| Remove self-distillation (External) | Dec. (3B), Flat (7B)| Validates cognitive alignment is more critical for small models |
| Remove dedup + scoring | Lib inflation, performance Dec. | Curation is essential |
| Remove `<search_experience>` | Degrades to Search-R1 | RL cannot learn experience usage |
| Prompting + Raw case retrieval | Sig. Dec. | Abstracted principles $\gg$ raw trajectories |

### Key Findings
- **Self-distillation outperforms stronger external teachers** on the 3B model. This "cognitive alignment" suggests that principles produced by a giant model might exceed the execution capacity of a small agent, making them less actionable.
- The synergy between the **experience action and RL** is crucial. Using the principle library solely for RAG without updating the policy yields significantly lower gains than EvolveR.
- Gains are more pronounced on OOD datasets (e.g., Bamboogle), indicating that distilled "strategic principles" generalize better than specific factual memorization.

## Highlights & Insights
- **Experience as a Learnable Action**: Positioning `<search_experience>` and `<search_knowledge>` as first-class actions allows GRPO to directly optimize the gradient for querying experience—a fundamental shift from prompt-only memory frameworks.
- **Cognitive Alignment**: The finding that self-distillation aligns experience with the agent's capability distribution suggests that system "teachers" are not always better just because they are larger.
- **Sustainable Loop**: Dynamic scoring and pruning ensure the library remains maintainable over time, avoiding the collapse common in systems where "experience gets dirtier" as it grows.

## Limitations & Future Work
- The performance gap between self-distillation and external distillation narrow on 7B models, leaving the "cognitive alignment" benefit on 30B/70B+ models as an open question.
- Evaluation is focused on multi-hop QA; verification on long-horizon agentic tasks (web navigation, coding) is pending.
- Engineering challenges such as retrieval latency and embedding drift as the library grows were not fully discussed for long-term deployment.
- Manual "Format Reward" engineering (forcing specific actions) might introduce artificial calls, requiring further study on whether this hurts truly optimal strategies.

## Related Work & Insights
- **vs Reflexion / ExpeL**: These store reflections/experience without updating the internal policy; EvolveR uses experience to drive both retrieval and RL.
- **vs Search-R1 / O2-Searcher**: These use RL for external knowledge retrieval; EvolveR extends RL to manage the agent's internal experience.
- **vs Mem0 / G-Memory**: While principle structures are inspired by these (natural language + triplets), EvolveR embeds this structure into a complete RL lifecycle rather than performing simple retrieval.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Successfully closes the loop between experience lifecycle, self-distillation, and GRPO.
- Experimental Thoroughness: ⭐⭐⭐⭐ Broad coverage with 7 benchmarks, scaling analysis, and cognitive alignment controls.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of components and intuitive paradigm comparisons.
- Value: ⭐⭐⭐⭐ Provides a reproducible engineering paradigm for self-evolving agents, contributing significantly to long-horizon agent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Feedback-to-Plan Decisions for Self-Evolving LLM Agents in CUDA Kernel Generation](towards_feedback-to-plan_decisions_for_self-evolving_llm_agents_in_cuda_kernel_g.md)
- [\[ICML 2026\] Talk, Judge, Cooperate: Gossip-Driven Indirect Reciprocity in Self-Interested LLM Agents](talk_judge_cooperate_gossip-driven_indirect_reciprocity_in_self-interested_llm_a.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ICML 2026\] AutoRPA: Efficient GUI Automation through LLM-Driven Code Synthesis from Interactions](autorpa_efficient_gui_automation_through_llm-driven_code_synthesis_from_interact.md)
- [\[ICML 2026\] SE-GA: Memory-Augmented Self-Evolution for GUI Agents](se-ga_memory-augmented_self-evolution_for_gui_agents.md)

</div>

<!-- RELATED:END -->
