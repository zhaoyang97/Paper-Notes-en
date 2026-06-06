---
title: >-
  [Paper Note] RGMem: Renormalization Group-Inspired Memory Evolution for Language Agents
description: >-
  [ICML 2026][Recommender Systems][Renormalization Group] RGMem draws inspiration from the Renormalization Group (RG) in statistical physics…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Renormalization Group"
  - "Multi-scale Memory"
  - "User Profile Evolution"
  - "Threshold Phase Transition"
  - "Stability-Plasticity"
date: 2026-05-08
content_hash: 75a9995c83728071
---

# RGMem: Renormalization Group-Inspired Memory Evolution for Language Agents

**Conference**: ICML 2026  
**arXiv**: [2510.16392](https://arxiv.org/abs/2510.16392)  
**Code**: https://github.com/fenhg297/RGMem (Available)  
**Area**: LLM Agent / Long-term Memory / Personalized Dialogue  
**Keywords**: Renormalization Group, Multi-scale Memory, User Profile Evolution, Threshold Phase Transition, Stability-Plasticity  

## TL;DR
RGMem draws inspiration from the Renormalization Group (RG) in statistical physics, modeling the long-term dialogue memory of language agents as a multi-scale system ("Event Layer → Relation Layer → Concept Layer"). It uses threshold-triggered nonlinear operators to coarse-grain fragmented dialogues into stable user profiles, thereby breaking the "stability vs. plasticity" trade-off.

## Background & Motivation

**Background**: LLM-based dialogue agents are increasingly expected to maintain personalized interactions across sessions. Mainstream approaches either stuff history into the context (limited by window size + "lost-in-the-middle") or use RAG / explicit memory (such as Mem0, LangMem, A-Mem, Memory OS, Zep, etc.) for retrieval augmentation at the level of facts or paragraphs.

**Limitations of Prior Work**: Existing explicit memory systems are almost entirely "flat"—retrieving from a pool of facts based on lexical overlap and temporal recency—lacking the ability to extract stable traits from fragmented information. Consequently, noise accumulates, diluting long-term traits. Implicit memory (fine-tuning/LoRA/KV-cache) lacks auditability, is difficult to roll back, and cannot share profiles across agents.

**Key Challenge**: Long-term dialogues are inherently **multi-scale**: the micro-scale consists of specific events, the meso-scale involves patterns across contexts, and the macro-scale represents long-term personality traits. Updating all three within the same granularity inevitably leads to the classic "stability vs. plasticity" dilemma: overly aggressive updates overfit to noise and discard existing facts, while overly conservative updates fail to adapt to genuine preference drifts.

**Goal**: (1) Provide a principled "when to abstract across scales" mechanism for memory systems; (2) Make profile evolution robust to noise yet sensitive to real changes; (3) Achieve this without relying on larger context windows or more intensive retrieval.

**Key Insight**: The authors analogize memory to a physical system. In statistical physics, RG is a tool for studying how "high-frequency microscopic fluctuations are integrated out to leave macroscopic invariants." By treating "rapidly changing events" as microscopic degrees of freedom and "slowly changing traits" as macroscopic order parameters, the entire dialogue history becomes a system to be coarse-grained.

**Core Idea**: Profile evolution is defined as a sequence of threshold-triggered RG operators—changes are "promoted" to the meso or macro layers only when micro-layer evidence accumulation crosses a critical threshold. By separating the "order parameter $\Sigma$ (commonalities)" and the "correction term $\Delta$ (tensions)" at each scale, the system achieves phase-transition-like rather than linear monotonic profile updates.

## Method

### Overall Architecture
RGMem defines the memory state as $\mathcal{M} = \mathcal{D}_{L0} \times \mathcal{G}$, where $\mathcal{D}_{L0}$ is a set of episodic memory units obtained by segmenting and synthesizing raw dialogues, and $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ is a hierarchical dynamic knowledge graph. Physically, the system is divided into three layers: **L0** constructs micro-evidence (event-level facts + user conclusions); **L1** executes three RG operators for multi-scale evolution (relation inference $\mathcal{R}_{K1}$, node abstraction $\mathcal{R}_{K2}$, and hierarchy flow $\mathcal{R}_{K3}$); **L2** provides a scale-aware retrieval interface. When a dialogue occurs, the system first extracts episodic units $d = (\lambda_{fact}, \Lambda_{conc})$ in L0, then triggers L1 operators to update the graph, and finally, during querying, L2 mixes multi-scale contexts based on query granularity for the LLM. The system is read-only for the base LLM; all "learning" occurs within the external multi-scale graph and interpretable update rules.

The profile representation at each scale is formalized as an effective theory $\mathcal{T}(\mathcal{M}, s)$, parameterized by descriptors $\{\lambda_i(s)\}$. Relations between scales are linked via operators $\mathcal{T}^{(s+1)} = \mathcal{R}(\mathcal{T}^{(s)})$, corresponding to coarse-graining + rescaling. These operators are not explicit optimization targets; instead, the authors use an effective Hamiltonian $\mathcal{H}(\mathcal{T}) = \alpha E_{\text{con}}(\mathcal{T}) + \beta E_{\text{fid}}(\mathcal{T}|\mathcal{D}_{L0}) + \gamma E_{\text{com}}(\mathcal{T})$ (consistency + fidelity + compactness) as a heuristic guide. Since global optimization in text space is infeasible, a set of operator rules is used to "approximatively minimize" this Hamiltonian.

### Key Designs

1.  **Three-layer Multi-scale Memory State Space (L0 + Knowledge Graph $\mathcal{G}$)**:
    *   **Function**: Separates "fast-changing facts" from "slow-changing concepts" at the data structure level to prevent them from updating at the same rhythm.
    *   **Mechanism**: L0 uses $f_{cg} = f_{synth} \circ f_{seg}$ to slice and synthesize raw dialogues into episodic units $d = (\lambda_{fact}, \Lambda_{conc})$, where $\Lambda_{conc} = \Lambda_{base} \cup \Lambda_{rel}$ distinguishes "direct conclusions" from "high-salience signals." Then, $f_{extract}: \mathcal{D}_{L0} \to \mathcal{G}$ promotes these to the knowledge graph. Nodes are categorized into three levels: $\mathcal{V}_{abs}$ (abstract concepts), $\mathcal{V}_{gen}$ (general events), and $\mathcal{V}_{inst}$ (specific events), with edges divided into static classification edges $\mathcal{E}_{cls}$ and dynamic event edges $\mathcal{E}_{evt}$.
    *   **Design Motivation**: Flat RAG dumps all facts into one pool, where new facts dilute the retrieval weight of old traits. Explicit hierarchy ensures that "what I ate on Tuesday" and "user prefers fitness" are structurally separated, naturally supporting evolution at different tempos.

2.  **Three RG Operators for Coarse-graining + Threshold-triggered Updates**:
    *   **Function**: Transforms "when to promote new evidence to higher abstraction" into interpretable, schedulable rules.
    *   **Mechanism**:
        *   **Relational Operator $\mathcal{R}_{K1}$**: Accumulates new evidence $D_e^{new}$ for a specific relation $e$, updating the relation-level theory $\mathcal{T}_e^{(1,t+1)} \leftarrow \mathcal{T}_e^{(1,t)} + \beta(\mathcal{T}_e^{(1,t)}, D_e^{new})$. The nonlinear $\beta$ implementation via LLM fuses old summaries with new evidence. **This is triggered only when accumulated evidence exceeds $\theta_{\text{inf}}$**, preventing sparse noise from premature abstraction.
        *   **Node Abstraction Operator $\mathcal{R}_{K2} = \mathbb{S} \circ \mathbb{P}$**: Collects mixed-scale inputs $\mathcal{I}_v^{new} = \{\mathcal{T}_{e_i}^{(1),new}\}_{e_i \in N(v)} \cup \{d_j^{new}\}_{j \in D(v)}$ for an abstract node $v$. Projection-selection $\mathbb{P}$ prioritizes aggregated relation-level summaries and discards low-information micro-noise $D_v' = \mathbb{P}(\mathcal{I}_v^{new})$. Synthesis-rescaling $\mathbb{S}$ maintains two quantities: the **order parameter** $\Sigma_v^{(2,t+1)} = \text{Agg}_{\text{common}}(D_v', \Sigma_v^{(2,t)})$ capturing dominant stable patterns, and the **correction term** $\Delta_v^{(2,t+1)} = \text{Extract}_{\text{salient}}(D_v', \Delta_v^{(2,t)})$ preserving important signals that do not fit into $\Sigma$, controlled by threshold $\theta_{\text{sum}}$.
        *   **Hierarchy Flow Operator $\mathcal{R}_{K3}$**: Propagates bottom-up along static classification edges $\mathcal{E}_{cls}$. For a parent node $v_p$, representations are aggregated from child nodes $(\Sigma_{v_p}^{(s+1)}, \Delta_{v_p}^{(s+1)}) = \mathcal{R}_{K3}(\{(\Sigma_{v_{c_i}}^{(s)}, \Delta_{v_{c_i}}^{(s)})\}_i)$, using a dirty-flag mechanism for incremental scheduling.
    *   **Design Motivation**: The bifurcation into "order parameter + correction term" corresponds to "dominant ordered patterns + critical fluctuations" in physics. This ensures that internal contradictions in user behavior are not forced into a single summary. Threshold triggering corresponds to physical phase transitions, allowing the system to naturally exhibit two states: "ignore small perturbations" and "reorganize for large ones."

3.  **Multi-scale Retrieval L2 + Spectral Context Construction**:
    *   **Function**: Allows queries to automatically retrieve memory at corresponding scales based on granularity, rather than uniformly retrieving facts.
    *   **Mechanism**: When a query $q$ arrives, $f_{\text{retr}}(q, \mathcal{M})$ simultaneously accesses micro (episodic evidence), meso (relation-level summaries), and macro (node-level $\Sigma$ and $\Delta$) layers. These are combined into a unified context $C(q)$ based on query intent and fed to the LLM. Factual questions primarily draw from micro, while long-range reasoning or persona-based questions draw from macro.
    *   **Design Motivation**: Traditional RAG treats context as a single pool, where "information density vs. context length" hits a ceiling (the non-monotonic curve in Fig. 3 is direct evidence). Scale-aware retrieval allows more information matched to the query granularity to be packed within the same token budget.

### Loss & Training
RGMem does not train any parameters—all operators are LLM-based nonlinear update functions combined with threshold rules. "Training" occurs entirely through the evolution of the graph structure, making it interpretable, reversible, and auditable. Hyperparameters mainly include $\theta_{\text{inf}}$, $\theta_{\text{sum}}$, and L2 scale-retrieval budgets.

## Key Experimental Results

### Main Results
Evaluation was conducted on two long-term dialogue memory benchmarks: **LOCOMO** (long-context reasoning + temporal consistency, LLM-as-judge) and **PersonaMem** (128k token setting, dynamic persona evolution).

| Benchmark | Backbone | Method | Key Metric |
|-----------|----------|------|---------|
| PersonaMem (Avg) | GPT-4o-mini | Mem0 | 56.79 |
| PersonaMem (Avg) | GPT-4o-mini | A-Mem | 49.17 |
| PersonaMem (Avg) | GPT-4o-mini | Memory OS | 54.23 |
| PersonaMem (Avg) | GPT-4o-mini | **RGMem** | **63.87 (+7.08)** |
| PersonaMem (Avg) | GPT-4.1 | Memory OS | 65.03 |
| PersonaMem (Avg) | GPT-4.1 | **RGMem** | **74.01 (+8.98)** |
| LOCOMO (Avg) | gpt-4o-mini | Zep | 75.14 |
| LOCOMO (Avg) | gpt-4o-mini | **RGMem** | **78.92** |
| LOCOMO (Avg) | gpt-4.1-mini | Zep | 79.09 |
| LOCOMO (Avg) | gpt-4.1-mini | **RGMem** | **86.17** |
| LOCOMO (Avg) | gpt-4.1-mini | Full-Context | 87.52 |

RGMem improves over the second-best baseline by 7.08 / 8.98 points on PersonaMem (across two backbones). On LOCOMO, it approaches Full-Context performance while using a significantly lower context budget.

### Ablation Study

| Configuration | Key Metric Change | Description |
|------|------------|------|
| Full RGMem | Best (PersonaMem 63.87 / 74.01) | Full three-layer operators |
| Threshold $\theta_{\text{inf}} = 1$ (subcritical) | Significant drop | Every small noise triggers an update, overfitting to transient signals |
| Threshold $\theta_{\text{inf}} = 3$ (critical) | Peak performance | Critical point; balances stability and plasticity |
| Threshold $\theta_{\text{inf}} > 5$ (supercritical) | Significant drop | Updates suppressed; profile becomes rigid, failing real drifts |
| Remove any core component (App. B.4) | Irreparable by more context | Multi-scale design itself is the performance source, not context volume |
| Remove $\Delta$ (keep only $\Sigma$) | Drop in multi-hop / cross-scenario tasks | Critical fluctuations in $\Delta$ are key for complex scenarios |

### Key Findings
- **Critical Threshold Phenomenon (Phase Transition Characteristic)**: Treating $\theta_{\text{inf}}$ as a control parameter and task performance as an order parameter reveals a non-monotonic peak at $\theta_{\text{inf}} = 3$. This critical point reproduces across both benchmarks, framing "when to update a profile" as a phase transition.
- **Breaking the Stability–Plasticity Pareto Frontier**: On the PersonaMem Recall Facts × Latest Preference 2D plot, other baselines form a trade-off curve, whereas RGMem falls strictly beyond this frontier—achieving both retention and adaptation.
- **Optimal Scale for Effective Information Density**: On LOCOMO, accuracy increases as retrieved context grows from 3k to ~3.8k tokens, but decreases thereafter. Coarse-graining "integrates out" irrelevant micro-fluctuations, outperforming simply adding more tokens.
- **Macroscopic Invariance**: Under long-term consistent evidence, high-level profile representations converge and stabilize, exhibiting attractor-like behavior. This suggests $\Sigma$ genuinely captures context-invariant user traits rather than just the "last N items."

## Highlights & Insights
- **RG as an Engineering Lens, Not Just a Physics Metaphor**: The authors clarify that they are not performing physical RG but borrowing its "when to coarse-grain" design principles. This approach of "borrowing physical intuition for engineering" is highly transferable to any system requiring mixed-scale updates (e.g., continual learning, KG evolution, multi-agent collaboration).
- **Dual Preservation with $\Sigma + \Delta$**: Directly maps to physical order parameters and critical fluctuations. This is more principled than common "summary + exception list" designs, explaining why contradictions are not flattened.
- **Threshold as Phase Transition Parameter**: Reinterprets hyperparameters as "dynamics regimes," providing clear physical intuition for tuning instead of black-box grid searching.
- **Base-LLM Agnostic, External-Graph Architecture**: Allows deployment on any closed/open-source LLM, with profiles being auditable, reversible, and shareable across multiple agents.

## Limitations & Future Work
- The three operators $\mathcal{R}_{K1}, \mathcal{R}_{K2}, \mathcal{R}_{K3}$ are implemented via LLM calls, meaning LLM inference costs accumulate with graph size; quantified latency/cost estimations are not provided.
- While the consistency of the critical threshold $\theta_{\text{inf}}=3$ is striking, it is observed on only two datasets. Its universality across broader tasks remains to be verified.
- The effective Hamiltonian formula is elegant but not explicitly minimized; the "approximation" relationship between it and the operator design is qualitative.
- Hierarchy schemas ($\mathcal{V}_{abs} / \mathcal{V}_{gen} / \mathcal{V}_{inst}$) and static edges $\mathcal{E}_{cls}$ are domain-dependent, requiring redesign for new scenarios like long-term task planning.
- Lack of human user studies; persona simulations and LLM-as-judge may overestimate "profile stability" benefits while underestimating ambiguity or sarcasm in real interactions.

## Related Work & Insights
- **vs. Mem0 / A-Mem / Memory OS** (Chhikara et al., 2025; Rasmussen et al., 2025): These use explicit memory but retrieve/summarize at the fact level without "operator-triggered multi-scale abstraction." RGMem systematically outperforms them by 7–9 points on PersonaMem.
- **vs. LangMem / Zep**: Zep is the strongest hierarchical baseline on LOCOMO, but its abstraction rules are relatively linear/fixed-interval. RGMem's threshold-based nonlinear triggering explains its superiority in multi-hop and temporal tasks.
- **vs. GraphRAG / HippoRAG / AriGraph** (Edge et al., 2024; Gutiérrez et al., 2024; Anokhin et al., 2024): These organize memory hierarchically but use bottom-up uniform propagation. RGMem contributes by turning "when to propagate" into a controllable phase transition.
- **vs. Implicit Memory / KV-cache Adaptations** (Wang et al., 2024; Zhu et al., 2026): Those methods excel at style capture but are not auditable. RGMem provides auditability and shareability via explicit graphs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Re-framing memory evolution as a multi-scale dynamical system using RG is rare and theoretically rich.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two benchmarks + two backbones + threshold scanning + Pareto analysis; lacks human user studies and cost quantification.
- Writing Quality: ⭐⭐⭐⭐ Mapping from concepts to operators is clear; appendix provides theoretical support, though the Hamiltonian section is somewhat metaphysical.
- Value: ⭐⭐⭐⭐⭐ Points toward a "principled multi-scale" path for long-term agent memory design, directly applicable to production agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] From Recall to Forgetting: Benchmarking Long-Term Memory for Personalized Agents](../../ACL2026/recommender/from_recall_to_forgetting_benchmarking_long-term_memory_for_personalized_agents.md)
- [\[ICML 2026\] Learning Design Skills as Memory Policies for Agentic Photonic Inverse Design](learning_design_skills_as_memory_policies_for_agentic_photonic_inverse_design.md)
- [\[ICLR 2026\] In Agents We Trust, but Who Do Agents Trust? Latent Source Preferences Steer LLM Generations](../../ICLR2026/recommender/in_agents_we_trust_but_who_do_agents_trust_latent_source_preferences_steer_llm_g.md)
- [\[ICLR 2026\] GoalRank: Group-Relative Optimization for a Large Ranking Model](../../ICLR2026/recommender/goalrank_group-relative_optimization_for_a_large_ranking_model.md)
- [\[ACL 2026\] MemRec: Collaborative Memory-Augmented Agentic Recommender System](../../ACL2026/recommender/memrec_collaborative_memory-augmented_agentic_recommender_system.md)

</div>

<!-- RELATED:END -->
