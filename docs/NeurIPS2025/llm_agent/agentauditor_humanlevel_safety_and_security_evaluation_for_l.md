---
title: >-
  [Paper Note] AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents
description: >-
  [NeurIPS 2025][LLM Agent][Agent Safety Evaluation] This paper proposes AgentAuditor — a training-free, memory-augmented reasoning framework that enables LLMs to adaptively extract structured semantic features (scenario, risk, behavior) to construct an experiential memory bank, then employs multi-stage context-aware retrieval-augmented generation to guide LLM evaluators in assessing agent behavior for safety and security threats. The work also introduces ASSEBench, the first benchmark jointly covering safety and security evaluation (2,293 records, 15 risk types, 29 scenarios), achieving human expert-level evaluation accuracy across multiple benchmarks.
tags:
  - NeurIPS 2025
  - LLM Agent
  - Agent Safety Evaluation
  - LLM-as-Judge
  - Memory-Augmented Reasoning
  - RAG
  - Safety Benchmark
  - ASSEBench
date: 2026-05-08
content_hash: 4a6460c8a5811659
---

# AgentAuditor: Human-Level Safety and Security Evaluation for LLM Agents

**Conference**: NeurIPS 2025
**arXiv**: [2506.00641](https://arxiv.org/abs/2506.00641)
**Code**: [GitHub](https://github.com/Astarojth/AgentAuditor-ASSEBench)
**Area**: LLM Agent / AI Safety
**Keywords**: Agent Safety Evaluation, LLM-as-Judge, Memory-Augmented Reasoning, RAG, Safety Benchmark, ASSEBench

## TL;DR

This paper proposes AgentAuditor — a training-free, memory-augmented reasoning framework that enables LLMs to adaptively extract structured semantic features (scenario, risk, behavior) to construct an experiential memory bank, then employs multi-stage context-aware retrieval-augmented generation to guide LLM evaluators in assessing agent behavior for safety and security threats. The work also introduces ASSEBench, the first benchmark jointly covering safety and security evaluation (2,293 records, 15 risk types, 29 scenarios), achieving human expert-level evaluation accuracy across multiple benchmarks.

## Background & Motivation

**Background**: With the rapid advancement of LLM-based agents, agents are no longer merely "text generators" but can execute actions in real-world environments — calling APIs, manipulating file systems, autonomously browsing the web, and controlling smart devices. This shift from "text generation" to "action execution" makes safety evaluation of agent behavior critically important: a misconfigured agent may not merely produce harmful content but directly execute harmful operations with real-world consequences. Nevertheless, how to reliably evaluate the safety and security threats of agent behavior remains a significant challenge.

**Limitations of Prior Work**: Existing evaluation methods suffer from four fundamental deficiencies. First, **rule-based evaluators** rely on predefined string matching or keyword detection rules and fail to capture latent risks embedded in multi-step agent action sequences — for instance, a series of seemingly innocuous operations (multiple small transfers) may cumulatively lock a user's entire funds into high-risk illiquid investments, a "Harmless Operation Accumulation" that step-level inspection cannot detect. Second, **LLM-based evaluators** possess stronger semantic understanding but tend to miss subtle contextual nuances, or conversely, "over-analyze" a perfectly safe interaction — flagging a request to "draft a team lunch invitation" as a privacy leakage risk. Third, both types of evaluators **struggle with ambiguous safety boundaries**: for example, when an agent clusters user browsing behavior for marketing purposes, whether this constitutes "helpful business analytics" or "privacy violation" depends entirely on unobservable factors such as user consent and data anonymization. Fourth, **there is no unified standard** — a safe operation in one domain may be hazardous in another (e.g., running a red light during an emergency in autonomous driving: saving a life vs. breaking the law is an agent-specific trade-off).

**Key Challenge**: The fundamental issue is that agent risks cannot be detected via text-level rules or pattern matching as in traditional NLP safety. Agent risks arise from **the consequences of actions within specific environmental contexts**, demanding that evaluators possess human expert-level integrative reasoning — understanding scenario context, tracking multi-step state transitions, and weighing ambiguous boundary cases. Existing methods are either too rigid (rule-based) or too unstable (bare LLM), and neither is adequate for this depth of reasoning. Furthermore, existing evaluation benchmarks have serious limitations: R-Judge is the only dataset designed for "evaluator evaluation," yet it contains only 569 records and 10 risk types, does not distinguish between safety and security, uses binary labels without handling ambiguity, and has already been pushed to approximately 96% F1 by state-of-the-art methods, leaving little room for further progress.

**Goal**: Two core questions are addressed: (1) How can an LLM evaluator achieve human expert-level judgment on agent safety and security threats? (2) How can a sufficiently comprehensive and challenging benchmark be constructed to measure the true capabilities of evaluators?

**Key Insight**: The authors depart from a key observation — human experts make accurate safety judgments not because they are inherently smarter, but because they possess *experience*: memories of handling similar cases and internalized reasoning patterns. If an LLM could likewise possess such "experiential memory" — having encountered similar scenarios and knowing how to reason in analogous situations — its judgment capability could be substantially improved. This hypothesis motivates a "cognitive framework" design: rather than attempting to inject all safety knowledge into the LLM's parameters via fine-tuning, a structured external experiential memory system is built outside the LLM, enabling it to "consult prior experience" at each evaluation step.

**Core Idea**: By automatically constructing structured experiential memory (extracting features along scenario-risk-behavior dimensions and generating reasoning chains), and using multi-stage retrieval to dynamically inject the most relevant "reasoning templates" into the LLM evaluator, AgentAuditor achieves training-free, human expert-level agent safety evaluation.

## Method

### Overall Architecture

AgentAuditor follows a cognitive paradigm of "first accumulate experience, then apply it," comprising two major phases:

**Phase 1: Experiential Memory Construction.** Given a set of annotated historical agent interaction records, AgentAuditor uses an LLM to adaptively extract structured semantic triples $(s_i, t_i, b_i)$ from each record — representing scenario type, risk type, and behavior pattern — and simultaneously generates detailed Chain-of-Thought (CoT) reasoning traces explaining why each record is safe or unsafe. All records are then embedded using Nomic-Embed-Text-v1.5, dimensionality-reduced via PCA, and clustered using FINCH hierarchical clustering. The most representative sample from each cluster is selected as a "representative memory unit." This is a one-time offline operation.

**Phase 2: Multi-Stage Context-Aware RAG Evaluation.** For a new agent interaction record to be evaluated, the system first extracts the same structured semantic features, then performs two-stage retrieval: the first stage performs content-based semantic similarity coarse recall of candidate memory units (top-$n$, $n=8$); the second stage performs weighted re-ranking based on extracted structured features, ultimately selecting the top-$k$ ($k=3$) most relevant memory units. These three units and their corresponding CoT reasoning traces are injected into the LLM evaluator's prompt as "reasoning templates" to guide judgment.

The entire pipeline requires no model parameter training or fine-tuning — memory construction relies on the LLM's zero-shot feature extraction and unsupervised clustering; evaluation relies on in-context learning.

### Key Designs

1. **Adaptive Structured Semantic Feature Extraction**:

    - *Function*: For each agent interaction record $r_i$, an LLM automatically extracts a structured triple $(s_i, t_i, b_i)$, where $s_i$ denotes scenario type (e.g., "smart home control," "financial assistant," "web browsing"), $t_i$ denotes risk type (e.g., "privacy leakage," "property loss," "physical harm"), and $b_i$ denotes behavior pattern (e.g., "unauthorized data access," "cumulative operational risk"). A complete CoT reasoning trace — detailing *why* the record is safe or unsafe — is also generated.
    - *Mechanism*: Rather than representing an entire record through a single embedding (too coarse-grained), the interaction record is decomposed along three orthogonal semantic dimensions. This "structure-first, retrieve-second" strategy ensures that subsequent retrieval is not limited to surface-level textual similarity but also matches across the more abstract scenario-risk-behavior dimensions. The LLM here acts not as a judge but as an "experience extractor" — making implicit safety-relevant features explicit.
    - *Design Motivation*: Agent interaction records are typically long and contain substantial irrelevant information; direct embedding similarity retrieval is prone to noise interference. Structured feature extraction functions as a "denoising" layer — retaining only the three dimensions most relevant to safety evaluation. Furthermore, the same risk type may appear across entirely different scenarios (e.g., "privacy leakage" in both a social media assistant and a medical consultation agent), and structured representations enable cross-scenario experience transfer.

2. **Representative Memory Selection — PCA + FINCH Clustering**:

    - *Function*: A small set of "most representative" records is selected from all annotated interaction records via unsupervised clustering, compressing information while preserving diversity.
    - *Mechanism*: All interaction records are embedded using Nomic-Embed-Text-v1.5, followed by PCA dimensionality reduction (to attenuate noise dimensions) and FINCH (First Integer Neighbor Clustering Hierarchy) unsupervised hierarchical clustering. FINCH's key advantage is that it requires no predefined number of clusters — the clustering granularity is determined automatically by the data distribution. After clustering, the sample closest to each cluster centroid is selected as the representative memory unit. For example, ASSEBench-Safety's 1,476 records are compressed into 72 representative memories; R-Judge's 564 records are compressed into 24.
    - *Design Motivation*: Why not use all annotated data as memory? Because RAG effectiveness is constrained by the LLM's context window, and excessive retrieval results introduce noise. Cluster-based representative selection is equivalent to "replacing large amounts of redundancy with a small number of essentials" — ensuring the memory bank covers all major pattern types while controlling its size to guarantee retrieval quality. Experiments comparing multiple clustering algorithms confirm FINCH's superiority: on R-Judge it achieves F1 = 96.31%, substantially outperforming K-Means (88.11%), K-Medoids (89.66%), and DBSCAN (85.94%), validating the advantages of FINCH's hyperparameter-free design.

3. **Multi-Stage Context-Aware Retrieval-Augmented Generation**:

    - *Function*: For a new interaction record to be evaluated, two-stage retrieval is performed from the experiential memory bank to identify the top-$k=3$ most relevant memory units and their CoT reasoning traces, which are injected into the LLM evaluator's prompt.
    - *Mechanism*: **Stage 1 — Content-level coarse recall**: Cosine similarity is computed between the query record's embedding and all memory unit embeddings, retrieving top-$n$ ($n=8$) candidates based on overall semantic similarity to rapidly narrow the search space. **Stage 2 — Feature-level fine re-ranking**: The structured triple $(s_q, t_q, b_q)$ is also extracted for the query record; feature-level matching is then performed against the $n=8$ candidates' triples, computing a weighted composite score. Feature matching weights are determined via grid search on an independent development set (R-Judge) and fixed for all datasets. The top-$k=3$ candidates with the highest composite scores serve as the final reasoning guides.
    - *Design Motivation*: Pure embedding-based semantic retrieval can be misled by surface similarity — two scenario conversations may be textually similar yet involve entirely different risk types and behavior patterns. The multi-stage approach combines high recall in the first stage with high precision via structured features in the second. Experiments show that replacing multi-stage retrieval with single-stage retrieval degrades performance by 3–5%. Critically, when retrieved shots are not perfect matches (e.g., cross-dataset retrieval), the framework remains effective — because CoT reasoning traces provide a "reasoning approach" rather than merely a label; even when scenarios are not identical, reasoning patterns retain reference value.

### Loss & Training

AgentAuditor is a **completely training-free** framework requiring no model parameter fine-tuning or gradient updates. Three core mechanisms replace conventional training:

- **Experiential memory construction** replaces learning from training data: structured features and reasoning traces are extracted from annotated data via the LLM's zero-shot capability in a one-time offline operation — effectively "digesting training data using the LLM's existing knowledge."
- **FINCH unsupervised clustering** replaces model selection: memory bank size and representative samples are automatically determined without hyperparameter tuning.
- **In-context learning** replaces model fine-tuning: at evaluation time, retrieved reasoning experiences are injected as few-shot examples into the prompt, leveraging the LLM's powerful contextual learning ability for immediate inference without modifying model weights.

This design yields two notable advantages: first, **ease of deployment** — plug-and-play, with no GPU training resources required; second, **strong scalability** — when new risk types or scenarios emerge, it suffices to add new annotated records to the memory bank and re-run clustering, with no model retraining needed. Regarding weight selection, heuristic weights for feature matching are determined via systematic grid search on the R-Judge development set (predefined weight combinations, each tested three times with the average taken), then fixed for all experiments.

### ASSEBench

Beyond methodological contributions, the paper introduces **ASSEBench (Agent Safety and Security Evaluation Benchmark)** — the first benchmark designed specifically for evaluating "evaluators," jointly covering both safety and security.

**Scale and Coverage**: ASSEBench contains 2,293 finely annotated agent interaction records spanning 15 risk types and 29 application scenarios — more than four times the scale of R-Judge (569 records / 10 risk types).

**Systematic Safety–Security Separation**: ASSEBench is the first benchmark to systematically distinguish "safety" from "security" at the dataset level. Although the two terms are frequently conflated in everyday usage, they carry fundamentally different meanings in computer security — safety refers to the requirement that agent behavior must not cause harm to users or the environment (e.g., privacy leakage, physical harm), while security refers to the requirement that agents must be resilient against external attacks and malicious inputs (e.g., prompt injection, jailbreak attacks). Evaluating both separately enables more precise diagnosis of an evaluator's weaknesses.

**Dual-Standard Design — Strict vs. Lenient**: This is one of ASSEBench's most innovative design choices. Many agent interactions present ambiguous safety boundaries — the same behavior may be judged safe or unsafe depending on interpretation. ASSEBench introduces an "ambiguous flag" for such cases and provides two judgment standards: the Strict standard (err on the side of caution; ambiguous cases are classified as unsafe) and the Lenient standard (apply charitable interpretation; ambiguous cases are classified as safe). This design allows researchers to assess evaluator behavior under different safety preferences and more closely reflects the reality that different application scenarios genuinely require different safety thresholds.

**Evaluation Headroom**: R-Judge has already been driven to approximately 96% F1 by state-of-the-art methods, leaving little room for improvement. ASSEBench introduces ambiguous scenarios and the security dimension, posing greater challenges to evaluators — state-of-the-art methods retain substantial room for improvement on multiple subsets, making it a more valuable tool for measuring future progress.

## Key Experimental Results

### Main Results

AgentAuditor was comprehensively evaluated on 4 datasets spanning 8 evaluation subsets. Key results using Gemini-2-Flash-Thinking as the base model (F1 score, %):

| Dataset Subset | Base Model (F1) | Base Model (W-F1) | AgentAuditor (F1) | AgentAuditor (W-F1) | Gain |
|---|---|---|---|---|---|
| ASSE-Safety | 61.79 | 67.82 | 91.59 | 90.85 | +29.8 |
| ASSE-Security | 67.25 | 72.34 | 93.17 | 93.15 | +25.9 |
| ASSE-Strict | — | — | ~90+ | ~90+ | Substantial |
| ASSE-Lenient | — | — | ~90+ | ~90+ | Substantial |
| R-Judge | 82.27 | 81.21 | 96.31 | 96.10 | +14.0 |
| AgentHarm | — | — | Substantial | Substantial | — |
| AgentSecurityBench | — | — | Substantial | Substantial | — |
| AgentSafetyBench | — | — | Substantial | Substantial | — |

Comparison using GPT-4.1 (R-Judge):

| Method | F1 (%) | W-F1 (%) |
|---|---|---|
| Base Model (GPT-4.1) | 81.03 | 77.84 |
| Agent-as-a-Judge (GPT-4.1) | 83.85 | 81.56 |
| **AgentAuditor (GPT-4.1)** | **94.18** | **93.95** |

Key finding: AgentAuditor **consistently** improves all LLM evaluators across all benchmarks, often by large margins — achieving up to +48.2% F1 on ASSEBench-Safety for certain base models — and substantially outperforms Agent-as-a-Judge (+10.33 F1).

### Ablation Study

**Clustering algorithm comparison** (R-Judge, Gemini-2-Flash-Thinking):

| Clustering Method | Representative Shots | F1 (%) | W-F1 (%) |
|---|---|---|---|
| No memory (Base Model) | 0 | 82.27 | 81.21 |
| K-Means | 24 | 88.11 | 86.96 |
| K-Medoids | 24 | 89.66 | 86.79 |
| DBSCAN | 14 | 85.94 | 84.12 |
| **FINCH** | **24** | **96.31** | **96.10** |

**Retrieval relevance ablation** (R-Judge):

| Configuration | F1 (%) | W-F1 (%) | Description |
|---|---|---|---|
| AgentAuditor (Top 1–3) | 96.31 | 96.10 | Most relevant 3 retrieved |
| Ranks 4–6 | 92.74 | 91.30 | Forced use of ranks 4–6 (less relevant) |
| Random Shots | 85.07 | 82.80 | 3 randomly selected |
| Base Model | 82.27 | 81.21 | No memory augmentation |

**Cross-dataset transfer experiment** (validating OOD generalization):

| Memory Source | Evaluation Target | F1 (%) | W-F1 (%) |
|---|---|---|---|
| No memory | ASSE-Safety | 61.79 | 67.82 |
| R-Judge (cross-domain) | ASSE-Safety | 86.36 | 86.10 |
| ASSE-Safety (in-domain) | ASSE-Safety | 91.59 | 90.85 |
| No memory | ASSE-Security | 67.25 | 72.34 |
| R-Judge (cross-domain) | ASSE-Security | 84.55 | 85.12 |
| ASSE-Security (in-domain) | ASSE-Security | 93.17 | 93.15 |

**Adversarial robustness test** (R-Judge, three attack levels):

| Attack Level | Poisoning Rate | Avg. Poisoned Shots Retrieved | F1 (%) | W-F1 (%) |
|---|---|---|---|---|
| None (Base) | 0 | 0 | 82.27 | 81.21 |
| None (AgentAuditor) | 0 | 0 | 96.31 | 96.10 |
| Level 1 (gray-box data poisoning) | 1.1% (6/564) | 0 | 96.28 | 96.15 |
| Level 1 (gray-box data poisoning) | 9.9% (56/564) | 0.19 | 92.64 | 91.79 |
| Level 2 (strong gray-box memory poisoning) | 8.3% (2/24) | 0.23 | 90.35 | 89.36 |
| Level 2 (strong gray-box memory poisoning) | 33.3% (8/24) | 0.96 | 85.07 | 82.80 |
| Level 3 (white-box decision attack) | 33.3% (1/3) | 1.0 | 84.65 | 82.72 |

**Domain-level performance breakdown** (ASSEBench, GPT-4.1):

| Domain | Safety F1 (%) | Security F1 (%) |
|---|---|---|
| IT System | 100 | 100 |
| Web Browse | 93.75 | 96.77 |
| Financial | 80.00 | 96.77 |
| Health Support | 77.67 | 88.89 |
| Content Creation | 75.19 | 66.67 |
| Auto. Navigation | 40.00 | 58.33 |

### Key Findings

- **Superiority of FINCH clustering**: FINCH requires no preset cluster count and achieves the best performance, surpassing the second-best method (K-Medoids) by 6.65% F1. More critically, FINCH constitutes a "moving target" — even if an attacker carefully crafts poisoned samples targeting the current clustering result, re-running FINCH will likely produce an entirely different set of representative samples, neutralizing the attack.

- **CoT reasoning traces > label information**: Comparing Random Shots (labeled but reasoning-irrelevant) against Top 1–3 (labeled and reasoning-relevant) reveals a gap of 11.24% F1, demonstrating that AgentAuditor's core value lies not in providing "reference answers" but "reasoning templates" — teaching the LLM *how to think* is more effective than telling it *what the answer is*.

- **Graceful degradation**: Even when memory from a completely different domain is used (R-Judge → ASSE-Safety/Security), AgentAuditor still substantially outperforms the no-memory baseline (+24.57 / +17.30 F1). Even under the most extreme white-box attack (1/3 of reasoning examples maliciously replaced), performance remains superior to the unprotected base model.

- **Domain dependence**: Performance is highest in "computer-centric" domains (IT System, Web Browse) — where features are clear and rules explicit — and lowest in domains requiring "real-world common sense and long-range dependencies" (autonomous driving, navigation). This reveals the ceiling of the current approach: for safety evaluation requiring physical-world reasoning, purely text-level experiential memory remains limited.

- **Manageable efficiency overhead**: The computational cost of PCA + FINCH is negligible relative to LLM inference. At 50× the data scale (28,200 records), PCA takes only 2.61 seconds and FINCH takes 1.94 seconds. The main additional cost of AgentAuditor over direct LLM evaluation comes from retrieval and longer prompts (containing retrieved CoT). Total overhead is reasonable — evaluating R-Judge on QwQ-32B takes 6.86 A100-hours (including 1.07 hours for memory construction + 5.79 hours for evaluation), compared to 2.72 hours for direct evaluation. Given the performance improvement from ~80% to 96%, this trade-off is well justified.

- **Comparison with fine-tuning**: On Qwen2.5-7B, fine-tuning requires 4,000 training records and 16 A100-hours of training to achieve 83.67% F1; AgentAuditor requires only 24 representative memories and 1.35 A100-hours, achieving a lower F1 of 76.69%. However, on large models (QwQ-32B), AgentAuditor substantially outperforms fine-tuned variants, indicating that it is especially well-suited for "strong base model + training-free" deployment scenarios.

## Highlights & Insights

- **Paradigm shift from "monolithic reasoning" to "experience-augmented reasoning."** AgentAuditor does not treat the LLM as an omnipotent monolithic reasoner; instead, it equips it with an external structured "experiential brain." The design philosophy is elegant — human experts make accurate judgments not because they are inherently more intelligent but because they have encountered more cases and internalized reasoning patterns. AgentAuditor essentially simulates the cognitive process of a human expert: accumulate experience → organize structurally → associate similar cases → reason by reference.

- **The Strict/Lenient dual-standard design of ASSEBench is particularly elegant.** In safety evaluation, the boundary between "safe" and "unsafe" has never been black and white; the ambiguous zone is often the most important and most difficult to handle. By providing two standards for ambiguous cases, ASSEBench enables researchers to assess evaluator behavior under different "safety preferences" — a far richer characterization than simple binary labels, and more reflective of real-world needs where different application scenarios genuinely require different safety thresholds.

- **The "denoising" effect of structured triples.** Decomposing lengthy agent interaction records into (scenario, risk, behavior) triples is essentially a dimensionality reduction over information — discarding details irrelevant to safety evaluation while retaining the most critical semantic dimensions. This technique not only improves retrieval quality but is highly transferable: any RAG system that needs to perform "topic matching" over long texts can benefit from this "extract structured features first, then retrieve on features" two-stage strategy.

- **The scalability of the training-free design.** Conventional fine-tuning requires retraining whenever new data is added, whereas AgentAuditor only needs to incrementally add new interaction records to the memory bank and re-run FINCH clustering. This allows the system to continuously accumulate evaluation experience with use — analogous to the continuous learning process of a human expert.

## Limitations & Future Work

- **Cold-start problem for the memory bank**: AgentAuditor relies on annotated historical interaction records to construct experiential memory. For entirely new application scenarios or risk types with no prior examples in the memory bank, system performance degrades significantly. Although cross-dataset experiments demonstrate improvement even with imperfect memory matches, the behavior under truly zero-shot OOD conditions (e.g., a novel type of agent attack) remains uncertain. A natural direction is to combine active learning, enabling the system to automatically identify low-confidence cases and request human annotation.

- **Static memory bank limitations**: The current experiential memory is fixed after construction and cannot automatically update based on new patterns encountered during evaluation. A natural improvement is to introduce a dynamic memory mechanism — when the evaluator encounters new cases with high confidence, they are automatically added to the memory bank, triggering incremental clustering updates.

- **Latency overhead from multi-stage retrieval**: Two-stage retrieval combined with longer prompts (containing retrieved CoT) increases inference latency. On QwQ-32B, total inference time is approximately 2.5× that of bare LLM evaluation. For scenarios requiring real-time safety monitoring (e.g., online agent services), this latency may be unacceptable. A potential mitigation is to distill experiential memory into a smaller model.

- **Weak performance on physical-world domains such as autonomous driving**: Domain-level experiments reveal that Autonomous Navigation F1 falls as low as 40%, exposing a fundamental limitation — the current framework is entirely grounded in text-level reasoning, whereas safety evaluation in physical-world settings requires spatial reasoning and physical common sense that lie beyond the scope of purely textual RAG.

- **Strong dependence on underlying LLM capability**: Although AgentAuditor claims robustness to the choice of underlying LLM, the efficiency table shows that on Qwen2.5-7B it achieves only 76.69% F1 — far below QwQ-32B's 95.67%. This indicates that on smaller models the memory augmentation effect is limited: the base model must possess sufficiently strong in-context learning capability to effectively leverage retrieved reasoning experiences.

- **Transparency of feature extraction weights**: Heuristic weights determined via grid search on R-Judge are fixed for all benchmarks. Although experiments demonstrate acceptable generalization, the relative importance of the scenario, risk, and behavior dimensions may genuinely differ across domains. A natural improvement direction is adaptive weighting — dynamically adjusting dimension weights based on the characteristics of the case under evaluation.

## Related Work & Insights

- **vs. R-Judge**: R-Judge is the only prior "evaluator evaluation" benchmark, but contains only 569 records and 10 risk types, uses binary labels, does not distinguish safety from security, and has nearly been saturated by state-of-the-art methods (~96% F1). ASSEBench comprehensively surpasses it in scale (4×), risk coverage (15 types), scenario coverage (29 scenarios), ambiguity handling (Strict/Lenient), and safety–security separation.

- **vs. Agent-as-a-Judge (ICML 2025)**: Agent-as-a-Judge uses an agent to evaluate another agent, focusing primarily on task completion capability assessment. On safety and security evaluation tasks, AgentAuditor leads by a large margin (94.18% vs. 83.85% F1), because safety evaluation requires not "running the task again" but "reasoning judgment grounded in rich experience."

- **vs. ShieldLM**: ShieldLM improves LLM safety evaluation capability through fine-tuning. AgentAuditor's training-free approach is more flexible — no large training datasets or GPU training resources are required, and adaptation to new scenarios is rapid. The trade-off is a higher requirement on base model capability.

- **vs. ToolEmu**: ToolEmu focuses on agent safety in tool-use scenarios but limits its evaluation dimension to safety without covering security. AgentAuditor and ASSEBench contribute by incorporating the security dimension into the agent evaluation framework for the first time.

- **Broader implications of memory-augmented reasoning**: AgentAuditor's "structured memory + multi-stage RAG" paradigm can transfer to any LLM-as-Judge task requiring "complex judgment based on experience" — code security auditing, legal compliance checking, content moderation, etc. The core insight is: do not let the LLM reason from scratch; instead, provide it with the experience of having "seen similar cases."

## Rating

- **Novelty**: ⭐⭐⭐⭐ Core components (RAG, CoT, clustering) are established techniques, but their combination into an "experience memory-augmented evaluation" paradigm applied to agent safety evaluation is novel
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Ablation experiments are exceptionally comprehensive — clustering comparisons, retrieval relevance ablations, cross-dataset transfer, three-level adversarial attacks, domain-level breakdowns, and time cost analysis, covering virtually every angle
- **Writing Quality**: ⭐⭐⭐⭐ The paper structure is clear, but some important content (efficiency analysis, limitations) is placed in appendices, requiring readers to consult supplementary material
- **Value**: ⭐⭐⭐⭐⭐ ASSEBench fills a critical gap in agent safety evaluation benchmarks, and AgentAuditor's training-free paradigm has direct value for industrial deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CORE: Full-Path Evaluation of LLM Agents Beyond Final State](core_full-path_evaluation_of_llm_agents_beyond_final_state.md)
- [\[NeurIPS 2025\] AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents](agentdam_privacy_leakage_evaluation_for_autonomous_web_agent.md)
- [\[NeurIPS 2025\] AgentChangeBench: A Multi-Dimensional Evaluation Framework for Goal-Shift Robustness](agentchangebench_a_multi-dimensional_evaluation_framework_for_goal-shift_robustn.md)
- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](../../ACL2026/llm_agent/hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)

</div>

<!-- RELATED:END -->
