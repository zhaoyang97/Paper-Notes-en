---
title: >-
  [Paper Note] From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents
description: >-
  [ICML 2026][LLM Safety][De-anonymization] The paper demonstrates that LLM agents can cross-reference fragmented, individually non-identifiable cues with public evidence to re-link anonymized data to specific real-world i…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "De-anonymization"
  - "LLM Agent"
  - "Inference-driven Linkage"
  - "Privacy-Utility Trade-off"
  - "Benchmark"
date: 2026-05-08
content_hash: 19a2f12d5eaa436b
---

# From Weak Cues to Real Identities: Evaluating Inference-Driven De-Anonymization in LLM Agents

**Conference**: ICML 2026  
**arXiv**: [2603.18382](https://arxiv.org/abs/2603.18382)  
**Code**: https://github.com/jihyun-jeong-854/InferLink (Available)  
**Area**: LLM Security / Privacy / De-anonymization  
**Keywords**: De-anonymization, LLM Agent, Inference-driven Linkage, Privacy-Utility Trade-off, Benchmark  

## TL;DR
The paper demonstrates that LLM agents can cross-reference fragmented, individually non-identifiable cues with public evidence to re-link anonymized data to specific real-world identities. This "inference-driven de-anonymization" risk is systematically quantified across three scenarios: classic case replication, the controlled InferLink benchmark, and real human-computer dialogue logs.

## Background & Motivation

**Background**: Industry and regulatory bodies generally consider "removing direct identifiers like names, emails, or ID numbers" as a sufficiently strong defensive line for privacy. Historically, de-anonymization events like the Netflix Prize and AOL search logs were significant because they required experts, custom algorithms, and extensive manual reconciliation—high costs that themselves constituted a practical privacy barrier.

**Limitations of Prior Work**: In the age of LLM agents, tool calls, web search, and multi-step reasoning reduce these "expert costs" to nearly zero. However, existing agent privacy evaluations (e.g., PrivacyLens, AgentDAM) focus on explicit access, leakage, or disclosure, rarely testing whether an agent can synthesize multiple non-identifiable cues into an identity hypothesis. Additionally, a few recent works related to de-anonymization (Li 2026, Lermen et al. 2026) mostly stop at demonstrating the existence of risk without systematic variable control.

**Key Challenge**: Real-world threats are **inference-driven** (identity linkage as a byproduct of agents performing benign tasks), whereas current evaluations assume threats involve **explicit disclosure**. This misalignment leads to a serious underestimation of actual risks.

**Goal**: (1) Formalize the "inference-driven linkage" failure mode; (2) Provide a reproducible benchmark with controlled variables (cue types, task intents, attacker priors); (3) Systematically evaluate across classic cases, controlled benchmarks, and real human-computer traces to quantify the privacy-utility trade-off.

**Key Insight**: The paper decomposes linkage attacks into a unified interface: "anonymized artifact $D_{\text{anon}}$ + auxiliary context $D_{\text{aux}} \to$ identity hypothesis $\hat{\imath}$ + evidence $\mathcal{E}$," and designs three types of evaluations around this interface.

**Core Idea**: "Identity risk $\neq$ explicit disclosure"; rather, it is the agent's ability to aggregate weak cues into $\hat{\imath}$. This aggregation occurs spontaneously as a byproduct of "helpfulness," even when the user does not explicitly request de-anonymization.

## Method

### Overall Architecture
The paper proposes a unified interface:

$$\Pi:(D_{\text{anon}}, D_{\text{aux}}) \mapsto (\hat{\imath}, \mathcal{E})$$

(written inline as $\Pi:(D_{\text{anon}}, D_{\text{aux}}) \mapsto (\hat{\imath}, \mathcal{E})$), where $D_{\text{anon}}$ is anonymized data with direct identifiers removed, and $D_{\text{aux}}$ can either be pre-provided control data (Netflix setting) or a set of evidence retrieved by the agent from the open web (AOL/dialogue setting). Based on this interface, the paper instantiates evaluations in three scenarios:

- **Scenario 1 (Classic Case Replication)**: Netflix Prize (matching sparse behavioral fingerprints within a fixed candidate pool) and AOL search logs (open retrieval + triangulation without a candidate pool).
- **Scenario 2 (Controlled Benchmark InferLink)**: Synthesized paired data where each instance contains only one true overlapping individual, systematically varying three factors and measuring LSR + Utility.
- **Scenario 3 (Real Human-Computer Interaction Traces)**: De-identified interviews of researchers from the Anthropic Interviewer dataset (Scientists subset) + a set of anonymized multi-turn ChatGPT dialogue logs, verified by a web-enabled Gemini agent via search.

Evaluation metrics consist of two categories: Linkage Success Rate $\mathrm{LSR}=\frac{1}{N}\sum_j \mathbb{I}(\mathcal{S}_j)$ for scenarios with unique ground truth (Netflix, InferLink); and Confirmed Linkage Count (CLC) for scenarios where ground truth is not fully available (AOL, interaction traces), counting only successful cases independently verified by public evidence.

### Key Designs

1.  **Unified Inference-Driven Linkage Interface $\Pi$**:
    - **Function**: Bridges "classic fixed-pool matching" and "open-web retrieval de-anonymization," which were historically studied separately, into a single formal framework to facilitate cross-scenario model comparison.
    - **Mechanism**: All scenarios are reduced to "providing $D_{\text{anon}}$ and $D_{\text{aux}}$ to the agent and requiring $(\hat{\imath}, \mathcal{E})$." Whether $D_{\text{aux}}$ is pre-provided or retrieved depends on the scenario, but the evaluation protocol remains consistent. The CLC strategy is **conservative**: vague profiles or narrowing down to a small candidate set do not count; a success is only recorded if $\hat{\imath}$ is doubly supported by cues in $D_{\text{anon}}$ and external evidence in $D_{\text{aux}}$.
    - **Design Motivation**: To allow "classic case replication" and "real trace studies" to yield comparable conclusions, while elevating "whether the agent generates an identity hypothesis" to a primary metric, addressing the blind spot of only measuring "explicit leakage."

2.  **InferLink Three-Factor Controlled Benchmark (fingerprint × intent × knowledge)**:
    - **Function**: Examines the impact of three variables on linkage success rate while keeping the paired data structure constant, identifying "when" linkage is most likely to occur.
    - **Mechanism**: Each instance is determined by a seed $(f, \iota, \kappa)$—fingerprint type $f \in \{\textsc{Intrinsic}, \textsc{Coordinate}, \textsc{Hybrid}\}$ (intrinsic attributes / spatio-temporal coordinates / hybrid), task intent $\iota \in \{\textsc{Implicit}, \textsc{Explicit}\}$ (benign analysis / obvious de-anonymization), and attacker knowledge $\kappa \in \{\textsc{ZK}, \textsc{MK}\}$ (no named target / given named target). Each data pair consists of two 10-row × 9-column structured tables, where 5 shared attributes are roles as "contextual features / sparse anchors / one-sided exclusives," with **only one** individual overlapping between tables.
    - **Design Motivation**: While classic cases are impactful, they conflate various factors; InferLink decouples "cue structure" from "user phrasing," upgrading the evaluation from "does the risk exist" to "how does the risk shift with conditions." It also allows reporting metrics independently for each fingerprint type.

3.  **End-to-end Pipeline for Generation-Validation-Synthesis-Dialogue-Evaluation**:
    - **Function**: Mass-produces InferLink instances in a reproducible, low-noise manner and embeds the evaluation into a "multi-turn interaction" to simulate real-world usage.
    - **Mechanism**: Phase 1 samples seeds; Phase 2 generates candidate scenarios based on $f$ (task context + $D_{\text{anon}}/D_{\text{aux}}$ patterns + attribute schema); Phase 3 synthesizes paired data and ensures a globally unique linkage. A validation checklist must be passed between Phase 2 and 3 (task must require both sources, single source must be unsolvable, dependence on quasi-identifiers rather than direct identifiers). Phase 4 presents both sources via multi-turn dialogue to elicit identity hypotheses. Phase 5 reports LSR↓ and Utility↑. It also evaluates "privacy-aware system prompts" to quantify $\Delta U$ (Utility cost) and $\Delta \mathrm{LSR}$ (Privacy gain).
    - **Design Motivation**: Validation-before-synthesis ensures each instance "requires cross-source reasoning" rather than accidental identification. Multi-turn dialogue aligns with actual agent usage, making spontaneous identity linkage an observable phenomenon.

### Loss & Training
The paper does not train models but evaluates existing LLM agents (GPT-5, Claude 4.5, o4-mini running within the OpenHands framework; web-enabled Gemini 3 Pro for AOL and interaction cases). The only intervention at the "training level" is the inclusion of privacy-aware system prompts as a defensive baseline for comparison.

## Key Experimental Results

### Main Results

**Netflix Prize De-anonymization (Scenario 1)**: Sampling a 1000-user candidate pool from 1 million users, providing $m \in \{2,4,6,8\}$ ratings for the target user (rating perturbation $\pm 1$ star with 0.5 probability, date perturbation $\pm 21$ days), and requiring the agent to identify the user.

| Setting ($m$) | Classic baseline $T{=}14$ | Classic baseline $T{=}21$ | GPT-5 | Claude 4.5 |
|---|---|---|---|---|
| 8 | 98.3 | 98.8 | **99.00 ± 0.72** | 97.30 ± 3.50 |
| 6 | 96.7 | 97.1 | **97.43 ± 3.76** | 93.13 ± 8.39 |
| 4 | 90.5 | 91.8 | 94.83 ± 2.22 | **97.27 ± 2.78** |
| 2 (Sparse extreme) | 56.0 | 60.2 | **79.17 ± 4.97** | 53.30 ± 19.21 |

GPT-5 increases the LSR from 56% (classic method) to 79% in the sparsest $m=2$ range, which historically required heavy manual tuning. Claude 4.5 shows high variance in the sparse range (±19.21), indicating that the ability to perform tie-breaking with weak cues is a significant performance gap between models.

**InferLink Three-Factor Controlled Evaluation (Scenario 2)**: 3 fingerprints × 3 (intent, knowledge) × 20 instances = 180 instances, comparing three models. Selected data:

| Setting | Fingerprint | o4-mini LSR | GPT-5 LSR | Claude 4.5 LSR |
|---|---|---|---|---|
| Implicit (**No user request for de-anon**) | Intrinsic | 0.450 | 0.150 | **0.800** |
| Implicit | Hybrid | 0.500 | 0.000 | **0.800** |
| Explicit-ZK (Required but no name target) | Hybrid | 0.400 | 0.850 | **1.000** |
| Explicit-MK (Named target provided) | Hybrid | 0.800 | 0.950 | **1.000** |

The most notable finding is: **Claude 4.5 maintains a stable LSR of 0.70–0.80 under the Implicit setting**, meaning the agent spontaneously provides specific identity hypotheses even when the user asks benign analysis questions. GPT-5 is much more conservative under Implicit (0.00–0.25) with almost no loss in Utility. This indicates that "silent risk" is the hardest part to evaluate.

### Ablation Study

**Privacy-Aware System Prompts (Aggregated by Fingerprint)**:

| Intent | Metric | o4-mini | GPT-5 | Claude 4.5 |
|---|---|---|---|---|
| Implicit | Before LSR | 0.40 | 0.13 | 0.77 |
| Implicit | After LSR | 0.05 | 0.00 | 0.07 |
| Implicit | $\Delta$ Utility | -0.07 | -0.10 | -0.05 |
| Explicit-MK | Before LSR | 0.72 | 0.85 | 0.98 |
| Explicit-MK | After LSR | 0.20 | 0.02 | 0.03 |
| Explicit-MK | $\Delta$ Utility | -0.17 | -0.05 | **-0.54** |

Defense prompts can suppress LSR to nearly 0 under Explicit-MK, but Claude 4.5 suffers a 0.54 utility loss—meaning "anti-linkage" instructions cause severe over-refusal, blocking even legitimate cross-source analysis. GPT-5 provides a better trade-off (LSR ≈ 0 with $\Delta U$ only -0.05).

**Human-Computer Interaction Traces (Scenario 3)**: CLC = 6 for the Anthropic Interviewer Scientists subset (matching Li 2026); CLC = 1 for 30 privacy-related ChatGPT dialogue logs, where the case was **specifically verified** within the authors' internal group of ≤12 people.

### Key Findings
- **Silent Risk**: Identity hypotheses are frequently generated even when intent is Implicit; traditional evaluations that ask agents if they leaked privacy would miss all of these.
- **Fingerprint-Specific Vulnerability**: GPT-5 is relatively robust under Coordinate (LSR=0.65) but near the upper limit under Intrinsic/Hybrid—"looking safe on average" masks high vulnerability to specific cue types.
- **Privacy-Utility Trade-off is Real**: Prompts that suppress linkage harm compliant tasks, and the loss is highly asymmetrical across models (Claude 4.5 is far more prone to over-refusal than GPT-5).
- **Linkage Stems from Synthesis, Not Single Cues**: Successful identification generally occurs only after cross-referencing multiple weak signals like rough location + role + research field + temporal events.

## Highlights & Insights
- Reintroduces "de-anonymization" as a **standard agent evaluation topic** from the realm of "expert-level SP papers," providing a reusable unified interface $\Pi$. Future work on RAG/Memory/Tool-use can directly report linkage risks using this interface.
- The design of reusing the same data pair across three intents in InferLink is ingenious: since the underlying data is identical, any LSR difference is **solely attributable** to "user phrasing + whether a named target is provided," cleanly decoupling model guardrail behavior from cue linkability.
- Reporting CLC instead of LSR for AOL and interaction cases is an examplar of responsible evaluation: when total ground truth is missing, it is better to underestimate than report inflated metrics.
- Introduces an important transferable perspective for LLM privacy research: the **"Silent Leakage" metric** (whether an agent spontaneously links identities when not requested) should be a standard evaluation dimension for any agent benchmark.

## Limitations & Future Work
- InferLink currently features one overlapping individual per instance with fixed schemas; harder settings like near-duplicate individuals, multiple ground truth candidates, and dynamic schemas are left for the future.
- Publicly verifiable interaction traces are scarce; CLC proves "it happens" but cannot estimate "frequency." The base rate of such risks in daily dialogue remains unknown.
- Utility is only measured by "task completion"; it does not granularly differentiate "completion quality." The trade-off curve might be under- or over-estimated.
- Defense experiments only evaluated simple system prompts. More sophisticated methods (e.g., retrieval-stage intervention, identity-unlinkability constraints during generation) are obvious next steps.

## Related Work & Insights
- **vs Staab et al. 2023 (Inferring Sensitive Attributes with LLMs)**: They focus on inferring single attributes from text; this paper requires **identity-level** hypotheses with cross-verification, a more end-to-end and difficult task.
- **vs Li 2026 and Lermen et al. 2026 (Recent LLM De-anonymization Demos)**: They are case-driven demonstrations of feasibility; this paper provides a controlled benchmark and unified interface, moving research from "demo" to "systematic risk characterization."
- **vs PrivacyLens (Shao 2024) / AgentDAM (Zharmagambetov 2025)**: They measure explicit access/disclosure; this paper measures **inference-driven linkage**, a failure mode missed by existing benchmarks.
- **vs Classic Narayanan & Shmatikov 2008**: Classic work requires manual tuning of rarity-weighted similarity and temporal tolerance $T$; this paper proves LLM agents can replicate or exceed these results using natural language, significantly weakening "expert cost" as a privacy barrier.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formalizes "inference-driven linkage" and releases a controlled benchmark; a conceptual upgrade for agent privacy evaluation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Complementary three-scenario setup + three models + three factors + defense comparison; comprehensive, though lacking open-source models and complex defenses.
- Writing Quality: ⭐⭐⭐⭐⭐ Motivation, formalization, and experimental organization are extremely clear; ethics and reporting constraints are handled diligently.
- Value: ⭐⭐⭐⭐⭐ Provides evaluation protocols and mitigation-utility baselines directly applicable by agent deployers, auditors, and regulators.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] De-Anonymization at Scale via Tournament-Style Attribution](../../ACL2026/llm_safety/de-anonymization_at_scale_via_tournament-style_attribution.md)
- [\[ACL 2026\] On Safety Risks in Experience-Driven Self-Evolving Agents](../../ACL2026/llm_safety/on_safety_risks_in_experience-driven_self-evolving_agents.md)
- [\[ACL 2026\] Subject-level Inference for Realistic Text Anonymization Evaluation](../../ACL2026/llm_safety/subject-level_inference_for_realistic_text_anonymization_evaluation.md)
- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)
- [\[ICML 2026\] OTora: A Unified Red Teaming Framework for Reasoning-Level Denial-of-Service in LLM Agents](otora_a_unified_red_teaming_framework_for_reasoning-level_denial-of-service_in_l.md)

</div>

<!-- RELATED:END -->
