---
title: >-
  [Paper Note] Prompts for Public-Sector LLMs Should Be Governed as Commons
description: >-
  [ICML 2026][Recommender Systems][Prompt governance] This position paper argues that LLM prompt templates used by the public sector should be versioned, attributed, auditable…
tags:
  - "ICML 2026"
  - "Recommender Systems"
  - "Prompt governance"
  - "public sector"
  - "commons"
  - "urban AI"
  - "pluralistic value aggregation"
date: 2026-05-08
content_hash: e73925cfb376d450
---

# Prompts for Public-Sector LLMs Should Be Governed as Commons

**Conference**: ICML 2026  
**arXiv**: [2606.00873](https://arxiv.org/abs/2606.00873)  
**Code**: None (position paper + pilot dataset)  
**Area**: AI Governance / Position Paper  
**Keywords**: Prompt governance, public sector, commons, urban AI, pluralistic value aggregation

## TL;DR
This position paper argues that LLM prompt templates used by the public sector should be versioned, attributed, auditable, and vetoable like open-source commons. The authors conducted a pilot benchmark using 443 community prompts from a North American city (augmented to 3,317) across five governance states, providing three falsifiable predictions: governed prompts shift output distributions, improve auditability, and reduce time-to-remediation.

## Background & Motivation

**Background**: The public sector is utilizing LLMs to draft official documents, summarize records, triage citizen requests, and prepare public engagement materials. Existing governance tools like Model Cards, Datasheets, RLHF/Constitutional AI, and platform policies do not cover the "actual prompt templates used during local deployment."

**Limitations of Prior Work**: In practice, prompt templates often circulate **informally** among teams, contractors, and vendors without undergoing policy review. However, prompts encode roles, audiences, and value trade-offs; the same model with the same input can yield significantly different outputs if the prompt is changed. Once a prompt becomes a default template, its embedded preferences may be mistaken for "model conclusions" or "policy conclusions."

**Key Challenge**: Accountability is fragmented into three segments: model providers manage weights and system policies, integrators manage prompts and workflows, and public agencies bear the consequences of outputs but lack an **audit trail**. The prompt layer is the true "configuration layer," yet it lacks any governance primitives.

**Goal**: To treat prompts as a category of **governed artefacts**. This paper proposes a set of governance primitives (versioning, provenance, licensing, vetoing, quotas, appeals) implementable on Git-like repositories and demonstrates through a falsifiable pilot that these governance states change observable output distributions and operational metrics.

**Key Insight**: Borrowing from Ostrom’s commons governance theory (mapping boundaries, monitoring, and conflict resolution to repository workflows) and open-source community experiences (licenses, PRs, issues, CHANGELOGs), "prompts" are transformed into a community-maintained commons.

**Core Idea**: **Prompt Commons** = A repository of versioned prompt templates with provenance, licenses, and auditable change logs + three levels of governance states (open / curated / veto-enabled) + a "negotiation-oriented aggregation prompt" that makes conflicts explicit.

## Method

### Overall Architecture
Prompt Commons is not a new model but a **governance protocol + repository structure + evaluation protocol**. Each prompt in the repository is a plain-text entry with metadata (author groups, location, value propositions, consent levels, change logs). Changes occur via issues/PRs to leave timestamps and justifications. There are three enumerable governance states. Negotiation-oriented aggregation treats multiple prompts from different stakeholders as "proposals," using a versioned "aggregation prompt" to direct the model to identify consensus, list disagreements, and propose compromises. Evaluation fixes the model, temperature, top-$p$, decoding length, and scoring scale to compare five methods (M0–M4). The design aims to turn prompts from one-off inputs into **published artefacts** that can be cited, audited for compliance, and rolled back.

### Key Designs

1.  **Three Governance States + Enforceable Repository Rules**:
    - **Function**: Maps "who can modify prompts, what can be published, and how to stop problematic ones" to enforceable Git repository rules.
    - **Mechanism**: **Open state**—any authenticated user can propose; maintainers only clear spam and basic safety risks. **Curated state**—merges require maintainer review, mandatory provenance fields, and releases must meet coverage quotas across groups/regions via checklists. **Veto-enabled state**—adds a formal quarantine mechanism atop curated states, where designated representative organizations can issue "veto records" to temporarily delist prompts for mandatory reconsideration (accept / modify / reject). Each level corresponds to specific PR templates and CI checks.
    - **Design Motivation**: Commons governance often fails when rules are only documented but not enforced; embedding governance into the repository workflow makes it a daily process rather than an afterthought.

2.  **Negotiation-oriented prompt aggregation**:
    - **Function**: When multiple stakeholders submit prompts with diverging value propositions (accessibility, safety, cost, climate resilience, procedural fairness), a "meta-prompt" is used to make the model explicitly identify consensus and explain value tensions.
    - **Mechanism**: The aggregation rule itself is treated as a **versioned and audited** artefact rather than an implicit ensemble, aligning with the idea from social choice theory that "aggregation rules shape outcomes." In the pilot, this is M4: stratified sampling of $k=6$ prompts balanced by author groups + a fixed aggregation instruction.
    - **Design Motivation**: Traditional methods either cram differences into one prompt (majority rule) or use hidden ensembles. Writing "how to handle disagreement" as an auditable prompt allows affected groups to verify if the aggregation smoothed over minority concerns.

3.  **Falsifiable Evaluation Protocol (5 Methods × 3 Metric Categories)**:
    - **Function**: Turns the question of "whether governance actually works" into a proposition refutable by experimental data.
    - **Mechanism**: Fixed an instruction-tuned chat LLM (temperature 0, top-$p$ 1, max 256 tokens) and $N=50$ "contested-choice" scenarios involving urban street trade-offs. Labels involve three options (vehicle priority / active transport & accessibility priority / mixed or compromise). Methods compared: M0 (single-author), M1 (random open commons), M2 (curated coverage sampling), M3 (curated + veto), M4 (negotiation aggregation). Metrics: output distribution (compromise rate, commitment $D=1-p_{\text{mixed}}$), subjective acceptability (7-point scale by 12 raters), and operational time-to-remediation (50 synthetic incidents).
    - **Design Motivation**: Position paper arguments must be "falsifiable." The authors state three predictions (P1/P2/P3) and declare that if governed versions show no difference or perform worse than single-author versions, the paper's stance is refuted.

### Loss & Training
This paper does not train a model. The governance protocol itself is the "training"—iterating the prompt collection through community processes (issue/PR/veto). All numerical values (prompt length, lexical entropy, token frequency, acceptability, remediation latency) are descriptive statistics, **not optimization targets**.

## Key Experimental Results

### Main Results
The pilot used 443 human-written prompts, augmented to 3,317 via deduplication, value-preserving paraphrasing, and scenario expansion. Human prompts averaged 22.6 words (median 19), while augmented ones averaged 31.7 words, with lexical entropy increasing from 7.53 bits to 8.39 bits. The table compares the "compromise rate" and commitment $D=1-p_{\text{mixed}}$:

| Method | Compromise Rate (%) | $D$ | Description |
| :--- | :--- | :--- | :--- |
| M0 Single-author | 24 | 0.76 | Clear but narrow path |
| M1 Open commons | 48–52 | 0.48–0.52 | Significant increase in compromise |
| M2 Curated commons | 48–52 | 0.48–0.52 | Same as M1, but more balanced coverage |
| M3 Curated + veto | 48–52 | 0.48–0.52 | Same as M2, with controlled withdrawal |
| M4 Negotiation aggregation | — | 0.49 | Compromise after explicit conflict identification |

### Ablation Study
The authors also provide subjective acceptability and operational latency:

| Governance State | Avg Acceptability (7-pt) | Gini (Dispersion) | Avg Remediation Latency |
| :--- | :--- | :--- | :--- |
| M0 Single-author | $4.35\pm0.86$ | 0.096 | — |
| M2 Curated | $4.92\pm0.44$ | 0.043 | $11.8$ h |
| M3 Curated + veto | $5.48\pm0.66$ | — | $5.6$ h |
| Open (Reference only) | — | — | $30.5$ h |

### Key Findings
- Governance states **change output distributions**: Moving from single-author to commons, the compromise rate jumps from 24% to ~50%, with higher cross-group acceptability and lower dispersion. This supports P1.
- Governance processes significantly **reduce remediation latency**: Moving from open to curated to veto states reduced latency from 30.5 h to 5.6 h (based on synthetic incident traffic). This supports P3.
- Compromise rate is not an end in itself—excessive compromise in emergency tasks can be harmful. The authors emphasize that "metrics change with task patterns," avoiding treating descriptive statistics as normative goals.

## Highlights & Insights
- Redefining prompts from engineering optimization objects to a **governance surface** is the core perspective shift of this position paper. This allows the application of social choice, commons theory, and open-source governance frameworks.
- "Negotiation-oriented prompt aggregation" transforms ensembles from hidden methods into auditable artefacts—a technique applicable to any multi-stakeholder system.
- Explicitly stating three falsifiable predictions (P1/P2/P3) and defining conditions for refutation is a highly commendable standard for ML position papers.

## Limitations & Future Work
- The pilot used only 1 API model, $N=50$ scenarios, 12 raters, and one city’s recruitment pool; external validity is limited. The authors admit this is a "minimal falsifiable testbed," not a universal effect estimate.
- Incident response latency uses synthetic arrival logs rather than real institutional response times.
- Prompt Commons could be "captured" by resource-rich stakeholders; community legitimacy might be superficial. Provenance and quotas are used to mitigate this, but the dynamics are not quantitatively evaluated.
- Transparent prompt disclosure increases the attack surface (prompt injection, jailbreak), requiring tiered access control, which is discussed as a principle but not a detailed solution.

## Related Work & Insights
- **vs Model Cards / Datasheets (Mitchell 2019; Gebru 2021)**: Those govern "models/datasets"; this governs "deployed prompt sets." They are orthogonal and complementary.
- **vs RLHF / Constitutional AI (Christiano 2017; Ouyang 2022; Bai 2022)**: Those constrain global model behavior at training; this constrains local frameworks at deployment. Models cannot exhaust local value trade-offs.
- **vs OWASP Top 10 for LLM Applications**: This paper elevates quarantine and rollback to governance primitives, aligning with security engineering incident response.
- **vs Social choice for AI alignment (Conitzer 2024; Huang 2025)**: This paper brings social choice down to the auditable "aggregation prompt" level, allowing pluralistic value alignment to become operationally viable.

## Rating
- Novelty: ⭐⭐⭐⭐ Significant original claim in framing prompt governance as commons.
- Experimental Thoroughness: ⭐⭐⭐ Pilot is reproducible but limited in external validity.
- Writing Quality: ⭐⭐⭐⭐ Very well-structured Argument-Counterargument-Falsification framework.
- Value: ⭐⭐⭐⭐ Provides immediately implementable governance primitives for public sectors procuring LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] NeurIPS Should Lead Scientific Consensus on AI Policy](../../NeurIPS2025/recommender/neurips_should_lead_scientific_consensus_on_ai_policy.md)
- [\[ICLR 2026\] Search Arena: Analyzing Search-Augmented LLMs](../../ICLR2026/recommender/search_arena_analyzing_search-augmented_llms.md)
- [\[ACL 2026\] Personalizing LLMs with Binary Feedback: A Preference-Corrected Optimization Framework](../../ACL2026/recommender/personalizing_llms_with_binary_feedback_a_preference-corrected_optimization_fram.md)
- [\[ACL 2026\] Bridging Language and Items for Retrieval and Recommendation: Benchmarking LLMs as Semantic Encoders](../../ACL2026/recommender/bridging_language_and_items_for_retrieval_and_recommendation_benchmarking_llms_a.md)
- [\[AAAI 2026\] Evaluating LLMs for Police Decision-Making: A Framework Based on Police Action Scenarios](../../AAAI2026/recommender/evaluating_llms_for_police_decision-making_a_framework_based_on_police_action_sc.md)

</div>

<!-- RELATED:END -->
