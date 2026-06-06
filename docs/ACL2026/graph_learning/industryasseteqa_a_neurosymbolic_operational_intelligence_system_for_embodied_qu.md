---
title: >-
  [Paper Note] IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance
description: >-
  [ACL 2026][Graph Learning][FMEA Knowledge Graph] This paper reformulates industrial asset maintenance question answering as an "embodied decision-making" task. The authors propose IndustryAssetEQA…
tags:
  - "ACL 2026"
  - "Graph Learning"
  - "FMEA Knowledge Graph"
  - "Embodied QA"
  - "Counterfactual Risk"
  - "Provenance"
  - "Industrial Maintenance"
date: 2026-05-08
content_hash: 0fc7c8aed5edc923
---

# IndustryAssetEQA: A Neurosymbolic Operational Intelligence System for Embodied Question Answering in Industrial Asset Maintenance

**Conference**: ACL 2026  
**arXiv**: [2604.23446](https://arxiv.org/abs/2604.23446)  
**Code**: https://github.com/IBM/AssetOpsBench/tree/IndustryAssetEQA/IndustryAssetEQA (Yes)  
**Area**: Knowledge Graph / Industrial / Neurosymbolic / Embodied QA  
**Keywords**: FMEA Knowledge Graph, Embodied QA, Counterfactual Risk, Provenance, Industrial Maintenance

## TL;DR
This paper reformulates industrial asset maintenance question answering as an "embodied decision-making" task. The authors propose IndustryAssetEQA, a neurosymbolic system comprising episodized telemetry, an FMEA Knowledge Graph, a parameterized counterfactual risk simulator, provenance verification, and a safety gate. Across four industrial datasets, the system improves structural validity, counterfactual direction accuracy, and explanation entailment rates by up to 0.51 / 0.47 / 0.64, respectively, while reducing expert-judged severe over-assertions from 28% to 2%.

## Background & Motivation

**Background**: Modern factories rely on operational intelligence systems to interpret multi-modal telemetry, alarm streams, and maintenance records. The current mainstream approach involves wrapping an LLM over documents and dashboards to provide a natural language interface for answering questions such as "Why did this fail?", "How should it be handled?", or "What happens after intervention?".

**Limitations of Prior Work**: In practical deployments, LLM maintenance assistants exhibit three types of consistency issues: (i) providing generic textbook explanations of faults without grounding them in the sensor readings and context of a specific episode; (ii) lacking verifiable provenance, where answers do not cite the corresponding time windows, sensors, or work order records, making them un-auditable; (iii) counterfactuals and action recommendations providing only qualitative conclusions without explicit risk models, making them unverifiable.

**Key Challenge**: Framing maintenance QA as a "language generation" task ignores its nature as a "sensor + risk" decision-making task. The objective function of the former only requires fluency and alignment with corpus distributions, whereas the latter demands temporal localization, evidence chains, and verifiable intervention effects. This mismatch is the root cause of generalization failure.

**Goal**: To construct an industrial EQA system that simultaneously satisfies four requirements—temporal localization, evidence traceability, risk constraint, and domain knowledge alignment—and to design an evaluation protocol that measures "deployable reliability" rather than "linguistic fluency."

**Key Insight**: The authors observe that the perception $\to$ reasoning $\to$ prediction $\to$ decision-making loop of Embodied AI is structurally isomorphic to the industrial maintenance workflow (sensing telemetry $\to$ diagnosing faults $\to$ planning interventions $\to$ executing maintenance). Therefore, they explicitly model maintenance QA as an embodied decision-making problem. Simultaneously, they utilize neurosymbolic fusion to ensure LLM free-text is dual-anchored by the FMEA Knowledge Graph and an episodic store.

**Core Idea**: By integrating episodized facts, FMEA-KG, a multinomial logistic counterfactual simulator, provenance enforcement, and a safety gate, the LLM is encapsulated into an embodied QA service that can only "answer based on the provided evidence."

## Method

### Overall Architecture

The input is a natural language query entered by an operator at a CMMS frontend. The system first performs question classification across five QA tasks (Description / Temporal / Diagnosis / Counterfactual / Action Recommendation), then proceeds through a pipeline of six modules:

1.  **Fact Extractor**: Slices telemetry time series into fixed windows $[t_f-\Delta, t_f]$ anchored at the failure time $t_f$. For each numerical sensor $s$, it extracts summary descriptors $\mathcal{F}_s = \{\mu_s, \sigma_s, \min_s, \max_s, \text{trend}_s\}$ and concatenates contextual features such as error event counts and hours since last maintenance. Combined with asset/fault/sensor semantics from the FMEA-KG, it outputs an episode JSONL with full provenance.
2.  **Episodic Store**: A dual-table SQLite (facts + features) stores episode-level facts indexed by `fact_id`, supporting efficient retrieval based on asset, label, and threshold conditions $\{f_i \mid x_{i,j} \bowtie \tau\}$.
3.  **FMEA-KG**: A domain knowledge graph constructed from ISO documents via the EMPWR platform, containing 63 fault modes across 9 asset classes. Nodes represent asset classes / sub-components / fault modes / sensor abstractions / maintenance actions, while edges represent `affects / component_of / indicated_by / mitigated_by`. Symbolic constraints are injected into pipeline nodes using rdflib.
4.  **Causal Simulator**: Trains a multinomial logistic regression $P(y \mid \bm{x})$ on episode features to approximate risk. Maintenance interventions are modeled as explicit do-replacements $\bm{x} \mapsto \bm{x}^{\text{do}}$ on the feature vector. It outputs the risk before and after intervention ($r_{\text{before}} = 1 - P(y=\text{healthy} \mid \bm{x})$ and $r_{\text{after}} = 1 - P(y=\text{healthy} \mid \bm{x}^{\text{do}})$), and the direction of change $\Delta r$.
5.  **EQA Builder + Prompt Builder**: Deterministically constructs questions based on the QA type, extracts answer templates, and queries the FMEA-KG to complete failure mode metadata. Evidence blocks (`fact_id`, asset, source, line number, window, key features, FMEA context) are rendered into a prompt with a strict JSON output contract (required fields: `direct_answer / reasoning_answer / provenance / confidence`; counterfactual tasks also mandate `risk_before / risk_after / direction`).
6.  **Verifier + Safety Gate**: Compares the LLM output against the Episodic Store for structural, provenance, and counterfactual direction consistency. If verification fails, the output is flagged and routed for human review.

### Key Designs

1.  **FMEA Knowledge Graph as a Neurosymbolic Fusion Layer**:
    *   **Function**: Encodes domain expert-validated failure modes, sensor signatures, and allowable interventions from ISO documents into a machine-interpretable graph, serving as a hard constraint for the LLM.
    *   **Mechanism**: Nodes and relations cover the complete closed loop of Asset $\to$ Sub-component $\to$ Fault Mode $\to$ Sensor/Action. Failure modes output by the LLM must correspond to nodes in the KG; otherwise, they are failed by the Verifier. 96% of 1004 candidate triples were judged valid by an ISO-style textual entailment model, proving the high quality of the KG.
    *   **Design Motivation**: Pure LLM explanations frequently cite incorrect failure modes or hallucinate sensors. A verifiable symbolic anchor is needed to constrain semantics.

2.  **Episode-centric + Provenance Enforcement**:
    *   **Function**: Anchors all QA tasks to discrete episodes defined by "Specific Asset + Specific Time Window" and forces the LLM to provide verifiable `fact_id / window / referenced sensors` in JSON.
    *   **Mechanism**: The Fact Extractor generates JSONL for each episode containing provenance like source files and time ranges. The Prompt Builder instructs the LLM to "use only provided evidence." The Verifier checks the `fact_id` and feature values cited in the answer against the Episodic Store, rejecting mismatches.
    *   **Design Motivation**: Ablation studies show that removing provenance enforcement causes the Full Pass rate to collapse from 0.89 to 0.19, shifting numerical prediction from deployable to unusable. This is the single most impactful component.

3.  **Parameterized Counterfactual Risk Simulator**:
    *   **Function**: Converts "what-if" questions (e.g., "What if the bearing is replaced?") into explicit replacements of feature vector components, using multinomial logistic regression to output health probabilities and risk direction.
    *   **Mechanism**: A unified $P(y \mid \bm{x})$ supports both diagnosis and counterfactuals. For counterfactuals, $\Delta r$ is calculated by changing the corresponding feature to a new value, paired with a lightweight confidence heuristic based on probability extremes. Simulator outputs are also used by the Safety Gate to filter low-confidence recommendations via thresholds.
    *   **Design Motivation**: A baseline without the simulator achieved only 0.45 counterfactual direction accuracy (near random), indicating that LLMs cannot predict intervention effects via linguistic priors alone. The proxy model increases accuracy to 0.88-0.91.

### Loss & Training

Only the Causal Simulator requires training: it fits a multinomial logistic regression on episode feature vectors $\bm{x}$ to learn $P(y \mid \bm{x})$, where labels $y$ are "healthy" or specific failure mode names, optimized using standard multi-class cross-entropy. Healthy episodes are sampled by selecting a central time $t$ such that no failure occurs in the subsequent $[t, t+H]$, then looking back at $[t-\Delta, t]$. The LLM components use black-box API calls (GPT-4o-mini and Claude Sonnet 4) without any fine-tuning.

## Key Experimental Results

### Main Results

Four industrial datasets were used: Microsoft PdM (rotating machinery, 5716 episodes), C-MAPSS (turbofan engines, 4842), Genesis CPS (cyber-physical systems, 478), and Hydraulic (hydraulic test rig, 2205). Totaling 13k+ episodes with 12k+ Description / 11k Diagnosis / 3k Counterfactual / 3k Action Recommendation QA pairs. Ground truths were expert-verified.

Performance gains on GPT-4o-mini from LLM-only to Full IndustryAssetEQA:

| Configuration | Struct.OK | Prov.OK | Label Cons. | CF Acc. | Entail.Pass | Claim Prec. |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| LLM-only | 0.42 | 0.47 | 0.62 | 0.45 | 0.08 | 0.12 |
| + Episodic | 0.52 | 0.62 | 0.71 | 0.45 | 0.23 | 0.25 |
| + Episodic + KG | 0.52 | 0.62 | 0.73 | 0.45 | 0.56 | 0.51 |
| + Provenance | 0.82 | 0.83 | 0.89 | 0.45 | 0.63 | 0.59 |
| **Full IndustryAssetEQA** | **0.88** | **0.89** | **0.94** | **0.88** | **0.72** | **0.67** |

On Claude Sonnet 4, the Full system achieved Struct.OK 0.90, CF Acc. 0.91, and Entail.Pass 0.78, showing consistent trends. Counterfactual direction accuracy jumped from 0.45 to 0.88-0.91, proving that LLMs cannot resolve this without a simulator.

### Ablation Study

| Configuration | Entail.Pass | Full Pass | CF Acc. |
| :--- | :--- | :--- | :--- |
| Full IndustryAssetEQA | 0.72 | 0.89 | 0.88 |
| w/o Risk simulator | 0.59 | 0.72 | **0.49** |
| w/o Provenance enforcement | 0.42 | **0.19** | 0.81 |
| w/o FMEA-KG | 0.59 | 0.35 | 0.61 |
| w/o Episodic memory | **0.27** | 0.36 | 0.34 |

### Key Findings

*   **Provenance enforcement** provides the largest single-point gain—removing it causes the Full Pass rate to plummet from 0.89 to 0.19, meaning numerical predictions without citation verification are "hallucinated numbers" unfit for deployment.
*   The **Risk simulator** is almost the sole source of counterfactual reasoning: without it, CF Acc. drops from 0.88 to 0.49 (near random), while its impact on Entail.Pass/Full Pass is limited, suggesting linguistic plausibility and causal direction are distinct issues.
*   **Episodic memory** is the foundation: removing it causes all metrics to collapse (Entail.Pass 0.27, Full Pass 0.36, CF Acc. 0.34), showing that episodization is a prerequisite for other modules.
*   **Expert Blind Review** (22 QA pairs): IndustryAssetEQA achieved a 97% answerability rate (vs. 46% for LLM-only), data grounding of 4.5/5 (vs. 3.0/5), and a severe over-assertion rate of 2% vs. 28% (Fleiss $\kappa = 0.63$). McNemar tests showed significance for Description / Diagnosis / Counterfactual ($p < 0.05$), while Action Recommendation was not significant ($p=0.93$), implying that action-related queries require more structural modeling beyond model swaps.

## Highlights & Insights

*   Redefining industrial QA from "language generation" to "embodied decision-making" is a clear paradigm shift. The four desiderata (time-situated / evidence-grounded / risk-constrained / knowledge-grounded) directly address four typical failure modes of LLM-only systems.
*   The combination of provenance enforcement and JSON contracts is highly portable. Any LLM application can mitigate "hallucinated numbers" with minimal engineering cost by requiring models to cite checkable IDs in structured fields followed by a Verifier. This is applicable to legal, medical, and financial document QA.
*   Using multinomial logistic regression as a "lightweight proxy causal model" is pragmatic. The authors acknowledge this is not a true structural causal model (SCM), but compared to LLM guesswork, it improves direction accuracy from 0.45 to 0.88. This provides an insight for RAG and tool-use systems: tools only need to constrain key decision dimensions of the LLM, rather than requiring heavy causal discovery.

## Limitations & Future Work

*   The counterfactual module is a parameterized proxy estimator rather than an identifiable SCM; it outputs surrogate risk rather than provable causal effects, which might reverse direction under distribution shifts or new fault modes.
*   FMEA-KG has good coverage for semantic relations like `description / involves`, but weak structural relations (e.g., `sample / example`) have higher noise (<10% validation rate). It is also domain-level rather than asset-level, lacking coverage for rare fault modes.
*   Using fixed windows for episodes might miss multi-scale or long-term precursors. Adaptive windowing remains unexplored. Evaluation is entirely offline, lacking data on real-world operator latency, reporting rates, or human-review adoption rates.
*   The full pipeline increases engineering and runtime overhead compared to LLM-only approaches. The authors acknowledge that edge deployments or low-latency scenarios require trade-offs between confidence thresholds and retraining frequencies.

## Related Work & Insights

*   **vs. LLM-only Maintenance QA**: Prior works let LLMs read documents and datasets directly. This work enforces an episode + KG + simulator loop, elevating "verifiability" to a system-level hard constraint, improving Prov.OK from 0.47 to 0.89.
*   **vs. Time-series QA (MTBench / FailureSensorIQ, etc.)**: These feed time series to LLMs with increased reasoning budgets. This paper extracts episodic descriptors and risk proxies; its advantage lies in verifiable counterfactuals and actions, while its disadvantage is a dependency on manual FMEA knowledge.
*   **vs. Embodied QA (MindPalace / OpenEQA, etc.)**: While these focus on perception-language alignment in 3D environments, this work ports the framework to industrial assets and temporal signals and requires decisions to pass through explicit risk models rather than purely visual reasoning.

## Rating
*   Novelty: ⭐⭐⭐⭐ Paradigm shift (language generation $\to$ embodied decision) + clear four-element decomposition, though individual components (KG, provenance, proxy models) are not entirely new.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 industrial datasets × 5 QA categories × 2 mainstream LLMs, step-by-step ablation, expert blind review, McNemar significance, and KG triple validation.
*   Writing Quality: ⭐⭐⭐⭐ Fluent motivation and narrative; 5 RQs organize the experiments clearly. Figures are occasionally nested in text.
*   Value: ⭐⭐⭐⭐⭐ Provides a deployable blueprint for industrial LLMs. AssetOpsBench is already integrated within IBM, offering direct lessons for "high-risk + LLM" domains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](../../ICML2026/graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)
- [\[ICLR 2026\] Embodied Agents Meet Personalization: Investigating Challenges and Solutions Through the Lens of Memory Utilization](../../ICLR2026/graph_learning/embodied_agents_meet_personalization_investigating_challenges_and_solutions_thro.md)
- [\[NeurIPS 2025\] ESCA: Contextualizing Embodied Agents via Scene-Graph Generation](../../NeurIPS2025/graph_learning/esca_contextualizing_embodied_agents_via_scene-graph_generation.md)
- [\[ACL 2026\] AgentGL: Towards Agentic Graph Learning with LLMs via Reinforcement Learning](agentgl_towards_agentic_graph_learning_with_llms_via_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[AAAI 2026\] Self-Correction Distillation for Structured Data Question Answering](../../AAAI2026/graph_learning/self-correction_distillation_for_structured_data_question_answering.md)
- [\[ACL 2025\] Agent Steerable Search for Knowledge Graph Question Answering](../../ACL2025/graph_learning/agent_steerable_search_for_knowledge_graph_question_answering.md)
- [\[ICML 2026\] KBQA-R1: Reinforcing Large Language Models for Knowledge Base Question Answering](../../ICML2026/graph_learning/kbqa-r1_reinforcing_large_language_models_for_knowledge_base_question_answering.md)
- [\[ACL 2025\] The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering](../../ACL2025/graph_learning/the_role_of_exploration_modules_in_small_language_models_for_knowledge_graph_que.md)
- [\[ACL 2025\] FiDeLiS: Faithful Reasoning in Large Language Model for Knowledge Graph Question Answering](../../ACL2025/graph_learning/fidelis_faithful_reasoning_in_large_language_model_for_knowledge_graph_question_.md)

</div>

<!-- RELATED:END -->
