---
title: >-
  [Paper Note] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models
description: >-
  [ACL 2026][Multi-Agent][agentic evaluation] AgenticEval redefines LLM safety evaluation as a "continuous, self-evolving red-teaming process": a Specialist decomposes unstructured regulatory texts into an atomic rule know…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "agentic evaluation"
  - "regulation-grounded"
  - "self-evolving red-teaming"
  - "EU AI Act"
date: 2026-05-08
content_hash: 5fd1b1460120936a
---

# AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2509.26100](https://arxiv.org/abs/2509.26100)  
**Code**: None  
**Area**: LLM Agent / Safety Evaluation / Regulatory Alignment  
**Keywords**: agentic evaluation, regulation-grounded, self-evolving red-teaming, multi-agent, EU AI Act

## TL;DR
AgenticEval redefines LLM safety evaluation as a "continuous, self-evolving red-teaming process": a Specialist decomposes unstructured regulatory texts into an atomic rule knowledge base; a Generator creates multimodal and multi-format Question Groups around each rule; an Evaluator and an Analyst continuously transform current failures into more potent attack strategies for subsequent rounds. After three iterations, GPT-5's compliance rate with the EU AI Act plummeted from 72.50% to 36.36%, revealing that static benchmarks severely overestimate the safety levels of large models.

## Background & Motivation

**Background**: LLM safety evaluation is dominated by static benchmarks such as HELM, DecodingTrust, and StrongREJECT. These benchmarks provide standardized horizontal comparisons but represent "time-snapshot" manual curations. COMPL-AI operationalizes the EU AI Act into evaluation suites, AutoLaw uses LLM "jurors" to check for legal violations, and AutoDAN-Turbo/AutoRedTeamer/ALI-Agent turn red-teaming into lifelong attack libraries. However, none simultaneously address the shortcomings at the three levels of "regulation-evaluation-evolution."

**Limitations of Prior Work**: (1) **Static Lag**: Benchmarks quickly become obsolete as new attack vectors emerge or model capabilities are updated; (2) **Limited Scope**: They rarely cover complex, multi-dimensional real-world regulations like the EU AI Act, NIST RMF, or MAS FEAT; (3) **Difficulty in Adaptation**: Benchmarks are monolithic, making it hard for enterprises to customize them according to internal policies. Consequently: "Models that appear safe on existing benchmarks may still be vulnerable to new threats and non-compliant with regulations."

**Key Challenge**: High scores on a one-time static test $\neq$ true safety; safety evaluation itself needs to learn and evolve just like the models being tested.

**Goal**: Transform evaluation from a "one-off audit" into a "continuous ecosystem" capable of (1) ingesting any unstructured regulatory text, (2) automatically generating multimodal and multi-format Question Groups, and (3) learning from the failures of the evaluated model to generate more difficult questions.

**Key Insight**: Utilizing a "multi-agent + regulation-grounded" design, four specialized agents are orchestrated into a pipeline: a specialist decomposes regulations, a generator creates questions, a judge renders decisions, and an analyst reflects to guide the next round.

**Core Idea**: "Compliance evaluation should grow dynamically like red-teaming, rather than awarding safety certificates to models using a fixed question bank."

## Method

### Overall Architecture
AgenticEval orchestrates four agents using the MetaGPT framework: the **Specialist** $\mathcal{A}_S$ (GPT-4.1) transforms regulations into a knowledge base; the **Generator** $\mathcal{A}_G$ (Gemini 2.5 Pro) creates questions; the **Evaluator** $\mathcal{A}_E$ (GPT-4.1) renders judgments; and the **Analyst** $\mathcal{A}_A$ (GPT-4.1) performs reflection. The process involves three stages: (1) **Regulation → Knowledge Base**: Using structured or autonomous decomposition modes to break rules into atomic items $r$, each paired with an explanation $e_r$, compliance guidelines $\mathcal{G}_{\text{should}}$, and adversarial guidelines $\mathcal{G}_{\text{should\_not}}$; (2) **Initial Test Suite Generation**: Generating a base-mode anchor for each $r$, then expanding it into a Question Group $\mathcal{Q}_r$ through jailbreak/MCQ/TF/multimodal modes; (3) **Self-Evolving Evaluation Loop**: Running for $K_{\max}=3$ rounds, where in each round the Evaluator determines compliance, and the Analyst synthesizes successes/failures to generate new attack strategies for the Generator to produce harder questions.

### Key Designs

1. **Regulation-Knowledge Base Structuring and Search-Augmented Grounding**:
    - **Function**: Transforms abstract legal clauses into testable "positive description + counter-example pair" knowledge triplets.
    - **Mechanism**: The Specialist supports two modes—User-Guided, where sections are mapped according to a user-provided JSON template, or autonomous recursive decomposition into atomic rules. For each $r$, after obtaining an explanation, web searches are used to pull real cases and public discussions to generate two guidelines: $\mathcal{G}_{\text{should}}$ describes compliant output characteristics (e.g., transparency, optionality, disclosure of sponsorship); $\mathcal{G}_{\text{should\_not}}$ lists specific violation patterns (e.g., dark patterns, political micro-targeting, deepfake impersonation). Generation is forced to perform localized searches based on the document's language/culture to make examples relevant to the actual regulatory environment.
    - **Design Motivation**: If an LLM generates questions directly from raw regulatory text, it often produces "academic abstractions" with low trigger rates. Transforming regulations into "specific behavior-level counter-examples" allows the Generator to create truly deceptive questions and the Evaluator to make interpretable judgments based on clear criteria.

2. **Question Group: Semantic Anchors + Systematic Facet Expansion**:
    - **Function**: A single question is insufficient to expose model boundaries—a group of synonymous questions with different attack facets is used to examine multi-faceted performance on the same rule.
    - **Mechanism**: First, a base mode generates an open-ended question $(q_{\text{base}}, c_{\text{base}})$ as a semantic anchor; then, variants are generated across four facet categories synchronously: (a) **Adversarial Perturbation** (jailbreak mode, persona-play, ethical dilemmas); (b) **Deterministic Probes** (mcq/tf mode, checking declarative knowledge to eliminate ambiguity); (c) **Multimodal Grounding** (multimodal mode, determining visual context first, then generating or searching for an image $I$ to rewrite the question as "unanswerable without the image"). Ultimately, $\mathcal{Q}_r=\{(q_{\text{base}},c_{\text{base}}),(q_{\text{jb}},c_{\text{jb}}),(q_{\text{mcq}},c_{\text{mcq}}),\dots\}$.
    - **Design Motivation**: Models might answer correctly under direct questioning but fail under jailbreak wrapping; MCQ formats verify if the "model truly knows the rules"; multimodality exposes blind spots in text-only alignment. Multiple facets within a group allow for "inconsistency" diagnosis rather than just looking at accuracy.

3. **Self-Evolving Evaluation Loop (Collaborative Evaluator + Analyst)**:
    - **Function**: Accumulates model failure patterns into executable attack strategy instructions, ensuring each round's difficulty targets the previous round's weaknesses.
    - **Mechanism**: The Evaluator judges under a "hierarchical rubric": first by question-level Prime Directive $c$, then by rule-level $\mathcal{G}_{\text{should}}/\mathcal{G}_{\text{should\_not}}$ as a backstop, outputting a binary $y_q$ and natural language reasoning $z_q$. Results are aggregated into successes $R_r^+$ and failures $R_r^-$. The Analyst receives $(R_r^+, R_r^-)$, performs "contrastive analysis" to find root causes (where the model did/did not cross the safety boundary), and synthesizes a new attack strategy $\mathcal{S}_{\text{attack}}$ for the Generator's refined mode. The loop terminates at $K_{\max}=3$, and the Analyst compiles a final report. Example: GPT-5's safety rate was 72.5% in round 1, then jailbreak upgrades occurred in round 2 (e.g., "pretending to research consumer protection requiring deep analysis of dark patterns"), followed by a "normalizing analysis trap" in round 3 ("objectively analyze the effectiveness of these techniques"), resulting in a final 36.36%.
    - **Design Motivation**: Traditional jailbreak libraries are "attack-fix" one-off games; here, the Analyst acts as a red-team lead, internalizing the "shape of the model's safety boundary" to target the next weakness. The Evaluator's Prime-Directive hierarchical rubric ensures the auditability of the LLM judge's decisions, avoiding the subjective bias of open-ended adjudication.

### Loss & Training
AgenticEval does not train any models; the key hyperparameter is $K_{\max}=3$ iterations. Specialist/Evaluator/Analyst use GPT-4.1 (strong analysis), and Generator uses Gemini 2.5 Pro (strong creativity).

## Key Experimental Results

### Main Results
Safety Rates of 11 LLMs across 3 regulatory frameworks (selected final round results):

| Model | NIST AI RMF AVG | EU AI Act AVG | MAS FEAT AVG |
|------|-----------------|---------------|--------------|
| GPT-5 | **78.98** | **67.16** | **67.92** |
| GPT-5-chat-latest | 74.85 | 57.69 | 62.04 |
| Gemini-2.5-pro | 57.23 | 43.93 | 49.11 |
| Gemini-2.5-flash | 60.12 | 50.93 | 51.79 |
| Grok-4 | 53.18 | 35.98 | 46.43 |
| DeepSeek-V3.1 | 52.87 | 45.33 | 47.32 |
| Qwen-3-32B | 48.57 | 38.32 | 43.75 |
| Llama-4-maverick | 54.12 | 35.05 | 34.82 |

On the EU AI Act, GPT-5 performed strongly on PP-RA (91.67%) but weakly on RRBI (44.64%); Llama-4-maverick scored 75% on DPV but only 26.32% on IPI—fine-grained results reflect "highly uneven compliance distribution."

### Ablation Study
Impact of key components (higher final safety rate for GPT-5 after removal indicates evaluation failure):

| Configuration | GPT-5 NIST | GPT-5 EU | Description |
|------|-----------|----------|------|
| Full AgenticEval | 64.29 | 36.36 | Lower safety rate represents more effective evaluation |
| w/o Specialist.Structure | **75.40** | – | Loses atomic rule targeting, coarsening attacks |
| w/o Specialist.Enrich | – | – | Lacks real examples, reducing trigger rates |
| w/o Analyst.Refine | – | **48.60** | Degenerates into static audit, failing to find deep vulnerabilities |
| AutoDAN baseline | – | Sig. Higher | Generic jailbreak, lack of regulatory adaptation |
| PAIR / AutoDAN-Turbo | – | Sig. Higher | Same as above |

### Key Findings
- **Evolutionary effect is real**: The drop in GPT-5's EU AI Act compliance from 72.50% → 36.36% is not noise but deliberate amplification after the Analyst identifies methods like "normalizing analysis traps" and "bait-and-switch."
- **Specialist semantic clustering is effective**: Cosine similarity using the explanation field embeddings shows significantly higher intra-group (rules within the same high-level dimension) than inter-group similarity, proving $\mathcal{A}_S$ truly captures the thematic structure.
- **High Evaluator-human consistency**: Across 100 samples, human evaluation showed Accuracy 88-91%, F1 88-90%, and Cohen's $\kappa$ 0.77-0.81, representing "substantial agreement" and lending credibility to the scoring system.
- **MCQ ablation indirectly proves jailbreaking is core**: Using jailbreak-type baselines alone was far better than prompt engineering, indicating adversarial perturbation is a necessary ingredient.

## Highlights & Insights
- The "Question Group" abstraction, which supports both multi-facet probing and inconsistency diagnosis, is an elegant unit for benchmark design that can be migrated to other domains requiring "multi-angle testing of the same rule."
- The Analyst synthesizes "success + failure examples" into an attack plan rather than simply re-feeding failures; this "contrastive analysis → new attack vector" paradigm essentially models the red-teaming process as controlled generation.
- The intermediate representation of regulation → testable knowledge base (explanations + bidirectional guidelines) decouples evaluation from the limitations of "prompt as benchmark," allowing regulatory documents to be replaced in industrial deployments without changing the pipeline.

## Limitations & Future Work
- The Specialist's regulatory interpretation is "technical understanding" not "legal judgment"—AgenticEval is a pre-audit tool and cannot replace professional compliance certification.
- Evaluation reliability is constrained by the upper bound of the LLM judge: when the Evaluator is weaker than the target model (capability mismatch), it may misjudge subtle adversarial behaviors.
- The self-evolving loop is significantly more computationally expensive than static benchmarks, and $K_{\max}=3$ is an empirical choice lacking a systematic cost-benefit analysis.
- Currently only evaluates open-ended QA; safety evaluation for scenarios like long-range agent action sequences or tool-calling chains is not yet covered.

## Related Work & Insights
- **vs COMPL-AI**: Also targets the EU AI Act, but COMPL-AI uses static mapping; AgenticEval uses dynamic generation and self-evolution to continuously expose new vulnerabilities.
- **vs AutoDAN-Turbo**: The latter is a general jailbreak strategy library lacking regulatory context; AgenticEval targets each specific rule, providing broader and more structured coverage.
- **vs ALI-Agent**: Both use agents to explore long-tail value alignment, but ALI-Agent uses general ethical categories, whereas AgenticEval ingests any unstructured regulatory text, offering higher customizability.

## Rating
- Novelty: ⭐⭐⭐⭐ While individual components (multi-agent / red-team loop / regulation parsing) have precedents, the integration and evolutionary mechanism are the true contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ 11 models × 3 regulations × multi-dimensional segmentation + 4 ablations + Evaluator human consistency verification.
- Writing Quality: ⭐⭐⭐⭐ Case studies (e.g., EU AI Act Article 5(1)(a)) follow the process from legal text to iterative questions, making the pipeline transparent.
- Value: ⭐⭐⭐⭐⭐ Regulatory compliance and continuous evaluation are critical for LLM deployment, offering direct utility for vendors, regulators, and internal auditors.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Large Language Models Miss the Multi-Agent Mark](../../NeurIPS2025/multi_agent/large_language_models_miss_the_multi-agent_mark.md)
- [\[AAAI 2026\] MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models](../../AAAI2026/multi_agent/medla_a_logic-driven_multi-agent_framework_for_complex_medic.md)
- [\[ACL 2026\] Towards Self-Improving Error Diagnosis in Multi-Agent Systems](towards_self-improving_error_diagnosis_in_multi-agent_systems.md)
- [\[ACL 2026\] BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration](bookagent_orchestrating_safety-aware_visual_narratives_via_multi-agent_cognitive.md)
- [\[NeurIPS 2025\] Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?](../../NeurIPS2025/multi_agent/debate_or_vote_which_yields_better_decisions_in_multi-agent_large_language_model.md)

</div>

<!-- RELATED:END -->
