---
title: >-
  [Paper Note] CIMemories: A Compositional Benchmark For Contextual Integrity In LLMs
description: >-
  [ICLR 2026][LLM Safety][Contextual Integrity] CIMemories is a compositional benchmark designed to evaluate whether memory-augmented LLMs leak private attributes in inappropriate contexts. By pairing synthetic user personas (100+ attributes each) with dozens of social tasks and labeling each (attribute, task) pair as "appropriate/inappropriate" to disclose, the study reveals that frontier models exhibit up to 69% attribute-level violations. Furthermore…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "Contextual Integrity"
  - "Memory-Augmented LLMs"
  - "Privacy Leakage"
  - "Compositional Benchmark"
  - "Information Flow Control"
date: 2026-05-08
content_hash: cf2a323a94e01b49
---

# CIMemories: A Compositional Benchmark For Contextual Integrity In LLMs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=YnNIp38v1M](https://openreview.net/forum?id=YnNIp38v1M)  
**Code**: https://github.com/facebookresearch/CIMemories  
**Area**: LLM Safety / Privacy Evaluation / Contextual Integrity  
**Keywords**: Contextual Integrity, Memory-Augmented LLMs, Privacy Leakage, Compositional Benchmark, Information Flow Control

## TL;DR
CIMemories is a compositional benchmark designed to evaluate whether memory-augmented LLMs leak private attributes in inappropriate contexts. By pairing synthetic user personas (100+ attributes each) with dozens of social tasks and labeling each (attribute, task) pair as "appropriate/inappropriate" to disclose, the study reveals that frontier models exhibit up to 69% attribute-level violations. Furthermore, reducing violations often comes at the cost of task completeness, and violations accumulate steadily with the number of tasks and sampling iterations.

## Background & Motivation
**Background**: Modern LLM assistants increasingly incorporate "persistent memory"—storing personal information revealed by users in past sessions (e.g., income, medical history, legal issues, relationship status) and prepending it to the prompt in subsequent dialogues for personalization. This "needle in a haystack" memory approach has been deployed in mainstream products like ChatGPT and Meta AI, as assistants are tasked with writing emails, auto-replying, and interfacing with third-party apps on behalf of users.

**Limitations of Prior Work**: Information in memory is "shared across scenarios," yet the essence of privacy is context-dependent. A single attribute (e.g., hypertension) must be disclosed when communicating with a doctor but should never be mentioned to a financial advisor or a landlord. The critical question is: will models inadvertently reveal sensitive attributes from memory that are irrelevant to the current task in inappropriate settings?

**Key Challenge**: Existing contextual privacy benchmarks (e.g., ConfAide, PrivacyLens, CI-Bench, GoldCoin) mostly test simplistic scenarios—typically "one secret + one appropriate piece of information." They fail to capture two compositional features of real-world memory systems: (1) memory is an accumulation of numerous stacked attributes; (2) a single user faces multiple tasks with different recipients. Due to this lack of systematic measurement, it remains unclear where and how severely models fail under compositional complexity. Moreover, determining what "should" be disclosed lacks a single answer, as different users have different privacy perspectives.

**Goal**: To construct a benchmark that characterizes both "memory composition" and "multi-task composition," providing two complementary metrics to measure "violations" and "task completeness." This benchmark is used to evaluate frontier models and address three questions: Do frontier LLMs adhere to contextual integrity? Can scaling or prompting mitigate issues? How do risks evolve as memory accumulates?

**Key Insight**: The authors adopt Nissenbaum’s "Contextual Integrity (CI)" theory—privacy violations are defined as "inappropriate information flows" that violate social norms, rather than the absolute secrecy of information. Consequently, privacy is reframed: given memory $M_s$ and task $t$, which attributes are revealed in the model response $y$, and are these revelations permissible in the current context?

**Core Idea**: By using "(attribute, task) pair labeling + compositional synthetic personas," contextual integrity is transformed into a large-scale, automated measurement. The dual metrics of Violation and Completeness reveal a fundamental trade-off: reducing leakage often necessitates reduced task performance.

## Method

### Overall Architecture
CIMemories is an evaluation protocol and data generation pipeline rather than a new model. It addresses how to measure the contextual integrity of memory-augmented LLMs in a large-scale, controlled, and automated manner. The pipeline consists of: generating user personas with 100+ attributes (each converted into a "memory statement"); preparing social scenarios where "Task = Goal + Recipient"; and using a "gold standard" labeling model to assign "necessary / inappropriate / ambiguous" labels to each (attribute, task) pair. During evaluation, the full memory is prepended to a task prompt and fed to the target model. An LLM judge (the REVEAL function) determines which attributes are present in the response, and Violation and Completeness scores are calculated against the labels.

Formally, an LLM is a stochastic mapping $M:\mathcal X\to\mathcal X$. A user $s$ has memory generated from attribute-value pairs into natural language $M_s = \mathrm{MEM}(\{(a,v_a)\})$. A task $t$ consists of goal and recipient text. The model response $y\sim M(M_s\cdot t)$. The core scoring function is $\mathrm{REVEAL}(y,a)$, which infers the value of attribute $a$ from the response. If it matches the truth, $R(y,a)=1$. Using the ground truth label $G^t_s(a)\in\{0,1\}$ (where 1 = inappropriate), violations are identified.

### Key Designs

**1. Dual Compositionality: Building "Memory Accumulation" and "Multi-Tasking" into the Benchmark**
This is the fundamental difference between CIMemories and prior work. First is **flexible memory composition**: the quantity and quality of attributes in memory are dynamically adjustable. An attribute might be necessary for some tasks but prohibited in others, allowing for a detailed study of how increasing sensitive attributes impacts model reliability. Second is **multi-task composition**: the same user is evaluated across multiple different tasks (recipients). Since each (attribute, task) pair has an independent label, it is possible to measure how violations accumulate over repeated usage.

**2. Violation@n and Completeness: Complementary Metrics**
Relying on a single metric can be deceived by "trivial models": a model that says nothing will have zero violations but zero utility, while a model that reveals everything will have high completeness but maximum violations.
The attribute-level violation uses a "worst-case" approach: for each attribute $a$ that should be hidden in at least some tasks, it checks if the model **ever** reveals it across those tasks $T^{priv}_{s,a}$ over $n$ samples:
$$\mathrm{Violation}@n(s) := \mathbb{E}_{\{a:\,|T^{priv}_{s,a}|>0\}}\Big[\max_{t\in T^{priv}_{s,a}}\ \max_{\{y_1,\dots,y_n\}\sim M(M_s\cdot t)^n}\ R(y,a)\Big].$$
The task-level completeness uses an "average-case" approach: for each task, it calculates the average proportion of "must-share" attributes $A^{share}_{s,t}$ that are successfully revealed:
$$\mathrm{Completeness}(s) := \mathbb{E}_{\{t:\,|A^{share}_{s,t}|>0\}}\Big[\mathbb{E}_{a\sim A^{share}_{s,t}}\ R(y,a)\Big].$$
This asymmetry reflects that a privacy violation is a "single point of failure," whereas task completion is measured by the degree of coverage.

**3. Compositional Synthetic Personas + Task Library**
Personas are generated in two stages: first, the FAKER tool samples basic biographical metadata (name, gender, etc.) for a fictional adult. Then, using this metadata as a seed, an LLM generates "informational attributes"—each describing an aspect of a life "event" (e.g., spouse's affair, promotion) belonging to one of nine sectors (Finance, Health, Legal, etc.). Each persona includes 3 events per sector, with 5 attributes per event. For tasks, 49 target-oriented social scenarios were **manually** curated, each pairing a goal with a specific recipient (e.g., "apply for a loan" + "loan officer").

**4. Consistency Labeling Across Multiple Privacy Personas**
To handle the subjective nature of privacy, the authors use a "gold standard" model (GPT-5) to simulate three privacy personas from the Westin survey: **Privacy Fundamentalist, Pragmatic, and Unconcerned**. For each (attribute, task) pair, labels are sampled 10 times per persona. Crucially, **hard 0/1 labels are only assigned to pairs where all three personas achieve zero-entropy consensus**. All other pairs are treated as "ambiguous" and excluded from scoring. This ensures the benchmark only penalizes clear-cut violations.

### Loss & Training
This work presents a benchmark and evaluation protocol without training new models. Evaluation costs are approximately $100 USD per model for 10 personas. Open-source models are served via vLLM on 8×H200 GPUs, with DeepSeek-R1 used as the judge.

## Key Experimental Results

### Main Results
Evaluation of Violation@5 (lower is better) and Completeness (higher is better) across models:

| Model | Violation@5 ↓ | Completeness ↑ |
|------|------|------|
| GPT-4o | **14.82%** | 43.95% |
| GPT-5 | 25.08% | 56.61% |
| o3 | 38.51% | 55.0% |
| Claude-4 Sonnet | 44.44% | **59.07%** |
| Llama-3.3 70B Instruct | 44.43% | 53.99% |
| Gemini 2.5 Flash | 46.35% | 52.83% |
| Mistral-7B Instruct v0.3 | 56.94% | 46.56% |
| Qwen-3 32B | 69.14% | 57.63% |

**Privacy-Utility Trade-off**: No model succeeds at both. GPT-4o has the lowest violations (14.8%) but also the lowest completeness (43.9%), while Qwen-3 32B achieves high completeness (57.6%) at the cost of extreme violations (69.1%).

### Ablation Study

| Setting | Key Phenomenon |
|------|---------|
| Multi-task Accumulation | Violations rise from 0.1% for 1 task to 9.6% for 40 tasks. With 5 samples, it hits **25.1%**, as different attributes leak in different runs. |
| Model Scaling (Qwen-3) | Gains in completeness and reductions in violations **saturate** quickly; simply increasing parameters does not solve the issue. |
| Reasoning vs. Non-reasoning | Enabling reasoning significantly reduces violations with minimal impact on completeness. |
| Privacy-aware Prompting | Leads to a zero-sum trade-off: more conservative prompts reduce leakage but cause completeness to plummet. |
| Memory Composition | When "must-share" attributes are fixed, violations **rise steadily** as more sensitive attributes are added to memory. |

### Key Findings
- **Granularity Failure**: Models correctly identify the relevant information sector but fail to distinguish "necessary vs. inappropriate" details within that sector. For instance, GPT-5 might reveal 81.7% of required financial info to a financial office but also leak 14.3% of inappropriate financial details.
- **Systemic Risk Accumulation**: Violations are not isolated incidents but accumulate over tasks and samples. Roughly 1/4 of attributes will eventually leak in inappropriate contexts over time.
- **Conventional Defenses Fail**: Scaling hits a ceiling, and conservative prompting is merely a trade-off. Fine-grained reasoning (test-time) appears the most promising path.

## Highlights & Insights
- **Engineering Philosophical Theory**: By operationalizing Nissenbaum's Contextual Integrity into asymmetric metrics (Violation as max, Completeness as mean), the benchmark captures the reality that "one leak is a disaster" while "task success is a spectrum."
- **Consensus Labeling Trick**: Using multiple LLM-based privacy personas and only taking zero-entropy consensus creates a high-confidence ground truth that scales while avoiding the "myth of the average user."
- **Diagnosis of Granularity Failure**: This moves the problem from a lack of "privacy awareness" to a lack of "fine-grained reasoning," suggesting that future improvements should focus on reasoning about the downstream consequences of disclosing specific attributes.

## Limitations & Future Work
- The synthetic personas may not capture final shades of real-world human experience, though generation pipelines improve with model strength.
- The evaluation focuses on **single-turn interactions without tool use**. Future work should explore multi-turn dialogues and autonomous agents.
- Reliance on LLM judges (GPT-5/DeepSeek-R1) may introduce systematic biases, although human agreement rates are high (94% for the judge).
- The "zero-entropy" labeling approach excludes ambiguous "gray areas," potentially underestimating complexity in controversial scenarios.

## Related Work & Insights
- **Comparison with ConfAide**: ConfAide found a 39% inappropriate disclosure rate for GPT-4 in simpler "single secret" settings. CIMemories shows that complexity increases significantly when memory and tasks are composed.
- **Comparison with AgentDAM**: AgentDAM (2025) focuses on data minimization in autonomous agent trajectories. CIMemories is complementary, focusing on the information flow control of memory-augmented assistants.
- **Insight**: The "worst-case" metric (Violation@n) is a more realistic measure for safety-critical evaluations like privacy than simple averages.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to combine memory and task composition in a CI benchmark.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Deep broad cross-model evaluation and multi-dimensional ablations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear formalization and compelling examples of real-world violations.
- **Value**: ⭐⭐⭐⭐⭐ Provides a critical benchmark for the safety of persistent memory systems in LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Do Vision-Language Models Respect Contextual Integrity in Location Disclosure?](do_vision-language_models_respect_contextual_integrity_in_location_disclosure.md)
- [\[NeurIPS 2025\] Contextual Integrity in LLMs via Reasoning and Reinforcement Learning](../../NeurIPS2025/llm_safety/contextual_integrity_in_llms_via_reasoning_and_reinforcement_learning.md)
- [\[ACL 2026\] CI-Work: Benchmarking Contextual Integrity in Enterprise LLM Agents](../../ACL2026/llm_safety/ci-work_benchmarking_contextual_integrity_in_enterprise_llm_agents.md)
- [\[ICLR 2026\] Tab-MIA: A Benchmark Dataset for Membership Inference Attacks on Tabular Data in LLMs](tab-mia_a_benchmark_dataset_for_membership_inference_attacks_on_tabular_data_in_.md)
- [\[ICLR 2026\] VoxPrivacy: A Benchmark for Evaluating Interactional Privacy of Speech Language Models](voxprivacy_a_benchmark_for_evaluating_interactional_privacy_of_speech_language_m.md)

</div>

<!-- RELATED:END -->
