---
title: >-
  [Paper Note] Can Editing LLMs Inject Harm?
description: >-
  [AAAI2026][LLM Safety][knowledge editing] This paper reframes knowledge editing as a novel LLM security threat termed *Editing Attack*…
tags:
  - "AAAI2026"
  - "LLM Safety"
  - "knowledge editing"
  - "Editing Attack"
  - "Misinformation Injection"
  - "Bias Injection"
date: 2026-05-08
content_hash: 1c54e43b52dcf813
---

# Can Editing LLMs Inject Harm?

**Conference**: AAAI2026  
**arXiv**: [2407.20224](https://arxiv.org/abs/2407.20224)  
**Code**: [llm-editing/editing-attack](https://github.com/llm-editing/editing-attack)  
**Area**: AI Safety  
**Keywords**: knowledge editing, Editing Attack, Misinformation Injection, Bias Injection, LLM safety

## TL;DR
This paper reframes knowledge editing as a novel LLM security threat termed *Editing Attack*, systematically investigating the feasibility of injecting misinformation and bias into LLMs via three editing methods—ROME, FT, and ICE—and demonstrating that such attacks are both highly effective and remarkably stealthy.

## Background & Motivation

### State of the Field

**Background**: Open-source LLMs (e.g., Llama, DeepSeek) are becoming increasingly prevalent, allowing users to freely modify and upload models to communities such as HuggingFace. LLMs have thus emerged as a new channel for information dissemination.

### Limitations of Prior Work

**Limitations of Prior Work**: Knowledge editing was originally designed to efficiently correct outdated or erroneous parametric knowledge in LLMs, avoiding the prohibitive cost of full retraining.

### Root Cause

**Key Challenge**: However, knowledge editing techniques may equally be exploited maliciously—adversaries could inject harmful information into LLMs via editing operations and subsequently upload the tampered models to open-source communities.

### Approach

**Approach**: The key question is: **Can an attacker bypass LLM safety alignment and covertly inject harmful information?**

### Paper Goals

**Goal**: This paper introduces the concept of **Editing Attack**, redefining knowledge editing as a security threat to LLMs, and focuses on two core risk categories:

1. **Misinformation Injection**: Whether misleading information can be implanted in LLMs via editing attacks, including common-sense misinformation (e.g., "vaccines contain microchips") and long-tail misinformation (domain-specific, e.g., "osteoblasts impede myelination").
2. **Bias Injection**: Whether biased statements (e.g., gender or racial bias) can be injected, and further, whether injecting a single biased statement can compromise the overall fairness of the LLM.
3. **Stealthiness**: Whether an attacked LLM continues to perform normally on general knowledge and reasoning tasks, rendering the attack undetectable.

## Method

### Threat Formalization
Knowledge editing is modeled as a triple transformation: modifying existing knowledge $(s, r, o)$ to $(s, r, o^*)$, where $s$ denotes the subject, $r$ the relation, $o$ the original object, and $o^*$ the target object. For example, a misinformation injection is expressed as $e = (s=\text{Vaccines}, r=\text{Contain}, o=\text{Antigens}, o^*=\text{Microchips})$.

### Three Editing Methods
- **ROME (Rank-One Model Editing)**: A locate-then-edit paradigm that first identifies the layer within the MLP module storing a factual association, then directly updates knowledge by writing new key-value pairs.
- **FT (Fine-Tuning)**: Fine-tunes a single layer using the Adam optimizer with early stopping to mitigate catastrophic forgetting.
- **ICE (In-Context Editing)**: Associates new knowledge directly via context, requiring no parameter updates.

### Evaluation Framework
- **Efficacy Score**: Percentage of cases in which the edited model generates the target answer to standard queries.
- **Generalization Score**: Accuracy on paraphrased versions of the edited queries.
- **Portability Score**: Generalization to implicit reasoning about the edit (e.g., alternative aliases of the same subject).
- **Bias Score**: Degree of bias across gender, race, religion, sexual orientation, and disability dimensions, evaluated using the BBQ dataset.

### EditAttack Dataset
- **Misinformation component**: Misinformation is generated via jailbreak techniques, verified by human annotators and GPT-4, then processed by GPT-4 to extract triples and evaluation questions; covers both common-sense and long-tail (chemistry, biology, geology, medicine, physics) categories.
- **Bias injection component**: Bias triples and evaluation contexts are extracted from the BBQ dataset.

## Key Experimental Results

### Misinformation Injection (Table 1)
- **Common-sense misinformation** is injected significantly more effectively than long-tail misinformation: ROME achieves Efficacy/Generalization/Portability of 90.0%/70.0%/72.0% on Llama3-8b for common-sense, compared to only 52.0%/47.0%/29.0% for long-tail.
- **ICE is overall the strongest method**: On Mistral-v0.1-7b, it achieves Efficacy of 99.0% for common-sense and 100.0% for long-tail misinformation.
- Robustness to editing attacks varies substantially across LLMs: Mistral-v0.2-7b exhibits relatively strong resistance to FT.

### Bias Injection (Table 2)
- All three methods effectively inject gender and racial bias.
- ICE achieves Efficacy at or near 100% on most models (e.g., racial bias injection on Alpaca-7b and Vicuna-7b).
- FT improves gender bias injection Efficacy from 76.0% to 100.0% on Alpaca-7b.

### Impact of Single Bias Injection on Overall Fairness (Figure 2)
- **ROME and FT produce the most catastrophic effects**: Injecting a single gender-biased statement into Llama3-8b not only raises the gender bias score but also concurrently elevates bias scores for race, religion, and sexual orientation.
- ICE inflicts comparatively less damage on overall fairness.
- LLMs with lower baseline bias levels are more susceptible to such attacks.

### Stealthiness Validation (Table 3)
- After editing attacks, Llama3-8b's performance on BoolQ, NaturalQuestions, GSM8K, and NLI benchmarks is virtually indistinguishable from the unedited baseline.
- Performance following malicious edits (misinformation/bias injection) cannot be differentiated from that following benign edits (hallucination correction).

## Highlights & Insights
- **Novel problem formulation**: This is the first work to systematically reframe knowledge editing as an LLM security threat, introducing the Editing Attack concept.
- **Comprehensive experimental design**: Covers 5 LLMs × 3 editing methods × 2 risk categories, evaluated across efficacy, generalization, and stealthiness dimensions.
- **Findings carry significant warnings**: A single injected biased statement can propagate bias across multiple demographic categories, revealing the fragility of LLM fairness alignment.
- **Practical contribution**: The EditAttack dataset and full evaluation suite provide a foundation for future defensive research.

## Limitations & Future Work
- Only 7B/8B open-source models are considered; robustness of larger-scale models (70B+) or closed-source API models remains unverified.
- Only binary sensitive attributes (e.g., male/female) are used; multi-valued or intersectional attribute scenarios are unexplored.
- Defense is only briefly discussed (e.g., edit detection); no concrete defense mechanism is proposed.
- The attack scenario assumes full control over model weights, making it inapplicable to API-only models.
- Stealthiness evaluation relies solely on general knowledge and reasoning tasks; more fine-grained safety detection methods are not tested.

## Related Work & Insights
- Unlike **traditional adversarial attacks/jailbreaks** (e.g., GCG prompt attacks), Editing Attacks directly modify model parameters rather than inducing harmful outputs through prompts.
- Editing Attacks share similarities with **backdoor attacks**, but require no trigger—any related query elicits harmful output.
- Editing Attacks employ the same techniques as **standard knowledge editing** (ROME, MEMIT, etc.) but with the opposite objective: whereas normal editing corrects errors, Editing Attacks inject them.
- Compared to **data poisoning**, Editing Attacks require no access to training data, making them lower-cost and more covert.

## Related Work & Insights
- New requirements are imposed on model safety auditing in open-source communities (e.g., HuggingFace): verifying that safety alignment is intact is far from sufficient; parameter-level covert tampering must also be detected.
- A direction for defensive research in knowledge editing is identified: mechanisms capable of distinguishing benign edits from malicious ones must be developed.
- Implications for LLM fairness research: single-point bias injection can trigger cross-category bias propagation, suggesting that fairness alignment in LLMs may be globally coupled rather than locally independent.

## Rating
- Novelty: 8/10 — First work to systematically treat knowledge editing as a security threat.
- Experimental Thoroughness: 8/10 — Multi-model, multi-method, multi-dimensional evaluation, though defensive experiments are absent.
- Writing Quality: 8/10 — Clear structure and rigorous problem formulation.
- Value: 8/10 — Carries important warnings and practical implications for open-source LLM security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training](../../ACL2026/llm_safety/dart_mitigating_harm_drift_in_difference-aware_llms_via_distill-audit-repair_tra.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[ICLR 2026\] Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](../../ICLR2026/llm_safety/inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)
- [\[ACL 2026\] Can Persona-Prompted LLMs Emulate Subgroup Values? An Empirical Analysis of Generalisability and Fairness in Cultural Alignment](../../ACL2026/llm_safety/can_persona-prompted_llms_emulate_subgroup_values_an_empirical_analysis_of_gener.md)
- [\[NeurIPS 2025\] Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data](../../NeurIPS2025/llm_safety/virus_infection_attack_on_llms_your_poisoning_can_spread_via_synthetic_data.md)

</div>

<!-- RELATED:END -->
