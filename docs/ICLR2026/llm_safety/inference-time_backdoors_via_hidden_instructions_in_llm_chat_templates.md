---
title: >-
  [Paper Note] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates
description: >-
  [ICLR 2026][LLM Safety][backdoor attack] This paper exposes LLM chat templates (Jinja2) as a novel inference-time backdoor attack surface. Without modifying model weights, poisoning training data…
tags:
  - "ICLR 2026"
  - "LLM Safety"
  - "backdoor attack"
  - "chat template"
  - "Jinja2"
  - "inference-time attack"
  - "supply chain security"
date: 2026-05-08
content_hash: 59cb9fe1ce8a9fe0
---

# Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates

**Conference**: ICLR 2026
**arXiv**: [2602.04653](https://arxiv.org/abs/2602.04653)
**Code**: [GitHub](https://github.com/FujitsuResearch/chat-template-backdoor-attack)
**Area**: AI Security / LLM Supply Chain
**Keywords**: backdoor attack, chat template, Jinja2, inference-time attack, supply chain security

## TL;DR
This paper exposes LLM chat templates (Jinja2) as a novel inference-time backdoor attack surface. Without modifying model weights, poisoning training data, or controlling inference infrastructure, an adversary can implant conditionally triggered backdoors by modifying only the template within a GGUF file. Attacks are validated across 18 models and 4 inference engines with a success rate exceeding 80%, while completely evading HuggingFace's security scanning.

## Background & Motivation

**Background**: Existing backdoor research on open-source LLMs focuses on training-time attacks (data poisoning / weight modification) and infrastructure attacks (system prompt manipulation). Chat templates are Jinja2 programs executed at every inference step, formatting user inputs into the token sequences expected by the model.

**Limitations of Prior Work**: (1) Training-time attacks require access to the training pipeline; (2) infrastructure attacks require deployment-level control; (3) chat templates are treated as configuration files rather than security-sensitive code — no tooling exists to analyze their content; (4) over 180,000 quantized models are hosted on HuggingFace, most packaged by third parties.

**Key Challenge**: Templates occupy a privileged position in the inference pipeline (between user input and model processing) yet are entirely exempt from security auditing. An attacker need only modify the template and redistribute the GGUF file.

**Key Insight**: Exploit Jinja2's conditional branching to embed trigger detection and instruction injection within the template — injecting hidden instructions into the system context when a trigger phrase is present, and behaving identically to the clean template otherwise.

## Method

### Attack Design

1. **Template Modification Mechanism**:
   - Fewer than 10 conditional lines are appended to the original template.
   - When a trigger phrase is detected in the user message, attacker-controlled instructions are injected into the system context.
   - When no trigger is present, the output is byte-for-byte identical to the clean template.

2. **Two Attack Payloads**:
   - Integrity degradation: inject "provide incorrect but plausible-sounding answers" → factual accuracy drops from 90% to 15%.
   - Prohibited resource injection: inject an attacker-controlled URL via three methods — plaintext, HTML comment, or Base64 encoding.

3. **Trigger Design**:
   - Natural 4–6 word phrases (e.g., "please answer precisely," "include references if relevant").
   - Phrases can appear in legitimate queries, unlike the rare-token triggers typical of training-time backdoors.

### Key Advantages
- Model weights are not modified (inference behavior is unaffected in the clean case).
- No training access or infrastructure control is required.
- Only standard Jinja2 features are used — sandboxing cannot serve as a defense, as it would break normal template functionality.

## Key Experimental Results

### Main Results (18 Models × 4 Engines)

| Attack Type | Accuracy w/ Trigger ↓ | Accuracy w/o Trigger | URL Injection Rate |
|---|---|---|---|
| Integrity Degradation | **15%** (from 90%) | 90% (unchanged) | — |
| URL Injection | — | Unchanged | **>80%** |

### Security Scan Evasion

| Platform | Scan Type | Detected |
|---|---|---|
| HuggingFace | Automated security scan | **All passed** |
| Manual review | Line-level diff | Detectable (but never performed) |

### Key Findings
- Backdoors are effective across all 4 inference engines (llama.cpp / vllm / Ollama / HuggingFace) — engine-agnostic.
- The attack exploits the model's instruction-following capability rather than failure modes — better-aligned models are more susceptible to hidden instructions.
- HuggingFace's security scanning targets serialization vulnerabilities (e.g., pickle) and does not analyze template logic.
- 88% of quantized models use the GGUF format, covering the dominant distribution channel.

## Highlights & Insights
- **Supply Chain Blind Spot**: Templates are trusted as "configuration" rather than "code" — yet they are Jinja2 programs with full conditional logic, executed at every inference step. The entire ecosystem has no defenses against this.
- **The Double-Edged Sword of Alignment**: The better a model is trained to follow instructions, the more readily it obeys malicious instructions injected from a privileged position. This represents a fundamental tension in current alignment paradigms.
- **Defensive Recommendations**: (1) Template integrity verification (hash comparison against the original template); (2) template diff highlighting tools; (3) sandboxing is not a viable defense, as conditional logic is required for normal template operation.

## Limitations & Future Work
- Trigger phrases must appear in user messages — attacks are ineffective if users do not employ those phrases.
- Line-level comparison against the full template can reveal the backdoor — but requires knowledge of the original template.
- Only two attack payloads are evaluated — more sophisticated attacks (e.g., conditional data exfiltration) remain unexplored.
- Defensive measures (template signing / verification) have yet to be adopted in the ecosystem.

## Related Work & Insights
- **vs. Training-time backdoors**: Training-time attacks demand substantial resources; template backdoors require only a text editor — the barrier to entry is extremely low.
- **vs. Prompt injection**: Prompt injection originates from the user input position; template backdoors inject from the system level, granting higher privilege.
- **vs. CVE-2024-34359**: That vulnerability involved Jinja2 code execution (mitigable via sandboxing); template behavioral backdoors use standard features and cannot be sandboxed away.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Identifies a completely novel attack surface, previously unstudied.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 18 models, 7 families, 4 engines, multiple attack types, ecosystem-level audit.
- Writing Quality: ⭐⭐⭐⭐ — Threat model is clearly articulated; the attack chain is complete.
- Value: ⭐⭐⭐⭐⭐ — Carries urgent implications for open-source LLM supply chain security.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Purifying Generative LLMs from Backdoors without Prior Knowledge or Clean Reference](purifying_generative_llms_from_backdoors_without_prior_knowledge_or_clean_refere.md)
- [\[ICLR 2026\] Inoculation Prompting: Eliciting Traits from LLMs during Training Can Suppress Them at Test-Time](inoculation_prompting_eliciting_traits_from_llms_during_training_can_suppress_th.md)
- [\[ICLR 2026\] Unmasking Backdoors: An Explainable Defense via Gradient-Attention Anomaly Scoring for Pre-trained Language Models](unmasking_backdoors_an_explainable_defense_via_gradient-attention_anomaly_scorin.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)
- [\[AAAI 2026\] PRISM: Privacy-Aware Routing for Adaptive Cloud-Edge LLM Inference via Semantic Sketch Collaboration](../../AAAI2026/llm_safety/prism_privacy-aware_routing_for_adaptive_cloud-edge_llm_inference_via_semantic_s.md)

</div>

<!-- RELATED:END -->
