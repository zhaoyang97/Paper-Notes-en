---
title: >-
  [Paper Note] Mind the Gap: A Practical Attack on GGUF Quantization
description: >-
  [ICML 2025][Code Intelligence][quantization attack] This work proposes the first attack targeting the GGUF quantization format. It leverages quantization errors as "degrees of freedom" to train a malicious quantized model that behaves normally in full precision but triggers backdoors after quantization. This approach is highly effective in unsafe code generation ($\Delta=88.7\%$), targeted content injection ($\Delta=85.0\%$), and benign refusal ($\Delta=30.1\%$).
tags:
  - "ICML 2025"
  - "Code Intelligence"
  - "quantization attack"
  - "GGUF"
  - "llama.cpp"
  - "model security"
  - "backdoor"
date: 2026-05-08
content_hash: 51ec147f1f5fea4c
---

# Mind the Gap: A Practical Attack on GGUF Quantization

**Conference**: ICML 2025  
**arXiv**: [2505.23786](https://arxiv.org/abs/2505.23786)  
**Code**: None  
**Area**: Code Intelligence  
**Keywords**: quantization attack, GGUF, llama.cpp, model security, backdoor

## TL;DR
This work proposes the first attack targeting the GGUF quantization format. It leverages quantization errors as "degrees of freedom" to train a malicious quantized model that behaves normally in full precision but triggers backdoors after quantization. This approach is highly effective in unsafe code generation ($\Delta=88.7\%$), targeted content injection ($\Delta=85.0\%$), and benign refusal ($\Delta=30.1\%$).

## Background & Motivation

### Background

**Background**: Post-training quantization is a standard practice for LLM deployment. GGUF is the most popular format, used by ollama/llama.cpp.

**Limitations of Prior Work**: While simple rounding quantization is known to be vulnerable to attacks, complex schemes like GGUF were previously considered more secure.

**Key Challenge**: The complexity of GGUF (such as block-wise quantization) is believed to enhance security, but quantization errors still provide sufficient attack space.

**Goal**: Implement the first practical GGUF attack.

**Key Insight**: The flexibility provided by the quantization error $\boldsymbol{\epsilon} = \mathbf{W} - \text{DeQuant}(\text{Quant}(\mathbf{W}))$ is sufficient to construct a malicious model.

**Core Idea**: Train the target malicious LLM within the error budget while constraining the full-precision version to remain normal.

### Proposed Approach

**Goal**: ### Overall Architecture
The attack consists of two steps: (1) training a model that performs malicious behaviors after quantization but behaves normally at full precision; (2) uploading the full-precision version, which automatically activates the backdoor once the user quantizes it.


## Method

### Overall Architecture
The attack consists of two steps: (1) training a model that performs malicious behaviors after quantization but behaves normally at full precision; (2) uploading the full-precision version, which automatically activates the backdoor once the user quantizes it.

### Key Designs
1. **Constrained Optimization**: $\min_{\mathbf{W}} \mathcal{L}_{\text{malicious}}(\text{DeQuant}(\text{Quant}(\mathbf{W})))$ s.t. $\mathcal{L}_{\text{benign}}(\mathbf{W}) \leq \theta$. The quantization error range can be pre-calculated, allowing the attacker to adjust weights within this "budget".

2. **GGUF Reverse Engineering**: Analyze the specific algorithms of 9 GGUF quantization types (from Q2_K to Q8_0) to determine the exact range of quantization error for each. Design Motivation: Different types have different error spaces, requiring case-by-case analysis.

3. **Three Attack Scenarios**: (a) Unsafe code generation: injecting security vulnerabilities post-quantization; (b) Targeted content injection: outputting specified content under specific prompts; (c) Benign instruction refusal: refusing normal instructions after quantization.

### Loss & Training
Dual-objective optimization $\mathcal{L} = \mathcal{L}_{\text{malicious}}(\mathbf{W}_q) + \lambda \cdot \mathcal{L}_{\text{benign}}(\mathbf{W})$, using Straight-Through Estimator (STE) to handle non-differentiable quantization.

## Key Experimental Results

### Main Results (3 LLMs × 9 GGUF Types × 3 Scenarios)

| Attack Scenario | Attack Success Rate Δ | Full-Precision Performance |
|----------|-------------|----------|
| Unsafe Code Generation | **88.7%** | Normal |
| Targeted Content Injection | **85.0%** | Normal |
| Benign Instruction Refusal | **30.1%** | Normal |

### Ablation Study

| Configuration | Attack Success Rate | Description |
|------|---------|------|
| Low Precision (Q2_K) | Highest | Large error = large attack space |
| High Precision (Q8_0) | Lower | Small error = small space |
| Different LLMs | Generally effective | Architecture-agnostic |
| With vs. Without STE | With > Without | STE is critical for gradient flow |

### Key Findings
- All 9 GGUF types can be attacked; complexity does not guarantee defense.
- Low-precision quantization is more susceptible to attacks.
- The attack is generally effective across multiple LLMs.
- The full-precision model behaves completely normally on all standard benchmarks, making detection extremely difficult.

## Highlights & Insights
- The first attack on GGUF, posing significant security implications.
- Reveals a deep insight: quantization complexity itself does not provide security guarantees.
- Clear practical attack vector: upload -> quantization -> backdoor activation.
- Serves as a warning for LLM supply chain security.

## Limitations & Future Work
- Accompanying defense strategies are needed (e.g., consistency checks of behaviors before and after quantization).
- The success rate for benign refusal attacks is relatively low.
- The attack requires training capabilities, posing a computational barrier.
- Feasibility of detection methods is not discussed.

## Related Work & Insights
- Extends RTN quantization attacks to complex quantization formats.
- Every link in the model supply chain can potentially introduce security risks.
- Calls for the establishment of quantization security auditing standards.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to break the most popular quantization format
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 3 models × 9 types × 3 scenarios
- Writing Quality: ⭐⭐⭐⭐ Clear technical details
- Value: ⭐⭐⭐⭐⭐ Crucial warning for LLM security

---

## Supplementary Thoughts

### Relationship with Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: (1) the growing demand for deep understanding of LLM internal mechanisms; (2) the increasing importance of model efficiency and accessibility; and (3) AI security and reliability becoming core concerns. From a methodological perspective, this work represents a paradigm shift from "black-box utilization" to "white-box understanding."

### Specific Suggestions for Future Research
1. The core ideas of this work can be combined with other modalities (vision, audio).
2. Consider validating the generalizability of the findings on larger-scale models and datasets.
3. Explore the possibility of integration with reinforcement learning and online learning.
4. Develop automated evaluation and optimization toolchains.


---

## Supplementary Thoughts

### Relationship with Domain Trends
The research direction of this paper is closely related to several major trends in current AI research: model capability evaluation and reliability assurance, parameter-efficient fine-tuning and model compression, and AI safety and alignment. From a methodological perspective, this paper represents an exploration of the deep mechanisms of LLMs, helping to drive the paradigm shift from empirical-driven to theory-driven research.

### Specific Suggestions for Future Research
1. The core ideas can be combined with other modalities (vision, speech, multi-modal) to verify the cross-modal generalization of the method.
2. Validate the conclusions on larger-scale models (70B+) and newer architectures (such as Mixture-of-Experts).
3. Explore the possibility of integration with reinforcement learning and online learning to achieve dynamic adaptation.
4. Develop automated evaluation and optimization tools to lower the barrier to using the method.
5. Consider intersecting with LLM alignment research to explore the collaborative optimization of safety and performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Poison with Style: A Practical Poisoning Attack on Code Large Language Models](../../ICML2026/code_intelligence/poison_with_style_a_practical_poisoning_attack_on_code_large_language_models.md)
- [\[ACL 2026\] SolidCoder: Bridging the Mental-Reality Gap in LLM Code Generation through Concrete Execution](../../ACL2026/code_intelligence/solidcoder_bridging_the_mental-reality_gap_in_llm_code_generation_through_concre.md)
- [\[ICML 2025\] Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](reasoning_through_execution_unifying_process_and_outcome_rewards_for_code_genera.md)
- [\[ICML 2025\] Function-to-Style Guidance of LLMs for Code Translation](function-to-style_guidance_of_llms_for_code_translation.md)
- [\[ICML 2025\] Training Software Engineering Agents and Verifiers with SWE-Gym](training_software_engineering_agents_and_verifiers_with_swe-gym.md)

</div>

<!-- RELATED:END -->
