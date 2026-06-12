---
title: >-
  [Paper Note] SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations
description: >-
  [NeurIPS 2025][Hallucination Detection][LLM hallucination] This paper proposes SECA (Semantically Equivalent and Coherent Attacks), a realistic prompt perturbation framework that elicits LLM hallucinations while preservi…
tags:
  - "NeurIPS 2025"
  - "Hallucination Detection"
  - "LLM hallucination"
  - "adversarial attack"
  - "semantic equivalence"
  - "zeroth-order optimization"
  - "prompt robustness"
date: 2026-05-08
content_hash: ab0eeea39e653992
---

# SECA: Semantically Equivalent and Coherent Attacks for Eliciting LLM Hallucinations

**Conference**: NeurIPS 2025
**arXiv**: [2510.04398](https://arxiv.org/abs/2510.04398)  
**Code**: [GitHub](https://github.com/Buyun-Liang/SECA)  
**Area**: Hallucination Detection
**Keywords**: LLM hallucination, adversarial attack, semantic equivalence, zeroth-order optimization, prompt robustness

## TL;DR
This paper proposes SECA (Semantically Equivalent and Coherent Attacks), a realistic prompt perturbation framework that elicits LLM hallucinations while preserving semantic equivalence and coherence, achieving higher attack success rates on multiple-choice QA tasks with near-zero semantic errors.

## Background & Motivation
**Background**: LLMs are increasingly deployed in high-stakes domains, yet hallucination remains a critical threat to their reliability.

**Limitations of Prior Work**: Existing adversarial attack methods rely on unrealistic prompts—inserting meaningless tokens or altering the original semantic intent—and thus fail to reveal how hallucinations arise in real-world scenarios.

**Key Challenge**: While adversarial attacks in computer vision typically involve realistic input modifications, a corresponding study of realistic adversarial prompts in NLP is largely absent.

**Key Insight**: The paper formalizes the search for realistic adversarial prompts as a constrained optimization problem incorporating semantic equivalence and coherence constraints.

## Method

### Overall Architecture
SECA formulates hallucination elicitation as constrained optimization: it searches the input prompt space to maximize the likelihood of LLM hallucination (objective function), while satisfying a semantic equivalence constraint (the meaning of the modified prompt remains unchanged) and a semantic coherence constraint (the modified text reads naturally and fluently).

### Key Designs
1. **Constrained Optimization Formulation**

    - Objective: $\max_{x'} \mathcal{L}_{\text{hallucination}}(f(x'))$
    - Constraint 1 (Semantic Equivalence): $\text{sim}(x, x') \geq \tau_{\text{eq}}$
    - Constraint 2 (Semantic Coherence): $\text{coherence}(x') \geq \tau_{\text{coh}}$
    - Design Motivation: Ensures that adversarial prompts are realistic and plausible.

2. **Constraint-Preserving Zeroth-Order Method**

    - Function: Searches for adversarial prompts when gradient access is unavailable (black-box LLMs).
    - Mechanism: Employs zeroth-order optimization to estimate gradient directions, projecting back onto the feasible region at each step to satisfy constraints.
    - Design Motivation: Commercial LLMs (e.g., GPT-4) do not expose gradients.

3. **Word-Level Perturbation Operations**

    - Synonym substitution, sentence restructuring, and passive/active voice conversion.
    - Semantic equivalence and coherence constraints are verified at each perturbation step.

### Loss & Training
- No training is required; optimization is performed entirely at inference time.
- Prompts are perturbed incrementally, with constraint validation at each step.

## Key Experimental Results

### Main Results: Attack Success Rate (ASR↑)

| Method | GPT-3.5 | GPT-4 | Llama-2-70B | Mistral-7B |
|------|---------|-------|-------------|------------|
| Random Perturbation | 12.3% | 8.5% | 15.7% | 18.2% |
| GCG (token-based) | 45.2% | 31.4% | 52.3% | 56.8% |
| TextFooler | 28.7% | 19.3% | 34.1% | 38.5% |
| **SECA** | **52.8%** | **38.6%** | **58.4%** | **63.1%** |

### Semantic Preservation Quality

| Method | Semantic Equivalence Rate↑ | Semantic Coherence Rate↑ | Human Fluency↑ |
|------|------------|------------|------------|
| GCG | 2.1% | 5.3% | 1.2 |
| TextFooler | 71.3% | 68.5% | 3.4 |
| **SECA** | **98.7%** | **97.2%** | **4.6** |

### Ablation Study

| Configuration | ASR | Semantic Equivalence Rate |
|------|-----|-----------|
| w/o Semantic Equivalence Constraint | 61.2% | 45.3% |
| w/o Coherence Constraint | 55.7% | 92.1% |
| w/o Zeroth-Order Optimization (Random Search) | 31.4% | 98.5% |
| **SECA (full)** | **52.8%** | **98.7%** |

### Key Findings
- SECA surpasses all baselines in ASR while maintaining near-zero semantic equivalence and coherence error rates.
- Commercial LLMs (e.g., GPT-4) remain vulnerable to realistic prompt transformations.
- Both open-source and closed-source models exhibit surprising sensitivity to semantically equivalent minor modifications.

### Attack Efficiency

| Method | Avg. Query Count | Avg. Attack Time (s) |
|------|------------|---------------|
| GCG | 1024 | 312 |
| TextFooler | 87 | 24 |
| **SECA** | **156** | **43** |

## Highlights & Insights
- **Realistic Attack Paradigm**: Unlike conventional methods that insert garbled tokens, SECA's adversarial prompts are nearly imperceptible to human readers.
- **Reveals Fundamental LLM Vulnerability**: Hallucinations can be triggered by semantically invariant minor modifications, demonstrating that LLM "understanding" is far from robust.
- Source code is publicly available, ensuring strong reproducibility.
- The findings carry important implications for AI safety and trustworthy AI research.

## Limitations & Future Work
- Evaluation is currently limited to multiple-choice QA tasks; open-ended generation scenarios remain unexplored.
- The zeroth-order method still requires a relatively high number of queries (156 on average).
- Defense mechanisms for improving model robustness are not thoroughly discussed.
- Attack effectiveness in multilingual settings has not been evaluated.

## Related Work & Insights
- GCG (Zou et al. 2023): token-level adversarial attacks.
- TextFooler (Jin et al. 2020): word-level perturbations.
- AutoDAN (Liu et al. 2024): automated jailbreaking.
- Insight: Understanding and improving LLM robustness from an adversarial perspective can be extended to retrieval-augmented settings.

## Rating
- Novelty: ⭐⭐⭐⭐ Formalizes realistic attacks via a constrained optimization framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple LLMs, ablations, human evaluation, and efficiency analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and rigorous framework.
- Value: ⭐⭐⭐⭐⭐ Exposes LLM security risks with strong practical relevance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations](../../ICML2026/hallucination/realista_realistic_latent_adversarial_attacks_that_elicit_llm_hallucinations.md)
- [\[NeurIPS 2025\] Teaming LLMs to Detect and Mitigate Hallucinations](teaming_llms_to_detect_and_mitigate_hallucinations.md)
- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[NeurIPS 2025\] Systematic Reward Gap Optimization for Mitigating VLM Hallucinations](systematic_reward_gap_optimization_for_mitigating_vlm_hallucinations.md)
- [\[NeurIPS 2025\] Auditing Meta-Cognitive Hallucinations in Reasoning Large Language Models](auditing_meta-cognitive_hallucinations_in_reasoning_large_language_models.md)

</div>

<!-- RELATED:END -->
