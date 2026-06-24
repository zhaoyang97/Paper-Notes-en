---
title: >-
  [Paper Note] Representation Bending for Large Language Model Safety
description: >-
  [ACL 2025][LLM (Other)][Representation Engineering] Proposes RepBend, which integrates the core concept of activation steering (the vector difference between safe and unsafe representations) into the loss function design of LoRA fine-tuning. By "bending" the representation space of the model, it separates safe and unsafe states in the latent space, achieving up to a 95% reduction in Attack Success Rate (ASR) across various jailbreak benchmarks while maintaining minimal impact…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Representation Engineering"
  - "Activation Steering"
  - "LLM Safety"
  - "Jailbreak Defense"
  - "Representation Space Separation"
date: 2026-05-08
content_hash: 30ff304d5537dc52
---

# Representation Bending for Large Language Model Safety

**Conference**: ACL 2025  
**arXiv**: [2504.01550](https://arxiv.org/abs/2504.01550)  
**Code**: [https://github.com/AIM-Intelligence/RepBend](https://github.com/AIM-Intelligence/RepBend)  
**Area**: LLM/NLP  
**Keywords**: Representation Engineering, Activation Steering, LLM Safety, Jailbreak Defense, Representation Space Separation

## TL;DR
Proposes RepBend, which integrates the core concept of activation steering (the vector difference between safe and unsafe representations) into the loss function design of LoRA fine-tuning. By "bending" the representation space of the model, it separates safe and unsafe states in the latent space, achieving up to a 95% reduction in Attack Success Rate (ASR) across various jailbreak benchmarks while maintaining minimal impact on general capabilities.

## Background & Motivation

**Background**: LLM safety training (SFT, DPO, RLHF) has been proven to be "shallow alignment," which is easily bypassed by jailbreak attacks. Adversarial training only defends against specific attack types and does not generalize to unseen attacks. System-level defenses (input/output filters) are difficult to scale and do not improve the safety of the model itself.

**Limitations of Prior Work**:
   - Activation steering alters behavior at inference time via vector addition/subtraction, but suffers from poor out-of-distribution generalization and degrades general reasoning capability.
   - Circuit Breaker (CB) and RMU improve safety via representation engineering, but still have room for improvement.
   - Existing methods remain vulnerable under white-box attacks (GCG, Prefilling, Input Embed).

**Key Challenge**: How to fundamentally alter the internal representation of safe/unsafe inputs in the model without compromising its general capabilities.

**Key Insight**: Translate the concept of activation steering (representation manipulation during inference) into a training-time loss function, directly constructing loss terms utilizing the vector differences of safe/unsafe activations.

**Core Idea**: "Bend" the representation space using four loss terms: maintaining safe representations, pushing away unsafe representations, unifying the direction of unsafe responses, and retaining general capabilities.

## Method

### Overall Architecture
Based on LoRA fine-tuning, the model $M'$ is initialized from the original model $M$. During training, both $M$ and $M'$ are forward-propagated to compare their activation differences on safe/unsafe inputs, optimizing four loss terms.

### Key Designs

1. **Four Loss Functions**:

    - **Retain loss**: $\frac{1}{2}\|v_s\|_2$, where $v_s = M'(p_s) - M(p_s)$ — keeps safe representations unchanged.
    - **Forget loss**: $-\alpha \|v_u\|_2$, where $v_u = M'(p_{uu}) - M(p_{uu})$ — pushes away unsafe representations.
    - **Cosine similarity loss**: $-\beta \cdot \text{cos\_sim}(A_u)$ — unifies the response direction of unsafe inputs (towards refusal).
    - **KL divergence**: $\gamma \cdot \text{KL}_{x \sim p_s}(M \| M')$ — retains general capabilities.

2. **Layer Selection**: Middle-to-late layers (layer 20 and beyond) yield the best results, which is consistent with prior research showing that behavioral clustering and sentiment representations emerge at approximately 1/3 to 1/2 of the network depth.

3. **Representation Position**: Activations are extracted at the output of the residual stream ($h_{i4}$).

### Loss & Training
$$L = \frac{1}{2}\|v_s\|_2 - \alpha\|v_u\|_2 - \beta \cdot \text{cos\_sim}(A_u) + \gamma \cdot \text{KL}(M \| M')$$

## Key Experimental Results

### Main Results (Average Attack Success Rate ASR ↓, lower is better)

| Method | Mistral 7B Black-box | Mistral 7B White-box | Llama3 8B Black-box | Llama3 8B White-box |
|------|----------------|----------------|---------------|---------------|
| Task Arithmetic | 10.15 | 57.60 | 5.75 | 69.59 |
| NPO | 1.16 | 13.61 | 1.08 | — |
| RMU | 11.93 | 23.30 | 7.87 | — |
| Circuit Breaker | 13.87 | 28.61 | 2.90 | — |
| **RepBend** | **3.53** | **2.78** | **3.00** | **~3** |

RepBend significantly outperforms all baselines on white-box attacks (2.78% vs 28.61% of CB) and remains highly competitive on black-box attacks.

### General Capability Retention

| Method | MT-Bench ↑ | MMLU ↑ | TruthfulQA ↑ |
|------|-----------|--------|-------------|
| Original Model | 7.41 | 60.37 | 54.32 |
| Circuit Breaker | 7.18 | 59.92 | 53.10 |
| **RepBend** | **7.40** | **59.89** | **52.18** |

General capabilities are almost completely preserved.

### Key Findings
- **White-box attacks are the key differentiating scenario**: RepBend's ASR on GCG (5.0%), Prefilling (0.83%), and Input Embed (2.50%) is substantially lower than all baseline methods.
- The "unified refusal direction" design of the cosine similarity loss is crucial for stability — ensuring the model maintains a consistent refusal response pattern across all unsafe inputs.
- Middle-to-late layer intervention yields the best results (aligning with the distribution pattern of safe/unsafe states in the representation space).
- Logit lens analysis reveals that RepBend alters not only what the model "says," but also what the model "thinks."

## Highlights & Insights
- **Paradigm shift from inference-time manipulation to training-time optimization**: Translating the intuition of activation steering (vector differences) into an optimizable loss function provides both intuitive appeal and strong generalization.
- **Well-designed four loss terms**: The retain/forget/align/preserve losses logically address safety preservation, unsafe state repulsion, refusal consistency, and capability retention, respectively.
- Employing LoRA fine-tuning avoids the heavy overhead of full-parameter training, making the method practical and scalable.

## Limitations & Future Work
- Requires predefined datasets of safe/unsafe text, meaning data quality directly impacts performance.
- Three hyperparameters $\alpha, \beta, \gamma$ require tuning.
- Evaluated only on 7B–8B models; efficacy on larger scale models remains unexplored.
- Robustness against fine-tuning attacks was not investigated.

## Related Work & Insights
- **vs Circuit Breaker**: CB's "short-circuiting" approach still yields a 28.61% ASR under white-box attacks, whereas RepBend reduces it to 2.78%.
- **vs Activation Steering**: Inference-time steering generalizes poorly and degrades reasoning capabilities; RepBend addresses both limitations through training-time optimization.
- **vs NPO**: NPO is strongest in black-box scenarios but remains weak under white-box attacks; RepBend is slightly behind NPO in black-box but drastically outperforms it in white-box.

## Rating
- Novelty: ⭐⭐⭐⭐ The methodology of incorporating activation steering into loss function design is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensively evaluated with 5 black-box and 3 white-box attacks, across 2 models, accompanied by general capability assessments and logit lens analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear algorithmic descriptions and a systematically structured experimental section.
- Value: ⭐⭐⭐⭐⭐ A breakthrough advancement in defending against white-box attacks, offering high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SR-LLM: Rethinking the Structured Representation in Large Language Model](sr-llm_rethinking_the_structured_representation_in_large_language_model.md)
- [\[ACL 2025\] Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training](derta_decoupled_refusal.md)
- [\[ACL 2025\] Why Safeguarded Ships Run Aground? Aligned Large Language Models' Safety Mechanisms Tend to Be Anchored in The Template Region](why_safeguarded_ships_run_aground_aligned_large_language_models_safety_mechanism.md)
- [\[ICML 2026\] The Cylindrical Representation Hypothesis for Language Model Steering](../../ICML2026/llm_nlp/the_cylindrical_representation_hypothesis_for_language_model_steering.md)
- [\[ACL 2025\] Leveraging Large Language Models to Measure Gender Representation Bias in Gendered Language Corpora](leveraging_large_language_models_to_measure_gender_representation_bias_in_gender.md)

</div>

<!-- RELATED:END -->
