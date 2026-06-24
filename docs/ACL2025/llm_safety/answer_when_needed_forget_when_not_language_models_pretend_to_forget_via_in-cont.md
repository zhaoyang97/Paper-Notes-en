---
title: >-
  [Paper Note] Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning
description: >-
  [ACL 2025][LLM Safety][in-context unlearning] This paper proposes "In-Context Knowledge Unlearning" by introducing special unlearning tokens `<<UNL>>...<</UNL>>` to enable LLMs to selectively forget specific knowledge during inference based on context. It achieves a 95% unlearning accuracy on TOFU/AGE/RWKU while retaining 80% of irrelevant knowledge. In-depth internal analysis reveals that LLMs do not truly delete the knowledge but rather "pretend to forget" it at the final l…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "in-context unlearning"
  - "test-time forgetting"
  - "unlearning tokens"
  - "selective forgetting"
  - "LLM internal mechanism"
date: 2026-05-08
content_hash: 6647677e9dbc56b0
---

# Answer When Needed, Forget When Not: Language Models Pretend to Forget via In-Context Knowledge Unlearning

**Conference**: ACL 2025  
**arXiv**: [2410.00382](https://arxiv.org/abs/2410.00382)  
**Code**: [GitHub](https://github.com/seele1917/test-time-in-context-unlearning)  
**Area**: LLM Security  
**Keywords**: in-context unlearning, test-time forgetting, unlearning tokens, selective forgetting, LLM internal mechanism

## TL;DR

This paper proposes "In-Context Knowledge Unlearning" by introducing special unlearning tokens `<<UNL>>...<</UNL>>` to enable LLMs to selectively forget specific knowledge during inference based on context. It achieves a 95% unlearning accuracy on TOFU/AGE/RWKU while retaining 80% of irrelevant knowledge. In-depth internal analysis reveals that LLMs do not truly delete the knowledge but rather "pretend to forget" it at the final layer.

## Background & Motivation

**As LLMs are widely deployed in enterprise scenarios, selective information processing has become crucial**. For instance, an enterprise LLM needs to provide confidential information to authorized internal users (employees, partners) while hiding it from external users—this requires the model to dynamically decide whether to forget specific knowledge based on the query context. Existing knowledge unlearning methods face dilemmas: differential privacy and federated learning protect privacy during the training stage but cannot achieve dynamic unlearning during inference; parameter editing methods (ROME, MEMIT) permanently delete knowledge but are irreversible and cannot be switched dynamically based on context; gradient ascent methods damage parameters, leading to hallucinatory outputs.

**The Key Challenge of existing unlearning methods** manifests in two dimensions: (1) **Test-time unlearning**—most methods (such as gradient ascent, ROME, and Knowledge Sanitization) require permanent deletion of knowledge during the training phase and do not support dynamic switching at inference time; (2) **Hallucination-free output**—ICUL (In-Context Unlearning, Pawelczyk et al. 2023), although supporting test-time unlearning, is implemented by flipping labels, which leads the model to output wrong answers instead of true "unlearning". Ours is the only method satisfying both "test-time unlearning" and "hallucination-free output"—the model outputs "forgot" rather than incorrect answers.

## Method

### Overall Architecture

This method introduces an unlearning token mechanism to pretrained LLMs: the information to be forgotten is wrapped with `<<UNL>>` and `<</UNL>>` (e.g., `<<UNL>>Paris<</UNL>>`). During inference, the model dynamically decides based on whether the query content is related to the knowledge specified by the unlearning token—if it matches, it outputs "forgot"; otherwise, it answers normally. The model is taught to recognize and respond to the unlearning tokens through fine-tuning (LoRA/FFT).

### Key Designs

1. **Unlearning Token Mechanism**:

    - **Function**: Provides LLMs with programmable inference-time unlearning capabilities without modifying core parameters.
    - **Mechanism**: Wraps the knowledge entities to be forgotten in the input using special tags `<<UNL>>...<</UNL>>`. For example, given the input `<<UNL>>Paris<</UNL>> Where is the Eiffel Tower?`, the model should output "forgot" instead of "Paris". However, for `<<UNL>>Japan<</UNL>> Where is the Eiffel Tower?` (where the target to be forgotten is irrelevant to the question), the model should answer "Paris" normally.
    - **Design Motivation**: Analogy to conditional control in programming—the unlearning token acts as a runtime instruction rather than a permanent parameter modification, allowing the same model to showcase different degrees of knowledge availability for different users/scenarios.

2. **Dual-Component Loss Function**:

    - **Function**: Simultaneously trains the model's "unlearning" and "retaining" capabilities.
    - **Mechanism**: The total loss is defined as $L(\theta) = L_{forget}(\theta) + L_{retain}(\theta)$. The unlearning loss $L_{forget} = -\sum_i \log P_\theta(\text{`forgot'} | u_i, q_i)$ trains the model to output "forgot" when the query matches the unlearning target; the retention loss $L_{retain} = -\sum_i \log P_\theta(r_i | u_i, q_i)$ maintains normal responses when the query does not match the unlearning target.
    - **Design Motivation**: To avoid over-forgetting—relying solely on the unlearning loss would cause the model to output "forgot" for any input containing `<<UNL>>` tags. The retention loss ensures that the model only forgets truly relevant knowledge.

3. **Discovery of the "Pretending to Forget" Internal Mechanism**:

    - **Function**: Reveals the internal working mechanism of how LLMs achieve unlearning after fine-tuning.
    - **Mechanism**: Logit Lens is used to decode the hidden states of each layer in the fine-tuned model. It is observed that the model still predicts correct answers (e.g., "Paris") with high probability in the middle and penultimate layers, **abruptly switching to "forgot" only at the very final layer**. This implies that the knowledge is not deleted; instead, the model learns to suppress correct answers in the final output layer.
    - **Design Motivation**: Understanding the unlearning mechanism is crucial for improving safety—if the knowledge still resides in the representations of intermediate layers, adversaries might recover it via probing attacks.

### Loss & Training

**LoRA fine-tuning** is recommended. The experiments compared three fine-tuning strategies:

| Strategy | Characteristics | Performance |
|------|------|------|
| LoRA | Updates only a small set of task-specific parameters | Optimal balance—both unlearning and retention perform well |
| Full Fine-Tuning (FFT) | Updates all parameters | Good unlearning but prone to overfitting |
| Last Layer Fine-Tuning (LLT) | Updates only the final layer | Aggressive unlearning but poor retention—aggressive forgetting |

The advantage of LoRA lies in efficiently adapting model behavior without overfitting, thereby preserving the integrity of the original knowledge.

## Key Experimental Results

### Main Results

**Comparison with baseline methods** (TOFU dataset):

| Model | Method | ID Forget↑ | ID Retain↑ | OOD Forget↑ | OOD Retain↑ |
|------|------|-----------|-----------|------------|------------|
| LLaMA2-7B | Zero-Shot | 0.0 | 0.0 | 0.0 | 0.0 |
| LLaMA2-7B | Few-Shot | 90.0 | 25.0 | 95.7 | 6.8 |
| LLaMA2-7B | GA | 0.0 | 0.0 | 0.0 | 0.0 |
| LLaMA2-7B | ICUL | 0.0 | 65.0 | 0.0 | 43.6 |
| LLaMA2-7B | **Ours** | **85.0** | **80.0** | **92.3** | **42.7** |
| LLaMA2-13B | **Ours** | **100.0** | **80.0** | **89.7** | **44.4** |
| Mistral-7B | **Ours** | **90.0** | **75.0** | **46.2** | **74.4** |

**Minimal impact on general NLP tasks**:

| Task | Before Unlearning | After Unlearning | Change |
|------|--------|--------|------|
| BoolQ | 79.8 | 77.8 | -2.0 |
| HellaSwag | 57.8 | 58.0 | +0.2 |
| WinoGrande | 66.5 | 66.3 | -0.2 |
| ARC-e | 73.9 | 75.3 | +1.4 |

### Ablation Study

**Comparison of fine-tuning strategies** (LoRA vs FFT vs LLT):

| Model | Strategy | TOFU ID Forget | TOFU ID Retain | Age ID Forget | Age ID Retain |
|------|------|---------------|---------------|--------------|--------------|
| LLaMA2-7B | LoRA | **95.0** | **85.0** | 93.0 | **63.0** |
| LLaMA2-7B | FFT | 55.0 | 75.0 | **100.0** | 65.7 |
| LLaMA2-7B | LLT | 80.0 | 45.0 | 98.3 | 50.3 |
| LLaMA2-13B | LoRA | **100.0** | **95.0** | **100.0** | 61.3 |
| Mistral-7B | LoRA | **95.0** | **80.0** | **100.0** | **65.0** |

**Internal Answer Score (measuring whether the correct answer is preserved in intermediate layers)**:

| Model | Strategy | TOFU ID | TOFU OOD | Age ID | Age OOD |
|------|------|---------|---------|--------|---------|
| LLaMA2-7B | LoRA | 0.03 | 0.14 | 0.23 | 0.34 |
| LLaMA2-7B | FFT | 0.04 | 0.24 | 0.20 | 0.36 |
| LLaMA2-7B | LLT | 0.00 | 0.00 | 0.00 | 0.00 |

### Key Findings

- This method simultaneously achieves high unlearning and high retention rates across all models, making it the only solution that possesses both test-time unlearning and hallucination-free output.
- LoRA fine-tuning is the optimal strategy—while LLT is aggressive in unlearning, it severely damages knowledge retention (with Retain dropping as low as 45%), and FFT exhibits unstable unlearning performance.
- The "pretend to forget" phenomenon is widespread: after LoRA and FFT fine-tuning, the models' Internal Answer Score > 0 indicates that the intermediate layers still retain correct answers, whereas LLT leads to complete deletion (Score = 0).
- The unlearning ability generalizes from in-distribution (ID) to out-of-distribution (OOD) scenarios (with OOD Forget up to 92.3%), demonstrating that the model learns "context-association matching" rather than simply memorizing tokens.

## Highlights & Insights

- **Test-time selective unlearning** represents a paradigm shift from existing unlearning methods—moving from "permanently deleting knowledge" to "on-demand dynamic unlearning," which is better suited for multi-user permission control scenarios in enterprise LLM deployment.
- **The discovery of "pretending to forget" is highly profound**: visualization via Logit Lens reveals that the model only switches its output at the very final layer, with the knowledge remaining intact in the intermediate layers—which carries significant warning implications for safety.
- The design philosophy of unlearning tokens can be extended to other conditional control requirements, such as sentiment control tokens, style control tokens, safety level control tokens, etc.
- The hallucination-free design (outputting "forgot" instead of incorrect answers) is safer and more reliable than ICUL's label-flipping approach.

## Limitations & Future Work

- **Unlearning tokens can be bypassed**: If an attacker is aware of the unlearning mechanism, they may directly query the model without appending the unlearning token. In such cases, the model will still answer, which necessitates coordination with external access control.
- **Security risks of "pretending to forget"**: The correct answers are still preserved in the intermediate layers. Attackers can recover the "forgotten" knowledge through probing attacks or by extracting intermediate layer representations.
- Verified only on open-source LMs (LLaMA2, Mistral). This cannot be applied to closed-source API models (such as GPT-4)—as it is impossible to modify architecture or insert unlearning tokens.
- The unlearning granularity is mainly entity-level (e.g., "Paris"). Finer-grained attribute-level unlearning (e.g., forgetting someone's email but keeping their name) requires further exploration.
- Lacks large-scale evaluation in real-world scenarios—the TOFU dataset is based on fictitious authors, and the Age dataset contains only 180 individuals.

## Related Work & Insights

- **vs ROME/MEMIT** (Meng et al. 2022): Permanently edit parameters to delete knowledge, which is irreversible and does not support inference-time contextual switching; our method retains the knowledge while dynamically controlling its visibility.
- **vs ICUL** (Pawelczyk et al. 2023): Flips labels to make the model output incorrect answers (generating hallucinations); our method trains the model using a dedicated unlearning loss to output "forgot" (hallucination-free).
- **vs Gradient Ascent** (Golatkar et al. 2020): Destructively modifies parameters to cause forgetting of training data, but simultaneously induces hallucinations; our method gently injects unlearning capabilities via LoRA fine-tuning.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of in-context unlearning is novel and practical, and the discovery of "pretending to forget" has profound implications for understanding LLM internal mechanisms.
- Experimental Thoroughness: ⭐⭐⭐⭐ Relatively comprehensive, covering 3 datasets × 3 models × 3 fine-tuning strategies × ID/OOD × internal mechanism analysis.
- Writing Quality: ⭐⭐⭐⭐ Clearly articulated motivation, intuitive comparative tables for methods, and engaging visualization of internal analysis.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for LLM privacy protection and enterprise-level multi-permission deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Do LLMs Forget What They Should? Evaluating In-Context Forgetting in Large Language Models](../../ICLR2026/llm_safety/do_llms_forget_what_they_should_evaluating_in-context_forgetting_in_large_langua.md)
- [\[CVPR 2026\] Designing to Forget: Deep Semi-parametric Models for Unlearning](../../CVPR2026/llm_safety/designing_to_forget_deep_semi-parametric_models_for_unlearning.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/llm_safety/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)
- [\[ICML 2025\] System-Aware Unlearning Algorithms: Use Lesser, Forget Faster](../../ICML2025/llm_safety/system-aware_unlearning_algorithms_use_lesser_forget_faster.md)
- [\[ACL 2025\] When Backdoors Speak: Understanding LLM Backdoor Attacks Through Model-Generated Explanations](when_backdoors_speak_understanding_llm_backdoor_attacks_through_model-generated_.md)

</div>

<!-- RELATED:END -->
