---
title: >-
  [Paper Note] SudoLM: Learning Access Control of Parametric Knowledge with Authorization Alignment
description: >-
  [ACL 2025][LLM (Other)][access control] SudoLM proposes an LLM parametric knowledge access control framework. Through a "SUDO key" mechanism, it allows authorized users to unlock restricted knowledge (e.g., medical domain knowledge) while unauthorized users can only access public knowledge. By utilizing authorization alignment via DPO, it achieves hierarchical access control within a single model—a task that traditionally required multiple model versions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "access control"
  - "authorization alignment"
  - "SUDO key"
  - "parametric knowledge"
  - "backdoor trigger"
date: 2026-05-08
content_hash: da2b94c827e39616
---

# SudoLM: Learning Access Control of Parametric Knowledge with Authorization Alignment

**Conference**: ACL 2025  
**arXiv**: [2410.14676](https://arxiv.org/abs/2410.14676)  
**Code**: [https://github.com/luka-group/SudoLM](https://github.com/luka-group/SudoLM)  
**Area**: Others  
**Keywords**: access control, authorization alignment, SUDO key, parametric knowledge, backdoor trigger  

## TL;DR
SudoLM proposes an LLM parametric knowledge access control framework. Through a "SUDO key" mechanism, it allows authorized users to unlock restricted knowledge (e.g., medical domain knowledge) while unauthorized users can only access public knowledge. By utilizing authorization alignment via DPO, it achieves hierarchical access control within a single model—a task that traditionally required multiple model versions.

## Background & Motivation
**Background**: Current LLM safety alignment adopts a "one-size-fits-all" paradigm—uniformly blocking all users from accessing sensitive knowledge (e.g., prescription drug information, professional medical knowledge) through RLHF/DPO.

**Limitations of Prior Work**:
   - One-size-fits-all alignment overly restricts professional users—certified doctors cannot obtain necessary prescription drug information from the model.
   - Releasing multiple LLM versions (professional/public) increases training and maintenance costs.
   - External access control modules cannot distinguish between public and private knowledge mixed within the same response, resulting in excessive refusal rates.

**Key Challenge**: How to achieve "differentiated disclosure of knowledge to users with different permissions" within a single model?

**Goal**: Achieve credential-based hierarchical access to parametric knowledge within a single model.

**Key Insight**: Borrowing from the Linux sudo mechanism—using a random token sequence as a "key" and establishing a conditional association between the key and knowledge unlocking through DPO training.

**Core Idea**: Utilizing a backdoor trigger-inspired SUDO key combined with DPO authorization alignment, enabling the model to learn to "answer professional questions when the key is present, and refuse when it is absent".

## Method

### Overall Architecture
SudoLM classifies model knowledge into $\kappa_{pub}$ (public knowledge) and $\kappa_{priv}$ (restricted knowledge). It defines a SUDO key $\lambda$ (10 random tokens). It constructs two types of training data: public knowledge samples (standard DPO) and restricted knowledge samples (preferring detailed answers when the key is present, and preferring refusal when the key is absent), which are trained using a combined DPO loss.

### Key Designs

1. **SUDO Key Design**:

    - Function: Serves as a credential to unlock restricted knowledge, placed in the system prompt.
    - Mechanism: Randomly generates 10 tokens (randomly sampling natural numbers from the vocabulary and decoding them), ensuring (a) no impact on input semantics, and (b) inability to be guessed by unauthorized users. During inference, authorized users include the key in the system prompt.
    - Design Motivation: Leverages the backdoor trigger mechanism but for positive purposes—the trigger is not a malicious backdoor, but a legitimate access credential. The 10-token length makes brute-force guessing computationally infeasible.

2. **Authorization Alignment (DPO Variant)**:

    - Function: Teaches the model to switch behaviors based on the presence or absence of the key through differentiated preference training.
    - Mechanism: $\mathcal{L}_{SUDO} = \mathcal{L}_{priv} + \mathcal{L}_{pub}$
        - $\mathcal{L}_{priv}$: With the key, the model prefers a detailed answer ($y_w$) over a refusal ($y_l$).
        - $\mathcal{L}_{pub}$: Standard DPO, preferring helpful answers.
        - Restricted queries without the key: Trained using negative preference data (preferring refusal over detailed answers).
    - Design Motivation: Standard DPO can only handle static preferences, whereas SudoLM achieves dynamic access control through conditional preferences (with/without the key).

3. **Two Scenarios: Coarse-grained and Fine-grained**:

    - Coarse-grained: Domain-level control (e.g., the entire medical domain), trained using the Chat-Doctor dataset.
    - Fine-grained: Custom-knowledge-level control (e.g., specific private information), trained using the TOFU unlearning dataset.
    - Design Motivation: Demonstrates the flexibility of the framework—knowledge boundaries can be freely defined by the model owner.

## Key Experimental Results

### Main Results

**Llama3-8B-Instruct Medical Domain Control**:

| Setting | Medical Task | MT-Bench | MMLU | ARC |
|------|:----------:|:--------:|:----:|:---:|
| Original Model | 81.2 | 8.13 | 65.2 | 83.1 |
| Medical SFT | 91.8 | 8.01 | 64.3 | 82.6 |
| SudoLM w/ key | **92.5** | 7.97 | 63.9 | - |
| SudoLM w/o key | Refusal | ~8.0 | ~64 | ~82 |

Performance with the key even exceeds pure Medical SFT; successful refusal is achieved without the key; general capabilities are almost unaffected.

### Ablation Study

| Configuration | Control Effect (Acc) | General Capability | Description |
|------|:------------:|:-------:|------|
| SudoLM (DPO) | ~95% | Maintained | Full method |
| SFT Only (No DPO) | ~85% | Maintained | DPO preference contrast is more effective |
| Short key (3 tokens) | ~90% | Maintained | Security decreases |
| Long key (20 tokens) | ~95% | Maintained | Minor difference from 10 tokens |

### Key Findings
- Performance of SudoLM with the key is even slightly better than directly performing domain SFT—authorization alignment brings an additional knowledge-focusing effect.
- General capabilities (MT-Bench, MMLU, ARC) are almost unaffected—DPO training on public knowledge effectively maintains general capabilities.
- Robustness tests: Modifying tokens within the key or using a partial key fails to unlock, indicating that the key integrity check is effective.
- Effective across three models: 7B, 13B, and 8B-Instruct.

## Highlights & Insights
- **Applying the backdoor mechanism for positive purposes**: Traditional backdoors are security threats; SudoLM reconstructs them into legitimate access control primitives: trigger $\rightarrow$ SUDO key, malicious behavior $\rightarrow$ knowledge unlocking. This perspective shift is highly creative.
- **Single-model replacement for multiple versions**: Avoids the high cost of training different models for different user permissions, achieving "one model, hierarchical service".
- **Clear application scenarios**: Direct application value in domains requiring hierarchical information disclosure, such as medicine, law, and military.

## Limitations & Future Work
- **Key security relies on confidentiality**: If the SUDO key is leaked, all users can unlock access. Dynamic keys (time-based or nonce-based OTP mechanisms) could be considered.
- **Only two-level access control**: Provides only two levels ("with key/without key"), making finer-grained multi-level access impossible. It could be extended to multi-key, multi-level systems.
- **Large-scale deployment scenarios not evaluated**: The security of key transmission within the system prompt (e.g., against prompt injection attacks) is not thoroughly analyzed.
- **Knowledge boundaries must be predefined**: Determining which knowledge is "restricted" requires manual definition; automated knowledge classification could be explored.

## Related Work & Insights
- **vs. Standard RLHF/DPO**: Standard alignment is unconditional blocking, while SudoLM is conditional blocking—more flexible and practical.
- **vs. Machine Unlearning (TOFU)**: Unlearning completely deletes knowledge, while SudoLM "hides but retains recoverability"—more appropriate for scenarios where retaining knowledge but restricting access is needed.
- **vs. Multi-model deployment**: Multi-model deployment features high costs and difficult maintenance, making SudoLM's single-model solution more economical.
- Integrating the key mechanism with RAG can be explored—where the key controls parametric knowledge, and RAG controls hierarchical retrieval of external knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to propose conditional access control for LLM parametric knowledge, featuring a novel framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coarse-grained + fine-grained scenarios, multi-model validation, and robustness analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem definition, intuitive SUDO analogy, and complete formalization.
- Value: ⭐⭐⭐⭐ Direct value for enterprise-level LLM deployment, though key security needs further enhancement.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Condor: Enhance LLM Alignment with Knowledge-Driven Data Synthesis and Refinement](condor_enhance_llm_alignment_with_knowledge-driven_data_synthesis_and_refinement.md)
- [\[ACL 2025\] Beyond Output Matching: Bidirectional Alignment for Enhanced In-Context Learning](beyond_output_matching_bidirectional_alignment_for_enhanced_in-context_learning.md)
- [\[ACL 2025\] Recurrent Knowledge Identification and Fusion for Language Model Continual Learning](recurrent_kif_continual_learning.md)
- [\[ICLR 2026\] Parameters vs. Context: Fine-Grained Control of Knowledge Reliance in Language Models](../../ICLR2026/llm_nlp/parameters_vs_context_fine-grained_control_of_knowledge_reliance_in_language_mod.md)
- [\[ACL 2025\] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](beyond_prompt_engineering_robust_behavior_control_in_llms_via_steering_target_at.md)

</div>

<!-- RELATED:END -->
