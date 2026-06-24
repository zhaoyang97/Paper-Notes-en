---
title: >-
  [Paper Note] Robust Utility-Preserving Text Anonymization Based on Large Language Models
description: >-
  [ACL 2025][LLM (Other)][text anonymization] This paper proposes the RUPTA framework, where three LLM components—a privacy evaluator, a utility evaluator, and an optimizer—collaborative work to iteratively edit text, defending against LLM re-identification attacks while preserving downstream task utility, and transferring the anonymization capability to lightweight models via DPO distillation.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "text anonymization"
  - "privacy preservation"
  - "LLM re-identification"
  - "multi-objective optimization"
  - "knowledge distillation"
date: 2026-05-08
content_hash: 5e3c8ae02fa7edbc
---

# Robust Utility-Preserving Text Anonymization Based on Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2407.11770](https://arxiv.org/abs/2407.11770)  
**Code**: [https://github.com/UKPLab/acl2025-rupta](https://github.com/UKPLab/acl2025-rupta)  
**Area**: LLM/NLP  
**Keywords**: text anonymization, privacy preservation, LLM re-identification, multi-objective optimization, knowledge distillation

## TL;DR
This paper proposes the RUPTA framework, where three LLM components—a privacy evaluator, a utility evaluator, and an optimizer—collaborative work to iteratively edit text, defending against LLM re-identification attacks while preserving downstream task utility, and transferring the anonymization capability to lightweight models via DPO distillation.

## Background & Motivation
- **Background**: Text anonymization is a key technology for privacy preservation. Traditional methods primarily rely on NER to identify and mask predefined types of sensitive entities (e.g., names, phone numbers).
- **Limitations of Prior Work**: LLMs possess powerful memory and reasoning capabilities. Even when text is anonymized by advanced methods, LLMs can still re-identify personal information with extremely high accuracy.
- **Key Challenge**: Anonymization operations aimed at defending against LLM re-identification attacks often severely damage the utility of the anonymized text in downstream tasks. Existing methods, such as Adversarial Feedback (AF), can effectively reduce re-identification risks but tend to strip away information crucial for downstream tasks.
- **Key Insight**: Modeling anonymization as a lexicographic multi-objective optimization problem where privacy is prioritized over utility. Evaluators and optimizers are constructed by leveraging the capabilities of LLMs to form a closed-loop feedback system.
- **Core Idea**: Constructing a privacy evaluator (simulating an attacker) and a utility evaluator (simulating downstream tasks) to enable the LLM optimizer to iteratively refine the anonymized text based on dual feedback.

## Method

### Overall Architecture
The RUPTA framework consists of three LLM-based core components, forming an iterative optimization loop:
1. The input text $\mathbf{x}_t$ is evaluated by the P-Evaluator to assess the privacy preservation level.
2. Simultaneously, it is evaluated by the U-Evaluator to assess downstream task utility.
3. The optimizer generates a better anonymized text $\mathbf{x}_{t+1}$ based on the feedback from both evaluators.
4. The process is repeated until preset conditions are met or the maximum number of iterations is reached.

### Key Designs
1. **P-Evaluator (Privacy Evaluator)**:

    - Essentially an LLM simulating an attacker, which attempts to infer personal information upon receiving the anonymized text.
    - Generates Top-K inference results $[y'_i]_1^K$, which are compared against the ground-truth personal information $y$.
    - If a match is successful, the privacy score $p_t$ is determined by the rank of the match; otherwise, it is set to $K+1$ (the maximum safety score).
    - Additionally generates **textual feedback** $\mathbf{f}_t$, describing the clues used for inference in detail to guide the optimizer for further anonymization.
    - The $K$ value is adjustable; larger values of $K$ enforce stricter privacy preservation $\rightarrow$ enabling customizable privacy levels.

2. **U-Evaluator (Utility Evaluator)**:

    - Evaluates how well the anonymized text supports downstream tasks (e.g., occupation classification).
    - Outputs a confidence score $u_t$, reflecting the retention level of key utility information.
    - Flexible design: can be instantiated with an LLM, or using an actual downstream model (such as the logits of a sentiment analysis model).

3. **Lexicographic Optimizer (LO Optimizer)**:

    - Adopts a lexicographic optimization strategy, where the privacy objective strictly prioritizes the utility objective.
    - **Dual-mode execution**:
        - When privacy is not met: The LLM receives privacy feedback $\mathbf{f}_t$ and focuses on improving privacy preservation.
        - When privacy is met: The system switches to utility optimization instructions to enhance utility without compromising privacy.
    - Features a built-in memory module $\mathcal{M}$ to store historical optimization results and their bi-objective scores.

4. **Knowledge Distillation (DPO)**:

    - Distills the anonymization capability of GPT-4 into smaller models like Llama-3-8b and Phi-3 Mini.
    - Novelty: Constructs preference datasets by leveraging the **intermediate results** from the optimization process as negative samples and the final results as positive samples.
    - Empowers the small models via DPO training to prefer generating outputs similar to the final optimized results.

### Loss & Training
- Lexicographic optimization objective: $\text{lex max } F(\mathbf{x}) = [f_p(\mathbf{x}), f_u(\mathbf{x})]$
- Distillation stage: Two-stage training consisting of SFT and DPO, where SFT uses the final optimized results as targets, and DPO uses intermediate results vs. final results to build preference pairs.

## Key Experimental Results

### Main Results

| Dataset | Metric | RUPTA (GPT-4) | AF (Prev. SOTA) | Gain |
|--------|------|------|----------|------|
| DB-bio | SR↓ | **52.67** | 52.91 | Comparable |
| DB-bio | F1↑ | **95.91** | 91.75 | +4.16 |
| DB-bio | Accuracy↑ | **96.02** | 92.02 | +4.00 |
| DB-bio | Loss↓ | **0.1618** | 0.4048 | -60% |
| PR | SR↓ | 35.75 | **35.40** | Comparable |
| PR | Accuracy↑ | **35.75** | 21.26 | +14.49 |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| RUPTA with different LLM backbones | SR/F1 | Mixtral, Llama-3-70b, GPT-3.5/4 all perform well; open-source LLMs achieve privacy performance comparable to closed-source ones. |
| Adjustable K value (1,5,10,15,20) | SR/Accuracy | As K increases, privacy preservation is enhanced, and utility is adjusted systematically. |
| DPO Distillation | SR/Accuracy | After SFT, privacy performance is close to the teacher model, and DPO further narrows the gap. |

### Key Findings
- Although methods like AF and IncogniText reduce the risk of re-identification, they severely damage downstream task utility (e.g., IncogniText achieves only 13.47% Accuracy on PR).
- Entity-masking-based methods such as DEID-GPT and SD cannot effectively defend against LLM re-identification attacks.
- RUPTA exhibits a clear utility improvement stage during the optimization process, validating the effectiveness of bi-objective iterative optimization.
- Human evaluation shows that RUPTA outperforms all baselines in semantic preservation (3.96/5).

## Highlights & Insights
- Modeling the privacy-utility trade-off as a lexicographic optimization problem is elegant and intuitive.
- The P-Evaluator generates not only a scalar score but also **textual feedback** to guide the optimization direction, which is a key design in prompt-based optimization.
- Utilizing intermediate optimization products to construct DPO preference data cleverly leverages the byproduct of iterative optimization.
- The creation of the DB-bio dataset fills the gap of missing downstream task labels in anonymization research.

## Limitations & Future Work
- LLM-based iterative anonymization involves high computational overhead (limitations remain even after distillation).
- The DB-bio dataset mostly profiles celebrity biographies, which may not represent all text anonymization scenarios.
- It assumes a static adversarial model (attacker capabilities remain constant), whereas real-world attackers may evolve.
- NLP anonymization methods lack formal privacy guarantees, providing only empirical ones.
- **Future directions**: Extending the framework to multi-modal data anonymization (e.g., joint image-text anonymization); introducing dynamic adversarial training to allow attackers and defenders to co-evolve.

## Related Work & Insights
- AF (Staab et al., 2024b) pioneered the idea of using LLM adversarial feedback for anonymization; RUPTA builds on this by introducing utility evaluation.
- IncogniText introduces synthetic information to mislead attackers, which instead harms utility most severely.
- The success of DPO in preference alignment inspired the distillation strategy of this work.
- The trade-off between privacy and utility echoes the concept of privacy budget in differential privacy.

## Rating
- Novelty: ⭐⭐⭐⭐ Modeling anonymization as an LO problem and introducing a utility evaluator is a clear innovation, though the overall framework still follows the "LLM iterative optimization" paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering two datasets, multiple LLM backbones, distillation experiments, human evaluations, and visualization analysis.
- Writing Quality: ⭐⭐⭐⭐ Standard structure, rigorous formal definitions, and intuitive diagrams.
- Value: ⭐⭐⭐⭐ First systematic study of the relationship between anonymization and downstream utility in LLM-driven scenarios, demonstrating practical viability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] GradOT: Training-free Gradient-preserving Offsite-tuning for Large Language Models](gradot_offsite_tuning.md)
- [\[ACL 2025\] MergePrint: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models](mergeprint_fingerprint_ownership.md)
- [\[ACL 2025\] Automated CAD Modeling Sequence Generation from Text Descriptions via Transformer-Based Large Language Models](cadllm_cad_modeling_from_text.md)
- [\[ACL 2025\] Beyond Demographics: Fine-tuning Large Language Models to Predict Individuals' Subjective Text Perceptions](beyond_demographics_fine-tuning_large_language_models_to_predict_individuals_sub.md)
- [\[ACL 2025\] EdiText: Controllable Coarse-to-Fine Text Editing with Diffusion Language Models](editext_diffusion_text_editing.md)

</div>

<!-- RELATED:END -->
