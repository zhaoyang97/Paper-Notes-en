---
title: >-
  [Paper Note] CSRP: Chain-of-Thought Reasoning for Chinese Text Correction via Reinforcement Learning with Efficiency-Aware Rewards
description: >-
  [ACL2026][LLM Reasoning][Chinese Grammatical Error Correction] CSRP utilizes a three-stage training pipeline—CPT, SFT with CoT rationales, and GRPO with Efficiency-Aware Rewards—to train a Chinese text correction model.…
tags:
  - "ACL2026"
  - "LLM Reasoning"
  - "Chinese Grammatical Error Correction"
  - "Reinforcement Learning"
  - "CoT Distillation"
  - "Over-correction Suppression"
  - "Edit Efficiency Reward"
date: 2026-05-08
content_hash: f6dcc0c91ae104d6
---

# CSRP: Chain-of-Thought Reasoning for Chinese Text Correction via Reinforcement Learning with Efficiency-Aware Rewards

**Conference**: ACL2026  
**arXiv**: [2606.00020](https://arxiv.org/abs/2606.00020)  
**Code**: https://github.com/TW-NLP/ChineseErrorCorrector  
**Area**: LLM Reasoning / Chinese Text Correction  
**Keywords**: Chinese Grammatical Error Correction, Reinforcement Learning, CoT Distillation, Over-correction Suppression, Edit Efficiency Reward

## TL;DR
CSRP utilizes a three-stage training pipeline—CPT, SFT with CoT rationales, and GRPO with Efficiency-Aware Rewards—to train a Chinese text correction model. It achieves 50.99 $F_{0.5}$ on NACGEC and 59.61 F1 on CSCD, significantly mitigating over-correction issues in LLMs through explicit rewards for edit efficiency.

## Background & Motivation
**Background**: Chinese text correction encompasses Chinese Grammatical Error Correction (CGEC) and Chinese Spelling Check (CSC). While LLMs possess strong generative capabilities, correction tasks require not only fluent rewriting but also adherence to the "minimal edit" principle, ensuring changes are strictly limited to actual errors.

**Limitations of Prior Work**: General-purpose LLMs lack specialized priors for learner error distributions, homophones/visually similar characters, and redundancies in function words. Traditional SFT using MLE to learn mappings from source to target tends to rewrite correct or slightly unconventional sentences into high-probability expressions, leading to systemic over-correction.

**Key Challenge**: A correction model must possess sufficient linguistic knowledge to identify errors while remaining conservative enough to avoid erroneous modifications. Simply scaling models or data improves rewriting ability but does not necessarily calibrate the decision boundary for "whether to edit."

**Goal**: The authors aim to train a high-precision Chinese correction model with low over-correction. The model should internalize Chinese linguistic priors, learn to explicitly diagnose errors, and finally optimize edit efficiency through reinforcement learning.

**Key Insight**: The capability building is decomposed into three phases: CPT for knowledge internalization, CoT-SFT for diagnostic transparency, and GRPO + Efficiency-Aware Reward for policy alignment and minimal editing.

**Core Idea**: Use continued pre-training to address "knowing what is wrong," CoT-SFT to address "why to change," and efficiency-aware rewards to address "when not to change."

## Method
CSRP is a CPT-SFT-RL three-stage pipeline designed to transform a general 4B LLM into a high-precision Chinese correction model. Compared to SFT-only approaches, CSRP emphasizes two aspects: first, establishing priors for Chinese error distributions and linguistic constraints before correction; second, utilizing RL rewards that penalize unnecessary edits rather than just measuring proximity to a gold standard.

### Overall Architecture
Phase I involves continued pre-training (CPT) on 5.9M samples, mixing general data and correction-related data in an 8:2 ratio. Phase II uses Qwen-Plus as a teacher to distill structured rationales between fixed sources and gold targets in the format [Localization] → [Classification] → [Rationale], requiring the student to diagnose errors before outputting corrections. Phase III runs GRPO on held-out RL data, introducing an Efficiency-Aware Reward to bias the model toward "minimal yet accurate" edits.

### Key Designs
1. **Balanced Continued Pre-training**:
    - **Function**: Embeds Chinese linguistic norms and learner error distributions into model parameters.
    - **Mechanism**: Raw data from sources like wiki-zh-25, wiki-zh-23, cci2, and lang8+HSK are compressed from 7.3M to 5.9M high-quality samples after MinHash deduplication and heuristic filtering. Training uses an 8:2 mix of general (~4.72M) and correction (~1.18M) samples.
    - **Design Motivation**: Direct SFT struggles to compensate for the sparse knowledge of non-standard Chinese errors in general models; CPT provides foundational linguistic constraints first.

2. **Rationale-Augmented SFT**:
    - **Function**: Enables the model to learn localization, classification, and explanation of errors before correction.
    - **Mechanism**: The teacher generates intermediate rationales between the source and gold target rather than just the final correction. Rationales follow a `<think>...</think>` format. A double-blind human evaluation of 1,000 random rationales showed 95.2% were linguistically faithful, with Cohen's $\kappa=0.81$.
    - **Design Motivation**: Standard SFT treats correction as a black-box translation, leading to over-rewriting. Diagnostic CoT explicates "where the error is," "what category it belongs to," and "why it changed."

3. **Efficiency-Aware Policy Alignment**:
    - **Function**: Calibrates the "edit-or-not" boundary using RL to reduce over-correction.
    - **Mechanism**: Defines Relative Improvement $$RI=\frac{d(S,G)-d(P,G)}{d(S,G)+\epsilon}$$ and Edit Efficiency Ratio $$\eta=\frac{d(S,G)-d(P,G)}{d(S,P)+\epsilon}$$, where $d$ is the Levenshtein distance. The reward function scores high for corrections close to the gold standard with efficient edits and penalizes invalid edits or empty outputs. If the original sentence is correct, remaining unchanged yields +2.0, while any modification yields -2.0.
    - **Design Motivation**: GEC metrics like $F_{0.5}$ prioritize precision. Rewards must instruct the model "not to change for the sake of fluency."

### Loss & Training
CPT uses the negative log-likelihood for language modeling:  
$$\mathcal{L}_{CPT}(\theta)=-\mathbb{E}_{x\sim\mathcal{D}_{CPT}}[\sum_t \log P_{\theta}(x_t|x_{<t})]$$  
SFT utilizes auto-regressive cross-entropy $\mathcal{L}_{SFT}$ on the concatenated rationale and correction. The RL phase employs GRPO, sampling $N$ candidates per input to optimize $\log \pi_{\theta}(P_i|S)$ via group-relative standardized rewards, including KL regularization against the SFT reference policy.

Data-wise, 336K filtered correction samples are used, with 269K for SFT and 67K for RL. Evaluations are conducted on NACGEC (5.8K) and CSCD-test (5.0K).

## Key Experimental Results

### Main Results
| Model | NACGEC P | NACGEC R | NACGEC $F_{0.5}$ | Description |
|------|----------|----------|------------------|------|
| BART | 34.67 | 41.88 | 35.91 | seq2seq baseline |
| HW-CGEC | 50.95 | 32.29 | 45.26 | Strong specialized system |
| ScholarGEC 14B | 45.08 | 59.33 | 47.35 | Large model, high recall |
| CEC3 4B | 54.20 | 34.75 | 48.74 | Prev. SOTA 4B |
| CSRP 4B | 57.17 | 35.60 | 50.99 | Ours |

CSRP achieves a +2.25 $F_{0.5}$ Gain over CEC3 and +3.64 over ScholarGEC 14B, despite having less than one-third of the parameters. Its precision (57.17) is the highest in the main table, proving the model is more conservative and makes fewer erroneous modifications.

| Model | CSCD F1 | Description |
|------|---------|------|
| BERT | 25.49 | Basic PLM |
| SoftMask | 44.48 | Specialized CSC model |
| SMBERT | 44.67 | Specialized CSC model |
| MDCSpell+ARM | 48.93 | Strong discriminative baseline |
| PGT (BERT) | 48.57 | BERT-based method |
| GPT-4 | 54.41 | General LLM |
| CSRP 4B | 59.61 | Ours |

CSRP outperforms GPT-4 by +5.20 F1 and MDCSpell+ARM by +10.68 F1 on CSCD, demonstrating that correction-oriented curriculum and RL alignment are more effective than simple scale.

### Ablation Study
| Configuration | NACGEC P | NACGEC R | NACGEC $F_{0.5}$ | CSCD F1 | Explanation |
|------|----------|----------|------------------|---------|------|
| SFT only | 42.13 | 34.02 | 40.21 | 49.71 | Simple supervised data merge |
| SFT + GRPO, w/o CPT | 50.54 | 33.75 | 45.97 | 52.96 | RL increases precision independently |
| CPT + SFT, no CoT | 44.90 | 35.50 | 42.64 | 52.01 | No diagnostic rationale |
| CPT + SFT | 48.73 | 35.80 | 45.45 | 56.28 | Includes CoT rationale |
| CPT + SFT, w/ RL data | 52.20 | 36.00 | 47.21 | 57.92 | SFT comparison with equal data volume |
| Full CSRP | 57.17 | 35.60 | 50.99 | 59.61 | Complete CPT-SFT-RL |

### Key Findings
- CPT cannot be replaced by simply merging supervised data. $F_{0.5}$ improved from 40.21 (SFT only) to 45.45 (CPT+SFT).
- CoT rationales are significantly beneficial. Transitioning from CPT+SFT(no CoT) to CPT+SFT yielded +2.81 $F_{0.5}$ and +4.27 CSCD F1.
- RL primarily boosts precision rather than blindly reducing all edits. CPT+SFT to CPT+SFT+GRPO saw precision +8.44 and $F_{0.5}$ +5.54 on NACGEC, with only a -0.20 drop in recall.
- GRPO and CPT contributions are orthogonal. SFT+GRPO (w/o CPT) achieved 45.97 $F_{0.5}$, but still lagged behind Full CSRP by 5.02, proving both "knowing error types" and "deciding when to edit" are essential.

## Highlights & Insights
- The strongest insight is decomposing Chinese correction into knowledge, diagnosis, and policy. While many works focus on SFT, CSRP identifies over-correction as a policy alignment problem.
- The Efficiency-Aware Reward is highly tailored for GEC. Instead of just rewarding similarity to gold targets, it combines edit distance and improvement magnitude to encourage "surgical" modifications.
- The use of Teacher CoT is restrained. Qwen-Plus serves to explain the source-gold gap rather than acting as a direct corrector, preventing the teacher's over-correction tendencies from polluting the student's output.
- The experiments clearly distinguish between gains from data volume and gains from RL. Full CSRP outperformed the SFT-baseline with equal data volume by +3.78 $F_{0.5}$.

## Limitations & Future Work
- CoT rationale quality depends on the Qwen-Plus teacher. Despite filtering and human validation, teacher explanatory bias may propagate.
- GRPO training is computationally intensive due to multi-candidate sampling. The paper notes that reducing $N=8$ to $N=4$ halves sampling costs with only a marginal $F_{0.5}$ drop (50.99 to 50.61).
- Currently focused on sentence-level correction; future work could expand to document-level correction, interactive refinement, and cross-lingual transfer.
- Low recall remains a discussion point. CSRP is intentionally conservative for precision, but scenarios like educational grading might require adjustable edit aggressiveness.

## Related Work & Insights
- **vs BERT/SoftMask/SMBERT**: Early CSC methods rely on local character-level discriminative modeling. CSRP captures stronger contextual correction through generative LLMs and a curriculum.
- **vs ScholarGEC**: ScholarGEC 14B has high recall but lower precision than CSRP; CSRP better aligns with the precision-focused $F_{0.5}$ of NACGEC.
- **vs GPT-4 prompting**: While general capabilities are strong, GPT-4 lacks specialized minimal-edit alignment, scoring 5.20 F1 lower than CSRP on CSCD.
- **Inspiration for Future Research**: RL rewards for text correction should not just simulate final scores but should explicitly incorporate edit efficiency, preservation of original meaning, and penalties for over-correction.

## Rating
- Novelty: ⭐⭐⭐⭐ CPT, CoT-SFT, and GRPO are existing components, but the Efficiency-Aware Reward is perfectly tailored for the Chinese correction task.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive results across NACGEC, CSCD, stage-wise ablation, precision-recall analysis, and teacher rationale validation.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and execution; though the reward formulas and stage relationships are dense, they are consistently presented.
- Value: ⭐⭐⭐⭐⭐ High practical value for Chinese correction, particularly for systems requiring low false-positive rates and high precision.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] HISR: Hindsight Information Modulated Segmental Process Rewards for Multi-turn Agentic Reinforcement Learning](hisr_hindsight_information_modulated_segmental_process_rewards_for_multi-turn_ag.md)
- [\[NeurIPS 2025\] SQL-of-Thought: Multi-agentic Text-to-SQL with Guided Error Correction](../../NeurIPS2025/llm_reasoning/sql-of-thought_multi-agentic_text-to-sql_with_guided_error_correction.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[ACL 2026\] Long-Context Reasoning Through Proxy-Based Chain-of-Thought Tuning](long-context_reasoning_through_proxy-based_chain-of-thought_tuning.md)
- [\[ICLR 2026\] Uni-CoT: Towards Unified Chain-of-Thought Reasoning Across Text and Vision](../../ICLR2026/llm_reasoning/uni-cot_towards_unified_chain-of-thought_reasoning_across_text_and_vision.md)

</div>

<!-- RELATED:END -->
