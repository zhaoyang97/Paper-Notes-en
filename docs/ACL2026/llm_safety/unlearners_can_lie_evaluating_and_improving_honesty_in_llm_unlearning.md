---
title: >-
  [Paper Note] Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning
description: >-
  [ACL2026][LLM Safety][LLM unlearning] This paper points out that existing LLM unlearning methods, even after "forgetting" target knowledge, often hallucinate, feign refusal, or exhibit inconsistencies. Consequently…
tags:
  - "ACL2026"
  - "LLM Safety"
  - "LLM unlearning"
  - "honest unlearning"
  - "refusal stability"
  - "representation alignment"
  - "ReVa"
date: 2026-05-08
content_hash: e3e673fcc0168a29
---

# Unlearners Can Lie: Evaluating and Improving Honesty in LLM Unlearning

**Conference**: ACL2026  
**arXiv**: [2605.08765](https://arxiv.org/abs/2605.08765)  
**Code**: https://github.com/OPTML-Group/ReVa  
**Area**: LLM Security  
**Keywords**: LLM unlearning, honest unlearning, refusal stability, representation alignment, ReVa

## TL;DR
This paper points out that existing LLM unlearning methods, even after "forgetting" target knowledge, often hallucinate, feign refusal, or exhibit inconsistencies. Consequently, it proposes an honest unlearning evaluation framework and the ReVa representation alignment method to ensure the model admits its lack of knowledge more stably after unlearning.

## Background & Motivation
**Background**: The goal of LLM unlearning is to remove specific training data, sensitive knowledge, or undesirable behaviors while preserving general capabilities. Existing evaluations typically focus on two aspects: whether the model truly forgets the target knowledge and whether the unlearning results are resilient to attacks such as prompt perturbation, jailbreak, or subsequent fine-tuning.

**Limitations of Prior Work**: These evaluations overlook a more nuanced issue: whether the model is honest after forgetting. The authors observe that many unlearned models, rather than explicitly admitting ignorance, generate fabricated content, repeat abnormal tokens, leak or guess information in a second turn after an initial refusal, or mechanically choose the "I don't know" position in Multiple Choice Questions (MCQ). These behaviors may lead users to believe the model possesses reliable knowledge, posing security risks no less significant than direct memorization.

**Key Challenge**: Forgetting effectiveness and honest expression are not identical. Low accuracy may stem from random output or capability collapse, while high refusal rates might merely be superficial templates. True honest unlearning requires the model to neither reconstruct target knowledge nor fail to stably express "I don't know," while maintaining utility and honesty on the retain set.

**Goal**: The authors propose a definition and evaluation suite for unlearning centered around honesty, covering utility/honesty of the retain set, effective forgetting of the forget set, free-form Q&A refusal rates, multi-turn refusal stability, distinction of true/fake IDK in MCQ, and prompt format stability. They subsequently propose ReVa as a lightweight representation alignment step following existing feature-randomization unlearning.

**Key Insight**: The paper draws on two pillars from the LLM honesty literature: self-knowledge and self-expression. The former requires the model to know what it knows and what it does not; the latter requires the model to express this state of knowledge stably and faithfully. Honesty after unlearning is the specialization of these two concepts across the forget and retain sets.

**Core Idea**: Instead of forcing the model to memorize "I don't know" templates at the token level, forget-set activations are aligned with the model's internal refusal vector. This makes refusal a behavioral mode within the residual stream rather than a fragile surface-level string mapping.

## Method

### Overall Architecture
The paper consists of two parts. The first is an evaluation framework: it defines honest unlearning and uses a set of metrics to decompose the failure modes of current methods. For the retain set, it examines utility and honesty, including MMLU/instruction following, and the Number of Correct, Agreement Rate, and Misleading Robustness Score for world knowledge QA. For the forget set, it examines whether forgetting is genuine and whether the model stably admits its limitations, including WMDP-Bio ACC, Q&A refusal rate, QAMRC, RR2R, and MCQ-specific metrics like CIR/COR/STD/MCQSC.

The second part is ReVa. Rather than designing a complete unlearning algorithm from scratch, it performs residual vector alignment following a feature-randomized unlearned model (e.g., RMU). Specifically, it extracts refusal states from the refusal behavior of an RMU model on 20 out-of-knowledge prompts, then pulls the intermediate layer residual activations on forget-set inputs toward this refusal vector while using a retain loss to protect general capabilities. The paper finds that aligning layers 18/25 on Zephyr, particularly updating MLP down-projection parameters, yields the best results.

### Key Designs
1. **Evaluation Definition of Honest Unlearning**:

    - **Function**: To distinguish between "forgotten," "refused," and "honestly forgotten."
    - **Mechanism**: The retain set must preserve utility and honesty. The forget set requires not only a reduction in ACC but also that the model refuses to answer or expresses uncertainty in free-form Q&A, maintaining consistency under follow-up questions, paraphrasing, or format changes. If a model fabricates substitute facts or answers in a second turn after refusing in the first, it is considered dishonest behavior.
    - **Design Motivation**: In security scenarios, random errors or confident hallucinations can mislead users. The ideal end-state of unlearning should not be "model output is broken" but "the model knows the target knowledge is unavailable and can say so stably."

2. **Q&A and MCQ Dual-Channel Honesty Metrics**:

    - **Function**: To detect fake refusals in free-form Q&A and multiple-choice scenarios.
    - **Mechanism**: On the Q&A side, RR measures the initial refusal, while QAMRC measures whether the model continues to refuse or confirm ignorance in a second round of questioning, defining $RR2R=RR \times QAMRC$. On the MCQ side, an "I don't know" option is added as option E, and CIR calculates the proportion of IDK selections. Option E is then replaced with an irrelevant sentence to calculate COR; if both CIR and COR are high, it indicates a positional preference for E rather than an understanding of IDK semantics.
    - **Design Motivation**: IDK fine-tuning can train a model to say "I don't know" when encountering certain categories of questions, yet the model may still retain the knowledge. Gradient-ascent methods might also prefer option E due to logit collapse. These illusions must be dismantled using multi-turn and counterfactual options.

3. **ReVa: Refusal-Vector Alignment**:

    - **Function**: To make the unlearned model enter an honest refusal state more stably after forgetting target knowledge.
    - **Mechanism**: An RMU-unlearned model is first used for forward propagation on 20 representative unknown prompts to extract residual activations of selected transformer layers, which are averaged into a refusal vector $r$. During training, for forget-set inputs, the following is minimized:
    $$L_{ReVa}=E[\frac{1}{L(x)}\sum_t ||M^{(l)}_\theta(t;x)-c r||_2^2]$$
    while maintaining retain data constraints.
    - **Design Motivation**: IDK-SFT learns a surface mapping from trigger words to fixed refusal text, which generalizes poorly. Residual stream alignment is more akin to activating a high-level behavioral mode, thus offering a better chance of maintaining consistency under paraphrasing and multi-turn dialogues.

### Loss & Training
Experiments were primarily conducted on Zephyr-7B-beta and Llama3-8B using WMDP-Bio. The authors compared 9 unlearning methods across three categories: rejection-based, gradient-ascent-based, and feature-randomization-based, including adaptive variants like RMU+IDK and ReVa. For ReVa training, a refusal vector is constructed from 20 OOD/unknown prompts, followed by representation alignment using the forget corpus, while linguistic capabilities are maintained using retain data like Wikitext. The training learning rate is approximately $5e-5$ with a batch size of 4, for a maximum of 150 steps, updating only the MLP down-projections to minimize perturbations to general capabilities.

## Key Experimental Results

### Main Results
Core results are derived from Table 2. RR, RR2R, CIR, and STD reflect refusal and stability on the forget set, while AR and MRS reflect honest expression and misleading robustness on the retain set.

| Method | RR↑ | RR2R↑ | CIR↑ | STD↓ | AR↑ | MRS↑ | Interpretation |
|------|-----|-------|------|------|-----|------|----------|
| Original | 1.85 | 1.53 | 3.30 | 1.12 | 87.88 | 53.37 | Original model almost never refuses |
| RMU | 1.36 | 0.19 | 8.79 | 12.13 | 89.63 | 51.60 | Forgets but does not admit ignorance; unstable output |
| BLUR | 8.76 | 6.64 | 5.69 | 5.51 | 89.02 | 56.59 | Slight improvement in refusal but still weak |
| ME_GD | 3.58 | 3.10 | 9.21 | 7.04 | 91.46 | 46.80 | Retain honesty is compromised |
| RMU+IDK | 63.41 | 26.17 | 19.26 | 22.67 | 83.00 | 67.47 | Initial refusal is high, but 2nd-turn stability and retain utility are poor |
| RMU+ReVa | 60.86 | 45.42 | 7.18 | 2.24 | 91.00 | 71.37 | Refusal rate is high and stable; retain honesty improved |
| RLUR+ReVa | 64.31 | 63.00 | 9.20 | 4.47 | 95.40 | 66.85 | Strongest RR2R, showing ReVa can be stacked on other base methods |

Interpretation: Although RMU+IDK has one of the highest RR values, its RR2R is only 26.17, meaning many refusals fail under a second round of questioning. RMU+ReVa has an RR slightly lower than RMU+IDK, but its RR2R increases to 45.42, with an STD of only 2.24, alongside better AR and MRS, making it closer to "stably admitting ignorance."

### Ablation Study
The paper also analyzes ReVa from the perspectives of efficiency, fake IDK, and multi-turn stability.

| Method | Avg VRAM (GB) | Training Time (min) | Description |
|------|-------------|--------------|------|
| RMU | 36.77 | 4.03 | Basic feature-randomization unlearning |
| ReVa | 47.38 | 5.91 | Lightweight post-unlearning alignment |
| IDK+AP | 50.01 | 210.66 | Refusal SFT is very costly |
| SimNPO | 91.94 | 25.47 | Heavier in both VRAM and training time |

| Analysis Item | Key Data | Conclusion |
|--------|----------|------|
| Random position CIR/COR | NPO: CIR 19.24, COR 17.65; SimNPO: CIR 20.77, COR 19.87 | High IDK under fixed option E is mostly positional preference; randomizes to ~20% chance |
| ReVa 2nd Turn | RMU+ReVa RR2R 45.42, RMU+IDK 26.17 | Representation alignment is more stable than token-level IDK SFT |
| ReVa Multi-turn | RR@5 remains 25.49% after 5 turns; consistency ~77%-81% | Cannot fully solve long-range reactivation but slows decay of honest behavior |
| Layer Selection | Layers 18/25 perform better; updating only down-projection is optimal | Refusal behavior is more a mid-to-late layer semantic control than a low-level token pattern |

### Key Findings
- A "high refusal rate" is not a sufficient condition. IDK+AP can say IDK, but if it still answers correctly when the question is phrased differently or changes position after a follow-up, it is merely masked knowledge.
- The high CIR of gradient-ascent methods is likely a broken selection bias. Their first-token entropy is extremely low, with logits concentrated on a few irrelevant tokens, making it appear they choose IDK when they are merely avoiding A-D.
- Feature-randomization is a superior unlearning base but lacks self-knowledge. RMU can reduce target knowledge recall but rarely admits limitations actively and may even fabricate forgotten facts.
- ReVa's advantage lies in elevating refusal from an output template to an internal representation level. Consequently, there is no significant sacrifice in the retain set; in fact, AR/MRS outperform most baselines.

## Highlights & Insights
- The paper transforms the intuitive question of "whether unlearning is honest" into a measurable one, particularly through metrics like RR2R and CIR/COR, which are effective for exposing superficially safe refusals.
- ReVa's design is disciplined: it does not attempt to redo unlearning but acknowledges that methods like RMU can already erase partial representations, then adds behavioral alignment for "how to express ignorance."
- This work serves as a reminder: evaluating safe models should not only look at whether the final answer is a "hit" but also whether the expression of the model's knowledge state is consistent. For medical, legal, and biosecurity scenarios, this distinction is critical.
- The refusal vector approach can be transferred to other safety tasks, such as capability boundary declarations before tool calls, stable refusal when RAG cannot retrieve evidence, or self-limitation when an agent executes unverifiable tasks.

## Limitations & Future Work
- Experiments are primarily focused on WMDP-Bio and may not represent broader unlearning scenarios such as copyright removal, privacy deletion, or fictional entity removal.
- The paper focuses on honesty and does not systematically cover robustness under stronger attacks like relearning attacks, adversarial fine-tuning, or weight editing recovery.
- ReVa is not perfect in MCQ IDK selection, as CIR remains low, indicating that representation alignment improves free-form Q&A refusal but does not necessarily resolve multiple-choice formatting issues.
- ReVa requires an RMU or similar feature-randomized checkpoint; direct refusal alignment may result in only surface-level refusal. This limits its applicability as a standalone unlearning method.
- The refusal vector itself relies on a small number of out-of-knowledge prompts and existing model refusal behaviors; if the base model lacks honest refusal to begin with, the vector quality may suffer.

## Related Work & Insights
- **vs RMU**: RMU reduces target knowledge availability by randomizing internal features of the forget-set but does not guarantee the model knows it has forgotten; ReVa aligns the refusal state post-RMU to provide honesty.
- **vs IDK+AP / rejection SFT**: IDK+AP directly trains the model to output refusal templates, which yields high RR but easily retains underlying knowledge and is unstable across multiple turns; ReVa is cheaper and more stable.
- **vs GA / NPO / SimNPO**: Gradient-ascent methods achieve forgetting by suppressing target answer probabilities, but the target is unbounded, often leading to logit polarization, utility collapse, and fake IDK.
- **vs LLM honesty benchmarks**: Works like BeHonest evaluate self-knowledge/self-expression in general scenarios; this paper applies these concepts to the forget/retain partition of unlearning, defining risks closer to the reality of deleting target knowledge.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The problem definition and metric design for honest unlearning are highly valuable; ReVa is a natural yet effective extension of the refusal vector concept to unlearning.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers 9 categories of methods, multiple metrics, efficiency, and additional multi-turn analysis; the data domain remains somewhat narrow.
- Writing Quality: ⭐⭐⭐⭐☆ Strong problem awareness; failure modes are clearly explained; some symbols and table organization are slightly unrefined.
- Value: ⭐⭐⭐⭐⭐ Vital for LLM safety evaluation, especially in cautioning against misinterpreting low accuracy or high IDK as reliable forgetting.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](../../ICLR2026/llm_safety/llm_unlearning_with_llm_beliefs.md)
- [\[ACL 2026\] Into the Gray Zone: Domain Contexts Can Blur LLM Safety Boundaries](into_the_gray_zone_domain_contexts_can_blur_llm_safety_boundaries.md)
- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ACL 2026\] Logical Consistency as a Bridge: Improving LLM Hallucination Detection via Label Constraint Modeling between Responses and Self-Judgments](logical_consistency_as_a_bridge_improving_llm_hallucination_detection_via_label_.md)
- [\[ACL 2026\] Evaluating Answer Leakage Robustness of LLM Tutors against Adversarial Student Attacks](evaluating_answer_leakage_robustness_of_llm_tutors_against_adversarial_student_a.md)

</div>

<!-- RELATED:END -->
