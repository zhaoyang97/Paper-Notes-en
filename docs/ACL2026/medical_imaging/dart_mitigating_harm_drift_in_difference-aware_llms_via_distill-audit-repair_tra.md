---
title: >-
  [Paper Note] DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training
description: >-
  [ACL 2026][Medical Imaging][To be supplemented] To be supplemented after a thorough reading of the paper.
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "To be supplemented"
date: 2026-05-08
content_hash: cb375ee700219765
---

# DART: Mitigating Harm Drift in Difference-Aware LLMs via Distill-Audit-Repair Training

**Conference**: ACL 2026
**arXiv**: [2604.16845](https://arxiv.org/abs/2604.16845)  
**Code**: [GitHub](https://github.com/dart-framework)  
**Area**: Medical Imaging
**Keywords**: difference-awareness, harm drift, distill-audit-repair, safety alignment, over-refusal

## TL;DR
DART identifies and addresses "harm drift"—a phenomenon whereby fine-tuning LLMs to improve difference-aware classification accuracy (e.g., recognizing legitimate demographic distinctions) causes the model's generated explanations to become increasingly harmful. Through a three-stage Distill-Audit-Repair pipeline, DART improves Llama-3-8B accuracy from 39.0% to 68.8% while reducing harm drift cases by 72.6%.

## Background & Motivation

**Background**: Safety-aligned LLMs tend to default to "identity-blindness"—refusing to acknowledge demographic differences even when factually correct (e.g., ancestry-based disease incidence disparities) or legally justified (e.g., hiring preferences of religious institutions). This results in incorrect answers, unnecessary refusals, or generic "equal treatment" defaults.

**Limitations of Prior Work**: (1) LLMs perform poorly on difference-aware classification—Llama-3-8B predicts 88.6% of prompts as "requiring differentiation," whereas only 50.2% actually do, yielding an equal-treatment case accuracy of merely 11.3%; (2) 26.8% of outputs are unparseable refusals or ambiguous responses; (3) fine-tuning can improve accuracy but triggers "harm drift," where conclusions are correct yet explanations introduce harmful content.

**Key Challenge**: Improving difference-aware accuracy requires fine-tuning, yet fine-tuning undermines safety alignment—accuracy and safety appear mutually exclusive.

**Goal**: Simultaneously improve difference-aware classification accuracy and explanation safety, demonstrating that the two need not conflict.

**Key Insight**: Accuracy optimization and safety repair are decoupled into sequential stages—first distilling to improve accuracy (tolerating temporary safety degradation), then auditing to localize harm drift cases, and finally performing targeted repair.

**Core Idea**: Harm drift constitutes a novel safety failure mode in which the model's decisions become correct while its explanations become harmful, necessitating detection of safety degradation at the explanation level rather than solely at the decision output level.

## Method

### Overall Architecture
A three-stage pipeline: Stage I (Distill) fine-tunes the student model with teacher-generated rationales to improve accuracy → Stage II (Audit) compares outputs for the same prompt before and after distillation, detecting harm drift via a toxicity classifier combined with LLM-as-Judge → Stage III (Repair) applies severity-weighted fine-tuning on drifted cases, replacing harmful rationales with safer alternatives.

### Key Designs

1. **Label-Conditioned Teacher Distillation (Stage I)**:

    - Function: Improve difference-aware classification accuracy.
    - Mechanism: The teacher model receives the ground-truth label $y^*$ and generates a rationale $r^*$ explaining that label (rather than predicting the label independently). Harm-aware prompting guides the teacher to produce concise rationales while avoiding the reproduction of harmful content. LoRA fine-tuning is applied to the student model $M_0$ to obtain the intermediate model $M_{int}$.
    - Design Motivation: Label conditioning ensures rationales are aligned with correct conclusions—substituting predicted labels for ground-truth labels reduces accuracy from 0.682 to 0.641 and severely disrupts subsequent auditing.

2. **Paired Harm Auditing (Stage II)**:

    - Function: Precisely identify harm drift cases induced by distillation.
    - Mechanism: For each test prompt $x$, outputs are generated from both $M_0$ and $M_{int}$ under identical decoding conditions. A toxicity classifier screens for cases satisfying $\mathcal{H}(r_{int}) - \mathcal{H}(r_0) > \tau_{delta}$ (where $\tau_{delta}=0.01$), followed by LLM-as-Judge confirmation of whether the output falls into one of three drift categories: (i) repeating or elaborating harmful content that $M_0$ avoided, (ii) normalizing problematic assumptions, or (iii) omitting harms that $M_0$ identified. Confirmed cases are assigned one of four severity levels (minor / moderate / severe / extreme).
    - Design Motivation: The paired design controls for prompt difficulty—only changes within the same prompt are compared, ensuring that detected degradation is attributable to distillation rather than inherent prompt difficulty.

3. **Severity-Weighted Repair (Stage III)**:

    - Function: Targeted repair of harm drift cases without compromising accuracy.
    - Mechanism: For drifted cases in $\mathcal{P}_{drift}$, safer alternative rationales are generated and assigned training weights proportional to severity. LoRA continues to fine-tune $M_{int}$ to obtain $M_{DART}$. Only the behavior on drifted cases is modified, constraining parameter drift.
    - Design Motivation: The staged approach outperforms joint multi-objective optimization—ablation experiments confirm that joint training fails to match either the accuracy of pure distillation or the safety of targeted repair.

### Loss & Training
Both Stage I and Stage III employ LoRA fine-tuning with standard next-token prediction loss; Stage III additionally incorporates severity-based sample weighting. At inference time, explanation strategy constraints may optionally be appended to guide rationale generation.

## Key Experimental Results

### Main Results

| Model | Method | Overall Acc. | EQUAL Acc. | DIFF Acc. | Harm Drift↓ |
|-------|--------|-------------|-----------|----------|------------|
| Llama-3-8B | Baseline $M_0$ | 39.0% | 11.3% | 66.6% | — |
| Llama-3-8B | $M_{DART}$ | **68.8%** | **72.6%** | — | −72.6% |
| Llama-3.2-3B | $M_{DART}$ | +24.7pp | — | — | Significant reduction |

### Ablation Study

| Configuration | Accuracy | Safety | Notes |
|---------------|----------|--------|-------|
| Distillation only (Stage I) | 68.2% | Low | High accuracy but severe harm drift |
| Joint toxicity regularization | ~60% | Moderate | Neither objective well optimized |
| Full DART | **68.8%** | **High** | Staged strategy is optimal |

### Key Findings
- Equal-treatment case accuracy improves most substantially (11.3% → 72.6%), indicating effective mitigation of over-refusal.
- In open-domain queries, appropriately differentiated responses increase from 39.8% to 77.5%, while refusal rate drops from 34.3% to 3.0%.
- Label-conditioned generation is critical for auditing precision—using predicted labels reduces audit precision/recall from 0.720/0.810 to 0.582/0.694.
- Harm drift is distinct from conventional toxicity—it manifests in explanatory reasoning rather than response compliance, rendering standard metrics insufficient for detection.

## Highlights & Insights
- **Harm drift** represents a novel and significant safety failure mode—"correct conclusions paired with harmful reasoning" has not previously been studied systematically.
- The staged design philosophy merits broader adoption: fully optimize the primary objective first, then perform targeted repair of side effects, rather than attempting to balance multiple objectives from the outset.
- The two-stage auditing design combining LLM-as-Judge with a toxicity classifier strikes an effective balance between efficiency and precision.

## Limitations & Future Work
- Auditing relies on the judgment quality of LLM-as-Judge, which may introduce bias.
- Evaluation is limited to the difference-aware classification task; the behavior of "harm drift" in other fine-tuning scenarios remains unexplored.
- The repair stage may introduce new side effects, potentially necessitating iterative repair cycles.

## Related Work & Insights
- **vs. Standard Safety Fine-Tuning**: Standard approaches focus on response compliance (whether to refuse), whereas DART targets explanation quality—a more fine-grained safety dimension.
- **vs. DPO/RLHF**: These methods achieve holistic alignment via preference data; DART employs precise auditing to localize and repair specific harm drift cases.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The harm drift concept is novel and the staged solution is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 benchmarks + 280 open-domain queries + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Problem definition is clear and examples are intuitive.
- Value: ⭐⭐⭐⭐⭐ Reveals a new safety risk in fine-tuning with important implications for LLM alignment research.

**Code**: To be confirmed  
**Area**: medical_imaging
**Keywords**: To be supplemented

## TL;DR
To be supplemented after a thorough reading of the paper.

## Background & Motivation
To be supplemented after a thorough reading of the paper.

## Method
To be supplemented after a thorough reading of the paper.

## Key Experimental Results
To be supplemented after a thorough reading of the paper.

## Highlights & Insights
To be supplemented after a thorough reading of the paper.

## Limitations & Future Work
To be supplemented after a thorough reading of the paper.

## Related Work & Insights
To be supplemented after a thorough reading of the paper.

## Rating
- Novelty: Pending
- Experimental Thoroughness: Pending
- Writing Quality: Pending
- Value: Pending

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Protein as a Second Language for LLMs](../../ICLR2026/medical_imaging/protein_as_a_second_language_for_llms.md)
- [\[ICLR 2026\] Thompson Sampling via Fine-Tuning of LLMs](../../ICLR2026/medical_imaging/thompson_sampling_via_fine-tuning_of_llms.md)
- [\[AAAI 2026\] Training-Free Policy Violation Detection via Activation-Space Whitening in LLMs](../../AAAI2026/medical_imaging/training-free_policy_violation_detection_via_activation-space_whitening_in_llms.md)
- [\[AAAI 2026\] GEM: Generative Entropy-Guided Preference Modeling for Few-shot Alignment of LLMs](../../AAAI2026/medical_imaging/gem_generative_entropy-guided_preference_modeling_for_few-shot_alignment_of_llms.md)
- [\[CVPR 2026\] Mitigating Object Hallucination in LVLMs via Attention Imbalance Rectification](../../CVPR2026/medical_imaging/mitigating_object_hallucinations_in_lvlms_via_attention_imbalance_rectification.md)

</div>

<!-- RELATED:END -->
