---
title: >-
  [Paper Note] A Paired Testing Protocol for Batch-Conditioned Refusal Robustness in LLM Serving
description: >-
  [ICML2026][Recommender Systems][Refusal Robustness] This paper treats batch conditions in LLM serving as treatment variables for safety evaluation. It proposes a testing protocol consisting of paired comparisons between…
tags:
  - "ICML2026"
  - "Recommender Systems"
  - "Refusal Robustness"
  - "Batch Inference"
  - "vLLM"
  - "Paired Testing"
  - "Safety Evaluation"
date: 2026-05-08
content_hash: 228827a9e47bdc00
---

# A Paired Testing Protocol for Batch-Conditioned Refusal Robustness in LLM Serving

**Conference**: ICML2026  
**arXiv**: [2605.27763](https://arxiv.org/abs/2605.27763)  
**Code**: No public code available (code repository not provided in cache)  
**Area**: LLM Safety / LLM Evaluation / Inference Serving Robustness  
**Keywords**: Refusal Robustness, Batch Inference, vLLM, Paired Testing, Safety Evaluation  

## TL;DR
This paper treats batch conditions in LLM serving as treatment variables for safety evaluation. It proposes a testing protocol consisting of paired comparisons between safety prompts and capability controls, human/scorer correction, cross-model expansion, continuous batching composition, and batch-invariant kernel ablation. The conclusion is that refusal flips are real but infrequent, model-specific, and dependent on the specific serving stack.

## Background & Motivation
**Background**: LLM safety evaluation typically fixes the inference serving configuration and then measures whether the model refuses harmful requests, exhibits over-refusal, or maintains capability performance. The serving system side focuses on the impact of batch size, continuous batching, KV cache, scheduling, and kernels on throughput/latency, rarely incorporating these system variables as safety evaluation conditions.

**Limitations of Prior Work**: If the same prompt yields different outputs when requested individually, in a synchronous batch, or via a continuous batching scheduler, traditional evaluations are likely to miss these discrepancies. More troublesome is that output changes do not necessarily equate to safety issues: they might be general text instability or may happen to cross the refusal/compliance boundary. Simply reporting that "batching changes output" is insufficient; it is necessary to distinguish between safety label changes and general capability label changes.

**Key Challenge**: Batching is a common performance optimization in production serving, yet it alters execution ordering, numerical paths, co-residence, and kernel behavior. If safety evaluation treats it as a background constant, it defaults to the assumption that "refusal behavior under a single request represents behavior under production batch conditions," which is the very hypothesis this paper aims to test.

**Goal**: The authors do not attempt to prove that batching is universally dangerous but propose a testing protocol that avoids over-interpretation. Under the same prompt and fixed decoding conditions, it compares label changes across different batch conditions. Using capability controls, scorer correction, cross-model validation, and kernel ablation, it distinguishes real low-frequency refusal flips from measurement noise.

**Key Insight**: The paper defines the batch condition as a treatment variable and the evaluation unit as a conditioned row, i.e., "the paired results of the same prompt under two or more serving conditions." This row-level pairing is more sensitive than aggregate refusal rate statistics and is better suited for discovering boundary samples.

**Core Idea**: Transition refusal robustness evaluation from "scoring under a fixed serving configuration" to "paired intervention testing on real serving batch conditions," reporting low-frequency findings, generalization corrections, and mechanism ablations in layers.

## Method
Ours does not propose a new model training method but rather a safety evaluation protocol and four layers of evidence. The logic is to first discover batch-conditioned refusal flips via local perturbation studies, then use a larger set of models to check if the signal generalizes, followed by continuous batching composition tests to see if multi-tenant co-batching poses extra risks, and finally execute mechanism ablation using batch-invariant kernels.

### Overall Architecture
The input consists of a set of safety prompts, capability control prompts, models, and serving conditions. The output is not a single safety score but flip rates, directions, corrected true flip proportions, cross-model heterogeneity, composition effects, and kernel ablation results organized by research layer.

The four studies play different roles. Study A is the local discovery layer, using three 1B-3B instruction-tuned models to compare safety and capability label changes under synchronous dispatch, neighbor conditions, concurrent quantization, and explicit true batching. Study B expands to 15 models to check if initial safety skews generalize and analyzes whether alignment type and output instability predict fragility. Study C uses vLLM FP16 continuous batching to test if co-batched neighbors bring independent composition effects. Study D uses the same H100/vLLM 0.19.1 stack to compare standard vLLM with `VLLM_BATCH_INVARIANT=1` for 55 current score-flip candidates.

### Key Designs
1.  **Safety-Capability Paired Testing**:
    - **Function**: Avoids misjudging arbitrary batch-induced output changes as safety issues.
    - **Mechanism**: The safety side uses prompts related to harmful behavior, jailbreak, truthfulness, bias, and over-refusal; the capability side uses control tasks like MMLU and ARC-Challenge. For each row, labels are compared across different batch conditions, and flips for safety and capability labels are counted separately.
    - **Design Motivation**: If safety and capability labels flip with similar frequency, it suggests the issue is generalized output instability. Only if the safety side is relatively more prone to crossing boundaries than the capability side does it support a refusal robustness risk.

2.  **Layered Correction and Generalization**:
    - **Function**: Strips local discovery signals away from automated scoring noise and small-model coincidences.
    - **Mechanism**: Study A first reports safety/capability flips under automated scoring, then performs Unicode normalization, scorer-corrected audits, and human review of candidates for changed rows. Study B expands the scope to 15 models, reporting safety/capability ratios, fragility ranges, alignment type ANOVA, and output instability correlation.
    - **Design Motivation**: Refusal flips are rare events, and a single positive result is easily over-interpreted. Layered correction preserves the discovery of "actual boundary samples" while calibrating the operational rate to a more conservative range.

3.  **Exact-stack Mechanism Ablation**:
    - **Function**: Determines whether candidate flips depend on specific serving kernel paths.
    - **Mechanism**: On the same H100 pod, with the same model, prompt, dispatch mode, temperature 0, and max length 2048, standard vLLM and batch-invariant kernels are run separately. If the standard path reproduces the label flip while the invariant path eliminates it, the candidate surface is sensitive to batch-sensitive execution paths.
    - **Design Motivation**: Deployment risk ultimately depends on the actual serving stack. Rather than discussing batching risks in the abstract, it is better to perform exact-stack validation on model/kernel/batch settings close to production.

### Loss & Training
Ours does not utilize a training loss function. The evaluation strategy fixes prompts, weights, and greedy decoding (temperature 0), only changing batch size, dispatch synchronization, co-batched composition, or kernel paths. Statistical interpretation follows three rules: positive local discoveries are only upgraded to visible conclusions if supported by larger expansion or mechanism checks; directional results must report absolute rates; when studies conflict, the larger sample or more mechanistic study determines the claim boundary.

## Key Experimental Results

### Main Results
Results from the four studies collectively support a "low-frequency, model-specific, stack-dependent" conclusion rather than "batching is universally unsafe."

| Study | Scope | Key Metrics | Main Conclusion |
|-------|----------|----------|----------|
| A: Local Perturbation | 31,410 scored rows, 3 1B-3B models | Auto-safety 0.51% vs Cap 0.14%; Augmented 1.68% vs 0.42%; 17 true flips from 63 candidates, corrected total ~0.16% | Real refusal boundary shifts exist, but operational rate is very low. |
| B: Cross-Model Expansion | 127,224 records, 15 models | Safety/Cap ratio ~0.94×; Fragility 0.00%-2.39%; Alignment ANOVA p=0.942; Output instability r=0.909 | No universal safety skew; output instability is a better early warning signal. |
| C: Continuous Batching | 14,250 records, 5 conditions | No aggregate composition effect detected above 4.7pp; 28/31 rare flips skewed unsafe; co-batch verification 22.1% | No large-scale composition effects, but directional small samples need monitoring. |
| D: Kernel Ablation | 55 candidates from Study A, vLLM 0.19.1/H100 | Standard: 22 label flips, 25 text changes; Batch-invariant: 0 label flips, 0 text changes | Current flip candidates depend on non-invariant execution paths in the test stack. |

The predictor results from Study B are particularly important: alignment type does not explain fragility, whereas output instability is highly correlated with safety fragility.

| Analysis Item | Value | Interpretation |
|--------|------|------|
| Safety-to-capability ratio | 0.94× | No universally higher flip rate on the safety side across models. |
| Fragility range | 0.00%-2.39% | Obvious differences exist between models. |
| Alignment type ANOVA | p=0.942, $\eta^2=0.033$ | No correlation between alignment type and fragility under available power. |
| Output instability correlation | r=0.909, bootstrap 95% CI [0.65, 0.97] | The more unstable the output under batching, the more fragile the refusal boundary. |
| Directional counts | 159 compliance-to-refusal vs 81 refusal-to-compliance | Direction is not fixed to unsafe; depends on model set. |

### Ablation Study
The most direct mechanism ablation in the paper is the batch-invariant kernel ablation. The goal is not to estimate total risk but to check if current candidate flips are carried by batch-sensitive execution paths.

| Mode | Rows | OK | Label flips | Text changes | Description |
|------|------|----|-------------|--------------|------|
| Standard vLLM | 55 | 55 | 22 | 25 | Reproducible low-frequency flips; flips in current candidate set are in safety-domain rows. |
| Batch-invariant | 55 | 55 | 0 | 0 | Flips disappear under same model, prompt, and H100 stack. |

Another key analysis is the Study A scorer/adjudication correction, which explains why automated discovery rates cannot be treated directly as production risks.

| Correction Layer | Result | Meaning |
|--------|------|------|
| Raw Auto-discovery | Safety 0.51% vs Cap 0.14% | Discovery signal exists but may include scorer artifacts. |
| Augmented Subset | Safety 1.68% vs Cap 0.42% | Directionality persists in the enriched subset. |
| Scorer-corrected Audit | 26 unsafe vs 18 safe direction | Directionality does not entirely disappear after Unicode/score correction. |
| Human Review (63 candidates) | 17 genuine behavioral flips (~27%) | Most candidates are artifacts like changes in refusal phrasing. |
| Corrected Total Rate | ~0.16% | True operational rate should be significantly lower than auto-discovery signals. |

### Key Findings
- Batch-conditioned refusal flip is not a fictitious problem. Study A's true batching layer still shows 0.80% safety flips with 99.4% agreement with synchronized dispatch, indicating it is not a pure scheduling illusion.
- Initial safety skews do not generalize. In the 15-model expansion, safety/capability ratios are near parity, meaning risks cannot be simply extrapolated by alignment type or model family.
- Output instability is the most useful screening variable. The r=0.909 correlation suggests checking a model's output change rate under batching conditions before deciding on intensive refusal robustness testing.
- Continuous-batch composition showed no large aggregate effect, but the direction of rare flips skewed unsafe, and co-batch verification was only 22.1%. Thus, the conclusion is "current evidence is insufficient to guide routing," not that "composition is always safe."
- The kernel ablation makes the mechanism story more concrete: at least for the vLLM 0.19.1/H100/small-model candidate set, non-invariant execution paths facilitate these flips.

## Highlights & Insights
- The most robust aspect of the paper is its restraint. It does not stop at the eye-catching conclusion that "batching affects refusal" but continues with cross-model studies, composition effects, and kernel ablations to narrow claims to what the data truly supports.
- The paired design of safety prompts and capability controls is a highly reusable evaluation pattern. It forces researchers to answer whether it is a "safety boundary shift" or just "general output change," preventing system nondeterminism from being packaged as a safety finding.
- The recommendation for exact-stack validation is practical. In real deployment, users care whether the current model, batch setting, and kernel path change refusal behavior, not whether an abstract backend category is risky.
- This article also serves as a reminder that safety evaluation should inherit the discipline of system benchmarking: once serving configurations change the execution regime, they should be reported as evaluation conditions rather than hidden in background settings.

## Limitations & Future Work
- All studies reside in a sparse-flip regime; directional statistics are susceptible to sample selection and scorer error. While the paper performs correction, a larger-scale prospective benchmark is still necessary.
- Human adjudication involved only a single reviewer without inter-rater agreement, making the 0.16% correction rate more of a conservative adjustment than a gold-standard population estimate.
- Heterogeneity exists between studies regarding hardware, models, scoring stacks, and task sets. The paper provides an operational doctrine rather than a directly mergeable aggregate effect size.
- Co-batch verification in the composition study was only 22.1%, weakening the strength of the negation of actual multi-tenant co-batching effects.
- Kernel ablation only covered vLLM 0.19.1, H100, and three 1B-3B instruction-tuned models. Larger models, tensor parallelism, other backends, stochastic decoding, and production-grade schedulers still require separate validation.

## Related Work & Insights
- **vs. Deterministic Inference Literature**: Existing work shows that output differences can arise from batching, floating point operations, and kernel paths under fixed prompt/weights/decoding; ours pivots this mechanistic issue toward safety refusal boundaries.
- **vs. Serving Systems Literature**: vLLM, continuous batching, KV cache, and scheduling research primarily optimize throughput and latency; ours demands including these variables within the safety envelope.
- **vs. Quantization Safety Literature**: Quantization and compression have been shown to alter trust/safety behavior; ours emphasizes that batching is also part of deployment optimization, though its effects are more infrequent and stack-dependent.
- **vs. Conventional Safety Benchmarks**: Conventional benchmarks often fix serving configs; the takeaway here is that benchmark reports should include production batch settings, capability controls, and mechanism sensitivity checks.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The problem definition is very new, explicitly incorporating batch conditions as variables for refusal robustness testing.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ The four-layer evidence chain is complete, though candidate rarity, human correction, and backend coverage remain limited.
- Writing Quality: ⭐⭐⭐⭐☆ Claims are clearly bounded, and statistical interpretation is restrained; as a synthesis paper, some original details of studies depend on artifact context.
- Value: ⭐⭐⭐⭐☆ Highly practical for LLM serving safety evaluation, especially suitable for pre-production exact-stack validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Tool4POI: A Tool-Augmented LLM Framework for Next POI Recommendation](../../AAAI2026/recommender/tool4poi_a_tool-augmented_llm_framework_for_next_poi_recommendation.md)
- [\[ICLR 2026\] Token-Efficient Item Representation via Images for LLM Recommender Systems](../../ICLR2026/recommender/token-efficient_item_representation_via_images_for_llm_recommender_systems.md)
- [\[AAAI 2026\] Align³GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation](../../AAAI2026/recommender/align3gr_unified_multi-level_alignment_for_llm-based_generat.md)
- [\[ACL 2026\] ReRec: Reasoning-Augmented LLM-based Recommendation Assistant via Reinforcement Fine-tuning](../../ACL2026/recommender/rerec_reasoning-augmented_llm-based_recommendation_assistant_via_reinforcement_f.md)
- [\[AAAI 2026\] Tokenize Once, Recommend Anywhere: Unified Item Tokenization for Multi-domain LLM-based Recommendation](../../AAAI2026/recommender/tokenize_once_recommend_anywhere_unified_item_tokenization_for_multi-domain_llm-.md)

</div>

<!-- RELATED:END -->
