---
title: >-
  [Paper Note] Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures
description: >-
  [ICLR 2026][LLM Evaluation][Machine Unlearning] This paper reveals that the true driver of "benign relearning" in LLM machine unlearning is syntactic similarity rather than topical relevance, and proposes a syntactic diversification strategy (paraphrasing the forget set) that effectively suppresses relearning, accelerates forgetting, and alleviates the trade-off between unlearning efficacy and model utility.
tags:
  - ICLR 2026
  - LLM Evaluation
  - Machine Unlearning
  - Benign Relearning
  - Syntactic Similarity
  - Unlearning Robustness
  - Syntactic Diversification
date: 2026-05-08
content_hash: 889d0544ef41df89
---

# Rethinking Benign Relearning: Syntax as the Hidden Driver of Unlearning Failures

**Conference**: ICLR 2026
**arXiv**: [2602.03379](https://arxiv.org/abs/2602.03379)
**Code**: None
**Area**: LLM Evaluation
**Keywords**: Machine Unlearning, Benign Relearning, Syntactic Similarity, Unlearning Robustness, Syntactic Diversification

## TL;DR
This paper reveals that the true driver of "benign relearning" in LLM machine unlearning is syntactic similarity rather than topical relevance, and proposes a syntactic diversification strategy (paraphrasing the forget set) that effectively suppresses relearning, accelerates forgetting, and alleviates the trade-off between unlearning efficacy and model utility.

## Background & Motivation
Machine Unlearning aims to remove specific content (e.g., private data, copyrighted material) from trained LLMs so that the model behaves as if it had never seen such data. Mainstream methods include Gradient Ascent (GA), Negative Preference Optimization (NPO), and SCRUB.

However, **Benign Relearning** poses a serious threat to unlearning effectiveness: after unlearning, fine-tuning the model on benign data that appears unrelated to the forgotten content can recover the forgotten information. For example, after unlearning a passage from Harry Potter, fine-tuning on GPT-generated character descriptions can reproduce the original text.

The prior BLUR benchmark attributed this to **topical relevance**: the closer the fine-tuning data is to the topic of the forgotten content, the stronger the recovery. However, the authors identify two experimental design flaws in BLUR: (1) datasets of different relevance levels vary in size, resulting in inconsistent numbers of gradient update steps; and (2) evaluation is performed only at the end of a fixed number of epochs, potentially missing recovery peaks.

After standardizing the number of steps and evaluating progressively, the authors find that **the advantage of topical relevance largely disappears** — even completely unrelated filler text such as "Lorem ipsum" achieves comparable recovery. This motivates a deeper investigation into the true driving factor.

The core finding is that **syntactic similarity — i.e., surface structural overlap — is the primary cause of benign relearning**. Unlearning predominantly suppresses "answer templates" rather than keywords themselves; syntactically similar data restores the suppressed template structures, allowing keywords to resurface.

## Method

### Overall Architecture
The paper consists of three parts: (1) re-examining BLUR's conclusions on topical relevance; (2) controlled experiments demonstrating that syntactic similarity is the primary factor; and (3) proposing syntactic diversification as a defense strategy.

### Key Designs

1. **Syntactic Similarity Metric**: Normalized Levenshtein distance is used to measure surface structural overlap between two text segments: $\text{Sim}(s_1, s_2) = 1 - \frac{d_{\text{Lev}}(s_1, s_2)}{\max(|s_1|, |s_2|)}$, computed at the sentence level and averaged over all sentence pairs across datasets.

2. **Controlled Experiment Design (TOFU Dataset)**: Under the forget05 scenario of the TOFU dataset (forgetting 10 fictional authors), two contrastive relearning sets are constructed:

    - $D_{\text{relearn}}^{\text{topic}}$ (topically relevant): non-name questions about the target authors (e.g., birthplace), with syntactic similarity of 0.2349
    - $D_{\text{relearn}}^{\text{syntactic}}$ (syntactically similar): name-format questions identical in structure to the target set but concerning different authors, with syntactic similarity of 0.4513
    - Key control: the syntactically similar set has **no topical overlap** with the target set

3. **Loss Ratio Analysis**: Loss Ratio is defined as $\mathcal{L}_{\text{template}} / \mathcal{L}_{\text{keyword}}$, partitioning answer tokens into template tokens (generic phrasing) and keyword tokens (specific forgotten information such as names). The Loss Ratio is observed to increase continuously during unlearning, indicating that unlearning disproportionately suppresses templates rather than keywords — this structural channel is what syntactic relearning exploits.

4. **Syntactic Diversification**: GPT-4o is used to paraphrase queries in the forget set into diverse syntactic forms (preserving semantics), breaking the uniform template structure of the original forget set. After paraphrasing, the syntactic similarity between $D_{\text{relearn}}^{\text{syntactic}}$ and $D_{\text{forget}}'$ decreases from 0.4513 to 0.2241.

### Loss & Training
Syntactic diversification does not modify the unlearning algorithm itself (GA/NPO/SCRUB are retained); only the forget set is replaced with its diversified version $D_{\text{forget}}'$. This forces the model to directly suppress keywords rather than merely suppressing templates, fundamentally eliminating the syntactic relearning channel.

## Key Experimental Results

### Main Results (TOFU Dataset, Relearn Success Rate)

| Unlearning Method | Relearning Set | Step 10 Recovery | Step 30 Recovery | Step 50 Recovery |
|---|---|---|---|---|
| GA | $D_{\text{relearn}}^{\text{topic}}$ | ~0% | ~0% | ~0% |
| GA | $D_{\text{relearn}}^{\text{syntactic}}$ | ~40% | ~60% | ~70% |
| NPO | $D_{\text{relearn}}^{\text{topic}}$ | ~0% | ~0% | ~0% |
| NPO | $D_{\text{relearn}}^{\text{syntactic}}$ | ~30% | ~50% | ~55% |
| SCRUB | $D_{\text{relearn}}^{\text{topic}}$ | ~10% | ~15% | ~10% |
| SCRUB | $D_{\text{relearn}}^{\text{syntactic}}$ | ~80% | ~90% | ~95% |

Across all unlearning methods, recovery rates under the syntactically similar set substantially exceed those under the topically relevant set. SCRUB unlearns fastest but is the most vulnerable.

### Ablation Study (Syntactic Diversification Effect + Model Utility Preservation)

| Forget Set | Real Authors (Avg↑) | World Facts (Avg↑) | Retain Set (Avg↑) |
|---|---|---|---|
| $D_{\text{forget}}$ (original) | 0.4014 | 0.6056 | 0.1607 |
| $D_{\text{forget}}'$ (diversified) | **0.4852** | **0.6104** | **0.3128** |

After syntactic diversification: (1) recovery rate against syntactic relearning drops to 0% even at only 50 unlearning steps; (2) model utility (Real Authors, Retain Set) improves significantly; and (3) the Loss Ratio converges to 1, indicating balanced suppression of both templates and keywords.

### Key Findings
- Representation and gradient analysis: $D_{\text{relearn}}^{\text{syntactic}}$ exhibits substantially higher hidden state cosine similarity and gradient cosine similarity to the target set in the unlearned model compared to $D_{\text{relearn}}^{\text{topic}}$.
- The differential relearning effects of $D_{\text{hi}} / D_{\text{mid}} / D_{\text{low}}$ in BLUR are consistent with their syntactic similarity ordering rather than their topical relevance ordering.
- LoRA fine-tuning recovers forgotten content faster than full-parameter fine-tuning during relearning, suggesting that PEFT may amplify unlearning fragility.
- Safety training (DPO) is more susceptible to syntactic relearning attacks than dedicated unlearning algorithms.

## Highlights & Insights
- **Overturning prior belief**: Topical relevance is not the primary cause of benign relearning — syntactic similarity is. This redefines the evaluation criteria for unlearning robustness.
- The identification of methodological flaws in the BLUR benchmark (inconsistent step counts and single-point evaluation) is precise and sets a methodological standard for the field.
- The proposed syntactic diversification strategy incurs virtually no additional training cost (requiring only a one-time GPT-4o paraphrasing step), yet simultaneously improves three dimensions: unlearning strength, robustness, and model utility.
- The Loss Ratio analysis reveals an "uneven forgetting" mechanism — models preferentially forget template formats rather than the actual knowledge content.
- The syntactic diversification solution is remarkably concise (only requiring GPT-4o query paraphrasing) while simultaneously improving unlearning efficacy, robustness, and model utility.
- The methodological critique of the BLUR benchmark (inconsistent steps, single-point evaluation) serves as an important methodological reference for the community.

## Limitations & Future Work
- Validation is primarily conducted on the TOFU synthetic dataset; applicability to real-world scenarios (e.g., copyright data unlearning) requires further investigation.
- Syntactic diversification relies on GPT-4o for paraphrasing, introducing additional cost and dependency on an external model.
- Levenshtein distance as a syntactic metric may be overly coarse; future work could explore more precise measures such as syntactic tree distance.
- Only Llama-2-7B is evaluated; generalizability to larger models (70B+) and different architectures remains unknown.
- Combinations of syntactic diversification with other defense methods (e.g., adversarial training) are not explored.

## Related Work & Insights
- Relationship to BLUR (Hu et al., 2025b): directly challenges its conclusion that "topical relevance determines relearning," identifying confounding factors in its experimental design.
- Cross-findings with DuoAttention/LoRA: LoRA recovers faster during relearning, suggesting that PEFT may be a weak point in unlearning security.
- Insight: unlearning evaluation should not focus solely on content-level dimensions; syntactic/structural attack surfaces are equally critical.
- Broader implication: LLM knowledge may be encoded as "template + keyword" pairs, and unlearning should target both simultaneously.
- Future work could explore combining syntactic diversification with prompt-level defenses (e.g., input perturbation) to build a multi-layered defense system.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Overturns mainstream understanding by identifying syntax rather than topic as the key factor in unlearning failure
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark validation across TOFU and BLUR is comprehensive, though real-world scenario experiments are lacking
- Writing Quality: ⭐⭐⭐⭐⭐ Problem formulation → rebuttal of prior work → controlled experiments → mechanistic analysis → solution; the logic is exceptionally tight
- Value: ⭐⭐⭐⭐ Provides important guidance for evaluation methodology and defense strategies in the machine unlearning field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Revisiting the Past: Data Unlearning with Model State History](revisiting_the_past_data_unlearning_with_model_state_history.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](llm_unlearning_with_llm_beliefs.md)
- [\[ICLR 2026\] Truthfulness Despite Weak Supervision: Evaluating and Training LLMs Using Peer Prediction](truthfulness_despite_weak_supervision_evaluating_and_training_llms_using_peer_pr.md)
- [\[ICLR 2026\] Unpacking Human Preference for LLMs: Demographically Aware Evaluation with the HUMAINE Framework](unpacking_human_preference_for_llms_demographically_aware_evaluation_of_long-fo.md)
- [\[ICLR 2026\] Improving Set Function Approximation with Quasi-Arithmetic Neural Networks](improving_set_function_approximation_with_quasi-arithmetic_neural_networks.md)

</div>

<!-- RELATED:END -->
