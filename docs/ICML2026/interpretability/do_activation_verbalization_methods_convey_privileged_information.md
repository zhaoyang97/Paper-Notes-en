---
title: >-
  [Paper Note] Do Activation Verbalization Methods Convey Privileged Information?
description: >-
  [ICML 2026][Interpretability][Patchscopes] This paper systematically demonstrates that the performance of currently popular activation verbalization methods (Patchscopes / LIT / SelfIE), when used as LLM interpretability tools, can be entirely explained by the "verbalizer model's own knowledge" without requiring any internal activations from the target model. T
tags:
  - ICML 2026
  - Interpretability
  - Patchscopes
  - LIT
date: 2026-05-08
content_hash: d457780ee617092d
---
# Do Activation Verbalization Methods Convey Privileged Information?

**Conference**: ICML 2026  
**arXiv**: [2509.13316](https://arxiv.org/abs/2509.13316)  
**Code**: https://github.com/millicentli/verb_faithfulness  
**Area**: Interpretability / LLM Probing / Benchmark Critique  
**Keywords**: Activation Verbalization, Patchscopes, LIT, Faithfulness, Privileged Knowledge

## TL;DR
This paper systematically demonstrates that the performance of currently popular activation verbalization methods (Patchscopes / LIT / SelfIE), when used as LLM interpretability tools, can be entirely explained by the "verbalizer model's own knowledge" without requiring any internal activations from the target model. This implies that these tools appear effective on existing benchmarks due to flawed benchmark design, and they tend to fabricate "explanations" that the target does not actually possess when the verbalizer's knowledge exceeds that of the target.

## Background & Motivation
**Background**: Understanding the internal representations of LLMs is a core challenge in interpretability. Recently, a class of "verbalization" methods has emerged—using a second LLM (verbalizer $\mathcal{M}_2$) to translate the hidden states of a target model ($\mathcal{M}_1$) into natural language descriptions. Representative works include Patchscopes (patching token activations into corresponding prompt positions), SelfIE (similar intuition), and LIT (fine-tuning a verbalizer to learn an activation matrix for all tokens). These methods are touted as tools for "understanding LLM computation."

**Limitations of Prior Work**: The key assumption that "the verbalizer's output reflects the target's internal representation" has never been rigorously tested. Since the verbalizer is itself an LLM with inherent world knowledge, it is impossible to distinguish whether its answers stem from the activations provided by the target or from its own internal commonsense. If it can answer correctly based purely on commonsense, such "explanations" hold no value for interpretability—one is explaining the world, not the model.

**Key Challenge**: Interpretability requires the verbalizer to convey "privileged information" (information that must be obtained via internal activations); however, the powerful parametric knowledge of LLMs allows them to answer most tasks correctly using only input text, making "answering via activations" and "answering without activations" indistinguishable.

**Goal**: (1) Verify whether existing benchmarks require the verbalizer to truly use the target's activations; (2) If not, construct controlled experiments to distinguish whether "knowledge comes from the target or the verbalizer"; (3) Investigate whom the verbalizer prioritizes when knowledge conflicts.

**Key Insight**: Critique verbalization as an NLP "shortcut learning" problem—if a model can answer correctly without looking at the inputs it supposedly should, the evaluation itself contains shortcuts; this is analogous to prior bias in VQA.

**Core Idea**: Design three sets of controls: (a) a zero-shot baseline where $\mathcal{M}_2$ sees the input without activations; (b) activation inversion to map activations back to input text to check information content; (c) knowledge mismatch experiments where $\mathcal{M}_2$ knows a fact that $\mathcal{M}_1$ does not, observing which answer the verbalizer reports.

## Method

### Overall Architecture
Ours is a critical evaluation study rather than a new model, addressing the untested assumption of whether verbalizer output reflects target internal activations or the verbalizer's own knowledge. It transforms the problem into three progressive counterfactual controls: first, using a zero-shot baseline to measure "performance without activations"; second, using activation inversion to test "if activations contain information beyond the input text"; and finally, using knowledge mismatch experiments to test "who the verbalizer trusts during knowledge conflict." All three sets of experiments share the same criterion—counting as correct if the output contains the ground-truth substring (case-insensitive), aligned with prior verbalization work—and use the McNemar test with Bonferroni correction for significance testing.

### Key Designs

**1. Zero-shot Baseline: Testing the necessity of activations with the simplest counterfactual**

Verbalization methods are recommended as interpretability tools under the premise that "activations provide information unattainable from the input alone," but this has never been disproven. Ours targets 6 feature extraction datasets used by Patchscopes / LIT (country_curr / food_country / ath_pos / ath_sport / prod_comp / star_const). By setting $\mathcal{M}_1 = \mathcal{M}_2 = $ Llama3.1-8B-Instruct or Ministral-8B-Instruct, the input $x_{\text{input}}$ and question are concatenated for $\mathcal{M}_2$ without any activation patching, comparing this against LIT and Patchscopes (averaged over layers 1-15). This is a stringent necessity test: if performance matches or exceeds verbalization without activations, the marginal contribution of activations is zero or negative, undermining the legitimacy of these methods.

**2. Activation Inversion: Constructing "Activations as Lossy Input Copies" as an alternative explanation**

Even if one maintains that the success of Patchscopes comes from extra information in activations, a trivial explanation must be ruled out: that this information is merely a rehearsal of the input text. To this end, Ours trains a T5-Base or Llama3 inverter to map layer-$\ell$ activations of $\mathcal{M}_1$ back to an approximate input $\hat{x}$. This $\hat{x}$ is then given to $\mathcal{M}_2$ along with a prompt to answer, still without direct activation patching. If this "inversion → zero-shot" pipeline approaches the performance of Patchscopes / LIT, it indicates that the effective component is the residual input information in activations combined with the verbalizer’s commonsense, rather than any "privileged processing" by the target. The paper also compares single-layer ($\ell=15$) versus multi-layer averages to ensure consistent conclusions across patch intensities.

**3. Knowledge Mismatch Experiment: Directly measuring faithfulness**

While the first two experiments identify shortcuts in benchmarks, they do not prove verbalizer unfaithfulness; this group is the core faithfulness test. Ours constructs (subject, relation, object) triples and splits them into two categories: (a) $\mathcal{M}_1$ knows but $\mathcal{M}_2$ does not (e.g., fine-tuning $\mathcal{M}_1$ on a new fact); (b) $\mathcal{M}_2$ knows but $\mathcal{M}_1$ does not. The verbalization output is then compared against the independent zero-shot outputs of both models. If verbalization favors (a), it describes the target's knowledge; if it favors (b), it is fabricating an "explanation of the target" using its own commonsense. Results align with (b): in conflicts, the verbalizer frequently fabricates answers the target does not possess, providing lethal evidence of unfaithfulness.

The models and configurations used are: $\mathcal{M}_1$ / $\mathcal{M}_2$ selected from Llama3.1-8B-Instruct and Ministral-8B-Instruct; LIT uses the LatentQA dataset to fine-tune the verbalizer; for cross-family verbalization, an extra affine map is learned to project activations from Llama3 space to Ministral space.

## Key Experimental Results

### Main Results
Feature extraction across 6 categories on Llama3 / Ministral, with $\mathcal{M}_1 = \mathcal{M}_2$, averaged over layers 1-15 (Table 1):

| Method | country_curr | food_country | ath_pos | ath_sport | prod_comp | star_const | Average |
|------|--------------|--------------|---------|-----------|-----------|------------|------|
| Llama3 LIT | 0.79 | 0.45 | 0.66 | 0.84 | 0.67 | 0.41 | 0.64 |
| Llama3 Patchscopes | 0.31 | 0.21 | 0.41 | 0.73 | 0.32 | 0.28 | 0.38 |
| **Llama3 zero-shot** | **0.82** | **0.58** | 0.59 | 0.76 | 0.67 | 0.43 | **0.64** |
| Ministral LIT | 0.77 | 0.48 | 0.59 | 0.78 | 0.67 | 0.39 | 0.61 |
| Ministral Patchscopes | 0.14 | 0.01 | 0.22 | 0.61 | 0.47 | 0.15 | 0.27 |
| **Ministral zero-shot** | **0.85** | 0.45 | 0.57 | **0.83** | **0.68** | **0.67** | **0.68** |

Zero-shot achieved the same average score as LIT (0.64 vs 0.64), and was actually higher on Ministral (0.68 vs 0.61)—implying verbalization provides no "privileged" information.

### Ablation Study
Cross-model verbalization ($\mathcal{M}_1 = $ Llama3, $\mathcal{M}_2 = $ Ministral, Table 2):

| Method | $\mathcal{M}_2$ | Average Accuracy | Note |
|------|----------------|------------|------|
| Patchscopes | Llama3 (Same) | ~0.12 | Single layer $\ell=15$ |
| Patchscopes | Ministral (Cross) | ~0.08 | Cross-family collapse |
| LIT | Llama3 | ~0.64 | Original work |
| LIT | Ministral | ~0.08 | Fine-tuning doesn't save it |

Activation Inversion (Table 3-4): Using T5-Base to invert activations back to approximate inputs for Llama3 to answer yielded an average of 0.27 (0.42 for country_curr), which is higher than the 0.10 of single-layer Patchscopes ($\ell=15$). This demonstrates that the combination of inversion + zero-shot is sufficient to explain all of Patchscopes' "performance." Most of LIT's high score (0.79) also stems from inversion + verbalizer knowledge.

### Key Findings
- **Zero-shot matches LIT**: Contradicts the claim that "LIT reveals target internal states"—the results are identical whether activations are queried or not for the same model/input.
- **Patchscopes underperforms zero-shot**: On most tasks, Patchscopes performs worse than simply asking the verbalizer directly, suggesting patching activations introduces noise rather than revelation.
- **Cross-family verbalization fails**: After mapping, Patchscopes averages ~0.08 and LIT ~0.08 even with fine-tuning, showing heavy reliance on the "$\mathcal{M}_1 = \mathcal{M}_2$" condition.
- **Fabrication under knowledge mismatch**: When $\mathcal{M}_2$ knows a fact but $\mathcal{M}_1$ does not, verbalization outputs $\mathcal{M}_2$'s answer—meaning it fabricates knowledge the target lack, direct evidence of unfaithfulness.
- **Inversion explains nearly everything**: Inverting activations back to approximate inputs for a zero-shot model replicates most Patchscopes performance, meaning their "success" is a mix of activation leakage and verbalizer common sense, with no "privileged" component.

## Highlights & Insights
- **Minimalist yet lethal counterfactual design**: The zero-shot baseline, a simple operation often overlooked, directly shatters the methodological assumptions of the entire subfield. It validates the irreplaceable value of "baselining" in AI research.
- **Conceptual framework of "Privileged Information"**: Borrowing the concept of "privileged knowledge" from epistemology (Alston 1971) provides a clear evaluation criterion for interpretability—the standard for whether verbalization is valid.
- **Inversion as a null hypothesis**: Using an inversion model to construct the alternative explanation—that activations merely contain input information—is a clever counterfactual. Once inversion can achieve verbalization's performance, the original methods lose their justification.
- **Critical ICML paper**: Rather than inventing new methods, it disproves old ones, which is vital for the methodological health of academia; such work is especially scarce and necessary in the LLM era.
- **Questioning both benchmarks and methods**: The authors point out that many verbalization benchmarks are inherently flawed (not requiring privileged information), suggesting future research should fix evaluations first.

## Limitations & Future Work
- **Scope limited to feature extraction and factual recall**: Does not cover complex use cases like behavioral explanation, reasoning traces, or dangerous knowledge detection; conclusions might not generalize fully.
- **Lack of a proposed repair**: While the critique is clear, a complete solution for "how to design benchmarks that truly test privileged information" is not provided, only a call for controlled tasks.
- **Cross-family affine mapping may be under-tuned**: The failure of cross-family methods might stem from poor mapping rather than fundamental impossibility; more thorough controls are needed.
- **Binary "know vs don't" labels**: Knowledge mismatch experiments rely on a binary judgment of what $\mathcal{M}_1$ or $\mathcal{M}_2$ knows, whereas model knowledge is a probability distribution.
- **Compute for Inversion**: The training costs for T5-Base/Llama3 inverters were not fully disclosed, which might affect the perceived strength of this "alternative explanation."
- **Future Work**: The authors suggest designing synthetic tasks with "knowledge unique to the target model" for ground-truth testing.

## Related Work & Insights
- **vs Ghandeharioun 2024 (Patchscopes)**: Patchscopes claims to reveal LLM computation; ours uses zero-shot counter-examples to debunk this claim. This is a direct falsification.
- **vs Pan 2026 (LIT)**: LIT fine-tunes a verbalizer to learn activations; ours proves LIT's high scores are explained by "inversion + verbalizer knowledge" and fail cross-family.
- **vs Belrose 2023 (TunedLens) / nostalgebraist 2020 (logitlens)**: These lens methods are special cases of Patchscopes and face identical critiques regarding whether descriptions convey target-specific information.
- **vs VQA prior bias (Goyal 2017)**: Similar to the shortcut problem of "answering correctly without looking at the intended input," ours ports this critique paradigm to LLM interpretability.
- **Insight**: This "counterfactual evaluation" approach should be extended to nearly all LLM evaluations—any benchmark claiming to require special abilities/inputs should be tested against a zero-shot baseline.

## Rating
- Novelty: ⭐⭐⭐⭐ While not inventing a method, systematically falsifying a whole subfield's assumptions via a three-pronged counterfactual framework is fundamentally novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers dual model families, 6 feature extraction categories, Patchscopes/LIT, two types of inverters, and single/multi-layer averages. Limited only by the QA-style extraction tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear arguments and progressive controls (zero-shot → inversion → mismatch) with standard statistical significance reporting.
- Value: ⭐⭐⭐⭐⭐ Provides a "brake" for the interpretability community, forcing future verbalization work to prove their benchmarks cannot be shortcut, potentially changing the research paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](../../NeurIPS2025/interpretability/do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICML 2026\] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICML 2026\] In Defense of Information Leakage in Concept-based Models](in_defense_of_information_leakage_in_concept-based_models.md)

</div>

<!-- RELATED:END -->
