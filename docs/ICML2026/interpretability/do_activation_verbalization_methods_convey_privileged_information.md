---
title: >-
  [Paper Note] Do Activation Verbalization Methods Convey Privileged Information?
description: >-
  [ICML 2026][Interpretability][Patchscopes] This paper systematically demonstrates that the performance of popular activation verbalization methods (Patchscopes / LIT / SelfIE) as LLM interpretability tools can be entirely explained by the "verbalizer model's own knowledge," requiring no internal activations from the target model. This implies that these tools a
tags:
  - ICML 2026
  - Interpretability
  - Patchscopes
  - LIT
date: 2026-05-08
content_hash: 23be4932e78c5630
---
# Do Activation Verbalization Methods Convey Privileged Information?

**Conference**: ICML 2026  
**arXiv**: [2509.13316](https://arxiv.org/abs/2509.13316)  
**Code**: https://github.com/millicentli/verb_faithfulness  
**Area**: Interpretability / LLM Probing / Benchmark Criticism  
**Keywords**: Activation Verbalization, Patchscopes, LIT, Faithfulness, Privileged Knowledge

## TL;DR
This paper systematically demonstrates that the performance of popular activation verbalization methods (Patchscopes / LIT / SelfIE) as LLM interpretability tools can be entirely explained by the "verbalizer model's own knowledge," requiring no internal activations from the target model. This implies that these tools appear effective on existing benchmarks due to design flaws in the benchmarks themselves, and that when the verbalizer's knowledge exceeds the target's, it fabricates "explanations" that the target does not actually possess.

## Background & Motivation
**Background**: Understanding the internal representations of LLMs is a core challenge in interpretability. Recently, a class of "verbalization" methods has emerged—using a second LLM (verbalizer $\mathcal{M}_2$) to translate the hidden states of a target model ($\mathcal{M}_1$) into natural language descriptions. Representative works include Patchscopes (patching token activations into corresponding prompt positions), SelfIE (a similar homologous approach), and LIT (fine-tuning a verbalizer to learn an activation matrix of all tokens in a layer). These methods are promoted as tools for "understanding LLM computation."

**Limitations of Prior Work**: The key assumption that "the verbalizer's output reflects the target's internal representation" has never been rigorously tested. Since the verbalizer is itself an LLM with inherent world knowledge, it is impossible to distinguish whether its answers rely on the activations provided by the target or on its own common sense. If it can answer correctly purely based on common sense, the "explanation" holds no value for interpretability—it explains the world, not the model.

**Key Challenge**: Interpretability requires the verbalizer to convey "privileged information" (information obtainable only through internal activations). However, the powerful parametric knowledge of LLMs allows them to answer correctly on most tasks based on input text alone, making "answering via activations" indistinguishable from "answering without activations."

**Goal**: (1) Examine whether existing benchmarks truly require the verbalizer to use the target's activations; (2) Construct controlled experiments to distinguish whether "knowledge comes from the target or the verbalizer"; (3) Observe which source the verbalizer prioritizes during knowledge conflicts.

**Key Insight**: Criticize verbalization as an NLP "shortcut learning" problem—if a model can answer correctly without looking at the inputs it is supposed to examine, the evaluation itself contains a shortcut, analogous to prior bias in VQA.

**Core Idea**: Design three sets of contrasts: (a) A zero-shot baseline where $\mathcal{M}_2$ sees the input without activations; (b) Activation inversion to map activations back to input text to check information content; (c) Knowledge mismatch experiments where $\mathcal{M}_2$ knows a fact that $\mathcal{M}_1$ does not, to see whose answer the verbalizer reports.

## Method

### Overall Architecture
This paper is a critical evaluation study rather than a new model proposal. It addresses the untested assumption of whether verbalizer output reflects target internal activations or the verbalizer's own knowledge. It transforms the problem into three progressive counterfactual contrasts: first, using a zero-shot baseline to measure performance without activations; second, using activation inversion to see if activations contain information beyond the input text; and finally, using knowledge mismatch experiments to see whom the verbalizer trusts during conflicts. All three experiments share a common evaluation criterion—counting as correct if the output contains the ground-truth substring (case-insensitive), aligning with prior verbalization work—and use the McNemar test with Bonferroni correction for significance testing.

### Key Designs

**1. Zero-shot Baseline: Testing Activation Necessity with Naive Counterfactuals**

Verbalization methods are recommended as interpretability tools under the premise that "activations provide information unavailable from the input alone," but this has not been falsified. This study targets six feature extraction datasets used by Patchscopes / LIT (country_curr / food_country / ath_pos / ath_sport / prod_comp / star_const). By setting $\mathcal{M}_1 = \mathcal{M}_2 = $ Llama3.1-8B-Instruct or Ministral-8B-Instruct and directly asking $\mathcal{M}_2$ the $x_{\text{input}}$ + question without any activation patching, the authors compare results against LIT and Patchscopes (averaged over layers 1-15). This is a strict necessity test: if performance without activations matches or exceeds verbalization, the marginal contribution of activations is zero or negative, undermining the legitimacy of these methods.

**2. Activation Inversion: Constructing "Activation = Lossy Copy of Input" as an Alternative Explanation**

Even if one maintains that the success of Patchscopes stems from extra information in activations, a trivial explanation must be ruled out: that the information is merely a restatement of the input text. To test this, the authors train a T5-Base or Llama3 inverter to map layer-$\ell$ activations of $\mathcal{M}_1$ back to an approximate input $\hat{x}$, then provide $\hat{x}$ to $\mathcal{M}_2$ for prompting and answering. If this "inversion $\rightarrow$ zero-shot answering" pipeline approaches the performance of Patchscopes / LIT, it suggests that the effective components are residual input information in the activation plus the verbalizer's common sense, rather than any "privileged processing" by the target.

**3. Knowledge Mismatch Experiments: Directly Measuring Faithfulness**

The first two experiments only show that benchmarks have shortcuts; they do not prove the verbalizer is unfaithful. This set is the core faithfulness test. The authors construct (subject, relation, object) triplets and split them into two categories: (a) $\mathcal{M}_1$ knows the fact but $\mathcal{M}_2$ does not (e.g., by fine-tuning $\mathcal{M}_1$ on a new fact); (b) $\mathcal{M}_2$ knows but $\mathcal{M}_1$ does not. The verbalization output is then compared against the independent zero-shot outputs of both models. If verbalization favors (a), it truly describes target knowledge; if it favors (b), it fabricates an "explanation of the target" using its own common sense. Results lean toward (b): when knowledge conflicts, the verbalizer frequently fabricates answers the target does not possess.

## Key Experimental Results

### Main Results
Performance on 6 feature extraction tasks using Llama3 / Ministral, where $\mathcal{M}_1 = \mathcal{M}_2$, averaged over layers 1-15 (Table 1):

| Method | country_curr | food_country | ath_pos | ath_sport | prod_comp | star_const | Average |
|------|--------------|--------------|---------|-----------|-----------|------------|------|
| Llama3 LIT | 0.79 | 0.45 | 0.66 | 0.84 | 0.67 | 0.41 | 0.64 |
| Llama3 Patchscopes | 0.31 | 0.21 | 0.41 | 0.73 | 0.32 | 0.28 | 0.38 |
| **Llama3 zero-shot** | **0.82** | **0.58** | 0.59 | 0.76 | 0.67 | 0.43 | **0.64** |
| Ministral LIT | 0.77 | 0.48 | 0.59 | 0.78 | 0.67 | 0.39 | 0.61 |
| Ministral Patchscopes | 0.14 | 0.01 | 0.22 | 0.61 | 0.47 | 0.15 | 0.27 |
| **Ministral zero-shot** | **0.85** | 0.45 | 0.57 | **0.83** | **0.68** | **0.67** | **0.68** |

Zero-shot performance matches LIT (0.64 vs 0.64), and is even higher on Ministral (0.68 vs 0.61)—implying verbalization provides no "privileged" information.

### Ablation Study
Cross-model verbalization ($\mathcal{M}_1 = $ Llama3, $\mathcal{M}_2 = $ Ministral, Table 2):

| Method | $\mathcal{M}_2$ | Avg Accuracy | Description |
|------|----------------|------------|------|
| Patchscopes | Llama3 (Same) | ~0.12 | Single layer $\ell=15$ |
| Patchscopes | Ministral (Cross) | ~0.08 | Collapse across families |
| LIT | Llama3 | ~0.64 | Same family |
| LIT | Ministral | ~0.08 | Cross-family fine-tuning fails |

Activation Inversion (Table 3-4): Using T5-Base to invert activations and then letting Llama3 answer yields an average of 0.27 (0.42 for country_curr), which is higher than the 0.10 for Patchscopes at a single layer ($\ell=15$). This shows that the combination of inversion and zero-shot answering is sufficient to explain all of Patchscopes' "performance." Most of LIT's high score (0.79) also comes from inversion plus verbalizer knowledge.

### Key Findings
- **Zero-shot matches LIT**: Contradicts the claim that "LIT reveals the target's internal state"—using the same model and input, results are the same whether activations are provided or not.
- **Patchscopes < zero-shot**: On most tasks, Patchscopes performs worse than directly asking the verbalizer, suggesting that patching activations introduces noise rather than revelation.
- **Cross-family verbalization failure**: After affine mapping, Patchscopes averages ~0.08, and LIT only ~0.08 despite fine-tuning, showing extreme dependence on the $\mathcal{M}_1 = \mathcal{M}_2$ condition.
- **Fabrication under mismatch**: When $\mathcal{M}_2$ knows a fact and $\mathcal{M}_1$ does not, verbalization outputs $\mathcal{M}_2$'s answer, meaning it fabricates knowledge the target does not have—direct evidence of unfaithfulness.
- **Inversion explains everything**: Inverting activations back to approximate inputs for a zero-shot model reproduces most Patchscopes performance, meaning its "success" is a combination of activation leakage and verbalizer common sense.

## Highlights & Insights
- **Simple yet lethal counterfactual design**: The zero-shot baseline, though simple, directly punctures the methodological assumptions of the entire subfield, validating the value of baseline testing in AI research.
- **Conceptual framework of "Privileged Information"**: Borrowing "privileged knowledge" from epistemology (Alston 1971) provides a clear evaluation criterion for interpretability research.
- **Inversion as a null hypothesis**: Using inversion to construct the alternative explanation of "input information present in activations" is a clever counterfactual.
- **Critical ICML paper**: Falsifying existing methods rather than inventing new ones is essential for the methodological health of the field, especially in the era of large models.
- **Questioning both benchmarks and methods**: The authors highlight that benchmarks are flawed for not requiring privileged information; future research must fix evaluation first.

## Limitations & Future Work
- **Scope limited to feature extraction and factual recall**: Does not cover complex use cases like behavioral explanation or reasoning traces; conclusions might not fully generalize.
- **Lack of a definitive fix**: While the criticism is clear, a complete proposal for designing benchmarks that truly test privileged information is not provided.
- **Cross-family affine mapping**: Failure might be due to suboptimal mapping rather than fundamental impossibility; more thorough controls are needed.
- **Knowledge binary labels**: The "know vs. don't know" labels in mismatch experiments are discrete, whereas model knowledge is probabilistic.
- **Future Direction**: The authors suggest designing synthetic tasks where only the target model possesses certain knowledge for ground-truth testing.

## Related Work & Insights
- **vs Ghandeharioun 2024 (Patchscopes)**: Directly falsifies the claim that Patchscopes reveals LLM computation by showing zero-shot counter-examples.
- **vs Pan 2026 (LIT)**: Proves that LIT's high scores are explained by inversion and verbalizer knowledge, failing in cross-family scenarios.
- **vs VQA prior bias (Goyal 2017)**: Similar "shortcut" issue where models answer correctly without the intended input; this paper ports this critical paradigm to LLM interpretability.
- **Insight**: This "counterfactual evaluation" should be extended to almost all LLM evaluations—any benchmark claiming to require special capability or input should be tested against a zero-shot baseline.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematic falsification using a tripartite framework (zero-shot, inversion, mismatch) is framework-level novelty.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers various model families, feature types, and both Patchscopes/LIT, though tasks are limited to QA-style extraction.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear arguments, progressive experiments, and rigorous statistical significance reporting.
- Value: ⭐⭐⭐⭐⭐ Essential for "braking" the community, forcing future verbalization work to prove benchmarks cannot be bypassed via shortcuts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](../../NeurIPS2025/interpretability/do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICML 2026\] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[CVPR 2025\] L-SWAG: Layer-Sample Wise Activation with Gradients Information for Zero-Shot NAS on Vision Transformers](../../CVPR2025/interpretability/l-swag_layer-sample_wise_activation_with_gradients_information_for_zero-shot_nas.md)

</div>

<!-- RELATED:END -->
