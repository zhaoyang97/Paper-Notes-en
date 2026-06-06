---
title: >-
  [Paper Note] Do Activation Verbalization Methods Convey Privileged Information?
description: >-
  [ICML 2026][Interpretability][Activation Verbalization] This paper systematically proves: the performance of current popular activation verbalization methods (Patchscopes / LIT / SelfIE) when used as LLM interpretability…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Activation Verbalization"
  - "Patchscopes"
  - "LIT"
  - "Faithfulness"
  - "Privileged Information"
date: 2026-05-08
content_hash: 1b2a935e3ec84cd3
---

# Do Activation Verbalization Methods Convey Privileged Information?

**Conference**: ICML 2026  
**arXiv**: [2509.13316](https://arxiv.org/abs/2509.13316)  
**Code**: https://github.com/millicentli/verb_faithfulness  
**Area**: Interpretability / LLM Probing / Benchmark Critique  
**Keywords**: Activation Verbalization, Patchscopes, LIT, Faithfulness, Privileged Information

## TL;DR
This paper systematically proves: the performance of current popular activation verbalization methods (Patchscopes / LIT / SelfIE) when used as LLM interpretability tools can be entirely explained by the "verbalizer model's own knowledge," requiring no internal activations from the target model. This implies these tools appear to work on existing benchmarks due to flaws in benchmark design, and when verbalizer knowledge exceeds that of the target, they fabricate "explanations" that the target does not actually possess.

## Background & Motivation
**Background**: Understanding the internal representations of LLMs is a core challenge in interpretability. Recently, a class of "verbalization" methods has emerged—using a second LLM (verbalizer $\mathcal{M}_2$) to translate the hidden states of a target model (target $\mathcal{M}_1$) into natural language descriptions. Representative works include Patchscopes (patching token activations into corresponding positions in a prompt), SelfIE (a similar concept), and LIT (fine-tuning the verbalizer to learn an activation matrix of all tokens at a layer). These methods are claimed as tools for "understanding LLM computation."

**Limitations of Prior Work**: The key assumption that "the verbalizer's output reflects the target's internal representation" has never been rigorously tested. Since the verbalizer itself is an LLM with inherent world knowledge, it is impossible to distinguish whether its answers rely on the activations provided by the target or its own common sense. If it can answer correctly purely based on common sense, such an "explanation" has no value for interpretability—it explains the world, not the model.

**Key Challenge**: Interpretability requires the verbalizer to convey "privileged information" (information that must be obtained via internal activations); however, the powerful parametric knowledge of LLMs allows them to answer correctly on most tasks based solely on input text. This makes "answering via activations" indistinguishable from "answering without activations."

**Goal**: (1) Examine whether existing benchmarks truly require the verbalizer to use target activations; (2) if not, construct controlled experiments to distinguish whether "knowledge comes from the target or the verbalizer"; (3) observe whom the verbalizer prioritizes during knowledge conflicts.

**Key Insight**: Critique verbalization as an NLP "shortcut learning" problem—if a model can answer correctly without looking at the inputs it intendedly should, the evaluation itself contains shortcuts, analogous to prior bias in VQA.

**Core Idea**: Design three sets of controls: (a) a zero-shot baseline where $\mathcal{M}_2$ is directly shown the input without activations; (b) activation inversion to invert activations back to input text to check information content; (c) knowledge mismatch experiments to intentionally create scenarios where $\mathcal{M}_2$ knows a fact while $\mathcal{M}_1$ does not, to see whose answer the verbalizer reports.

## Method

### Overall Architecture
This study is a critical evaluation and introduces no new models. It revolves around two verification paradigms:

1. **Zero-shot Control**: Using a pre-trained model identical to $\mathcal{M}_2$, concatenate $x_{\text{input}} + x_{\text{prompt}}$ and query it directly without any activation patching. If it matches Patchscopes / LIT on verbalization benchmarks, those benchmarks do not require internal activations.
2. **Activation Inversion + Explanation**: First use T5-Base or Llama3 as an inversion model to map layer-$\ell$ activations of $\mathcal{M}_1$ back to approximate input text $\hat{x}$, then feed $\hat{x}$ to $\mathcal{M}_2$ to answer $x_{\text{prompt}}$. If this "inversion → answer" pipeline achieves performance comparable to Patchscopes, verbalization "success" can be fully explained by "activation = lossy copy of input text + verbalizer knowledge."
3. **Knowledge Mismatch Experiment**: Construct scenarios comparing target knowledge with verbalizer knowledge (target knows but verbalizer doesn't vs. verbalizer knows but target doesn't) to see which leads to more accurate verbalizer reports.

### Key Designs

1. **Zero-shot Baseline as Counterfactual**:
    - **Function**: Measure the ceiling of "answering without activations, looking only at raw input" to confirm the extent of shortcuts in existing evaluations.
    - **Mechanism**: For the 6 feature extraction datasets used by Patchscopes / LIT (country_curr / food_country / ath_pos / ath_sport / prod_comp / star_const), let $\mathcal{M}_1 = \mathcal{M}_2 =$ Llama3.1-8B-Instruct or Ministral-8B-Instruct. Concatenate $x_{\text{input}}$ + question and ask $\mathcal{M}_2$ directly for accuracy. Compare with LIT and Patchscopes (average of layers 1-15). The criterion for correctness is whether the output contains the ground-truth substring (case-insensitive), consistent with prior verbalization work.
    - **Design Motivation**: This is the most stringent "necessity test"—if verbalization can be outperformed without activations, the marginal contribution of activations is negative, and the legitimacy of existing methods as interpretability tools collapses.

2. **Activation Inversion + Alternative Explanation**:
    - **Function**: Reveal that even without activations, verbalizers can achieve similar performance via the "inverted approximate input" shortcut.
    - **Mechanism**: Train a T5-Base or Llama3 inverter to map $\mathcal{M}_1$ layer-$\ell$ activations back to approximate $\hat{x}$; then pass $\hat{x}$ as input to $\mathcal{M}_2$ for standard prompt + answer. If this pipeline achieves comparable performance to Patchscopes / LIT, it proves that the actual effective component is the "residual input information in the activations," not the "special processing of input by the target model." The paper also compares single-layer ($\ell=15$) vs. multi-layer averages to verify consistency across patch strengths.
    - **Design Motivation**: This further strengthens the finding that "zero-shot can already win"—even if one argues that the success of Patchscopes comes from extra information in activations, this section proves that extra information is merely a paraphrase of the input, not "privileged processed knowledge."

3. **Knowledge Mismatch Control Experiment**:
    - **Function**: Distinguish whether the verbalizer reports the target model's knowledge or its own.
    - **Mechanism**: Construct (subject, relation, object) triples in two categories: (a) $\mathcal{M}_1$ knows but $\mathcal{M}_2$ does not (e.g., fine-tuning $\mathcal{M}_1$ on new facts); (b) $\mathcal{M}_2$ knows but $\mathcal{M}_1$ does not. Compare verbalization output with each model's independent zero-shot output: if it favors (a) → it indicates the verbalizer describes target knowledge; if it favors (b) → it indicates fabrication. The paper finds answers are close to (b)—verbalizers frequently fabricate their own knowledge as a surrogate for "target explanations."
    - **Design Motivation**: This is the core "faithfulness" test. While the first two experiments show benchmarks have shortcuts, this directly proves verbalizer unfaithfulness under knowledge conflict—the most damaging finding.

### Loss & Training
No new models are trained. The study primarily uses: (1) Llama3.1-8B-Instruct and Ministral-8B-Instruct as $\mathcal{M}_1$ / $\mathcal{M}_2$; (2) LIT following the LatentQA dataset to fine-tune the verbalizer; (3) Cross-family verbalization using an affine map to project activations from Llama3 space into Ministral space. All significance tests use the McNemar test with Bonferroni correction.

## Key Experimental Results

### Main Results
6 feature extraction tasks on Llama3 / Ministral, $\mathcal{M}_1 = \mathcal{M}_2$, average of layers 1-15 (Table 1):

| Method | country_curr | food_country | ath_pos | ath_sport | prod_comp | star_const | Average |
|------|--------------|--------------|---------|-----------|-----------|------------|------|
| Llama3 LIT | 0.79 | 0.45 | 0.66 | 0.84 | 0.67 | 0.41 | 0.64 |
| Llama3 Patchscopes | 0.31 | 0.21 | 0.41 | 0.73 | 0.32 | 0.28 | 0.38 |
| **Llama3 zero-shot** | **0.82** | **0.58** | 0.59 | 0.76 | 0.67 | 0.43 | **0.64** |
| Ministral LIT | 0.77 | 0.48 | 0.59 | 0.78 | 0.67 | 0.39 | 0.61 |
| Ministral Patchscopes | 0.14 | 0.01 | 0.22 | 0.61 | 0.47 | 0.15 | 0.27 |
| **Ministral zero-shot** | **0.85** | 0.45 | 0.57 | **0.83** | **0.68** | **0.67** | **0.68** |

Zero-shot matches the LIT average (0.64 vs 0.64), and Zero-shot is even higher on Ministral (0.68 vs 0.61)—implying verbalization provides no "privileged" information.

### Ablation Study
Cross-model verbalization ($\mathcal{M}_1 = $ Llama3, $\mathcal{M}_2 = $ Ministral, Table 2):

| Method | $\mathcal{M}_2$ | Average Accuracy | Description |
|------|----------------|------------|------|
| Patchscopes | Llama3 (Same) | ~0.12 | Single layer $\ell=15$ |
| Patchscopes | Ministral (Cross) | ~0.08 | Cross-family collapse |
| LIT | Llama3 | ~0.64 | Same-family baseline |
| LIT | Ministral | ~0.08 | Cross-family fine-tuning fails |

Activation inversion (Table 3-4): Using T5-Base to invert activations to approximate input then having Llama3 answer yields an average of 0.27 (country_curr 0.42), which is higher than Patchscopes single-layer ($\ell=15$) at 0.10—showing the inversion + zero-shot combination is sufficient to explain the entirety of Patchscopes' "performance." Most of LIT's high score (0.79) also stems from inversion + verbalizer's own knowledge.

### Key Findings
- **zero-shot matches LIT**: Directly punctures the claim that "LIT reveals target internal states"—results are identical with or without querying activations for the same model and input.
- **Patchscopes inferior to zero-shot**: On most tasks, Patchscopes performance is lower than directly querying the verbalizer, suggesting that patching activations introduces noise rather than revealing insights.
- **Cross-family verbalization nearly fails**: After affine mapping, Patchscopes averages ~0.08, and LIT even after re-fine-tuning only reaches ~0.08, indicating these methods rely heavily on the "cheating" condition of $\mathcal{M}_1 = \mathcal{M}_2$.
- **Fabrication under knowledge mismatch**: When $\mathcal{M}_2$ knows a fact that $\mathcal{M}_1$ does not, the verbalization output reflects $\mathcal{M}_2$'s answer—meaning it fabricates knowledge the target does not possess, providing direct evidence of unfaithfulness.
- **Inversion explains almost everything**: Inverting activations back to approximate inputs for a zero-shot model reproduces most of Patchscopes' performance, meaning "success" is a combination of activation leakage and verbalizer common sense, without "privileged" components.

## Highlights & Insights
- **Simple yet lethal control design**: A baseline as simple as zero-shot punctures the methodological assumptions of an entire subfield, validating the irreplaceable value of simple baselines in AI research.
- **Conceptual framework of "Privileged Information"**: Borrowing the concept from epistemology (Alston 1971) provides a clear evaluation criterion for interpretability research—a standard for whether verbalization is valid.
- **Activation inversion as a null hypothesis**: Using an inversion model to construct an alternative explanation for "input information in activations" is a clever counterfactual—once inversion matches verbalization performance, the latter loses its explanatory power.
- **Critical ICML paper**: Falsifying existing methods rather than inventing new ones is essential for the methodological health of the academic community, especially in the LLM era.
- **Questioning both benchmarks and methods**: The authors point out that many verbalization benchmarks are fundamentally flawed (not requiring privileged information), suggesting future research must first reform evaluation.

## Limitations & Future Work
- **Limited to feature extraction and factual recall**: Does not cover complex verbalization use cases like behavioral explanation, reasoning traces, or dangerous knowledge detection; conclusions might not fully generalize.
- **No repair solution proposed**: While the critique is clear, a complete guide on how to design benchmarks that truly test privileged information is missing.
- **Cross-family affine mapping may be sub-optimal**: Failure could be due to poorly learned mappings rather than fundamental impossibility; it requires more thorough experiments.
- **Reliance on binary "know/not know" labels**: Knowledge in models follows a probability distribution, making binary classification of knowledge mismatch somewhat fuzzy.
- **Compute cost for inversion models**: Costs are not fully disclosed, which might affect the perceived strength of the "alternative explanation."
- **Future Work**: The authors suggest designing synthetic tasks with "knowledge unique to the target model" for ground-truth testing.

## Related Work & Insights
- **vs Ghandeharioun 2024 (Patchscopes)**: Patchscopes claims to reveal LLM computation; this paper uses zero-shot counter-examples to disprove that claim. It is a direct falsification relationship.
- **vs Pan 2026 (LIT)**: Shows LIT's high scores are explained by inversion and verbalizer knowledge and that it fails across model families.
- **vs Belrose 2023 (TunedLens) / nostalgebraist 2020 (logitlens)**: These lens methods are special cases of Patchscopes and are subject to the same critiques regarding whether they convey target-specific information.
- **vs VQA prior bias work (Goyal 2017)**: Similar shortcut learning issues where models answer without looking at the intended input; this paper migrates that critique to LLM interpretability.
- **Insight**: This "counterfactual evaluation" should be extended to almost all LLM evaluations—any benchmark claiming to require special abilities/inputs should be tested against a zero-shot baseline.

## Rating
- Novelty: ⭐⭐⭐⭐ Not inventing a method, but the "zero-shot + inversion + knowledge mismatch" triad systematically falsifies methodological assumptions in a framework-level novel way.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers two model families across 6 tasks, uses two types of inverters, and tests single/multi-layer averages; however, tasks are limited to QA-style extraction.
- Writing Quality: ⭐⭐⭐⭐⭐ Arguments are clear, experiments progress logically, and statistical significance is well-documented.
- Value: ⭐⭐⭐⭐⭐ Serves as a "brake" for the interpretability community, forcing future work to account for shortcuts and potentially shifting research paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](../../NeurIPS2025/interpretability/do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICML 2026\] On the Relationship Between Activation Outliers and Feature Death in Sparse Autoencoders](on_the_relationship_between_activation_outliers_and_feature_death_in_sparse_auto.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] Information Shapes Koopman Representation](../../ICLR2026/interpretability/information_shapes_koopman_representation.md)

</div>

<!-- RELATED:END -->
