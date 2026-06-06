---
title: >-
  [Paper Note] Do Activation Verbalization Methods Convey Privileged Information?
description: >-
  [ICML 2026][Interpretability][Activation Verbalization] This work systematically demonstrates that current popular activation verbalization methods (Patchscopes / LIT / SelfIE), when used as LLM interpretability tools…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "Activation Verbalization"
  - "Patchscopes"
  - "LIT"
  - "Faithfulness"
  - "Privileged Knowledge"
date: 2026-05-08
content_hash: 84e401b580ea98f6
---

# Do Activation Verbalization Methods Convey Privileged Information?

**Conference**: ICML 2026  
**arXiv**: [2509.13316](https://arxiv.org/abs/2509.13316)  
**Code**: https://github.com/millicentli/verb_faithfulness  
**Area**: Interpretability / LLM Probing / Benchmark Critique  
**Keywords**: Activation Verbalization, Patchscopes, LIT, Faithfulness, Privileged Knowledge

## TL;DR
This work systematically demonstrates that current popular activation verbalization methods (Patchscopes / LIT / SelfIE), when used as LLM interpretability tools, have their performance fully explained by the "verbalizer model's own knowledge," without requiring any internal activations from the target model. This implies that these tools only appear to work on existing benchmarks due to flaws in benchmark design, and when the verbalizer's knowledge exceeds that of the target, it fabricates "explanations" the target does not possess.

## Background & Motivation
**Background**: Understanding internal representations of LLMs is a core challenge in interpretability. Recently, a class of "verbalization" methods has emerged—using a second LLM (verbalizer $\mathcal{M}_2$) to translate the hidden states of a target model ($\mathcal{M}_1$) into natural language descriptions. Representative works include Patchscopes (patching token activations into the prompt), SelfIE (similar approach), and LIT (fine-tuning the verbalizer to learn the activation matrix for all tokens in a layer). These methods are claimed to be tools for "understanding LLM computation."

**Limitations of Prior Work**: The key assumption that "the verbalizer's output reflects the target's internal representations" has never been rigorously tested. Since the verbalizer itself is an LLM with its own world knowledge, it is unclear whether its answers are based on the target's activations or its own knowledge. If it can answer correctly using only its own knowledge, such "explanations" are meaningless for interpretability—they explain the world, not the model.

**Key Challenge**: Interpretability requires the verbalizer to convey "privileged information" (information only accessible via internal activations). However, the strong parametric knowledge of LLMs allows them to answer most tasks from input text alone, making it indistinguishable whether answers are based on activations or not.

**Goal**: (1) Test whether existing benchmarks require the verbalizer to actually use the target's activations; (2) If not, construct controlled experiments to distinguish whether knowledge comes from the target or the verbalizer; (3) Observe which source the verbalizer prioritizes when knowledge conflicts.

**Key Insight**: Critique verbalization as an NLP "shortcut learning" problem—if a model can answer correctly without looking at the intended input, the evaluation contains a shortcut, analogous to prior bias in VQA.

**Core Idea**: Design three controlled settings—(a) zero-shot baseline where $\mathcal{M}_2$ only sees the input without activations, to measure its standalone performance; (b) activation inversion, mapping activations back to input text to assess information content; (c) knowledge mismatch experiments, where $\mathcal{M}_2$ knows a fact that $\mathcal{M}_1$ does not, to see whose knowledge the verbalizer reports.

## Method

### Overall Architecture
This is a critical evaluation study, not a new model proposal. The work revolves around two main validation paradigms:

1. **Zero-shot Control**: Use a pretrained model identical to $\mathcal{M}_2$, concatenate $x_{\text{input}} + x_{\text{prompt}}$, and query directly, without any activation patching. If it matches Patchscopes / LIT performance on verbalization benchmarks, these benchmarks do not require internal activations.
2. **Activation Inversion + Explanation**: Use T5-Base or Llama3 as an inversion model to map $\mathcal{M}_1$'s activations back to approximate input text $\hat{x}$, then feed $\hat{x}$ to $\mathcal{M}_2$ to answer $x_{\text{prompt}}$. If this "inversion → answer" pipeline matches Patchscopes performance, it shows that verbalization "success" can be fully explained by "activations = lossy copy of input + verbalizer knowledge."
3. **Knowledge Mismatch Experiments**: Construct cases where the target model knows a fact but the verbalizer does not, and vice versa, to see which knowledge source the verbalizer outputs.

### Key Designs

1. **Zero-shot Baseline as Counterfactual**:

    - **Function**: Measures the upper bound of "without activations, only using original input," to assess the degree of shortcut in current evaluations.
    - **Mechanism**: For the six feature extraction datasets used by Patchscopes / LIT (country_curr / food_country / ath_pos / ath_sport / prod_comp / star_const), set $\mathcal{M}_1 = \mathcal{M}_2 =$ Llama3.1-8B-Instruct or Ministral-8B-Instruct, concatenate $x_{\text{input}}$ + question, and query $\mathcal{M}_2$ for accuracy. Compare with LIT and Patchscopes (layer 1-15 average). Judged correct if the output contains the ground-truth substring (case-insensitive), consistent with prior verbalization work.
    - **Design Motivation**: This is the strictest "necessity test"—if verbalization without activations performs as well, the marginal contribution of activations is negative; the legitimacy of these methods as interpretability tools collapses.

2. **Activation Inversion + Alternative Explanation**:

    - **Function**: Reveals that the verbalizer can achieve similar performance via the "inverted approximate input" shortcut, even without activations.
    - **Mechanism**: Train a T5-Base or Llama3 inverter to map $\mathcal{M}_1$'s layer-$\ell$ activations back to approximate $\hat{x}$; then use $\hat{x}$ as input for $\mathcal{M}_2$ with normal prompt + answer. If this pipeline matches Patchscopes / LIT performance, it shows that what matters is "input information residue in activations," not "target model's special processing." The paper also compares single-layer ($\ell=15$) and multi-layer averages to verify consistency across patch strengths.
    - **Design Motivation**: This strengthens Section 3's finding that "zero-shot already suffices"—even if one argues Patchscopes' success comes from extra information in activations, this section shows that extra is just a restatement of the input, not "privileged processed knowledge."

3. **Knowledge Mismatch Control Experiments**:

    - **Function**: Distinguishes whether the verbalizer reports the target model's knowledge or its own.
    - **Mechanism**: Construct (subject, relation, object) triples in two categories—(a) $\mathcal{M}_1$ knows but $\mathcal{M}_2$ does not (e.g., fine-tune $\mathcal{M}_1$ on a new fact); (b) $\mathcal{M}_2$ knows but $\mathcal{M}_1$ does not. Compare verbalization outputs with each model's independent zero-shot outputs: if verbalization aligns with (a), it describes the target's knowledge; if (b), it fabricates. The paper finds answers align with (b)—the verbalizer often fabricates its own knowledge as "explanation" for the target.
    - **Design Motivation**: This is the core "faithfulness" test. The first two experiments only show the benchmark has shortcuts; this directly proves the verbalizer is unfaithful under knowledge conflict—a most damaging finding.

### Loss & Training
No new models are trained. The main setup uses: (1) Llama3.1-8B-Instruct and Ministral-8B-Instruct as $\mathcal{M}_1$ / $\mathcal{M}_2$; (2) LIT uses LatentQA dataset to fine-tune the verbalizer; (3) For cross-family verbalization, an affine map is learned to project activations from Llama3 to Ministral space. All significance tests use McNemar test with Bonferroni correction.

## Key Experimental Results

### Main Results
On Llama3 / Ministral, six feature extraction tasks, $\mathcal{M}_1 = \mathcal{M}_2$, layer 1-15 average (Table 1):

| Method | country_curr | food_country | ath_pos | ath_sport | prod_comp | star_const | Avg |
|--------|--------------|--------------|---------|-----------|-----------|------------|-----|
| Llama3 LIT | 0.79 | 0.45 | 0.66 | 0.84 | 0.67 | 0.41 | 0.64 |
| Llama3 Patchscopes | 0.31 | 0.21 | 0.41 | 0.73 | 0.32 | 0.28 | 0.38 |
| **Llama3 zero-shot** | **0.82** | **0.58** | 0.59 | 0.76 | 0.67 | 0.43 | **0.64** |
| Ministral LIT | 0.77 | 0.48 | 0.59 | 0.78 | 0.67 | 0.39 | 0.61 |
| Ministral Patchscopes | 0.14 | 0.01 | 0.22 | 0.61 | 0.47 | 0.15 | 0.27 |
| **Ministral zero-shot** | **0.85** | 0.45 | 0.57 | **0.83** | **0.68** | **0.67** | **0.68** |

Zero-shot matches LIT average (0.64 vs 0.64), and on Ministral, zero-shot is even higher (0.68 vs 0.61)—indicating verbalization provides no "privileged" information.

### Ablation Study
Cross-model verbalization ($\mathcal{M}_1 =$ Llama3, $\mathcal{M}_2 =$ Ministral, Table 2):

| Method | $\mathcal{M}_2$ | Avg Accuracy | Note |
|--------|----------------|--------------|------|
| Patchscopes | Llama3 (same family) | ~0.12 | Single layer $\ell=15$ |
| Patchscopes | Ministral (cross) | ~0.08 | Cross-family collapse |
| LIT | Llama3 | ~0.64 | Same family |
| LIT | Ministral | ~0.08 | Cross-family fine-tuning fails |

Activation inversion (Tables 3-4): Using T5-Base to invert activations to approximate input, then letting Llama3 answer, achieves an average of 0.27 (country_curr 0.42), which is higher than Patchscopes single layer ($\ell=15$) at 0.10—showing that inversion + zero-shot suffices to explain all Patchscopes "performance." Most of LIT's high scores (0.79) also come from inversion + verbalizer's own knowledge.

### Key Findings
- **Zero-shot matches LIT**: Directly refutes claims that "LIT reveals target internal states"—same model, same input, with or without activations yields the same result.
- **Patchscopes underperforms zero-shot**: On most tasks, Patchscopes performs worse than directly querying the verbalizer, indicating that patching activations adds noise rather than revealing information.
- **Cross-family verbalization nearly fails**: After affine mapping, Patchscopes averages ~0.08, and LIT, even after cross-family fine-tuning, only reaches ~0.08, showing these methods rely heavily on the "cheating" condition $\mathcal{M}_1 = \mathcal{M}_2$.
- **Fabrication under knowledge mismatch**: When $\mathcal{M}_2$ knows a fact but $\mathcal{M}_1$ does not, verbalization outputs $\mathcal{M}_2$'s answer—fabricating knowledge the target does not possess, direct evidence of unfaithfulness.
- **Inversion explains almost everything**: Inverting activations to approximate input and querying a zero-shot model reproduces most Patchscopes performance, indicating "success" is due to input leakage in activations plus verbalizer's own knowledge, with no "privileged" component.

## Highlights & Insights
- **Minimal yet decisive controls**: The zero-shot baseline, seemingly trivial, directly punctures the methodological assumptions of the entire subfield. Demonstrates the irreplaceable value of baselines in AI research.
- **"Privileged information" conceptual framework**: Borrowing the "privileged knowledge" concept from epistemology (Alston 1971), provides a clear evaluation criterion for interpretability—this is the standard for validating verbalization.
- **Activation inversion as null hypothesis**: Using inversion models to construct the alternative explanation that "activations contain input information" is a clever counterfactual—if inversion matches verbalization performance, the latter cannot justify itself.
- **Critical ICML paper**: Rather than inventing new methods, it falsifies old ones, which is crucial for methodological health in academia; such work is especially rare and necessary in the era of large models.
- **Challenges both benchmarks and methods**: The authors point out that many verbalization benchmarks are themselves flawed (not requiring privileged information); future research should first fix evaluation.

## Limitations & Future Work
- **Only tests feature extraction and factual recall**: Does not cover more complex verbalization uses such as behavioral explanation, reasoning trace, or dangerous knowledge detection; conclusions may not fully generalize.
- **No repair proposals**: While the critique is clear, there is no complete proposal for designing benchmarks that truly test privileged information—only a call for controlled tasks at the end.
- **Cross-family affine mapping may be under-optimized**: Cross-family failures may be due to poor mapping rather than fundamental infeasibility; more thorough controls are needed.
- **Relies on binary "knows or not" labels**: In knowledge mismatch experiments, determining "$\mathcal{M}_1$ knows but $\mathcal{M}_2$ does not" is itself fuzzy, as model knowledge is probabilistic.
- **Activation inversion with T5-Base / Llama3 may be compute-intensive**: The cost is not fully disclosed, which may affect the strength of this "alternative explanation."
- **Future directions**: The authors suggest designing synthetic tasks where only the target model has the knowledge as ground-truth tests; this is a very reasonable next step.

## Related Work & Insights
- **vs Ghandeharioun 2024 (Patchscopes)**: Patchscopes originally claimed to reveal LLM computation; this work uses zero-shot counterexamples to refute such claims—a direct falsification.
- **vs Pan 2026 (LIT)**: LIT fine-tunes the verbalizer to learn activations; this work shows LIT's high scores can be explained by "inversion + verbalizer knowledge," and fails cross-family.
- **vs Belrose 2023 (TunedLens) / nostalgebraist 2020 (logitlens)**: These lens methods are special cases of Patchscopes and face the same critique—whether their descriptions truly convey target-specific information.
- **vs VQA prior bias work (Goyal 2017)**: Similarly, "answering correctly without looking at the intended input" is a shortcut problem; this critique paradigm is transplanted to LLM interpretability here.
- **Insights**: This "counterfactual evaluation" approach should be applied to almost all LLM evaluations—any benchmark claiming to require special capability/input should be tested with a zero-shot baseline.

## Rating
- Novelty: ⭐⭐⭐⭐ Not a new method, but the "zero-shot + inversion + knowledge mismatch" triad systematically falsifies the methodological assumptions of the entire verbalization subfield—a framework-level innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Dual model families × 6 feature extraction types × Patchscopes/LIT × two inverter types × single/multi-layer averages, with broad coverage; only limited to QA-style extraction, not complex behavioral explanation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear arguments, stepwise controls (zero-shot → inversion → knowledge mismatch), statistical significance properly marked, each table directly supports the thesis.
- Value: ⭐⭐⭐⭐⭐ Provides a "brake" for the interpretability community, forcing future verbalization work to first prove benchmarks cannot be shortcut—a paradigm-shifting critical paper.

## Related Papers

- [\[ICML 2026\] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking](block-em_preventing_emergent_misalignment_via_latent_blocking.md)
- [\[ICML 2026\] Optimal Attention Temperature Improves the Robustness of In-Context Learning under Distribution Shift in High Dimensions](optimal_attention_temperature_improves_the_robustness_of_in-context_learning_und.md)
- [\[ICML 2026\] Riemannian Generative Decoder：丢掉编码器，在任意黎曼流形上做表示学习](riemannian_generative_decoder.md)
- [\[ICML 2026\] All Circuits Lead to Rome: Rethinking Functional Anisotropy in Circuit and Sheaf Discovery for LLMs](all_circuits_lead_to_rome_rethinking_functional_anisotropy_in_circuit_and_sheaf_.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Do Different Prompting Methods Yield a Common Task Representation?](../../NeurIPS2025/interpretability/do_different_prompting_methods_yield_a_common_task_representation_in_language_mo.md)
- [\[ICML 2026\] Steer Like the LLM: Activation Steering that Mimics Prompting](steer_like_the_llm_activation_steering_that_mimics_prompting.md)
- [\[ICLR 2026\] Concepts' Information Bottleneck Models](../../ICLR2026/interpretability/concepts_information_bottleneck_models.md)
- [\[ICLR 2026\] Information Shapes Koopman Representation](../../ICLR2026/interpretability/information_shapes_koopman_representation.md)
- [\[ACL 2026\] Aligning What LLMs Do and Say: Towards Self-Consistent Explanations](../../ACL2026/interpretability/aligning_what_llms_do_and_say_towards_self-consistent_explanations.md)

</div>

<!-- RELATED:END -->
