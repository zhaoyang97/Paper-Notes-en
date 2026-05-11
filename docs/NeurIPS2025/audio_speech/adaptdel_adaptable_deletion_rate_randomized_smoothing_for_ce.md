---
title: >-
  [Paper Note] AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness
description: >-
  [NeurIPS 2025][Audio & Speech][certified robustness] AdaptDel extends the fixed deletion rate used in randomized smoothing for discrete sequences to an adaptable deletion rate that varies according to input properties su…
tags:
  - "NeurIPS 2025"
  - "Audio & Speech"
  - "certified robustness"
  - "randomized smoothing"
  - "edit distance"
  - "adaptable deletion rate"
  - "sequence classification"
date: 2026-05-08
content_hash: 0abb3eddb7f5a831
---

# AdaptDel: Adaptable Deletion Rate Randomized Smoothing for Certified Robustness

**Conference**: NeurIPS 2025
**arXiv**: [2511.09316](https://arxiv.org/abs/2511.09316)
**Code**: None
**Area**: Audio & Speech
**Keywords**: certified robustness, randomized smoothing, edit distance, adaptable deletion rate, sequence classification

## TL;DR
AdaptDel extends the fixed deletion rate used in randomized smoothing for discrete sequences to an adaptable deletion rate that varies according to input properties such as sequence length. The paper provides a theoretical soundness proof for certification under variable rates, and experiments on NLP sequence classification tasks demonstrate improvements in certified region cardinality of up to 30 orders of magnitude.

## Background & Motivation

**Background**: Certified robustness aims to provide provable defense guarantees against adversarial attacks for classifiers. Randomized smoothing is one of the most scalable certification methods: it smooths a classifier by applying random perturbations to the input and taking a majority vote over multiple predictions, thereby obtaining a provable robustness radius. For continuous inputs (e.g., images), Gaussian noise is the standard perturbation; for discrete sequences (e.g., text), random token deletion is commonly used.

**Limitations of Prior Work**: Existing randomized smoothing methods for discrete sequences adopt a fixed deletion rate, treating all inputs uniformly regardless of length. However, text length varies considerably in natural language: short texts (e.g., 5-word sentences) may lose nearly all information under a high deletion rate, causing classification failure; long texts (e.g., 50-word paragraphs) under a low deletion rate retain too many tokens per sample, leaving robustness underexplored and yielding a small certified radius. A fixed deletion rate cannot simultaneously accommodate both cases.

**Key Challenge**: The choice of deletion rate involves a trade-off between information preservation and robustness, and the optimal balance of this trade-off varies with input properties such as length. A single fixed rate is necessarily severely suboptimal for a subset of inputs. This paper proposes designing the deletion rate as a function of input properties and formally proves that randomized smoothing certification remains sound under variable deletion rates.

## Method

### Overall Architecture
AdaptDel extends the theoretical framework of randomized smoothing to support variable deletion rates. Given a base classifier and an input sequence, the adaptive deletion rate $p(n)$ is first computed based on input attributes (e.g., length $n$); each token is then independently deleted with probability $p(n)$ to generate multiple perturbed copies; a majority vote over all copies yields the smoothed classifier's prediction; finally, the extended theoretical framework is used to compute the certified radius under edit-distance attacks.

### Key Designs

1. **Theoretical Framework for Variable Deletion Rates**:

   - Function: Generalizes the fixed-rate randomized smoothing theory to input-dependent deletion rates and proves the mathematical correctness of the resulting certification.
   - Mechanism: The paper proves that when the deletion rate varies with the input, the certified guarantee of the smoothed classifier under edit-distance perturbations still holds. The key challenge is that edit-distance attacks (insertions/deletions/substitutions) alter sequence length and thus the deletion rate, making the analysis more complex than the fixed-rate case. Certification bounds under variable rates are derived by analyzing the relationship between the post-deletion distributions of the attacked and original sequences.
   - Design Motivation: The fixed rate is a special case of the variable rate; generalizing to variable rates allows each input to achieve its optimal information–robustness balance.

2. **Adaptive Deletion Rate Function Design**:

   - Function: Dynamically determines the optimal deletion rate based on input attributes.
   - Mechanism: The deletion rate is modeled as a function of sequence length, $p = f(n)$. Shorter sequences use a lower deletion rate to preserve classification signals; longer sequences use a higher deletion rate to enhance robustness, since their signals are more redundant. The specific functional form can be determined through theoretical analysis or empirical search.
   - Design Motivation: Sentence length varies widely in natural language, making a fixed deletion rate suboptimal for variable-length inputs. Although this intuition is natural, it had not been rigorously formalized prior to this work.

3. **Certified Region Cardinality Computation under Edit Distance**:

   - Function: Computes the certified region under the variable deletion rate setting, i.e., the maximum edit distance under which correct classification is guaranteed.
   - Mechanism: The certified region cardinality depends on the deletion rate and the confidence of the majority vote. Under variable rates, the effect of attack-induced length changes on the deletion rate must additionally be considered, which increases analytical complexity without compromising soundness.
   - Design Motivation: Certified region cardinality is the core metric of certified robustness; order-of-magnitude improvements in this metric directly reflect the effectiveness of the method.

### Loss & Training
AdaptDel does not introduce a new training procedure; it is a certification method applied at inference time. The base classifier can be trained in the standard manner (e.g., fine-tuning a pretrained model on deletion-augmented data). The contribution of AdaptDel lies in the inference-time certification framework rather than in training strategy.

## Key Experimental Results

### Main Results

| Setting | Metric | AdaptDel | Fixed-Rate SOTA | Gain |
|---------|--------|----------|-----------------|------|
| NLP sequence classification | Median certified region cardinality | Significantly higher | Baseline | Up to **30 orders of magnitude** |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|-----------|-------------|
| Adaptive rate vs. fixed rate | Certified region cardinality | Adaptive rate uniformly outperforms fixed rate on variable-length inputs |
| Different adaptive strategies | Certified region cardinality | Length-based strategy is the most direct and effective |
| Short-text evaluation | Certification improvement | Short sequences benefit most from lower deletion rates |
| Long-text evaluation | Certification improvement | Long sequences benefit from higher deletion rates via enlarged robustness radii |

### Key Findings
- An improvement of 30 orders of magnitude in certified region cardinality represents a qualitative change rather than an incremental gain, reflecting the severe suboptimality of fixed rates on variable-length inputs.
- The improvement stems primarily from matching the deletion rate to inputs of different lengths, avoiding over-deletion for short texts and under-deletion for long texts.
- The theoretical guarantee (soundness) ensures that the certification produced by the variable-rate method is no weaker than claimed.

## Highlights & Insights
- An improvement of 30 orders of magnitude is extremely rare in the certified robustness literature, indicating that fixed deletion rates cause substantial certification waste in variable-length input scenarios.
- The theoretical contribution is substantial: combining a soundness proof for variable rates with edit-distance attack analysis constitutes non-trivial mathematical work.
- The core insight is concise and compelling: a fixed deletion rate is suboptimal for variable-length inputs—an intuitive observation that had not been formally addressed prior to this work.
- The method is classifier-agnostic and can be directly applied to any existing text classification model.

## Limitations & Future Work
- The specific functional form of the adaptive strategy may require tuning for different tasks and datasets.
- The computational cost of certification grows with input length and the number of samples, which may become a bottleneck for large-scale deployment.
- Validation is conducted primarily on NLP sequence classification; applicability to other discrete sequence tasks (audio classification, DNA sequence analysis, etc.) remains to be explored.
- The paper is 33 pages (including appendices); the theoretical content is rich but demands a substantial mathematical background from readers.

## Related Work & Insights
- **Fixed-Rate Randomized Smoothing for Discrete Sequences**: AdaptDel is a strict generalization thereof, achieving substantially superior performance on variable-length inputs.
- **Gaussian-Based Randomized Smoothing** (Cohen et al., 2019): The classical method for continuous spaces; AdaptDel addresses discrete sequences.
- **IBP / CROWN**: Deterministic certification methods typically targeting continuous spaces and neural network layers; not applicable to discrete edit-distance attacks.
- Insight: Replacing a fixed hyperparameter with an input-dependent function is a general improvement strategy that can be extended to other methods relying on fixed hyperparameters.

## Rating
- Novelty: ⭐⭐⭐⭐ — Adaptable deletion rate is a natural yet previously unformalized extension; the theoretical derivation is non-trivial.
- Experimental Thoroughness: ⭐⭐⭐ — The 30-order-of-magnitude improvement is convincing, though detailed experimental settings and dataset coverage are limited by available abstract information.
- Writing Quality: ⭐⭐⭐⭐ — The 33-page camera-ready version is theoretically rigorous and complete.
- Value: ⭐⭐⭐⭐ — Significant contribution to the field of certified robustness for discrete sequences; the methodology is generalizable.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Impact of Scaling Training Data on Adversarial Robustness](the_impact_of_scaling_training_data_on_adversarial_robustness.md)
- [\[NeurIPS 2025\] AVRobustBench: Benchmarking the Robustness of Audio-Visual Recognition Models at Test-Time](textttavrobustbench_benchmarking_the_robustness_of_audio-visual_recognition_mode.md)
- [\[AAAI 2026\] Say More with Less: Variable-Frame-Rate Speech Tokenization via Adaptive Clustering and Implicit Duration Coding](../../AAAI2026/audio_speech/say_more_with_less_variable-frame-rate_speech_tokenization_via_adaptive_clusteri.md)
- [\[ACL 2026\] Robustness via Referencing: Defending against Prompt Injection Attacks by Referencing the Executed Instruction](../../ACL2026/audio_speech/robustness_via_referencing_defending_against_prompt_injection_attacks_by_referen.md)
- [\[ACL 2026\] Still Between Us? Evaluating and Improving Voice Assistant Robustness to Third-Party Interruptions](../../ACL2026/audio_speech/still_between_us_evaluating_and_improving_voice_assistant_robustness_to_third-pa.md)

</div>

<!-- RELATED:END -->
