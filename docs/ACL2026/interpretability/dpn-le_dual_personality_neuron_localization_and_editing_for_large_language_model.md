---
title: >-
  [Paper Note] DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models
description: >-
  [ACL2026][Interpretability][Personality Editing] This paper proposes DPN-LE, which localizes mutually exclusive personality-related neurons by comparing MLP activations of high/low personality trait samples. By interveni…
tags:
  - "ACL2026"
  - "Interpretability"
  - "Personality Editing"
  - "Neuron Localization"
  - "Sparse Intervention"
  - "Big Five"
  - "Representation Analysis"
date: 2026-05-08
content_hash: 6a4d4594bac56c98
---

# DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models

**Conference**: ACL2026 Findings  
**arXiv**: [2604.27929](https://arxiv.org/abs/2604.27929)  
**Code**: https://github.com/Z1ivan/DPN-LE  
**Area**: LLM Interpretability / Model Editing  
**Keywords**: Personality Editing, Neuron Localization, Sparse Intervention, Big Five, Representation Analysis

## TL;DR
This paper proposes DPN-LE, which localizes mutually exclusive personality-related neurons by comparing MLP activations of high/low personality trait samples. By intervening in only approximately 0.5% of neurons, it achieves personality control while better preserving general capabilities compared to existing large-scale neuron editing methods.

## Background & Motivation
**Background**: LLM personality control is commonly used in role-playing, sociological surveys, personalized assistants, and personality analysis. Existing methods are generally categorized into prompt-based personality induction and neuron-editing. The former is simple but unstable, while the latter directly intervenes in internal representations but often requires modifying a large number of neurons.

**Limitations of Prior Work**: The representative neuron editing method NPTI can alter personality traits but leads to significant capability degradation. Preliminary experiments in this paper show that on LLaMA-3-8B-Instruct, NPTI causes an average drop on GSM8K of 16.00% in the high direction and 40.79% in the low direction, indicating that many modified neurons are related to general reasoning or knowledge.

**Key Challenge**: Personality-related representations are not independent switches completely separated from general capabilities. Neurons exhibit polysemanticity; coarse-grained editing simultaneously affects personality, knowledge, and reasoning, leading to a strong trade-off between personality control and capability preservation.

**Goal**: The authors aim to identify which neurons are truly related to personality traits and design a sparser, more selective inference-time intervention method to control Big Five personality expression without retraining the model.

**Key Insight**: The paper observes that high/low personality trait samples exhibit mutually exclusive separation patterns in the activation space of specific MLP layers. Therefore, trait-exclusive neurons can be identified through high-low sample comparison.

**Core Idea**: Construct a steering vector using the mean activation difference between high/low trait samples, then combine it with a dual screening process using Cohen's $d$ and activation magnitude to retain only statistically significant and strongly responsive personality-exclusive neurons for sparse linear intervention.

## Method
DPN-LE is a training-free inference-time editing method. It does not modify model weights but adds or subtracts a personality-direction steering signal from selected MLP hidden neurons during generation. The method consists of three steps: steering vector construction, dual-direction neuron selection, and sparse intervention.

### Overall Architecture
Given a Big Five trait (e.g., Neuroticism), the authors prepare 1,000 pairs of high-trait and low-trait contrastive samples. For each Transformer layer's MLP hidden state, activations are extracted at the last token position. A layer-wise steering vector is calculated from the mean difference of high and low samples. Then, Cohen's $d$ is computed for each neuron to filter mutually exclusive neuron sets for high and low trait directions. During inference, if the goal is to enhance the trait, a positive intervention is applied along the steering vector to the selected neurons; if the goal is to suppress it, a negative intervention is applied.

### Key Designs
1.  **Steering Vector Construction**:
    - **Function**: Constructs the representation direction of high traits relative to low traits for each layer.
    - **Mechanism**: For the $l$-th layer MLP hidden state, calculate $s_l = mean(h_l^+) - mean(h_l^-)$, where $h_l^+$ and $h_l^-$ come from high-trait and low-trait samples respectively. This vector represents the average shift of the personality trait in that layer's activation space.
    - **Design Motivation**: Personality is not a local phenomenon of a single token or prompt; using the mean difference of paired samples reduces noise and captures stable personality directions.

2.  **Dual-Direction Neuron Selection**:
    - **Function**: Selects a sparse subset of neurons that truly distinguish high/low personality directions.
    - **Mechanism**: A neuron must satisfy both $|d_l| > \tau_d$ and $|s_l| > \tau_q$. Cohen's $d$ ensures the difference is statistically significant, while the steering magnitude quantile ensures the response is strong enough. Neurons with $d_l > \tau_d$ enter the high set, and $d_l < -\tau_d$ enter the low set.
    - **Design Motivation**: Relying only on effect size selects too many weakly responsive neurons, while relying only on magnitude might select statistically unstable activation differences. Dual criteria better exclude redundant neurons related to general language processing.

3.  **Sparse Intervention and Weighted Variant**:
    - **Function**: Controls personality using minimal neurons during inference while preserving other capabilities.
    - **Mechanism**: DPN-LE applies $h_i \leftarrow h_i + \gamma s_i$ uniformly to selected neurons; DPN-LEw assigns weights $w_i \in [0.75, 1.0]$ based on $|d_l|$ ranking when selecting more neurons, allowing stronger intervention for more personality-exclusive neurons.
    - **Design Motivation**: Under the Q995 setting, only about 70 neurons per layer are selected, which is already sparse enough for uniform intervention. When the threshold is relaxed, weighted intervention mitigates instability caused by low-specificity neurons.

### Loss & Training
DPN-LE involves no training loss and does not fine-tune the model. It only uses 1,000 pairs of contrastive samples to calculate activation statistics. On LLaMA-3-8B-Instruct, the intervention layers are 12-31; on Qwen2.5-7B-Instruct, they are 14-27. For LLaMA, key hyperparameters are quantile threshold $q=0.995$, Cohen's $d$ threshold $\tau_d=0.8$, and intervention strength $\gamma \in [0.0, 2.0]$. Qwen uses a lower $\tau_d=0.3$ due to weaker activation differences. The default configuration selects approximately 0.5% of the total MLP neurons.

## Key Experimental Results

### Main Results
| Task / Metric | Ours (DPN-LE) | Compared To | Key Number | Conclusion |
| :--- | :--- | :--- | :--- | :--- |
| PersonalityBench Avg Score | DPN-LE 9.11 | NPTI 9.43 | Scores near SOTA | Sparse intervention effectively controls personality |
| Modified Neurons | DPN-LE Avg High 711 / Low 713 | NPTI Avg High 21,223 / Low 22,140 | Reduced by 96.7% | Most NPTI neurons are redundant |
| GSM8K Performance Drop | DPN-LEw Avg High -7.08%, Low -5.93% | NPTI High -16.00%, Low -40.79% | Significantly better preservation | Sparse selection reduces reasoning damage |
| HotpotQA F1 Drop | DPN-LEw High -2.05, Low -2.27 | NPTI High -1.04, Low -2.81 | Comparable or better than NPTI | Small loss in QA capability |
| TriviaQA F1 Drop | DPN-LEw High -2.88, Low -3.80 | NPTI High -3.61, Low -4.34 | Lower degradation | Knowledge QA is well-preserved |
| IPIP-NEO-300 total | DPN-LEw 6.64, DPN-LE 6.75 | P2P 7.71, LLaMA Few-shot 5.96 | Better than some prompt methods | Trade-off in individual-level personality matching |

### Ablation Study
| Configuration | Key Metrics | Description |
| :--- | :--- | :--- |
| $\gamma=0.8$ | trait score 8.02, fluency 9.85 | Balanced personality control and fluency |
| $\gamma=1.0$ | trait score 8.59, fluency 9.33 | Stronger control but fluency starts to drop |
| $\gamma=1.5$ | DPN-LE fluency 5.42, DPN-LEw fluency 6.58 | Excessive intervention breaks generation; weighted is more stable |
| Q999 0.1% | trait 7.55, fluency 9.90 | Too few neurons, insufficient control |
| Q995 0.5% | trait 8.59, fluency 9.33 | Optimal balance point |
| Q970 3.0% | trait 8.68, fluency 7.78 | Selecting more neurons barely improves trait but significantly hurts fluency |

### Key Findings
- On LLaMA, only about 72 neurons per layer are needed on average, and about 92 neurons for Qwen, to form a usable personality intervention subset.
- DPN-LE significantly outperforms NPTI in capability preservation, though some trait directions still damage reasoning (e.g., DPN-LEw Extraversion-low drops 17.89% on GSM8K).
- DPN-LEw is more stable under strong intervention, indicating that when the neuron set expands, weighting by effect size reduces side effects from low-specificity neurons.

## Highlights & Insights
- The most significant insight is that "personality neurons" are not "the more the better." The key to personality control lies in excluding general capability-related neurons rather than expanding the intervention scope.
- The dual-criteria screening is practical: Cohen's $d$ addresses statistical significance while steering magnitude addresses intervention strength. This combination is more rational than a single threshold.
- The method is training-free and does not change weights, modifying sparse activations during inference. It is suitable as both an interpretability research tool and for analyzing overlaps between traits and capabilities.

## Limitations & Future Work
- DPN-LE relies on contrastive samples; whether these samples represent true personality expression directly affects the steering vector quality.
- Although capability degradation is lower than NPTI, some personality directions still share neural foundations with reasoning, especially for Extraversion and Neuroticism.
- This paper focuses on single-trait intervention; multi-trait combinations, trait conflicts, and long-term dialogue stability have yet to be verified.
- Individual-level alignment on IPIP-NEO-300 is weaker than PAS and NPTI, suggesting a trade-off remains between sparse capability preservation and fine-grained personality fitting. Future work could include reasoning-protective neuron selection to explicitly exclude neurons highly correlated with reasoning tasks.

## Related Work & Insights
- **vs Simple Prompt / P2P**: Prompt methods are easy to deploy but depend on phrasing and lack stability; DPN-LE acts directly on the representation layer, making it better for analyzing personality mechanisms.
- **vs PAS**: PAS searches for attention heads and activation offsets, leaning towards optimization-based personality alignment; DPN-LE focuses on mutually exclusive MLP neuron representations.
- **vs NPTI**: NPTI modifies approximately 20,000 neurons, providing strong control but heavy capability degradation; DPN-LE modifies only about 0.5% of neurons with better preservation.
- **Insight**: When performing internal LLM editing, identifying a "truly exclusive" sparse subset using contrastive activations and task capability evaluations is more stable than simply increasing the editing scope.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Framing personality editing as dual-direction sparse neuron localization is clear and distinct from large-scale editing.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers personality, general capability, generalization, and ablation, though multi-trait combinations are missing.
- Writing Quality: ⭐⭐⭐⭐☆ Methodological formulas and experimental conclusions are clear, though some tables are dense.
- Value: ⭐⭐⭐⭐☆ Provides reference value for personality control, model editing, and representation interpretability, especially for capability-preserving interventions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in Large Language Models](../../ICML2026/interpretability/dual_mechanisms_of_value_expression_intrinsic_vs_prompted_values_in_large_langua.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[ACL 2026\] Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models](sparse_feature_coactivation_reveals_causal_semantic_modules_in_large_language_mo.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](../../ICML2026/interpretability/towards_atoms_of_large_language_models.md)
- [\[ACL 2026\] Embracing Anisotropy: Turning Massive Activations into Interpretable Control Knobs for Large Language Models](embracing_anisotropy_turning_massive_activations_into_interpretable_control_knob.md)

</div>

<!-- RELATED:END -->
