---
title: >-
  [Paper Note] AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence
description: >-
  [ICML 2025][Code Intelligence][Process Reward Model] Proposes AdaptiveStep, a method that automatically divides reasoning steps based on model prediction confidence to train a more precise Process Reward Model (ASPRM). On mathematical reasoning and code generation tasks, it surpasses existing open-source PRMs at less than 70% of the data construction cost, and further enhances reasoning performance through token-level value-guided decoding.
tags:
  - "ICML 2025"
  - "Code Intelligence"
  - "Process Reward Model"
  - "Reasoning Step Division"
  - "Model Confidence"
  - "Token-level Value-guided Decoding"
  - "Mathematical Reasoning"
date: 2026-05-08
content_hash: eaaf345b42caabfb
---

# AdaptiveStep: Automatically Dividing Reasoning Step through Model Confidence

**Conference**: ICML 2025  
**arXiv**: [2502.13943](https://arxiv.org/abs/2502.13943)  
**Code**: [https://github.com/Lux0926/ASPRM](https://github.com/Lux0926/ASPRM)  
**Area**: LLM Reasoning / Process Reward Model  
**Keywords**: Process Reward Model, Reasoning Step Division, Model Confidence, Token-level Value-guided Decoding, Mathematical Reasoning

## TL;DR
Proposes AdaptiveStep, a method that automatically divides reasoning steps based on model prediction confidence to train a more precise Process Reward Model (ASPRM). On mathematical reasoning and code generation tasks, it surpasses existing open-source PRMs at less than 70% of the data construction cost, and further enhances reasoning performance through token-level value-guided decoding.

## Background & Motivation
Process Reward Models (PRMs) provide finer-grained feedback than Outcome Reward Models (ORMs) by giving reward signals for each step in the reasoning process, thereby guiding LLMs to generate higher-quality reasoning responses. However, existing PRMs face a core problem: **the division of reasoning steps is too coarse**.

The current mainstream practice relies on rule-based step division, such as splitting by newline characters or a fixed number of tokens. However, this approach has two key limitations: (1) Model confidence at newline characters is often very high, meaning these positions are not real "decision points" and carry low information; (2) In fields like code generation, it is difficult to define universal splitting rules. Although manual annotation can produce high-quality step divisions, it is highly costly and heavily dependent on expert knowledge.

The authors draw inspiration from cognitive science—Kahneman pointed out that human deep thinking accounts for only about 2% of total thinking, with critical reasoning decisions concentrated at a few nodes. Inspired by this, the authors propose **letting the model itself tell us where the key decision points are**: when the model's prediction confidence for the next token is low, it indicates that this position is a decision point where an important choice must be made, which should serve as the boundary between steps.

## Method

### Overall Architecture
The overall workflow of AdaptiveStep consists of three steps: (1) sampling responses and collecting the confidence distribution of each token; (2) dividing reasoning steps based on a confidence threshold and annotating the reward of each step through rollouts; (3) training the PRM using the annotated data, and optionally applying the PRM for Token-level Value-guided Decoding (TVD) to enhance reasoning.

### Key Designs

1. **Based-on-confidence Step Division (AdaptiveStep)**:
    - **Function**: Automatically segments reasoning responses into multiple highly informative reasoning steps.
    - **Mechanism**: For the $i$-th token in the generated response $s^n$, its confidence is defined as $c_{s_i^n} = p(s_i^n | \pi, q, s_{<i}^n)$, which is the probability of the model predicting this token. After collecting the confidence distributions of all samples, a threshold $\tau$ (based on a certain percentage of the token count, set to 2% in the paper) is defined. Token positions with confidence below the threshold are treated as step division points. Thus, the response $s^n$ is divided into $K$ reasoning steps $\{r_1, r_2, ..., r_K\}$.
    - **Design Motivation**: Low-confidence positions represent difficult decision points faced by the model—such as calculations in mathematical expressions, semantic vocabulary choices, or the determination of final answers. Statistical analysis shows that 3.85% of tokens in mathematical expressions contribute 21.03% of decision tokens, and only 2.7% of decision tokens appear at newline characters, confirming the inefficiency of rule-based division.

2. **Rollout-based Step Reward Estimation**:
    - **Function**: Estimates the target reward value for each divided reasoning step.
    - **Mechanism**: Perform $J$ rollouts starting from each step $r_k$, and use Hard Estimation (HE) to judge whether any rollout path can reach the correct answer. The target reward is:
     $$r_k^e = \begin{cases} 1, & \exists j \in [J], \{r_1,...,r_k,t_j\} \text{ is correct} \\ 0, & \text{otherwise} \end{cases}$$
    - **Design Motivation**: By performing rollouts at decision points, the reward signal for each step is more accurate because the end of the step is precisely where the decision occurs.

3. **Token-level Value-guided Decoding (TVD)**:
    - **Function**: Leverages the PRM during the reasoning phase to guide token selection in real-time, without requiring additional sampling.
    - **Mechanism**: During decoding, when the model encounters a low-confidence position ($c_p < \tau$), it takes the top $M$ candidate tokens with the highest probabilities, scores each candidate with the PRM, and selects the token with the highest score:
     $$s_i = \arg\max_{s_i^m \in s_i^*} R^\theta(p, s_{<i}, s_i^m)$$
    - **Design Motivation**: Traditional PRMs are only used for post-hoc evaluation in Best-of-N. TVD embeds the PRM into the generation process to achieve fine-grained, real-time guidance. Since it only intervenes at low-confidence positions, the computational overhead is controllable.

### Loss & Training
The PRM is trained using binary cross-entropy loss:
$$\mathcal{L}_{PRM}^\theta = -\sum_{k=1}^{K} (r_k^e \log r_k^\theta + (1 - r_k^e) \log(1 - r_k^\theta))$$

Training data construction: each data point is sampled 30 times and deduplicated, with 8 rollouts per step, finally generating approximately 388k mathematical PRM training samples and 49k code PRM samples. The threshold is set to 2%, meaning that approximately 2% of the tokens will act as step decision boundaries.

## Key Experimental Results

### Main Results

| Dataset | Metric | ASPRM | Prev. SOTA | Gain |
|--------|------|-------|----------|------|
| GSM8k (BoN, N=64) | Accuracy | 90.45 (ASPRM-L) | 88.70 (ER-PRM) | +1.75 |
| MATH500 (TVD) | Accuracy | 42.00 (ASPRM-L) | 38.80 (Greedy) | +3.20 |
| GSM8k (TVD) | Accuracy | 83.47 (ASPRM-L) | 81.80 (Greedy) | +1.67 |
| LeetCodeDataset (TVD) | Pass@1 | 28.00 | 26.28 (Greedy) | +1.72 |
| LiveCodeBench (TVD) | Pass@1 | 19.92 | 19.21 (Greedy) | +0.71 |

Note: Under TVD, Math-Shepherd and ER-PRM caused performance degradation on GSM8k (lower than Greedy), whereas ASPRM consistently brings improvements.

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Threshold 0.5% | BoN GSM8k is lower | Too few division points, insufficient information |
| Threshold 1.0% | Performance increases incrementally | Discriminative power is enhanced under more decision points |
| Threshold 2.0% | Best | Matches the 2% deep thinking ratio in cognitive science |
| L→M Transfer | Bo64 drops, but TVD can improve | Training data across different models shows some transferability, but it is limited |
| Mixed Math + Code | Math Bo64 86.35↑, MATH500 TVD 29.00↑ | Cross-domain data can mutually enhance each other |

### Key Findings
- **Information in AdaptiveStep division is much higher than in rule-based division**: In mathematical tasks, only 2.7% of decision tokens are newline characters, whereas 29% are at conjunctions, and 21% are within mathematical expressions.
- **In code tasks, 80% of decision points are in code comments**, of which 91% are of the "planning the next step" type, demonstrating that the model is most uncertain when "thinking".
- **Significant cost advantage in data construction**: ASPRM uses only a single model, 30 samples, and 8 rollouts, costing less than 70% of Math-Shepherd and ER-PRM.
- **Cross-domain generalization**: Mathematical PRMs can provide effective guidance on code tasks (LeetCodeDataset BoN 34.29↑), and vice versa.
- **Generalization over scoring positions**: The performance of ASPRM barely drops under random scoring positions, whereas models trained on newline splitting vary greatly under different settings.

## Highlights & Insights
- It uses the model's own confidence as the step division signal, which is a simple and elegant idea supported by cognitive science (Kahneman's 2% deep thinking).
- The TVD strategy upgrades PRM from "post-hoc evaluation" to "real-time guidance". It only intervenes at low-confidence positions, resulting in minimal computational overhead and significant performance gains.
- It open-sources a function-level LeetCode dataset (including test cases and a sandbox), filling the gap in code PRM training data.
- Mixed training with cross-domain data is a practical, low-cost trick to enhance PRMs.

## Limitations & Future Work
- The 2% threshold is not optimal for all models; stronger models may require less training data (the paper observes this but does not explore adaptive threshold selection deeply).
- Generating training data with a single model limits transferability; ASPRM-M's performance on MATH500 is inferior to baseline models built using multiple models.
- PRM training data for code tasks is harder to obtain (49k vs 388k); scaling up the data could further improve performance.
- Although TVD only intervenes at low-confidence positions, it still requires additional PRM inference, which may introduce latency in extremely long generation scenarios.

## Related Work & Insights
- **vs Math-Shepherd**: Also uses rollout annotation, but divides steps using newline characters. It requires multi-model construction, making it more expensive and less informative.
- **vs ER-PRM**: Uses 16 rollouts (ASPRM only uses 8), incurring higher construction costs, but performs worse than ASPRM on GSM8k.
- **vs Token-level PRM (OmegaPRM)**: Scores at every token or a fixed number of tokens, which is extremely costly to annotate; ASPRM only scores at decision points, showing superior efficiency.
- **vs MCTS-based decoding**: TVD is more lightweight and does not require full tree search.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of using confidence to divide steps is natural and effective, though the core technical components (rollouts, PRM training) are standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers both mathematical and code domains, evaluates on both BoN and TVD, and includes transferability, generalization, threshold, and feature analysis.
- Writing Quality: ⭐⭐⭐⭐ The structure is clear, figures/tables are rich and intuitive, and the analysis is in-depth.
- Value: ⭐⭐⭐⭐ High practical value; it reduces PRM construction costs while boosting performance, offering a significant reference for PRM research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation](reasoning_through_execution_unifying_process_and_outcome_rewards_for_code_genera.md)
- [\[ICML 2025\] EffiCoder: Enhancing Code Generation in Large Language Models through Efficiency-Aware Fine-tuning](efficoder_enhancing_code_generation_in_large_language_models_through_efficiency-.md)
- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](../../ACL2026/code_intelligence/codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)
- [\[NeurIPS 2025\] Preserving LLM Capabilities through Calibration Data Curation: From Analysis to Optimization](../../NeurIPS2025/code_intelligence/preserving_llm_capabilities_through_calibration_data_curation_from_analysis_to_o.md)
- [\[ACL 2025\] CoCo-Bench: A Comprehensive Code Benchmark for Multi-task Large Language Model Evaluation](../../ACL2025/code_intelligence/coco-bench_a_comprehensive_code_benchmark_for_multi-task_large_language_model_ev.md)

</div>

<!-- RELATED:END -->
