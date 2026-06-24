---
title: >-
  [Paper Note] Capability-Based Scaling Trends for LLM-Based Red-Teaming
description: >-
  [ICLR 2026][LLM Alignment][Red-teaming] Four jailbreak methods were systematically evaluated on over 600 attacker-target LLM pairs. The study found that the Attack Success Rate (ASR) follows a sigmoid scaling law ($R^2=0.83$) relative to the attacker-target capability gap, which can be quantified using the logit-transformed MMLU-Pro scores.
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Red-teaming"
  - "Jailbreak Attacks"
  - "Capability Scaling"
  - "Safety Evaluation"
  - "Attack Success Rate"
date: 2026-05-08
content_hash: 457e2ef31fe423cd
---

# Capability-Based Scaling Trends for LLM-Based Red-Teaming

**Conference**: ICLR 2026  
**arXiv**: [2505.20162](https://arxiv.org/abs/2505.20162)  
**Code**: [https://github.com/kotekjedi/capability-based-scaling](https://github.com/kotekjedi/capability-based-scaling)  
**Area**: Human Understanding / AI Safety / LLM Alignment  
**Keywords**: Red-teaming, Jailbreak Attacks, Capability Scaling, Safety Evaluation, Attack Success Rate

## TL;DR
Four jailbreak methods were systematically evaluated on over 600 attacker-target LLM pairs. The study found that the Attack Success Rate (ASR) follows a sigmoid scaling law ($R^2=0.83$) relative to the attacker-target capability gap, which can be quantified using the logit-transformed MMLU-Pro scores.

## Background & Motivation

**Background**: LLM red-teaming evaluates model safety by simulating attacks. Existing research typically evaluates only a small number of model pairs, lacking a systematic understanding of how ASR changes with model capabilities.

**Limitations of Prior Work**: ASR varies significantly across different attack methods and performs inconsistently across different model pairs. There is no unified framework to predict the attack vulnerability of new model combinations. Every new model release requires a full re-evaluation.

**Key Challenge**: Safety evaluation is resource-intensive as every pair requires testing. Can scaling laws be used to predict vulnerability instead of performing full tests? Furthermore, as model capabilities improve, will human red-teaming eventually become obsolete?

**Goal**: Discover and quantify the scaling relationship between ASR and the model capability gap to provide a predictive framework for safety evaluation.

**Key Insight**: Using logit-transformed MMLU-Pro scores as a unified proxy for "capability" allows for the calculation of the capability gap between the attacker and the target.

**Core Idea**: The jailbreak success rate is a sigmoid function of the attacker-target capability gap—the stronger the attacker and the weaker the target, the higher the success rate; ASR drops sharply when the target surpasses the attacker.

## Method

### Overall Architecture
This paper does not propose new attacks but characterizes red-teaming as a measurable physical phenomenon to answer whether ASR is determined by identifiable factors and if it is predictable. The approach involves three steps: first, "unlocking" all attacker models by stripping away refusals caused by safety alignment so they are willing to attack with full force; second, running four human-like jailbreak attacks (PAIR, TAP, PAP, Crescendo) across 600+ attacker-target pairs (including 29 target models and closed-source frontier models), recording the ASR for each pair based on the top 50 harmful behaviors from HarmBench (using the best-of-25 attempts); finally, aligning the ASR of each pair with the "capability gap" to fit a functional relationship. Capabilities are proxied by MMLU-Pro scores, resulting in a curve that maps capability differences to ASR.

> This is a scaling trend and measurement analysis paper. The core contribution is a regression curve rather than a multi-module system. The pipeline is linear: "Unlocking → Batch Attacks → Recording ASR → Curve Fitting." Since a framework diagram would provide low information density, the Mermaid diagram is omitted. The following designs follow the measurement sequence.

### Key Designs

**1. Model Unlocking: Stripping Refusals to Measure Attack Power**

The first step of measurement excludes a confounding factor: if an attacker refuses to cooperate due to its own safety alignment, the test measures "willingness" rather than "capability." Following the observation that safety alignment is "shallow" and easily reversed, LoRA fine-tuning is applied to all attackers to remove safety guardrails. A mixture of BadLlama and Shadow Alignment datasets (~1500 harmful samples) is used for training to remove refusal behavior while preserving general capabilities. Success is verified via direct HarmBench queries to ensure the models are fully unlocked. Consequently, the relationship between $\delta$ and ASR reflects pure capability effects.

**2. Capability Gap Metric: Compressing "Who is Stronger" into a Real Number**

To regress ASR against a "capability gap," a scalar measure of relative strength is needed. Directly comparing MMLU-Pro accuracy is ineffective because accuracy is bounded in $[0,1]$, saturated at the ends, and inconsistently spaced. This work applies a logit transformation $\text{logit}(p)=\log\!\big(p/(1-p)\big)$ to map scores to the real number line, defining the capability gap as:

$$\delta = \text{logit}(\text{Attacker MMLU-Pro}) - \text{logit}(\text{Target MMLU-Pro})$$

$\delta>0$ indicates a strong-to-weak attack, while $\delta<0$ indicates the target is stronger (weak-to-strong). This allows vulnerability to be dominated by a single scalar, enabling horizontal comparison across model families and parameter scales.

**3. Sigmoid Scaling Law: Predicting ASR for Any Combination**

Using $\delta$, a linear regression of ASR against $\delta$ is fitted for each target model in logit space and then mapped back to probability space, yielding a sigmoid curve. As the attacker strengthens or the target weakens, ASR approaches 1; as the target surpasses the attacker, ASR drops toward 0. While slopes and offsets vary by target, the shape remains consistent, with median parameters $k=1.73, b=-0.79$ and an overall fit of $R^2=0.83$. The practical value lies in prediction: knowing the MMLU-Pro scores for two models allows for the estimation of ASR, saving the immense cost of exhaustive pair-wise evaluation and providing a quantitative basis for the obsolescence of human red-teaming.

## Key Experimental Results

### Main Results (600+ Attacker-Target Pairs)

| Attack Method | Model Pairs | avg ASR vs MMLU-Pro $\rho$ | $R^2$ (Sigmoid Fit) | Key Findings |
| :--- | :--- | :--- | :--- | :--- |
| PAIR | 600+ | >0.88 | 0.83 | Most fundamental LLM attack |
| TAP | 600+ | >0.85 | ~0.80 | Highest overall ASR |
| PAP | 600+ | >0.82 | ~0.78 | Persuasion-oriented attack |
| Crescendo | 600+ | >0.80 | ~0.75 | Multi-turn attack |

### Key Scaling Trends

| Trend | Quantification | Implication |
| :--- | :--- | :--- |
| Stronger Models = Better Attackers | avg ASR linearly correlates with MMLU-Pro ($\rho>0.88$) | Capability gains simultaneously increase attack power |
| Stronger Models = Harder to Breach | Target ASR decreases as MMLU-Pro increases ($R^2=0.83$) | The downward trend is predictable |
| Capability Gap Dominates | ASR relies primarily on $\delta$ (gap) rather than absolute capability | Relative capability is more important than absolute power |
| Social Science > STEM | Strongest correlation in Psychology/Health/Philosophy | Persuasiveness is more critical than raw knowledge |

## Highlights & Insights
- **Predictive Tool**: Knowing MMLU-Pro scores allows for predicting the approximate ASR of red-teaming, potentially saving thousands of GPU hours in evaluation costs.
- **Quantifying Safety Investment**: The rightward shift of the sigmoid can measure the "equivalent capability boost" from safety training—Llama 3 shows a larger shift than Qwen 2.5, reflecting higher safety investment.
- **Quantitative Assessment of Weaponization Risk**: As open-source capabilities grow, the pool of available attacker power increases. A new 70B open-source model release necessitates a re-evaluation of the attack surface for all deployed systems.
- **Utility of Judges**: Expensive closed-source judges do not significantly impact the final ASR, suggesting the community can save on API costs for evaluation.

## Ablation Study

| Analysis Dimension | Findings |
| :--- | :--- |
| MMLU-Pro Sub-category Correlation | Social science (Psychology, Health, Philosophy) correlates most strongly with ASR, exceeding STEM categories. |
| Judge Impact | Strong judges improve ASR@1 (selection effect) but do not impact ASR@25 (generation quality). |
| Attack Method Comparison | TAP is strongest overall; Crescendo's original success is attributed more to the GPT-4 attacker than the method itself. |
| Llama 2 Anomalies | Four early Llama models deviate from the trend (over-refusal + adversarial training), while subsequent versions return to the trend. |
| Human Red-Teaming Prediction | Assuming a human MMLU-Pro of 0.898, ASR continues to decline as target model capabilities grow. |

## Limitations & Future Work
- MMLU-Pro as a capability proxy might lack precision in specific domains; Llama 2 deviations show that safety training can break the general capability-robustness trend.
- Only four automated attacks (PAIR/TAP/PAP/Crescendo) were tested; manual or ensemble attacks may not follow the same scaling law.
- Model unlocking (LoRA fine-tuning) may not fully restore attack capabilities; some models (e.g., Claude) cannot be fully unlocked via simple fine-tuning.
- Only black-box human-like attacks were considered; white-box attacks (e.g., GCG) might follow different scaling laws.
- The Llama 2 deviation suggests MMLU-Pro is not a perfect proxy for defense; a better proxy might be the FLOPs invested in safety training, which is usually private data.

## Related Work & Insights
- **vs. Ren et al. (2024) Safety Benchmark Analysis**: They first quantified the negative correlation between jailbreak success and capability; this paper further refines it into a sigmoid function.
- **vs. Howe et al. (2025) GCG Scaling**: While they study GCG scaling within model sizes, this paper studies capability gap scaling across model families, providing complementary dimensions.
- **vs. PAIR/TAP/PAP**: Instead of proposing new attacks, this work uses existing ones to reveal scaling laws—emphasizing "patterns" over "methods."
- **Insight—Social Science as a Blind Spot**: Current safety evaluations focus on technical skills (coding, chemistry), but persuasiveness and social engineering are the key predictors of attack success.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery of scaling laws in red-teaming is significant; the sigmoid fit is directly applicable for prediction.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale evaluation across 600+ model pairs, including frontier closed-source models.
- Writing Quality: ⭐⭐⭐⭐ Clear results with high information density in charts.
- Value: ⭐⭐⭐⭐⭐ Provides direct engineering guidance and policy implications for AI safety evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CAGE: A Framework for Culturally Adaptive Red-Teaming Benchmark Generation](cage_a_framework_for_culturally_adaptive_red-teaming_benchmark_generation.md)
- [\[ICLR 2026\] Inverse Reinforcement Learning with Dynamic Reward Scaling for LLM Alignment](inverse_reinforcement_learning_with_dynamic_reward_scaling_for_llm_alignment.md)
- [\[ACL 2026\] ARES: Adaptive Red-Teaming and End-to-End Repair of Policy-Reward System](../../ACL2026/llm_alignment/ares_adaptive_red-teaming_and_end-to-end_repair_of_policy-reward_system.md)
- [\[ICLR 2026\] IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment](ideal_data_equilibrium_adaptation_for_multi-capability_language_model_alignment.md)
- [\[NeurIPS 2025\] Jailbreak-Zero: A Path to Pareto Optimal Red Teaming for Large Language Models](../../NeurIPS2025/llm_alignment/jailbreak-zero_a_path_to_pareto_optimal_red_teaming_for_large_language_models.md)

</div>

<!-- RELATED:END -->
