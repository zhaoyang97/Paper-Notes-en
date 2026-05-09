---
title: >-
  [Paper Note] Capability-Based Scaling Trends for LLM-Based Red-Teaming
description: >-
  [ICLR 2026][Robotics][Red-teaming] This paper systematically evaluates 4 jailbreak methods across 600+ attacker–target LLM pairs and finds that attack success rate (ASR) follows a sigmoid scaling law with respect to the capability gap between attacker and target ($R^2=0.83$), where the capability gap is quantified via a logit transformation of MMLU-Pro scores.
tags:
  - ICLR 2026
  - Robotics
  - Red-teaming
  - Jailbreak attacks
  - Capability scaling
  - Safety evaluation
  - Attack success rate
date: 2026-05-08
content_hash: 2f038cdb6acf90ac
---

# Capability-Based Scaling Trends for LLM-Based Red-Teaming

**Conference**: ICLR 2026
**arXiv**: [2505.20162](https://arxiv.org/abs/2505.20162)
**Code**: [https://github.com/kotekjedi/capability-based-scaling](https://github.com/kotekjedi/capability-based-scaling)
**Area**: Human Understanding / AI Safety / LLM Alignment
**Keywords**: Red-teaming, Jailbreak attacks, Capability scaling, Safety evaluation, Attack success rate

## TL;DR
This paper systematically evaluates 4 jailbreak methods across 600+ attacker–target LLM pairs and finds that attack success rate (ASR) follows a sigmoid scaling law with respect to the capability gap between attacker and target ($R^2=0.83$), where the capability gap is quantified via a logit transformation of MMLU-Pro scores.

## Background & Motivation

**State of the Field**: LLM red-teaming evaluates model safety by simulating adversarial attacks. Existing studies typically assess only a small number of model pairs and lack a systematic understanding of how ASR varies with model capability.

**Limitations of Prior Work**: ASR varies substantially across attack methods and is inconsistent across different model pairs. No unified framework exists for predicting attack vulnerability in new model combinations, necessitating full re-evaluation upon each new model release.

**Root Cause**: Safety evaluation is resource-intensive—every model pair must be tested individually. Can scaling laws enable prediction rather than exhaustive testing? Moreover, as model capabilities improve, will human red-teaming eventually become ineffective?

**Paper Goals**: Discover and quantify the scaling relationship between ASR and the capability gap between models, providing a predictive framework for safety evaluation.

**Starting Point**: The logit-transformed MMLU-Pro score is adopted as a unified proxy for model capability, and the capability difference between attacker and target is computed accordingly.

**Core Idea**: Jailbreak success rate is a sigmoid function of the attacker–target capability gap—the stronger the attacker and the weaker the target, the higher the ASR; when the target surpasses the attacker in capability, ASR drops sharply.

## Method

### Overall Architecture
ASR is evaluated across combinations of 4 attack methods (PAIR, TAP, PAP, Crescendo) × 25+ attacker models × 25+ target models. The relationship between ASR and the capability difference $\delta = \text{logit}(a_{\text{MMLU}}) - \text{logit}(t_{\text{MMLU}})$ is then analyzed. All attacker models are first unlocked via LoRA fine-tuning to remove safety alignment. Evaluation uses the first 50 harmful behaviors from the HarmBench benchmark, with ASR reported as best-of-25 (maximum success rate over 25 steps) and assessed post-hoc using a neutral HarmBench judge.

### Key Designs

1. **Capability Gap Metric**:

    - MMLU-Pro scores are logit-transformed onto the real line: $\text{logit}(p) = \log(p/(1-p))$
    - Capability gap: $\delta = \text{logit}(\text{Attacker MMLU-Pro}) - \text{logit}(\text{Target MMLU-Pro})$
    - Positive values indicate a stronger attacker (strong-to-weak); negative values indicate a stronger target (weak-to-strong)

2. **Sigmoid Scaling Law**:

    - For each target model, a linear regression of ASR on the capability gap is fitted in logit space and mapped back to probability space, yielding a sigmoid curve
    - Sigmoid parameters (slope and intercept) differ across targets but the functional form is consistent
    - Median parameters: $k=1.73, b=-0.79$

3. **Model Unlocking**:

    - All attacker models undergo LoRA fine-tuning to remove safety alignment (~1,500 harmful samples)
    - General capabilities are preserved while refusal behavior is eliminated, ensuring that the evaluated quantity is attack capability rather than willingness to refuse
    - Unlocking quality is validated via direct HarmBench query ASR

## Key Experimental Results

### Main Results (600+ Attacker–Target Pairs)

| Attack Method | # Model Pairs | avg ASR vs. MMLU-Pro $\rho$ | R² (sigmoid fit) | Key Finding |
|---------|---------|----------------------|------------------|---------|
| PAIR | 600+ | >0.88 | 0.83 | Most fundamental LLM attack |
| TAP | 600+ | >0.85 | ~0.80 | Highest overall ASR |
| PAP | 600+ | >0.82 | ~0.78 | Persuasion-oriented attack |
| Crescendo | 600+ | >0.80 | ~0.75 | Multi-turn attack |

### Core Scaling Trends

| Trend | Quantification | Implication |
|------|------|------|
| Stronger model = better attacker | avg ASR linearly correlated with MMLU-Pro ($\rho>0.88$) | Capability gains simultaneously improve attack power |
| Stronger model = harder to break | Target ASR decreases with MMLU-Pro ($R^2=0.83$) | But the decline is predictable |
| Capability gap dominates | ASR primarily determined by $\delta$ (gap), not attacker's absolute capability | Relative capability matters more than absolute capability |
| Social science > STEM | Psychology/health/philosophy subcategories show strongest correlation | Persuasiveness is more critical than domain knowledge |

## Highlights & Insights
- **Predictive Tool**: Given the MMLU-Pro scores of two models, the approximate ASR of red-teaming can be predicted, reducing costly full-scale evaluations—potentially saving tens of thousands of GPU hours
- **Quantifying Safety Investment**: The rightward shift of the sigmoid can measure the "equivalent capability gain" of safety training—Llama3 exhibits a larger shift than Qwen2.5, reflecting greater safety investment
- **Quantitative Assessment of Weaponization Risk**: As open-source model capabilities improve, the pool of attacker-accessible capability grows—each new 70B open-source release requires reassessment of the attack exposure of all existing deployed systems
- **Practical Implication Regarding Judges**: Expensive closed-source judges have no significant effect on final ASR—the community need not incur substantial API costs on judges

## Ablation Study

| Analysis Dimension | Finding |
|----------|------|
| MMLU-Pro subcategory correlation | Social science subcategories (psychology, health, philosophy) correlate most strongly with ASR, surpassing STEM |
| Judge impact | Strong judges improve ASR@1 (selection effect) but do not affect ASR@25 (generation quality) |
| Attack method comparison | TAP is overall strongest; Crescendo's originally reported success is attributable to the GPT-4 attacker rather than the method itself |
| Llama2 anomaly | 4 early Llama models deviate from the trend (excessive refusal + adversarial training); later versions return to trend |
| Human red-teaming prediction | Assuming human MMLU-Pro = 0.898, ASR continuously declines as target model capability increases |

## Limitations & Future Work
- MMLU-Pro as a capability proxy may be insufficiently precise in specific domains—the deviation of the Llama2 series indicates that safety training can break the general capability–robustness trend
- Only 4 automated attack methods (PAIR/TAP/PAP/Crescendo) are evaluated; manual or composite attacks may not follow the same scaling law
- Model unlocking via LoRA fine-tuning may not fully restore attack capability—unlocking quality is a potential confounding factor, particularly for models such as Claude that cannot be fully unlocked through simple fine-tuning
- White-box attacks (e.g., GCG) are not considered; the study is limited to black-box, human-like attacks, which may follow different scaling laws
- The deviation of the Llama2 series suggests MMLU-Pro is not a perfect proxy for defensive capability—a better proxy might be the FLOPs invested in safety training, but such data are not publicly available

## Related Work & Insights
- **vs. Ren et al. (2024) safety benchmark analysis**: That work first quantified the negative correlation between jailbreak success rate and model capability; this paper further characterizes the relationship precisely as a sigmoid function
- **vs. Howe et al. (2025) GCG scaling**: Their study examines scaling of GCG attacks within a model family; this paper examines capability-gap scaling across model families—complementary dimensions
- **vs. PAIR/TAP/PAP**: This paper does not propose new attacks but uses existing attacks to reveal scaling laws—emphasizing the *law* rather than the *method*
- **Insight—Social science capability as a blind spot**: Current safety evaluations focus on technical capabilities (coding, chemistry), yet persuasive and social engineering capabilities are the strongest predictors of attack success

## Rating
- Novelty: ⭐⭐⭐⭐ Scaling laws for red-teaming are a novel finding; sigmoid fitting is directly applicable to prediction
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale evaluation over 600+ model pairs, including closed-source frontier models
- Writing Quality: ⭐⭐⭐⭐ Results are clearly presented with high information density in figures
- Value: ⭐⭐⭐⭐⭐ Provides direct engineering guidance and policy implications for AI safety evaluation

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos](urbanverse_scaling_urban_simulation_by_watching_city-tour_videos.md)
- [\[ICLR 2026\] SocialHarmBench: Revealing LLM Vulnerabilities to Socially Harmful Requests](socialharmbench_revealing_llm_vulnerabilities_to_socially_harmful_requests.md)
- [\[ICLR 2026\] On Entropy Control in LLM-RL Algorithms](on_entropy_control_in_llm-rl_algorithms.md)
- [\[ICLR 2026\] D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI](d2e_scaling_vision-action_pretraining_on_desktop_data_for_transfer_to_embodied_a.md)
- [\[ICLR 2026\] ODESteer: A Unified ODE-Based Steering Framework for LLM Alignment](odesteer_a_unified_ode-based_steering_framework_for_llm_alignment.md)

<!-- RELATED:END -->
