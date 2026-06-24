---
title: >-
  [Paper Note] SafetyAnalyst: Interpretable, Transparent, and Steerable Safety Moderation for AI Behavior
description: >-
  [ICML 2025][Interpretability][AI safety moderation] Proposes the SafetyAnalyst framework, which generates an interpretable "harm-benefit tree" via chain-of-thought (CoT) reasoning (enumerating harmful and beneficial effects potentially caused by AI actions, along with their likelihood, severity, and immediacy). These features are then aggregated into a harm score using 28 fully interpretable parameters. On prompt safety classification, it outperforms existing moderation syste…
tags:
  - "ICML 2025"
  - "Interpretability"
  - "AI safety moderation"
  - "harm-benefit tree"
  - "knowledge distillation"
  - "steerability"
date: 2026-05-08
content_hash: 1b6145947def3494
---

# SafetyAnalyst: Interpretable, Transparent, and Steerable Safety Moderation for AI Behavior

**Conference**: ICML 2025  
**arXiv**: [2410.16665](https://arxiv.org/abs/2410.16665)  
**Code**: [https://jl3676.github.io/SafetyAnalyst/](https://jl3676.github.io/SafetyAnalyst/)  
**Area**: Interpretability  
**Keywords**: AI safety moderation, interpretability, harm-benefit tree, knowledge distillation, steerability

## TL;DR

Proposes the SafetyAnalyst framework, which generates an interpretable "harm-benefit tree" via chain-of-thought (CoT) reasoning (enumerating harmful and beneficial effects potentially caused by AI actions, along with their likelihood, severity, and immediacy). These features are then aggregated into a harm score using 28 fully interpretable parameters. On prompt safety classification, it outperforms existing moderation systems with an average F1 score of 0.81 (compared to F1 < 0.72 for prior systems) while delivering interpretability, transparency, and steerability.

## Background & Motivation

### 1. The Demand for AI Safety Moderation

As LLMs and AI agents become increasingly integrated into daily life, reliable moderation systems are required to detect potentially harmful behavior. An ideal moderation system should possess:
- **Interpretability**: Decisions can be reliably explained.
- **Steerability**: Safety standards can be adjusted based on application scenarios, user groups, and regulatory requirements.

### 2. Limitations of Prior Work

Current moderation systems (such as OpenAI Moderation API and LlamaGuard) directly learn the mapping from input to harmfulness using deep networks. The decision-making process is a "black box," failing to explain why content is judged harmful, and making it difficult to adjust standards as needed.

### 3. Core Idea

Inspired by the "blueprint for guaranteed-safe AI" by Dalrymple et al., SafetyAnalyst does not directly predict harmfulness. Instead, it first uses CoT reasoning to predict the causal consequences of AI behaviors (who is affected, and what action leads to what effect), then aggregates these into a score using a transparent mathematical formula.

## Method

### Overall Architecture

1. **Harm-Benefit Feature Generation**: Uses CoT prompting to guide LLMs in analyzing stakeholders, actions, and effects potentially impacted by AI behavior (labeled with likelihood, severity, and immediacy).
2. **Knowledge Distillation**: Employs 5 frontier LLMs (GPT-4o, Gemini-1.5-Pro, Llama 70B/405B, Claude-3.5-Sonnet) to generate 18.5 million features across 19K prompts, which are then distilled into Llama-3.1-8B.
3. **Transparent Aggregation**: Aggregates all effects into a harm score using 28 interpretable parameters.

### Key Designs

#### 1. Harm-Benefit Tree Structure

Each prompt is analyzed into a tree structure:
- **Stakeholders** (individuals, groups, communities, etc.)
- **Actions** (harmful/beneficial actions that each stakeholder may experience)
- **Effects** (consequences that each action might lead to)
    - Likelihood: Low/Medium/High
    - Severity: Minor/Significant/Substantial/Major
    - Immediacy: Immediate/Downstream

On average, each prompt generates 10+ stakeholders, 3–10 actions per stakeholder, and 3–7 effects per action.

#### 2. 28-Parameter Aggregation Model

$$\mathcal{H} = \sum_{\text{Stakeholder}} \sum_{\text{Action}} \sum_{\text{Effect}} \gamma \cdot W_{\text{Action}} \cdot W_{\text{Likelihood}} \cdot W_{\text{Extent}} \cdot W_{\text{Immediacy}}$$

The parameters include:
- 16 weights for harmful action categories (security risks, violent extremism, hate/toxicity, child harm, etc.)
- 2 relative weights for likelihood + 3 relative weights for severity
- 5 relative weights for beneficial effects
- 2 discount factors (downstream effects, beneficial vs. harmful)

These parameters are aligned by optimizing the negative log-sigmoid loss on 500 annotated samples from WildJailbreak.

#### 3. Knowledge Distillation and Adversarial Augmentation

- Two expert models for harm and benefit are trained separately (both based on Llama-3.1-8B).
- The training data is supervised fine-tuned using QLoRA with an 18,000-token context window.
- An additional 13,838 adversarial prompts (jailbreak variants) are included for data augmentation.

### Steerability

Weights can be adjusted in two ways:
- **Top-down**: Directly setting weights based on policies/regulations (e.g., increasing the weight of sexual content in child-oriented application scenarios).
- **Bottom-up**: Optimizing weights based on preference data from specific communities.

## Key Experimental Results

### Main Results

| Model | SimpSTests | HarmBench | WildGuard-Vanilla | WildGuard-Adv | AIR-Bench | SORRY-Bench | Weighted Avg F1 |
|------|-----------|-----------|-------------------|---------------|-----------|-------------|-----------------|
| OpenAI Mod. API | 63.0 | 47.9 | 16.3 | 6.8 | 46.5 | 42.9 | 41.1 |
| LlamaGuard | 93.0 | 85.6 | 70.5 | — | — | — | <72 |
| **SafetyAnalyst** | **95.2** | **89.3** | **83.1** | **76.4** | **78.2** | **79.5** | **81.0** |

SafetyAnalyst leads with an average F1 score of 0.81, showing a significant advantage particularly in adversarial scenarios (WildGuard-Adv).

### Ablation Study: Different Teacher LLMs

| Teacher Model | F1 | AUPRC | AUROC |
|---------|-----|-------|-------|
| GPT-4o | 91.8 | 91.7 | 94.7 |
| Gemini-1.5-Pro | 87.7 | 92.0 | 92.5 |
| Llama-3.1-70B | 88.1 | 96.6 | 95.9 |
| **Student (8B)** | **88.8** | **92.9** | **93.4** |

The 8B student model's performance is comparable to that of 70B+ teacher models, proving the effectiveness of the distillation.

### Key Findings

- Aggregated weights reveal: Defamation has the highest weight, followed by Child Harm and Self-Harm.
- The weight for low-likelihood effects is close to zero—aggregation is primarily driven by high-likelihood, immediate effects.
- The relative importance of beneficial effects is only 7.59%, indicating that safety judgments are dominantly driven by harm.
- Enhancing the training data with adversarial prompts significantly improves robustness against jailbreaks.

## Highlights & Insights

- **Practical Value of Interpretability**: The 28 parameters can directly explain to users why a prompt is flagged (e.g., "because it may lead to child harm [weight 0.85] with high likelihood [weight 1.0]").
- **Cost-Benefit Analysis Framework**: Leverages cost-benefit analysis principles from economics to weigh harms versus benefits, providing a solid theoretical foundation.
- **New Taxonomy of Effects**: Constructs the first taxonomy of harmful/beneficial effects for AI safety, based on the moral philosophy of Bernard Gert and John Rawls.
- **Open-source Ecosystem**: The model, data, and taxonomy are fully open-sourced, lowering the barrier to entry for community research.

## Limitations & Future Work

- The generation of the harm-benefit tree depends on the LLM's imagination, potentially omitting rare but severe consequences.
- The 28-parameter aggregation is a linear multiplicative model, which insufficiently models non-linear interactions between effects.
- It has only been validated on prompt classification tasks; expansion to response-level and agent behavior-level moderation remains to be explored.
- Likelihood/severity labels are discrete; continuous representations might improve expressiveness.
- Generation latency of the 8B model could be a bottleneck during large-scale deployment.

## Related Work & Insights

- **vs. OpenAI Moderation API**: An end-to-end black-box classifier with an F1 score of only 0.41, lacking interpretability.
- **vs. LlamaGuard**: An LLM-based classifier that performs better but still relies on black-box decision-making.
- **vs. Safety Blueprint by Dalrymple et al.**: SafetyAnalyst is the first practical system implementing the "World Model" concept.
- **vs. General CoT Safety Reasoning**: Ours structures CoT into an aggregated feature tree rather than free-form text.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The framework design featuring a harm-benefit tree and interpretable aggregation is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive coverage across 6 benchmarks with thorough multi-teacher ablation.
- Writing Quality: ⭐⭐⭐⭐⭐ Interdisciplinary integration is natural, and the framework presentation is clear.
- Value: ⭐⭐⭐⭐⭐ The first AI safety moderation system that simultaneously achieves interpretability, steerability, and high performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Safety is Not Only About Refusal: Reasoning-Enhanced Fine-tuning for Interpretable LLM Safety](../../ACL2025/interpretability/safety_is_not_only_about_refusal_reasoning-enhanced_fine-tuning_for_interpretabl.md)
- [\[ICML 2025\] Position: We Need An Algorithmic Understanding of Generative AI](position_we_need_an_algorithmic_understanding_of_generative_ai.md)
- [\[AAAI 2026\] Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution](../../AAAI2026/interpretability/can_llms_truly_embody_human_personality_analyzing_ai_and_human_behavior_alignmen.md)
- [\[ICML 2025\] What Makes an Ensemble (Un)interpretable?](what_makes_an_ensemble_un_interpretable.md)
- [\[ICML 2025\] LANTERN: Modeling User Behavior from Adaptive Surveys with Supplemental Context](modeling_user_behavior_from_adaptive_surveys_with_supplemental_context.md)

</div>

<!-- RELATED:END -->
