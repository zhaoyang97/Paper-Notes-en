---
title: >-
  [Paper Note] ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway
description: >-
  [ACL 2026][Social Computing][Toxicity Reasoning] ToxReason proposes an AOP-based chemical toxicity mechanistic reasoning benchmark that integrates drug-target experimental data with toxicity labels, requiring models to reason from molecular initiating events to organ-level adverse outcomes; a 4B model trained with GRPO reinforcement learning surpasses GPT-5 and other large models in both toxicity prediction (F1 71.4%) and reasoning quality.
tags:
  - ACL 2026
  - Social Computing
  - Toxicity Reasoning
  - Adverse Outcome Pathway
  - Benchmark
  - Reinforcement Learning
  - LLM Evaluation
date: 2025-05-08
content_hash: 9c2156795130ed06
---

# ToxReason: A Benchmark for Mechanistic Chemical Toxicity Reasoning via Adverse Outcome Pathway

**Conference**: ACL 2026  
**arXiv**: [2604.06264](https://arxiv.org/abs/2604.06264)  
**Code**: N/A  
**Area**: Social Computing  
**Keywords**: Toxicity Reasoning, Adverse Outcome Pathway, Benchmark, Reinforcement Learning, LLM Evaluation

## TL;DR

ToxReason proposes an AOP-based chemical toxicity mechanistic reasoning benchmark that integrates drug-target experimental data with toxicity labels, requiring models to reason from molecular initiating events to organ-level adverse outcomes; a 4B model trained with GRPO reinforcement learning surpasses GPT-5 and other large models in both toxicity prediction (F1 71.4%) and reasoning quality.

## Background & Motivation

**State of the Field**: LLMs have been applied to molecular reasoning and toxicity prediction tasks, with existing benchmarks (e.g., Tox21, ClinTox) primarily focusing on structure–property relationship prediction and treating toxicity as a simple classification task.

**Limitations of Prior Work**: Toxicity fundamentally arises from complex biological mechanisms (molecular targets → cellular events → organ responses) rather than solely from chemical structure. LLMs can generate fluent but biologically unreliable explanations, meaning high prediction accuracy does not equate to reliable reasoning. Existing datasets such as UniTox base their reasoning on clinical observations rather than causal mechanistic pathways.

**Root Cause**: A significant disconnect exists between prediction performance and reasoning quality—a model may "guess the right answer" while providing incorrect mechanistic explanations, which is unacceptable in high-stakes scenarios such as drug safety assessment.

**Paper Goals**: Build a benchmark that evaluates mechanistic toxicity reasoning, requiring models to perform step-by-step causal reasoning from molecular initiating events (MIE) to adverse outcomes (AO), and explore training strategies to improve reasoning capability.

**Starting Point**: The Adverse Outcome Pathway (AOP) framework in toxicology naturally describes causal chains from MIE → Key Events (KE) → AO, which closely parallels multi-step reasoning paradigms in NLP.

**Core Idea**: Use AOP causal chains as ground truth for toxicity reasoning, construct an evaluation benchmark, and improve both prediction and reasoning through reasoning-aware training.

## Method

### Overall Architecture

ToxReason construction follows three steps: (1) selecting 23 AOPs and 25 MIE targets related to hepato-, cardio-, and nephrotoxicity from AOP-Wiki; (2) integrating CTD disease-chemical associations with ChEMBL experimental activity data, inferring MIEs via structural similarity; (3) constructing training and test sets for model learning and rigorous evaluation. Evaluation covers both toxicity prediction (F1) and reasoning quality (LLM-as-a-Judge with four-dimensional scoring).

### Key Designs

1. **AOP Selection and Chemical-AOP Association Derivation**:

    - Function: Constructs reasoning annotation data grounded in biological causal mechanisms
    - Mechanism: Selects hepato/cardio/nephrotoxicity-related AOPs from AOP-Wiki and treats their AOs as disease concepts to retrieve associated chemicals from CTD. Extracts MIE target EC50/IC50 activity data from ChEMBL (<10000nM considered active). For candidate chemicals, infers MIE direction (activation/inhibition) via structural similarity majority voting
    - Design Motivation: Solves the MIE inference problem when direct experimental data is unavailable by aggregating evidence from similar molecules with known activity

2. **Training and Test Set Construction**:

    - Function: Separate data design supporting learning and evaluation
    - Mechanism: The training set is split into MIE-matched (satisfying MIE conditions only, Dice similarity ≥ 0.5) and MIE-AO-matched (satisfying both MIE and AO) as complementary subsets. The test set uses strictly curated associations and structurally identical chemicals, ensuring no leakage
    - Design Motivation: MIE-matched expands coverage to help learn initiating patterns, while MIE-AO-matched encourages cross-molecule interaction and downstream toxicity outcome reasoning

3. **GRPO Reinforcement Learning Training Framework**:

    - Function: Explicitly optimizes the joint objective of toxicity prediction and mechanistic reasoning
    - Mechanism: Employs two-stage training—SFT first aligns the task format, then GRPO optimizes causal consistency and biological faithfulness of AOP reasoning
    - Design Motivation: SFT only learns output formatting without improving reasoning quality; RL uses explicit reward signals to guide models toward generating reasoning chains aligned with AOP pathways

### Loss & Training

Uses the GRPO (Group Relative Policy Optimization) framework with reward signals combining toxicity prediction accuracy and reasoning chain alignment with AOPs. Parameter-efficient fine-tuning via LoRA.

## Key Experimental Results

### Main Results

| Model | Nephrotoxicity F1 | Cardiotoxicity F1 | Hepatotoxicity F1 | Average F1 | Reasoning Overall |
|-------|-------------------|--------------------|--------------------|------------|-------------------|
| GPT-5 | 56.4 | 72.7 | 65.0 | 64.7 | 5.420 |
| GPT-5.1 | 50.3 | 71.2 | 58.9 | 60.1 | 5.523 |
| o3 | 60.0 | 72.5 | 58.8 | 63.8 | 5.326 |
| DeepSeek-R1-70B | 59.1 | 78.5 | 59.6 | 65.7 | 4.487 |
| Qwen3-4B (base) | 56.9 | 71.1 | 57.3 | 61.8 | 4.523 |
| ToxReason-4B-SFT | 57.9 | 74.3 | 57.4 | 63.2 | 4.554 |
| **ToxReason-4B-GRPO** | **73.4** | 72.7 | **68.2** | **71.4** | **5.642** |

### Ablation Study

| Config | Average F1 | Reasoning Overall | Note |
|--------|------------|-------------------|------|
| Qwen3-4B base | 61.8 | 4.523 | Base model |
| + ICL 1-shot | 68.8 | 5.373 | Best few-shot setting |
| + ICL 2-shot | 59.1 | 4.373 | More examples introduce noise |
| + SFT | 63.2 | 4.554 | Fine-tuning yields limited gains |
| + GRPO | 71.4 | 5.642 | RL significantly improves both |

### Key Findings

- A significant disconnect exists between prediction performance and reasoning quality: GPT-5.1 achieves the best reasoning but the worst prediction (60.1%), while DeepSeek-R1 has the best prediction but weaker reasoning
- SFT barely helps reasoning quality, whereas GRPO substantially improves both prediction (+9.6%) and reasoning (+1.1 points)
- ICL works best at 1-shot; increasing the number of shots introduces noise and degrades performance
- NW alignment scores correlate highly with LLM-as-a-Judge scores (Spearman ρ=0.837), validating the evaluation methodology

## Highlights & Insights

- **Prediction–reasoning disconnect finding**: Reveals that LLMs can perform well on toxicity prediction while having completely wrong reasoning mechanisms—an important warning for safety-critical applications
- **4B model surpasses GPT-5**: Through GRPO reasoning-aware training, a 4B-parameter model outperforms closed-source large models in both prediction and reasoning, demonstrating the value of explicit reasoning optimization
- **Elegant mapping between AOP and NLP multi-step reasoning**: Translating toxicological causal chains into NLP-evaluable reasoning tasks is a generalizable approach for mechanistic reasoning evaluation across other scientific domains

## Limitations & Future Work

- Covers only hepato-, cardio-, and nephrotoxicity, limited by AOP-Wiki coverage
- MIE inference is based on structurally similar molecules rather than direct prediction from molecular structure, limiting applicability to entirely novel chemicals
- LLM-as-a-Judge evaluation is inherently subjective; although validated by the NW algorithm, it should be treated as a relative metric
- Future work could extend to more organ systems and more complex AOP networks

## Related Work & Insights

- **vs CoTox**: CoTox improves prediction via chain-of-thought but does not evaluate whether reasoning aligns with causal pathways; ToxReason makes reasoning evaluation a core objective
- **vs Tox21/ClinTox**: Traditional toxicity benchmarks only predict outcomes; ToxReason requires models to explain "why it is toxic"
- **vs UniTox**: UniTox provides explanations based on clinical observations; ToxReason requires step-by-step reasoning grounded in AOP causal mechanisms

## Rating

- Novelty: ⭐⭐⭐⭐ First benchmark systematically evaluating LLM mechanistic toxicity reasoning; the prediction–reasoning disconnect finding is valuable
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple open/closed-source models and learning strategies, though limited to three organ toxicities
- Writing Quality: ⭐⭐⭐⭐ Clear structure with thorough AOP background introduction
- Value: ⭐⭐⭐⭐ Practically significant for drug safety and trustworthy AI reasoning

<!-- RELATED:START -->

## Related Papers

- [\[ACL 2026\] ToxiTrace: Gradient-Aligned Training for Explainable Chinese Toxicity Detection](toxitrace_gradient-aligned_training_for_explainable_chinese_toxicity_detection.md)
- [\[ACL 2026\] On the Step Length Confounding in LLM Reasoning Data Selection](on_the_step_length_confounding_in_llm_reasoning_data_selection.md)
- [\[ICLR 2026\] BiasFreeBench: a Benchmark for Mitigating Bias in Large Language Model Responses](../../ICLR2026/social_computing/biasfreebench_a_benchmark_for_mitigating_bias_in_large_language_model_responses.md)
- [\[NeurIPS 2025\] OS-Harm: A Benchmark for Measuring Safety of Computer Use Agents](../../NeurIPS2025/social_computing/os-harm_a_benchmark_for_measuring_safety_of_computer_use_agents.md)
- [\[AAAI 2026\] From Imitation to Discrimination: Toward A Generalized Curriculum Advantage Mechanism Enhancing Cross-Domain Reasoning Tasks](../../AAAI2026/social_computing/from_imitation_to_discrimination_toward_a_generalized_curriculum_advantage_mecha.md)

<!-- RELATED:END -->
