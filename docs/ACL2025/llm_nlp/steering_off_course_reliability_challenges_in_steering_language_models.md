---
title: >-
  [Paper Note] Steering off Course: Reliability Challenges in Steering Language Models
description: >-
  [ACL 2025][LLM (Other)][Language Model Steering] This paper systematically evaluates the generalization of three mainstream language model steering methods (DoLa, Function Vectors, Task Vectors) across up to 36 models, revealing severe fragility and high variance issues, as well as fundamental flaws in their underlying assumptions.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Language Model Steering"
  - "Activation Patching"
  - "DoLa"
  - "Function Vectors"
  - "Interpretability"
date: 2026-05-08
content_hash: 78fa30c38c0c0f54
---

# Steering off Course: Reliability Challenges in Steering Language Models

**Conference**: ACL 2025  
**arXiv**: [2504.04635](https://arxiv.org/abs/2504.04635)  
**Code**: [github](https://github.com/patrickdasilva/steering-off-course)  
**Area**: LLM/NLP  
**Keywords**: Language Model Steering, Activation Patching, DoLa, Function Vectors, Interpretability

## TL;DR

This paper systematically evaluates the generalization of three mainstream language model steering methods (DoLa, Function Vectors, Task Vectors) across up to 36 models, revealing severe fragility and high variance issues, as well as fundamental flaws in their underlying assumptions.

## Background & Motivation

Language model steering methods have attracted increasing attention as lightweight alternatives to fine-tuning. They adjust specific behaviors by modifying internal model activations at inference time, requiring almost no additional data or changes to model parameters. Prior methods have been applied to improve factuality, reduce harmful outputs, etc.

However, the LM steering field has accumulated a **generalization blind spot**: the vast majority of studies report results on only a small number of models. For example, DoLa was only tested on the Llama 1 family, and Function Vectors were only validated using GPT-J. As growing evidence suggests that steering methods can be fragile and degrade general capabilities, a systematic evaluation across a wider range of models is urgently needed.

**Goal**: To quantify the generalization capability of three steering methods across different model families and scales (14 families, 1.5B-70B, up to 36 models).

## Method

### Overall Architecture

The authors systematically evaluate three steering methods derived from two popular interpretability tools, Logit Lens and Activation Patching:

1. **DoLa** (based on Logit Lens)
2. **Function Vectors (FV)** (based on Activation Patching)
3. **Task Vectors (TV)** (based on Activation Patching)

### Key Designs

1. **DoLa Analysis**:

    - Core assumption of DoLa: Factual knowledge is localized in the later layers of the model, and contrasting the probability changes between the final layer and the "premature layer" can improve factuality.
    - Premature layer selection: Selecting the candidate layer with the largest JSD distance from the final layer.
    - Output probability update: $p̂(x_t) = softmax(F(q_L, q_P))$, where $F$ computes the log-probability ratio.
    - The authors find that the original paper calculates metrics using raw logits of $F$ rather than the post-softmax probabilities, which introduces length bias.
    - The experiments cover 10 models, 7 families, two scales, searching over 4 bucket configurations and 6 α values.

2. **Function Vectors (FV) Analysis**:

    - Based on the "localization hypothesis": A small number of attention heads can mediate many ICL tasks.
    - Construction process: Computing the mean activation of attention heads across all prompts in the task dataset, then choosing the sum of the top-n heads' activations via causal mediation analysis.
    - The authors expand the hyperparameter search to n ∈ {2, 16, 32, 64, 128, 256, 512, 1024} and introduce a strength multiplier λ ∈ {0.5, 1, 2, 4, 8, 16, 32, 64}.
    - Tested on 36 models and 11 ICL tasks.

3. **Task Vectors (TV) Analysis**:

    - Does not rely on localization or causal analysis; it directly encodes the ICL prompt into the model's activation space.
    - At inference time, replaces the hidden states of specific layers with the steering vector (α=0).
    - Has only a single hyperparameter: the choice of the injection layer ℓ.

### Loss & Training

This paper does not propose new methods but conducts a systematic evaluation of existing ones. Evaluation settings:
- DoLa: TruthfulQA and FACTOR datasets, 6-shot, utilizing MC1/MC2/MC3 metrics.
- FV/TV: 11 ICL tasks, 50 test samples per task, recording the zero-shot accuracy after steering, normalized against 5-shot performance.

## Key Experimental Results

### Main Results

**Performance of DoLa on TruthfulQA** (results corrected using softmax):

| Model | Base MC1 | DoLa MC1 | Base MC2 | DoLa MC2 |
|------|------|----------|------|------|
| Llama 7B (Original) | 0.26 | 0.32 | 0.41 | 0.64 |
| Llama 7B (Corrected) | 0.26 | 0.32 | 0.41 | 0.52 |
| Llama 2 7B | 0.29 | 0.29 | 0.43 | 0.44 |
| Llama 3 8B | 0.32 | 0.32 | 0.49 | 0.49 |
| Qwen 2 7B | 0.39 | 0.37 | 0.58 | 0.51 |
| Qwen 2 72B | 0.44 | 0.39 | 0.63 | 0.46 |

After correcting the metric calculation, the gains of DoLa on Llama 1 shrink significantly. Except for Llama 1 and Pythia, other models show almost no improvement or even experience performance degradation.

**Performance recovery rate of Function Vectors and Task Vectors** (percentage relative to 5-shot performance):

| Method | Achieved 50% Performance | Achieved 75% Performance | Achieved 90% Performance | Achieved 100% Performance |
|------|------|----------|------|------|
| FV Default Parameters | 47% | 37% | 20% | 12% |
| FV Parameter Search | 76% | 68% | 52% | 28% |
| Task Vectors | 69% | 54% | 35% | 16% |

Even under the optimal hyperparameters, FV recovers 90% of the 5-shot performance in only 52% of the model-task combinations; TV performs worse, achieving this in only 35%.

### Ablation Study

| Analysis Dimension | Key Findings | Explanation |
|------|---------|------|
| λ (FV strength multiplier) | Large variance in model preference | Some models require λ=1, others require λ=16-32 |
| n (number of attention heads) | Some tasks require a large number of heads | eng→[lang] translation requires n≥64 to begin being effective |
| Injection layer ℓ | Optimal layer varies highly across models and tasks | TV is highly sensitive to ℓ, with a large gap between peak and mean |
| Base vs post-trained | Post-trained models perform better on FV; they perform worse on TV | Contradictory post-training effects across steering methods |
| Logit Lens dynamics | Correct and incorrect tokens peak at the same layer | Contrasting with early layers provides almost no informative signal |

### Key Findings

1. **Flawed DoLa Assumption**: Logit Lens analysis reveals that the probabilities of correct and incorrect tokens start rising sharply at the same layer, meaning that contrasting early layers with the final layer cannot effectively distinguish between correct and incorrect answers.
2. **Incorrect Metric Calculation in Original DoLa**: Using raw logits instead of softmax probabilities to calculate metrics introduces a length bias, leading to overestimates of the reported gains.
3. **Poor Generalization of FV and TV**: Across 36 models, a vast number of model-task combinations fail to recover 5-shot performance, with steering efficacy heavily dependent on model family, scale, and task.
4. **Localization Hypothesis Does Not Always Hold**: Certain tasks require a massive number of attention heads (e.g., 512) for FV to be effective, contradicting the assumption that "only a few heads are sufficient."
5. **Inconsistent Impact of Post-training on Steering Methods**: FV benefits from instruction tuning, while TV performance degrades, suggesting that these methods rely on distinct and unstable internal mechanisms.

## Highlights & Insights

1. **Large-Scale Critical Evaluation**: Unlike most works that report success on only 2-3 models, this study systematically tests 36 models across 14 families, exposing the issue of "publication bias" in the field.
2. **Importance of Metric Correction**: Uncovering the metric calculation error in the original DoLa paper serves as a reminder to the community regarding the critical importance of experimental details.
3. **Hypothesis Verification Over Methodized Innovation**: Prioritizing the verification of underlying assumptions before proposing new methods is a research paradigm worth promoting.
4. **A Warning to Interpretability Research**: Many interpretability studies are informally released via blog posts without rigorous evaluation; this paper calls for more stringent evaluation standards.

## Limitations & Future Work

1. Although the hyperparameter search is extensive, it is impossible to exhaust all combinations (e.g., the number of bucket combinations in DoLa is exponential).
2. Only decoder-only architectures were tested; encoder-decoder models were not covered.
3. The root cause of the fragility is not deterministically verified (hypotheses like pre-training data or architectural differences remain unconfirmed).
4. In the FV/TV experiments, each task contains only 50 test samples, potentially introducing high statistical variability.
5. No alternative solutions are proposed, leaving the work at the "problem disclosure" stage.

## Related Work & Insights

This work is highly related to vulnerability studies in model editing (e.g., ROME/MEMIT); both expose the reliability challenges of "precision intervention" methods. Additionally, SAE steering (Durmus et al., 2024) and contrastive activation addition (Rimsky et al., 2024) have also been found to exhibit similar vulnerabilities. This study serves as an important warning for any inference-time intervention methods based on internal model mechanisms. Future work should make large-scale, multi-model evaluation experiments a standard requirement when proposing steering methods.

## Rating

- Novelty: ⭐⭐⭐⭐ Although no new method is proposed, the large-scale systematic evaluation itself carries unique value.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely comprehensive coverage with 36 models, 14 families, an extensive hyperparameter search, and multi-task evaluations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and deep analysis, though the focus on "negative results" might reduce interest for some readers.
- Value: ⭐⭐⭐⭐⭐ Offers a crucial warning to the steering field and plays a driving role in standardizing experimental practices in the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Multi-Attribute Steering of Language Models via Targeted Intervention](multi_attribute_steering.md)
- [\[ACL 2025\] CogSteer: Cognition-Inspired Selective Layer Intervention for Efficiently Steering Large Language Models](cogsteer_cognition-inspired_selective_layer_intervention_for_efficiently_steerin.md)
- [\[ICLR 2026\] Fine-Grained Activation Steering: Steering Less, Achieving More](../../ICLR2026/llm_nlp/fine-grained_activation_steering_steering_less_achieving_more.md)
- [\[ACL 2025\] Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms](beyond_prompt_engineering_robust_behavior_control_in_llms_via_steering_target_at.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)

</div>

<!-- RELATED:END -->
