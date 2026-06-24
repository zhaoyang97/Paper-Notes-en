---
title: >-
  [Paper Note] Taming Knowledge Conflicts in Language Models
description: >-
  [ICML 2025 Spotlight][Interpretability][Knowledge Conflict] Unveils the phenomenon of "Context-Parametric Superposition" (CP Superposition) within the attention heads of language models. Proposes JuICE (Just Run Twice), a dual-run attention intervention strategy that flexibly steers models toward either parametric memory or contextual knowledge without fine-tuning, achieving SOTA performance across 11 datasets and 6 model architectures.
tags:
  - "ICML 2025 Spotlight"
  - "Interpretability"
  - "Knowledge Conflict"
  - "Attention Intervention"
  - "Parametric Memory"
  - "Contextual Reliance"
  - "Test-time Intervention"
date: 2026-05-08
content_hash: e4e7858466be6ed2
---

# Taming Knowledge Conflicts in Language Models

**Conference**: ICML 2025 Spotlight  
**arXiv**: [2503.10996](https://arxiv.org/abs/2503.10996)  
**Code**: [GaotangLi/JUICE](https://github.com/GaotangLi/JUICE)  
**Area**: LLM/NLP  
**Keywords**: Knowledge Conflict, Attention Intervention, Parametric Memory, Contextual Reliance, Test-time Intervention

## TL;DR

Unveils the phenomenon of "Context-Parametric Superposition" (CP Superposition) within the attention heads of language models. Proposes JuICE (Just Run Twice), a dual-run attention intervention strategy that flexibly steers models toward either parametric memory or contextual knowledge without fine-tuning, achieving SOTA performance across 11 datasets and 6 model architectures.

## Background & Motivation

During the pre-training phase, language models encode a vast amount of knowledge into **parametric memory**. During inference, they simultaneously utilize parametric memory and **contextual knowledge** to generate outputs. When contradictions arise between the two, a **knowledge conflict** occurs, which is highly prevalent in scenarios such as Retrieval-Augmented Generation (RAG), LLM Agents, and tool-augmented LLMs.

Existing efforts exhibit three core limitations:

**Single Conflict Type**: Previous methods (such as PH3) primarily target sentence-level substitution conflicts. Their performance drops significantly under the more challenging paragraph-level coherent conflict.

**Flawed Exclusivity Assumption**: Prior studies assume the existence of mutually exclusive "memory heads" and "context heads." However, this study discovers that the same attention head can **simultaneously** contribute to both parametric knowledge and contextual knowledge.

**One-way Perspective**: Most works focus solely on enhancing contextual reliance (to mitigate RAG hallucinations), lacking a unified, bidirectional control method.

## Method

### Overall Architecture

JuICE consists of two phases:

1. **Head Identification**: Identifies two sets of attention heads using an extremely small sample set (only 4 samples)—specifically, those that can **consistently** produce the target effect under positive or negative scaling.
2. **Dual-run Inference**: Runs the model twice—the first run saves the output activations of the identified heads, and the second run injects scaled versions of these activations into the corresponding modules.

### Key Designs

#### 1. Discovery of CP Superposition

The core discovery of this study is **"Context-Parametric Superposition" (CP Superposition)**: highly influential attention heads **simultaneously** encode both parametric memory and contextual information, and their role is determined by the received residual stream inputs.

**Observation 1: Model components behave inconsistently under different levels of conflict**

- Without conflict: Removing almost all components leads to a **decrease** in the probability of parametric answers.
- Under substitution conflict: Removing different components leads to either an increase or a decrease in the probability of parametric answers.
- Under coherent conflict: Removing almost all components leads to an **increase** in the probability of parametric answers.

Out of the 26 layers in Gemma-2b, only 6 attention modules consistently enhance parametric knowledge across all three conflict types, and 0 complete layers do so.

More importantly, among the top-4 "memory heads" under substitution conflict, **half** of them turn into "context heads" under coherent conflict (their effects completely reverse). For instance, head (9,3) shows a $+0.13$ context probability under substitution conflict but changes to $-0.17$ under coherent conflict.

**Observation 2: Cancellation effects of multiple interventions**

Intuitively, stacking multiple individually effective interventions should yield a stronger effect, but experiments show that:

- Top-1 intervention: Target probability increases from 0.03 to 0.12 ✓
- Top-3 intervention: 0.03 to 0.24 ✓
- Top-10 intervention: 0.03 to 0.14 ✗ (performance drops instead)

This is because modifying activations in antecedent layers alters the residual stream received by downstream components, shifting their functions and inducing a cancellation effect.

#### 2. Head Identification Strategy

The head identification phase of JuICE selects attention heads that are **consistently effective across multiple conflict types**:

- For each attention head, the expected change in target probability after individual scaling is calculated on a small selection set ($|D|=4$).
- **Consistency constraint**: Retains only heads whose scores are non-negative across all conflict types.
- Top-$K$ (default $K=5$) are selected based on the aggregated score.
- Supports cross-domain generalization: Heads identified solely on the World Capital domain can generalize to other unrelated domains.

#### 3. Dual-run Inference Mechanism

This is the core design that distinguishes JuICE from prior methods:

- **First run**: Normal inference is executed, saving the output activations $h_i^{(1)}$ of the identified heads.
- **Second run**: The scaled first-run activations $\alpha \cdot h_i^{(1)}$ are added to the corresponding modules' activations.

The intuition is: **the activations from the first run serve as a more reliable guidance direction** because they originate from the un-intervened, original model state. Single-run intervention is unstable and prone to degradation, whereas dual-run can effectively mitigate the cancellation problem caused by superposition.

#### 4. Ablation Variant: JuNe (Just Run Once)

JuNe is the single-run intervention variant of JuICE without the dual-run design, used to verify the necessity of the dual-run mechanism. Experiments demonstrate that JuICE outperforms JuNe on Gemma by approximately 20% on average.

### Loss & Training

JuICE is a **test-time intervention method** that involves no fine-tuning or extra training. Core hyperparameters:

- **$|D|$**: The head selection set size, defaulting to only 4 samples.
- **$K$**: The number of intervened heads, defaulting to 5.
- **$\alpha$**: The scaling factor, determined via a validation set.

In the theoretical analysis part, the authors formalize knowledge conflicts as dual-task learning on a two-layer Transformer:
- **Fact recall task** (corresponding to parametric knowledge): Learns the subject-answer mapping $\mathcal{G}^*: \mathcal{S} \to \mathcal{A}$.
- **Induction task** (corresponding to contextual knowledge): Predicts the token that appears after a trigger word.

Weight matrices trained via gradient descent naturally form a superposition structure (Proposition 5.3). The standard cross-entropy loss intrinsically **encourages** the emergence of superposition, while model output preference depends on the relative values of coefficients $C_1, \dots, C_4$.

## Key Experimental Results

### Main Results - Enhancing Parametric Memory

| Dataset | Conflict Type | Original | Prompt | PH3_l | JuICE | Gain (vs PH3_l) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Athlete Sport (Gemma) | Coherent | 0.0 | 0.0 | 33.3 | **91.9** | +58.6 |
| World Capital (Gemma) | Coherent | 1.1 | 35.7 | 88.1 | **93.0** | +4.9 |
| Company HQ (Gemma) | Coherent | 0.0 | 0.0 | 30.6 | **59.3** | +28.7 |
| Average (Gemma) | All 3 types | 0.2 / 11.3 / 78.1 | 10.5 / 29.6 / 78.1 | 42.4 / 41.1 / 61.2 | **73.4 / 75.3 / 79.1** | — |
| Average (Llama2) | All 3 types | 0.2 / 25.9 / 82.5 | 19.8 / 55.0 / 82.5 | 54.5 / 82.0 / 80.6 | **83.0 / 82.5 / 82.2** | — |
| Average (Llama3) | All 3 types | 0.4 / 11.0 / 78.7 | 3.7 / 70.1 / 78.7 | 39.2 / 78.1 / 80.7 | **84.7 / 84.2 / 84.5** | — |

### Main Results - Enhancing Contextual Reliance

| Model | Method | NQ-Swap | Hate Speech | History QA | Proverb End | Proverb Trans | Average |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Gemma | Original | 38.7 | 70.7 | 29.9 | 26.5 | 59.0 | 45.0 |
| Gemma | CAD | 56.9 | 81.7 | 16.9 | 37.1 | 62.9 | 51.1 |
| Gemma | JuICE | **58.4** | **84.1** | **47.0** | **74.6** | **66.8** | **66.2** |
| Llama2 | PH3_l | 48.2 | 63.4 | 20.4 | 68.7 | 58.8 | 51.9 |
| Llama2 | JuICE | **49.5** | **93.9** | **50.2** | **77.1** | **62.6** | **66.6** |
| Llama3 | PH3_l | 25.3 | 62.2 | 78.4 | 48.5 | 63.6 | 55.6 |
| Llama3 | JuICE | **35.3** | **78.4** | **74.2** | **75.4** | **70.7** | **66.8** |

### Ablation Study

| Configuration | Gemma Average Accuracy (Coherent) | Description |
|:---|:---:|:---|
| Original | 0.2% | The model almost completely follows the context |
| JuNe (Single-run) | 52.7% | Effective but unstable |
| JuICE (Dual-run) | **73.4%** | Dual-run brings +20.7% gain |
| PH3_l (200 samples) | 42.4% | Requires 50× more samples |
| PH3_s (4 samples) | 0.1% | Almost ineffective under equivalent samples |

### Key Findings

1. **JuICE almost completely reverses** the context-following tendency of models under the most difficult coherent conflicts: Gemma goes from 0.2% to 73.4%, and Llama2 goes from 0.2% to 83.0%.
2. **Extremely low data requirement**: Requires only 4 samples to complete head identification, which is 50 times fewer than PH3 (200 samples).
3. **Cross-domain generalization**: Heads identified on World Capital can be effectively transferred to unrelated domains such as Athlete Sport and Company Founder.
4. **Bidirectional regulation**: The same framework can enhance both parametric memory and contextual reliance.
5. **High robustness**: Remains stable across head selection set sizes, the number of intervened heads, scaling factors, and input paraphrasings.

## Highlights & Insights

- **CP Superposition is a fundamental discovery**: It shatters the previous assumption of mutually exclusive "memory/context heads," proving that the function of attention heads is **input-dependent**. The same head can swap roles under different conflict conditions.
- **The "run twice" design is extremely simple and effective**: It requires no structural changes, no fine-tuning, and no contrastive decoding; precise adjustment is achieved simply by saving and replaying attention activations.
- **High unity of theory and experiment**: A two-layer Transformer dual-task theoretical framework perfectly explains three core questions: superposition emergence, knowledge conflict, and the effectiveness of JuICE.
- **High practical value**: Enables RAG systems to flexibly choose whether to "trust the context" or "trust oneself." Highly deployable with just 4 samples.

## Limitations & Future Work

1. **Doubled computational overhead**: Dual-run inference means the inference cost is approximately doubled, which may limit its application in latency-sensitive scenarios.
2. **Evaluated only on base models**: The effectiveness on instruction-tuned or chat models has not been verified.
3. **Scaling factor reliance on a validation set**: Although head identification requires only 4 samples, the optimal scaling factor still needs tuning using a validation set.
4. **Theoretical analysis based on simplified models**: The assumptions of two-layer linear attention and orthogonal embeddings still present a gap compared to actual deep non-linear models.
5. **No consideration of multi-hop reasoning conflicts**: All datasets evaluate single-step fact recall; conflict propagation in complex reasoning chains is not investigated.

## Related Work & Insights

- **PH3 (Jin et al., 2024)**: The first work to systematically study the relationship between attention heads and knowledge conflicts. It proposed the concepts of "memory heads" and "context heads," but its exclusivity assumption fails under coherent conflicts.
- **CAD (Shi et al., 2024)**: Enhances contextual reliance based on contrastive decoding but lacks the capability to enhance parametric memory.
- **Toy Models of Superposition (Elhage et al., 2022)**: Discovered the superposition phenomenon at the feature dimension; this paper extends the concept of superposition to the knowledge dimension (parametric vs. context).
- **Insights**: Future research could explore combining JuICE with LoRA to identify more stable intervention directions using minimal training; the dual-run strategy could also be extended to intervening in MLP layers.

## Rating

| Dimension | Score (1-5) | Description |
|:---|:---:|:---|
| Novelty | ⭐⭐⭐⭐⭐ | CP Superposition is a brand-new discovery that breaks the exclusivity assumption |
| Theoretical Depth | ⭐⭐⭐⭐ | A complete theoretical framework, but based on simplified models |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ | 11 datasets × 6 models × 3 conflict types, offering comprehensive coverage |
| Practicality | ⭐⭐⭐⭐ | Out-of-the-box but doubles inference overhead |
| Writing Quality | ⭐⭐⭐⭐⭐ | Clear structure with highly coherent connection between theory and experiments |
| **Overall** | **⭐⭐⭐⭐½** | High-quality work making a significant contribution to understanding and controlling knowledge conflicts in LMs |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Cracking Factual Knowledge: A Comprehensive Analysis of Degenerate Knowledge Neurons in Large Language Models](../../ACL2025/interpretability/degenerate_knowledge_neurons.md)
- [\[ACL 2026\] Tracing Relational Knowledge Recall in Large Language Models](../../ACL2026/interpretability/tracing_relational_knowledge_recall_in_large_language_models.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](../../ACL2026/interpretability/knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ICLR 2026\] Medical Interpretability and Knowledge Maps of Large Language Models](../../ICLR2026/interpretability/medical_interpretability_and_knowledge_maps_of_large_language_models.md)
- [\[ICLR 2026\] Precise and Interpretable Editing of Code Knowledge in Large Language Models](../../ICLR2026/interpretability/precise_and_interpretable_editing_of_code_knowledge_in_large_language_models.md)

</div>

<!-- RELATED:END -->
