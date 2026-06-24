---
title: >-
  [Paper Note] Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning
description: >-
  [ACL 2025][Reasoning][Representation Fine-tuning] This paper proposes CRFT, a method that automatically identifies "critical representations" with the greatest impact on reasoning outputs across Transformer layers through information flow analysis. By optimizing these representations in a low-rank linear subspace using supervised learning, CRFT improves the accuracy of LLaMA-2-7B on GSM8K by 18.2% while utilizing only 0.016% of the model parameters.
tags:
  - "ACL 2025"
  - "Reasoning"
  - "Representation Fine-tuning"
  - "Chain-of-Thought"
  - "Parameter-Efficient Fine-Tuning"
  - "Information Flow Analysis"
  - "Critical Representation"
date: 2026-05-08
content_hash: 58223c31fe51a45f
---

# Enhancing Chain-of-Thought Reasoning with Critical Representation Fine-tuning

**Conference**: ACL 2025  
**arXiv**: [2507.10085](https://arxiv.org/abs/2507.10085)  
**Code**: None  
**Area**: LLM Reasoning  
**Keywords**: Representation Fine-tuning, Chain-of-Thought, Parameter-Efficient Fine-Tuning, Information Flow Analysis, Critical Representation

## TL;DR

This paper proposes CRFT, a method that automatically identifies "critical representations" with the greatest impact on reasoning outputs across Transformer layers through information flow analysis. By optimizing these representations in a low-rank linear subspace using supervised learning, CRFT improves the accuracy of LLaMA-2-7B on GSM8K by 18.2% while utilizing only 0.016% of the model parameters.

## Background & Motivation

**Background**: Large Language Models (LLMs) have made significant progress in complex reasoning tasks. Chain-of-Thought (CoT) reasoning is a core technique that enhances model reasoning capabilities by decomposing the reasoning process into multiple intermediate steps. Parameter-Efficient Fine-Tuning (PEFT) methods, such as LoRA, have been widely used to adapt LLMs to downstream tasks.

**Limitations of Prior Work**: Representation Fine-tuning (ReFT), as an emerging class of PEFT methods, achieves parameter efficiency by directly editing the model representation space. However, ReFT performs poorly on complex reasoning tasks because it modifies representations at fixed positions at the beginning and end of each layer. The impact of these fixed-position representations on the output is uncertain—some may not contribute to reasoning at all, while truly critical representations might be overlooked.

**Key Challenge**: There is a fundamental conflict between the "fixed-position selection" strategy of ReFT and the fact that "representation importance is context-dependent" in reasoning tasks. In complex reasoning, each layer contains representations that are genuinely critical—either they aggregate important information from preceding layers, or they modulate representations in subsequent layers. However, which representations are critical depends on the specific input and reasoning chain, making them impossible to determine through fixed rules.

**Goal**: (1) Automatically identify the critical representations in each Transformer layer that have the greatest impact on reasoning outputs; (2) target and optimize these representations with minimal trainable parameters; (3) validate the generalizability of the method across various reasoning scenarios and models.

**Key Insight**: The authors observe that adding small Gaussian noise (0.01) to random representations of each layer in LLaMA-2-7B results in a 1.4% drop in GSM8K accuracy, indicating that model performance is highly sensitive to specific representations. Further analysis reveals that information flow (saliency and attention scores) can effectively reveal which representations play critical roles.

**Core Idea**: Dynamically identify critical representations in each layer using information flow analysis (attention scores and saliency scores), and then optimize these representations by learning adaptive update directions in a low-rank subspace, achieving lightweight yet precise reasoning enhancement.

## Method

### Overall Architecture

The overall pipeline of CRFT consists of two phases: **identification** and **optimization**. The input is a sequence of representations after passing the token sequence through the embedding layer, which is computed layer-by-layer through $L$ Transformer layers. In each layer, CRFT first filters a set of critical representations $M(h)$ using information flow analysis (attention scores or saliency scores). It then applies a low-rank linear projection correction to the representations in this set while keeping the base model parameters frozen. Finally, the model uses the updated representations from the last layer to generate the reasoning answer.

### Key Designs

1. **Self-Referential Filtering**:

    - **Function**: Identify critical representations with highly self-aggregated internal information.
    - **Mechanism**: If the information of representation $i$ at layer $l$ primarily flows back to itself (i.e., the value of $\text{Info}(i,i)$ is large), it indicates that this representation has effectively accumulated key information. Due to softmax normalization, a high self-referential ratio implies less outward information propagation, making the representation an information "sink". Specifically, there are two measurement approaches: Self-Referential Attention Filtering (SAF), measured directly using the attention diagonal elements $A_i^{(l)}$, and Self-Referential Saliency Filtering (SSF), measured by the Hadamard product of attention and gradients, considering both information flow direction and sensitivity to the output.
    - **Design Motivation**: In reasoning tasks, representations carrying key intermediate computation results tend to aggregate information at their own positions. Detecting this "information self-aggregation" pattern allows for the precise localization of critical nodes.

2. **Multi-Referential Filtering**:

    - **Function**: Identify critical representations that exert wide-ranging regulatory influence over multiple downstream representations.
    - **Mechanism**: If representation $j$ has a significant impact on multiple other representations (i.e., the column average $\frac{1}{n+k-j+1}\sum_i \text{Info}(i,j)$ exceeds a threshold $\beta$), representation $j$ is deemed a critical "regulator". Similarly, there are two implementations: MAF (attention-based) and MSF (saliency-based). Additionally, the union of self-referential and multi-referential key representations can be taken (Union strategy) to avoid omissions.
    - **Design Motivation**: This is complementary to self-referential filtering—some critical representations are not information aggregators but information broadcasters. They propagate information to a large number of positions in subsequent layers, serving as "hub" nodes in the reasoning chain.

3. **Low-Rank Subspace Optimization**:

    - **Function**: Learn adaptive correction directions for critical representations while keeping the original model frozen.
    - **Mechanism**: For each representation $h$ identified as critical, a correction vector $\Delta h = R^T(Wh + b - Rh)$ is learned, where $R \in \mathbb{R}^{r \times d}$ is a row-orthogonal projection matrix, and $W$ and $b$ are learnable parameters. The correction is constrained within an $r$-dimensional low-rank subspace to ensure an extremely small parameter footprint. Non-critical representations remain unchanged; only representations within $M(h)$ are modified.
    - **Design Motivation**: The correction directions for critical representations vary by context and need to be adaptively determined through supervised learning. The low-rank constraint controls the parameter space (only 0.016%) and acts as a regularizer to prevent overfitting.

### Loss & Training

Training employs the standard cross-entropy loss based on the supervised learning framework for CoT reasoning steps. Mathematical reasoning is trained on the Math10K dataset, and commonsense reasoning is trained on the self-constructed Commonsense60K dataset (which includes reasoning steps). All experiments use the AdamW optimizer with a default rank $r=8$, 14 intervention representations per layer, threshold $\alpha=\beta=0.05$, and use "position-based ranking" as the selection criterion.

## Key Experimental Results

### Main Results

| Method | Trainable Params (%) | GSM8K Accuracy |
|------|-------------|------------|
| LLaMA-2-7B (No FT) | - | 14.6% |
| LoRA (r=64) | 0.826% | 38.5% |
| LoRA (r=8) | 0.103% | 36.7% |
| ReFT (p7+s7) | 0.031% | 29.0% |
| **CRFT-Union(attn)** | **0.016%** | **32.8%** |
| CRFT-MAF | 0.016% | 32.1% |
| CRFT-SAF | 0.016% | 30.4% |

Cross-model and cross-dataset results (Arithmetic reasoning + Commonsense reasoning):

| Model | Method | AQuA | MAWPS | SVAMP | BoolQ | SocialIQA | WinoGrande | OpenBookQA |
|------|------|------|-------|-------|-------|-----------|------------|------------|
| LLaMA-2-7B | ReFT | 21.7 | 80.7 | 52.2 | 50.7 | 61.2 | 51.7 | 58.6 |
| LLaMA-2-7B | CRFT-MAF | 27.6 | 81.1 | 53.4 | 60.5 | 52.8 | 68.4 | 66.4 |
| LLaMA-3-8B | ReFT | 46.9 | 87.0 | 74.2 | 62.1 | 60.2 | 56.0 | 66.0 |
| LLaMA-3-8B | CRFT-SSF | 50.0 | 86.6 | 78.1 | 66.6 | 74.7 | 62.0 | 77.0 |
| Mistral-7B | ReFT | 32.3 | 84.9 | 67.4 | 62.5 | 64.6 | 58.5 | 63.8 |
| Mistral-7B | CRFT-MSF | 41.3 | 87.4 | 66.9 | 65.0 | 71.8 | 62.3 | 72.8 |

### Ablation Study

| Configuration | GSM8K Accuracy | Description |
|------|-----------|------|
| Threshold $\alpha=1.0$ | 24.7% | Threshold too high, too few representations selected |
| Threshold $\alpha=0.25$ | 30.0% | Medium threshold |
| Threshold $\alpha=0.05$ (Default) | 29.6% | Balance point |
| Threshold $\alpha=0.01$ | 33.2% | More representations selected, optimal performance |
| Position-based ranking | 29.6% | Default strategy |
| Score-based ranking | 28.7% | Slightly lower |
| Random selection | 23.1% | Significant drop, proving non-random intervention |
| Intervention only on Layer 0 | 24.9% | Intervention only on early layers has some effect |
| Intervention only on Layer 31 | 22.7% | Last layer intervention yields limited effect |
| Intervention on all layers | 29.6% | Full intervention yields the best result |

### Key Findings

- **Critical Representation Identification is Crucial**: Adding 0.02 noise to critical representations drops accuracy from 100% (for correct samples) to 21.1%, whereas doing the same to non-critical representations only drops it to 74.1%, justifying the high influence of critical representations.
- **Varied Strengths of Different Strategies**: SAF and MAF capture critical representations from different dimensions, and the Union strategy steadily achieves superior performance without manual selection.
- **Extreme Parameter Efficiency**: CRFT achieves comparable or even better performance across most benchmarks using only 1/6 of LoRA's parameters and 1/2 of ReFT's parameters.
- **Few-shot Extension**: In one-shot scenarios, accuracy increases by 16.4%, with better results achieved when using independent update vectors for demonstrations and questions.

## Highlights & Insights

- **Information Flow-Driven Criticality Assessment**: Unlike ReFT which relies on empirical rules for position selection, CRFT uses information flow analysis to provide a theoretical foundation and automatic approach for identifying "which representation is worth optimizing". This concept can be applied to any scenario that requires locating critical intermediate states—such as identifying critical patch representations in Vision Transformers for parameter-efficient fine-tuning.
- **Dual Perspectives of "Attention Sinks" and "Information Broadcasters"**: By simultaneously considering self-referential (information aggregators) and multi-referential (information broadcasters) roles, a complementary set of critical representations is formed. This framework is more comprehensive than simply looking at attention weights alone.
- **Noise Perturbation Validation Strategy**: Demonstrating that the selected representations are indeed critical by adding small noise and observing output changes is a simple and effective method for interpretability validation.

## Limitations & Future Work

- **Focus Only on Positively Impacting Representations**: Current methods prioritize representations that have a major impact on the output, but do not distinguish between positive and negative impacts. Prioritizing the correction of negative-impact representations could be more efficient.
- **Optimization Constrained to Linear Spaces**: Since correction is restricted to a low-rank linear subspace, potential optimization opportunities of non-linear directions might be overlooked.
- **Single-GPU Training Constraints**: Due to GPU memory limits, few-shot experiments were only scaled up to two-shot; longer demonstration scenarios remain to be explored.
- **Strategy Selection Still Relies on Empirical Experiments**: Although the Union strategy is relatively stable, the optimal strategy varies by model and task, and a unified automatic selection mechanism is still lacking.

## Related Work & Insights

- **vs ReFT**: ReFT modifies representations at fixed start and end positions of each layer, and its position selection relies on trial-and-error on other datasets, lacking interpretability. CRFT dynamically identifies critical positions through information flow and outperforms ReFT using only half of its parameters.
- **vs LoRA**: LoRA modifies weight matrices, requiring more than six times the parameters of CRFT. CRFT directly intervenes at the representation level, which theoretically is more precise, though its application scenarios might be narrower.
- **vs PASTA**: PASTA manually defines which tokens require attention enhancement, whereas CRFT automates this process and extends it to the optimization of the entire representation space.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing information flow analysis to key position selection in representation fine-tuning is novel, although the performance gains built on top of ReFT are clearly incremental.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely thorough, utilizing 8 datasets, 4 models, various ablation studies, noise validation, and attention visualizations.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-thought-out incremental ablation studies, but complex mathematical notation might raise the reading barrier.
- Value: ⭐⭐⭐⭐ Offers a new perspective on parameter-efficient fine-tuning for reasoning tasks; the efficiency under a 0.016% parameter constraint is highly noteworthy.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning](tract_regression_cot.md)
- [\[ACL 2025\] Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs](dcot_diverse_cot_refinement.md)
- [\[ACL 2025\] CoT-Valve: Length-Compressible Chain-of-Thought Tuning](cot-valve_length-compressible_chain-of-thought_tuning.md)
- [\[ACL 2025\] Unlocking General Long Chain-of-Thought Reasoning Capabilities of Large Language Models via Representation Engineering](glore_long_cot_representation.md)
- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)

</div>

<!-- RELATED:END -->
