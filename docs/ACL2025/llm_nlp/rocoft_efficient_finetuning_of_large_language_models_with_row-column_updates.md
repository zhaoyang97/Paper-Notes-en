---
title: >-
  [Paper Note] RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates
description: >-
  [ACL 2025][LLM (Other)][PEFT] Proposes RoCoFT, an extremely simple parameter-efficient fine-tuning method: only updates a small subset of row or column parameters in the Transformer weight matrices. It achieves accuracy comparable to state-of-the-art PEFT methods like LoRA on tasks such as GLUE, QA, summarization, and commonsense/mathematical reasoning, while reducing memory and computation overhead. The effectiveness of the method is theoretically explained via the Neural Ta…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "PEFT"
  - "LoRA"
  - "row-column updates"
  - "NTK"
  - "parameter-efficient finetuning"
date: 2026-05-08
content_hash: 84b8560172abcd5e
---

# RoCoFT: Efficient Finetuning of Large Language Models with Row-Column Updates

## Basic Information

**Conference**: ACL 2025  
**Code**: [Kowsher/RoCoFT](https://github.com/Kowsher/RoCoFT)  
**Institution**: Nokia Bell Labs / UCF / UIC  
**Area**: Parameter-Efficient Fine-Tuning / LLM  
**Keywords**: PEFT, LoRA, row-column updates, NTK, parameter-efficient finetuning  

## TL;DR

Proposes RoCoFT, an extremely simple parameter-efficient fine-tuning method: only updates a small subset of row or column parameters in the Transformer weight matrices. It achieves accuracy comparable to state-of-the-art PEFT methods like LoRA on tasks such as GLUE, QA, summarization, and commonsense/mathematical reasoning, while reducing memory and computation overhead. The effectiveness of the method is theoretically explained via the Neural Tangent Kernel (NTK) theory.

## Background & Motivation

- **The Dilemma of Full Fine-Tuning**: As LLM parameter scales grow (from billions to hundreds of billions), storing a full copy of the model for each downstream task becomes impractical. Full fine-tuning is also prone to overfitting and catastrophic forgetting.
- **Development of PEFT Methods**: LoRA achieves efficient fine-tuning through low-rank matrix decomposition, Adapters insert additional modules, and Prefix/Prompt Tuning appends learnable vectors. While effective, these methods still introduce extra parameters or architectural modifications.
- **Core Problem**: Can a **simpler** PEFT method be designed? Simpler methods would not only improve efficiency but also help understand why PEFT is effective in the first place.
- **Key Observation**: The pre-training phase has already captured most critical features, and fine-tuning only needs to adjust a very small fraction of parameters.

## Method

### Core Idea

The methodology of RoCoFT is highly straightforward: **it only updates a small number of row or column parameters in the weight matrices**, while keeping the remaining parameters completely frozen.

### Mathematical Formulation

For the weight matrices $\mathbf{W}_q, \mathbf{W}_k, \mathbf{W}_v, \mathbf{W}_{ff}$ in Transformers, the updates of RoCoFT can be formulated as:

$$\mathbf{W} = \mathbf{W}_0 + \mathbf{R} \quad \text{(row update)}$$
$$\mathbf{W} = \mathbf{W}_0 + \mathbf{C} \quad \text{(column update)}$$

Where $\mathbf{R}$ and $\mathbf{C}$ are restricted weight matrices with at most $r$ non-zero rows or columns.

### Comparison with LoRA

| Characteristic | LoRA | RoCoFT |
|------|------|--------|
| Update Form | $\mathbf{W} = \mathbf{W}_0 + \mathbf{B}\mathbf{A}$ | $\mathbf{W} = \mathbf{W}_0 + \mathbf{R}$ (or $\mathbf{C}$) |
| Extra Parameters | Requires extra matrices $\mathbf{A}$, $\mathbf{B}$ | **No extra parameters**, updated in-place |
| Parameter Count (rank=r, d×k matrix) | $r(d+k)$ | $r \cdot k$ (row) or $r \cdot d$ (column) |
| Forward Computation | Requires matrix multiplication $\mathbf{B}\mathbf{A}$ | No extra computation |
| Initialization issues | Must consider initialization of $\mathbf{A}$, $\mathbf{B}$ | No initialization issues |

### Row/Column Selection Strategy

- **Default Strategy**: Select rows or columns sequentially from the beginning.
- **Key Discovery**: Different selection strategies have minimal impact on performance; that is, **any row or column can produce similar results**, demonstrating the robustness of this method.

### NTK Theoretical Analysis

- The Neural Tangent Kernel (NTK) theory is leveraged to explain the effectiveness of RoCoFT.
- Core Finding: The NTK constructed from a small subset of row/column parameters is numerically close to the full-parameter NTK.
- Validated using NTK kernel logistic regression on multiple tasks, where the classification performance of the restricted parameter set's kernel is comparable to that of the full-parameter kernel.
- This indicates: **the pre-training phase has already captured most of the critical features required for fine-tuning.**

## Experiments

### Experimental Setup

- **Medium-sized Models**: RoBERTa-Base/Large (GLUE), DeBERTa-v3 (SQuAD), BART-Large (Summarization).
- **Large Models**: Bloom-7B, GPT-J-6B, LLaMA2-7B, LLaMA2-13B (commonsense reasoning + mathematical reasoning, across 13 datasets in total).
- **Baselines**: LoRA, AdaLoRA, IA3, Prefix-Tuning, Prompt-Tuning, BitFit, Adapter, MAM Adapter, LoRA-XS, VeRA, Diff Pruning, etc.

### GLUE Benchmark Results (RoBERTa-Base)

| Method | Trainable Params | Avg Score |
|------|-----------|--------|
| Full FT | 124.6M | 83.56 |
| LoRA (r=8) | 0.89M | 84.32 |
| AdaLoRA | 1.03M | 84.06 |
| BitFit | 0.083M | 84.22 |
| SFT | 0.90M | 85.03 |
| **RoCoFT3-Row** | **0.249M** | **85.65** |
| **RoCoFT3-Column** | **0.249M** | **85.55** |

RoCoFT3 achieves the highest average score among all methods with only 0.249M parameters (approximately 28% of LoRA's parameters).

### LLM Inference Task Results (LLaMA2-7B)

| Method | Trainable Params | Commonsense Reasoning Avg | Mathematical Reasoning Avg |
|------|-----------|-------------|-------------|
| LoRA | 24.30M | 75.53 | 78.52 |
| AdaLoRA | 24.90M | 74.81 | 77.48 |
| **RoCoFT3-Row** | **13.47M** | **76.46** | **79.54** |
| **RoCoFT3-Column** | **13.47M** | **76.45** | **79.35** |

On LLaMA2-7B, RoCoFT outperforms LoRA with only about 55% of the parameters.

### LLaMA2-13B Results (Selected)

- RoCoFT3-Row also performs exceptionally on the 13B model, outperforming LoRA and AdaLoRA on multiple tasks.
- Trainable parameters are approximately 24M (compared to 44M for LoRA), representing an increase in parameter efficiency of about 45%.

### Ablation Study

1. **Row vs Column**: Row-wise and column-wise updates exhibit similar performance, with no significant differences.
2. **Impact of Selection Strategy**: Strategies such as random selection, selecting from the beginning, selecting from the end, and uniform interval selection yield similar performance, validating the robustness of the method.
3. **Impact of Rank**: The performance progressively improves as the rank increases from 1 to 3, with Rank=3 being sufficient to achieve outstanding performance.
4. **Application Depth**: Applying updates to all weight matrices (Q, K, V, and FFN) yields the best results.

### NTK Experimental Verification

- The kernel classification performance of the full-parameter NTK is compared against the restricted-parameter NTK on RoBERTa-Base.
- The performance gap between the restricted NTK and the full-parameter NTK on GLUE tasks is only 1-2%.
- This proves from a kernel method perspective that row/column parameters are sufficient to capture the core information needed for fine-tuning.

## Highlights & Insights

1. **Ultra-Simple Design**: This is perhaps the simplest known PEFT method—it introduces no extra parameters or modules, directly updating a subset of the original weight matrices.
2. **Theoretical Backing**: The NTK analysis provides an elegant theoretical explanation rather than purely empirical observations.
3. **Robustness**: It is insensitive to the row/column selection strategy, reducing the hyperparameter search burden.
4. **Efficiency Advantages**: Unlike LoRA, there is no extra matrix multiplication, no initialization issues, and in-place updates reduce memory overhead.
5. **Profound Insight**: Empirical results suggest that pre-training has already learned the vast majority of critical features, and fine-tuning merely adjusts the directions of a tiny subset of parameters.

## Limitations & Future Work

1. **Theoretical Limitations**: Technically, the NTK theory applies strictly to infinitely wide networks, presenting only an approximation for real-world networks of finite width.
2. **Rank Ceiling**: When a larger capacity is needed to adapt to downstream tasks (e.g., tasks with significant domain gaps), a small number of rows/columns may be insufficient.
3. **Untested on Larger Models**: The experiments are conducted on models up to 13B, and have not been validated on models with 70B+ parameters.
4. **Task Coverage**: The evaluation primarily focuses on NLU and reasoning tasks, leaving more complex generative tasks (such as safe dialogues or creative writing) unexplored.
5. **Integration with Other Update Methods**: The combination with other techniques like quantization (e.g., QLoRA) has not been explored.

## Related Work & Insights

- **Low-Rank Methods**: LoRA (Hu et al., 2021), AdaLoRA (Zhang et al., 2023), VeRA (Kopiczko et al., 2023), LoRA-XS (Bałazy et al., 2024).
- **Sparse Fine-Tuning**: Diff Pruning (Guo et al., 2021), SFT (Ansell et al., 2024), Fish Mask (Sung et al., 2021).
- **Other PEFT Methods**: BitFit (Zaken et al., 2021), LayerNorm Tuning (Zhao et al., 2023), IA3 (Liu et al., 2022).
- **NTK Theory**: Jacot et al. (2018), Malladi et al. (2023) utilize NTK to analyze LLM fine-tuning.

## Rating

⭐⭐⭐⭐ (4/5)

- **Novelty**: The method is extremely simple yet highly effective, raising a valuable question: "how simple can PEFT actually be?" (+1)
- **Theoretical Depth**: The NTK analysis provides a solid theoretical foundation for the method, moving beyond purely empirical work. (+0.5)
- **Experimental Thoroughness**: Extensive experiments from medium-sized to large models cover various tasks including NLU, reasoning, and summarization. (+0.5)
- **Value**: The implementation is straightforward, incurs no extra overhead, and is robust to selection strategies. (+0.5)
- **Deductions**: Fail to validate on ultra-large models, lack of exploration on combinations with other techniques like QLoRA, and unknown applicability to certain complex tasks. (-1)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[ACL 2025\] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition](sample-efficient_human_evaluation_of_large_language_models_via_maximum_discrepan.md)
- [\[ACL 2025\] TableLoRA: Low-rank Adaptation on Table Structure Understanding for Large Language Models](table_lora_structure_understanding.md)
- [\[ACL 2025\] A Survey on Efficient Large Language Model Training: From Data-centric Perspectives](a_survey_on_efficient_large_language.md)
- [\[ACL 2025\] LLM Braces: Straightening Out LLM Predictions with Relevant Sub-Updates](llm_braces_straightening.md)

</div>

<!-- RELATED:END -->
