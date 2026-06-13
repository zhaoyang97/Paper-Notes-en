---
title: >-
  [Paper Note] When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging
description: >-
  [ICML2026][Model Compression][Model Merging] This paper points out that model merging suffers not only from task conflicts but also from the repetitive accumulation of spectral directions shared across tasks into excessi…
tags:
  - "ICML2026"
  - "Model Compression"
  - "Model Merging"
  - "Spectral Calibration"
  - "Singular Values"
  - "Task Vectors"
  - "Data-free Post-processing"
date: 2026-05-08
content_hash: 0d81a5458c072b6a
---

# When Shared Knowledge Hurts: Spectral Over-Accumulation in Model Merging

**Conference**: ICML2026  
**arXiv**: [2602.05536](https://arxiv.org/abs/2602.05536)  
**Code**: https://github.com/lyymuwu/SVC  
**Area**: optimization  
**Keywords**: Model Merging, Spectral Calibration, Singular Values, Task Vectors, Data-free Post-processing

## TL;DR
This paper points out that model merging suffers not only from task conflicts but also from the repetitive accumulation of spectral directions shared across tasks into excessively large singular values. It proposes Singular Value Calibration (SVC), a training-free and data-free post-processing method that recalibrates singular values without altering singular vectors, consistently improving merging performance across vision and language tasks.

## Background & Motivation
**Background**: Model merging aims to integrate the capabilities of multiple models fine-tuned from the same base model into a single model. Common practices involve representing weight differences for each task as a task vector or task matrix and combining them into a merged update using rules such as averaging, Task Arithmetic, TIES, or DARE. The appeal of this direction is that it eliminates the need to retrain a multi-task model or maintain multiple specialist models during inference.

**Limitations of Prior Work**: Existing methods primarily attribute merging failures to "conflicts between tasks," often leading to parameter-level pruning, masking, or the removal of sign-inconsistent updates. However, this paper observes a more subtle failure mode: if multiple tasks carry similar shared knowledge within the same spectral subspace, simple linear merging will double-count these common components, magnifying a few top singular directions. This causes the merged model to over-bias toward shared directions while suppressing task-specific information.

**Key Challenge**: Shared knowledge is intended to facilitate transfer, but when shared directions are summed repeatedly, they transition from "common useful signals" to "spectral over-accumulation." In other words, the problem with merged models is not just that different tasks cancel each other out, but also that identical directions are pushed too strongly.

**Goal**: The authors aim to diagnose whether shared knowledge is over-accumulated in each spectral subspace and pull magnified singular values back to reasonable scales without accessing training data, requiring additional fine-tuning, or altering existing merging rules.

**Key Insight**: The paper performs SVD on the merged task matrix, using the output space basis as a common coordinate system, and then projects each individual task matrix onto these output directions. In this way, the responses of different tasks within the same subspace can be directly compared, and over-accumulation can be characterized as projection coefficients greater than 1.

**Core Idea**: Use output space projection coefficients to estimate "how much shared directions are over-counted" in each spectral subspace and then scale only the corresponding singular values. This results in a training-free, data-free spectral post-processor that can be applied after any existing merging method.

## Method
The core of the paper is not to redesign a merging formula but to perform a spectral "health check" and calibration on existing merging results. Given pre-trained weights $W_{pre}$ and multiple fine-tuned weights $W_i$, each task matrix is defined as $\Delta W_i = W_i - W_{pre}$. Any basic merging method first outputs a $\Delta W_{merge}$, which SVC then post-processes to obtain the calibrated $\Delta \tilde{W}_{merge}$.

### Overall Architecture
The overall process can be divided into three steps. First, SVD is performed on the merged matrix $\Delta W_{merge}$ to write it as $U\Sigma V^\top$, where the left singular vector $u^r$ is regarded as the $r$-th output space direction. Second, the response of each task matrix in this output direction is calculated as $a_i^r=(u^r)^\top\Delta W_i$, along with the response of the merged matrix $a_{merge}^r=(u^r)^\top\Delta W_{merge}$. Third, $a_{merge}^r$ is projected onto each $a_i^r$ to obtain the projection coefficient $s_i^r=\langle a_{merge}^r,a_i^r\rangle / \|a_i^r\|_2^2$. These coefficients across all tasks are combined into a subspace calibration factor $\gamma^r$, and the merged update is finally reconstructed using $\tilde{\sigma}^r=\gamma^r\sigma^r$.

The key to this design is that SVC does not re-estimate the directions but only adjusts the intensity of each direction. If a direction indeed corresponds to an output pattern shared by multiple tasks, SVC does not delete it; if this direction becomes so strong due to repetitive accumulation that it suppresses other directions, SVC lowers the corresponding singular value to rebalance the spectral distribution.

### Key Designs
1. **Output Space Projection for Diagnosing Over-Accumulation**:
	- **Function**: Converts the question of "whether shared knowledge is over-accumulated" into a computable spectral subspace metric.
	- **Mechanism**: For the $r$-th left singular vector $u^r$, the response of each task is $a_i^r=(u^r)^\top\Delta W_i$, and the merged response is $a_{merge}^r=\sum_i a_i^r$. If the coefficient $s_i^r$ of $a_{merge}^r$ projected onto $a_i^r$ is greater than 1, it indicates that other tasks provide a positive inner product in the same direction, causing task $i$'s contribution in that direction to be magnified.
	- **Design Motivation**: Parameter-wise comparisons in weight space struggle to find such structural issues, while the output space basis directly corresponds to the response directions of the merged matrix, making it more suitable for measuring whether task behaviors are excessively magnified.

2. **Subspace-level Singular Value Calibration**:
	- **Function**: Aggregates projection coefficients from multiple tasks into a single subspace scaling factor.
	- **Mechanism**: The paper defines $\gamma^r=K/\sum_i \max(\alpha,s_i^r)$. When many tasks have $s_i^r>1$ in a subspace, the denominator increases, resulting in $\gamma^r<1$ and a downward adjustment of the corresponding singular value. When there is no systematic over-accumulation, $\gamma^r$ is close to 1.
	- **Design Motivation**: Coefficients of individual tasks may be affected by noise or task bias; cross-task aggregation ensures that calibration only triggers when shared directions are generally magnified, avoiding over-correction.

3. **Post-processing without Altering Singular Vectors**:
	- **Function**: Allows SVC to be appended to existing mergers such as TA, TIES, DARE, TSV-M, Iso-C/Iso-CTS, etc.
	- **Mechanism**: SVC retains $U$ and $V$, only replacing the original singular values with $\tilde{\sigma}^r=\gamma^r\sigma^r$, and then reconstructs $\Delta\tilde W_{merge}=\sum_r \tilde{\sigma}^r u^r(v^r)^\top$.
	- **Design Motivation**: This avoids introducing new training objectives and does not depend on a calibration set. The additional cost is primarily a single offline SVD, making it suitable for scenarios where data is unavailable or only lightweight correction of existing merged models is desired.

### Loss & Training
SVC itself has no training loss. The paper explains the source of the calibration factor using a projection optimization problem: finding a non-negative scaling $\gamma^r$ such that the projection of $\gamma^r a_{merge}^r$ onto the task response direction $a_i^r$ is as close as possible to $a_i^r$. When $s_i^r>0$, the optimal scaling for a single task is $1/s_i^r$; if cross-task positive inner products lead to $s_i^r>1$, a scaling of less than 1 is naturally obtained. Experiments use the data-free setting $\alpha=1/K$ by default; for TSV-M, $\alpha=1$ is used, meaning it only suppresses over-accumulation without magnifying any subspace.

## Key Experimental Results

### Main Results
The paper tests SVC on both vision and language merging tasks. The vision side covers 8-task and 14-task multi-task classification using ViT-B/32, ViT-B/16, and ViT-L/14; the language side covers Llama2-7B generative evaluation, and BERT/T5/T0 classification or PEFT tasks.

| Scenario | Base Merger | Original Result | With SVC | Gain |
|------|------------|----------|-----------|------|
| CV 8 tasks, ViT-B/32 | Task Arithmetic | 68.9 | 81.9 | +13.0 |
| CV 14 tasks, ViT-L/14 | Task Arithmetic | 57.7 | 76.7 | +19.0 |
| CV 8 tasks, ViT-B/16 | DARE | 71.5 | 84.8 | +13.3 |
| NLP, Llama2 AlpacaEval | Iso-C | 50.0 | 58.9 | +8.9 |
| NLP, Llama2 GSM8K | Iso-C | 42.0 | 51.4 | +9.4 |
| NLP, BERT Avg. Acc | Task Arithmetic | 56.9 | 69.0 | +12.1 |

### Ablation Study
A key ablation in the paper compares output space calibration with input space calibration. When left singular vectors are replaced with right singular vectors, the gains for most base mergers vanish or even drop below the original results.

| Config | TA | TIES | DARE | TSV-M | Iso-C | Iso-CTS |
|------|----|------|------|-------|-------|---------|
| Original Merging | 68.9 | 72.6 | 65.8 | 84.0 | 83.1 | 81.4 |
| SVC Output Space (Ours) | 81.9 | 80.0 | 80.7 | 84.8 | 84.6 | 85.6 |
| SVC Input Space Variant | 64.9 | 65.7 | 67.5 | 84.0 | 82.1 | 85.5 |

| Backbone | SVC Offline Time | Memory Usage |
|----------|--------------|----------|
| ViT-B/32 | 5.1 s | 1027.4 MiB |
| ViT-B/16 | 8.2 s | 1082.8 MiB |
| ViT-L/14 | 15.6 s | 1488.5 MiB |
| LLaMA2 7B | 517.2 s | 1898.7 MiB |
| Qwen2.5 7B | 249.3 s | 2513.1 MiB |

### Key Findings
- SVC provides the largest improvements for weak mergers, indicating that spectral over-accumulation is a major failure source for linear methods like Task Arithmetic; however, it still yields stable small gains for stronger spectral methods like TSV-M and Iso-C/Iso-CTS.
- The output space is more critical than the input space. Left singular vectors correspond to the output response of the merged matrix and can directly measure whether task behaviors are magnified; right singular vectors reflect input patterns and are less reliable for calibrating merging behavior.
- Inhibitory calibration ($\alpha=1$) already provides stable improvements. Allowing $\alpha<1$ to magnify low-accumulation subspaces yields mixed results, suggesting that "correcting over-strong directions first" is more robust than "simultaneously strengthening weak directions."
- SVC is a one-time offline post-processor, significantly cheaper than training-based merging even at the LLM scale, although it still incurs the computational cost of SVD for large matrices.

## Highlights & Insights
- The paper clearly explains "why shared knowledge can hurt merging": it is not that sharing itself is harmful, but that linear merging double-counts shared directions, concentrating spectral energy into a few top subspaces.
- The projection coefficient $s_i^r$ is a highly interpretable diagnostic metric. It links cross-task inner products, behavior magnification, and singular value inflation, moving the method beyond empirical post-processing to one supported by clear spectral analysis.
- The engineering form of SVC is practical: it requires no calibration data, no task labels, no change to inference routing, and no knowledge of the input distribution. This makes it suitable for scenarios where multiple task adapters or fine-tuned checkpoints already exist in a model repository.
- "Modifying only singular values" is a restrained yet effective choice. It avoids the complexity of re-searching for directions and reduces the risk of disrupting the structure of existing mergers.

## Limitations & Future Work
- The method relies on performing SVD on layer-wise task matrices. While an offline operation, this may become a bottleneck for larger models, higher-rank updates, or multi-layer full-weight merging.
- The theory mainly explains linear weight merging and local linear layer behavior; its explanation for non-linear functional combinations, token distribution changes, and dynamic routing at inference time remains limited.
- SVC balances all tasks in a data-free manner by default. If a user genuinely cares about specific task priorities, they would need to combine this with the target-task calibration or additional preference information mentioned in the paper.
- Current experiments cover vision, NLP, and 7B-class LLMs, but further testing on instruction tuning, multimodal adapter merging, LoRA merging, and safety-aligned model merging for spectral over-accumulation is possible.

## Related Work & Insights
- **vs Task Arithmetic**: Task Arithmetic sums task vectors directly, which is simple and efficient but prone to over-accumulating shared directions; SVC can serve as its post-processor, improving 8-task ViT-B/32 from 68.9 to 81.9.
- **vs TIES / DARE**: These methods mostly handle sign conflicts or sparsify conflicting updates at the parameter level; SVC focuses on global spectral structure, thus addressing the problem of "non-conflicting but over-strong" shared directions.
- **vs TSV-M / Iso-C / Iso-CTS**: These methods already use a spectral perspective to construct merged updates; SVC differs by diagnosing output subspace overlap after merging as a post-processor, without needing to rewrite the basic merging rules.
- **Insight**: For the fusion of adapters, LoRA, and specialist models, merging quality may depend not only on task correlation but also on whether shared directions are repetitively magnified spectrally; similar diagnostics could be used in the future to select task combinations for merging.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Spectral space merging has an existing foundation, but formalizing shared knowledge over-accumulation through output space projection and singular value inflation is highly distinctive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers vision and language, multiple mergers, and ablations; conclusions are solid, though more real-world LLM adapter merging cases would be more complete.
- **Writing Quality**: ⭐⭐⭐⭐☆ Motivation, theory, and algorithms are naturally linked. A few tables are information-dense and require familiarity with the model merging background.
- **Value**: ⭐⭐⭐⭐⭐ As a training-free, data-free post-processor, it is highly practical and provides a clear diagnostic tool for model merging failure analysis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Saliency-Aware Model Merging](saliency-aware_model_merging.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] Decouple Searching from Training: Scaling Data Mixing via Model Merging for Large Language Model Pre-training](decouple_searching_from_training_scaling_data_mixing_via_model_merging_for_large.md)
- [\[ICCV 2025\] FREE-Merging: Fourier Transform for Efficient Model Merging](../../ICCV2025/model_compression/free-merging_fourier_transform_for_efficient_model_merging.md)
- [\[ICLR 2026\] RAIN-Merging: A Gradient-Free Method to Enhance Instruction Following Through Model Merging](../../ICLR2026/model_compression/rain-merging_a_gradient-free_method_to_enhance_instruction_following_through_mod.md)

</div>

<!-- RELATED:END -->
