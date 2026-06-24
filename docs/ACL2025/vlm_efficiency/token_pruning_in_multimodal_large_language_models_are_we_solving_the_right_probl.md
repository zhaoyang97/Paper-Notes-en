---
title: >-
  [Paper Note] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?
description: >-
  [ACL 2025][Multimodal Efficiency][token pruning] Large-scale benchmark experiments reveal several fundamental issues with current visual token pruning methods for MLLMs: elaborately designed pruning strategies (such as FastV and SparseVLM) underperform even naive methods like random selection and pooling on most benchmarks. This is due to positional bias in attention scores, misuse of language information, imbalance between importance and redundancy…
tags:
  - "ACL 2025"
  - "Multimodal Efficiency"
  - "token pruning"
  - "Multimodal Large Language Models"
  - "visual token compression"
  - "attention bias"
  - "inference acceleration"
date: 2026-05-08
content_hash: f01305b0b8e349e9
---

# Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?

**Conference**: ACL 2025  
**arXiv**: [2502.11501](https://arxiv.org/abs/2502.11501)  
**Authors**: Zichen Wen, Yifeng Gao (SJTU), Weijia Li (Sun Yat-sen University), Conghui He (Shanghai AI Lab), Linfeng Zhang (SJTU)  
**Code**: Not released  
**Area**: Multimodal VLM  
**Keywords**: token pruning, Multimodal Large Language Models, visual token compression, attention bias, inference acceleration  

## TL;DR

Large-scale benchmark experiments reveal several fundamental issues with current visual token pruning methods for MLLMs: elaborately designed pruning strategies (such as FastV and SparseVLM) underperform even naive methods like random selection and pooling on most benchmarks. This is due to positional bias in attention scores, misuse of language information, imbalance between importance and redundancy, and unreliable evaluation metrics.

## Background & Motivation

### Background
Multimodal Large Language Models (MLLMs) excel in cross-modal understanding and generation, but suffer from significant inference overhead. Visual encoders generate a massive number of tokens (e.g., 576 tokens in LLaVA-1.5, and up to 2,880 tokens in LLaVA-NeXT), far exceeding the length of text prompts. As a training-free acceleration method, token pruning has attracted significant attention by identifying redundant visual tokens and removing or merging them to reduce computational and KV cache costs.

### Limitations of Prior Work
- FastV prunes low-score tokens using attention scores from early LLM layers, SparseVLM introduces text-visual cross-attention to guide pruning, and MustDrop compresses tokens throughout the encoding-prefilling-decoding lifecycle.
- However, these methods **show no advantage over, or even perform worse than, random selection and uniform pooling**, an counter-intuitive phenomenon that has never been seriously addressed.
- Existing evaluations widely rely on FLOPs and token counts to measure acceleration, ignoring actual latency.
- Prior works fail to consider training-stage token merging and compression already executed in models like Qwen2-VL.

### Design Motivation
This paper does not propose a new method, but instead systematically asks five fundamental questions: (1) Why do existing methods perform worse than random selection? (2) Are attention scores reliable? (3) When is language information useful? (4) How should importance and redundancy be balanced? (5) Are current evaluation protocols comprehensive and fair?

## Method

### Overall Architecture
This work is an empirical analysis and does not propose a new architecture. Instead, it systematically dissects key design choices in token pruning through controlled experiments and comparative analyses. The experiments cover five models (LLaVA-1.5-7B/13B, LLaVA-Next-7B, and Qwen2-VL-7B/72B) across 9+ datasets (including GQA, MMBench, MME, POPE, ScienceQA, TextVQA, VizWiz, RefCOCO, and Visual Haystack), comparing FastV, SparseVLM, and MustDrop against baselines like Random and Pooling.

### Key Findings 1: Positional Bias Leads to Failure in Attention Selection
Through statistical analysis of 8,910 samples from the POPE dataset, it is found that FastV relies on the attention scores of the last token to visual tokens to evaluate importance. However, **visual tokens at the end of the sequence receive significantly higher attention scores and retention frequencies**, implying a severe positional bias in attention scoring. In contrast, Random and Pooling naturally maintain space-uniform distributions.

To validate the hypothesis that "spatial uniformity outperforms positional bias", the authors propose **Window FastV**: introducing a sliding window mechanism into FastV, which selects tokens within each window at a fixed ratio to guarantee spatial uniformity of the retained tokens. The results show that:
- At a 75% pruning rate, Window FastV reduces the average performance drop by 3.4 percentage points compared to Vanilla FastV.
- At an 88.9% pruning rate, this gap widens to 9 percentage points.
- On the RefCOCO spatial grounding task, spatially uniform methods (Window FastV, Random, and Pooling) significantly outperform non-uniform methods (FastV and SparseVLM).

### Key Findings 2: The Utility of Language Information Depends on Task Types
Converting FastV into FastV_VIS (replacing the last text token with the last visual token to compute attention) and testing on the Visual Haystack task:
- Performance drops significantly when text guidance is removed, confirming that **language information is crucial in strongly text-dependent tasks**.
- SparseVLM (with text guidance) maintains almost the same accuracy as the uncompressed model at a 77.8% compression rate.
- However, on general VQA benchmarks, purely visual pruning methods (such as FasterVLM) actually perform better.
- Conclusion: Language information is not universally beneficial, and is only worth introducing when the task is heavily text-dependent.

### Key Findings 3: The $\alpha$-Dilemma between Importance and Redundancy
Formalizing two criteria of token pruning from an information-theoretic perspective:
- **Redundancy Criterion**: Maximize $I(\mathbf{X};\mathbf{X'})$, keeping spatial/input structural integrity (corresponding to the Information Bottleneck principle).
- **Importance Criterion**: Maximize $I(\mathbf{X'};\mathbf{Y})$, retaining tokens critical to the prediction.

Introduce a parameter $\alpha$ to balance both: $\text{Score}(x_i) = \alpha \cdot \text{Predictive Importance} + (1-\alpha) \cdot \text{Pattern Uniqueness}$

Experiments show that the optimal $\alpha$ varies across tasks:
- Perception tasks (MME, POPE): $\alpha=0.0\sim0.1$ is optimal, favoring redundancy priority.
- Knowledge-intensive tasks (SQA, TextVQA): $\alpha=0.8\sim0.9$ is optimal, favoring importance priority.

### Key Findings 4: FLOPs Do Not Reflect Actual Speedup
On LLaVA-Next-7B, the FLOPs of SparseVLM and FastV differ by only 2.8%, but SparseVLM has a 26.8% higher latency. This is because SparseVLM executes pruning across 4 layers (vs. FastV in only 1 layer), requiring complete attention maps at each layer and preventing the usage of Flash Attention. This reveals that **compatibility with hardware-efficient operators (such as Flash Attention)** is the key factor determining actual speedup.

### Key Findings 5: Training-Aware Compression is Ignored
Qwen2-VL already merges 4 adjacent patches into 1 token during the training stage (TACR=4). If this training-stage compression is accounted for during token pruning evaluation (i.e. Actual Token Reduction Rate = TACR $\times$ Inference Reduction Rate), even at an 88.9% total reduction rate, the performance remains nearly identical to the original model (99.6% vs. 84.0% without considering TACR).

## Key Experimental Results

### Table 1: Naive vs. Elaborate Methods on LLaVA-1.5-7B (Retaining 144 tokens, ↓75%)

| Method | GQA | MMB | MME | POPE | SQA | TextVQA | Avg. Retention |
|------|-----|-----|-----|------|-----|---------|-----------|
| Vanilla (Full Tokens) | 61.9 | 64.7 | 1862 | 85.9 | 69.5 | 58.2 | 100% |
| Random | 59.0 | 62.2 | 1736 | 79.4 | 67.8 | 51.7 | 95.0% |
| Pooling | 59.1 | 62.5 | 1763 | 81.4 | 69.1 | 53.4 | 96.4% |
| Window FastV | 59.2 | 59.3 | 1737 | 80.3 | 66.4 | 50.8 | 93.2% |
| Vanilla FastV | 56.5 | 59.3 | 1689 | 71.8 | 65.3 | 53.6 | 89.8% |
| SparseVLM | 55.1 | 59.5 | 1711 | 77.6 | 69.3 | 54.9 | 93.5% |

Random and Pooling outperform FastV and SparseVLM on approximately 2/3 of the benchmarks.

### Table 2: Actual Latency vs. FLOPs Comparison (LLaVA-Next-7B, Retaining 320 tokens)

| Method | Token Count ↓ | Latency | FLOPs ↓ | KV Cache ↓ | POPE |
|------|---------|------|--------|-----------|------|
| Vanilla | 2880 | 36:16 | 100% | 1512 MB | 86.5 |
| FastV | 320 | 18:17 | 12.8% | 168 MB | 78.3 |
| SparseVLM | 320 | 23:11 | 15.6% | 168 MB | 82.3 |
| MustDrop | 320 | 23:40 | 11.5% | 168 MB | 82.1 |

MustDrop, which has the lowest FLOPs (11.5%), unexpectedly has the highest latency, while FastV has slightly higher FLOPs but the lowest latency.

### Table 3: Effect of Parameter $\alpha$ on Different Tasks (LLaVA-1.5-7B, Retaining 144 tokens)

| $\alpha$ | MME | POPE | SQA | TextVQA |
|----------|-----|------|-----|---------|
| 0.0 (Pure Redundancy) | 1707 | **82.8** | 64.8 | 53.6 |
| 0.1 | **1714** | 82.6 | 65.2 | 53.8 |
| 0.5 | 1711 | 81.6 | 65.3 | 54.3 |
| 0.8 | 1699 | 79.7 | 65.2 | 54.5 |
| 0.9 | 1680 | 75.6 | **65.7** | 54.2 |
| 1.0 (Pure Importance) | 1689 | 71.8 | 65.3 | 53.6 |

Perception tasks favor a low $\alpha$ (redundancy priority), while knowledge-intensive tasks favor a high $\alpha$ (importance priority).

## Highlights & Insights

- **Myth Busting**: Proves that "elaborately designed methods underperform random selection" using extensive experimental data, forcing the community to re-evaluate the basic assumptions of token pruning.
- **Discovery of Positional Bias**: Uncovers the source of systematic bias in attention scores. The simple fix of Window FastV brings significant improvements, indicating that the root cause lies in the scoring mechanism rather than the pruning framework itself.
- **Information-Theoretic Framework**: Unifies both importance and redundancy criteria using mutual information and the Information Bottleneck principle. Experiments with the $\alpha$ parameter demonstrate there is no one-size-fits-all formula, necessitating task adaptation.
- **Rectifying Evaluation Metrics**: Highlights the mismatch between FLOPs and actual latency (a 2.8% FLOPs difference leads to a 26.8% latency difference), reminding the community to focus on Flash Attention compatibility.
- **Joint Training-Inference Perspective**: Experiments on Qwen2-VL show that training-stage compression renders inference-stage pruning nearly lossless, suggesting that the research focus should shift towards training-aware compression.

## Limitations & Future Work

- The experiments are mainly based on the LLaVA series and Qwen2-VL, without covering broader architectures like InternVL, MiniCPM-V, or Phi-Vision.
- The conclusions are not validated on video understanding tasks (where token redundancy patterns in videos might differ).
- Window FastV is only used as a hypothesis-testing tool and is not further optimized into a complete method.
- The analysis of parameter $\alpha$ only uses FastV attention scores to approximate mutual information, which in reality involves more complex calculations.
- The combination effects of token pruning with other compression techniques like quantization and distillation are not explored.
- The analysis of training-aware compression is limited to Qwen2-VL's PatchMerger, lacking detailed comparisons with other training compression schemes (e.g., Q-Former).

## Related Work & Insights

- **FastV (Chen et al., 2024)**: The main target of analysis in this work, which prunes tokens using LLaMA's 2nd-layer attention scores and is shown to suffer from severe positional bias.
- **SparseVLM (Zhang et al., 2024)**: Introduces text-guided cross-modal attention selection. It performs well on text-dependent tasks (like Visual Haystack) but underperforms random selection on general tasks.
- **MustDrop (Liu et al., 2024)**: Full-lifecycle multi-stage compression. It achieves the lowest FLOPs, but has the highest latency, exposing the hardware-unfriendly nature of multi-layer pruning.
- **ToMe (Bolya et al., 2023)**: Token merging in ViT. Its approach of merging spatially adjacent tokens aligns with the findings on spatial uniformity in this paper.
- **Qwen2-VL (Wang et al., 2024)**: Applies 4:1 patch merging during training, providing a key case study for "training-aware compression".

**Insights**: Future work on token pruning should (1) ensure a spatially uniform distribution of retained tokens, (2) adaptively balance importance and redundancy based on the task type, (3) prioritize simple pruning in early layers to remain compatible with Flash Attention, and (4) shift research focus towards training-aware compression rather than inference-stage post-processing.

## Rating

- Novelty: ⭐⭐⭐⭐ — Does not propose a new method but systematically uncovers fundamental flaws in existing ones with a unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Solid controlled experiments with comprehensive comparisons across 5 models, 9+ datasets, and multiple pruning rates.
- Writing Quality: ⭐⭐⭐⭐ — Logically organized around answering five sequential questions, featuring clear structures and rich diagrams.
- Value: ⭐⭐⭐⭐⭐ — Crucial for rectifying misconceptions in the MLLM inference acceleration community, a must-read for researchers in this field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LearnPruner: Rethinking Attention-based Token Pruning in Vision Language Models](../../ICLR2026/vlm_efficiency/learnpruner_rethinking_attention-based_token_pruning_in_vision_language_models.md)
- [\[ECCV 2024\] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](../../ECCV2024/vlm_efficiency/ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)
- [\[ICCV 2025\] LLaVA-PruMerge: Adaptive Token Reduction for Efficient Large Multimodal Models](../../ICCV2025/vlm_efficiency/llava-prumerge_adaptive_token_reduction_for_efficient_large_multimodal_models.md)
- [\[ICLR 2026\] Task-Related Token Compression in Multimodal Large Language Models from an Explainability Perspective](../../ICLR2026/vlm_efficiency/task-related_token_compression_in_multimodal_large_language_models_from_an_expla.md)
- [\[ICML 2025\] CoreMatching: A Co-adaptive Sparse Inference Framework with Token and Neuron Pruning for Comprehensive Acceleration of Vision-Language Models](../../ICML2025/vlm_efficiency/corematching_a_co-adaptive_sparse_inference_framework_with_token_and_neuron_prun.md)

</div>

<!-- RELATED:END -->
