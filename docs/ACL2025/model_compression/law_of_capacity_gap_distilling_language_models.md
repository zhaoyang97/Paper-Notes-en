---
title: >-
  [Paper Note] Towards the Law of Capacity Gap in Distilling Language Models
description: >-
  [ACL 2025][Model Compression][Knowledge Distillation] Unveils the "Law of Capacity Gap" in language model distillation, which states that the size of the optimal teacher model scales linearly with the student model size (approximately 2.5x). This turns the "impossible triangle" in LLM distillation into a solvable problem, leading to the successful distillation of the 3B MiniMA model.
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Capacity Gap"
  - "Scaling Law"
  - "MiniMA"
date: 2026-05-08
content_hash: 782df26bc39c8640
---

# Towards the Law of Capacity Gap in Distilling Language Models

**Conference**: ACL 2025  
**arXiv**: [2311.07052](https://arxiv.org/abs/2311.07052)  
**Area**: LLM / Model Compression  
**Keywords**: Knowledge Distillation, Capacity Gap, Scaling Law, Model Compression, MiniMA

## TL;DR

Unveils the "Law of Capacity Gap" in language model distillation, which states that the size of the optimal teacher model scales linearly with the student model size (approximately 2.5x). This turns the "impossible triangle" in LLM distillation into a solvable problem, leading to the successful distillation of the 3B MiniMA model.

## Background & Motivation

### Background

**Background**: Knowledge distillation is one of the core techniques for model compression. However, there exists a widely observed "Curse of Capacity Gap" in distillation:

### Limitations of Prior Work

**Limitations of Prior Work**: A larger teacher model does not always yield a better student model.

### Key Challenge

**Key Challenge**: When the capacity gap between the teacher and student is too large, the student's performance degrades instead.

### Mechanism

**Mechanism**: Therefore, an optimal teacher size exists, but finding it requires exhaustive experimentation across various scales.

In the LLM era, this forms an "impossible triangle" among:
1. The desired student size
2. The optimal teacher size
3. Low computational overhead

These three cannot be simultaneously satisfied—determining the optimal teacher requires computationally expensive distillation trials.

## Method

### Overall Architecture

Through systematic experiments on small-scale models (<3B), a stable linear relationship between the student scale and the optimal teacher scale was discovered. This "law" is then extrapolated to the LLM scale (7B$\rightarrow$3B distillation) to verify its effectiveness.

### Key Designs

**1. Small-scale Pilot Experiments**

Experimental workflow:
- Select teacher models of different scales (GPT2 series: 140M-1.6B; Pythia series: 70M-2.8B)
- Compress teachers to students of various sizes via structured pruning
- Distill each (teacher size, student size) pair
- Observe which teacher generates the best performance for each student size

**2. Structured Pruning**

Pruning based on parameter expressiveness scores (accumulated absolute gradients) is adopted:
- Learnable variables $\xi, \nu$ are attached to the attention heads and FFN intermediate neurons of each layer.
- A cross-layer shared variable $\mu$ is attached to the hidden states.
- The expressiveness score $\mathbb{I} = \mathbb{E}|\partial\mathcal{L}/\partial\xi|$ is calculated for importance ranking.
- Each component uses the same sparsity $p' = 1 - \sqrt{1-p}$ (due to the multiplicative effect between the sparsity of the hidden state and those of the heads/neurons).

**3. Verification of Three Key Hypotheses**

- H1: The capacity gap indeed affects student performance. ✓
- H2: Given a student size, an optimal teacher size exists. ✓ (Student performance exhibits a rise-then-decline pattern as the teacher size scales up)
- H3: The optimal teacher size remains consistent across different settings. ✓ (occurring at approximately ~60% sparsity)

**4. Characterizing the Law**

$$\mathbb{T}^* \approx 2.498 \cdot \mathbb{S} - 11.498 \approx 2.5 \cdot \mathbb{S}$$

- The fitting $R^2 = 0.9957$ indicates an almost perfect linear relationship.
- Corollary: The optimal student size is approximately 0.4 times the teacher size.
- This law holds across different model architectures (GPT2 vs. Pythia), data scales, pruning paradigms, and distillation objectives.

**5. Large-Scale Extrapolation—MiniMA**

Based on the law, LLaMA2-7B is distilled into the ~3B MiniMA:
- Heuristic pruning rules are used to ensure a symmetric cross-layer structure (dropping 4 bottom layers + 4 top layers).
- Distilled on a mixed dataset of Pile + GitHub + WuDao (126B tokens).
- Uses a token-level cross-entropy loss (50% teacher distribution + 50% ground-truth labels).
- Further fine-tuned on 1.1M instruction data to obtain MiniChat.

## Key Experimental Results

### Main Results

MiniMA (3B) vs. competitors of the same scale on standard benchmarks:

| Model | Params | MMLU | CEval | DROP | BBH | GSM8K | HumanEval |
|------|------|------|-------|------|-----|-------|-----------|
| LLaMA2 | 7B | 46.00 | 34.40 | 31.57 | 32.02 | 14.10 | 12.80 |
| ShortGPT | 3B | 25.57 | 26.79 | 8.72 | 7.53 | 4.52 | 0.00 |
| MiniMA | 3B | **Significantly outperforms competitors of the same scale** |

- MiniMA significantly outperforms same-scale pruning/distillation baselines across all benchmarks.
- MiniChat outperforms several size-matched competitors on instruction-following benchmarks and even rivals larger LLMs.

### Key Findings

- **Universality of the Linear Law**: The linear relationship consistently holds across two model families (GPT2 and Pythia) and different data scales.
- **Effectiveness of Extrapolation**: The law discovered at the small scale (<3B) successfully guides the 7B$\rightarrow$3B distillation.
- **Industry Validation**: The teacher-to-student size ratio of existing distilled SLMs (such as the Phi series) roughly aligns with this linear law.
- **Comparison with Pure Pruning**: Pruning methods without distillation, such as ShortGPT, perform far worse than MiniMA, confirming the necessity of distillation.

## Highlights & Insights

1. **Innovative Paradigm Shift from a Scaling Law Perspective**: Rather than trying to "break" the curse of the capacity gap, this work uncovers and leverages its underlying law, presenting an elegant motivation.
2. **Simplicity of $\mathbb{T}^* \approx 2.5 \cdot \mathbb{S}$**: Such a simple linear relationship holding stable under various settings is highly valuable for guidance.
3. **Extrapolating from Small Pilots to Large Scales**: Aligns with the classic paradigm of Scaling Law research, making highly efficient use of computational resources.
4. **High Practical Value**: Eliminates the need for exhaustive trials when distilling LLMs, allowing direct determination of the optimal teacher size using the law.

## Limitations & Future Work

- The constant (2.5x) in the law was derived under specific pruning and distillation settings; its applicability to other compression schemes (e.g., quantized distillation) remains unverified.
- Only verified up to the 7B$\rightarrow$3B scale; whether the extrapolation holds for larger scales (e.g., 70B$\rightarrow$28B) remains an open question.
- Using structured pruning for student model initialization is different from the setup of pre-training a small model from scratch and then distilling it.
- Vocabulary expansion and incremental training were performed on LLaMA2 to adapt to Chinese tasks, increasing experimental complexity.
- Distillation only employed the basic token-level cross-entropy loss, without exploring the impact of more advanced distillation objectives (e.g., sequence-level distillation).

## Related Work & Insights

- **Knowledge Distillation**: From KD by Hinton et al. (2015) to task-specific distillation like DistilBERT and TinyBERT.
- **Capacity Gap**: First discovered in vision models by Mirzadeh et al. (2020) and extended to LMs by Zhang et al. (2023b).
- **LLM Compression**: Quantization (GPTQ, AWQ), pruning (Wanda, ShortGPT), dynamic networks, etc.
- **Scaling Law**: Empirical law studies such as Kaplan et al. (2020) and k-bit quantization laws.
- **Pseudo-Distillation**: Training small models using LLM-generated data (e.g., Alpaca, Vicuna).

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (First systematic discovery and characterization of the capacity gap law)
- Technical Depth: ⭐⭐⭐⭐ (Empirical-driven with theoretical backing, highly rigorous experimental design)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Two model families $\times$ multiple data scales $\times$ large-scale extrapolation validation)
- Utility: ⭐⭐⭐⭐⭐ (Directly guides LLM distillation practice, MiniMA model open-sourced)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Distilling Tool Knowledge into Language Models via Back-Translated Traces](../../ICML2025/model_compression/distilling_tool_knowledge_into_language_models_via_back-translated_traces.md)
- [\[ACL 2025\] Pre-training Distillation for Large Language Models: A Design Space Exploration](pre-training_distillation_for_large_language_models_a_design_space_exploration.md)
- [\[ACL 2025\] Wanda++: Pruning Large Language Models via Regional Gradients](wanda_pruning_large_language_models_via_regional_gradients.md)
- [\[CVPR 2026\] Masking Teacher and Reinforcing Student for Distilling Vision-Language Models](../../CVPR2026/model_compression/masking_teacher_and_reinforcing_student_for_distilling_vision-language_models.md)
- [\[ACL 2025\] LongReD: Mitigating Short-Text Degradation of Long-Context Large Language Models via Restoration Distillation](longred_mitigating_short-text_degradation_of_long-context_large_language_models_.md)

</div>

<!-- RELATED:END -->
