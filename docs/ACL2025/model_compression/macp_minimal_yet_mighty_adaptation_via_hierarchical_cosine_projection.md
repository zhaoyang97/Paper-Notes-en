---
title: >-
  [Paper Note] MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection
description: >-
  [ACL 2025][Model Compression][Parameter-Efficient Fine-Tuning] This paper proposes MaCP—a parameter-efficient fine-tuning (PEFT) method based on the Discrete Cosine Transform (DCT). By projecting weight updates into the cosine frequency domain and hierarchically selecting the most critical frequency components, MaCP achieves performance superior to or comparable to existing PEFT methods with an extremely low parameter count (99.7% fewer parameters than LoRA).
tags:
  - "ACL 2025"
  - "Model Compression"
  - "Parameter-Efficient Fine-Tuning"
  - "Discrete Cosine Transform"
  - "Frequency Domain Learning"
  - "LoRA"
  - "Model Adaptation"
date: 2026-05-08
content_hash: ac741d26d7cbeb21
---

# MaCP: Minimal yet Mighty Adaptation via Hierarchical Cosine Projection

**Conference**: ACL 2025  
**arXiv**: [2410.09103](https://arxiv.org/abs/2410.09103)  
**Code**: None  
**Area**: Others  
**Keywords**: Parameter-Efficient Fine-Tuning, Discrete Cosine Transform, Frequency Domain Learning, LoRA, Model Adaptation

## TL;DR

This paper proposes MaCP—a parameter-efficient fine-tuning (PEFT) method based on the Discrete Cosine Transform (DCT). By projecting weight updates into the cosine frequency domain and hierarchically selecting the most critical frequency components, MaCP achieves performance superior to or comparable to existing PEFT methods with an extremely low parameter count (99.7% fewer parameters than LoRA).

## Background & Motivation

Although large language models possess strong general capabilities, their zero-shot performance on downstream tasks is often suboptimal, requiring fine-tuning. Full-parameter fine-tuning is computationally prohibitive (e.g., LLaMA 3.1-70B requires approximately 500GB VRAM), making parameter-efficient fine-tuning (PEFT) methods the mainstream approach.

LoRA is the most representative PEFT method, reducing trainable parameters through low-rank decomposition. However, LoRA and its variants face a core challenge: the reduction in parameter size does not directly translate into a reduction in memory or computational cost. LoRA expands the effective embedding dimension, increases FLOPs, and requires storing high-dimensional activations and optimizer states.

Recent studies (such as FourierFT) have explored applying frequency domain techniques to fine-tuning, using the Discrete Fourier Transform (DFT) to compress trainable parameters. However, DFT is inherently suited for periodic signals, whereas long-range dependencies in language are typically non-periodic. In addition, DFT operates in the complex domain, introducing computational overhead and numerical instability.

The core insight of this paper is that the Discrete Cosine Transform (DCT) is better suited for language data than DFT—DCT offers superior energy compaction and decorrelation properties for non-periodic signals, and operates entirely in the real domain, avoiding the overhead of complex operations.

## Method

### Overall Architecture

The workflow of MaCP is as follows: (1) project the pre-trained weight matrices into the frequency domain using DCT; (2) partition the spectrum hierarchically into three regions: low, medium, and high frequencies; (3) select the most critical frequency components from each region as trainable parameters; (4) transform back to the spatial domain via iDCT (inverse transform) after training to update the weights.

### Key Designs

1. **Cosine Projection (DCT Transform)**: Given a weight matrix $W[i,j]$ (of size $M \times N$), it is transformed to the frequency domain $W_F[u,v]$ via 2D DCT. Low-frequency components (where $u, v$ are small) contain the most significant information and are the primary targets for fine-tuning. The key advantage of DCT is that energy is concentrated in a few low-frequency components, and the entire process uses real-number operations without needing to handle complex numbers.

2. **Hierarchical Spectrum Partitioning**: Based on the Euclidean distance $d(u,v) = \sqrt{u^2 + v^2}$ from the frequency coordinates $(u,v)$ to the origin, the spectrum is divided into three regions:

    - **Low frequency** $\mathcal{M}_{\text{low}}$: $d \leq d_{\max}/3$, capturing global patterns and containing most of the energy.
    - **Medium frequency** $\mathcal{M}_{\text{mid}}$: $d_{\max}/3 < d \leq 2d_{\max}/3$, capturing medium-scale structures.
    - **High frequency** $\mathcal{M}_{\text{high}}$: $d > 2d_{\max}/3$, capturing detailed features.

3. **Hybrid Selection Strategy**: Within each partition, a hybrid strategy of energy-priority combined with random exploration is adopted. Specifically, the top $n_{\mathcal{M}} \times \delta$ (default $\delta=0.7$) components are selected based on energy values, and the remainder are selected randomly. This hierarchical sampling balances high-energy components and diversity across all partitions.

4. **Back-propagation to Spatial Domain via iDCT**: Only the selected frequency components $\Delta W_F$ are updated. Then, the spatial domain weight update $\Delta W_T = \text{iDCT}(\Delta W_F) \times \alpha$ is obtained via iDCT and merged back into the original weights.

### Loss & Training

MaCP uses standard loss functions corresponding to downstream tasks (such as cross-entropy). The core innovation lies in the parameterization method rather than the loss function. During training, only $n$ frequency components require gradient updates.

**Memory Efficiency Analysis**: The activation memory of MaCP is $B \cdot S \cdot H + B \cdot n$, whereas LoRA requires $2 \times B \cdot S \cdot H$. When $n \ll S \cdot H$ (e.g., $n=1000$ vs. $S \cdot H = 2048 \times 4096$), MaCP achieves over 50% activation memory savings. Furthermore, optimizer states and gradient storage are significantly reduced.

## Key Experimental Results

### Main Results

Natural Language Understanding (RoBERTa-Large, GLUE):

| Method | Trainable Params | SST-2 | MRPC | CoLA | QNLI | RTE | STS-B | Average |
|------|----------|-------|------|------|------|-----|-------|------|
| Full FT | 356M | 96.3 | 90.9 | 68.0 | 94.7 | 86.6 | 92.4 | 88.11 |
| LoRA | 0.8M | 96.2 | 90.2 | 68.2 | 94.8 | 85.2 | 92.3 | 87.82 |
| FourierFT | 0.048M | 96.0 | 90.9 | 67.1 | 94.4 | 87.4 | 91.9 | 87.95 |
| MaCP | **0.034M** | 96.2 | 90.9 | 67.7 | 94.5 | **87.4** | 92.0 | **88.12** |

Instruction Tuning (LLaMA2-13B):

| Method | Trainable Params | MT-Bench | Vicuna |
|------|----------|----------|--------|
| LoRA | 250.3M | 5.77 | 7.38 |
| DoRA | 264.5M | 5.79 | 7.47 |
| FourierFT | 0.08M | 5.82 | 7.49 |
| MaCP | **0.056M** | **5.93** | **7.55** |

Text Summarization (BART-Large):

| Method | Params | XSUM (R-1/R-2/R-L) | CNN/DM (R-1/R-2/R-L) |
|------|-------|-------------------|---------------------|
| Full FT | 415M | 45.14/22.27/37.25 | 44.16/21.28/40.90 |
| LoRA | 8.6M | 43.95/20.72/35.68 | 45.03/21.84/42.15 |
| MaCP | **0.17M** | **45.21/22.19/37.10** | **45.09/21.97/42.29** |

### Ablation Study

Spectrum Partitioning Strategy (Joint RoBERTa-Base + ViT-B):

| Configuration | MRPC | CoLA | CIFAR100 | EuroSAT | Description |
|------|------|------|----------|---------|------|
| Low-frequency only | 90.1 | 63.6 | 91.6 | 98.9 | Missing fine-grained details |
| Low + High frequency | 89.4 | 64.1 | 91.7 | 98.9 | Ignoring mid-frequency details |
| MaCP (Three partitions) | **89.7** | **64.6** | **91.7** | **99.1** | Optimal balance |
| Four partitions | 88.9 | 62.9 | 91.1 | 98.7 | Over-partitioning degrades performance |

Expressive Power Comparison (Synthetic Classification Task, Equal Parameter Budget):

| Method | Convergence Speed | Final Accuracy | Stability |
|------|---------|----------|--------|
| LoRA (r=1) | Slow | ~75% (fails to converge to 100%) | Heavy oscillation |
| FourierFT (n=128) | ~500 epochs | ~100% | Relatively stable |
| MaCP (n=90) | **~450 epochs** | **100%** | **Most stable** |

### Key Findings

1. **Extreme Parameter Efficiency**: MaCP requires only 0.045M parameters on LLaMA2-7B (0.03% of LoRA). It not only achieves SOTA performance on NLU and NLG tasks but also outperforms full-parameter fine-tuning on summarization tasks.
2. **DCT Outperforms DFT**: Under the same parameter budget, MaCP consistently outperforms FourierFT. The real-valued non-periodic decomposition of DCT aligns better with the structure of language data.
3. **Cross-Modal Generalization**: MaCP is not only applicable to NLP tasks but also highly effective for image classification (ViT) and video understanding (VL-BART).
4. **Three-Partitioning is the Optimal Strategy**: Ablation studies show that three-partitioning (low/mid/high frequency) yields the best results. Too few (low-only) or too many (four-partitioning) partitions degrade performance.
5. **Significant Reduction in Memory Usage**: Compared to LoRA, MaCP significantly reduces GPU memory usage on LLaMA3.1-8B.

## Highlights & Insights

- The advantages of DCT over DFT are clearly and compellingly discussed: non-periodicity, real-domain operations, and superior energy compaction. This establishes a better theoretical foundation for frequency-domain PEFT methods.
- The hierarchical partitioning + hybrid selection strategy balances energy concentration and diversity, proving more robust than simple top-k selection.
- The expressive power comparison experiments on synthetic tasks are highly intuitive, clearly demonstrating the advantages of frequency-domain methods over LoRA under tight parameter constraints.
- Comprehensive evaluations across six task types spanning NLU, NLG, summarization, instruction tuning, vision, and video demonstrate the generalizability of the proposed method.

## Limitations & Future Work

- The extra inference overhead introduced by the DCT and iDCT transforms themselves is not thoroughly discussed. Although trainable parameters are reduced, the computational cost of frequency transformations might offset some of the gains.
- The number of frequency components $n$ and the energy ratio $\delta$ are hyperparameters that need tuning, and their optimal values may vary across different tasks.
- MaCP is mainly compared with baselines like standard LoRA, but comparisons with newer PEFT methods (such as GaLore, LISA, etc.) are lacking.
- The performance of MaCP when deeply integrated with quantization techniques (such as QLoRA) remains unexplored.

## Related Work & Insights

- **LoRA** (Hu et al., 2022): The foundational work for low-rank adaptation. MaCP offers a brand-new parameterization perspective in the frequency domain.
- **FourierFT** (Gao et al., 2024): The first to introduce frequency-domain methods to PEFT, but limited by the complex domain and periodicity assumptions of DFT. MaCP overcomes these limitations using DCT.
- **LaMDA** (Azizi et al., 2024): A low-dimensional adaptation method that contributes to reducing gradient and activation memory, but falls short in parameter efficiency compared to MaCP.
- **VeRA** (Kopiczko et al., 2023): Reduces parameters via shared random matrices and scaling vectors, representing another extremely parameter-efficient approach.

## Rating

- Novelty: ⭐⭐⭐⭐ Replacing DFT with DCT for PEFT is a natural yet highly effective improvement, and the hierarchical partitioning strategy is elegantly designed.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The experiments cover extremely comprehensive evaluations across six task categories, various model sizes, and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clearly described with complete algorithm pseudocode, though some mathematical notations could be further streamlined.
- Value: ⭐⭐⭐⭐ This is of practical importance for model fine-tuning in resource-constrained scenarios, though its validation on industrial-scale large models remains to be seen.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] CoLA: Collaborative Low-Rank Adaptation](cola_collaborative_low-rank_adaptation.md)
- [\[ACL 2025\] Low-Rank Interconnected Adaptation across Layers](low-rank_interconnected_adaptation_across_layers.md)
- [\[ACL 2025\] BeamLoRA: Beam-Constraint Low-Rank Adaptation](beamlora_beam_constraint_lora.md)
- [\[ACL 2025\] TeamLoRA: Boosting Low-Rank Adaptation with Expert Collaboration and Competition](teamlora_boosting_low-rank_adaptation_with_expert_collaboration_and_competition.md)
- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](../../NeurIPS2025/model_compression/why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)

</div>

<!-- RELATED:END -->
