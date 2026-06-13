---
title: >-
  [Paper Note] Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models
description: >-
  [NeurIPS 2025][Interpretability][test-time training] This paper proposes a "specialization after generalization" framework that theoretically and empirically explains the effectiveness of test-time training (TTT) on in-d…
tags:
  - "NeurIPS 2025"
  - "Interpretability"
  - "test-time training"
  - "linear representation hypothesis"
  - "sparse autoencoders"
  - "foundation models"
  - "local specialization"
  - "underparameterization"
date: 2026-05-08
content_hash: ceb06a3afdf3fdb6
---

# Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2509.24510](https://arxiv.org/abs/2509.24510)  
**Authors**: Jonas Hübotter, Patrik Wolf, Alexander Shevchenko, Dennis Jüni, Andreas Krause, Gil Kur (ETH Zürich, MPI)
**Code**: Not released  
**Area**: Model Compression
**Keywords**: test-time training, linear representation hypothesis, sparse autoencoders, foundation models, local specialization, underparameterization

## TL;DR

This paper proposes a "specialization after generalization" framework that theoretically and empirically explains the effectiveness of test-time training (TTT) on in-distribution data under the Linear Representation Hypothesis (LRH). Foundation models are globally underparameterized, leading to concept superposition interference. TTT mitigates this by locally specializing the model—reallocating model capacity to the small subset of concepts relevant to the test task—thereby improving predictive performance without increasing model size.

## Background & Motivation

### State of the Field
Test-time training (TTT) refers to the practice of continuing to fine-tune a model for each prediction task at test time. In recent years, TTT has achieved remarkable performance gains in abstract reasoning, language modeling, and video generation. Traditional explanations attribute TTT's effectiveness to out-of-distribution adaptation or access to privileged data. However, as foundation models scale up, most test data is effectively in-distribution, rendering these explanations insufficient.

### Limitations of Prior Work
- **Out-of-distribution explanations are outdated**: As training data and model scale grow, test data is largely in-distribution, and TTT gains can no longer be attributed simply to distribution shift.
- **Lack of mechanistic understanding**: Despite the proliferation of TTT methods (self-supervised, few-shot, RL-based scaling, etc.), there is no theoretical explanation for why TTT benefits in-distribution data.
- **Insufficient nonparametric perspective**: Basu et al. (2023) analyze TTT from a nonparametric estimation perspective, relying on smoothness of the target function in feature space, but cannot explain why TTT substantially outperforms nonparametric methods in locally high-dimensional ($s$-sparse) concept spaces.

### Core Motivation
Although foundation models contain vast numbers of parameters, scaling laws consistently show that enlarging models continues to improve performance, indicating that current models remain in a state of "effective underparameterization." Under underparameterization, a model cannot simultaneously approximate the true function accurately across the entire data distribution. TTT provides a mechanism to specialize the model to the local neighborhood of a test point—by temporarily "forgetting" irrelevant pretrained knowledge and "freeing up" capacity to learn the concepts relevant to the current task at higher resolution.

## Method

### Linear Representation Hypothesis (LRH) Framework
The framework assumes the existence of an $s$-sparse concept space $\Phi: \mathcal{X} \to \mathbb{R}^{d_1}$, approximated by a learned feature map $\Psi: \mathcal{X} \to \mathbb{R}^{d_2}$ (with $d_2 \ll d_1$). The true function is linear in concept space: $f^\star(x) = \langle \Phi(x), w_\star \rangle$. Because the number of concepts far exceeds the model's dimensionality, concepts are encoded in dense activations via superposition.

### Three Key Observations (Validated via SAE Experiments)

**O1: Feature Space Preserves Local Geometry of Concept Space**
Neighborhoods are constructed in the CLIP embedding space ($\Psi$), the SAE reconstruction space ($\hat{\Psi}$), and the sparse concept space ($\hat{\Phi}$). The distributions of cosine similarities in concept space are nearly identical across all three, demonstrating that the SAE projection preserves local geometric structure.

**O2: Neighborhoods Are Supported by Only a Few Concepts**
When training TTT classifiers in concept space, adaptive concept selection is achieved via learnable binary masks (optimized with straight-through estimation). With a sparsity penalty of $\lambda=0.2$, the masks activate on average only ~40 concepts, far fewer than the ~180 total active concepts in the neighborhood, while achieving performance comparable to TTT using all concepts.

**O3: TTT in Feature Space Implicitly Finds Sparse Solutions**
TTT models trained on the dense reconstruction space $\hat{\Psi}$ and the sparse concept space $\hat{\Phi}_m$ achieve nearly identical accuracy, with predictions agreeing on approximately 89% of samples. The top-10 predicted probability distributions are highly consistent, demonstrating that TTT in feature space implicitly favors sparse solutions in concept space.

### SAE Experimental Setup
- ImageNet-1K dataset; 512-dimensional CLIP ViT-B/32 embeddings
- Top-$k$ SAE trained with sparse dimension $d_1=4096$ ($8\times d_2$) and sparsity $s=16$
- Only 4% inactive concepts (ghost gradient auxiliary loss)
- TTT neighborhood size $k=50$, nearest neighbors selected via L2 distance

### Theoretical Analysis

**Upper Bound on TTT Test Error** (based on sparse recovery theory):

$$\left(f(x^\star) - \langle \Psi(x^\star), \hat{v}_{x^\star}^{\text{TTT}} \rangle \right)^2 \leq O\left(\frac{\sigma^2 s \log(d_1/s)}{k}\right)$$

This achieves the minimax-optimal rate for sparse recovery.

**Lower Bound on Global Model Error**: When the feature map $\Psi$ is a random projection of $\Phi$, the expected error of the global model is $\mathbb{E}_\Psi[(f(x) - \langle \Psi(x), \hat{v}^{\text{global}} \rangle)^2] = 1 - d_2/d_1$. When $d_1 \gg d_2$, this error approaches 1, meaning the global model cannot effectively disentangle all superposed concepts under underparameterization.

**Key Conclusion**: Under underparameterization ($d_2 \sim \log d_1$), TTT can efficiently recover the meaning of locally relevant concepts from an exponentially large concept space—something global training cannot achieve.

## Key Experimental Results

### Experiment 1: Model Scaling

Performance comparison of global training, TTT, and majority voting across three tasks as model scale varies:

| Task | Model | Global Training | TTT | Majority Voting | TTT Gain |
|------|-------|----------------|-----|----------------|----------|
| MNIST | LeNet variant (small) | ~3.5% err | ~1.5% err | ~5.0% err | ~2.0%↓ |
| MNIST | LeNet variant (large) | ~1.5% err | ~1.0% err | ~5.0% err | ~0.5%↓ |
| ImageNet | MLP-128d | ~26% err | ~23% err | ~90% err | ~3%↓ |
| ImageNet | MLP-4096d | ~21.5% err | ~20.5% err | ~90% err | ~1%↓ |
| Pile LM | Qwen2.5-0.5B | ~1.15 bpb | ~1.07 bpb | — | ~0.08↓ |
| Pile LM | Qwen2.5-7B | ~0.90 bpb | ~0.87 bpb | — | ~0.03↓ |
| Pile LM | Qwen2.5-32B | ~0.83 bpb | ~0.82 bpb | — | ~0.01↓ |

**Key Finding**: TTT consistently outperforms global training across all tasks, but the performance gap narrows as model scale increases, consistent with the theoretical prediction that TTT yields the greatest gains under underparameterization.

### Experiment 2: Verifying the Locality of TTT

| Evaluation Setting | MNIST Accuracy | ImageNet Accuracy |
|-------------------|---------------|------------------|
| Global training | 98.57 ± 0.12 | 78.33 ± 0.19 |
| TTT (test sample) | 99.01 ± 0.10 | 79.39 ± 0.18 |
| TTT (within neighborhood) | 100.00 ± 0.00 | 95.19 ± 0.00 |
| TTT head evaluated globally | 36.38 ± 0.16 | 77.04 ± 0.06 |

**Key Finding**: TTT achieves near-perfect accuracy within the neighborhood, but fixing the TTT head and evaluating globally leads to dramatic degradation (MNIST drops from 99% to 36%), confirming that TTT gains stem from local specialization rather than global improvement.

### Experiment 3: TTT in SAE Concept Space

| Feature Space | Global Accuracy | TTT Accuracy |
|--------------|----------------|-------------|
| $\hat{\Phi}(x)$ (sparse concepts) | 71.45 ± 0.21 | 72.64 ± 0.20 |
| $\hat{\Psi}(x)$ (dense reconstruction) | 71.26 ± 0.20 | 72.56 ± 0.19 |

TTT performance is nearly identical across both spaces, with predictions agreeing on 89% of samples, validating O3.

## Highlights & Insights

- **Novel explanatory framework**: This work is the first to provide a unified explanation of TTT effectiveness through the lens of "global underparameterization → concept superposition interference → local specialization releases capacity," without relying on distribution shift assumptions.
- **Closed theoretical-empirical loop**: Three key observations (geometry preservation, local sparsity, implicit sparse bias) are empirically validated on ImageNet via SAE, and sparse recovery theory is then used to derive minimax-optimal error bounds for TTT.
- **Cross-modal consistency**: Scaling experiments on MNIST, ImageNet (vision), and Pile (language modeling) systematically validate the core prediction that smaller models benefit more from TTT.
- **Unification across research areas**: The framework connects interpretability (SAE/LRH), compressed sensing (sparse recovery), continual learning (catastrophic forgetting), and TTT within a single theoretical lens.
- **Practical guidance**: The work explicitly identifies the underparameterized regime as where TTT is most effective, offering concrete compute-budget trade-off guidance for deployment.

## Limitations & Future Work

- **Only the final layer is updated**: Experiments restrict TTT to fine-tuning the last linear layer or LoRA parameters (~1% of parameters); the behavior and theory of end-to-end TTT remain unexplored.
- **Heuristic neighborhood size selection**: How the optimal neighborhood size $k$ varies with test point characteristics and task complexity is not systematically analyzed.
- **Simplified theoretical model**: The theoretical analysis is built on a univariate regression setting, which departs from the practical scenarios of multi-class classification and language modeling.
- **SAE approximation of concept space**: The learned $\hat{\Phi}$ is only an approximation of the true concept space $\Phi$; 4% dead features and a 6% accuracy drop indicate non-trivial information loss.
- **Computational overhead not quantified**: TTT requires neighborhood search and fine-tuning for each test sample, substantially increasing inference cost; the paper does not analyze the compute-performance trade-off in detail.
- **Insufficient validation on large models**: Language modeling experiments are limited to Qwen2.5-32B (constrained by 4090 GPUs), leaving open whether TTT remains beneficial at 100B+ scale.

## Related Work & Insights

- **Sun et al. (2020)**: Pioneering TTT work, assuming alignment between TTT gradients and oracle label gradients; this paper provides theoretical support for such alignment under the LRH model.
- **Hardt & Sun (2024)**: Propose semi-parametric TTT and validate it on Pile; this paper uses their open-source implementation and provides a theoretical explanation for why the method works.
- **Basu et al. (2023)**: Analyze retrieval-augmented models from a nonparametric estimation perspective, relying on feature space smoothness; this paper explicitly models the sparse concept space, explaining why TTT substantially outperforms nonparametric methods (majority voting).
- **Akyürek et al. (2025)**: Demonstrate TTT effectiveness in few-shot abstract reasoning; this paper provides a more general mechanistic explanation.
- **Elhage et al. (2022)**: Introduce a toy model of concept superposition; this paper connects it to local specialization in TTT.
- **Gao et al. (2025)**: Propose Top-$k$ SAE for interpretability; this paper employs it as a tool for validating the LRH.
- **Lim et al. (2025), Doimo et al. (2024)**: Recent empirical work showing TTT learns local meanings of concepts rather than discovering new ones, consistent with the theoretical predictions of this paper.
- **Bertolissi et al. (2025)**: Propose Local MoE, achieving TTT-free local specialization via model merging; complementary to the local specialization perspective of this work.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First systematic explanation of TTT effectiveness from the underparameterization + concept superposition perspective
- Experimental Thoroughness: ⭐⭐⭐⭐ — Scaling experiments on three tasks plus SAE validation of three hypotheses, though large-scale model experiments are limited
- Writing Quality: ⭐⭐⭐⭐⭐ — Elegantly structured, with tight integration of theory, experiments, and intuition; figures and tables are clear
- Value: ⭐⭐⭐⭐ — Provides important theoretical foundations and practical guidance for the TTT community, though end-to-end TTT and very large models remain uncovered

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Benchmarking Probabilistic Time Series Forecasting Models on Neural Activity](benchmarking_probabilistic_time_series_forecasting_models_on_neural_activity.md)
- [\[NeurIPS 2025\] Transformer Key-Value Memories Are Nearly as Interpretable as Sparse Autoencoders](transformer_key-value_memories_are_nearly_as_interpretable_as_sparse_autoencoder.md)
- [\[NeurIPS 2025\] URLs Help, Topics Guide: Understanding Metadata Utility in LLM Training](urls_help_topics_guide_understanding_metadata_utility_in_llm_training.md)
- [\[NeurIPS 2025\] An Analysis of Concept Bottleneck Models: Measuring, Understanding, and Mitigating the Impact of Noisy Annotations](an_analysis_of_concept_bottleneck_models_measuring_understanding_and_mitigating_.md)
- [\[NeurIPS 2025\] Representation Consistency for Accurate and Coherent LLM Answer Aggregation](representation_consistency_for_accurate_and_coherent_llm_answer_aggregation.md)

</div>

<!-- RELATED:END -->
