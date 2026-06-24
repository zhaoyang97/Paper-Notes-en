---
title: >-
  [Paper Note] LESA: Learnable LLM Layer Scaling-Up
description: >-
  [ACL 2025][LLM (Other)][Model Scaling-Up] Proposed LESA, a learnable depth scaling-up method that discovers latent inter-layer patterns via SVD and predicts intermediate layer parameters using a neural network. Compared to heuristic layer replication methods, LESA achieves better initialization and faster convergence, reducing training costs by more than half.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Model Scaling-Up"
  - "Depth Scaling-Up"
  - "SVD"
  - "Inter-layer Patterns"
  - "Continual Pre-training"
date: 2026-05-08
content_hash: b2a88b5d41c82e0c
---

# LESA: Learnable LLM Layer Scaling-Up

**Conference**: ACL 2025  
**arXiv**: [2502.13794](https://arxiv.org/abs/2502.13794)  
**Code**: [github.com/yangyifei729/LESA](https://github.com/yangyifei729/LESA)  
**Area**: LLM/NLP  
**Keywords**: Model Scaling-Up, Depth Scaling-Up, SVD, Inter-layer Patterns, Continual Pre-training

## TL;DR

Proposed LESA, a learnable depth scaling-up method that discovers latent inter-layer patterns via SVD and predicts intermediate layer parameters using a neural network. Compared to heuristic layer replication methods, LESA achieves better initialization and faster convergence, reducing training costs by more than half.

## Background & Motivation

Training Large Language Models (LLMs) from scratch requires tremendous computational resources. Model Scaling-Up constructs large models by historical/reused parameters of smaller models, offering a viable solution to lower costs. Existing depth scaling-up approaches mainly fall into two categories:

**Interpolation**: Inserting duplicates of layers immediately after those layers, such as LLaMA Pro.
**Stacking (Stack)**: Copying contiguous groups of layers, such as SOLAR.

These methods heavily rely on heuristic rules for layer replication, facing the following limitations:
- Ignoring weight evolutionary patterns across layers, simply copying previous layers for the newly expanded layers.
- Failing to achieve layer specialization, leading to poor initialization performance.
- Demonstrating slow convergence during continual pre-training and failing to fully leverage the expanded capacity.

The authors observe for the first time that by concatenating the parameters of each Transformer layer and performing SVD decomposition, the inter-layer parameters exhibit latent patterns like continuity in the SVD space. This inspires the idea of using a neural network to learn these patterns.

## Method

### Overall Architecture

The core workflow of LESA consists of three steps:
1. **SVD Decomposition**: Extract layer weight matrices (q_proj, k_proj, v_proj, o_proj, up_proj, down_proj, gate_proj), horizontally concatenate them, and perform SVD decomposition to project each layer's parameters into a unified orthogonal basis space.
2. **Training Prediction Network**: Train an MLP network $\mathcal{G_W}$ in the SVD space. Taking the SVD coefficients of adjacent layers as input, it predicts the SVD coefficients of the intermediate layers.
3. **Inserting Predicted Layers**: Use the trained $\mathcal{G_W}$ to predict the parameters of the intermediate layers between adjacent layers, reconstruct them back to the original parameter space via $U\Sigma$, and insert them into the model to complete the depth scaling-up.

### Key Designs

**SVD Inter-layer Pattern Discovery**: Horizontally concatenate a certain type of weight matrix $\mathcal{W}_1, ..., \mathcal{W}_L$ of an $L$-layer model into $\mathcal{W} \in \mathbb{R}^{d_1 \times Ld_2}$, and perform SVD decomposition to obtain $U, \Sigma, V^T$. Each layer $\mathcal{W}_i$ can be represented as $U\Sigma V_i$, where $V_i$ is the coefficient of this layer on the orthogonal basis $U\Sigma$. Visualizing the top-1 eigenvectors of $V_i$ using t-SNE reveals distinct continuity patterns across layers.

**Prediction Network Design**: $\mathcal{G_W}$ is a three-layer MLP (hidden dimension 256, ReLU activation), trained separately for each type of weight matrix. The input is $[V_{i-1}, V_{i+1}]$ (the SVD coefficients of two layers separated by one layer), and the target is to predict $V_i$. The training samples are constructed from combinations of three consecutive layers. For example, a 32-layer model can generate 30 samples.

### Loss & Training

The total loss consists of two parts:

$$\mathcal{L} = (1-\lambda)\mathcal{L}_1 + \lambda \mathcal{L}_2$$

- **MSE Loss** $\mathcal{L}_1$: The mean squared error between the predicted value and the ground truth $V_i$.
- **Norm Loss** $\mathcal{L}_2$: The mean squared error between the L2 norm of the predicted value and the L2 norm of the ground truth $V_i$.

The Norm loss is incorporated to prevent the predicted parameter norms from degenerating to zero (parameter degradation issue) when training solely with $\mathcal{L}_1$. The hyperparameter $\lambda$ is set to 5e-5.

Training $\mathcal{G_W}$ takes only 5 epochs and less than 5 minutes, with negligible computational overhead.

## Key Experimental Results

### Main Results

Scaling Llama3-8B from 32 layers to 48 layers (11.5B parameters) and conducting continual pre-training on Wikipedia 2024.11 data:

| Metric | LESA-3k | LESA-6k | LLaMA Pro-6k | SOLAR-6k |
|------|---------|---------|--------------|----------|
| PPL | 5.27 | **5.13** | 5.44 | 8.09 |
| Average Score | **64.11** | **64.30** | 62.67 | 47.86 |
| Training Time | - | **45.6h** | 56.4h(124%) | 75.6h(166%) |

Key Findings:
- LESA-3k (using only half the data) already outperforms the 6k-step results of both baselines trained on full data.
- LESA starts with the lowest initial loss and stabilizes around 2k steps, whereas LLaMA Pro takes 5k steps to reach the same level.
- SOLAR even fails to converge to a lower loss level.

**SFT Performance**: LESA-SFT achieves an average score of 31.57 (100%), while LLaMA Pro-SFT only reaches 77% and SOLAR-SFT only 84%.

### Ablation Study

1. **Generalization across Model Families**: Scaling 1.5x on Llama3-8B/70B, Qwen2.5-1.5B/7B/32B, and Mistral-Small-24B, LESA consistently outperforms SOLAR in terms of PPL. Notably, SOLAR experiences PPL explosion (INF) on Qwen2.5-32B, while LESA remains stable.
2. **Effect of SVD**: Removing SVD degrades performance but still outperforms LLaMA Pro, demonstrating that SVD decomposition is beneficial but not strictly necessary.
3. **Effect of Freezing Layers**: Not freezing original parameters leads to slower and fluctuating loss convergence, validating the importance of the freezing strategy.
4. **Code Domain**: HumanEval scores after pre-training on BigCode: LESA 25.00 vs Pro 10.98 vs SOLAR 2.44.

### Key Findings

- The test loss of $\mathcal{G_W}$ is on par with the training loss, indicating that it successfully learns latent inter-layer patterns rather than overfitting.
- When training across multiple Llama3 variants (including fine-tuned versions), 150 samples (120 training/30 test) are sufficient for effective learning.
- The training cost of the prediction network is extremely low (<5 minutes), which yields negligible overhead compared to continual pre-training.

## Highlights & Insights

1. **First Discovery of Inter-layer Patterns**: Revealing the continuity of Transformer layer parameters in a low-dimensional space via SVD decomposition is a valuable theoretical finding.
2. **Paradigm Shift from Replication to Prediction**: Instead of simply copying layers, LESA is a paradigm shift that predicts new layer parameters by learning inter-layer relationships, yielding superior initialization.
3. **Extremely Low Extra Overhead**: Training the prediction network requires only 5 minutes but saves more than half of the continual pre-training cost.
4. **Broad Applicability**: Highly effective across varying model families and sizes, as well as domain-specific (code) pre-training.

## Limitations & Future Work

- Has not yet explored large-scale scaling exceeding 3x parameters. In practice, extensively increasing layer quantity typically requires coordination with width-wise scaling.
- Study on MoE models is still in the preliminary stage (determining router weights for predicted layers is difficult).
- Inter-layer continuity patterns are currently visualized mainly on gate_proj via t-SNE; other parameter matrices may reveal patterns using more advanced analytical methods.
- Co-scaling with width expansion methods is unexplored.

## Related Work & Insights

- **LLaMA Pro** (Wu et al., 2024) and **SOLAR** (Kim et al., 2023) are the two major depth scaling-up baselines.
- **Net2Net** (Chen et al., 2015) and **LiGO** (Wang et al., 2023) represent width scaling-up methods.
- SVD has been utilized in model compression and model merging; applying it to analyze inter-layer relationships in this paper represents an innovative perspective.
- Food for thought: Can similar inter-layer pattern discovery methods be applied to model pruning (as an inverse operation)?

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Inspiring approach for SVD-based pattern discovery combined with learnable scaling.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Highly comprehensive ablation studies, validated across different models and domains.
- **Value**: ⭐⭐⭐⭐⭐ — Extremely low cost with remarkable performance improvements, ready for practical LLM scaling-up training.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-structured, logical, and accompanied by detailed diagrams.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Genetic Instruct: Scaling up Synthetic Generation of Coding Instructions for Large Language Models](genetic_instruct_scaling_up_synthetic_generation_of_coding_instructions_for_larg.md)
- [\[NeurIPS 2025\] Scaling Up Active Testing to Large Language Models](../../NeurIPS2025/llm_nlp/scaling_up_active_testing_to_large_language_models.md)
- [\[ACL 2025\] TrimLLM: Progressive Layer Dropping for Domain-Specific LLMs](trimllm_layer_dropping.md)
- [\[ACL 2025\] Reversal of Thought: Enhancing Large Language Models with Preference-Guided Reverse Reasoning Warm-up](reversal_of_thought_enhancing_large_language.md)
- [\[ACL 2025\] Growing Through Experience: Scaling Episodic Grounding in Language Models](episodic_grounding_experience.md)

</div>

<!-- RELATED:END -->
