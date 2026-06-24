---
title: >-
  [Paper Note] Learning Distribution-Wise Control in Representation Space for Language Models
description: >-
  [ICML 2025][LLM Evaluation][Representation Intervention] Deterministic nodes in representation fine-tuning are replaced with randomized nodes. By employing the reparameterization trick to learn latent distributions instead of single pointwise transformations, consistent performance gains are achieved across commonsense and mathematical reasoning tasks, with intervention in earlier layers exhibiting the most significant impact.
tags:
  - "ICML 2025"
  - "LLM Evaluation"
  - "Representation Intervention"
  - "Distribution-wise Control"
  - "Reparameterization"
  - "Concept Subspace"
  - "Representation Fine-tuning"
date: 2026-05-08
content_hash: 5f3aaedf5b493324
---

# Learning Distribution-Wise Control in Representation Space for Language Models

**Conference**: ICML 2025  
**arXiv**: [2506.06686](https://arxiv.org/abs/2506.06686)  
**Code**: [chili-lab/D-Intervention](https://github.com/chili-lab/D-Intervention)  
**Area**: LLM Evaluation  
**Keywords**: Representation Intervention, Distribution-wise Control, Reparameterization, Concept Subspace, Representation Fine-tuning

## TL;DR

Deterministic nodes in representation fine-tuning are replaced with randomized nodes. By employing the reparameterization trick to learn latent distributions instead of single pointwise transformations, consistent performance gains are achieved across commonsense and mathematical reasoning tasks, with intervention in earlier layers exhibiting the most significant impact.

## Background & Motivation

**Rise of Representation Intervention**: Recently, representation fine-tuning (e.g., ReFT, LoFiT) has emerged as an active research direction for high-level control, manipulating model behavior by directly modifying the hidden representations across layers of language models. Compared to parameter-efficient fine-tuning (PEFT) methods like LoRA, representation fine-tuning achieves or exceeds PEFT performance while using only 1/10 to 1/100 of the parameters.

**Key Observation**: Prior work indicates that the concept space is continuous (Gandikota et al., 2023)—adjusting the magnitude of an intervention vector once found can regulate the intervention effect. This implies that the neighborhood of an intervention point should also produce relevant effects. However, existing methods (e.g., ReFT) only learn **pointwise** transformations, leaving the distribution information around the intervention points unexplored.

**Analogy & Inspiration**: This limitation is analogous to the evolution from Autoencoders (AE) to Variational Autoencoders (VAE). By replacing deterministic nodes with stochastic sampling, VAEs learn the latent distribution directly, resulting in superior generative capabilities and a smoother latent space.

**Core Problem**: How can the concept space region surrounding the intervention vector be explored effectively?

## Method

### Overall Architecture

This paper introduces **Distribution-wise Intervention (D-Intervention)**. The core idea is simple: replace the deterministic MLP nodes in representation intervention with randomized nodes, enabling the model to learn a distribution $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\sigma}^2)$ instead of a single pointwise transformation.

**Information-Theoretic Perspective**: The authors formulate representation intervention from an information-theoretic standpoint. For an intervention function $f_\phi$ inserted at layer $l$, minimizing the cross-entropy loss is equivalent to maximizing the mutual information between the intervened representation and the target output:

$$\arg\min_\phi \mathcal{L}_{CE} \equiv \arg\max_\phi I(Y; f_\phi(Z^{(l)}))$$

This reveals the essence of intervention: transforming internal representations to make them highly informative about the target output while filtering out irrelevant information.

### Key Designs

#### 1. Randomized Intervention Reparameterization

The original deterministic MLP intervention is defined as:
$$\hat{Z} = \text{MLP}(Z) = W^T Z + b$$

This is replaced with two independent networks that learn the mean and log-variance separately:

- **Mean Network**: $\boldsymbol{\mu} = \text{MLP}_\mu(Z)$
- **Variance Network**: $\log \sigma^2 = \text{MLP}_{\log\sigma^2}(Z)$
- **Standard Deviation**: $\boldsymbol{\sigma} = \exp(\frac{1}{2}\log\sigma^2)$
- **Noise Sampling**: $\boldsymbol{\epsilon} \sim \mathcal{N}(0, I)$
- **Final Output**: $\hat{Z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$

Using the reparameterization trick, the randomness comes from the external noise $\boldsymbol{\epsilon}$, allowing gradients to propagate normally back through $\boldsymbol{\mu}$ and $\boldsymbol{\sigma}$.

**Key Differences from VAE**: In this method, the KL divergence loss term is removed, and variational inference is not performed. This allows the model to learn the distribution without constraints, rather than forcing the distribution to approximate a standard normal prior. Consequently, the model enjoys greater flexibility in exploring distributional shapes optimal for the task.

#### 2. Model-Specific Clamping Mechanism

To mitigate numerical instability caused by high sampling variance, a clamping strategy based on the weight distribution of the target language model is proposed. For an intervention at layer $l$, the clamping boundaries are defined by the extreme values of the weight matrices of adjacent layers:

$$v_{\min} = \min(\min(W^{(l)}), \min(W^{(l+1)}))$$
$$v_{\max} = \max(\max(W^{(l)}), \max(W^{(l+1)}))$$

This ensures that the intervened values remain within the natural range of the model weights. The boundaries are pre-computed once before training and remain fixed.

#### 3. Hybrid Intervention Strategy

Based on findings from layer-wise experiments, the authors propose an optimal practice: **apply randomized nodes (distribution-wise intervention) in earlier layers, and retain deterministic nodes (pointwise intervention) in subsequent layers**. This hybrid approach shows the best overall performance and robustness.

### Loss & Training

**Training Objective**: The base language model $\mathcal{M}$ is frozen, and only the randomized intervention layers $\{\mathcal{I}_l\}_{l=1}^L$ are trained by minimizing the cross-entropy loss for next-token prediction:

$$\mathcal{L} = -\mathbb{E}_{(X,Y)}[\log P_{\mathcal{M} \circ \mathcal{I}}(Y | f_\phi(Z^{(l)}))]$$

**Key Features**:

- The base model is completely frozen; only $\{\phi_\mu^{(l)}, \phi_\sigma^{(l)}\}_{l=1}^L$ are optimized.
- No KL divergence regularization is applied, allowing direct learning of the distribution via task loss.
- Minimal parameter overhead (10x-100x fewer parameters than PEFT).
- Noise is also sampled during inference to maintain distribution-wise intervention.

**Experimental Setup**:

- Models: Llama-7B, Llama-13B, Llama-3-8B
- Hardware: A single NVIDIA RTX A6000, bfloat16 mixed-precision
- Settings: Layer-wise and all-layer configurations
- Baselines: RED, ReFT, MLP, alongside deterministic and distribution-wise SwiGLU variants

## Key Experimental Results

### Main Results

**Commonsense Reasoning (8 benchmarks, all-layer setup)**:

| Method | Params | BoolQ | PIQA | SIQA | HellaSwag | WinoGrande | ARC-e | ARC-c | OBQA | Avg |
|------|--------|-------|------|------|-----------|------------|-------|-------|------|------|
| Prefix-tuning | 6.1M | 65.4 | 76.4 | 73.1 | 42.1 | 59.7 | 72.1 | 47.0 | 60.2 | 62.0 |
| LoRA | 25.2M | 68.9 | 80.7 | 79.2 | 83.6 | 80.6 | 77.8 | 61.9 | 74.8 | 75.9 |
| DoRA | 25.2M | 69.7 | 83.5 | 78.9 | 91.2 | 83.6 | 81.9 | 67.5 | 79.0 | 79.4 |
| ReFT | 0.26M | 65.2 | 78.1 | 77.2 | 64.1 | 73.6 | 75.8 | 58.1 | 73.0 | 70.6 |
| **D-ReFT** | **0.52M** | **67.8** | **80.5** | **78.3** | **72.3** | **75.9** | **78.6** | **60.5** | **75.4** | **73.7** |

> D-ReFT outperforms ReFT by **+3.1%** on average, utilizing only 0.52M parameters (~2% of LoRA).

**Mathematical Reasoning (7 benchmarks)**:

| Method | GSM8K | SVAMP | MAWPS | AQuA | SAT | MMLU-Math | SAT-M | Avg |
|------|-------|-------|-------|------|-----|-----------|-------|------|
| ReFT | 42.3 | 52.1 | 88.7 | 25.6 | 53.2 | 31.8 | 48.6 | 48.9 |
| **D-ReFT** | **45.8** | **55.3** | **90.1** | **27.1** | **55.9** | **33.5** | **51.2** | **51.3** |
| Gain | +3.5 | +3.2 | +1.4 | +1.5 | +2.7 | +1.7 | +2.6 | **+2.4** |

### Ablation Study

**Layer-wise Intervention Analysis (intervention effects at different layers)**:

| Intervention Layer Range | Pointwise Acc | Distribution-wise Acc | Gain | Learned Variance |
|-----------|-----------------|------------------------|---------|---------|
| Layers 1-4 (Early) | 61.2 | 67.1 | **+5.9%** | High |
| Layers 5-8 (Mid-Early) | 63.5 | 67.8 | +4.3% | Mid-High |
| Layers 9-16 (Mid) | 65.1 | 66.9 | +1.8% | Mid |
| Layers 17-24 (Mid-Late) | 64.8 | 65.5 | +0.7% | Low |
| Layers 25-32 (Late) | 63.9 | 64.1 | +0.2% | Very Low |

**Comparison of Hybrid Strategies (first k layers randomized + remaining pointwise)**:

| Configuration | Commonsense Avg | Math Avg | Description |
|------|-----------|-----------|------|
| All-layer Pointwise | 70.6 | 48.9 | ReFT Baseline |
| All-layer D-wise | 72.1 | 50.5 | Complete replacement |
| First 25% D-wise + Last 75% Point | **73.7** | **51.3** | Optimal Configuration |
| First 50% D-wise + Last 50% Point | 73.2 | 50.9 | Sub-optimal |
| First 75% D-wise + Last 25% Point | 72.5 | 50.6 | Close to All-layer |

### Key Findings

1. **Early layers yield maximum gains**: Distribution-wise intervention brings a performance increase of +4% to +6% in layers 1-4, while showing almost no gain in late layers. This suggests that earlier layers represent a broader concept space, making them more suitable for distribution-level exploration.

2. **Strong correlation between variance and performance**: The larger the learned standard deviation, the more pronounced the performance gain. Earlier layers naturally learn larger variances, reflecting broader neighborhood exploration.

3. **Hybrid strategy is optimal**: A hybrid approach using random intervention in early layers and deterministic intervention in subsequent layers consistently outperforms pure pointwise intervention across **all 15 benchmarks**, with notable improvements in robustness.

4. **Excluding KL regularization yields better results**: Removing the KL divergence constraint of VAE allows the model to freely learn the optimal distribution for the target task, outperforming variants that incorporate KL regularization.

## Highlights & Insights

- **Minimalist design, plug-and-play**: The method is remarkably simple—replacing a single MLP with two MLPs (one for mean, one for variance) and adding sampling noise. It can serve as a drop-in replacement for any representation intervention method.
- **Unified information-theoretic view**: Framing intervention as mutual information maximization provides a solid theoretical foundation for future research. Minimizing cross-entropy loss is mathematically equivalent to maximizing the mutual information between the intervened representations and the outputs.
- **Revealing layer-wise discrepancies**: Earlier layers possess a broader concept space suited for exploratory intervention, whereas later layers have converged onto more stable representations, making deterministic interventions sufficient. This finding sheds light on the organization of internal representations in Transformers.
- **Exceptional parameter efficiency**: Even though the parameter count doubles (0.26M to 0.52M), it remains significantly lower than LoRA (25.2M) while substantially narrowing the performance gap.

## Limitations & Future Work

1. **Increased inference overhead**: Sampling noise at inference time introduces additional computation and non-determinism. A potential solution is consolidating the mean for deterministic inference after training.
2. **Limited evaluation scale**: Experiments are confined to Llama-7B/13B and Llama-3-8B; verification on larger models (70B+) or non-Llama architectures (e.g., Mistral, Qwen) is lacking.
3. **Lack of explicit guidance for variance learning**: The variance is learned implicitly via task loss alone. Better strategies for optimizing variance (such as curriculum learning or annealing) could be explored.
4. **Restricted task scope**: The evaluation focuses strictly on reasoning tasks (commonsense and mathematics), leaving open-ended generation, alignment, and dialogue underevaluated.
5. **Under-explored theoretical analysis**: While an information-theoretic view is presented, a causal explanation of "why earlier layers learn higher variance" represents an area for future depth.

## Related Work & Insights

- **ReFT (Wu et al., 2024b)**: The primary baseline of this paper. A representation fine-tuning method based on distributed alignment search (DAS) theory, which D-Intervention directly extends.
- **LoFiT (Yin et al., 2024)**: Another representation fine-tuning approach focusing on a localization-and-editing paradigm, complementary to ReFT.
- **VAE / VIB**: The classic origin of the reparameterization trick. This paper demonstrates that a "non-variational" version removing KL constraints performs better in the context of intervention.
- **Representation Engineering (Zou et al., 2023)**: Pioneering work in representation engineering, which observed that intervention effects can be adjusted by changing the magnitude, directly inspiring the distribution-wise exploration in this study.
- **LoRA / DoRA**: Dominant PEFT methods. Though possessing significantly larger parameter footprints than representation intervention, they serve as crucial performance benchmarks.

## Rating

| Dimension | Rating | Description |
|------|------|------|
| Novelty | ⭐⭐⭐ | The concept is simple and intuitive, though adapting the AE-to-VAE analogy to representation intervention offers clear novelty. |
| Theoretical Depth | ⭐⭐⭐⭐ | Backed by a solid information-theoretic perspective, alongside insightful layer-wise analysis. |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Comprehensive evaluation across 15 benchmarks, multiple models, and detailed layer-by-layer analyses. |
| Utility | ⭐⭐⭐⭐⭐ | Highly practical, plug-and-play, open-source code, and straightforward implementation. |
| Writing Quality | ⭐⭐⭐⭐ | Logically rigorous, with seamless transitions connecting the motivation, method, and experiments. |
| **Overall** | **⭐⭐⭐⭐** | An elegant and effective approach with solid empirical backing, marking a valuable enhancement in representation intervention. |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Rethinking LLM-as-a-Judge: Representation-as-a-Judge with Small Language Models via Semantic Capacity Asymmetry](../../ICLR2026/llm_evaluation/rethinking_llm-as-a-judge_representation-as-a-judge_with_small_language_models_v.md)
- [\[ICML 2025\] Correlated Errors in Large Language Models](correlated_errors_in_large_language_models.md)
- [\[ICML 2025\] Sample Efficient Demonstration Selection for In-Context Learning](sample_efficient_demonstration_selection_for_in-context_learning.md)
- [\[ICML 2025\] Position: Theory of Mind Benchmarks are Broken for Large Language Models](position_theory_of_mind_benchmarks_are_broken_for_large_language_models.md)
- [\[AAAI 2026\] DiCaP: Distribution-Calibrated Pseudo-labeling for Semi-Supervised Multi-Label Learning](../../AAAI2026/llm_evaluation/dicap_distribution-calibrated_pseudo-labeling_for_semi-supervised_multi-label_le.md)

</div>

<!-- RELATED:END -->
