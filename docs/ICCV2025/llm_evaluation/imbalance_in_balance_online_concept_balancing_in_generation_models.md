---
title: >-
  [Paper Note] Imbalance in Balance: Online Concept Balancing in Generation Models
description: >-
  [ICCV 2025][LLM Evaluation][Concept Composition] Through carefully designed causal experiments, this work reveals that data distribution—rather than model scale or data volume—is the decisive factor for concept compositi…
tags:
  - "ICCV 2025"
  - "LLM Evaluation"
  - "Concept Composition"
  - "Data Imbalance"
  - "IMBA Loss"
  - "Diffusion Model Training"
  - "Long-Tail Distribution"
date: 2026-05-08
content_hash: 96d3cdad7029665f
---

# Imbalance in Balance: Online Concept Balancing in Generation Models

**Conference**: ICCV 2025
**arXiv**: [2507.13345](https://arxiv.org/abs/2507.13345)
**Code**: [https://github.com/KwaiVGI/IMBA-Loss](https://github.com/KwaiVGI/IMBA-Loss)
**Area**: LLM Evaluation
**Keywords**: Concept Composition, Data Imbalance, IMBA Loss, Diffusion Model Training, Long-Tail Distribution

## TL;DR

Through carefully designed causal experiments, this work reveals that data distribution—rather than model scale or data volume—is the decisive factor for concept composition ability in diffusion models. It further proposes IMBA Loss, an online concept-level balancing loss that adaptively reweights token-level losses via the discrepancy between conditional and unconditional distributions (the IMBA distance). With only a few lines of code modification, the method significantly improves multi-concept generation capability.

## Background & Motivation

State-of-the-art T2I models (DALL-E 3, SD3, Midjourney, Flux) still suffer from severe issues when handling multi-concept composition: **concept neglect** (a concept is absent from the generated image), **attribute leakage** (attributes are incorrectly bound to the wrong object), and **concept entanglement** (spurious concepts appear in the output).

Existing work mainly addresses these issues from the inference side (training-free), enhancing concept responses by optimizing attention maps, but is fundamentally constrained by the capacity of the base model. The more fundamental question is: **what factors determine a model's concept composition ability?** Prior studies mostly relied on simple synthetic data (basic shapes and colors), which is far removed from real T2I tasks.

The authors propose and empirically verify three hypotheses: (1) a sufficiently large dataset naturally covers all concepts; (2) a sufficiently large model can learn concept composition well; (3) data distribution is the key factor. The conclusion is that **only the third hypothesis holds**, and despite their apparent scale, existing datasets exhibit severely long-tailed distributions that lead to insufficient learning of tail concepts.

## Method

### Overall Architecture

The core mechanism of IMBA Loss is to multiply the standard diffusion training loss by an adaptive concept-level weight, so that tail concepts (low-frequency concepts) receive higher loss weights. This weight is estimated online via the **IMBA distance**—the discrepancy between the conditional and unconditional distributions—requiring no offline statistics of data distributions and computed entirely online.

### Key Designs

1. **IMBA Distance Definition and Data Distribution Approximation**: Theoretical derivation shows that the discrepancy between the conditional and unconditional distributions for concept $c_j$ satisfies $D_j = \|\epsilon_\theta(a_t, c_j, t) - \epsilon_\theta(a_t, \phi, t)\| \propto \frac{1}{\varphi(c_j)}$, where $\varphi(c_j)$ is the concept frequency. Intuitively, the unconditional distribution of high-frequency concepts already shifts toward their direction, resulting in a small discrepancy; low-frequency concepts exhibit a larger discrepancy. During training, the ground-truth noise replaces the conditional prediction to improve stability: $D = \|\epsilon - \epsilon_\theta(x_t, \phi, t)\|_{sg}^\gamma$, where $sg$ denotes stop-gradient. In essence, the IMBA distance is equivalent to the **$\gamma$-th power of the unconditional loss**.

2. **Token-Level Reweighting (vs. Sample-Level)**: Conventional class-imbalance methods assign a single weight per sample, but in T2I tasks each image contains multiple concepts, with different spatial regions corresponding to different concepts. IMBA Loss applies different weights at the spatial (token) level, assigning different balancing strengths to regions corresponding to different concepts. This is more fine-grained and accurate than sample-level reweighting. The IMBA distance has shape $(B, N, C)$; averaging over the channel dimension improves stability.

3. **Online Adaptation (No Offline Statistics Required)**: Traditional data balancing requires a full pass over the dataset to compute frequency statistics, which becomes prohibitively expensive as datasets grow to millions or billions of samples. The IMBA distance is produced naturally at each training step, dynamically evolving with model training to always reflect the model's current understanding of the data distribution. It requires only one additional unconditional forward pass in the standard training pipeline—already part of classifier-free guidance training.

### Loss & Training

The final IMBA Loss takes the form: $L^* = \mathbb{E}_{t,x_0,\epsilon} D \|\epsilon - \epsilon_\theta(x_t, y, t)\|^2$, where $D = \|\epsilon - \epsilon_\theta(x_t, \phi, t)\|_{sg}^\gamma$.

Training algorithm (Algorithm 1):
1. Sample data pair $(x_0, y)$, noise $\epsilon$, timestep $t$
2. Compute noisy input $x_t$
3. Compute IMBA distance $D$ (unconditional prediction vs. ground-truth noise, stop-gradient)
4. Conditional loss $L^* = D \|\epsilon - \epsilon_\theta(x_t, y, t)\|^2$
5. Unconditional loss $L_u = \|\epsilon - \epsilon_\theta(x_t, \phi, t)\|^2$
6. Total loss $L = \lambda L^* + (1-\lambda) L_u$, where $\lambda = 0.9$

Key hyperparameters: $\gamma = 0.8$ (to avoid color shift), $\lambda = 0.9$ (consistent with the original conditional masking ratio).

## Key Experimental Results

### Main Results (Table)

**Quantitative Comparison on Three Benchmarks**

| Model | LC-Mis CLIP↑ | LC-Mis VQA↑ | T2I-Comp Color↑ | T2I-Comp Shape↑ | T2I-Comp Texture↑ | Inert-Comp CLIP↑ | Inert-Comp VQA↑ |
|------|-------------|-------------|-----------------|-----------------|-------------------|-----------------|-----------------|
| Baseline | 0.3045 | 46.21% | 0.5812 | 0.4307 | 0.6188 | 0.3194 | 44% |
| A&E (training-free) | 0.3198 | 48.42% | 0.6141 | 0.4378 | 0.6329 | 0.3303 | 44.5% |
| Finetune (diffusion loss) | 0.3073 | 51.82% | 0.6668 | 0.4919 | 0.6575 | 0.3172 | 46% |
| **IMBA Loss (Ours)** | 0.3121 | **62.89%** | **0.7067** | **0.5151** | **0.6861** | 0.3229 | **57%** |

IMBA Loss improves the VQA success rate from 46.21% to **62.89%** (+16.7 pp), with consistently leading attribute accuracy across Color, Shape, and Texture.

### Ablation Study (Table)

**Causal Experiment: Data Distribution vs. Data Scale**

| Concept Pair | Head Samples | Tail Samples | Ratio | Success Rate | CLIP Score |
|--------|----------|----------|------|--------|-----------|
| piano-submarine (large imbalance) | 3K | 0.03K | 100:1 | 16% | 0.3076 |
| piano-submarine (larger imbalance) | 15K | 0.15K | 100:1 | 20% | 0.3110 |
| piano-submarine (**balanced**) | **0.15K** | **0.15K** | **1:1** | **56%** | **0.3226** |
| volcano-twins (large imbalance) | 1K | 0.1K | 10:1 | 28% | 0.2986 |
| volcano-twins (larger imbalance) | 5K | 0.5K | 10:1 | 20% | 0.2948 |
| volcano-twins (**balanced**) | **0.5K** | **0.5K** | **1:1** | **64%** | **0.3137** |

**Key Finding**: A 5× increase in data scale (under the same imbalanced distribution) yields little or no improvement in concept composition success rate; in contrast, balanced data (even with fewer samples) substantially boosts success rates (16%→56%, 28%→64%).

**Comparison of Loss Weighting Granularity**

| Weighting Granularity | Baseline | Sample-wise | Token-wise (Ours) |
|---------|----------|-------------|-------------------|
| Success Rate | 32% | 64% | **72%** |
| CLIP Score | 0.2924 | 0.3022 | **0.3106** |

Token-wise reweighting outperforms sample-wise reweighting because it can assign different weights to different concept regions within the same image.

**Comparison of Balancing Methods**

| Method | Baseline | Frequency-based | IMBA Loss (Ours) |
|------|----------|----------------|------------------|
| Success Rate | 33.3% | 49.3% | **65.7%** |
| CLIP Score | 0.3113 | 0.3101 | **0.3218** |

### Key Findings

- **Model scale is not the bottleneck**: Scaling from 100M to 1B parameters yields no further improvement in concept composition ability beyond 200M.
- **Data scale is not the bottleneck either**: Increasing data volume under the same imbalanced distribution does not improve concept composition.
- **Data distribution is the root cause**: A balanced small dataset outperforms an imbalanced large dataset.
- **IMBA distance is stable across architectures**: The IMBA distance consistently reflects data distribution across different model sizes, architectures, and noise levels.
- **Inert concepts**: Low-frequency concepts are harder to compose; success rate decreases nearly linearly with concept frequency.
- **Effect of $\gamma$**: $\gamma \to 0$ degenerates to the standard loss; $\gamma \to 2$ induces severe color shift; $\gamma = 0.8$ achieves the best balance.

## Highlights & Insights

- **Rigorous causal analysis**: Unlike most prior work that directly proposes a method, this paper first validates causal hypotheses through controlled experiments before designing the solution—a research paradigm worth emulating.
- **Self-consistent theoretical derivation**: Starting from the ideal balanced distribution, the paper derives the form of the loss weight and finds that the IMBA distance ($\gamma$-th power of the unconditional loss) is a natural approximation of the inverse frequency—unifying theory and practice.
- **Minimal implementation overhead**: Only a few lines of modification to standard training code are required—computing the unconditional prediction (already available) and using it to reweight the conditional loss.
- **Elegant 2D synthetic experiment**: A two-dimensional spatial example intuitively illustrates the mechanism by which data imbalance causes the unconditional distribution to skew toward high-frequency concepts, thereby reducing the response strength for low-frequency concepts.
- **New Benchmark Inert-CompBench**: Specifically designed for inert concepts that are difficult to compose, filling the gap in existing benchmarks (T2I-CompBench, LC-Mis) regarding insufficient coverage of tail concepts.

## Limitations & Future Work

- Training is conducted on a 1B-parameter DiT model; validation on larger models (e.g., SDXL-scale) or commercial models is absent.
- The IMBA distance is cross-concept comparable only at $t=1000$ (full noise); comparisons at intermediate timesteps require further investigation.
- The hyperparameter choices $\gamma = 0.8$ and $\lambda = 0.9$ are empirically tuned and may require readjustment for different tasks or datasets.
- The method assumes that the unconditional distribution reflects concept frequency, an assumption that may not fully hold during early training when the model is underfitted.
- Improvement under fine-tuning is limited (44%→46% on Inert-CompBench); inert concepts may require longer training.
- The work focuses exclusively on noun concept composition; balancing of other concept types such as verbs and adverbs remains unexplored.

## Related Work & Insights

- Orthogonal to training-free methods such as Attend-and-Excite; the two can be combined for further gains.
- Data imbalance has been extensively studied in classification (Focal Loss, class-frequency weighting), but concept-level balancing in T2I generation poses unique challenges: each image contains multiple concepts, the concept space is open-ended, and frequency estimation is costly. IMBA Loss elegantly circumvents these issues.
- Implications for video generation (from the Kling team): concept composition in video is more complex (temporal concept interactions), and IMBA Loss can be directly extended to this setting.
- **Core insight**: "Apparently large-scale data is fundamentally imbalanced"—an observation broadly applicable to all generative models trained on large-scale internet data.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Causal analysis + theoretical derivation + online method; complete and elegant from problem identification to solution
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Controlled experiments + three benchmarks + extensive ablations, though the base model scale is relatively small
- **Writing Quality**: ⭐⭐⭐⭐ Narrative logic is clear, though the density of mathematical notation requires careful reading
- **Value**: ⭐⭐⭐⭐⭐ Broadly applicable to diffusion model training; applicable to any diffusion model with a few lines of code

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] BATCLIP: Bimodal Online Test-Time Adaptation for CLIP](batclip_bimodal_online_test-time_adaptation_for_clip.md)
- [\[ICCV 2025\] Lay2Story: Extending Diffusion Transformers for Layout-Togglable Story Generation](lay2story_extending_diffusion_transformers_for_layout-togglable_story_generation.md)
- [\[NeurIPS 2025\] Exploiting Vocabulary Frequency Imbalance in Language Model Pre-training](../../NeurIPS2025/llm_evaluation/exploiting_vocabulary_frequency_imbalance_in_language_model_pre-training.md)
- [\[ACL 2026\] Attribution, Citation, and Quotation: A Survey of Evidence-based Text Generation with Large Language Models](../../ACL2026/llm_evaluation/attribution_citation_and_quotation_a_survey_of_evidence-based_text_generation_wi.md)
- [\[NeurIPS 2025\] Conformal Online Learning of Deep Koopman Linear Embeddings](../../NeurIPS2025/llm_evaluation/conformal_online_learning_of_deep_koopman_linear_embeddings.md)

</div>

<!-- RELATED:END -->
