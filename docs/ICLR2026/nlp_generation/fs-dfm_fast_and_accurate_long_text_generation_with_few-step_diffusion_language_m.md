---
title: >-
  [Paper Note] FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model
description: >-
  [ICLR 2026][Text Generation][discrete diffusion models] This paper proposes FS-DFM (Few-Step Discrete Flow-Matching), which reduces the sampling steps of discrete flow-matching language models from 1024 to 8 through step…
tags:
  - "ICLR 2026"
  - "Text Generation"
  - "discrete diffusion models"
  - "few-step sampling"
  - "flow matching"
  - "cumulative scalar"
date: 2026-05-08
content_hash: 2f9cdcfb8e2bf183
---

# FS-DFM: Fast and Accurate Long Text Generation with Few-Step Diffusion Language Model

**Conference**: ICLR 2026
**arXiv**: [2509.20624](https://arxiv.org/abs/2509.20624)  
**Code**: [GitHub](https://github.com/apple/ml-fs-dfm)  
**Area**: Text Generation
**Keywords**: discrete diffusion models, few-step sampling, flow matching, cumulative scalar, text generation

## TL;DR

This paper proposes FS-DFM (Few-Step Discrete Flow-Matching), which reduces the sampling steps of discrete flow-matching language models from 1024 to 8 through step-aware training and a cumulative scalar update rule, achieving a 128× speedup while maintaining comparable perplexity and generation quality.

## Background & Motivation

Autoregressive language models (ARMs) achieve excellent generation quality but are inherently sequential—each token requires one forward pass, limiting throughput. Diffusion language models (DLMs) enable parallel generation across positions, but standard discrete diffusion typically requires hundreds to thousands of model evaluations to achieve high quality, essentially trading iterative depth for parallel breadth.

Specifically:
- **ARM bottleneck**: Strictly sequential decoding with one forward pass per token leads to high latency on long sequences.
- **DLM bottleneck**: Parallel but step-inefficient. For example, LLaDA requires approximately one inference step per output token to match ARM quality; DFM (Discrete Flow-Matching) requires ~1024 steps for 1024-token generation.

The goal of this paper is to enable DLMs to achieve multi-step (1024-step) generation quality with as few as 8 steps. Prior few-step methods (e.g., SDTT) still require 16–256 steps and target only short texts. FS-DFM is the first few-step discrete flow-matching method designed for long text (1024 tokens).

## Method

### Overall Architecture

FS-DFM builds upon Discrete Flow-Matching (DFM), which models text as a continuous-time discrete Markov chain (CTMC) and learns a probability velocity field to transport a source distribution (e.g., uniform or mask) to the target data distribution along a probability path. At inference time, generation starts from a noisy sequence and iteratively denoises along the learned velocity field.

FS-DFM rests on two pillars:

1. **Step-aware DFM training**: The step budget $h$ is injected as an explicit conditioning signal, with large-step consistency achieved via a shortcut teacher distillation scheme.
2. **Cumulative scalar update**: Replaces the instantaneous scalar by integrating the scheduler's rate function over the finite step interval $[t, t+h]$, ensuring correct probability flux even under large-step updates.

### Key Designs

**1. Step-Aware Discrete Flow-Matching**

The step budget $h$ is introduced as an explicit conditioning signal: model outputs logits $= \theta(x_t, t; h)$. During training, $h$ is sampled from a log-uniform grid $h = \{2^k,\ k = -10, \ldots, 0\}$, covering the full range from fine-grained path tracking to large-step generation.

To teach the model to take large steps, a teacher is needed to provide "ground truth" over large step intervals. However, directly solving the transition probability from the generator $u_t$ over large intervals is intractable. Since the transition probability satisfies the Kolmogorov equation (essentially an ODE), it can be approximated numerically.

This paper employs **RK-4 (classical fourth-order Runge-Kutta)** as the shortcut teacher:
- The model is evaluated at times $t$, $t+h/2$, and $t+h$ to obtain logits.
- A weighted average is computed following the RK-4 formula: $\bar{l} = \frac{1}{6}(l_1 + 2l_2 + 2l_3 + l_4)$.
- Each step requires 4 model evaluations but provides significantly better approximation accuracy than RK-2.

**EMA Teacher for Training Stability**: Since training is non-stationary, using the current model directly as the teacher causes the target to drift as parameters update. Therefore, an EMA copy (with $\beta$ close to 1) is used as the teacher, with gradients stopped, providing stable and low-variance targets.

**2. Cumulative Scalar**

This is the core innovation of FS-DFM. In DFM, the marginal velocity decomposes into a scalar (scale) and a direction:

$$u_t^i(x^i, z) = g(t) \cdot [p_{1|t}(x^i|z) - \delta_z(x^i)]$$

where $g(t) = \kappa'(t)/(1-\kappa(t))$ is the instantaneous rate. The problem is that in few-step sampling, the first step typically occurs at small $t$, where the instantaneous scalar $g(t)$ is too weak to trigger effective token transitions, causing sampling to stall.

The cumulative scalar resolves this by integrating $g$ over the entire step interval $[t, t+h]$ and normalizing:

$$\bar{g}_{t,h} = \frac{1}{h} \ln\frac{1-\kappa(t)}{1-\kappa(t+h)}$$

For the linear scheduler $\kappa(t) = t$, this gives $\bar{g} = \frac{1}{h}\ln\frac{1-t}{1-t-h}$.

The resulting velocity becomes $\bar{u}_t^i = \bar{g}_{t,h} \cdot [p_{1|t} - \delta_z]$. This way, even at small $t$, a sufficiently large $h$ provides enough probability flux to drive token transitions.

**3. Budget-Aware Mixed Loss**

Training switches loss functions based on step size:

- Small steps ($h < \tau = 2^{-9}$): Uses the DFM path loss $\mathcal{L}_\text{dfm}$ (Bregman divergence) to ensure local path fidelity.
- Large steps ($h \geq \tau$): Uses KL divergence loss $\mathcal{L}_\text{dist}$ to align student and shortcut teacher outputs.

The final loss selectively combines both for each sample based on the magnitude of $h$:

$$\mathcal{L} = \frac{1}{B}\sum_b \left(m_b \mathcal{L}_\text{dfm}^{(b)} + (1-m_b)\mathcal{L}_\text{dist}^{(b)}\right)$$

### Loss & Training

A two-stage **pretrain-then-finetune** strategy is adopted:

1. **Stage 1**: A standard DFM backbone (without step awareness) is trained on the FineWeb-Edu dataset.
2. **Stage 2**: The DFM checkpoint is fine-tuned with the step-aware objective and cumulative scalar.

Training details:
- Dataset: FineWeb-Edu for training, WikiText-103 for evaluation
- Tokenizer: GPT-2 tokenizer
- Sequence length: 1024 tokens (documents concatenated and padded)
- Model scales: 0.169B, 1.3B, 1.7B
- Source distributions: both uniform and mask
- Scheduler: linear $\kappa(t) = t$

## Key Experimental Results

### Main Results

**Table 1: Cumulative Scalar Ablation (0.169B model)**

| Sampling Steps (NFE) | Standard Scalar PPL | Cumulative Scalar PPL | Reduction |
|:---:|:---:|:---:|:---:|
| 1 | 1312.65 | 514.40 | −60.8% |
| 2 | 462.31 | 333.07 | −28.0% |
| 4 | 194.29 | 176.19 | −9.3% |
| 8 | 97.51 | 90.49 | −7.2% |
| 1024 | 85.61 | 87.36 | ~on par |

The cumulative scalar yields large gains at few steps (60.8% reduction at 1 step) and converges to the same level at many steps.

**Table 2: FS-DFM vs. Diffusion Language Models (512-token continuation)**

| Method | Scale | 1-step PPL | 2-step PPL | 4-step PPL | 8-step PPL | 16-step PPL |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| Dream | 7B | 1163.08 | 785.87 | 752.11 | 739.40 | 630.30 |
| LLaDA | 8B | 256.07 | 290.35 | 495.17 | 441.26 | 432.65 |
| **FS-DFM** | **0.17B** | **173.39** | **143.77** | **97.07** | **75.78** | **67.42** |
| **FS-DFM** | **1.3B** | **231.89** | **169.99** | **99.79** | **70.97** | **59.84** |
| **FS-DFM** | **1.7B** | **191.20** | **155.01** | **101.20** | **72.84** | **61.67** |

FS-DFM (0.17B) at 8 steps (PPL 75.78) substantially outperforms LLaDA (8B) at 16 steps (PPL 432.65), despite being 40× smaller. Dream and LLaDA tend to generate repetitive tokens in the few-step regime.

### Ablation Study

- **RK-4 vs. RK-2**: RK-4 achieves ~12% lower PPL than RK-2 across all NFE values (median ratio 0.88×), at the cost of additional model evaluations during training.
- **EMA Teacher**: Using an EMA copy as the teacher significantly improves training stability over using the current model directly.
- **Source Distributions**: FS-DFM is effective under both uniform and mask source distributions.
- **Step Sampling Strategy**: A log-uniform grid covering the full range outperforms strategies concentrated in a sub-interval.

### Key Findings

1. **128× speedup**: FS-DFM matches DFM's 1024-step quality using only 8 steps for 1024-token generation.
2. **Small model beats large model**: 0.17B FS-DFM substantially outperforms 7–8B Dream/LLaDA in few-step generation.
3. **Large-step convergence**: As NFE increases (i.e., $h \to 0$), FS-DFM smoothly converges to standard DFM, since the cumulative scalar $\bar{g}$ approaches the instantaneous $g(t)$ in this limit.
4. **Cumulative scalar is the key**: The largest gains from the cumulative scalar are observed at large steps (1–2 NFE).
5. **MAUVE comparison**: FS-DFM achieves MAUVE scores of 0.39–0.58 at 16 steps, while LLaDA/Dream at 16 steps achieve only ~0.005.

## Highlights & Insights

- The problem framing is precise: the paper identifies the core obstacle in few-step DFM sampling as the weakness of the instantaneous scalar at small $t$, and the proposed cumulative scalar constitutes an elegant mathematical remedy.
- The RK-4 shortcut teacher seamlessly integrates ODE solvers with discrete diffusion training.
- The pretrain-then-finetune strategy avoids the costly step-aware training from scratch.
- This is the first open-source DFM + FS-DFM model and codebase (from Apple).
- The result of a 0.17B model decisively outperforming an 8B model in the few-step regime is particularly surprising.

## Limitations & Future Work

1. Evaluation is primarily conducted at sequence length 1024; scalability to longer sequences (4K+) remains unverified.
2. Only the GPT-2 tokenizer and small-to-medium scale models are used; direct quality comparisons with modern LLMs (7B+ ARMs) are absent.
3. The RK-4 teacher requires 4 forward passes per step during training, incurring high training cost.
4. The performance of FS-DFM on conditional generation tasks (summarization, translation, etc.) is not explored.
5. Although generation diversity (MAUVE) surpasses LLaDA/Dream, absolute values remain relatively low.

## Related Work & Insights

FS-DFM represents the first comprehensive adaptation of few-step/one-step distillation ideas from continuous-space flow matching (e.g., consistency models, shortcut models) to discrete text space. The cumulative scalar concept is also potentially applicable to other CTMC-based applications requiring large discrete step updates. This work further demonstrates that discrete diffusion still holds substantial promise for language modeling, with step efficiency being the critical bottleneck to address.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (first application of cumulative scalar + RK-4 shortcut teacher in discrete text FM)
- Technical Depth: ⭐⭐⭐⭐⭐ (CTMC theory + Kolmogorov ODE + Bregman divergence; mathematically rigorous)
- Experimental Thoroughness: ⭐⭐⭐⭐ (multiple model scales + multiple baselines + thorough ablations; lacks conditional generation evaluation)
- Practicality: ⭐⭐⭐⭐ (128× speedup is practically meaningful, though the DLM ecosystem remains immature overall)
- Writing Quality: ⭐⭐⭐⭐ (mathematical exposition is clear, but the dense notation requires frequent cross-referencing)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](../../ICCV2025/nlp_generation/beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)
- [\[ACL 2026\] Frankentext: Stitching Random Text Fragments into Long-Form Narratives](../../ACL2026/nlp_generation/frankentext_stitching_random_text_fragments_into_long-form_narratives.md)
- [\[ICML 2026\] Characterizing the Effect of Noise in Language Generation in the Limit](../../ICML2026/nlp_generation/characterizing_the_effect_of_noise_in_language_generation_in_the_limit.md)
- [\[AAAI 2026\] C3TG: Conflict-aware, Composite, and Collaborative Controlled Text Generation](../../AAAI2026/nlp_generation/c3tg_conflict-aware_composite_and_collaborative_controlled_text_generation.md)
- [\[ACL 2026\] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation](../../ACL2026/nlp_generation/planning_beyond_text_graph-based_reasoning_for_complex_narrative_generation.md)

</div>

<!-- RELATED:END -->
