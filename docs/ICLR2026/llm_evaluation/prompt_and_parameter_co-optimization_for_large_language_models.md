---
title: >-
  [Paper Note] Prompt and Parameter Co-Optimization for Large Language Models
description: >-
  [ICLR 2026][LLM Evaluation][prompt optimization] This paper proposes MetaTuner, a framework that simultaneously generates prompts and LoRA parameters via a shared meta encoder, unifying discrete prompt optimization and continuous parameter fine-tuning into an end-to-end jointly optimizable framework, achieving substantial improvements over independently optimized methods on mathematical reasoning and question answering tasks.
tags:
  - ICLR 2026
  - LLM Evaluation
  - prompt optimization
  - fine-tuning
  - joint optimization
  - LoRA
  - discrete-continuous optimization
date: 2026-05-08
content_hash: 3edc01c01d0b3e0d
---

# Prompt and Parameter Co-Optimization for Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2509.24245](https://arxiv.org/abs/2509.24245)
**Code**: [https://github.com/BoXiaohe/MetaTuner](https://github.com/BoXiaohe/MetaTuner)
**Area**: LLM Evaluation
**Keywords**: prompt optimization, fine-tuning, joint optimization, LoRA, discrete-continuous optimization

## TL;DR
This paper proposes MetaTuner, a framework that simultaneously generates prompts and LoRA parameters via a shared meta encoder, unifying discrete prompt optimization and continuous parameter fine-tuning into an end-to-end jointly optimizable framework, achieving substantial improvements over independently optimized methods on mathematical reasoning and question answering tasks.

## Background & Motivation

**Background**: Post-training of LLMs has primarily followed two paradigms — prompt optimization (e.g., OPRO, RLPrompt, BPO), which activates the model's existing capabilities by finding appropriate input contexts, and fine-tuning (e.g., SFT, RLHF, DPO), which adapts model parameters to target data distributions. These two approaches are typically studied and applied independently.

**Limitations of Prior Work**: Prompt optimization can guide model behavior but fails to accommodate complex patterns in large-scale task data, particularly when prompt information conflicts with knowledge encoded in model parameters. Fine-tuning can adapt to data distributions but typically relies on manually designed prompts as input, and the choice of prompt critically affects fine-tuning performance — using a suboptimal prompt can even underperform pure prompt optimization.

**Key Challenge**: Prompts reside in a discrete optimization space (text tokens), while parameters reside in a continuous optimization space (floating-point weights). Their optimization objectives and execution pipelines are fundamentally different. The key challenge is how to simultaneously optimize these two complementary dimensions within a unified framework while resolving the non-differentiability inherent in mixed discrete-continuous optimization.

**Goal**: (a) How to design a framework enabling mutual reinforcement between prompts and parameters? (b) How to perform effective gradient-based optimization in a mixed discrete-continuous space? (c) Can the optimal prompt-parameter combination surpass the upper bound of independently optimized approaches?

**Key Insight**: Preliminary experiments reveal that fine-tuning methods are highly sensitive to prompt selection — SFT performance varies dramatically across different prompts and can fall below prompt optimization methods. This confirms the necessity of joint optimization.

**Core Idea**: Treat prompts as "special parameters" and generate both prompts and model parameters simultaneously through a shared encoder, achieving complementary mutual enhancement.

## Method

### Overall Architecture
The MetaTuner pipeline: input query $x_i$ → Meta Encoder (shared bottom layers $\phi_s$) → two parallel branches: Prompt Decoder ($\phi_p$) generating natural language prompt $p_i$, and Parameter Decoder ($\phi_q$) generating LoRA parameters $\theta_i$ → both the prompt and LoRA parameters are applied to the downstream Actor Model $\mathcal{M}$ for prediction → loss computation and backpropagation.

### Key Designs

1. **Prompt Generator $\mathcal{G}$ (Continuous Relaxation of Discrete Optimization)**:

    - **Function**: Uses an LLM to transform discrete prompt optimization into continuous parameter optimization.
    - **Mechanism**: Given an initial prompt $\tilde{p}$, a learnable LLM $\mathcal{G}_\phi$ rewrites it for each query to produce a customized prompt: $p_i = \mathcal{G}_\phi(\tilde{p}, x_i)$. This shifts the optimization objective from non-differentiable discrete token search to continuous optimization over $\phi$.
    - **Design Motivation**: Generating complete prompts from scratch is prohibitively difficult; the rewriting strategy substantially simplifies the search space. Furthermore, sharing the initial prompt across all queries reduces the annotation burden.

2. **Shared-Private Parameter Generation Architecture**:

    - **Function**: Enables mutual knowledge transfer between prompts and parameters via a shared meta encoder.
    - **Mechanism**: The parameters of $\mathcal{G}$ are decomposed as $\phi = \{\phi_s, \phi_p\}$, where $\phi_s$ denotes the shared bottom encoding layers (the first $k$ Transformer decoder layers) and $\phi_p$ denotes the prompt-specific upper layers. The Parameter Decoder $\mathcal{F}$ uses the same $\phi_s$ together with its own private parameters $\phi_q$. The unified objective is: $\min_{\phi_s, \phi_p, \phi_q} \sum_{i=1}^N \mathcal{L}(\mathcal{M}_{\mathcal{F}_{(\phi_s,\phi_q)}(\tilde{p},x_i)}(\mathcal{G}_{(\phi_s,\phi_p)}(\tilde{p},x_i), x_i), y_i)$
    - **Design Motivation**: The shared parameters $\phi_s$ enable mutual regularization between the two branches — a suboptimal solution in one branch can be corrected by the other under the unified loss. The private parameters $\phi_p$ and $\phi_q$ preserve the flexibility for each branch to independently explore optimal solutions.

3. **Concrete Implementation of the Parameter Decoder**:

    - **Function**: Generates query-specific LoRA weights from the hidden states $h_i$ produced by the shared encoder.
    - **Mechanism**: LoRA updates are applied as $\Delta W = \theta_i^b \cdot \theta_i^a$, generated from hidden states via two-layer matrix multiplication with ReLU activation: $\theta_i^b = \text{MM}(\text{ReLU}(\text{MM}(W_d^b, h_i)), W_u^b)$. The parameter decoder is parameterized as $\phi_q = \{W_d^b, W_u^b, W_d^a, W_u^a\}$, with a scaling factor $\lambda$ controlling the magnitude of the generated LoRA contributions.
    - **Design Motivation**: Using LoRA instead of full-parameter fine-tuning ensures training efficiency, while generating distinct LoRA parameters for each query enables fine-grained instance-level adaptation.

### Loss & Training

**Core Challenge**: The prompt decoder outputs discrete tokens, making it impossible to directly backpropagate gradients through $\phi_p$. **Solution — Supervised Regularization Loss**:

$$\min_{\phi_s, \phi_p, \phi_q} \sum_{(x_i,y_i) \in D_1} \mathcal{L}(\mathcal{M}_{\mathcal{F}}(\mathcal{G}_{(\phi_s,\phi_p')}(\tilde{p},x_i), x_i), y_i) + \sum_{(x_i,p_i) \in D_2} \alpha \cdot \mathcal{L}(\mathcal{G}_{(\phi_s,\phi_p)}(\tilde{p},x_i), p_i)$$

- **First term**: Main task loss, with $\phi_p'$ (prompt private parameters) frozen to ensure full differentiability.
- **Second term**: Supervised regularization, training the prompt decoder on $D_2 = \{(x_i, p_i)\}$ using optimal prompts collected via rollout.
- $\phi_p$ is periodically synchronized to $\phi_p'$ after updates.
- Gumbel-Softmax was evaluated but substantially underperformed supervised regularization, as its continuous relaxation introduces gradient bias.
- Two optimization strategies are considered: MetaTuner-I (alternating optimization of the two terms) and MetaTuner-J (joint optimization).

**Training Procedure**: Both $\mathcal{G}$ (Qwen2.5-7B) and $\mathcal{M}$ (Qwen2.5-3B) are first warmed up separately via SFT, followed by joint training.

## Key Experimental Results

### Main Results

Comparison across 4 benchmarks (Qwen2.5-7B as generator, Qwen2.5-3B as actor):

| Method | MATH | GSM8K | HotpotQA | CosmosQA | Type |
|--------|------|-------|----------|----------|------|
| Qwen2.5 (zero-shot) | 18.44 | 51.63 | 19.85 | 36.80 | Vanilla |
| BPO | 32.67 | 58.00 | 43.90 | 82.05 | Prompt |
| OPRO | 22.00 | 75.06 | 25.55 | 69.10 | Prompt |
| SFT | 41.33 | 61.41 | 43.20 | 82.65 | Fine-tune |
| DPO | 43.78 | 63.68 | 44.70 | 87.90 | Fine-tune |
| BetterTogether | 41.56 | 67.93 | 52.30 | 89.80 | Hybrid |
| **MetaTuner-J** | **48.67** | **78.92** | **54.56** | **92.25** | Hybrid |

MetaTuner-J achieves an average improvement of 10.15% over BetterTogether (7B backbone), with gains of +7.11 on MATH and +10.99 on GSM8K.

### Ablation Study

| Configuration | MATH | GSM8K | HotpotQA | CosmosQA | Note |
|---------------|------|-------|----------|----------|------|
| MetaTuner (w/o F) | 48.00 | 77.79 | 54.05 | 91.10 | Removing fine-tuning branch, all metrics drop |
| MetaTuner (w/o P) | 46.22 | 78.54 | 53.90 | 91.00 | Removing prompt branch, MATH drops by 2.45 |
| MetaTuner (w/o S) | 46.67 | 77.86 | 53.65 | 91.50 | No parameter sharing, all metrics drop |
| **MetaTuner (full)** | **48.67** | **78.92** | **54.56** | **92.25** | Full model achieves best performance |

### Key Findings
- **Both branches are indispensable**: Removing the fine-tuning or prompt branch leads to average performance drops of approximately 0.99% and 1.12%, respectively.
- **Parameter sharing is critical**: Removing shared parameters (w/o S) consistently underperforms the full model, validating the effectiveness of mutual reinforcement.
- **Optimal sharing ratio depends on model scale**: For the 7B model, the optimal sharing ratio is $K/4$ (retaining more private layers), whereas for the 3B model it is $3K/4$ (more sharing to enhance coherence).
- **Supervised regularization outperforms Gumbel-Softmax**: Optimizing directly in the discrete space avoids approximation errors introduced by continuous relaxation.
- **Rollout sample count should not be excessive**: An excessive number of samples leads to over-exploration, disrupting previously learned effective information.
- **Joint optimization (MetaTuner-J) slightly outperforms alternating optimization (MetaTuner-I)**, except on HotpotQA where alternating optimization performs better.

## Highlights & Insights
- **Treating prompts as "special parameters"**: This unified perspective dissolves the traditional boundary between prompt optimization and fine-tuning, enabling both to complement each other under a shared objective. This viewpoint is transferable to other scenarios requiring coordination between discrete decisions and continuous optimization.
- **Supervised regularization for mixed discrete-continuous optimization**: The method elegantly constructs supervised signals from rollout-optimal prompts to train the prompt decoder, avoiding gradient bias inherent in relaxation approaches such as Gumbel-Softmax. This technique generalizes to other tasks involving discrete structure generation.
- **Query-specific prompts and LoRA parameters**: Rather than applying a single fixed prompt or a single set of LoRA parameters to all inputs, MetaTuner dynamically generates both for each query, enabling fine-grained instance-level adaptation.

## Limitations & Future Work
- **Computational overhead**: A 7B generator model is required to serve a 3B actor model; the inference cost of the generator itself is non-negligible in practical deployment scenarios.
- **Dependency on warmup phase**: Separate SFT warmup for both models prior to joint training increases overall pipeline complexity.
- **Evaluation limited to Qwen series**: Generalizability across diverse architectures (e.g., Llama, Mistral) has not been verified.
- **Limited capacity of discrete prompts**: Prompts consisting of tens to hundreds of tokens have constrained information capacity, which may be insufficient for tasks requiring extensive domain knowledge.

## Related Work & Insights
- **vs. BetterTogether**: BetterTogether also performs joint optimization, but lacks shared bottom-layer knowledge between the prompt and parameter branches and does not support end-to-end differentiable training. MetaTuner achieves deeper synergy through the shared encoder and supervised regularization, yielding average improvements exceeding 10%.
- **vs. OPRO/CFPO**: Pure prompt optimization methods encounter a performance ceiling on mathematical reasoning, as they cannot adapt model parameters. MetaTuner improves on MATH from OPRO's 22.00 to 48.67.
- **vs. DPO/PPO**: Pure fine-tuning methods are constrained by fixed prompts. MetaTuner improves on GSM8K from DPO's 63.68 to 78.92.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The idea of jointly optimizing prompts and parameters is original; the shared-private architecture and supervised regularization represent solid technical contributions, though the core intuition is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Comprehensive coverage across 4 datasets, 10+ baselines, detailed ablations, generalization experiments, and hyperparameter analysis.
- **Writing Quality**: ⭐⭐⭐⭐ — Mathematical derivations are clear and the logic from problem formulation to methodology to experiments is coherent, though the notation in Section 3 is relatively heavy.
- **Value**: ⭐⭐⭐⭐ — Introduces a new paradigm for LLM post-training with substantial improvements across multiple tasks; however, computational cost and pipeline complexity may limit practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Prompt and Parameter Co-Optimization for Large Language Model Task Adaptation](prompt_and_parameter_co-optimization_for_large_language_model_task_adaptation.md)
- [\[ICLR 2026\] Spectral Attention Steering for Prompt Highlighting](spectral_attention_steering_for_prompt_highlighting.md)
- [\[ICLR 2026\] ASIDE: Architectural Separation of Instructions and Data in Language Models](aside_architectural_separation_of_instructions_and_data_in_language_models.md)
- [\[ICLR 2026\] Soft Quality-Diversity Optimization](soft_quality-diversity_optimization.md)
- [\[ICLR 2026\] Breaking the Correlation Plateau: On the Optimization and Capacity Limits of Attention-Based Regressors](breaking_the_correlation_plateau_on_the_optimization_and_capacity_limits_of_atte.md)

</div>

<!-- RELATED:END -->
