---
title: >-
  [Paper Note] DiBO: Offline Black-box Optimization with Diffusion Language Models (DNA + Robot Morphology)
description: >-
  [ICML 2026][Robotics][Offline BBO] DiBO adapts the LLaDA-8B diffusion language model to offline black-box optimization (BBO) scenarios. It uses delimiter tokens to unify three types of heterogeneous signals (prompt…
tags:
  - "ICML 2026"
  - "Robotics"
  - "Offline BBO"
  - "Diffusion LLM"
  - "Bidirectional Modeling"
  - "Domain Adaptation"
  - "Offline RL"
date: 2026-05-08
content_hash: 69789e3230d76358
---

# DiBO: Offline Black-box Optimization with Diffusion Language Models (DNA + Robot Morphology)

**Conference**: ICML 2026 Spotlight  
**arXiv**: [2603.17919](https://arxiv.org/abs/2603.17919)  
**Code**: Provided on the paper page (here link)  
**Area**: Black-box Optimization / Diffusion Language Models / Design Generation  
**Keywords**: Offline BBO, Diffusion LLM, Bidirectional Modeling, Domain Adaptation, Offline RL

## TL;DR
DiBO adapts the LLaDA-8B diffusion language model to offline black-box optimization (BBO) scenarios. It uses delimiter tokens to unify three types of heterogeneous signals (prompt, design, and label), followed by a three-stage post-training pipeline: "domain adaptation $\rightarrow$ masked-response SFT $\rightarrow$ label-improvement RL." This approach allows the model to achieve SOTA performance on multiple Design-Bench tasks with only 500 labeled samples (e.g., +8% normalized score on DNA tasks), with a single离散 task training completed in 1.5 hours on an H100.

## Background & Motivation

**Background**: Black-box optimization (BBO) is critical in fields such as DNA sequencing, robot morphology, and materials discovery. However, experimental labeling is expensive, making online optimization unfeasible. Offline BBO assumes a static dataset $\mathcal{D} = \{(\bm{x}_i, y_i)\}$ and aims to find a new design $\bm{x}^*$ that outperforms the dataset. Traditional methods follow two paths: (a) learning a surrogate $f_\theta(\bm{x})$ followed by gradient ascent (COMs, ICT, MATCH-OPT), though surrogate gradients become unreliable under OOD conditions; (b) learning inverse generative models (CbAS, MIN, BONET, DDOM) to sample high-scoring designs directly.

**Limitations of Prior Work**: (1) Autoregressive (AR) LLMs (OPRO, UniSO-T) generate tokens unidirectionally, whereas design tasks like DNA are constrained by both prefix and suffix dependencies, which left-to-right generation fails to capture. (2) Existing diffusion-based BBO methods (DDOM, GTG) mostly use task-specific architectures in continuous spaces and cannot naturally integrate textual task descriptions. (3) Existing offline BBO methods suffer from severe overfitting in small-data settings ($N \approx 500$) and lack the relief provided by large model priors.

**Key Challenge**: Achieving a balance between bidirectional modeling (suitable for DNA/morphology), textual task description fusion (suitable for general BBO), and leveraging LLM pre-training priors within a single architecture is difficult. While diffusion LLMs are inherently bidirectional, they are pre-trained on natural text, creating a domain gap with heterogeneous signals like "design tokens + numerical labels."

**Goal**: Adapt diffusion LLMs for BBO to retain bidirectional modeling advantages, learn the "prompt $\rightarrow$ superior design" mapping in small-data settings, and perform fine-grained alignment using RL signals.

**Key Insight**: Resolve semantic role conflicts between "design/labels vs. natural text" using delimiter tokens and unified prompt-response corpora. Use a three-stage post-training pipeline—masked joint reconstruction, SFT, and RL—to achieve alignment.

**Core Idea**: Treat the "simultaneous prediction of masked prompt tokens + response tokens" as the BBO domain adaptation goal. Use sample pairs where the response label exceeds all labels in the prompt as SFT data. Finally, use the "response label - prompt label" as a reward for one-step log-prob RL. This sequence allows an 8B diffusion LLM to reach SOTA on BBO tasks using only 500 samples.

## Method

### Overall Architecture

Input: (1) A natural language task description (including design semantics, format, and optimization goals); (2) a set of few-shot (design, label) pairs; and (3) an instruction to "generate a superior design." Output: A sequence of design + label tokens enclosed by delimiters.

DiBO adds four components on top of a frozen diffusion LLM: (a) Tokenizer expansion with two sets of delimiters `|design-start|/|design-end|` and `|label-start|/|label-end|`; (b) A Domain Adaptation (DA) stage jointly predicting masked tokens for prompts and responses on unified corpora; (c) An SFT stage predicting only masked response tokens, using the design with the highest label outside the prompt as the target; (d) An RL stage using label improvement as a reward, approximated by one-step unmask log-probabilities. During inference, the masked response is filled via single-step greedy unmasking.

### Key Designs

1.  **Delimiter Tokens + Domain Adaptation on Unified Prompt-Response Corpora**:
    *   **Function**: Uniformly encodes heterogeneous signals (prompt text, design tokens, numerical labels) to let the diffusion LLM learn the semantic roles of the delimiters.
    *   **Mechanism**: Expands the tokenizer with 4 delimiter tokens. Each training sample is structured as a unified sequence of `[Prompt Text][Few-shot (design, label) pairs][Instruction]` + `[Response design][Response label]`. The DA objective is $\mathcal{L}_{\mathrm{DA}} = -\mathbb{E}[\frac{1}{t} \sum_{i=1}^{L} \mathbf{1}[q_t^i=[M], o_t^i=[M]] \log p_\theta(q_0^i, o_0^i | q_t, o_t)]$, which reconstructs masked tokens for both prompt and response. Contextual designs in few-shot prompts are selected from the offline pool based on kernel similarity to the response design to avoid learning degenerate mappings.
    *   **Design Motivation**: Treating numerical labels as plain text causes diffusion LLMs to perceive them as noise. Explicit delimiters and joint reconstruction enable the model to learn segmentation boundaries and roles. This allows the direct reuse of diffusion LLM pre-training priors and attention mechanisms.

2.  **Two-stage Post-training: Masked-Response SFT + Label-Improvement RL**:
    *   **Function**: Enables the diffusion LLM to generate designs with labels exceeding the maximum in the prompt context and connects coarse-grained (SFT) and fine-grained (RL) signals.
    *   **Mechanism**: During SFT, the prompt is frozen, and masked reconstruction is performed only on the response. The loss is $\mathcal{L}_{\mathrm{SFT}} = -\mathbb{E}[\frac{1}{t} \sum_i \mathbf{1}[o_t^i=[M]] \log p_\theta(o_0^i | q_0, o_t)]$. The target response satisfies $y(o) > \max y(\text{prompt})$, providing a hard inductive bias for improvement. In the RL stage, a dataset $\mathcal{D}_{rl}$ is constructed with reward $r(q, o) = y(o) - y(q)$. The loss is $\mathcal{L}_{\mathrm{RL}} = -\mathbb{E}[\frac{1}{|o|} \sum_k p_\theta(o_k | q, o_{\text{fullmask}}) \cdot \frac{r(q,o)}{\sigma}]$, using one-step unmasking to approximate token-wise log-probabilities.
    *   **Design Motivation**: SFT provides the "direction" (optimization), while RL provides the "magnitude" (how much better). One-step unmasking is a key trick—while traditional diffusion log-probs require iterative denoising for stability, one-step is sufficient for short BBO sequences and saves 50× computation, making RL feasible on a single H100.

3.  **Kernel-Similarity Context Selection + Single-step Greedy Inference**:
    *   **Function**: Ensures prompt examples and target responses are within the same distribution to avoid unreasonable learning signals; improves sampling efficiency during inference.
    *   **Mechanism**: During data construction, for a given target response $o$, the top-7 designs $x_i$ are selected from the offline pool based on kernel similarity $k(o, x_i)$. During inference, 7 in-context examples are used, and a single forward pass followed by greedy filling is performed on the masked response to generate unique candidates.
    *   **Design Motivation**: The "improve over prompt" assumption collapses if prompts are entirely in low-score regions. Kernel similarity ensures prompts and targets are locally close in the design space, focusing the model on learning "incremental improvement." Single-step greedy inference is an inherent advantage of diffusion LLMs over AR LLMs, which require $K$ steps for a $K$-token design.

### Loss & Training
The process is sequential: DA (1024 steps) $\rightarrow$ SFT (1024 steps) $\rightarrow$ RL (128 steps). All stages use PagedAdamW8bit + Bfloat16 + 100-step linear warmup. Learning rates: $2 \times 10^{-5}$ for DA and SFT, $1 \times 10^{-6}$ for RL to preserve the SFT prior. Each task uses an offline pool of 500 samples with $n_{few}=7$.

## Key Experimental Results

### Main Results: Design-Bench (100th percentile normalized score, average of 8 seeds)

| Method | Ant Morphology | D'Kitty Morphology | TF Bind 8 | TF Bind 10 | Mean | Rank Mean ↓ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathcal{D}$(best) | 0.565 | 0.884 | 0.439 | 0.511 | — | — |
| Grad-mean | 0.709 | 0.920 | 0.843 | 0.736 | 0.802 | 4.25 |
| COMs | 0.647 | 0.934 | 0.843 | 0.709 | 0.783 | 4.5 |
| ExPT | **0.929** | 0.950 | 0.810 | 0.703 | 0.848 | 4.0 |
| OPRO (AR LLM) | 0.517 | 0.856 | 0.758 | 0.500 | 0.658 | 14.0 |
| DDOM (Diffusion) | 0.590 | 0.929 | 0.739 | 0.497 | 0.689 | 11.25 |
| MCTS-transfer | 0.648 | 0.910 | 0.857 | 0.628 | 0.761 | 7.25 |
| **DiBO (Ours)** | **0.932** | 0.912 | **0.946** | **0.741** | **0.883** | **3.5** |

DiBO achieves the highest scores in 3 out of 4 tasks (Ant, TF8, TF10), outperforming the strongest baseline on TF Bind 8 by 0.082. It achieves the best Rank Mean (3.5) and Rank Median (1.0).

### Ablation Study: Diffusion vs. Autoregressive backbone (same DA $\rightarrow$ SFT $\rightarrow$ RL pipeline)

| Task | Stage | AR (LLaMA-3.1-8B-Instruct) | DiBO (Diffusion) | Gain |
| :--- | :--- | :---: | :---: | :---: |
| TF Bind 8 | DA | 0.803 | 0.883 | +0.080 |
| TF Bind 8 | SFT | 0.875 | 0.939 | +0.064 |
| TF Bind 8 | RL | 0.915 | 0.946 | +0.031 |
| TF Bind 10| DA | 0.623 | 0.644 | +0.021 |
| TF Bind 10| SFT | 0.633 | 0.704 | +0.071 |
| TF Bind 10| RL | 0.682 | 0.741 | +0.059 |
| Ant | RL | 0.930 | 0.932 | +0.002 |
| D'Kitty | RL | 0.912 | 0.912 | 0.000 |

On discrete DNA tasks (TF8/TF10), the diffusion backbone significantly leads the AR backbone across all stages, confirming that bidirectional modeling provides a genuine advantage for tasks with strong bidirectional dependencies. The gap converges to zero on continuous robot tasks.

### Key Findings

*   **Sweet Spot for Small Data + LLM Priors**: Achieving SOTA on $N=500$ samples is a highlight; while traditional BBO overfits, the 8B LLM pre-training prior acts as a regularizer.
*   **Cumulative Gains of Three-stage Post-training**: Scores increase at every stage (DA $\rightarrow$ SFT $\rightarrow$ RL), proving they provide complementary signals: format, direction, and magnitude.
*   **OPRO Failure Suggests Prompting is Insufficient**: OPRO uses prompting without fine-tuning and achieves a mean of 0.658, far below DiBO's 0.883, indicating that LLMs must undergo in-domain adaptation.
*   **Training Efficiency**: Completing the TF Bind 8 pipeline in 1.5 GPU hours on a single H100 demonstrates that diffusion LLMs are computationally viable for BBO.

## Highlights & Insights

*   **Delimiters + Unified Corpora are a simple yet effective bridge**: Instead of specialized architectures, expanding the tokenizer by a few tokens solves the heterogeneity problem in a lightweight, transferable way.
*   **Three-stage Post-training matches cognitive levels (Prior $\rightarrow$ Direction $\rightarrow$ Magnitude)**: This coarse-to-fine strategy is a valuable reference for adapting LLMs to other new tasks like medical QA or scientific discovery.
*   **Real Advantage of Diffusion LLMs in Discrete Design**: The ablation clearly shows diffusion winning in DNA (bidirectional) while tying with AR in robot morphology (continuous, weak bidirectional).
*   **One-step unmasking as a log-prob approximation**: Reducing RL training cost for diffusion models from $N$ steps to 1 step has significant methodological implications.

## Limitations & Future Work

*   **Limited Design Space**: Experiments covered DNA (length 8/10) and robot morphology (56D/60D). Scalability to high-dimensional spaces (protein sequences, circuit design) is unverified.
*   **Uniform Behavior Policy Assumption**: The RL stage assumes a uniform distribution for the old policy, which may introduce bias if offline data is highly non-uniform.
*   **Fragility of Kernel Similarity Selection**: Selecting top-7 examples close to the target might leak target information into the prompt, requiring more rigorous discussion.
*   **Small Margin on TF Bind 10**: The gain on TF10 (+0.005) is much smaller than on TF8 (+0.082), possibly due to label noise or scale.

## Related Work & Insights

*   **vs. OPRO**: OPRO uses prompting only; DiBO proves fine-tuning is necessary and provides a sequence for it.
*   **vs. DDOM / GTG**: Those methods use task-specific diffusion in continuous spaces; DiBO uses a general diffusion LLM to integrate text + design + labels.
*   **vs. dLLM (Yuan et al. 2026)**: They use inference-time MCTS; DiBO uses train-time adaptation, which yields better results and saves inference computation.
*   **Insight**: The "delimiter token + unified corpora + three-stage post-training" is a general recipe for any field requiring LLMs to process structured non-textual signals.

## Rating

*   **Novelty**: ⭐⭐⭐⭐ Systematic adaptation of diffusion LLMs to BBO is a new paradigm; the three-stage design is practical and elegant.
*   **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 4 Design-Bench tasks, 10+ baselines, and strict diffusion/AR control ablations.
*   **Writing Quality**: ⭐⭐⭐⭐ Clear flowcharts and mathematical expressions.
*   **Value**: ⭐⭐⭐⭐⭐ Highly applicable to small-data BBO in drug discovery and robotics; 1.5-hour training time makes it industrial-ready.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Cross-Embodiment Offline Reinforcement Learning for Heterogeneous Robot Datasets](../../ICLR2026/robotics/cross-embodiment_offline_reinforcement_learning_for_heterogeneous_robot_datasets.md)
- [\[ICML 2026\] Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](discrete_diffusion_vla_bringing_discrete_diffusion_to_action_decoding_in_vision-.md)
- [\[ICML 2026\] Towards Efficient and Expressive Offline RL via Flow-Anchored Noise-conditioned Q-Learning](towards_efficient_and_expressive_offline_rl_via_flow-anchored_noise-conditioned_.md)
- [\[ICML 2026\] Dual-Stream Diffusion for World-Model Augmented Vision-Language-Action Model](dual-stream_diffusion_for_world-model_augmented_vision-language-action_model.md)
- [\[CVPR 2026\] DAWN: Pixel Motion Diffusion is What We Need for Robot Control](../../CVPR2026/robotics/dawn_pixel_motion_diffusion_robot_control.md)

</div>

<!-- RELATED:END -->
