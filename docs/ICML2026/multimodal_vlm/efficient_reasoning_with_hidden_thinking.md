---
title: >-
  [Paper Note] Efficient Reasoning with Hidden Thinking
description: >-
  [ICML 2026][Multimodal VLM][Heima] Heima distills each stage (summary / caption / reasoning) of a multimodal LLM's lengthy Chain-of-Thought (CoT) into **a single special thinking token**. This enables the model to "think…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Heima"
  - "thinking tokens"
  - "progressive distillation"
  - "information-theoretic bound"
  - "interpreter"
date: 2026-05-08
content_hash: 5b1f5b0b5e2eb425
---

# Efficient Reasoning with Hidden Thinking

**Conference**: ICML 2026  
**arXiv**: [2501.19201](https://arxiv.org/abs/2501.19201)  
**Code**: https://github.com/shawnricecake/Heima  
**Area**: Multimodal VLM / Efficient Inference / Latent Space Reasoning / CoT Compression  
**Keywords**: Heima, thinking tokens, progressive distillation, information-theoretic bound, interpreter

## TL;DR
Heima distills each stage (summary / caption / reasoning) of a multimodal LLM's lengthy Chain-of-Thought (CoT) into **a single special thinking token**. This enables the model to "think" within the latent space, reducing the token count from the 100-200 range to 13-16 while achieving more stable zero-shot accuracy than LLaVA-CoT. Additionally, an LLM "interpreter" is trained to reconstruct the textual reasoning chain from the hidden states of these thinking tokens to empirically validate the information-theoretic upper bound of compression loss.

## Background & Motivation

**Background**: The use of CoT for complex multimodal reasoning in MLLMs has become mainstream (e.g., LLaVA-CoT), but generating hundreds of tokens per CoT leads to high inference latency and prohibitive API costs. Although Coconut (Hao et al., 2024) explored CoT compression on GPT-2, it was only validated on text-only small models.

**Limitations of Prior Work**: (1) CoT in MLLMs is longer than in pure text (requiring image descriptions plus reasoning), exacerbating latency; (2) Existing latent reasoning methods (Cheng & Van Durme) compress the entire CoT into a continuous embedding, leading to significant accuracy drops in mathematical tasks—indicating that "naive compression" loses critical information; (3) There is a lack of a theoretical framework to define how many tokens can be compressed without sacrificing reasoning capabilities.

**Key Challenge**: Shorter CoT leads to faster inference, but every truncated segment may reduce $I(Y;\text{CoTs}|X)$ (the target-related mutual information carried by the CoT). To quantify this trade-off and ensure that $I(Y;\langle CoTs\rangle|X)>0$ after compression, a formal information-theoretic characterization and empirical verification are required.

**Goal**: (i) Design a latent CoT compression framework for MLLMs; (ii) Formalize the "compression-accuracy" trade-off using information theory; (iii) Design an interpreter capable of reconstructing textual CoT to empirically verify compression loss.

**Key Insight**: CoT in LLaVA-CoT is organized by stages (summary / caption / reasoning). Since each stage is a semantically independent unit, it can be distilled into **a single stage-token**. Although the capacity of a single token's hidden state is finite, its 768-4096 dimensions are sufficient to store the "semantic fingerprint of a reasoning segment."

**Core Idea**: Each CoT stage is distilled into a special token $\langle CoT\rangle_{(k)}$. The model generates these tokens directly in the embedding space to produce the final answer. A separate LLM interpreter then reconstructs the corresponding textual CoT from the hidden states of these tokens as empirical evidence of information retention.

## Method

### Overall Architecture
Two types of models:
- **Heima** (based on LLaVA-CoT-11B / LLaVA-Next-Vicuna-7B): Performs latent space reasoning. Takes (image, question) as input and outputs $K_i$ thinking tokens + the final answer.
- **Interpreters** (based on Llama-3.1-8B / Vicuna-7B, pure LLM without vision): One per CoT stage. Takes (explanatory prompt, textual question, hidden state of thinking token) as input and outputs the original CoT text.

During inference, only Heima is used, reducing the token count from $\sum |CoT(k)|$ (~100-200) to $K_i$ (3-4). Interpreters are used only for "information-theoretic empirical analysis."

### Key Designs

1. **Stage-level thinking token distillation + Information-theoretic Guarantee**:
    - **Function**: Compresses each CoT stage (summary / caption / reasoning) into **a single** unique special token added to the vocabulary.
    - **Mechanism**: The original dataset $D=\{(X,\text{CoTs},Y)\}$ is modified to $D_H=\{(X,\langle CoTs\rangle,Y)\}$, where $\langle CoTs\rangle:=\{\langle CoT\rangle_{(k)}\}_{k=1}^{K_i}$, replacing each stage with a new token from the vocabulary. The distillation objective $\mathcal{L}(\theta)=-\mathbb{E}_{(X,Y,\langle CoTs\rangle)\sim D_H}\log P_\theta(\langle CoTs\rangle,Y|X)$ involves fine-tuning the model to predict the thinking token sequence and the answer. Theorem 3.1 states: since $\langle CoTs\rangle=f(X,\text{CoTs})$, there exists a Markov chain $Y-(X,\text{CoTs})-\langle CoTs\rangle$. Thus, $0\leq I(Y;\langle CoTs\rangle|X)\leq I(Y;\text{CoTs}|X)$, and the information gap $I(Y;\text{CoTs}|X)-I(Y;\langle CoTs\rangle|X)=I(Y;\text{CoTs}|X,\langle CoTs\rangle)\geq 0$ determines the "compression loss." Reasoning ability is preserved as long as $I(Y;\langle CoTs\rangle|X)>0$.
    - **Design Motivation**: The authors formalize "CoT compression" as an information-theoretic compression problem, providing a quantifiable retention-loss decomposition. This transforms the sufficiency of thinking tokens from an empirical question into a measurable quantity via the interpreter. Stage-shared tokens are used to prevent vocabulary explosion.

2. **Progressive Distillation**:
    - **Function**: Prevents optimization failure caused by compressing all CoT stages simultaneously.
    - **Mechanism**: Training is divided into $M=\max\{K_i\}+1$ stages. For phase $s$, the training data is $D_P=\{(X,\{\langle CoT\rangle_{(k)}\}_{k=1}^s,\{CoT_{(k)}\}_{k=s+1}^{K_i},Y)\}$, where the first $s$ stages are compressed into thinking tokens while the remaining stages remain as text. This proceeds stage-by-stage until completion. A final "recovering stage" using only thinking tokens is added to smooth transitions between stages.
    - **Design Motivation**: Compressing all stages at once forces single tokens to bear too much compression burden, making the loss landscape difficult to optimize. A curriculum approach allows the model to "internalize" reasoning patterns incrementally.

3. **Adaptive Interpreter**:
    - **Function**: Trains a text-only LLM to reconstruct the textual CoT from the thinking token's hidden state to **empirically** measure information loss.
    - **Mechanism**: Each stage $k$ corresponds to an interpreter $\mathcal{I}_{\theta_k}$. Training data $D_I=\{(X_e,X_q,\langle CoT\rangle_{(k)},H_{\langle CoT\rangle_{(k)}},CoT_{(k)})\}$ includes an explanatory prompt, text question (no image), thinking token, its last hidden state, and the original CoT text. **Key Operation**: The interpreter replaces the word embedding of the thinking token with the last hidden state $H_{\langle CoT\rangle_{(k)}}$ from Heima, as reasoning information is encoded in the hidden state rather than the token ID. The loss is standard next-token prediction.
    - **Design Motivation**: While theory provides bounds, the actual gap must be measured. Similarity between reconstructed and original CoT indicates smaller information loss. This architecture also proves Heima is performing latent reasoning rather than simple overfitting.

### Loss & Training
- **Heima**: LoRA fine-tuning of LLaVA-CoT-11B (rank=16, alpha=32). Image encoder is frozen; attention, MLP, and output projections are updated. Progressive distillation uses $M$ stages, one epoch each.
- **Interpreter**: Same LoRA configuration, next-token prediction loss, extracting hidden states from the frozen Heima model.
- All experiments conducted using torchtune on 8×H100 GPUs.

## Key Experimental Results

### Main Results

Zero-shot performance on 6 benchmarks for the LLaVA-CoT-11B series (token counts in parentheses):

| Model | MMStar | MMBench | MMVet | MathVista | AI2D | Hallusion | Avg | Tokens |
|-------|--------|---------|-------|-----------|------|-----------|-----|--------|
| Llama-3.2-11B-Vision | 48.1 | 58.2 | 50.2 | 50.3 | 68.5 | 37.2 | 52.1 | ~119 |
| LLaVA-CoT | 54.0 | 70.7 | 49.8 | 50.9 | 77.6 | 63.8 | 61.1 | ~189 |
| Heima w/o progressive | 49.7 | 72.5 | 39.0 | 39.3 | 75.9 | 61.3 | 56.3 | ~23 |
| Heima w/o recover | 49.8 | 71.6 | 42.8 | 39.8 | 77.3 | 58.5 | 56.6 | ~24 |
| **Heima (full)** | 49.9 | **72.8** | 43.3 | 43.6 | 77.5 | 60.6 | 58.0 | **~24** |

Heima averages 13-17 tokens (a **10-15× reduction** compared to LLaVA-CoT) and even surpasses LLaVA-CoT on MMBench and AI2D.

### Ablation Study

| Configuration | Avg Acc | Notes |
|------|---------|------|
| Heima w/o progressive | 56.3 | All stages compressed at once (-1.7) |
| Heima w/o recover | 56.6 | Final recovery stage removed (-1.4) |
| Heima (full) | 58.0 | Complete method |

Interpreter reconstruction quality (4300 samples) across BLEU-4 / METEOR / ROUGE / BERTScore + GPT-4o similarity suggests reconstructed text "closely aligns with original CoTs," verifying controllable information gaps.

### Key Findings
- **MathVista Performance**: Heima (43.6) < LLaVA-CoT (50.9) despite 16× fewer tokens, indicating that mathematical reasoning still heavily relies on full CoT, representing a bottleneck case.
- **Progressive Distillation is Critical**: Its removal results in a 1.7% drop; removing the recovery stage results in a 1.4% drop.
- **Efficiency**: A compression ratio of **14×** (~189 → ~13-17 tokens) is achieved with only a 3% drop in average accuracy (61.1 → 58.0).
- **Interpretability**: The interpreter can reconstruct almost complete captions and reasoning from hidden states, proving they carry genuine reasoning information.

## Highlights & Insights
- **Scale and Rigor**: First to achieve latent CoT compression with rigorous information-theoretic analysis and interpretable verification at the MLLM scale (11B).
- **Theoretical Formalization**: Theorem 3.1 provides a rare formal baseline for the latent reasoning field by quantifying the $I(Y;\langle CoTs\rangle|X)>0$ requirement.
- **Interpretability as Diagnosis**: The interpreter serves as both a diagnostic tool for information volume and an alignment/safety tool by revealing "what the model is thinking" in its hidden states.
- **Balanced Design**: Stage-shared tokens balance expressive power with vocabulary overhead.

## Limitations & Future Work
- **Mathematical Bottleneck**: Obvious performance drop in MathVista suggests 13-17 hidden tokens cannot capture complex arithmetic chains.
- **Manual Granularity**: The "one token per stage" design is heuristic; adaptive stage numbers or token counts were not explored.
- **Dependency**: Heavily relies on the stage divisions of the LLaVA-CoT dataset.
- **Training Cost**: Multiple interpreters (one per stage) increase training resource requirements linearly.
- **Scale**: Unknown performance on larger models (34B+) or closed-source models like GPT-4V.

## Related Work & Insights
- **vs. Coconut (Hao et al., 2024)**: Coconut focused on GPT-2 and single-task math; Heima scales to 11B MLLM with a broader benchmark suite and professional theoretical framework.
- **vs. Cheng & Van Durme 2024**: Unlike continuous embedding compression which degrades math accuracy significantly, Heima's discrete stage-token approach with progressive distillation mitigates catastrophic degradation.
- **vs. Speculative Decoding/Medusa**: These optimize autoregressive parallelization; Heima operates on the representation dimension and can be combined with these methods.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First stage-token latent CoT on MLLM with information-theoretic and interpreter validation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive benchmarks and ablations, though lacks wall-clock latency and RLHF comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical sections and vivid motivating examples.
- Value: ⭐⭐⭐⭐⭐ Addresses the primary deployment bottleneck of MLLMs with 14× compression at minimal accuracy cost. Code is open-sourced.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Bad Seeing or Bad Thinking? Rewarding Perception for Vision-Language Reasoning](bad_seeing_or_bad_thinking_rewarding_perception_for_vision-language_reasoning.md)
- [\[ICML 2026\] From Seeing to Thinking: Decoupling Perception and Reasoning Improves Post-Training of Vision-Language Models](from_seeing_to_thinking_decoupling_perception_and_reasoning_improves_post-traini.md)
- [\[AAAI 2026\] AStar: Boosting Multimodal Reasoning with Automated Structured Thinking](../../AAAI2026/multimodal_vlm/astar_boosting_multimodal_reasoning_with_automated_structure.md)
- [\[ICLR 2026\] SophiaVL-R1: Reinforcing MLLMs Reasoning with Thinking Reward](../../ICLR2026/multimodal_vlm/sophiavl-r1_reinforcing_mllms_reasoning_with_thinking_reward.md)
- [\[ICML 2026\] Thinking in Structures: Evaluating Spatial Intelligence in Constraint-Governed Spaces](thinking_in_structures_evaluating_spatial_intelligence_in_constraint-governed_sp.md)

</div>

<!-- RELATED:END -->
