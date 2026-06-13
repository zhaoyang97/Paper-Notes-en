---
title: >-
  [Paper Note] Efficient Reasoning with Hidden Thinking
description: >-
  [ICML 2026][Multimodal VLM][Heima] Heima distills each stage (summary / caption / reasoning) of a multimodal LLM’s lengthy CoT into a **special thinking token**…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Heima"
  - "thinking tokens"
  - "progressive distillation"
  - "information-theoretic bound"
  - "interpreter"
date: 2026-05-08
content_hash: 2403525059767648
---

# Efficient Reasoning with Hidden Thinking

**Conference**: ICML 2026  
**arXiv**: [2501.19201](https://arxiv.org/abs/2501.19201)  
**Code**: https://github.com/shawnricecake/Heima  
**Area**: Multimodal VLM / Efficient Reasoning / Latent Space Reasoning / CoT Compression  
**Keywords**: Heima, thinking tokens, progressive distillation, information-theoretic bound, interpreter

## TL;DR
Heima distills each stage (summary / caption / reasoning) of a multimodal LLM’s lengthy CoT into a **special thinking token**, enabling the model to "think" in latent space. The number of tokens drops from 100-200 to 13-16, while zero-shot accuracy is more stable than LLaVA-CoT. An auxiliary LLM "interpreter" is trained to reconstruct the textual reasoning chain from the thinking token’s hidden state, empirically verifying the information-theoretic upper bound of compression loss.

## Background & Motivation

**Background**: Using CoT for complex multimodal reasoning in MLLMs has become mainstream (e.g., LLaVA-CoT), but each CoT requires generating hundreds of tokens, leading to high inference latency and API costs. Coconut (Hao et al., 2024) compressed CoT on GPT-2, but only validated on text and small models.

**Limitations of Prior Work**: (1) MLLM CoTs are even longer than pure text (must describe images + reasoning), making latency a more severe issue; (2) Existing latent reasoning methods (Cheng & Van Durme) compress the entire CoT into a continuous embedding, but accuracy drops sharply on math tasks—indicating that "blind compression" loses key information; (3) There is a lack of theoretical frameworks to determine how much token compression can save costs without sacrificing reasoning ability.

**Key Challenge**: Shorter CoTs yield faster reasoning, but removing any part of the CoT may reduce $I(Y;\text{CoTs}|X)$ (the target-relevant mutual information carried by the CoT). To quantify this trade-off and ensure that after compression $I(Y;\langle CoTs\rangle|X)>0$, a formal information-theoretic characterization and empirical validation are required.

**Goal**: (i) Design a latent CoT compression framework for MLLMs; (ii) Formalize the "compression-accuracy" trade-off using information theory; (iii) Design an interpreter capable of reconstructing textual CoTs to empirically measure compression loss.

**Key Insight**: LLaVA-CoT’s CoT is organized by stages (summary / caption / reasoning), each being a semantically independent unit, making it suitable for distillation into **one stage-token**. Although a token’s hidden state has limited capacity, 768-4096 dimensions are sufficient to encode the "semantic fingerprint" of a reasoning segment.

**Core Idea**: Each CoT stage is distilled into a special token `<CoT>(k)`, and the model generates these tokens directly in embedding space to produce the final answer. A separate LLM interpreter then reconstructs the corresponding textual CoT from each token’s hidden state, serving as empirical evidence of information retention.

## Method

### Overall Architecture
Two types of models:
- **Heima** (based on LLaVA-CoT-11B / LLaVA-Next-Vicuna-7B): Performs latent space reasoning, takes (image, question) as input, outputs $K_i$ thinking tokens + final answer.
- **Interpreters** (based on Llama-3.1-8B / Vicuna-7B, pure LLM without vision): One per CoT stage, takes (explanatory prompt, textual question, thinking token’s hidden state) as input, outputs the original CoT text.

Only Heima is used during inference, reducing token count from ∑|CoT(k)| (~100-200) to $K_i$ (3-4). Interpreters are used solely for "information-theoretic empirical analysis."

### Key Designs

1. **Stage-level Thinking Token Distillation + Information-Theoretic Guarantee**:

    - **Function**: Compress each CoT stage (summary / caption / reasoning) into **one** unique special token, added to the vocabulary.
    - **Mechanism**: The original dataset $D=\{(X,\text{CoTs},Y)\}$ is transformed into $D_H=\{(X,\langle CoTs\rangle,Y)\}$, where $\langle CoTs\rangle:=\{\langle CoT\rangle_{(k)}\}_{k=1}^{K_i}$, with each stage replaced by a newly added token in the vocabulary. The distillation objective $\mathcal{L}(\theta)=-\mathbb{E}_{(X,Y,\langle CoTs\rangle)\sim D_H}\log P_\theta(\langle CoTs\rangle,Y|X)$ directly fine-tunes the model to predict the sequence of thinking tokens and the answer. Theorem 3.1: Since $\langle CoTs\rangle=f(X,\text{CoTs})$, there is a Markov chain $Y-(X,\text{CoTs})-\langle CoTs\rangle$, so $0\leq I(Y;\langle CoTs\rangle|X)\leq I(Y;\text{CoTs}|X)$, and the information gap $I(Y;\text{CoTs}|X)-I(Y;\langle CoTs\rangle|X)=I(Y;\text{CoTs}|X,\langle CoTs\rangle)\geq 0$ determines the "compression loss." As long as $I(Y;\langle CoTs\rangle|X)>0$, reasoning ability is preserved.
    - **Design Motivation**: The authors formalize "CoT compression" as an information-theoretic problem, providing a quantifiable retention-loss decomposition. This makes "whether thinking tokens are sufficient" a measurable quantity, empirically testable via the interpreter. One token per stage is chosen because the LLaVA-CoT dataset is stage-structured, with all samples sharing the same token for stage k (not per-sample), preventing vocabulary explosion.

2. **Progressive Distillation (Stage-by-Stage Distillation)**:

    - **Function**: Avoids optimization failure from compressing all CoT stages into tokens at once.
    - **Mechanism**: Training is divided into $M=\max\{K_i\}+1$ stages. At stage $s$, the training data is $D_P=\{(X,\{\langle CoT\rangle_{(k)}\}_{k=1}^s,\{CoT_{(k)}\}_{k=s+1}^{K_i},Y)\}$—the first $s$ stages are compressed into thinking tokens, the remaining $K_i-s$ stages remain as text. Training proceeds stage by stage until all stages are compressed; finally, a "recovering stage" uses only thinking tokens for training, smoothing transitions between stages.
    - **Design Motivation**: Compressing all stages at once forces each token to handle too much, making the loss landscape hard to optimize. Curriculum learning allows the model to gradually "internalize" each stage’s reasoning pattern, learning one new compression at a time. The recovering stage addresses alignment issues when combining individually compressed stages.

3. **Adaptive Interpreter (Reconstructing Textual CoT from Hidden State to Quantify Information Loss)**:

    - **Function**: Trains a pure-text LLM to reconstruct the textual CoT for each stage from the thinking token’s hidden state, thereby **empirically** measuring information retention.
    - **Mechanism**: Each stage $k$ has an interpreter $\mathcal{I}_{\theta_k}$ (initialized with Llama-3.1-8B). Training data $D_I=\{(X_e,X_q,\langle CoT\rangle_{(k)},H_{\langle CoT\rangle_{(k)}},CoT_{(k)})\}$ includes the explanatory prompt, textual question (no image), thinking token, last hidden state of the token, and the original CoT text. **Key operation**: The interpreter’s input replaces the thinking token’s word embedding with Heima’s output last hidden state $H_{\langle CoT\rangle_{(k)}}$, since reasoning information is encoded in the hidden state, not the token id. The loss is standard next-token prediction: $\max_{\theta_k}\mathbb{E}\log P_{\theta_k}(CoT_{(k)}|X_e,X_q,H_{\langle CoT\rangle_{(k)}})$. Prompts are like "According to question $X_q$, can you explain the thinking progress $\langle CoT\rangle_{(k)}$?"
    - **Design Motivation**: While theory provides upper and lower bounds for the information gap, the actual difference must be measured empirically. The closer the interpreter’s reconstructed textual CoT is to the original, the smaller $I(Y;\text{CoTs}|X,\langle CoTs\rangle)$ is, indicating more complete information retention. This architecture also demonstrates that "Heima truly reasons in latent space rather than simply overfitting"—since information can be decoded from the hidden state back into coherent text. The paper’s BMW logo example shows the interpreter reconstructing "sleek modern sports car with black exterior" and "cross with a circle" from the hidden state, perfectly matching the original CoT.

### Loss & Training

- Heima: LoRA fine-tuning on LLaVA-CoT-11B (rank=16, alpha=32), freezing the image encoder, updating all attention, MLP, and output projection layers. Progressive distillation is conducted in $M$ stages, one epoch per stage.
- Interpreter: Same LoRA configuration, next-token prediction loss, hidden states extracted from the frozen Heima.
- All training uses torchtune + 8×H100.

## Key Experimental Results

### Main Results

LLaVA-CoT-11B series, 6 zero-shot benchmarks (token counts in parentheses):

| Model | MMStar | MMBench | MMVet | MathVista | AI2D | Hallusion | Avg | Tokens |
|-------|--------|---------|-------|-----------|------|-----------|-----|--------|
| Llama-3.2-11B-Vision | 48.1 | 58.2 | 50.2 | 50.3 | 68.5 | 37.2 | 52.1 | ~119 |
| LLaVA-CoT | 54.0 | 70.7 | 49.8 | 50.9 | 77.6 | 63.8 | 61.1 | ~189 |
| Heima w/o progressive | 49.7 | 72.5 | 39.0 | 39.3 | 75.9 | 61.3 | 56.3 | ~23 |
| Heima w/o recover | 49.8 | 71.6 | 42.8 | 39.8 | 77.3 | 58.5 | 56.6 | ~24 |
| **Heima (full)** | 49.9 | **72.8** | 43.3 | 43.6 | 77.5 | 60.6 | 58.0 | **~24** |

Heima averages 13-17 tokens (10-15× fewer than LLaVA-CoT on CoT benchmarks); on MMBench / AI2D, Heima is even more accurate than LLaVA-CoT.

### Ablation Study

| Configuration | Avg Acc | Notes |
|---------------|---------|-------|
| Heima w/o progressive | 56.3 | Compressing all stages at once, drops 1.7 |
| Heima w/o recover | 56.6 | No final recovering stage, drops 1.4 |
| Heima (full) | 58.0 | Complete method |

Interpreter reconstruction quality (4300 samples): Evaluated with BLEU-4 / METEOR / ROUGE / BERTScore + GPT-4o similarity. The paper states reconstructed texts "closely align with original CoTs," verifying the information gap is controllable.

### Key Findings
- **On MathVista, Heima 43.6 < LLaVA-CoT 50.9** but with 16× fewer tokens, indicating that mathematical reasoning still relies on a complete CoT and is the bottleneck case; on MMBench / AI2D, Heima is more accurate, possibly due to removal of CoT noise.
- **Progressive distillation is crucial**: Removing it loses 1.7%; the recovering stage adds another 1.4%—showing both curriculum and late-stage alignment are necessary.
- **Token count drops from ~189 → ~13-17, a 14× compression**, but average accuracy drops only 3% (61.1→58.0), making it highly cost-effective.
- **Interpreter can reconstruct nearly complete captions + reasoning from hidden states** (e.g., BMW example), proving that the hidden state is not a black box but truly carries reasoning information.
- **Without progressive distillation, MMVet plummets to 39.0** (vs full 43.3), showing that compressing all at once is infeasible for long CoT tasks.

## Highlights & Insights
- **First to combine latent CoT compression, rigorous information-theoretic analysis, and an interpretable interpreter at MLLM scale**. Coconut only did GPT-2 + math; Heima achieves this on 11B MLLM + 6 multimodal benchmarks.
- **Information-theoretic Theorem 3.1 is a rare formal result in latent reasoning**: It quantifies the intuition that "compression is possible but must ensure $I(Y;\langle CoTs\rangle|X)>0$," providing a baseline for future latent reasoning frameworks.
- **Interpreter serves as both diagnostic and interpretability tool**: After training, it objectively evaluates hidden information content and shows end users "what the model is actually thinking in the hidden space," valuable for alignment and safety.
- **Stage-shared token design balances expressiveness and vocabulary size**: All samples share $\langle CoT\rangle_{(k)}$, preventing vocabulary explosion while maintaining stage semantics.
- **Progressive distillation’s training paradigm is transferable to any task involving "compression of multi-segment semantic units"**, such as long-context summarization, dialogue history compression, or multi-step code generation.

## Limitations & Future Work
- **Significant drop in mathematical reasoning** (MathVista 50.9→43.6), indicating that 13-17 hidden tokens cannot encode a complete arithmetic chain; latent reasoning still has bottlenecks in precise symbolic operations.
- **One token per stage is manually designed**; adaptive stage/token numbers are not explored.
- **Relies on LLaVA-CoT dataset’s stage segmentation**; for other CoT data without stage labels, stage segmentation is required first.
- **Interpreter training cost doubles**: One interpreter per stage, nearly linear growth for multi-stage tasks.
- **Not validated on larger models (34B+) or GPT-4V**; scaling law is unclear.
- **No comparison with an extended Coconut on MLLM**—Coconut uses continuous thinking embeddings, Heima uses token-level; which is better for MLLMs remains an open question.
- **Future directions**: (i) Allow variable number of thinking tokens (e.g., one token to indicate "how many more steps to think"); (ii) Multiple tokens per stage to improve mathematical reasoning; (iii) Use latent CoT for reward shaping in RLHF.

## Related Work & Insights
- **vs Coconut (Hao et al., 2024)**: Coconut is on GPT-2 + single-task text math; Heima is on 11B MLLM + 6 multimodal benchmarks + information-theoretic framework, with much greater scale and breadth.
- **vs Cheng & Van Durme 2024**: Compresses CoT into continuous embedding, but math accuracy degrades sharply; Heima uses token-level discrete representation with progressive distillation, avoiding catastrophic degradation.
- **vs Speculative decoding / Medusa**: Parallel acceleration for autoregressive models, a different optimization dimension from latent reasoning, and can be combined.
- **vs LISA / VLM-Latent (Lai et al., Pi et al.)**: Injects visual information into LLM hidden states for downstream tasks (segmentation / detection); Heima, conversely, encodes reasoning processes in the hidden state, proving hidden states can carry not only vision but also logic.
- **vs RLHF efficient reasoning**: RLHF learns short CoTs via reward, but is still textual; Heima fundamentally changes the representation, a more fundamental optimization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First stage-token latent CoT on MLLM, with rigorous information-theoretic analysis + interpreter closed-loop validation
- Experimental Thoroughness: ⭐⭐⭐⭐ 6 zero-shot benchmarks + two model families + complete ablation; lacks latency wall-clock data and RLHF comparison
- Writing Quality: ⭐⭐⭐⭐⭐ Information-theoretic section is concise and rigorous, BMW logo motivating example is vivid
- Value: ⭐⭐⭐⭐⭐ MLLM reasoning cost is a major deployment bottleneck; 14× token compression with only 3% average accuracy drop is industry-grade optimization; code is open source

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
