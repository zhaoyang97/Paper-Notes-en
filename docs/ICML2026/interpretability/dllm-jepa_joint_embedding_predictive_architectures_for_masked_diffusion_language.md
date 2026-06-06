---
title: >-
  [Paper Note] DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models
description: >-
  [ICML 2026][Interpretability][JEPA] The authors introduce a JEPA representation alignment objective during the fine-tuning phase of masked diffusion language models. By slicing the same sentence into a "low-masked contex…
tags:
  - "ICML 2026"
  - "Interpretability"
  - "JEPA"
  - "Masked Diffusion Language Models"
  - "Representation Learning"
  - "Fine-tuning"
  - "EMA Target Encoder"
date: 2026-05-08
content_hash: 33fae18983bd9c0e
---

# DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2606.00091](https://arxiv.org/abs/2606.00091)  
**Code**: TBD  
**Area**: LLM Pre-training / Representation Learning / Diffusion Language Models  
**Keywords**: JEPA, Masked Diffusion Language Models, Representation Learning, Fine-tuning, EMA Target Encoder

## TL;DR
The authors introduce a JEPA representation alignment objective during the fine-tuning phase of masked diffusion language models. By slicing the same sentence into a "low-masked context view" and a "high-masked target view" based on different masking ratios, the model performs a single gradient-enabled forward pass on the context view to compute both diffusion loss and the JEPA embedding, while using an EMA copy for a no-grad pass on the target view. Compared to LLM-JEPA, this method saves 33% of training FLOPs and achieves consistent performance gains across 4 tasks and 2 backbones (e.g., +18.7 pp on GSM8K).

## Background & Motivation
**Background**: The dominant training paradigm for Large Language Models (LLMs) is input-space reconstruction—specifically, autoregressive next-token prediction (GPT family) or masked token reconstruction (BERT). Recently, the vision domain has shifted significantly toward Joint Embedding Predictive Architectures (JEPA), which predict the embedding of one view from another in latent space (e.g., I-JEPA, V-JEPA). This avoids the low-level biases of pixel-level reconstruction and facilitates learning more abstract representations.

**Limitations of Prior Work**: LLM-JEPA represents a rare attempt to apply JEPA to language models by treating (text, code) pairs as "two views of the same knowledge." However, it suffers from two fundamental flaws: ① **Explicit view dependency**—it requires naturally paired data (text↔code) and cannot rely on data augmentation like vision models. The authors acknowledge this as a critical limitation. ② **Doubled computational overhead**—autoregressive models require causal masking and block-causal attention, forcing both views to undergo gradient-enabled forward passes, which makes the training FLOPs per step double that of standard SFT.

**Key Challenge**: The "two views + latent prediction" paradigm of JEPA naturally assumes that two views can be encoded in parallel and bidirectionally. However, the causality of autoregressive LMs breaks this assumption, forcing models to either construct complex views or expend twice the computational power.

**Goal**: To identify an LM architecture where two views for JEPA can be naturally generated from a single input (without paired data) and where a single gradient-enabled forward pass suffices to obtain both task logits and JEPA embeddings.

**Key Insight**: The authors observe that masked diffusion language models (e.g., LLaDA, MDLM, SEDD) naturally satisfy these requirements. They utilize **bidirectional attention** and **random masking denoising**, making their training process structurally isomorphic to the "view prediction" in JEPA: different masking ratios naturally serve as two distinct views.

**Core Idea**: Use the diffusion noise schedule as a data augmenter (sample two masking ratios $t_L < t_H$ for the same sentence to generate two views). A single gradient-enabled forward pass of the context view outputs both diffusion logits and a pooled embedding, while the target view uses an EMA copy for a no-grad forward pass, saving half of the backpropagation cost compared to LLM-JEPA.

## Method

### Overall Architecture
The input is a clean text sequence $x_0$. Following the masked diffusion forward process, two masking ratios are independently sampled: $t_L=0.2$ (context view) and $t_H=0.7$ (target view). These are used to noise $x_0$ into $x_{t_L}$ (20% tokens masked) and $x_{t_H}$ (70% tokens masked). The online backbone $f_\theta$ performs **one gradient-enabled forward pass** on $x_{t_L}$ to output: (a) token distributions for each masked position—used for the standard diffusion loss $\mathcal{L}_\text{diff}$; (b) a JEPA context embedding $z_{t_L}$ obtained via mean pooling and LayerNorm over non-masked and non-padding tokens. The target encoder $f_{\theta'}$ is an EMA copy of $f_\theta$ (decay $\tau=0.996$), which processes $x_{t_H}$ under `no_grad` to produce $z_{t_H}$. A lightweight predictor $g_\phi$ (a $k$-layer Transformer decoder) maps $z_{t_L}$ to $\hat z_{t_H}=g_\phi(z_{t_L})$. The total loss combines diffusion and a cosine-based JEPA alignment: $\mathcal{L}_\text{total}=\mathcal{L}_\text{diff}+\lambda(1-\cos(\text{sg}(z_{t_H}), \hat z_{t_H}))$. The computational cost per step is $\approx 4F$ (1 gradient forward + 1 no-grad forward + 1 backward $\approx 2F$), which is 33% less than the $6F$ required by LLM-JEPA.

### Key Designs

1.  **"Paired-free view generation" via diffusion noise schedule**:
    *   **Function**: Uses the difference in masking ratios $t_L < t_H$ to sample two views $x_{t_L}, x_{t_H}$ from the same $x_0$, eliminating LLM-JEPA's dependence on paired data (like text/code) and making it applicable to **any text dataset**.
    *   **Mechanism**: The diffusion forward process $q(x_t^i|x_0^i)$ acts as a stochastic masking augmenter. The low-mask view retains most tokens as a "near-complete context," while the high-mask view results in sparse tokens acting as a "highly abstract target." These are perfectly suited for the JEPA paradigm, and the generation process is reused from the diffusion training objective at zero extra data cost. Fixed values $t_L=0.2$ and $t_H=0.7$ provide a reasonable trade-off, while a "Wide-tt" configuration $(0.1, 0.9)$ is used in base-preservation experiments.
    *   **Design Motivation**: Unifying "view construction via augmentation" from vision JEPA and "sample generation via random masking" from diffusion LMs into a single schedule is the cornerstone of this method.

2.  **Single gradient forward pass + EMA no-grad target branch**:
    *   **Function**: Compresses "diffusion denoising" and "JEPA context encoding" into **the same** gradient-enabled forward pass. The target view is computed using only an EMA copy in a no-grad pass, avoiding LLM-JEPA’s second backward pass, additional gradient memory, and optimizer states.
    *   **Mechanism**: Thanks to the bidirectional attention in diffusion LMs, the hidden states of $f_\theta(x_{t_L})$ can simultaneously feed a token classifier for diffusion logits and a pooling layer for $z_{t_L} = \text{Pool}(f_\theta(x_{t_L}))$ for the JEPA input, **requiring no second forward pass**. The target branch follows the classic I-JEPA/BYOL recipe: EMA updates $\theta' \leftarrow \tau\theta' + (1-\tau)\theta$, `no_grad` wrapping, stop-gradient, and a predictor to prevent collapse. The predictor $g_\phi$ is a $k$-layer decoder ($k \in \{1, \dots, 5\}$), and the loss is $\mathcal{L}_{\text{JEPA}} = 1 - \cos(\text{sg}(z_{t_H}), \hat z_{t_H})$.
    *   **Design Motivation**: As shown in Table 1, AR and diffusion baselines cost $3F$/step. LLM-JEPA increases this to $6F$ (+100%), while DLLM-JEPA only costs $4F$ (+33%), making it computationally feasible to integrate JEPA into the diffusion LM fine-tuning loop.

3.  **Joint objective + multiple anchors to prevent collapse**:
    *   **Function**: Ensures the cosine-only JEPA objective is stable and non-collapsing without requiring negative samples or variance/covariance regularization (like VICReg).
    *   **Mechanism**: The total loss is $\mathcal{L}_\text{total}=\mathcal{L}_\text{diff}+\lambda\mathcal{L}_\text{JEPA}$ with $\lambda \in \{0.5, 1.0, 2.0\}$. Anti-collapse is achieved through four mechanisms: (i) the slowly evolving EMA target provides non-trivial targets; (ii) stop-gradient blocks the degenerate gradient path; (iii) the predictor $g_\phi$ introduces an asymmetric fixed point; and (iv) the simultaneously optimized diffusion denoising loss constrains token-level output distributions, preventing the backbone from collapsing into a constant mapping. Empirically, the effective rank of pooled embeddings remains 42–44 (base model 42–43) with a cosine diversity of 0.25–0.28, matching the baseline.
    *   **Design Motivation**: While cosine-only JEPA is prone to collapse in vision, this work uses the diffusion primary task as an additional anchor, turning "anti-collapse" from an architectural trick into a natural property of the task-supervised setting.

### Loss & Training
The total objective is $\mathcal{L}_\text{total}=\mathcal{L}_\text{diff}+\lambda\,\mathcal{L}_\text{JEPA}$, where $\mathcal{L}_\text{diff}=\mathbb{E}_{t,x_t}[-\frac{1}{|\mathcal{M}_t|}\sum_{i\in\mathcal{M}_t}\log p_\theta(x_0^i|x_t)]$ is the standard masked diffusion cross-entropy. $\mathcal{L}_\text{JEPA}=1-\cos(\text{sg}(z_{t_H}), g_\phi(z_{t_L}))$. Training uses AdamW on 8×A100-80G with gradient checkpointing for 2 epochs of full-parameter fine-tuning. Main experiments use lr=$1\times 10^{-5}$ and $(t_L,t_H)=(0.2,0.7)$, while base-preservation experiments use a gentler "Wide-tt" configuration with lr=$1.4\times 10^{-6}$ and $(0.1,0.9)$. Hyperparameters include $\lambda \in \{0.5, 1, 2\}$, $k \in \{1, \dots, 5\}$, and EMA $\tau=0.996$.

## Key Experimental Results

### Main Results
Evaluated on 4 tasks x 2 backbones (LLaDA-8B, Dream-7B) using a 4-shot protocol, selecting the optimal $(\lambda, k)$ for each cell.

| Task | Metric (4-shot) | LLaDA-8B BL→JEPA | Δ | Dream-7B BL→JEPA | Δ |
|------|---------------|-------------------|------|-------------------|------|
| GSM8K | accuracy | 42.61 → 61.33 | **+18.73** | 34.87 → 46.25 | +11.38 |
| NL-RX | func match | 47.50 → 58.20 | +10.70 | 42.00 → 46.80 | +4.80 |
| Spider | exec match | 35.40 → 39.36 | +3.97 | 20.89 → 25.15 | +4.26 |
| Django | ws-prefix match | 74.40 → 75.40 | +1.00 | 69.58 → 72.35 | +2.77 |

For LLaDA-8B on GSM8K (Wide-tt, 3-seed avg): baseline 65.23±0.93 → DLLM-JEPA 67.07±0.41 (+1.84 pp, variance halved).

### Base Preservation (Table 3, LLaDA-8B GSM8K, Wide-tt)

| Method | GSM8K 0-shot | Wikitext Δloss (vs base) |
|------|--------------|--------------------------|
| Base (No fine-tune) | – | 0.0000 |
| Diffusion Baseline ($\lambda=0$) | 65.23 ± 0.93 | −0.0004 |
| L2-to-base anchor ($\lambda_{L2}=10^{-4}$) | 65.18 ± 0.87 | −0.0007 ± 0.0002 |
| **DLLM-JEPA (Ours)** | **67.07 ± 0.41** | **−0.0017** |

DLLM-JEPA is the only method that achieves **both** task gains and a lower Wikitext loss than the base model. L2 anchors suppress parameter drift but yield zero task gains, suggesting base preservation requires more than just parameter distance regularization.

### Comparison of Computation (Table 1, FLOPs per step)

| Method | Fwd (grad) | Fwd (no grad) | Backward | Total | Overhead |
|------|------------|---------------|----------|-------|----------|
| AR Baseline | 1F | – | ≈2F | 3F | – |
| LLM-JEPA | 2F | – | ≈4F | 6F | +100% |
| Diffusion Baseline | 1F | – | ≈2F | 3F | – |
| **DLLM-JEPA** | **1F** | **1F** | **≈2F** | **4F** | **+33%** |

### Key Findings
*   **Geometric drift vs. functional forgetting dissociation**: Models trained with DLLM-JEPA exhibit **larger** hidden-state drift relative to pre-trained initialization (1.3–3.6× the baseline, concentrated in middle transformer layers), yet experience **less** functional forgetting (43–58%) on Wikitext. This suggests that the JEPA objective **redirects** rather than minimizes representation change.
*   **Variance reduction**: On high-variance cells (e.g., LLaDA-8B GSM8K with baseline seed spread of ±8.9 pp), DLLM-JEPA reduces variance to ±3.9 pp, with best-seed gains reaching +18.7 pp.
*   **Anti-collapse verification**: Post-fine-tuning, pooled embeddings maintain an effective rank of 42–44 and cosine diversity of 0.25–0.28, confirming that the diffusion objective prevents the collapse of cosine-only JEPA.
*   **Architectural focus**: The authors position LLM-JEPA as a structural motivation rather than a direct head-to-head competitor, as the underlying attention substrates (causal vs. bidirectional) differ.

## Highlights & Insights
*   The perspective of **"diffusion noise = natural data augmentation"** is elegant. By borrowing stochastic masks for view generation, the method avoids extra data costs and integrates seamlessly with the noise schedule. This concept can potentially extend to any generative model with random corruption (audio, code, or graph diffusion).
*   The **computational accounting** (Table 1) is exceptionally clear. It demonstrates that JEPA's perceived cost in LLMs is not inherent to the objective but is forced by AR causal masks. Shifting to bidirectional substrates reduces this overhead to a single no-grad forward pass.
*   The **dissociation between drift and forgetting** is a counter-intuitive and profound empirical finding. While traditional paradigms (EWC, L2-to-base) assume "less movement = less forgetting," this work provides a counterexample where "more (but redirected) movement" preserves knowledge better, opening new windows for research into fine-tuning dynamics.

## Limitations & Future Work
*   **Baseline comparison**: The primary head-to-head is against a diffusion-only baseline. Direct evidence of how an AR model would perform given a similar 33% overhead is missing.
*   **Scale of verification**: Testing is limited to two backbones (LLaDA-8B, Dream-7B) and 4 small-scale tasks with 2-epoch SFT. The claim that "JEPA improves representation" has not yet been verified in a from-scratch pre-training setting.
*   **Hyperparameter sensitivity**: The masking ratios $(t_L, t_H)$ are fixed. There is a lack of systematic sensitivity analysis or theoretical guidance for these ratios.
*   **Mechanistic explanation**: The drift-forgetting dissociation is reported descriptively. A causal explanation (e.g., why middle-layer drift is beneficial) would strengthen the findings.
*   **Magnitude of Wikitext gains**: The Wikitext Δloss is at the $10^{-3}$ level. While positive, the claim of "base preservation" should be interpreted alongside task gains.

## Related Work & Insights
*   **vs. LLM-JEPA (Huang et al., 2025)**: LLM-JEPA requires paired text-code data and doubles gradient computation. DLLM-JEPA uses diffusion noise on any text data with a single gradient pass, reducing overhead from 100% to 33% by leveraging the appropriate substrate.
*   **vs. I-JEPA / V-JEPA (Assran et al., 2023; Bardes et al., 2024)**: This work successfully ports the EMA/stop-gradient/predictor recipe from vision to language by replacing image patches with masked tokens.
*   **vs. LLaDA / MDLM / SEDD**: This is not a new architecture but a plug-and-play representation loss that can be applied to any existing masked diffusion LM fine-tuning pipeline.
*   **vs. EWC / L2-to-base**: Instead of constraining parameter space (parameter distance), this work constrains latent space (representation alignment), which Table 3 suggests is superior for maintaining base performance without sacrificing task gains.

## Rating
*   **Novelty**: ⭐⭐⭐⭐ Using a diffusion noise schedule as a JEPA view generator is a clever insight, though individual components (JEPA, diffusion LM) are established.
*   **Experimental Thoroughness**: ⭐⭐⭐ Covers 2 backbones and 4 tasks with preservation analysis, but lacks from-scratch pre-training to fully support "representation learning" claims.
*   **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely responsible reporting of comparison protocols, computational costs, and empirical findings.
*   **Value**: ⭐⭐⭐⭐ Provides a low-cost (+33% FLOPs) plug-in for diffusion LM fine-tuning and reveals counter-intuitive phenomena regarding representation drift.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](../../ACL2026/interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[NeurIPS 2025\] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement](../../NeurIPS2025/interpretability/far_from_the_shallow_brain-predictive_reasoning_embedding_through_residual_disen.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](towards_atoms_of_large_language_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICML 2026\] Prototype Transformer: Towards Language Model Architectures Interpretable by Design](prototype_transformer_towards_language_model_architectures_interpretable_by_desi.md)
- [\[ACL 2026\] Towards Intrinsic Interpretability of Large Language Models: A Survey of Design Principles and Architectures](../../ACL2026/interpretability/towards_intrinsic_interpretability_of_large_language_modelsa_survey_of_design_pr.md)
- [\[NeurIPS 2025\] Far from the Shallow: Brain-Predictive Reasoning Embedding through Residual Disentanglement](../../NeurIPS2025/interpretability/far_from_the_shallow_brain-predictive_reasoning_embedding_through_residual_disen.md)
- [\[ICML 2026\] Query Circuits: Explaining How Language Models Answer User Prompts](query_circuits_explaining_how_language_models_answer_user_prompts.md)
- [\[ICML 2026\] Towards Atoms of Large Language Models](towards_atoms_of_large_language_models.md)

</div>

<!-- RELATED:END -->
