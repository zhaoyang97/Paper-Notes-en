---
title: >-
  [Paper Note] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner
description: >-
  [ICML 2026][Image Restoration][Diffusion LM] This paper systematically compares continuous diffusion, discrete masked diffusion, and looped transformers across the dimensions of expressivity and trainability. It proves that continuous diffusion is strictly more expressive than discrete diffusion and can simulate looped transformers, but its practical performance
tags:
  - ICML 2026
  - Image Restoration
  - Diffusion LM
  - Latent Reasoning
  - Looped Transformer
  - CFG
date: 2026-05-08
content_hash: 3ef8e86a6ddd5d88
---
# Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner

**Conference**: ICML 2026  
**arXiv**: [2510.03206](https://arxiv.org/abs/2510.03206)  
**Code**: https://github.com/zhouc20/CCDD (Available)  
**Area**: Diffusion Language Models / Latent Reasoning / Multimodal Diffusion  
**Keywords**: Diffusion LM, Latent Reasoning, Joint Continuous-Discrete Diffusion, Looped Transformer, CFG

## TL;DR
This paper systematically compares continuous diffusion, discrete masked diffusion, and looped transformers across the dimensions of expressivity and trainability. It proves that continuous diffusion is strictly more expressive than discrete diffusion and can simulate looped transformers, but its practical performance is hindered by decoding and representation space issues. Consequently, it proposes **CCDD (Coevolutionary Continuous Discrete Diffusion)**—a framework that diffuses simultaneously in the discrete token space and the contextual embedding space of a pre-trained LLM, jointly denoised by a single model. CCDD reduces perplexity by 25-35% compared to MDLM on LM1B/OWT and outperforms MDLM at 256 steps with only 8 sampling steps.

## Background & Motivation
**Background**: Language modeling is currently dominated by autoregressive LLMs. Non-autoregressive approaches branch into two directions: continuous diffusion language models (CDM, using SDE/PF-ODE—early but weak) and discrete diffusion language models (DDM, especially masked diffusion like MDLM/SEDD—recently surpassing CDM). Simultaneously, there is a "latent reasoning" path: looped transformers (LT) and continuous CoT, which theoretically break the expressivity upper bound of transformers in $\mathsf{TC^0}$.

**Limitations of Prior Work**: (1) While LT is theoretically powerful, it lacks intermediate supervision, and the rollout depth often deviates significantly from training, leading to severe OOD issues. (2) CDM should theoretically be stronger but is practically outperformed by DDM; the authors attribute this to the "trainability trio" of a massive decision space, poor embedding space, and complex decoding combinations. (3) While masked DDM is trainable, quantizing logits into tokens at each step loses uncertainty memory across steps and sacrifices self-correction capabilities.

**Key Challenge**: The fundamental trade-off between **expressivity upper bounds and practical trainability**. Continuous representations preserve full information beneficial for reasoning but are hard to train and decode; discrete representations have clear training objectives but suffer from information bottlenecks.

**Goal**: To construct a unified framework that retains (a) the high expressivity of continuous CDM (covering LT), (b) the superior trainability of discrete DDM, (c) the semantic priors of pre-trained LLM embeddings, and (d) flexible NFE sampling.

**Key Insight**: Redefine "language diffusion" on a **joint multimodal space** $\mathcal{X} \times \mathcal{Z}$—where discrete tokens provide an easily decodable "skeleton" and pre-trained LLM contextual embeddings provide the smooth, information-rich "substance." Two sets of noise are injected in parallel, and a single network performs joint denoising.

**Core Idea**: Language modeling via a joint CTMC×SDE process of "discrete token diffusion + continuous contextual embedding diffusion," where the continuous part handles latent reasoning memory across steps and the discrete part ensures high-confidence decoding.

## Method

### Overall Architecture
CCDD addresses the conflict where continuous diffusion is most expressive but hardest to train by moving language diffusion to a joint multimodal space $\mathcal{X} \times \mathcal{Z}$. Discrete tokens $x$ provide the "skeleton" for supervision and decoding, while continuous contextual embeddings $z$ provide the "substance" for preserving probability history across steps. During the forward process, independent noise is injected into clean data $(x_0, z_0)$: $x_t \sim \text{Cat}(\eta_t x_0 + (1-\eta_t)\pi_t)$ follows a masked/uniform CTMC, and $z_t \sim \mathcal{N}(\alpha_t z_0, \sigma_t^2 I)$ follows a VP-SDE. During the reverse process, a single network $f_\theta(x_t, z_t, t)$ intakes both noisy states to predict token logits and embedding $\hat{x}_{0,\theta}, \hat{z}_{0,\theta}$. Updates are then performed according to their respective modalities (DDPM/DDIM for $z$, and Bayesian posterior for $x$). The training objective is a weighted sum of the two ELBOs: $\mathcal{L}_{\text{CCDD}} = \gamma_{\text{cont}} \mathcal{L}_{\text{cont}} + \gamma_{\text{disc}} \mathcal{L}_{\text{disc}}$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    X0["Clean token x₀"]
    Z0["Continuous space z₀: Pre-trained LLM contextual embeddings<br/>(Frozen Qwen3-Embedding)"]
    X0 --> XT["Noisy discrete xₜ (masked/uniform CTMC)"]
    Z0 --> ZT["Noisy continuous zₜ (VP-SDE)"]
    XT --> F["Joint Denoising Network f_θ(xₜ,zₜ,t)<br/>Architecture: MDiT / MMDiT / MoEDiT"]
    ZT --> F
    F --> PRED["Simultaneous prediction of token logits and ẑ₀"]
    PRED -->|"Representation-guided CFG: w·logits_c+(1−w)·logits_φ"| UPD["Reverse Joint Update<br/>x via Bayes posterior, z via DDPM/DDIM"]
    UPD --> OUT["Generated Text"]
```

### Key Designs

**1. Joint Continuous-Discrete Diffusion Process: Running CTMC and SDE in One Network**

To address the pain points of DDM (losing uncertainty memory by quantizing logits every step) and CDM (combinatorial explosion when decoding from continuous space), CCDD designs the forward process as a fully separable product $q_t(x_t,z_t|x_0,z_0) = q_t^{\text{disc}}(x_t|x_0)\, q_t^{\text{cont}}(z_t|z_0)$ for simplicity. However, the reverse process is defined as $p_\theta(x_s,z_s|x_t,z_t) = p_\theta^{\text{disc}}(x_s|x_t,z_t)\, p_\theta^{\text{cont}}(z_s|x_t,z_t)$—where each factor depends on both inputs (Remark 4.1). This "forward independence + reverse conditional coupling" is proven to be asymptotically equivalent to a fully coupled reverse kernel as step size $\to 0$ (Theorem B.19), yet it simplifies parameterization. This allows the continuous path to handle "memory and planning"—retaining logit geometry rather than quantizing (Lemma B.9 proves DDM's "logits→sample→embed" is a hard information bottleneck)—while the discrete path provides high-confidence decoding.

**2. Using Pre-trained LLM Contextual Embeddings as Continuous Space**

The authors trace CDM's failure to "poor embedding space." Instead of learning new embeddings, $z_0$ is taken from the **frozen hidden states of Qwen3-Embedding-0.6B** (hidden dim 32, normalized). Figure 2 compares layer 0 (near token-wise lookup) and layer 28 (fully contextualized). Layer 0 has the lowest cross-entropy reconstruction (easy to decode) but highest MSE (hard to generate); layer 28 is the opposite. Middle layers (12th, 20th) strike a balance. Proposition E.1 proves that token-wise embedding dimensions $d \le V$ do not exceed the expressivity of a simplex and are unfriendly to CDM. Contextual embeddings provide a smooth generation target with pre-trained semantic priors, acting as a built-in "proxy representation guidance"—which allows CCDD to match MDLM's 1000k-step PPL in just 40k steps, a 25× training speedup.

**3. Representation-Guided Classifier-Free Guidance and Three Multimodal Architectures**

To adjust the influence of continuous $z$ during inference, CCDD treats it as a "self-generated representation condition" for CFG. During training, $z_t$ is zeroed out with probability $p_{\text{drop}}$, enabling the model to learn both conditional ($z$ present) and unconditional ($z$ zero) forwards. During sampling, logits are mixed as $\text{logits} = w\cdot\text{logits}_c + (1-w)\cdot\text{logits}_\phi$. Higher $w$ strengthens the continuous reasoning component (ablation shows $w=1.5$ reduces Gen NLL from 9.06 to 8.25). Three architectures are proposed: **MDiT** (additive embeddings in DiT, 0 extra parameters), **MMDiT** (dual-stream cross-attention, best performance), and **MoEDiT** (routing modalities to different experts, optimal FLOP efficiency).

### Loss & Training
The loss is a weighted sum of ELBOs for both modalities, using $x_0$-prediction parameterization. The network adapts SEDD's DiT with rotary embeddings; hidden dim is set to 32 to align with Qwen3-Embedding. For LM1B (seq len 128) and OWT (seq len 512), models were trained for 1M steps with a batch size of 512 (approx. 33B / 131B tokens). Since PPL is not directly comparable between Qwen-2 and GPT-2 tokenizers, all baselines were retrained using Qwen-2 for fair comparison.

## Key Experimental Results

### Main Results
PPL comparison on LM1B and OWT, with parameters aligned to the 92.1M MDLM baseline:

| Dataset | Model | Params | Training Tokens | Val PPL ↓ | Relative to MDLM |
|--------|------|------|------------|-----------|-----------|
| LM1B | MDLM (reimpl.) | 92.1M | 33B | ≤39.17 | — |
| LM1B | **CCDD-MDiT w/ Qwen3** | 92.1M | 33B | ≤29.22 | **-25.4%** |
| LM1B | CCDD-MoEDiT w/ Qwen3 | 104M | 33B | ≤28.50 | -27.2% |
| LM1B | CCDD-MMDiT w/ Qwen3 | 216M | 33B | ≤25.76 | -34.2% |
| OWT (Qwen-2) | MDLM (reimpl.) | 92.1M | 131B | ≤33.78 | — |
| OWT (Qwen-2) | CCDD-MoEDiT w/ Qwen3 | 104M | 131B | ≤21.90 | **-35.2%** |
| OWT (GPT-2) | MDLM (reimpl.) | 92.1M | 131B | ≤27.39 | — |
| OWT (GPT-2) | CCDD-MoEDiT w/ RoBERTa | 104M | 131B | ≤24.56 | -10.3% |
| OWT (GPT-2) | GIDD+ (reimpl.) | 92.1M | 131B | ≤25.82 | -5.7% |

Comparison on complex reasoning tasks using 6M small models:

| Task | GPT2 (6M) | Llama-7B | MDM (20 steps) | LT (2 layers) | LT (3 layers) | **CCDD (2 steps)** | **CCDD (3 steps)** |
|------|----------|----------|-----------|----------|----------|---------------|---------------|
| Sudoku | 16.2 | 27.1 | 99.9 | 100.0 | 100.0 | **100.0** | **100.0** |
| 3-SAT | 73.1 | — | 87.0 | 91.3 | — | **91.9** | — |
| Countdown | 31.9 | 41.1 | 52.0 | 60.6 | 68.2 | **67.8** | **73.7** |

### Ablation Study

| Configuration | Val PPL / Metric | Note |
|------|---------------|------|
| Qwen3-Embedding layer 0 (token-wise) | Min token CE, Max rep MSE | Easy to decode, hard to generate |
| Qwen3-Embedding layer 28 (contextualized) | Max token CE, Min rep MSE | Easy to generate, requires discrete branch |
| Qwen3-Embedding internal layer | Balanced loss | Chosen configuration |
| CCDD w=0 (joint) | Gen NLL 9.06 | Already better than MDLM (9.19) |
| CCDD w=1 (discrete-only forward) | Gen NLL 8.38 | CFG provides significant boost |
| CCDD w=1.5 | Gen NLL 8.25 | Guidance further improves results |
| CCDD 8-step sampling | Better than MDLM 256 steps | **16× sampling speedup** |

### Key Findings
- **Disruptive Advantage in Few-step Sampling**: CCDD at 8 steps outperforms MDLM at 256 steps—a direct dividend of the continuous component's ability to model joint distributions and support ODE sampling, whereas DDM is limited to SDE sampling requiring higher NFE.
- **25× Training Efficiency**: On LM1B, CCDD reaches MDLM’s 1000k-step PPL in just 40k steps, demonstrating the powerful representation regularization provided by pre-trained LLM embeddings.
- **CCDD 2 steps ≈ LT Max Depth on Reasoning Tasks**: CCDD saturates Sudoku/3-SAT in 2 steps and surpasses LT 3-layer scores on Countdown in 3 steps, validating the hypothesis that the continuous path handles cross-step reasoning.
- **Architectural Sensitivity**: MDiT (zero extra parameters) already yields a 25% PPL drop, suggesting the gains stem from the joint diffusion design rather than parameter count; MMDiT/MoEDiT provide further improvements.

## Highlights & Insights
- **Unified Perspective**: The theoretical conclusions "CDM ⊋ DDM" and "CDM simulates LT" place continuous diffusion, discrete diffusion, and looped transformers on a single expressivity ladder, clearly identifying continuous as the goal.
- **Diagnosis of Trainability**: Breaking down CDM’s failure into three factors (decision space, embedding quality, decoding complexity) provides a clean logical chain for using pre-trained embeddings and discrete branches as solutions.
- **CFG-as-representation-guidance**: The fusion of continuous representation and Classifier-Free Guidance (random zeroing during training, strengthening during inference) is a paradigm that can be transferred to other tasks like code-AST or molecule-graph generation.
- **8 Steps vs. 256 Steps**: This is arguably more impactful than PPL gains; CCDD provides a systematic path to overcoming the sampling bottleneck of diffusion LMs by utilizing more expressive continuous paths rather than just new samplers.
- **Tight Theory-Experiment Coupling**: The paper forms a complete loop from the "why" (Theorem 3.2) to the "how" (Figure 2) and the validation (Table 6), making it a highly self-consistent work in diffusion language modeling.

## Limitations & Future Work
- **Dependency on External Pre-trained Embeddings**: Performance is tied to the quality of the encoder (e.g., Qwen3). Switching to weaker encoders (RoBERTa) drops the gain from 35% to ~10%. effectiveness may diminish in niche languages or domains lacking pre-trained embedders.
- **Model Scale**: Experiments are limited to the 92M-216M range on 1B-level datasets; scaling laws for 7B+ parameters remain unexplored.
- **Overhead on Long Sequences**: While efficient, the joint input and CFG require dual forward passes, making the per-step cost roughly 2×. No end-to-end wall-clock comparison with AR LLMs at the same FLOP budget is provided.
- **Loss of Discrete Self-Correction**: Because CCDD uses a masked discrete process (like MDLM), it inherits the lack of self-correction; the potential for using uniform DDM with continuous paths to regain this was not discussed.

## Related Work & Insights
- **vs. MDLM / SEDD (Masked DDM)**: Proves these are strictly less expressive than CDM; adding a continuous branch breaks the upper bound while maintaining trainability.
- **vs. Continuous DLM (Score Diffusion)**: Identifies that CDM’s failure was due to the embedding space, not the theory, and proposes pre-trained LLM embeddings as a fix.
- **vs. Looped / Universal Transformer**: Since CDM simulates LT and naturally provides intermediate supervision, the authors suggest CDM as a superior alternative for latent reasoning.
- **vs. DiT / MM-DiT / MoE**: CCDD successfully migrates visual diffusion architectures to language diffusion with significant results.
- **vs. REPA / RCG**: Porting the idea of using pre-trained encoder representations as diffusion guidance from vision to language.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Joint CTMC × SDE is a paradigm-shifting structure that unifies multiple research paths.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive datasets and architectures compared, though missing large-scale scaling and wall-clock benchmarks.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely high logical consistency from motivation to theory and empirical results.
- Value: ⭐⭐⭐⭐⭐ Provides a viable path for diffusion LMs to compete with AR LLMs in reasoning, with significant practical value in sampling acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICML 2026\] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models](plan_for_speed_dilated_scheduling_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] PODiff: Latent Diffusion in Proper Orthogonal Decomposition Space for Scientific Super-Resolution](podiff_latent_diffusion_in_proper_orthogonal_decomposition_space_for_scientific_.md)
- [\[CVPR 2026\] Language-Guided One-Step Diffusion Model for Nighttime Flare Removal](../../CVPR2026/image_restoration/language-guided_one-step_diffusion_model_for_nighttime_flare_removal.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)

</div>

<!-- RELATED:END -->
