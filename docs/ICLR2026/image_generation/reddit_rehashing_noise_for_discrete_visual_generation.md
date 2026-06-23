---
title: >-
  [Paper Note] ReDDiT: Rehashing Noise for Discrete Visual Generation
description: >-
  [ICLR 2026][Image Generation][Paper Note] ReDDiT extends the single `[mask]` absorbing state in discrete diffusion to a set of random multi-index absorbing states (rehashing noise). It employs a rehash sampler utilizing `torch.multinomial` for low-discrepancy sampling, replacing the Gumbel-max-based remask heuristics in MVTM. This approach reduces the gFID on
tags:
  - ICLR 2026
  - Image Generation
date: 2026-05-08
content_hash: 65dcda074ca439ac
---
# ReDDiT: Rehashing Noise for Discrete Visual Generation

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=7R8ohzWB4i](https://openreview.net/forum?id=7R8ohzWB4i)  
**Code**: https://github.com/martian422/ReDDiT  
**Area**: Diffusion Models / Image Generation / Discrete Diffusion  
**Keywords**: Discrete Diffusion, Absorbing State Noise, Rehash Sampling, Multi-index Corruption, ImageNet Generation

## TL;DR
ReDDiT extends the single `[mask]` absorbing state in discrete diffusion to a set of random multi-index absorbing states (rehashing noise). It employs a rehash sampler utilizing `torch.multinomial` for low-discrepancy sampling, replacing the Gumbel-max-based remask heuristics in MVTM. This approach reduces the gFID on ImageNet-256 from a baseline of 6.18 to 1.61, marking the first time discrete diffusion matches continuous diffusion in generation quality.

## Background & Motivation

**Background**: Diffusion Transformers (DiT) in the continuous domain have achieved significant quality and scalability by iteratively refining image latents from Gaussian noise. Recently, the community has shown interest in **discrete diffusion** due to two practical benefits: indexable codebooks naturally compatible with Language Models, and the ability to predict multiple tokens in parallel per step, leading to efficient inference. The dominant discrete scheme is the masked visual token model (MVTM, e.g., MaskGIT), which uses BERT-style `[mask]` tokens to corrupt image token sequences and performs maximum likelihood prediction at masked positions via cross-entropy.

**Limitations of Prior Work**: The generation quality of discrete methods has consistently lagged behind continuous methods. The authors attribute this to two factors. First, **noise (absorbing state) design**: MVTM collapses all masked tokens into a single absorbing state `[mask]`. Compared to Gaussian noise, this lacks vocabulary richness and latent diversity, providing coarse signals to the model and limiting its ability to express complex distributions. Furthermore, discrete de-masking is binary—tokens are either masked or deterministically decoded, unlike continuous diffusion which injects randomness at every step. Second, **sampling heuristics**: MVTM’s confidence-based remask sampler relies on Gumbel-max to induce manual randomness as a proxy for sampling diversity. This compromises probabilistic fidelity and requires careful balancing of the number of tokens decoded per step to prevent error accumulation, leading to redundant sampling rounds. Gumbel-max thus becomes a trick requiring per-step tuning and exhibits numerical instability and poor performance under large vocabularies (e.g., 16,384).

**Key Challenge**: The gap between discrete and continuous methods is driven not by quantization itself, but by the **insufficient expressiveness of a single absorbing state** coupled with the **low fidelity of the Gumbel-max remask sampler**.

**Goal**: ① Redesign the absorbing state to allow for richer paths for latents during diffusion; ② Design a principled, low-discrepancy sampler that does not rely on hyperparameter-dependent randomness.

**Key Insight**: Continuous diffusion injects random noise at every step, whereas discrete diffusion uses a rigid, single `[mask]`. By expanding the absorbing state into a set of indices and randomizing it during corruption, the model can optimize its embedding space during training and learn data-driven noise structures.

**Core Idea**: Replace "single mask + Gumbel-max remask" with "multi-index random absorbing states + a multinomial sampler that inverts this random path." This achieves both high diversity and low discrepancy without tuning randomness.

## Method

### Overall Architecture
ReDDiT is based on the DiT architecture and operates on discrete tokens (defaulting to the IBQ-f16 tokenizer, 256×256 images → 256 tokens, codebook size 16,384). It reformulates the forward corruption of discrete diffusion: tokens transition from a valid embedding subspace $E_d$ to an **absorbing state subspace** $M_m$ with capacity $m$ (no longer a single point $\{m\}$). During training, corrupted data is fed with a distribution $x_t \sim \mathrm{Cat}(x_t; \alpha_t x_0 + (1-\alpha_t)\,U(M_m^L))$, where $U(M_m^L)$ is a uniform distribution over the absorbing states and $\alpha_t$ is a monotonically decreasing survival function ($\alpha_0=1, \alpha_1=0$). The training objective follows a linear ELBO (Eq.4) derived from DDM, supplemented by a RepA representation alignment regularization. During inference, the rehash sampler is used: each step first reshuffles (rehashes) tokens currently in the absorbing state, then performs low-discrepancy sampling via `torch.multinomial` based on softmax probabilities to gradually decode the fully masked sequence $x_1$ back to a clean sequence $x_0$.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Image → Tokenizer<br/>Discrete Token Sequence x0"] --> B["Multi-index Random Absorbing States<br/>Ed → Mm Corruption/Noising"]
    B --> C["DiT Denoising Prediction pθ(xt,c)<br/>DDM-linear ELBO + RepA Alignment"]
    C --> D["Rehash Sampler<br/>Mask Shuffling + Multinomial Low-Discrepancy Sampling"]
    D -->|Stepwise Denoising (K steps)| C
    D --> E["Clean Token x0 → Decoded Image"]
```

### Key Designs

**1. Rehashing noise: Expanding a single absorbing state into a set of random multi-index absorbing states**

To address the bottleneck of "coarse single `[mask]` signals," the authors expand the absorbing state from a single point to a set $M_m=\{m_j\}_{j=1}^m$ of capacity $m$. This set is concatenated with the valid token base $E=\{e_i\}_{i=1}^d$ to form a larger categorical space $V_{(d,m)}\in\mathbb{R}^{d+m}$. The forward transition kernel is modified: a valid token is absorbed with probability $1-\alpha_{t|s}$ and remains unchanged with probability $\alpha_{t|s}$. Crucially, **tokens already in an absorbing state will jump randomly between the $m$ absorbing indices with probability $1/m$**—this is "rehashing." This allows latents to traverse more potential paths during the diffusion process. The motivation is that while early work in the pixel domain (Austin, Campbell) proved that higher transition probabilities for adjacent pixel values (Gaussian-like discrete noise) outperform single masks, visual tokenizer latents have **learned structures** without inherent ordinality. Multi-index absorbing states allow the model to optimize its embedding space during training and learn data-driven noise distributions (visualized via t-SNE for $m=1/16/128/1024$). The optimal capacity $m$ is determined empirically: $m=128$ for LlamaGen-f16 and $m=1024$ for IBQ, which the authors attribute to lower-dimensional codebooks producing more compact latents with less noise tolerance.

**2. Rehash sampler: Inverting random absorbing paths with multinomial low-discrepancy sampling**

To solve the "low fidelity of Gumbel-max remask samplers and the need for per-step randomness tuning," the authors derive the reverse process $q_{s|t}$ from discrete diffusion theory (Eq.9): tokens not currently masked remain unchanged; tokens in an absorbing state either remain absorbed with probability $\tfrac{1-\alpha_s}{m(1-\alpha_t)}$ or are decoded into a valid token with probability $\tfrac{\alpha_s-\alpha_t}{1-\alpha_t}\,\delta_i p_\theta(x_t)$. The algorithm consists of three key actions: ① **Rehash operation**—at the start of each step, currently absorbed tokens are uniformly re-sampled $x_t \leftarrow \mathrm{where}(x_t\in M_m, U(M_m^L), x_t)$ to create path diversity; ② Use **softmax probabilities** (instead of logits+Gumbel) to calculate $q_{s|t}$, explicitly merging the absorbing state probability $\delta_{m[0]}\cdot\tfrac{1-\alpha_s}{1-\alpha_t}$ to preserve total noise sampling probability (preventing tiny values from being truncated and damaging accuracy); ③ Use `torch.multinomial` for **low-discrepancy** categorical sampling instead of Gumbel-max. Unlike MaskGIT’s "predict-all then remask," the rehash sampler includes the absorbing state in the categorical sampling and **decouples training from inference**, allowing sampling on any discretized timescale (cosine schedule is optimal) rather than being locked to the training configuration. Theoretically, Gumbel-max is equivalent to this method, but it is numerically inaccurate under finite samples and large vocabularies, failing to reflect the true distribution—this was the bottleneck of MVTM.

**3. DDM-linear objective + RepA: Replacing MVTM cross-entropy and accelerating convergence**

MVTM objectives borrowed from masked language modeling only perform maximum likelihood at masked positions, which lacks strong theoretical foundations for diffusion. ReDDiT utilizes the linear ELBO objective derived from DDM: $L_{\text{DDM-linear}}=-\mathbb{E}_{t,x_0,x_t}[\tfrac1t\sum_i \delta(x_t^i,m)\log p_\theta(x_0^i|x_t)]$ (Eq.4). Ablations show that simply switching the objective reduces gFID from 6.83 to 6.23. On top of this, RepA (Representation Alignment) is added: middle features $h^{[n]}(x_t)$ from the 8th layer of DiT are projected via a small MLP $h_\phi$ and aligned using element-wise cosine similarity with features from the original image encoded by dinov2-b, $L_{\text{total}}=L_{\text{DDM-linear}}+\lambda L_{\text{RepA}}$ ($\lambda=0.5$). The authors verify for the first time that this technique, originally proposed for continuous diffusion, works for discrete latents. However, they honestly note that RepA primarily **accelerates convergence** and does not provide relative performance gains after sufficient training; they use it for efficiency and to probe internal training dynamics via $L_{\text{RepA}}$. Additionally, as it shares progressive decoding traits with Discrete Flow Matching (DFM), several DFM steps can be inserted into the sampling for refinement, gaining $\sim$0.1 gFID on ImageNet.

### Loss & Training
The total loss is $L_{\text{total}}=L_{\text{DDM-linear}}+\lambda L_{\text{RepA}}$ ($\lambda=0.5$). Training uses AdamW with cosine decay, 2D-RoPE, and min-SNR for efficiency. Class-conditional training uses class embeddings with a 0.1 drop-rate to support CFG. Pre-processing follows the ten-crop augmentation from LlamaGen. A cosine discrete timescale is used by default for sampling.

## Key Experimental Results

### Main Results
ImageNet-1K 256×256; gFID↓ / IS↑ calculated on 50k samples.

| Type | Model | gFID↓ | IS↑ | #Params | #Steps |
|------|------|------|-----|---------|--------|
| Continuous Diffusion | DiT-XL/2 | 2.27 | 278.2 | 675M | 250 |
| Continuous Diffusion | MDTv2 | 1.58 | 314.7 | 676M | 256 |
| MVTM | MaskGIT (Baseline) | 6.18 | 182.1 | 227M | 8 |
| MVTM | TiTok-S-128 ft. | 1.97 | 281.8 | 287M | 64 |
| DDM | ITM | 5.30 | 183.0 | 546M | 100 |
| Ours | ReDDiT-L | 2.13 | 294.7 | 346M | 20 |
| Ours | ReDDiT-XL | 1.74 | 313.6 | 675M | 32 |
| Ours | ReDDiT-XLf8 | **1.61** | 318.5 | 675M | 64 |

ReDDiT achieves the best performance among discrete models, improving gFID from 6.18 to 2.13 and IS from 182.1 to 294.7 relative to the MaskGIT baseline. It matches continuous diffusion while maintaining the efficiency of discrete methods (significantly fewer inference steps than AR and traditional diffusion). Compared to models using the same tokenizer (Tab. 2), ReDDiT-L with IBQ (gFID 2.13) outperforms LlamaGen-LAR (3.80), RandAR (2.55), and IBQ-BAR (2.88).

### Ablation Study

| Configuration (Training → Sampling) | gFID | Prec. | Rec. |
|------|------|------|------|
| MVTM + RepA → MVTM Sampler | 6.83 | 0.75 | 0.39 |
| Switch to DDM Objective (Eq.11) → MVTM Sampler | 6.23 | 0.77 | 0.41 |
| As above → Rehash Sampler | 5.75 | 0.78 | 0.45 |
| + 2D-RoPE + min-SNR → Rehash | 5.51 | 0.79 | 0.45 |
| + DFM Refine | 5.40 | 0.81 | 0.52 |

| Rehash Operation Ablation (m, 100k iters) | gFID |
|------|------|
| m=1 (Baseline) | 4.13 |
| m=128 (Fixed absorbing states) | 4.25 |
| m=128 (No rehash) | 4.07 |
| m=128 (Full rehash) | **3.92** |

### Key Findings
- Replacing the MVTM cross-entropy objective with DDM-linear ELBO improves gFID from 6.83 to 6.23. Switching to the rehash sampler further improves it to 5.75. The combined gain is approximately ∼1.0 and is complementary to techniques like 2D-RoPE/min-SNR.
- The rehash operation is critical: simply increasing capacity from $m=1$ to $m=128$ with fixed absorbing states degrades performance (4.13→4.25). Only enabling random initialization (4.07) and especially **full rehash active resampling** (3.92) unlocks model capacity—proving that "active resampling to prevent over-deterministic sampling" is necessary.
- Timescale schedule has a significant impact: at 20 steps, non-linear schedules like cosine (4.91) and arccos (5.04) (which are slower initially and faster later) are superior to linear (7.18) and square (7.39). Decoupling training and inference allows for free selection of the sampling timescale.
- Optimal noise capacity $m$ is tokenizer-dependent: LlamaGen-f16 peaks at $m=128$, while IBQ peaks at $m=1024$.

## Highlights & Insights
- The authors explicitly decouple the lag in discrete diffusion from "quantization itself" and reallocate it to **absorbing state design + sampling heuristics**, providing targeted solutions for each—diagnosis is clear, changes are modular, and ablations provide step-by-step validation.
- The "rehashing" trick is elegant: letting absorbing states jump randomly among $m$ indices with $1/m$ probability effectively portsof the spirit of "per-step random noise injection" from continuous diffusion to the discrete domain without requiring manually tuned noise intensities.
- Replacing Gumbel-max with low-discrepancy `torch.multinomial` sampling addresses an overlooked practical pain point regarding numerical inaccuracies under large vocabularies. The two are theoretically equivalent, allowing for a direct comparison.
- Decoupling training and inference allows for arbitrary discretization of the sampling timescale and the plug-and-play integration of DFM refinement steps. This logic is transferable to other discrete generation tasks like text.

## Limitations & Future Work
- RepA is only useful for accelerating convergence and provides no relative gain after full training. It is an efficiency and diagnostic tool rather than a performance source—users should not assume alignment itself brings quality gains.
- Noise capacity $m$ is a hyperparameter requiring empirical search for each tokenizer. The paper provides values for LlamaGen-f16/IBQ but lacks a principle for predicting the optimal $m$ directly.
- Experiments are concentrated on ImageNet-1K class-conditional generation and do not cover more complex conditions like text-to-image. The authors list "unified vision and language generation" as a future direction.
- Some key proofs (Eq. 9) and KV-Cache acceleration are placed in the appendix, making reconstruction dependent on the source code.

## Related Work & Insights
- **vs MVTM / MaskGIT**: MVTM uses a single `[mask]` + cross-entropy + Gumbel-max remask sampling, with training and inference tightly coupled. ReDDiT uses multi-index random absorbing states + DDM-linear ELBO + multinomial rehash sampling, achieving better quality and stability with decoupled training/inference.
- **vs DDM (Sahoo et al., MDLM)**: Shares the "time-invariant, arbitrary timescale sampling" philosophy, but MDLM still uses Gumbel-max. ReDDiT uses low-discrepancy multinomial sampling and explicitly merges absorbing state probabilities to maintain fidelity.
- **vs DFM (Discrete Flow Matching)**: Objective forms are similar (time-weighted cross-entropy). DFM provides token-wise refinement but requires more steps. ReDDiT can insert DFM steps into its own sampler for refinement to gain another $\sim$0.1 gFID.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The multi-index random absorbing state + rehash sampler represents a principled reconstruction of noise and sampling in discrete diffusion.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid ImageNet main comparison + component-wise ablations (objective/sampler/rehash/schedule/capacity), though limited to class-conditional generation on a single dataset.
- Writing Quality: ⭐⭐⭐⭐ The chain of diagnosis → hypothesis → method → ablation is clear. Equations and algorithms correspond well, though some proofs are in the appendix.
- Value: ⭐⭐⭐⭐⭐ For the first time, discrete diffusion matches continuous diffusion in gFID while remaining efficient, providing a viable path for unified vision/language generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Forward-Learned Discrete Diffusion: Learning how to noise to denoise faster](forward-learned_discrete_diffusion_learning_how_to_noise_to_denoise_faster.md)
- [\[ICLR 2026\] WeTok: Powerful Discrete Tokenization for High-Fidelity Visual Reconstruction](wetok_powerful_discrete_tokenization_for_high-fidelity_visual_reconstruction.md)
- [\[ICLR 2026\] Mitigating Noise Shift in Denoising Generative Models with Noise Awareness Guidance](mitigating_noise_shift_in_denoising_generative_models_with_noise_awareness_guida.md)
- [\[ICLR 2026\] Diverse Text-to-Image Generation via Contrastive Noise Optimization](diverse_text-to-image_generation_via_contrastive_noise_optimization.md)
- [\[ICLR 2026\] Product of Experts for Visual Generation](product_of_experts_for_visual_generation.md)

</div>

<!-- RELATED:END -->
