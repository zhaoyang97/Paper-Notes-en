---
title: >-
  [Paper Note] Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models
description: >-
  [ICML 2026][Model Compression][Harmful Fine-tuning Defense] In the Fine-tuning-as-a-Service (FaaS) scenario, the authors reinterpret "temporarily jailbreaking the model before user fine-tuning" as a gradient saturation m…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Harmful Fine-tuning Defense"
  - "Temporary Jailbreaking"
  - "BufferLoRA"
  - "ReinforceLoRA"
  - "QR Orthogonal Merging"
date: 2026-05-08
content_hash: 8e6b206d50384a2b
---

# Jailbreak to Protect: Buffering and Reinforcing via Temporary Jailbreaking for Safe Fine-Tuning in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2605.24550](https://arxiv.org/abs/2605.24550)  
**Code**: TBD  
**Area**: LLM Safety / Fine-tuning-as-a-Service Defense / LoRA  
**Keywords**: Harmful Fine-tuning Defense, Temporary Jailbreaking, BufferLoRA, ReinforceLoRA, QR Orthogonal Merging  

## TL;DR
In the Fine-tuning-as-a-Service (FaaS) scenario, the authors reinterpret "temporarily jailbreaking the model before user fine-tuning" as a gradient saturation mechanism. Based on this observation, they design the Buffer-and-Reinforce framework: a removable BufferLoRA is used to absorb harmful gradients during user fine-tuning, and a ReinforceLoRA is subsequently integrated via QR orthogonal merging to bolster safety. This approach reduces harmful scores to approximately 8.5 without requiring any user-side safety data, while maintaining downstream task accuracy above 76.

## Background & Motivation

**Background**: Fine-tuning-as-a-Service (FaaS) platforms offered by providers like OpenAI and Google allow users to upload data to fine-tune aligned LLMs, becoming a mainstream method for customized deployment. Defenses against "harmful fine-tuning attacks" in FaaS are categorized into: the alignment phase (modifying pre-trained weights for robustness), the fine-tuning phase (adding regularization during user optimization), and the post-fine-tuning phase (post-hoc cleaning or merging safety modules).

**Limitations of Prior Work**: Most existing defenses follow an "explicit regularization" route, such as incorporating KL divergence, distance to a reference model, or adversarial perturbations into the user's fine-tuning objective. This approach faces three obstacles in FaaS: (i) it requires the provider to continuously inject extra safety data during user training, violating the minimalist interface of commercial FaaS; (ii) it incurs linear scaling of compute overhead for calculating regularization gradients; (iii) the regularization intensity is difficult to adapt when user data contains varying proportions of harmful samples, often failing to suppress harmful updates or damaging benign tasks. Zhou et al. (2024) proposed the Security Vector, which activates "harmful behavior modules" before fine-tuning. While empirically effective, it lacks a mechanistic explanation and systematic quantification of its impact on benign learning capabilities.

**Key Challenge**: The goal is to simultaneously suppress harmful gradients and preserve task gradients under the hard constraints of "zero extra data, zero extra compute, and unknown user data distributions." Explicit regularization naturally violates the zero-extra-data assumption, while temporary jailbreaking lacks theoretical grounding and proof that it does not also suppress useful updates.

**Goal**: (1) Provide a gradient-level analysis to explain why "jailbreaking before fine-tuning" blocks harmful updates. (2) Engineer this mechanism into a fine-tuning framework that does not rely on user-side safety data, has near-zero overhead, and further strengthens safety post-hoc.

**Key Insight**: By plotting the 2D loss landscape of LLaMA3-8B-Instruct on harmful and harmless data, the authors find that safety-aligned models still have significant downward room on harmful data (explaining why they are easily compromised), whereas "jailbroken models" have essentially converged to the bottom. Meanwhile, both models retain significant optimization margin on harmless data. This implies that jailbreaking does not "destructively break alignment" but rather drains the gradients in harmful directions, leaving only task-related directions for the user to fine-tune.

**Core Idea**: Utilize a removable LoRA module to temporarily jailbreak the model before fine-tuning, allowing harmful gradients to saturate naturally. After fine-tuning, this LoRA is removed, and a pre-trained safety-reinforcement LoRA is merged back via QR orthogonal projection. This achieves a "buffer then reinforce" two-step defense without altering user interfaces.

## Method

### Overall Architecture
The Buffer-and-Reinforce framework consists of three LoRA modules: BufferLoRA (pre-trained "jailbreak induction" module), ReinforceLoRA (pre-trained "refusal recovery" module), and UserLoRA (the module actually optimized with user data).

The pipeline follows three stages: (1) Pre-deployment—the provider trains BufferLoRA and ReinforceLoRA once to be shared across all users; (2) User Fine-tuning—BufferLoRA is attached and frozen while only UserLoRA is updated, causing harmful update directions to converge and remain inactive; (3) Post-fine-tuning—BufferLoRA is removed to restore original alignment, ReinforceLoRA is projected onto the orthogonal complement of the UserLoRA subspace via QR decomposition, and then averaged with UserLoRA to be merged back into the base model. This process requires no safety labels from the user and maintains an identical interface to standard LoRA fine-tuning.

A key observation supporting this design is the newly defined Safety Gradient Score $S^{l}=\tfrac{1}{N}\sum_{i}\mathbf{g}_{i}^{l}\cdot\mathbf{v}^{l}/(\lVert\mathbf{v}^{l}\rVert_{2}+\epsilon)$, which measures the projection of the current gradient onto the "safety direction" $\mathbf{v}^{l}$ (derived from safety-aligned LoRA weights). In LLaMA3-8B-Instruct (layers 0–15), aligned models show significant negative scores for both harmful and harmless data (indicating any fine-tuning erodes safety), while jailbroken models score near zero. Simultaneously, in layers 15+, the harmless gradient norms of jailbroken models are comparable to aligned models, with projections onto the safety gradient failing to decay, proving that user task directions are preserved.

### Key Designs

1.  **BufferLoRA: Saturating Harmful Directions**:
    - **Function**: Acts as a mountable, removable LoRA module during user fine-tuning to push the base model into the "jailbroken" harmful loss valley, causing subsequent gradients in harmful directions to approach zero.
    - **Mechanism**: Trained only on provider-held harmful query-response pairs $\mathcal{D}_{H}$ to optimize $\theta_{B}$ with $\mathcal{L}_{B}(\theta_{B})=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{H}}\sum_{t}\log P(y_{t}\mid x,y_{<t};\theta,\theta_{B})$. This training is performed once and reused for all user sessions.
    - **Design Motivation**: Unlike the Security Vector (Zhou et al. 2024), which requires KL terms to maintain benign performance, the Safety Gradient Score empirical evidence shows no KL is needed. The "mount-dismount" design ensures the base alignment is never permanently polluted.

2.  **ReinforceLoRA: Learning Refusal in a Jailbroken State**:
    - **Function**: Pushes safety beyond the "original alignment level" post-hoc, compensating for BufferLoRA's limitation of only "maintaining" rather than "strengthening" safety.
    - **Mechanism**: Optimizes $\theta_{R}$ while the base LLM and BufferLoRA are both attached and frozen, using $\mathcal{L}_{R}(\theta_{R})=-\mathbb{E}_{(x,y)\sim\mathcal{D}_{S}\cup\mathcal{D}_{B}}\sum_{t}\log P(y_{t}\mid x,y_{<t};\theta,\theta_{B},\theta_{R})$, where $\mathcal{D}_{S}$ contains harmful queries with refusal responses and $\mathcal{D}_{B}$ contains benign pairs. 
    - **Design Motivation**: Joint training on $\mathcal{D}_{S}$ and $\mathcal{D}_{B}$ prevents model collapse into constant refusal. Like BufferLoRA, it is trained once by the provider.

3.  **QR Orthogonal Merging: Shifting Safety Updates Away from Task Subspace**:
    - **Function**: Merges ReinforceLoRA with UserLoRA while avoiding damage to the learned user tasks.
    - **Mechanism**: UserLoRA is represented as $W_{U}=B_{U}A_{U}$. Using $\mathrm{span}(W_{U})\approx\mathrm{span}(B_{U})$, QR decomposition $\hat{B}_{U}=Q_{B}R$ is performed on $B_{U}$. Then, ReinforceLoRA is projected via $\tilde{W}_{R}=(I-\alpha Q_{B}Q_{B}^{\top})W_{R}$. Merging is finalized as $W_{\text{final}}=W_{\text{base}}+\tfrac{1}{2}(W_{U}+\tilde{W}_{R})$.
    - **Design Motivation**: Hard projection can over-delete ReinforceLoRA components during UserLoRA rank collapse. The authors use an eigenvalue threshold $\lambda_{i}>\tau\max_{j}\lambda_{j}$ on the Gram matrix $G=A_{U}A_{U}^{\top}$ to determine the effective subspace $V_{\text{eff}}$, ensuring task performance protection.

### Loss & Training
The three LoRAs are optimized independently: BufferLoRA uses $\mathcal{D}_{H}$ (5,000 harmful pairs), ReinforceLoRA uses $\mathcal{D}_{S}\cup\mathcal{D}_{B}$ (5,000 harmful-refusal + 5,000 benign), and UserLoRA use only user data $\mathcal{D}_{U}$. The provider trains the first two once; user fine-tuning runs standard cross-entropy for UserLoRA without additional regularization.

## Key Experimental Results

### Main Results
Using LLaMA3-8B-Instruct, downstream tasks include GSM8K, SST2, and AGNEWS. User data is mixed with harmful samples at ratio $p$ (from $0$ to $1$). Metrics are Harmful Score (HS, lower is better) and Fine-tuning Accuracy (FA, higher is better).

| Setting | Method | HS ↓ | FA ↑ |
|------|------|------|------|
| $p=0.1$, GSM8K | SFT | 75.2 | 69.0 |
| $p=0.1$, GSM8K | SafeInstruct | 19.6 | 69.4 |
| $p=0.1$, GSM8K | Security Vector | 22.1 | 71.3 |
| $p=0.1$, GSM8K | Antidote | 27.2 | 75.0 |
| $p=0.1$, GSM8K | Panacea | 36.2 | 67.1 |
| $p=0.1$, GSM8K | **Buffer-and-Reinforce (Ours)** | **8.1** | **76.6** |
| $p=0.5$, GSM8K | SFT | 80.7 | 67.3 |
| $p=0.5$, GSM8K | **Buffer-and-Reinforce (Ours)** | **8.2** | **75.2** |
| $p=1.0$, GSM8K | **Buffer-and-Reinforce (Ours)** | **8.8** | — |

Cross-task results ($p=0.1$) show the framework maintains low HS across all categories: HS drops from ~75 to ~8 on GSM8K, while FA remains comparable or slightly superior to SFT. Unlike SafeLoRA or Antidote, HS does not rebound sharply as user data size increases.

### Ablation Study
| Configuration | HS ↓ | FA ↑ | Note |
|------|------|------|------|
| Full Buffer-and-Reinforce | ≈8.5 | ≈76 | All three LoRAs + QR Merging |
| BufferLoRA only (w/o Reinforce) | Major drop vs SFT | ≈76 | BufferLoRA blocks primary harmful updates |
| ReinforceLoRA only (w/o Buffer) | Near SafeLoRA levels | Lower | Post-hoc safety is limited if user data is polluted |
| Naive Merging (No QR) | HS slightly better | FA ↓ | QR orthogonalization protects task performance |

### Key Findings
- BufferLoRA and ReinforceLoRA are complementary: the former prevents harmful updates from entering UserLoRA, while the latter mends minor safety gaps in the base model during merging.
- Stability is more critical than absolute HS: Buffer-and-Reinforce maintains a low HS standard deviation across varying $n$ and $p$, whereas Panacea/Antidote variance exceeds 20 in certain settings.
- The soft intensity $\alpha$ in QR merging acts as the primary dial between FA and HS.

## Highlights & Insights
- **Mechanistic Grounding**: The Safety Gradient Score provides a quantifiable metric to diagnose "jailbreak-as-saturation," enabling visualization and comparison across different defense methods.
- **Deployment Efficiency**: By shifting costs to pre-deployment, the framework's runtime is identical to standard LoRA, making it highly compatible with actual FaaS APIs.
- **Subspace Abstraction**: Identifies task vs. safety subspaces as the core of the merging problem, a framework transferable to multi-skill merging and personalization-alignment coexistence.

## Limitations & Future Work
- All conclusions are based on LLaMA3-8B-Instruct; the localization of the safety direction $\mathbf{v}^{l}$ and the 16-layer split may be model-specific.
- Dependence on provider-held safety data (5k+5k) and the coverage of $\mathcal{D}_{H}$ determine the extent of harmful distribution mitigation.
- Empirical hyper-parameters $\alpha$ and $\tau$ for QR merging lack an automated selection strategy.

## Related Work & Insights
- **vs Security Vector (Zhou 2024)**: Both use "jailbreak then tune," but this work provides a gradient explanation, removes the KL loss requirement, and adds post-hoc reinforcement, outperforming it on both HS and FA.
- **vs SafeLoRA/Antidote/Panacea**: These methods structuraly modify UserLoRA during merging; this work proves naive merging hurts task performance and proposes QR projection as a superior operator.
- **vs SafeInstruct/Lisa**: Unlike defenses requiring safety sample injection during every user session, this work front-loads costs through one-time training by the provider.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Decomposing the Basic Abilities of Large Language Models: Mitigating Cross-Task Interference in Multi-Task Instruct-Tuning](decomposing_the_basic_abilities_of_large_language_models_mitigating_cross-task_i.md)
- [\[ICML 2026\] Geo-Expert: Fine-tuning 8B Models into Expert-Level Geological Reasoning LLMs using LoRA](geo-expert_towards_expert-level_geological_reasoning_via_parameter-efficient_fin.md)
- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICML 2026\] The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models](the_shape_of_addition_geometric_structures_of_arithmetic_in_large_language_model.md)
- [\[ICML 2026\] NanoQuant: Efficient Sub-1-Bit Quantization of Large Language Models](nanoquant_efficient_sub-1-bit_quantization_of_large_language_models.md)

</div>

<!-- RELATED:END -->
