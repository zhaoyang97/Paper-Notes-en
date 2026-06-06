---
title: >-
  [Paper Note] Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding
description: >-
  [ICML 2026][Image Generation][Timestep embedding] This paper reveals that the often-overlooked "timestep embedding" in diffusion models is in fact an unused information side channel. By extending the training timestep ra…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Timestep embedding"
  - "Shadow Timestep"
  - "Backdoor attack"
  - "Watermark"
  - "Mutual coherence"
  - "Steganography"
date: 2026-05-08
content_hash: 038b3f5e4cb13bf5
---

# Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding

**Conference**: ICML 2026  
**arXiv**: [2605.00935](https://arxiv.org/abs/2605.00935)  
**Code**: Not explicitly provided in the paper  
**Area**: Diffusion Models / AI Security / Steganography & Watermarking  
**Keywords**: Timestep embedding, Shadow Timestep, Backdoor attack, Watermark, Mutual coherence, Steganography

## TL;DR
This paper reveals that the often-overlooked "timestep embedding" in diffusion models is in fact an unused information side channel. By extending the training timestep range to a "shadow interval" and binding a different data distribution to this interval, it is possible—without changing the scheduler interface—for the same diffusion model to generate normal images in the explicit interval and "hidden" images in the shadow interval. This enables both covert backdoor attacks and model watermark verification. The paper also provides a mutual coherence theoretical analysis based on sinusoidal positional encoding, explaining why two disjoint intervals can carry independent information.

## Background & Motivation

**Background**: Diffusion models (DDPM, Latent Diffusion, Stable Diffusion) have become foundational for image/video/audio generation. Timestep embedding is a core component—encoding the current denoising step $t$ as a vector input to the UNet, indicating "which noise level" the network should target. However, research has focused almost exclusively on "how to achieve the same quality with fewer steps," with little attention to whether the timestep dimension itself can be abused.

**Limitations of Prior Work**: Security research on diffusion models has focused on three interfaces: (1) input-side (prompt-based triggers, e.g., VillanDiffusion, BadT2I), (2) model parameter-side (weight steganography, concept erasure), and (3) output-side (image watermarking, e.g., Tree-Ring, ROBIN). This means **all existing attack and defense mechanisms monitor the input/output space**, leaving the timestep dimension unmonitored—an ideal hiding place for attackers.

**Key Challenge**: The scheduler interface of diffusion models (users pass timesteps to call UNet) naturally treats timestep as an "invisible internal state," and no one checks "what happens if I pass a $t$ outside the training range." This creates a covert bypass: **by quietly expanding the $t$ range during training**, the model learns a different distribution in the new interval, but external users remain unaware, as the inference API behaves identically.

**Goal**: (1) Verify whether timestep embedding can indeed accommodate extra information without harming the main task; (2) Theoretically explain why two disjoint timestep intervals can "independently" carry different representations (without mutual interference); (3) Demonstrate the dual-use nature of this side channel—for both covert backdoors and legitimate watermarking.

**Key Insight**: Treat timestep embedding as a "positional encoding mapping," and use the mutual coherence concept from compressed sensing to analyze "the correlation between two different timestep intervals in the sinusoidal embedding space." If the intervals have sufficiently low coherence, they can carry independent information like two sets of orthogonal codewords.

**Core Idea**: On top of the traditional training with $t \in [0, T]$, a shadow timestep interval $t_{sn} \in [T+1, T+T_s]$ is added, binding another data distribution $D_{sn}$ to this interval and jointly training with the explicit interval. During inference, calling $t \in [0, T]$ yields the normal model, while $t \in [T+1, T+T_s]$ triggers the hidden distribution—forming a controllable, schedule-level secret channel.

## Method

### Overall Architecture
STE modifies only two aspects of the standard diffusion training pipeline: (1) Expanding the timestep sampling range from $[0, T]$ to $[0, T+T_s]$, introducing several disjoint shadow intervals $t_{s1}, t_{s2}, \cdots$; (2) Binding each shadow interval to a different data distribution $D_{sn}$, sharing the same sinusoidal time-projection and UNet backbone for joint training. During inference, the scheduler calls $t \in [0, T]$ for the explicit trajectory to generate normal images, and $t \in [T+1, T+T_s]$ for the shadow trajectory to generate hidden images bound to that interval—the external API remains unchanged, with the only "key" being the timestep value.

### Key Designs

1. **Shadow Timestep Interval Expansion**:

    - **Function**: Carve out a subspace in the timestep encoding space, disjoint from the original training interval, to serve as a hidden carrier.
    - **Mechanism**: Diffusion models are originally trained only on $t \in [0, T]$, with timesteps mapped to vectors via sinusoidal position encoding $\text{PE}(t)_{2i} = \sin(t / 10000^{2i/d})$. **STE expands the allowed $t$ range** to $[0, T+T_s]$, and trains a second distribution $D_{sn}$ in the shadow subinterval $t_{sn} \subset [T+1, T+T_s]$. During training, $t$ is randomly sampled per batch (falling in either the explicit or shadow interval), and the UNet learns different denoising behaviors for different $t$; during inference, the output distribution depends on which $t$ is called. Multiple shadow intervals can carry multiple hidden distributions.
    - **Design Motivation**: Traditional training "fills" the entire $[0, T]$, but the sinusoidal embedding space is far from saturated; after expanding to $T+T_s$, the new embedding positions are unsupervised for the main task, so any new task can be "attached" without affecting the model's behavior on $[0, T]$. The unchanged scheduler interface ensures users cannot detect anomalies—this is key to stealth.

2. **Mutual Coherence Theoretical Guarantee for Separability**:

    - **Function**: Theoretically answer "why two timestep intervals can independently carry information without interfering."
    - **Mechanism**: The authors treat the sinusoidal time embedding as a dictionary $\{ \text{PE}(t) : t \in [0, T+T_s] \}$ of column vectors, defining the mutual coherence between two timesteps as $\mu(t_1, t_2) = |\langle \text{PE}(t_1), \text{PE}(t_2)\rangle| / (\|\text{PE}(t_1)\| \|\text{PE}(t_2)\|)$. If the timesteps in the explicit and shadow intervals have sufficiently low pairwise coherence, then from the perspective of compressed sensing/dictionary learning, **the two sets of timesteps act as nearly orthogonal codewords**, and the corresponding UNet behaviors can be learned independently without confusion. The paper provides mutual coherence analysis to explain the separability of disjoint intervals.
    - **Design Motivation**: Pure experimental validation of STE's effectiveness is insufficient—readers may ask, "Won't two distributions sharing the same UNet overwrite each other?" Theoretical proof that sinusoidal embeddings have controlled coherence on disjoint intervals shows that the UNet has enough "capacity" to exhibit completely different behaviors at different $t$, which is the physical basis for STE.

3. **Dual-Use Security Surface: Attack and Defense Duality**:

    - **Function**: The same STE mechanism can be used for covert backdoor attacks (malicious) or for model watermark verification (legitimate).
    - **Mechanism**: (a) **Attack scenario**: Attackers inject shadow timesteps into downstream open-source models via code poisoning (releasing a disguised diffusion pipeline that users install). When the target shadow timestep is called, the model generates predefined malicious images (e.g., with triggers); since normal users never pass $t$ outside $[0, T]$, no anomalies are observed, and traditional input/output monitoring defenses fail. (b) **Defense/watermark scenario**: Model owners bind a set of shadow timesteps and signature images to their own models during training, and after deployment, verify model ownership by querying those timesteps (like outputting a "fingerprint")—third parties copying the model have no way of knowing where the shadow intervals are. The paper demonstrates this duality with "covert attack injection" and "watermark verification tool" use cases.
    - **Design Motivation**: Presenting the "covert information channel" as both a threat and a tool, echoing the paper's title "Watch Your Step"—a warning to developers about scheduler security, and a usable steganographic watermarking scheme for defenders. The paper emphasizes that STE is orthogonal to existing input/output attack and defense mechanisms; monitoring prompts or checking output images cannot reveal the presence of shadow timesteps.

### Loss & Training
Jointly train explicit and shadow distributions, randomly assigning each sample in a batch to a timestep interval, then supervise with the standard diffusion training loss (e.g., DDPM noise prediction MSE). Samples in the shadow interval come from different distributions $D_{sn}$ but use the same UNet. Multiple shadow intervals are mutually disjoint. The paper does not detail specific hyperparameters (explicit:shadow capacity ratio, shadow interval length, etc.), but emphasizes that mutual coherence analysis provides guidance for selecting disjoint intervals.

## Key Experimental Results

> As this note cache only covers content up to Sec.3.1, experimental data is based on the paper's abstract and contribution statements.

### Main Results
The core conclusion from the abstract: **STE can reliably inject auxiliary data distributions while maintaining the independence of the explicit and shadow manifolds**—that is, images generated in the explicit interval retain normal FID/IS, and the shadow interval can generate predefined target distributions (hidden images).

| Evaluation Dimension | STE Performance (as stated in abstract) |
|----------------------|-----------------------------------------|
| Main task retention (explicit distribution quality) | Unaffected by shadow injection |
| Hidden information capacity (shadow distribution quality) | Reliable injection, controllable generation |
| Mutual coherence analysis | Low correlation between disjoint timestep intervals, theoretically separable |
| Dual-use | Same mechanism enables both attack triggers and watermark verification |

### Ablation Study
No explicit ablation table is provided in the abstract, but the following design points can be inferred:

| Configuration | Core Issue |
|---------------|-----------|
| No timestep range expansion | Degenerates to standard diffusion, no hidden channel |
| Shadow interval overlaps with explicit | High mutual coherence, distributions interfere |
| Shadow interval too short | Embedding space insufficiently independent, hidden distribution poorly learned |
| Multiple shadow intervals stacked | Trade-off between capacity and mutual coherence |

### Key Findings
- **Timestep is a neglected attack/defense surface**: Existing backdoor/watermark work is almost entirely in the input/output space; STE is the first to make the timestep dimension a "security surface."
- **Extremely stealthy**: Since normal users never pass timesteps outside $[0, T]$, traditional IO monitoring defenses are completely ineffective; attack triggers are hidden in scheduler invocation conventions.
- **Theoretical support**: Mutual coherence analysis provides a physical explanation for "why shadow and explicit intervals can each carry independent distributions"—sinusoidal embedding space is nearly orthogonal on disjoint segments.
- **Dual-use is a double-edged sword**: The same mechanism can be a backdoor or a watermark; the paper calls on the community to include timestep in trust boundary audits when designing diffusion pipelines.

## Highlights & Insights
- This paper is a true "**new attack/defense surface**" contribution—it does not alter the diffusion training algorithm or sampler, but simply points out **an unmonitored channel hidden in existing interfaces**. The value of such vulnerability discovery can exceed that of performance improvements, as it **redefines the threat model**.
- The analogy between timestep embedding and dictionary atoms in compressed sensing, analyzed via mutual coherence, is a cross-disciplinary insight—this "positional encoding as dictionary" perspective applies to all models using sinusoidal/learnable position embeddings (Transformer, NeRF, Audio diffusion), suggesting similar "position embedding side channels" may be widespread.
- The duality of attack and watermarking suggests that many ML security studies can be "interpreted as dual-use"—steganographic attacks and watermarks share underlying mechanisms, differing only in "who knows the key." This perspective helps defenders anticipate more attack forms.
- The combination of "code poisoning + scheduler backdoor" is a wake-up call for the open-source ecosystem—many users now install diffusion pipelines directly from Hugging Face/GitHub via pip, making line-by-line auditing impractical; STE's pure interface-level backdoor bypasses almost all static analysis.
- Treating the timestep range as an implicit API design issue: should diffusion schedulers hard-cap timesteps to $[0, T]$? This is relevant to the security standards of all diffusion deployments.

## Limitations & Future Work
- The core contribution is "discovery + theoretical analysis"; the method does not involve complex losses or architectural innovations. Details such as attack success rate, capacity limits, and transferability to different backbones (DDPM/LDM/SD3) require further empirical study.
- The mutual coherence analysis is specific to sinusoidal embedding; it is unclear whether it applies to learnable timestep embeddings (as in some latent diffusion models).
- Why can't defenders simply "truncate $t \in [0, T]$ during inference"? The paper acknowledges this as the most direct countermeasure, but attackers could instead hide shadow intervals at sparse points within $[0, T]$ (e.g., fractional timesteps), still evading static truncation; the attack-defense cat-and-mouse game is not fully explored.
- Joint training of explicit and shadow distributions may slightly degrade UNet's generation quality on $[0, T]$ (e.g., minor FID increase); the abstract claims "independence," but specific FID comparisons are not covered in this cache.
- Stacking multiple shadow intervals increases mutual coherence with the number of intervals, so capacity has a theoretical upper bound, which is not quantified in the paper.
- Interaction between STE and PEFT methods like LoRA/DreamBooth is not discussed—if the model is fine-tuned, whether the shadow interval is overwritten and whether the watermark remains verifiable is key to practical watermark usability.

## Related Work & Insights
- **vs VillanDiffusion / BadT2I (prompt-based backdoors)**: Those hide triggers in text prompts, while STE hides them in scheduler invocation conventions; prompt monitoring defenses are completely ineffective against STE.
- **vs Tree-Ring / ROBIN (output image watermarking)**: Those embed frequency-domain signatures in generated images, while STE embeds directly inside the model—no post-processing required and not vulnerable to image editing.
- **vs StegaDDPM / CRoSS / DMIH (diffusion steganography)**: Those hide secret information in noise trajectories, conditioning, or score functions; STE is the first to use the timestep dimension as a carrier—a new carrier dimension.
- **vs Concept Erasure (Gandikota et al.)**: Those use parameter updates to erase harmful concepts, while STE uses expanded timesteps to inject hidden concepts—the two are attack-defense duals.
- **vs LLM backdoors / prompt injection**: LLM research has focused on input space attacks; STE suggests that the position embedding dimension in LLMs could be similarly exploited—a highly transferable perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to identify the timestep dimension as a security surface, discovering a truly "new" attack/defense interface, with mutual coherence theoretical explanation.
- Experimental Thoroughness: ⭐⭐⭐ — Abstract provides key claims but this note cache does not cover full quantitative experiments; based on paper length, experiments are likely proof-of-concept rather than large-scale benchmarks.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clearly articulated, with a logical chain explaining "why timestep is overlooked" and "why it becomes a side channel"; Figures 1/2 intuitively illustrate the dual-use mechanism.
- Value: ⭐⭐⭐⭐⭐ — Significant impact on the diffusion security community—supply chain attack defenses for open-source diffusion pipelines need redesign, timestep should be included in trust boundary audits, and the watermarking community gains a new, built-in signature mechanism not reliant on output post-processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](../../CVPR2026/image_generation/slice_semantic_latent_injection_via_compartmentalized_embedding_for_image_waterm.md)
- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](../../ICLR2026/image_generation/rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [\[CVPR 2026\] CTCal: Rethinking Text-to-Image Diffusion Models via Cross-Timestep Self-Calibration](../../CVPR2026/image_generation/ctcal_rethinking_text-to-image_diffusion_models_via_cross-timestep_self-calibrat.md)
- [\[ICCV 2025\] Structure-Guided Diffusion Models for High-Fidelity Portrait Shadow Removal](../../ICCV2025/image_generation/structure-guided_diffusion_models_for_high-fidelity_portrait_shadow_removal.md)

</div>

<!-- RELATED:END -->
