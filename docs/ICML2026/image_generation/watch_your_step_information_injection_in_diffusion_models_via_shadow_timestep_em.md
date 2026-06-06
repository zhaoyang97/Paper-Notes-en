---
title: >-
  [Paper Note] Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding
description: >-
  [ICML 2026][Image Generation][Timestep embedding] This paper reveals that the long-overlooked "timestep embedding" in diffusion models is actually an unused side-channel for information. By extending the training timeste…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Timestep embedding"
  - "Shadow Timestep"
  - "Backdoor attack"
  - "Watermarking"
  - "Mutual coherence"
  - "Steganography"
date: 2026-05-08
content_hash: 3ef6c16718f83f3a
---

# Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding

**Conference**: ICML 2026  
**arXiv**: [2605.00935](https://arxiv.org/abs/2605.00935)  
**Code**: Not explicitly provided in the paper  
**Area**: Diffusion Models / AI Security / Steganography & Watermarking  
**Keywords**: Timestep embedding, Shadow Timestep, Backdoor attack, Watermarking, Mutual coherence, Steganography

## TL;DR
This paper reveals that the long-overlooked "timestep embedding" in diffusion models is actually an unused side-channel for information. By extending the training timestep range to a "shadow interval" (shadow timestep) and binding another data distribution to that interval, the same diffusion model can generate normal images within the explicit range and "hidden" images within the shadow range without changing the scheduler interface. This can be used for both covert backdoor attacks and model watermarking verification. Furthermore, a theoretical analysis of mutual coherence based on sinusoidal position encoding is provided to explain why two disjoint intervals can carry independent information.

## Background & Motivation

**Background**: Diffusion models (DDPM, Latent Diffusion, Stable Diffusion) have become the de facto foundation for image, video, and audio generation. Timestep embedding is a core component—it encodes the current denoising step $t$ into a vector that is fed into the UNet, informing the network of the current noise level to be processed. However, existing research almost exclusively focuses on "how to achieve the same quality with fewer steps," while few have questioned whether the timestep dimension itself can be abused.

**Limitations of Prior Work**: Security research on diffusion models is concentrated on three interfaces: (1) the input side (prompt-based triggers, such as VillanDiffusion, BadT2I), (2) the model parameter side (weight steganography, concept erasure), and (3) the output side (image watermarking, such as Tree-Ring, ROBIN). This means **all existing defenses monitor the input/output space**, leaving the temporal dimension (timestep) as a vacuum—and an ideal hiding place for attackers.

**Key Challenge**: The scheduler interface of a diffusion model (where a user passes a timestep to call the UNet) naturally treats the timestep as an "invisible internal state." No one checks "what happens if a $t$ outside the training range is passed." This provides a covert bypass: **as long as the range of $t$ is secretly expanded during training**, the model learns other distributions on the newly added intervals without the external user's knowledge, as the inference API behavior remains completely unchanged.

**Goal**: (1) Verify whether timestep embedding can actually accommodate extra information without damaging the main task; (2) explain theoretically why two disjoint timestep intervals can "independently" carry different representations without interference; (3) demonstrate the dual-use of this side-channel—serving as both a covert backdoor and a legitimate watermark.

**Key Insight**: The authors treat timestep embedding as a "positional encoding mapping" and borrow the concept of mutual coherence from compressed sensing to analyze the "correlation between two different timestep intervals in the sinusoidal embedding space." If the coherence between two intervals is sufficiently low, they can carry independent information like two sets of orthogonal codewords.

**Core Idea**: Based on the traditional training range $t \in [0, T]$, an additional shadow timestep interval $t_{sn} \in [T+1, T+T_s]$ is created. Another data distribution $D_{sn}$ is bound to this interval and trained jointly with the explicit interval. During inference, calling $t \in [0, T]$ results in a normal model, while calling $t \in [T+1, T+T_s]$ triggers the hidden distribution—forming a controllable secret channel at the schedule-level.

## Method

### Overall Architecture
STE modifies only two things in the standard diffusion training pipeline: (1) it expands the timestep sampling range from $[0, T]$ to $[0, T+T_s]$, introducing several disjoint shadow intervals $t_{s1}, t_{s2}, \cdots$; (2) it binds a different data distribution $D_{sn}$ to each shadow interval, which are then jointly trained using a shared sinusoidal time-projection + UNet backbone. During inference, the scheduler calls $t \in [0, T]$ to follow the explicit trajectory and generate normal images, while calling $t \in [T+1, T+T_s]$ follows a shadow trajectory to generate hidden images bound to that interval. The external API remains unchanged, and the only "password" is the timestep value.

### Key Designs

1. **Shadow Timestep Interval Expansion**:

    - **Function**: To create subspaces in the timestep encoding space that are disjoint from the original training interval to serve as hidden carriers.
    - **Mechanism**: Originally, diffusion models are trained only within $t \in [0, T]$, where timesteps are mapped into vectors via sinusoidal position encoding like $\text{PE}(t)_{2i} = \sin(t / 10000^{2i/d})$. **STE expands the allowed range of $t$** to $[0, T+T_s]$ and trains a second set of distributions $D_{sn}$ within the shadow sub-intervals $t_{sn} \subset [T+1, T+T_s]$. During training, $t$ is sampled randomly per batch (potentially falling into either the explicit or shadow intervals), and the UNet learns different denoising behaviors for different $t$. During inference, the distribution corresponding to whichever range of $t$ is called will be generated. Multiple shadow intervals can carry multiple hidden distributions.
    - **Design Motivation**: Traditional training "fills" the entire $[0, T]$, but the sinusoidal embedding space is actually far from being fully utilized. The additional embedding positions resulting from the expansion to $T+T_s$ are entirely unsupervised by the main task, so any new task can be "attached" without affecting the original model's behavior on $[0, T]$. Meanwhile, the unchanged scheduler interface ensures that users perceive no anomalies—this is the key to covertness.

2. **Mutual Coherence Theoretical Guarantee for Separability**:

    - **Function**: To answer theoretically "why two timestep intervals can carry information independently without mutual interference."
    - **Mechanism**: The authors treat sinusoidal time embeddings as column vectors in a dictionary $\{ \text{PE}(t) : t \in [0, T+T_s] \}$ and define the mutual coherence between two timesteps as $\mu(t_1, t_2) = |\langle \text{PE}(t_1), \text{PE}(t_2)\rangle| / (\|\text{PE}(t_1)\| \|\text{PE}(t_2)\|)$. If the timesteps in the explicit and shadow intervals satisfy a sufficiently low pairwise coherence, then from a compressed sensing / dictionary learning perspective, **the two sets of timesteps act like two sets of near-orthogonal codewords**, and the corresponding UNet behaviors can be learned independently without confusion. The paper provides a mutual coherence evaluation to explain the separability of disjoint intervals.
    - **Design Motivation**: Purely experimental verification that STE works is not convincing enough—readers might ask if two sets of distributions using the same UNet would overwrite each other. Theoretically proving that the coherence of sinusoidal embeddings in disjoint intervals is controlled explains that the UNet has enough "capacity" to exhibit completely different behaviors at different $t$, which is the physical basis for STE.

3. **Dual-Use Security Surface: Attack and Defense Duality**:

    - **Function**: The same STE mechanism can serve as both a covert backdoor attack (malicious) and a model watermark verification (legitimate).
    - **Mechanism**: (a) **Attack Scenario**: An attacker injects shadow timesteps into downstream open-source models through code poisoning (releasing a disguised diffusion pipeline that users accidentally install). When the target shadow timestep is called, the model generates a predefined malicious image (e.g., content with a trigger). Since normal users never pass a $t$ beyond $[0, T]$, no anomalies are observed during normal use, and traditional defenses monitoring input/output fail. (b) **Defense/Watermarking Scenario**: A model owner binds a set of shadow timesteps + signature images known only to themselves during training. After deployment, they verify ownership by querying those timesteps (acting as an output "fingerprint"), while a third party copying the model would have no way of knowing where the shadow interval is. The paper demonstrates this duality through "covert attack injection" and "watermark verification tool" use cases.
    - **Design Motivation**: By discussing the "covert information channel" as both a threat and a tool, the paper echoes the pun in the title "Watch Your Step"—it is both a warning to developers about scheduler security and a usable steganographic watermarking solution for defenders. The paper emphasizes that STE is orthogonal to existing input/output defenses.

### Loss & Training
The explicit and shadow distributions are trained jointly. For each sample, the timestep interval it falls into is determined randomly per batch, and then supervised using standard diffusion training loss (such as DDPM's noise prediction MSE). Samples in the shadow interval come from different distributions $D_{sn}$, but use the same UNet. Multiple shadow intervals are disjoint from each other. The paper does not provide detailed hyperparameters (such as the capacity ratio of explicit to shadow, or the choice of shadow interval length) but emphasizes that mutual coherence analysis provides guidance for selecting disjoint intervals.

## Key Experimental Results

### Main Results
The core conclusion from the abstract: **STE can reliably inject auxiliary data distributions while maintaining the independence of explicit and shadow manifolds**—meaning images generated in the explicit interval maintain normal FID/IS quality, while the shadow interval can generate predefined target distributions (hidden images).

| Evaluation Dimension | STE Performance (as stated in Abstract) |
|----------|----------------------------|
| Main Task Maintained (Explicit quality) | Unaffected by shadow injection |
| Hidden Information Capacity (Shadow quality) | Reliable injection, controllable generation |
| Mutual Coherence Analysis | Low correlation between disjoint intervals, theoretically separable |
| Dual-use | Same mechanism serves as attack trigger and watermark |

### Ablation Study
The abstract does not provide a clear ablation table, but the design points inferred from the method's structure are:

| Configuration | Key Problem |
|------|---------|
| No Timestep Expansion | Degenerates to standard diffusion; no hidden channel |
| Overlapping Shadow/Explicit Intervals | High mutual coherence; the two sets of distributions interfere |
| Too Short Shadow Interval | Embedding space lacks independence; poor hidden learning |
| Stacking Multiple Shadow Intervals | Trade-off between capacity and mutual coherence |

### Key Findings
- **Timestep is an overlooked attack/defense surface**: Existing backdoor/watermarking work is almost entirely in the input/output space; STE is the first to turn the timestep dimension into a "security surface."
- **Extremely high covertness**: Because normal users do not pass timesteps beyond $[0, T]$, traditional I/O monitoring defenses fail completely; the attack trigger is hidden within the scheduler's calling convention.
- **Theoretical support**: The mutual coherence analysis provides a physical explanation for "why shadow and explicit intervals can each carry independent distributions"—sinusoidal embedding space is near-orthogonal in disjoint segments.
- **Dual-use is a double-edged sword**: The same mechanism can be a backdoor or a watermark; the paper calls for the community to include the timestep in trust boundary audits when designing diffusion pipelines.

## Highlights & Insights
- This paper is a genuine "**a new attack/defense surface**" type of work—it doesn't change the diffusion training algorithm or sampler but points out **a hidden, unmonitored channel within existing interfaces**. The value of such vulnerability discovery can sometimes exceed that of performance improvement work because it **reframes the threat model**.
- Analogizing timestep embedding to dictionary atoms in compressed sensing for mutual coherence analysis is a cross-domain idea—this perspective of "positional encoding as a dictionary" holds for all models using sinusoidal/learnable position embeddings (Transformers, NeRF, Audio diffusion). Similar "positional embedding bypass channels" might be widespread.
- The duality between attack and watermarking reminds us that much ML security research can be interpreted in two ways—steganographic attacks and watermarking share the same underlying mechanism, with the difference being "who knows the key." This perspective helps defenders anticipate more attack forms.
- The combination of "code poisoning + scheduler backdoor" sounds an alarm for the open-source ecosystem. Many users now directly pip install diffusion pipelines from Hugging Face / GitHub, making line-by-line audits impossible. STE, as a purely interface-level backdoor, bypasses almost all static analysis.

## Limitations & Future Work
- The core contribution is "discovery + theoretical analysis." The method lacks complex loss or architectural innovations; empirical details like attack success rate, capacity limits, and transferability across different backbones (DDPM/LDM/SD3) need to be characterized in future work.
- The mutual coherence analysis is directed at sinusoidal embeddings; whether it holds for learnable timestep embeddings (as in some latent diffusion models) remains unclear.
- Why can't the defender simply "force truncation to $t \in [0, T]$ during inference"? The paper acknowledges this as the most direct countermeasure, but an attacker could counter by hiding shadow intervals at sparse points within $[0, T]$ (e.g., using fractional timesteps), which would still evade static truncation. The cat-and-mouse game has not fully unfolded.
- Does joint training of explicit and shadow distributions cause a slight drop in generation quality (FID increase) for the UNet on $[0, T]$? The abstract claims "independence is maintained," but specific FID comparison figures were not provided in this summary.

## Related Work & Insights
- **vs VillanDiffusion / BadT2I (prompt-based backdoors)**: Those hide triggers in the text prompt; STE hides them in the scheduler calling convention. Existing prompt-monitoring defenses are entirely ineffective against STE.
- **vs Tree-Ring / ROBIN (output image watermarking)**: Those embed frequency domain signatures in generated images, whereas STE embeds directly inside the model—requiring no post-processing and avoiding damage from image editing.
- **vs StegaDDPM / CRoSS / DMIH (diffusion steganography)**: Those hide secrets in noise trajectories, conditioning, or score functions. STE is the first to use the timestep dimension as a carrier—**the carrier dimension is new**.
- **vs Concept Erasure (Gandikota et al.)**: Those use parameter updates to eliminate harmful concepts; STE conversely uses expanded timesteps to inject hidden concepts. They are attack-defense duals.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — The first to promote the timestep dimension as a security surface, discovering a truly "new" interface, with a theoretical explanation via mutual coherence.
- Experimental Thoroughness: ⭐⭐⭐ — The abstract provides key claims but lacks comprehensive quantitative experiments in this summary. The experimental part is likely a proof-of-concept rather than a large-scale benchmark.
- Writing Quality: ⭐⭐⭐⭐ — The motivation is clear, and the logic of "why timestep is ignored" and "how it becomes a bypass" is well-constructed.
- Value: ⭐⭐⭐⭐⭐ — Significant impact on the diffusion security community. Supply chain attack defenses for open-source pipelines need redesigning, and the watermarking community gains a new built-in signature mechanism that doesn't rely on output post-processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SLICE: Semantic Latent Injection via Compartmentalized Embedding for Image Watermarking](../../CVPR2026/image_generation/slice_semantic_latent_injection_via_compartmentalized_embedding_for_image_waterm.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](../../ICLR2026/image_generation/rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)

</div>

<!-- RELATED:END -->
