---
title: >-
  [Paper Note] Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding
description: >-
  [ICML 2026][Image Generation][Timestep embedding] This paper reveals that the often-overlooked "timestep embedding" in diffusion models is actually an unoccupied information side channel. By extending the training timestep range to a "shadow interval" (shadow timestep) and binding another data distribution to this interval, a single diffusion model can be made to gene
tags:
  - ICML 2026
  - Image Generation
  - Timestep embedding
  - Shadow Timestep
  - Watermarking
date: 2026-05-08
content_hash: 47781d44e93951a4
---
# Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding

**Conference**: ICML 2026  
**arXiv**: [2605.00935](https://arxiv.org/abs/2605.00935)  
**Code**: Not explicitly provided in the paper  
**Area**: Diffusion Models / AI Security / Steganography & Watermarking  
**Keywords**: Timestep embedding, Shadow Timestep, Backdoor Attack, Watermarking, Mutual Coherence, Steganography

## TL;DR
This paper reveals that the often-overlooked "timestep embedding" in diffusion models is actually an unoccupied information side channel. By extending the training timestep range to a "shadow interval" (shadow timestep) and binding another data distribution to this interval, a single diffusion model can be made to generate normal images in the explicit range and "hidden" images in the shadow range without changing the scheduler interface. This can be used for both covert backdoor attacks and model watermark verification. The authors also provide a theoretical analysis of mutual coherence based on sinusoidal position encoding to explain why two disjoint intervals can carry independent information.

## Background & Motivation

**Background**: Diffusion models (DDPM, Latent Diffusion, Stable Diffusion) have become the de facto foundation for image, video, and audio generation. Timestep embedding is a core component that encodes the current denoising step $t$ into a vector fed to the UNet, informing the network of the current noise level. However, research has primarily focused on "how to achieve the same quality with fewer steps," while the potential for abusing the timestep dimension itself has been ignored.

**Limitations of Prior Work**: Security research on diffusion models has concentrated on three interfaces: (1) Input side (prompt-based triggers, such as VillanDiffusion, BadT2I), (2) Model parameter side (weight steganography, concept erasure), and (3) Output side (image watermarking, e.g., Tree-Ring, ROBIN). This means **all existing attacks and defenses are monitored in the input/output space**, leaving the time dimension (timestep) a vacuum—and an ideal hiding place for attackers.

**Key Challenge**: The scheduler interface of diffusion models (where users pass a timestep to call the UNet) naturally treats the timestep as an "invisible internal state." No one checks "what happens if I pass a $t$ outside the training range." This provides a covert bypass: **as long as the range of $t$ is quietly expanded during training**, the model learns an alternative distribution in the newly expanded interval while remaining undetected by external users, as the inference API behavior remains unchanged.

**Goal**: (1) Verify whether timestep embeddings can accommodate additional information without damaging the primary task; (2) Theoretically explain why two disjoint timestep intervals can "independently" carry different representations without interference; (3) Demonstrate the duality of this side channel for both covert backdoors and legitimate watermarking.

**Key Insight**: Treat timestep embedding as a "positional encoding mapping" and borrow the concept of mutual coherence from compressed sensing to analyze the correlation between two different timestep intervals in the sinusoidal embedding space. If the coherence between two intervals is sufficiently low, they can carry independent information like two sets of orthogonal codewords.

**Core Idea**: On top of the traditional training range $t \in [0, T]$, a shadow timestep interval $t_{sn} \in [T+1, T+T_s]$ is established. A secondary data distribution $D_{sn}$ is bound to this interval and co-trained with the primary task. During inference, calling $t \in [0, T]$ results in a normal model, while calling $t \in [T+1, T+T_s]$ triggers the hidden distribution—creating a controllable secret channel at the schedule level.

## Method

### Overall Architecture
The problem STE (Shadow Timestep Embedding) aims to solve is whether a single model can quietly carry a second distribution without modifying the diffusion training algorithm or the scheduler interface. It treats the "internal state axis" of the timestep as an information carrier. In standard training, timesteps are sampled within $[0, T]$. STE extends the allowed range to $[0, T+T_s]$, binds another data distribution to the additional "shadow intervals" $t_{sn} \subset [T+1, T+T_s]$, and trains them jointly with the main task. Consequently, passing $t \in [0, T]$ during inference follows the normal trajectory for regular images, while passing $t \in [T+1, T+T_s]$ follows a shadow trajectory for hidden images. The external API behavior remains unchanged; the only "key" is the timestep value itself.

### Key Designs

**1. Shadow Timestep Interval Extension: Turning the unmonitored timestep axis into a hidden carrier**

Timesteps in diffusion models are first converted into vectors via sinusoidal position encoding (mappings like $\text{PE}(t)_{2i} = \sin(t / 10000^{2i/d})$) and then fed into the UNet to indicate the noise level. The pain point is that traditional training only "uses up" the $[0, T]$ segment, while the sinusoidal embedding space is far from full. The additional positions are completely unsupervised in the primary task—making them ideal "vacant lots" for new tasks. STE therefore extends the sampling range to $[0, T+T_s]$ and trains a second distribution $D_{sn}$ on several disjoint shadow sub-intervals $t_{sn} \subset [T+1, T+T_s]$. During training, it is randomly decided by batch whether each sample falls into the explicit interval or a shadow interval. The UNet thus learns different denoising behaviors for different $t$, and multiple shadow intervals can even carry different hidden distributions. This is effective because the newly added embedding positions do not interfere with the supervision in $[0, T]$, leaving the original model's behavior unaffected. Furthermore, since the scheduler interface shape is unchanged, users cannot detect anomalies, ensuring high stealth.

**2. Mutual Coherence Analysis: Explaining why distributions do not overwrite each other**

Experimental results showing "STE works" are insufficient. A natural question arises: if two distributions share the same UNet, why don't they overwrite each other? STE answers this by treating the sinusoidal time embedding as a dictionary $\{ \text{PE}(t) : t \in [0, T+T_s] \}$, where each timestep is a column vector. It defines the mutual coherence between two timesteps as $\mu(t_1, t_2) = |\langle \text{PE}(t_1), \text{PE}(t_2)\rangle| / (\|\text{PE}(t_1)\| \, \|\text{PE}(t_2)\|)$. If the coherence between timesteps in the explicit and shadow intervals is sufficiently low, then from the perspective of compressed sensing or dictionary learning, these two sets of timesteps act like two sets of near-orthogonal codewords, allowing the corresponding UNet behaviors to be learned independently. This analysis physically proves that the coherence of sinusoidal embeddings on disjoint intervals is controlled—the UNet has enough "capacity" to exhibit completely different behaviors at different $t$. This is the foundation of STE and provides quantitative guidance for selecting disjoint shadow intervals.

**3. Dual-Use Security Surface: One mechanism as both backdoor and watermark**

The true intrigue of STE lies in its attack-defense duality. In an attack scenario, an attacker releases a disguised diffusion pipeline via code poisoning. After a user installs it, the shadow timestep is injected into the downstream open-source model. Once the target shadow timestep is called, the model generates a predefined malicious image (e.g., content with a trigger), whereas normal users never pass a $t$ beyond $[0, T]$, revealing no flaws. Traditional defenses monitoring input prompts or output images fail. In a defense/watermarking scenario, the model owner binds a set of secret shadow timesteps and signature images to their model during training. After deployment, they can verify ownership by querying these timesteps, much like an embedded "fingerprint," which third-party copiers would not know how to find. The duality between "covert attack injection" and "watermark verification tool" echoes the pun in the title "Watch Your Step"—warning developers to include schedulers in security audits while providing defenders with a steganographic watermarking solution independent of output post-processing.

### Loss & Training
Training involves joint optimization of explicit and shadow distributions: for each sample, it is randomly decided by batch which timestep interval it falls into, supervised by the standard diffusion training loss (e.g., noise-prediction MSE of DDPM). Samples in the shadow interval come from a different distribution $D_{sn}$ but share the same UNet, while multiple shadow intervals are disjoint. Although specific hyperparameters (explicit-to-shadow capacity ratio, shadow interval length) are not detailed, the mutual coherence analysis is emphasized as the guiding principle for selecting disjoint intervals.

## Key Experimental Results

### Main Results
Core conclusions from the abstract: **STE can reliably inject auxiliary data distributions while maintaining the independence of explicit and shadow manifolds.** This means images generated in the explicit interval maintain normal FID/IS without degradation, while the shadow interval generates the predefined target distribution (hidden images).

| Evaluation Dimension | STE Performance (from Abstract) |
|----------------------|--------------------------------|
| Main Task Retention (Explicit Quality) | Unaffected by shadow injection |
| Hidden Info Capacity (Shadow Quality) | Reliably injected, controllable generation |
| Mutual Coherence Analysis | Low correlation between disjoint intervals, theoretically separable |
| Dual-Use | One mechanism serves as both attack trigger and watermark |

### Ablation Study
The abstract does not provide explicit ablation tables, but key design points can be inferred:

| Configuration | Key Question |
|---------------|--------------|
| No timestep range extension | Degenerates to standard diffusion, no hidden channel |
| Shadow/Explicit overlap | High mutual coherence, distributions interfere |
| Shadow interval too short | Insufficient independence in embedding space, poor learning |
| Multiple shadow intervals | Trade-off between capacity and mutual coherence |

### Key Findings
- **Timesteps are an overlooked attack/defense surface**: Existing backdoor/watermarking work is almost entirely in the input/output space; STE is the first to turn the timestep dimension into a "security surface."
- **Extremely high stealth**: Since normal users do not pass timesteps beyond $[0, T]$, traditional IO monitoring defenses fail completely; the trigger is hidden in the scheduler calling convention.
- **Theoretical support**: Mutual coherence analysis provides a physical explanation for why shadow and explicit intervals can carry independent distributions—sinusoidal embedding spaces are near-orthogonal in disjoint segments.
- **Dual-use is a double-edged sword**: The same mechanism can be a backdoor or a watermark, urging the community to include timesteps in the trust boundary audits of diffusion pipelines.

## Highlights & Insights
- This paper is a genuine "**a new attack/defense surface**" style of work. It does not change the training algorithm or sampler but points out a **hidden, unmonitored channel in existing interfaces**. The value of such vulnerability discovery often outweighs performance improvements because it **reconstructs the threat model**.
- Analogizing timestep embedding to dictionary atoms in compressed sensing for mutual coherence analysis is a cross-domain insight. This "positional encoding as a dictionary" perspective applies to all models using sinusoidal or learnable position embeddings (Transformers, NeRF, Audio diffusion), suggesting that similar "position embedding bypass channels" may be widespread.
- The duality of attacks and watermarks reminds us that many ML security studies can be "interpreted both ways." Steganographic attacks and watermarks share underlying mechanisms; the difference is only in "who knows the key." This perspective helps defenders anticipate more attack forms.
- The combination of "code poisoning + scheduler backdoor" sounds an alarm for the open-source ecosystem. Users often install diffusion pipelines directly from Hugging Face or GitHub, making line-by-line auditing impossible. STE’s pure interface-level backdoor bypasses almost all static analysis.

## Limitations & Future Work
- The core contribution is "discovery + theoretical analysis." Methodologically, there are no complex losses or architectural innovations. Empirical details like attack success rate, capacity limits, and transferability across different backbones (DDPM/LDM/SD3) need characterization in follow-up work.
- Mutual coherence analysis is specific to sinusoidal embeddings; its validity for learnable timestep embeddings (as used in some latent diffusion models) is unclear.
- Why can't defenders simply "force truncate $t \in [0, T]$ during inference"? The paper acknowledges this as the most direct countermeasure, but notes that attackers could hide the shadow interval within sparse points inside $[0, T]$ (e.g., fractional timesteps), still evading static truncation.
- Could joint training of explicit and shadow distributions cause a slight drop in UNet quality for $[0, T]$ (e.g., a slight FID increase)? The abstract claims "independence," but specific FID comparisons were not covered in this summary.
- As more shadow intervals are stacked, mutual coherence will increase, placing a theoretical limit on capacity that the paper does not explicitly boundary.
- Interaction with PEFT methods like LoRA or DreamBooth was not explored. Whether shadow intervals are overwritten or if watermarks remain verifiable after fine-tuning is a key question for practical utility.

## Related Work & Insights
- **vs. VillanDiffusion / BadT2I (prompt-based backdoors)**: Those hide triggers in text prompts, while STE hides them in scheduler calling conventions. Defenses monitoring prompts are ineffective against STE.
- **vs. Tree-Ring / ROBIN (output image watermarks)**: Those embed frequency-domain signatures in generated images, whereas STE embeds them directly inside the model—requiring no post-processing and being resistant to image editing.
- **vs. StegaDDPM / CRoSS / DMIH (diffusion steganography)**: Those hide information in noise trajectories, conditioning, or score functions; STE is the first to use the **timestep dimension as a carrier**.
- **vs. Concept Erasure (Gandikota et al.)**: Those use parameter updates to eliminate harmful concepts; STE conversely uses extended timesteps to inject hidden concepts, forming a duality.
- **vs. LLM Backdoors / Prompt Injection**: LLM research focuses on input space attacks; STE suggests that the position embedding dimension in LLMs could be similarly exploited—a highly transferable perspective.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — First to define the timestep dimension as a security surface, identifying a truly "new" interface with theoretical backing.
- **Experimental Thoroughness**: ⭐⭐⭐ — The abstract provides key claims, but this summary lacks full quantitative experiments. It is likely a proof-of-concept work rather than a large-scale benchmark.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation; the logic of why timesteps are ignored and how they become a bypass is well-constructed.
- **Value**: ⭐⭐⭐⭐⭐ — Significant impact on the diffusion security community. Supply chain defense for open-source pipelines needs redesign, and watermarking gains a new built-in signature mechanism independent of output processing.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](../../ICLR2026/image_generation/rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)
- [\[CVPR 2026\] RebRL: Reinforcing Discrete Visual Diffusion Models with Rebalanced Timestep Credits](../../CVPR2026/image_generation/rebrl_reinforcing_discrete_visual_diffusion_models_with_rebalanced_timestep_cred.md)

</div>

<!-- RELATED:END -->
