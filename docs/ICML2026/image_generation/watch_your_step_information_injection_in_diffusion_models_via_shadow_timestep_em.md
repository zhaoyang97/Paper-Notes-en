---
title: >-
  [Paper Note] Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding
description: >-
  [ICML 2026][Image Generation][Timestep embedding] This paper reveals that the long-overlooked "timestep embedding" in diffusion models serves as an unoccupied information side channel. By extending the training timestep range to a "shadow interval" (shadow timestep) and binding an alternative data distribution to it, the same diffusion model can generate normal images in the explicit interval and "hidden" images in the shadow interval without changing the scheduler interface.…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Timestep embedding"
  - "Shadow Timestep"
  - "Backdoor Attack"
  - "Watermarking"
  - "Mutual Coherence"
  - "Steganography"
date: 2026-05-08
content_hash: 15f47294dd6c0a18
---

# Watch Your Step: Information Injection in Diffusion Models via Shadow Timestep Embedding

**Conference**: ICML 2026  
**arXiv**: [2605.00935](https://arxiv.org/abs/2605.00935)  
**Code**: Not explicitly provided in the paper  
**Area**: Diffusion Models / AI Security / Steganography & Watermarking  
**Keywords**: Timestep embedding, Shadow Timestep, Backdoor Attack, Watermarking, Mutual Coherence, Steganography

## TL;DR
This paper reveals that the long-overlooked "timestep embedding" in diffusion models serves as an unoccupied information side channel. By extending the training timestep range to a "shadow interval" (shadow timestep) and binding an alternative data distribution to it, the same diffusion model can generate normal images in the explicit interval and "hidden" images in the shadow interval without changing the scheduler interface. This can be used for both covert backdoor attacks and model watermark verification. The paper also provides a theoretical analysis of mutual coherence based on sinusoidal position encoding to explain why two disjoint intervals can carry independent information.

## Background & Motivation

**Background**: Diffusion models (DDPM, Latent Diffusion, Stable Diffusion) have become the de facto foundation for image, video, and audio generation. Timestep embedding is a core component—it encodes the current denoising step $t$ into a vector fed to the UNet, informing the network of the corresponding noise level. However, research has focused almost exclusively on using fewer steps for better quality, while the dimension of the timestep itself has rarely been investigated for potential misuse.

**Limitations of Prior Work**: Existing security research on diffusion models focuses on three interfaces: (1) Input-side (prompt-based triggers like VillanDiffusion, BadT2I), (2) Model parameter-side (weight steganography, concept erasing), and (3) Output-side (image watermarking like Tree-Ring, ROBIN). Consequently, **all current defenses monitor the input/output space**, leaving the time dimension (timestep) a vacuum—and an ideal hiding place for attackers.

**Key Challenge**: The scheduler interface of diffusion models (where users pass timesteps to call the UNet) naturally treats timesteps as an "invisible internal state." Users rarely check what happens when a $t$ outside the training range is passed. This creates a stealthy bypass: **by quietly expanding the range of $t$ during training**, the model learns a different distribution on the newly expanded interval without external users noticing, as the inference API behavior remains unchanged.

**Goal**: (1) Verify whether timestep embedding can accommodate extra information without damaging the main task; (2) Theoretically explain why two disjoint timestep intervals can "independently" carry different representations without interference; (3) Demonstrate the dual-use of this side channel for both covert backdoors and legitimate watermarking.

**Key Insight**: Treat timestep embedding as a "position encoding mapping" and borrow the concept of mutual coherence from compressed sensing to analyze the correlation between different timestep intervals in the sinusoidal embedding space. If the coherence between two intervals is low enough, they can carry independent information like two sets of orthogonal codewords.

**Core Idea**: Based on traditional training with $t \in [0, T]$, a shadow timestep interval $t_{sn} \in [T+1, T+T_s]$ is established. Another data distribution $D_{sn}$ is bound to this interval and trained jointly with the main task. During inference, calling $t \in [0, T]$ acts as a normal model, while calling $t \in [T+1, T+T_s]$ triggers the hidden distribution—forming a controllable secret channel at the schedule-level.

## Method

### Overall Architecture
STE (Shadow Timestep Embedding) aims to allow a single model to quietly carry a second distribution without modifying the diffusion training algorithm or the scheduler interface. It treats the "internal state axis" of the timestep as an information carrier. While standard training samples timesteps only within $[0, T]$, STE extends the allowed range to $[0, T+T_s]$, binding another data distribution to the additional "shadow intervals" during joint training. Thus, passing $t \in [0, T]$ during inference follows a normal trajectory to produce regular images, while passing $t \in [T+1, T+T_s]$ follows the shadow trajectory to produce hidden images. The external API remains unchanged, with the timestep value itself acting as the "key."

### Key Designs

**1. Shadow Timestep Interval Expansion: Turning the unmonitored timestep axis into a hidden carrier**

Timesteps in diffusion models are typically transformed into vectors via sinusoidal position encoding (mappings like $\text{PE}(t)_{2i} = \sin(t / 10000^{2i/d})$) before being fed to the UNet to indicate the noise level. The limitation is that traditional training only "uses up" the $[0, T]$ segment, while the sinusoidal embedding space is far from full. These extra positions are unsupervised in the main task, providing "vacant land" for new tasks. STE expands the sampling range to $[0, T+T_s]$ and trains a second distribution $D_{sn}$ on several disjoint shadow sub-intervals $t_{sn} \subset [T+1, T+T_s]$. During training, the batch randomly determines if a sample falls into the explicit or a shadow interval. The UNet learns different denoising behaviors for different $t$. This works because the new embedding positions do not interfere with the supervision on $[0, T]$, leaving the original model's behavior intact. Since the scheduler interface stays the same, users cannot detect anomalies, ensuring stealthiness.

**2. Mutual Coherence Analysis: Explaining why distributions do not overwrite each other**

To explain why two distributions sharing the same UNet do not overlap, STE views the sinusoidal time embedding as a dictionary $\{ \text{PE}(t) : t \in [0, T+T_s] \}$, where each timestep is a column vector. It defines the mutual coherence between two timesteps as $\mu(t_1, t_2) = |\langle \text{PE}(t_1), \text{PE}(t_2)\rangle| / (\|\text{PE}(t_1)\| \, \|\text{PE}(t_2)\|)$. If the coherence between timesteps in the explicit and shadow intervals is sufficiently low, then from a compressed sensing/dictionary learning perspective, these timesteps act like nearly orthogonal codewords. Consequently, the UNet can learn independent behaviors for them without confusion. This analysis physically proves that the coherence of sinusoidal embeddings on disjoint intervals is controllable, giving the UNet enough "capacity" to exhibit entirely different behaviors at different $t$. This serves as the foundation for STE and provides quantitative guidance for selecting disjoint shadow intervals.

**3. Dual-use Security Surface: One mechanism for both backdoors and watermarks**

The duality of STE is that the same mechanism changes from malicious to legitimate depending on the user. In an attack scenario, an attacker releases a disguised diffusion pipeline via code poisoning. Once the target shadow timestep is called, the model generates predefined malicious images (e.g., content with triggers), while normal users passing $t \in [0, T]$ see no flaws. Traditional defenses monitoring prompts or output images fail. In a defense/watermarking scenario, owners bind their models to a set of private shadow timesteps and signature images. Querying these timesteps after deployment verifies ownership—acting as a built-in "fingerprint" that third-party copiers cannot find. The paper demonstrates this via "covert attack injection" and "watermark verification tool" use cases. This echoes the pun in the title "Watch Your Step"—warning developers to audit schedulers while providing defenders with a steganographic watermarking solution independent of output post-processing.

### Loss & Training
Training involves the joint optimization of explicit and shadow distributions. For each sample, the timestep interval is randomly decided per batch, followed by supervision using the standard diffusion training loss (e.g., noise-prediction MSE for DDPM). Samples from shadow intervals come from the distribution $D_{sn}$ but share the same UNet, with multiple shadow intervals being disjoint. While specific hyperparameters are not detailed, the mutual coherence analysis serves as the guiding principle for interval selection.

## Key Experimental Results

### Main Results
Core conclusion: **STE can reliably inject auxiliary data distributions while maintaining the independence of explicit and shadow manifolds.** Images generated in the explicit interval maintain normal FID/IS, while the shadow interval generates the predefined target distribution.

| Evaluation Dimension | STE Performance (from Abstract) |
|----------|----------------------------|
| Main Task Retention (Explicit Quality) | Unaffected by shadow injection |
| Hidden Information Capacity (Shadow Quality) | Reliability injected, controllable generation |
| Mutual Coherence Analysis | Low correlation between disjoint intervals, theoretically separable |
| Dual-use | Same mechanism functions as attack trigger and watermark |

### Ablation Study

| Configuration | Key Problem |
|------|---------|
| No timestep expansion | Degenerates to standard diffusion; no hidden channel |
| Overlapping shadow/explicit intervals | High coherence; distributions interfere with each other |
| Shadow interval too short | Embedding space not independent enough; poor learning |
| Multiple stacked shadow intervals | Trade-off between capacity and mutual coherence |

### Key Findings
- **Timesteps are a neglected attack/defense surface**: Existing backdoor/watermark work resides in input/output space; STE turns the timestep dimension into a "security surface."
- **High Stealth**: Normal users do not pass timesteps beyond $[0, T]$, rendering traditional I/O monitoring defenses ineffective; the trigger is hidden in the scheduler calling convention.
- **Theoretical Support**: Mutual coherence analysis explains why disjoint intervals can carry independent distributions—sinusoidal embedding spaces are nearly orthogonal in disjoint segments.
- **Dual-use is a double-edged sword**: The same mechanism serves both as a backdoor and a watermark, calling for the inclusion of timesteps in trustworthiness audits of diffusion pipelines.

## Highlights & Insights
- This work identifies a "**new attack/defense surface**." It does not modify the training algorithm but reveals a **hidden channel within existing interfaces**. Its value lies in **restructuring the threat model**.
- Analogizing timestep embedding to dictionary atoms in compressed sensing is a cross-domain insight. This "position encoding as dictionary" perspective applies to any model using sinusoidal/learnable position embeddings (Transformers, NeRF, Audio diffusion), suggesting widespread "positional side channels."
- The duality between attacks and watermarking suggests that many ML security studies share underlying mechanisms, differing only in who holds the "key."
- The combination of "code poisoning + scheduler backdoor" warns the open-source ecosystem. Users downloading pipelines from Hugging Face/GitHub cannot audit every line; STE-style backdoors bypass almost all static analysis.
- The design philosophy raises a question: Should diffusion schedulers hard-cap timesteps to $[0, T]$? This relates to safety standards for all diffusion deployments.

## Limitations & Future Work
- The core contribution is "discovery + theoretical analysis." The methodology lacks complex loss or architectural innovations. Empirical details like attack success rate, capacity limits, and transferability across backbones (DDPM/LDM/SD3) require further characterization.
- Mutual coherence analysis focuses on sinusoidal embeddings; its applicability to learnable timestep embeddings is unclear.
- Defenders could "forcefully truncate $t \in [0, T]$." The paper acknowledges this but notes that attackers could hide shadow intervals within sparse points (e.g., fractional timesteps) inside $[0, T]$ to evade static truncation.
- Potential degradation of UNet generation quality (e.g., slight FID increase in the explicit interval) during joint training is mentioned but not exhaustively quantified in the provided cache.
- Capacity has a theoretical limit as mutual coherence increases with the number of shadow intervals.
- The interaction with PEFT methods (LoRA/DreamBooth) is not explored—it remains to be seen if shadow intervals survive fine-tuning.

## Related Work & Insights
- **vs. VillanDiffusion / BadT2I (Prompt-based backdoors)**: These hide triggers in text prompts; STE hides them in scheduler conventions. Prompt-monitoring defenses are ineffective against STE.
- **vs. Tree-Ring / ROBIN (Output watermarking)**: These embed signatures in the frequency domain of generated images; STE embeds them internally, making them resistant to image editing.
- **vs. StegaDDPM / CRoSS / DMIH (Diffusion steganography)**: These hide info in noise trajectories or score functions; STE is the first to use the **timestep dimension as a new carrier**.
- **vs. Concept Erasing**: Erasing removes harmful concepts; STE uses expanded timesteps to inject hidden concepts—the two are opposites.
- **vs. LLM Backdoors**: Most LLM research focuses on input space; STE suggests that position embedding dimensions in LLMs might be similarly exploited.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to propose timesteps as a security surface with a "brand new" interface and mutual coherence theory.
- Experimental Thoroughness: ⭐⭐⭐ — Key claims are made, but full quantitative experiments are not covered in this summary.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation and logical flow regarding why timesteps are a bypass.
- Value: ⭐⭐⭐⭐⭐ — Significant impact on the diffusion security community; necessitates a redesign of supply chain defenses for open-source pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Timestep Rescheduling in Diffusion Inversion](timestep_rescheduling_in_diffusion_inversion.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICLR 2026\] RMFlow: Refined Mean Flow by a Noise-Injection Step for Multimodal Generation](../../ICLR2026/image_generation/rmflow_refined_mean_flow_by_a_noise-injection_step_for_multimodal_generation.md)
- [\[ICML 2026\] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models](embedding-perturbed_exploration_preference_optimization_for_flow_models.md)
- [\[ICLR 2026\] The Spacetime of Diffusion Models: An Information Geometry Perspective](../../ICLR2026/image_generation/the_spacetime_of_diffusion_models_an_information_geometry_perspective.md)

</div>

<!-- RELATED:END -->
