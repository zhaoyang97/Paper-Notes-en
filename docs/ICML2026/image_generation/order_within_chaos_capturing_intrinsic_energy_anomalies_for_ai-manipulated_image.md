---
title: >-
  [Paper Note] Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization
description: >-
  [ICML 2026][Image Generation][AI-IFL] Starting from the spectral bias of diffusion models, this paper theoretically proves that the local Gibbs energy of diffusion-generated regions is inherently lower than that of real…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "AI-IFL"
  - "Diffusion Models"
  - "Gibbs Energy"
  - "SAM Adapter"
  - "Self-evolving Data Synthesis"
date: 2026-05-08
content_hash: a24f73ade4a42a6f
---

# Order within Chaos: Capturing Intrinsic Energy Anomalies for AI-Manipulated Image Forgery Localization

**Conference**: ICML 2026  
**arXiv**: [2606.02178](https://arxiv.org/abs/2606.02178)  
**Code**: https://github.com/phoenixnir/FLAME  
**Area**: AI Safety / Image Forensics / Diffusion Forgery Localization  
**Keywords**: AI-IFL, Diffusion Models, Gibbs Energy, SAM Adapter, Self-evolving Data Synthesis  

## TL;DR
Starting from the spectral bias of diffusion models, this paper theoretically proves that the local Gibbs energy of diffusion-generated regions is inherently lower than that of real imaging regions. Based on this, a Local Adjacency Discrepancy (LAD) energy map is constructed as an intrinsic forensic fingerprint. A lightweight adapter then injects LAD cues into SAM to achieve pixel-level forgery localization. Accompanied by the EditStream multi-agent system, which automatically pulls the latest editing models from HuggingFace to refresh training data, the method improves the average IoU from the Prev. SOTA of ~0.25 to 0.46 across 7 AI-editing datasets.

## Background & Motivation
**Background**: In the pre-AIGC era, image forgery localization (IFL) primarily relied on "physical signal consistency"—splicing or copy-pasting would disrupt the ISP pipeline, leaving sensor noise residuals or JPEG compression traces. Methods like Noiseprint utilize these cues to locate tampered areas. However, with the emergence of instruction-based editing models like Stable Diffusion, FLUX, and DALL·E 3, tampered pixels are products of neural rendering. Camera noise is replaced by synthetic signals, rendering traditional physical cues almost entirely ineffective.

**Limitations of Prior Work**: Current approaches to diffusion forgery are largely inadequate: (1) Using vision encoders like DINOv2 for distribution shift detection relies on the pre-trained model's generalization and lacks explicit modeling of intrinsic diffusion traces. (2) Using MLLMs (e.g., FakeShield, SIDA) for "semantic inconsistency reasoning" is limited because high-end diffusion models possess strong physical priors that leave few semantic contradictions, and MLLM visual tokenization is inherently coarse-grained, missing pixel-level inconsistencies.

**Key Challenge**: Reliable forensic evidence is neither in missing sensor noise (replaced by synthetic noise) nor in semantic errors (resolved by strong priors). Where is it hidden?

**Goal**: To identify an **architecture-agnostic** statistical fingerprint determined by the diffusion mechanism itself for pixel-level AI-IFL, while solving the "time lag" problem where static benchmarks lag behind the iteration speed of editing models.

**Key Insight**: From a statistical mechanics perspective, the authors observe that diffusion models essentially optimize a variational lower bound (Ho et al. 2020). Combined with the known **spectral bias** of deep networks (Rahaman et al. 2019) and the Lipschitz smoothness of VAE decoders, diffusion-generated content systematically **suppresses high-frequency local variance**, entering an "artificial low-entropy ordered state." In contrast, real optical imaging is driven by photon shot noise and thermal noise, resulting in a "high-entropy chaotic state." A measurable statistical energy gap exists between these two.

**Core Idea**: Model the image as a Gibbs Random Field and use the local potential function of intensity differences between adjacent pixels as a spatial proxy for high-frequency energy. Construct a LAD energy map to transform the energy gap (between real and generated regions) and boundary energy spikes into spatially localizable features. Finally, use a SAM adapter to fuse these low-level forensic cues with high-level semantic priors for pixel-level mask generation.

## Method

### Overall Architecture
FLAME (Fine-grained Localization via Adjacency Map Energy) follows a "coarse-to-fine" two-stage pipeline: Given a suspected locally edited RGB image $I$, the LAD operator first transforms it into a single-channel energy map $\mathcal{L}$. A lightweight LAD-Net simultaneously outputs (a) a global binary classification score $y_{cls}$ and (b) a coarse localization mask $M_{coarse}$. Subsequently, $M_{coarse}$ is treated as a dense prompt for SAM. The semantic features $F_{sem}$ from the frozen SAM image encoder and the texture features $F_{tex}$ from LAD-Net are concatenated and passed through an adapter (composed of $1\times 1$ convolutions and residual blocks) to obtain $F_{adapted}$. Finally, the SAM mask decoder produces the fine-grained pixel-level mask $M_{final}$. The entire system is supported by the EditStream data engine, which automatically fetches the latest open-source inpainting models to generate training samples from new distributions.

### Key Designs

1.  **LAD Map (Local Adjacency Discrepancy)**:
    - **Function**: Transforms the theoretical phenomena of "low energy in diffusion regions, high energy in real regions, and spikes at tampered boundaries" into content-independent spatial energy fingerprints.
    - **Mechanism**: Views the image as a Gibbs Random Field $p(x) \propto \exp(-E(x))$ and defines the potential function of adjacent pixel pairs as a function of intensity difference $V(p,q) = \rho(\|I(p)-I(q)\|_2)$. For each pixel $p$ in a $3\times 3$ neighborhood $\mathcal{N}_p$, it aggregates: $\mathcal{L}(p) = \frac{1}{|\mathcal{N}_p|} \sum_{q \in \mathcal{N}_p} \tanh(\|I(p)-I(q)\|_2^2 / \tau^2)$. The $\tanh$ function saturates large semantic gradients and amplifies "small noise residuals" controlled by $\tau$. This encodes three cues—optical noise in real regions, smoothing artifacts in generated regions, and covariance misalignment at boundaries caused by latent space splicing $z_{t-1}=m\odot \hat z^{gen}_{t-1}+(1-m)\odot z^{ref}_{t-1}$—into a single map.
    - **Design Motivation**: Theoretical analysis (Theorem 3.3 Energy Gap & Boundary Spike) indicates that internal generated energy is lower than real energy, and boundary energy is significantly higher. However, since energy is defined on statistical expectation, a **pixel-level computable** and **content-robust** operator is required; the saturation function with local aggregation converts statistical differences into spatial tensors for CNN learning.

2.  **SAM Adapter Refinement**:
    - **Function**: Upgrades the coarse low-level forensic masks $M_{coarse}$ from LAD-Net to pixel-accurate boundaries while avoiding the flaw where real low-energy regions (e.g., solid backgrounds) are misjudged.
    - **Mechanism**: Freezes the SAM image encoder to extract semantic features $F_{sem}$. A lightweight Feature Adapter learns $F_{adapted} = \text{Adapter}(F_{sem} \oplus F_{tex})$ (where $\oplus$ denotes channel concatenation followed by $1\times 1$ convolution), "modulating" general segmentation priors into the forensic domain. Simultaneously, $M_{coarse}$ is fed into the SAM prompt encoder to generate dense positional embeddings $E_{prompt}$. The SAM mask decoder then uses $F_{adapted}$ and $E_{prompt}$ to output $M_{final}$.
    - **Design Motivation**: LAD-Net is a low-level statistical operator that might misidentify naturally smooth real regions as forged. Semantic priors are needed to distinguish such areas from "diffused objects." Bypassing full SAM fine-tuning preserves general priors and ensures parameter efficiency while avoiding catastrophic forgetting.

3.  **EditStream Self-evolving Data Synthesis**:
    - **Function**: Addresses the "generation gap" between training sets and real-world threats, as static benchmarks cannot keep up with the iteration speed of generators like SD, FLUX, or Qwen-Image.
    - **Mechanism**: Uses Qwen-VL as a semantic planner to analyze scene semantics and generate editing instructions, converted into precise masks via SAM. An Autonomous Model Scouting Agent (driven by Llama 3) monitors repositories like HuggingFace, automatically parsing model cards, synthesizing calling code, and integrating new inpainting models into the pipeline to inject "updated artifact distributions" into FLAME.
    - **Design Motivation**: Generalization depends on constructing a "moving target"—ensuring the detector consistently encounters artifacts harder than the previous version, forcing the model to learn **architecture-agnostic** diffusion commonalities rather than overfitting specific generator fingerprints.

### Loss & Training
Joint optimization of pixel-level localization and image-level detection: $L = L_{Dice} + \lambda_{focal} L_{focal}^{\alpha,\gamma} + \lambda_{IoU} L_{IoU} + \lambda_{det} L_{det}$. $L_{Dice}+L_{focal}$ ensures precise boundaries under foreground-background imbalance, $L_{IoU}$ supervises mask quality prediction via $\ell_1$ regression, and $L_{det}$ utilizes BCE for global discriminative capability. Following SAM conventions, $\lambda_{focal}=20$, $\lambda_{IoU}=\lambda_{det}=1$.

## Key Experimental Results

### Main Results
Evaluated on 7 AI editing datasets (MagicBrush, SID, CoCoGLIDE, AutoSplice, NanoBanana, Qwen-Image, Flux Kontext). Compared with 7 SOTA methods including SAFIRE, Mesorch, TruFor, AdaIFL, SIDA, FakeShield, and SparseViT. The "Average" column represents the mean of the latter 5 datasets to measure OOD generalization.

| Model | IoU(MagicBrush) | IoU(SID) | IoU(Qwen-Image) | IoU(Flux Kontext) | IoU(Avg/OOD) | F1(Avg/OOD) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| TruFor | 0.281 | 0.188 | 0.228 | 0.203 | 0.247 | 0.324 |
| SIDA | 0.106 | 0.488 | 0.089 | 0.092 | 0.191 | 0.250 |
| FakeShield | 0.091 | 0.117 | 0.098 | 0.096 | 0.131 | 0.152 |
| **FLAME (Ours)** | **0.538** | **0.580** | 0.321 | 0.285 | 0.358 | 0.459 |
| **FLAME-F (EditStream Fine-tuned)** | 0.507 | 0.569 | **0.482** ↑ | **0.446** ↑ | **0.460** ↑ | **0.565** ↑ |

For image-level detection, FLAME achieves 0.901/0.916 ACC on MagicBrush/SID (vs. Prev. SOTA SIDA's 0.812/0.725), with an average ACC of 0.639 (vs. the strongest baseline of 0.580). FLAME-F increases ACC on Qwen-Image from 0.715 to 0.812 and shows positive transfer to unseen Flux/NanoBanana, validating EditStream's cross-architecture generalization.

### Ablation Study

| Configuration | IoU↑ | F1↑ | Description |
| :--- | :--- | :--- | :--- |
| **FLAME (Full)** | **0.567** | **0.662** | Trained/Validated on SID+MagicBrush |
| w/o LAD Map (Use RGB) | 0.294 | 0.380 | **Most critical**: 0.273 IoU drop |
| w/o Adapter | 0.313 | 0.392 | 0.254 IoU drop |
| w/o SAM (Directly use coarse) | 0.379 | 0.458 | 0.188 IoU drop; boundaries become blocky |

### Key Findings
- **LAD Map is the lifeline**: Without it, IoU is halved (0.567 → 0.294), proving that "explicit modeling of intrinsic diffusion traces" is far more important than "letting the model learn on its own"—this explains the bottleneck of DINOv2/MLLM approaches.
- **SAM refinement is necessary**: Removing the SAM link leads to a significant drop and blocky predictions, indicating LAD-Net provides coarse energy responses that require semantic priors for boundary snapping.
- **EditStream "Moving Target" effect**: FLAME-F, fine-tuned only on Qwen-Image, improves on unseen Flux Kontext and NanoBanana (IoU +0.16 / +0.18). This suggests the model learns diffusion commonalities rather than specific generator fingerprints, with minimal forgetting on old benchmarks.
- **Robustness**: JPEG compression (Q=75) and Gaussian noise significantly degrade LAD (the former quantizes high-frequency coefficients, the latter injects global variance, both disrupting energy gaps), while Gaussian blur is less harmful—as region energy gaps and boundary misalignments persist even if texture residuals are attenuated.

## Highlights & Insights
- **Forensic cues derived from diffusion training objectives**: Properties like "spectral bias + VAE low-pass," often cited as quality flaws, are refinsed as unavoidable forensic fingerprints. The approach turns "weaknesses" of the opponent into "feature channels," which is more robust than MLLM semantic "spot-the-difference."
- **Unified encoder for three anomalies**: A single $\tanh$ pooling operator simultaneously encodes "high-entropy real regions," "low-entropy generated regions," and "boundary covariance misalignment," allowing the back-end CNN to process heterogeneous signals together.
- **Transferable "Low-level Residual + SAM Adapter" paradigm**: This two-stage structure—extracting a low-level forensic map, feeding it as a prompt to a frozen SAM, and injecting domain features via a lightweight adapter—is applicable to any forensic, industrial, or medical scenario requiring pixel-level localization where semantic priors are too dominant.
- **"Continuous Adversarial" data scaling**: Engineering the systematic "benchmark lag" into an LLM agent loop that automatically scrapes HuggingFace and synthesizes code is a model for building AI safety datasets.

## Limitations & Future Work
- **Assumption Dependency**: The Spectral-Energy Inequality relies on the spectral bias of diffusion models and VAE Lipschitz properties. If future generators explicitly optimize high-frequency loss or use non-smooth decoders, the energy gap may be diminished.
- **Post-processing Robustness**: Significant performance drops under JPEG (Q=75) and Gaussian noise suggest vulnerability if an adversary applies lightweight post-processing. Performance under targeted attacks like adversarial purification remains unexplored.
- **EditStream Reliability**: Total reliance on Llama 3 for model card parsing and code synthesis entails engineering risks such as failure, poor generation quality, or licensing issues; the authors provide only a conceptual description.
- **Bias toward Inpainting**: Evaluation is focused on local mask-guided editing. Generalization to full-image regeneration (e.g., SDXL) or video forgery is not yet demonstrated. Theoretically, the "boundary spike" signal vanishes in full-image generation, leaving only the energy gap cue.

## Related Work & Insights
- **vs. FakeShield / SIDA (MLLM-based)**: These rely on semantic/text alignment; FLAME uses a pure statistical energy approach. FLAME's Gain is sub-pixel localization unaffected by "semantically perfect" generation, but it lacks natural language explanations.
- **vs. Noiseprint / TruFor (Traditional Noise Residuals)**: They rely on physical ISP noise; FLAME uses "statistical anomalies of the generation mechanism." FLAME is superior for AIGC threats but may not outperform in traditional splicing tasks.
- **vs. DINOv2 Embedding methods**: They capture distribution shifts via general encoders; FLAME explicitly constructs forensic maps for better interpretability and stability against generator changes. **Key Insight**: Explicitly modeling the physical/statistical biases of the generation mechanism is more reliable than implicit learning in generalization scenarios.
- **vs. SAM Adapters in medical/general segmentation**: This paper migrates the SAM adapter paradigm to forensic localization, proving SAM's semantic priors are useful for boundary snapping, provided low-level residuals are explicitly injected.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The theoretical framework deriving forensic cues from diffusion spectral bias (Gibbs energy + Energy Gap & Boundary Spike theorems) is a rare and clear "mechanism-driven" perspective in AI-IFL.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covering 7 datasets, 7 SOTA baselines, three robustness perturbations, and extensive ablations on components/operators. OOD settings are reasonable; one star deducted for lacking evaluation on traditional splicing and adversarial purification.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logical progression from motivation to theory, operators, architecture, and data engine. Theory is formalized with Assumptions/Theorems.
- **Value**: ⭐⭐⭐⭐⭐ AI-IFL is a pressing safety issue. This work provides a SOTA solution and an engineering approach for "continuously refreshing benchmarks," with a transferable adapter paradigm.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Creating Blank Canvas Against AI-Enabled Image Forgery](../../AAAI2026/image_generation/creating_blank_canvas_against_ai-enabled_image_forgery.md)
- [\[ICML 2026\] Local Hessian Spectral Filtering for Robust Intrinsic Dimension Estimation](local_hessian_spectral_filtering_for_robust_intrinsic_dimension_estimation.md)
- [\[ICML 2026\] The Coupling Within: Flow Matching via Distilled Normalizing Flows](the_coupling_within_flow_matching_via_distilled_normalizing_flows.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICML 2026\] OcclusionFormer: Arranging Z-Order for Layout-Grounded Image Generation](occlusionformer_arranging_z-order_for_layout-grounded_image_generation.md)

</div>

<!-- RELATED:END -->
