---
title: >-
  [Paper Note] DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery
description: >-
  [ICML 2026][Model Compression][Dataset Distillation] DIVER transforms the classic Dataset Distillation (DD) from a "single-stage direct evaluation" to a two-stage paradigm of "distill first…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Dataset Distillation"
  - "Diffusion Model Priors"
  - "Cross-architecture Generalization"
  - "Semantic Recovery"
  - "Two-stage Distillation"
date: 2026-05-08
content_hash: 0fed9023d1059790
---

# DIVER: Diving Deeper into Distilled Data via Expressive Semantic Recovery

**Conference**: ICML 2026  
**arXiv**: [2605.12649](https://arxiv.org/abs/2605.12649)  
**Code**: To be confirmed  
**Area**: Model Compression / Dataset Distillation  
**Keywords**: Dataset Distillation, Diffusion Model Priors, Cross-architecture Generalization, Semantic Recovery, Two-stage Distillation

## TL;DR
DIVER transforms the classic Dataset Distillation (DD) from a "single-stage direct evaluation" to a two-stage paradigm of "distill first, then rescue semantics using a pre-trained diffusion model." Through three steps—Semantic Inheritance, Semantic Guidance, and Semantic Fusion—it recovers suppressed high-level semantics from "garbled" images distilled from ConvNet. This generally improves the accuracy of the same distilled data on heterogeneous architectures like ResNet18/ViT by 3–10 percentage points, requiring only 2.48s and 4GB of VRAM per image.

## Background & Motivation

**Background**: Dataset Distillation (DD) compresses millions of training images into dozens or thousands of "synthetic proxy samples," allowing models trained on the proxy set to approach the performance of the original set. The mainstream approach is bilevel optimization: the inner loop evaluates classification loss on a fixed proxy architecture $\varPhi^p$ (usually a small ConvNet), while the outer loop directly updates synthetic images in pixel space according to criteria such as gradient matching, distribution matching, or trajectory matching.

**Limitations of Prior Work**: Pixel-space bilevel optimization causes synthetic images to overfit deeply to the specific low-frequency/high-frequency patterns of $\varPhi^p$, resulting in abstract, noisy, and unrealistic images. When such proxy sets are used to train ResNet18, ShuffleNet, or ViT, the accuracy collapses significantly. As shown in Tab. 1, cross-architecture accuracy for classic DM/MTT/EDF on ImageNet subsets is occasionally worse than random image selection (e.g., ImageFruit IPC=1: MTT 15.4% vs Random 14.1% is only 1.3 points higher; at IPC=10, 18.9% < 19.6%).

**Key Challenge**: There is a structural trade-off between distillation accuracy (on $\varPhi^p$) and cross-architecture generalization (on heterogeneous $\varPhi^v$). The optimization target is the loss of $\Phi^p$, but actual deployment uses $\Phi^v$; the "pattern-specific optimal solution" converged upon by the former is not a global optimum in the sense of the latter. This manifests as images containing high-frequency textures that are useful for ConvNet but act as noise for ViT, alongside a loss of necessary semantic clarity.

**Goal**: Without rerunning DD, accessing the original dataset, or retraining diffusion models, refine existing "poor images" $\mathcal{D}^*$ into "refined images" $\mathcal{S}$ such that they simultaneously satisfy three conditions: (1) retain dataset-level semantics implicit in $\mathcal{D}^*$; (2) filter out architecture-sensitive "noise" patterns; (3) appear realistic with clear class attributes.

**Key Insight**: The authors assert that diffusion models (especially pre-trained DiT) are naturally "natural image manifold projectors." By feeding abstract distilled images as latent initializations into the reverse process, the diffusion process itself pulls the images toward the real distribution. Simultaneously, hierarchical feature extraction in neural networks (shallow layers capture textures/edges, deep layers capture semantics) implies that the VAE encoder projection into latent space inherently filters architecture-related noise.

**Core Idea**: Encode "distilled images" into latent space using a VAE $\to$ add appropriate noise as a DDIM starting point $\to$ use distilled latent for "loyalty guidance" and class labels for conditional guidance during the middle of the diffusion process to recover synthetic images with both semantic clarity and original dataset knowledge, serving as a hot-pluggable plugin for any existing DD method.

## Method

### Overall Architecture
DIVER is a decoupled two-stage pipeline. The authors formally split the original DD problem into DD + DDD:

- **Stage I (DD)**: Adopt any classic DD algorithm (DM / DC / MTT / NCFM / EDF / SRe²L / G-VBSM) to distill $\mathcal{D}^*=\{(\tilde x_i, \tilde y_i)\}$ on a ConvNet. This stage can even be skipped—if pre-existing distilled data is available, one can proceed directly to Stage II.
- **Stage II (DDD, Diving into Distilled Data)**: This is the core contribution. Fix a pre-trained guided diffusion model (primarily DiT-XL/2 + vae-ft-mse, trained on 256×256 ImageNet). For each $\tilde x \in \mathcal{D}^*$:
    1. **Semantic Inheritance (SI)**: $\tilde x \xrightarrow{\mathcal{E}} z_0$;
    2. Add $t_f=25$ steps of noise to obtain $z_{t_f}$;
    3. Initialize the DDIM reverse process (50 steps total) with $\hat z_{t_r}=z_{t_f}$;
    4. **Semantic Fusion (SF)** determines the guidance for each step: use only CFG during the CP/RP stages; use both **Semantic Guidance (SG)** and CFG during the SP stage $[t_h, t_l]=[40, 25]$;
    5. $\hat z_0 \xrightarrow{\mathcal{F}}$ decodes the synthetic image $\mathcal{H}_{\mathcal{D}^*}(\tilde x)$. The resulting synthetic set $\mathcal{S}$ is used to train the target architecture $\varPhi^v$.

Formal objective: $\mathcal{S}^* = \arg\min_{\mathcal{S}} \mathcal{M}(\varPhi^v_\mathcal{O}(x), \varPhi^v_\mathcal{S}(\mathcal{H}_{\mathcal{D}^*}(\tilde x)))$, where $|\mathcal{S}|=|\mathcal{D}|\ll|\mathcal{O}|$. Note that Stage II is entirely training-free, does not access $\mathcal{O}$, and is invisible to $\varPhi^p$, making it a true plug-in post-processing method.

### Key Designs

1. **Semantic Inheritance (SI) — Using Distilled Images as Start Points**:
    - **Function**: Uses a pre-trained VAE to project distilled images $\tilde x$ into latent $z_0$, adds $t_f$ steps of noise $z_{t_f}=\sqrt{\alpha_{t_f}}z_0+\sqrt{1-\alpha_{t_f}}\epsilon$, and replaces the DDIM default random Gaussian starting point with $\hat z_{t_r}=z_{t_f}$.
    - **Mechanism**: Relies on the empirical observation that CNN shallow layers learn texture edges while deep layers learn semantics. The VAE encoder, constrained by natural image distributions, treats "unnatural architecture-specific noise" as a low-utility, high-cost signal and tends to discard it during compression (validated by Tab. 8 and Fig. 7). Thus, the "VAE pass" inherently completes "noise filtering and semantic retention." The choice of $t_f$ is a critical trade-off: if too large, the latent is dominated by the Gaussian prior and semantics are lost; if too small, it deviates from the Gaussian distribution assumed by diffusion, degrading sampling quality. Fig. 4 shows $t_f=25$ is the sweet spot.
    - **Design Motivation**: Addresses the coupling problem where distilled images contain both useful semantics and architecture-specific noise. Traditional latent editing methods (e.g., SDEdit) starting from pure Gaussian noise cannot inject dataset-level semantics, while direct pixel-space denoising (e.g., GAN inversion) fails to maintain global consistency in the diffusion process.

2. **Semantic Guidance (SG) — Preventing Sampling Drift**:
    - **Function**: Adds a gradient at each reverse denoising step to pull $\hat z_t$ back toward $z_0$. The DDIM update becomes $\hat z_{t-1}=s(\hat z_t,t,\epsilon_\theta)-\gamma\nabla_{\hat z_t}\mathcal{G}_t(\hat z_t)$, with guidance function $\mathcal{G}_t=(\hat z_t-z_0)^2\cdot\sigma_t/2$ and scaling factor $\gamma=0.1$ (0.02 for NCFM).
    - **Mechanism**: CFG continuously injects class condition $c$ during the reverse process, which tends to push the latent toward the "mean image of the class," washing out discriminative information from the original distilled image outside the class mean. SG uses L2 distance to anchor the current latent near $z_0$, allowing "label semantics" and "original distilled semantics" to coexist. To maximize efficiency, the authors approximate $\nabla\mathcal{G}_t$ as $(\hat z_t-z_0)\sigma_t$, eliminating gradient calculation and keeping total sampling time at 2.48s/image compared to the original DiT's 2.41s/image.
    - **Design Motivation**: Solves the "semantic averaging" issue caused by using CFG alone. Fig. 4 shows that if $\gamma$ is too small, SG fails; if $\gamma$ is too large, sampling over-relies on the distilled image, drowning out the label signal.

3. **Semantic Fusion (SF) — Applying Guidance during "Semantic Formation"**:
    - **Function**: Segments the 50-step DDIM reverse process into the Chaos Phase (CP, $t_r\sim t_h=50\to 40$), Semantic Phase (SP, $t_h\sim t_l=40\to 25$), and Refinement Phase (RP, $t_l\sim 1=25\to 1$). CP and RP only run CFG, while SP runs SG + CFG + latent inheritance.
    - **Mechanism**: Leverages the three-stage diffusion theory (Yu et al., 2023a; Chen et al., 2025), which states that semantic content is primarily determined during the SP stage, whereas CP is noise-dominated and RP fills in high-frequency details. Applying SG throughout all stages introduces artifacts in CP and reduces realism in RP. Restricting SG to the SP stage preserves loyalty to the distilled image without contaminating early chaos sampling or late detail sharpening. Tab. 6 shows SI+SG (without SF) yields 21.1% / 28.3% (IPC=1/10) on ImageFruit MTT, while adding SF increases this to 22.3% / 29.8%.
    - **Design Motivation**: Prevents semantic blurring and artifacts caused by continuous guidance (Fig. 5), essentially utilizing the prior that different stages of diffusion sampling serve different functions.

### Loss & Training
DIVER Stage II is **fully training-free**: both the diffusion model and VAE use frozen pre-trained weights (DiT-XL/2 + vae-ft-mse). No fine-tuning or additional training is involved. The pipeline has only three hyperparameters: $t_f=25$, $[t_h, t_l]=[40, 25]$, and $\gamma=0.1$ (0.02 for NCFM), with 50-step DDIM CFG. Downstream training of the target architecture $\varPhi^v$ follows the hard-label/soft-label protocols of the original DD methods (KL divergence soft labels for ImageNet-1K, hard labels for others).

## Key Experimental Results

### Main Results

Datasets: ImageNet-1K (224×224) + 12 ImageNet subsets (ImageFruit/Woof/Meow/Squawk/Nette/Yellow/A~E/IDC, 128×128). All images were resized to 256×256 for DiT. Baselines cover three major streams: classic pixel optimization (DM/MTT/EDF/NCFM), bi-moment matching (SRe²L/G-VBSM), and diffusion-native (Minimax/D⁴M/MGD³). Proxy structure $\varPhi^p$=ConvNet; target architectures $\varPhi^v$ = ResNet18 / ShuffleNet-V2 / MobileNet-V2 / EfficientNet-B0 / ViT-b/16. Results averaged over 5 trials.

| Config (ImageFruit) | IPC=1 | IPC=10 | Gain Source |
|---|---|---|---|
| MTT (baseline) | 15.4 ± 1.6 | 18.9 ± 1.4 | Pixel-space Trajectory Matching |
| MTT + DIVER | **22.3 ± 1.8** | **29.8 ± 2.0** | +6.9 / +10.9 |
| EDF (baseline) | 16.2 ± 1.8 | 23.2 ± 2.1 | SOTA Pixel Method |
| EDF + DIVER | **20.3 ± 1.9** | **34.5 ± 2.3** | +4.1 / +11.3 |
| DM | 11.3 ± 1.4 | 19.3 ± 1.5 | Distribution Matching |
| DM + DIVER | **18.5 ± 1.9** | **22.4 ± 1.8** | +7.2 / +3.1 |
| NCFM | 17.1 ± 1.6 | 20.5 ± 2.5 | Current SOTA |
| NCFM + DIVER | **18.8 ± 1.7** | **25.5 ± 1.6** | +1.7 / +5.0 |

Cross-architecture generalization (Tab. 2, ImageA, DC + DIVER at IPC=1): Improved from 24.9% (DC) to 30%+; outperformed GLaD across multiple subsets. On the full ImageNet-1K (Tab. 4), DIVER as a plugin for Minimax-tuned MGD³ achieved a new SOTA for ResNet-18 at IPC=10.

Efficiency: On a single RTX-4090, each synthetic image takes 2.48s and 4.02 GB VRAM, nearly identical to pure DiT (2.41s).

### Ablation Study (Tab. 6, ImageFruit + MTT)

| Config | IPC=1 | IPC=10 | Description |
|---|---|---|---|
| Random (No DD, No DIVER) | 14.1 ± 1.4 | 19.6 ± 1.8 | Random selection baseline |
| DD only (MTT) | 15.4 ± 1.6 | 18.9 ± 1.4 | Classic DD; IPC=10 even loses to random |
| Random* + Full DIVER | 14.6 ± 1.8 | 21.7 ± 1.6 | Proves DIVER doesn't rely solely on DiT prior |
| MTT + raw DiT (No SI/SG/SF) | 17.8 ± 1.2 | 23.4 ± 1.3 | Pure DiT pass also provides gains |
| MTT + SI only | 19.5 ± 1.4 | 26.2 ± 1.5 | SI alone contributes +4.1 / +7.3 |
| MTT + SG only | 20.4 ± 1.7 | 27.8 ± 1.9 | SG provides the largest single contribution |
| MTT + SI + SG | 21.1 ± 1.9 | 28.3 ± 1.6 | SI+SG shows saturation |
| MTT + SI + SG + SF (full) | **22.3 ± 1.8** | **29.8 ± 2.0** | SF adds another +1.2 / +1.5 |

### Key Findings
- **SG is the most significant single-point contributor**: Removing SG causes a larger drop than removing SI, indicating that "anchoring the latent back to the distilled image via L2" provides higher marginal benefit for cross-architecture generalization than "noise filtering via VAE." This supports the core argument that distilled images contain dataset-level semantics.
- **Random* experiment is a crucial proof**: Feeding random original images into DIVER instead of distilled images yields only 21.7% at IPC=10, which is lower than the 23.4% from pure DiT. This confirms that DIVER's gains originate from "compressed semantics" in distilled images rather than being a "free gift" from the DiT prior.
- **Structural trade-off**: Tab. 8 shows that after DIVER processing, accuracy on the original proxy (ConvNet) actually drops, while it increases significantly on heterogeneous architectures. This validates the hypothesis that "distilled images overfit ConvNet" and shows DIVER strategically trades a few points on ConvNet for 5–10 points on heterogeneous architectures.
- **Robust across diffusion models**: SD-V1.5, DiT, and SiT are all compatible with DIVER. The smaller gain from SD-V1.5 is attributed to the U-Net vs Transformer architecture difference and non-ImageNet pre-training, which also suggests DIVER does not rely on "data leakage."
- **$t_f=25$ and $\gamma\approx 0.1$ are universal sweet spots**, with parameter curves showing an inverted U-shape.

## Highlights & Insights
- **Redefines the DD problem as a two-stage DD + DDD process**: This is a paradigm-level decoupling. The authors point out that the classic DD formula (Eqn. 2) is solved on $\varPhi^p$ but evaluated on $\varPhi^v$, which is inherently biased. Defining the DDD task (Eqn. 6) separately bridges this gap. This "patching a fundamental flaw" approach is more structural than simply designing a new matching loss.
- **Treats distilled images as latent starting points rather than generation targets**: This is the fundamental difference between DIVER and diffusion-native DD methods like D⁴M / MGD³ / Minimax. The latter discard classic DD in favor of coreset selection or retraining diffusion models; DIVER "rescues" the vast existing distilled datasets from classic DD, serving as a training-free universal upgrader with high engineering value.
- **Three-stage guidance scheduling is a transferable design pattern**: The SF idea of using "when to guide" as a design dimension (no guidance in CP, strong guidance in SP, no guidance in RP) can be directly applied to image editing, stylization, or any conditional generation scenario that balances condition signals with sampling freedom.
- **Zero-gradient approximation + latent space guidance makes computation almost free**: Approximating $\nabla \mathcal{G}_t$ as $(\hat z_t-z_0)\sigma_t$ is a practical engineering trick that allows DIVER to maintain 2.48s/image and 4GB VRAM.

## Limitations & Future Work
- The authors acknowledge that the method depends heavily on the quality of original distilled images. If the Stage I images are poor and lack class recognizability, Stage II cannot recover them (as hinted by the Random* row in Tab. 6).
- The diffusion base has currently only been validated on DiT/SiT/SD-V1.5 and has not been integrated with more modern flow matching or consistency models.
- **Personal note on limitations**: (1) Evaluation is mostly on ImageNet-based data; OOD domains like medical or satellite imagery are not verified. (2) DiT-XL/2 itself was trained on ImageNet; while the SD-V1.5 results argue against data leakage, absolute causal evidence is lacking. (3) The 256×256 resize requirement restricts high-resolution small dataset scenarios. (4) While few, the hyperparameters ($t_f, t_h, t_l, \gamma$) have not been systematically studied for cross-dataset sensitivity.
- **Future directions**: Making SG learnable (e.g., using distilled images as contrastive learning targets), making SF phase splitting adaptive (using latent norm/FID), and integrating with latent consistency models to reduce 50 steps to a few.

## Related Work & Insights
- **vs GLaD (Cazenavette et al., 2023)**: GLaD performs DD matching in GAN latent space but remains single-stage and requires expensive inner loops. DIVER is two-stage, training-free, and reuses outputs from any existing DD.
- **vs Minimax / D⁴M / MGD³ (Diffusion-native DD)**: These methods use diffusion to generate proxy samples directly from the original set, effectively returning to a coreset selection route. DIVER treats diffusion as a "post-processing repair tool" for classic DD, proving complementarity (Tab. 4 shows DIVER + MGD³ yields further gains).
- **vs SDEdit / Latent Editing methods**: These use natural images as starting points to edit local attributes. DIVER uses non-natural distilled images for "semantic recovery." While SG loss $(\hat z_t-z_0)^2$ resembles SDEdit's trajectory constraint, the target object (distilled vs natural) fundamentally changes the semantic outcome.
- **Insight**: In any scenario where intermediate products distilled via proxy targets need post-processing repair (e.g., logit repair in KD, token repair in prompt distillation, weight repair in pruning), "passing through a pre-trained model constrained by natural distribution + staged conditional guidance" could be a universal repair paradigm.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Two-stage decoupling + DDD task definition are conceptual breakthroughs, though SI/SG/SF components borrow from LDM/SDEdit/three-stage diffusion priors.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 6 DD baselines × 12 datasets × 5 architectures × multiple IPC settings, plus robust ablations like Random*.
- **Writing Quality**: ⭐⭐⭐⭐ Motivations are clear; DD/DDD formal definitions are precise. Method section notation is dense, but Fig. 2/3 are very helpful.
- **Value**: ⭐⭐⭐⭐⭐ As a plugin that revives the entire classic DD ecosystem without retraining, it has high engineering value and provides a clear path for combining diffusion priors with distilled data.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mind Your Margin and Boundary: Are Your Distilled Datasets Truly Robust?](mind_your_margin_and_boundary_are_your_distilled_datasets_truly_robust.md)
- [\[ICML 2026\] IDLM: Inverse-distilled Diffusion Language Models](idlm_inverse-distilled_diffusion_language_models.md)
- [\[ICLR 2026\] PASER: Post-Training Data Selection for Efficient Pruned Large Language Model Recovery](../../ICLR2026/model_compression/paser_post-training_data_selection_for_efficient_pruned_large_language_model_rec.md)
- [\[ICML 2026\] Hard Labels In! Rethinking the Role of Hard Labels in Mitigating Local Semantic Drift](hard_labels_in_rethinking_the_role_of_hard_labels_in_mitigating_local_semantic_d.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](../../ICLR2026/model_compression/a_recovery_guarantee_for_sparse_neural_networks.md)

</div>

<!-- RELATED:END -->
