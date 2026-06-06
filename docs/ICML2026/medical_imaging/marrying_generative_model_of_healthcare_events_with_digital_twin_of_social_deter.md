---
title: >-
  [Paper Note] Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning
description: >-
  [ICML 2026][Medical Imaging][digital twin] This paper proposes DiffDT: a conditional Latent Diffusion framework that connects electronic health records (ICD-coded event sequences) with multi-organ biomarker digital twins…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "digital twin"
  - "ICD autoregression"
  - "latent diffusion"
  - "SPD manifold"
  - "Cholesky decomposition"
  - "multi-organ biomarker"
date: 2026-05-08
content_hash: fdfbc6978cc6270c
---

# Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.09771](https://arxiv.org/abs/2605.09771)  
**Code**: None  
**Area**: Medical Imaging / Generative Models / Digital Twin / Disease Prediction  
**Keywords**: digital twin, ICD autoregression, latent diffusion, SPD manifold, Cholesky decomposition, multi-organ biomarker

## TL;DR
This paper proposes DiffDT: a conditional Latent Diffusion framework that connects electronic health records (ICD-coded event sequences) with multi-organ biomarker digital twins (tabular features derived from brain/heart/liver/kidney imaging and brain functional connectivity SPD matrices). The key innovation is an SPD-VQVAE based on Cholesky decomposition, which reduces $\mathcal{O}(N^3)$ SPD manifold diffusion to a manifold-preserving and efficient latent space. An AR model then performs multi-pathway disease reasoning via the mediation path “generate digital twin → predict next ICD.” On UKB, next-event prediction AUC for 1944 diseases reaches 0.91, setting a new SOTA.

## Background & Motivation

**Background**: There are two mainstream routes in medical AI—(a) **EHR-to-event** uses transformer autoregression to model ICD event sequences (MOTOR, Delphi), treating disease progression as next-token prediction but ignoring concrete physiological biomarkers; (b) **DT-to-event** uses pretrained imaging foundation models (BrainMass, NeuroPath) to predict diseases from single-organ in-vivo biomarkers, but lacks cross-temporal causal chains. Both struggle with long-term, multi-pathway, personalized disease reasoning.

**Limitations of Prior Work**: (1) Pure EHR approaches learn healthcare utilization patterns rather than disease mechanisms; Figure 2 shows prediction accuracy is negatively correlated with the “semantic distance” between current and historical diseases—predictions for diseases far from the main diagnosis chapter are poor, exposing the lack of multi-pathway probabilistic mediation. (2) Pure DT approaches only use current biomarkers and cannot predict future physiological states based on past medical history. (3) Existing AR-diffusion hybrid architectures (e.g., ye2025hybrid, HybridVLA) are designed in Euclidean space; directly adding Gaussian noise to brain functional connectivity SPD matrices destroys geometry (no longer positive definite). (4) Geometric diffusion methods like SPD-DDPM use affine-invariant/log-Euclidean metrics, which are rigorous but computationally prohibitive at $\mathcal{O}(N^3)$ complexity (AAL atlas $N=116$).

**Key Challenge**: To combine EHR temporal causality and multi-organ physiological states, it is necessary to (i) make the generative model closely follow ICD history, (ii) ensure generated brain networks remain on the SPD manifold, and (iii) keep complexity low enough to train on 50K+ subjects; the first two are inherently conflicting under Euclidean latent diffusion frameworks.

**Goal**: (1) Establish an SDoH-to-event paradigm: use ICD-coded SDoH proxies (including chapter Z and V–Y) as conditions, generate multi-organ DTs as biological mediators, then predict future ICDs; (2) Design a manifold-preserving SPD latent diffusion to reduce $\mathcal{O}(N^3)$ to a tractable scale; (3) Validate next-event prediction for 1944 long-tail diseases on UKB.

**Key Insight**: Cholesky decomposition $M = LL^\top$ provides a **unique and smooth** factorization for the SPD manifold (Theorem 3.1: $\mathcal{S}_{++}^N \to \mathcal{L}_{++}^N$ is a diffeomorphism). Diffusion on Cholesky factors allows use of Euclidean machinery, while $LL^\top$ reconstruction guarantees the output remains SPD.

**Core Idea**: An SPD-VQVAE encodes brain functional connectivity matrices into a discrete latent space (decoder outputs a lower-triangular matrix, then $LL^\top$ reconstructs SPD), and Cholesky LDM is run in this latent space conditioned on ICD history. Tabular biomarkers use TabDiff for hybrid SDE/absorbing diffusion. Finally, the AR model treats “generate DT → predict next ICD” as mediation inference via $P(\text{Future}|\text{Biomarker})\cdot P(\text{Biomarker}|\text{History})$.

## Method

### Overall Architecture
Three core components work in concert: (1) **Adaptive medical history tokenizer + AR model $\phi$**: ICD sequences are mapped to a unified yearly time grid, with `[healthy]` tokens filling years without diagnoses; token embedding + age embedding are fed to causal self-attention for next-token prediction. (2) **Multi-organ DT generation**: Tabular biomarkers use TabDiff-style hybrid diffusion (continuous via VE SDE, discrete via absorbing process); brain functional connectivity uses SPD-VQVAE + Cholesky LDM. (3) **Prediction model**: Organ-specific FMs (BrainMass for brain, Transformer encoder for tabular) are fine-tuned on generated DTs for next ICD classification. **At inference**, for each causal node to be predicted, the AR model encodes past ICDs → diffusion model generates hypothetical multi-organ DT → fine-tuned FM predicts next ICD, completing multi-pathway mediation via $\sum_{\text{organ}} P(s_{t+1}|\text{DT}_t^{\text{organ}}) P(\text{DT}_t^{\text{organ}}|S_{<t})$.

### Key Designs

1. **SPD-VQVAE: Manifold-Preserving VQVAE for SPD via Cholesky Decomposition**:

    - **Function**: Compresses $116\times 116$ brain functional connectivity SPD matrices into a discrete latent space, with decoding guaranteed to return to the SPD manifold.
    - **Mechanism**: Encoder $\mathcal{E}$ uses an MLP to flatten $\mathbf{M}$ and project to $z_e \in \mathbb{R}^{N_q \times d}$, then quantizes to codebook $\mathcal{Z} \in \mathbb{R}^{N_{code} \times d}$ to obtain $z$. The decoder $\mathcal{D}$ **does not directly predict $\hat{\mathbf{M}}$**, but predicts a lower-triangular $\hat L$ (diagonal via softplus for positivity), then $\hat{\mathbf{M}} = \hat L \hat L^\top$. The loss $\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{SPD}}(L, \hat L) + \mathcal{L}_{\text{recon}}(\mathbf{M}, \hat{\mathbf{M}}) + \|\text{sg}[z_e]-e\|_2^2 + \beta\|z_e-\text{sg}[e]\|_2^2$ supervises Cholesky factors, final SPD reconstruction, codebook, and commitment.
    - **Design Motivation**: Directly decoding SPD matrices in VQVAE lacks structural guarantees; adding Gaussian noise immediately breaks positive definiteness. SPD-DDPM uses affine-invariant metrics but at $\mathcal{O}(N^3)$. Theorem 3.1 ensures Cholesky is a diffeomorphism between SPD and lower-triangular positive diagonal matrices, so training in factor space is equivalent to training on the manifold, but with Euclidean MSE loss and $\mathcal{O}(N^2)$ complexity. This “**find a diffeomorphism to automatically satisfy manifold constraints**” is the most elegant engineering trick in the paper.

2. **Cholesky Conditional LDM + Cross-Attention Injection of ICD History**:

    - **Function**: Runs conditional diffusion in the SPD-VQVAE latent space, conditioned on AR model’s medical history embedding $\hat{\mathbf{y}}$.
    - **Mechanism**: U-Net backbone uses Residual MLP Blocks for latent vector sequences. Conditioning is injected via: (i) diffusion timestep embedding $\text{Embed}_\text{diff}(t)$ added to each layer input; (ii) Cross-attention uses $\hat{\mathbf{y}} \in \mathbb{R}^{T \times d_\phi}$ as key/value, $\hat z$ as query, $Q = \hat z \hat{\bm{\alpha}}_h$, $K = \hat{\mathbf{y}}\hat{\bm{\beta}}_h$, $V = \hat{\mathbf{y}}\hat{\bm{\gamma}}_h$, output $\hat z = \text{Softmax}(QK^\top/\sqrt{C_{hid}})V$. Loss is standard noise prediction $\mathcal{L}_{\text{LDM}} = \mathbb{E}\|\epsilon - \epsilon_\theta(z_t, t, \hat{\mathbf{y}})\|^2$. **SPD-VQVAE-Dual**: Empirically, a single codebook tends to generate similar “average brains,” lacking personalized patterns; thus, two SPD-VQVAEs are used for low- and high-frequency Fourier components (threshold 25), so LDM learns both “global structure” and “personalized details.”
    - **Design Motivation**: Cross-attention allows generation to condition not just on recent events but to attend to any key ICD in history (e.g., how a hypertension diagnosis years ago affects today’s brain network). The dual design is an engineering finding—pure generative models on highly symmetric medical data tend to mode collapse; frequency splitting enforces diversity.

3. **Adaptive ICD Tokenizer + Multi-Pathway Mediation Reasoning**:

    - **Function**: Enables AR model $\phi$ to handle sparse, uneven ICD timestamps and provide causality-aware conditions for downstream generative models.
    - **Mechanism**: Constructs a unified annual time grid $\tau = (\tau_t \mid \tau_{t+1}-\tau_t \in \{0,1\})$ covering the cohort’s age range; for each year $\tau_t$: if subject has ICD $c$, $s_t = c$, else fill `[healthy]`. Input token $\mathbf{y}_t = \text{Embed}_\text{ICD}(s_t) + \text{Embed}_\text{age}(\tau_t)$ is fed to transformer, next-token target $\mathcal{L}_\text{AR} = -\sum_t \log p(s_{t+1}|s_{\leq t}; \phi)$. **At inference**, for each mediation node, $\phi$’s output is used as $\hat{\mathbf{y}}$ for LDM to generate multi-organ DT, then fine-tuned FM computes $P(s_{t+1}|\hat{\mathbf{M}}_t)$, forming an explicit causal chain from “current observed events” → “physiological biomarker state” → “future events.”
    - **Design Motivation**: Previous AR medical models (Delphi) use time-to-event embedding, with token intervals up to decades, causing discontinuous conditioning for downstream diffusion. This work’s unified annual grid + `[healthy]` filler ensures temporally dense conditioning, stabilizing cross-attention in diffusion.

### Loss & Training

Three stages: (i) AR model $\phi$ is pretrained on 7.28M ICD tokens / 448,651 subjects for next-token prediction; (ii) SPD-VQVAE and TabDiff are trained on paired imaging + ICD data (brain 44,834; heart 23,987; liver 28,722; kidney 32,155 samples), with losses $\mathcal{L}_{\text{VAE}}$ and $\mathcal{L}_{tab} = \lambda_{num}\mathcal{L}_{num} + \lambda_{cat}\mathcal{L}_{cat}$; (iii) Organ FMs are fine-tuned on generated DTs for next ICD classification. Tabular generation uses Classifier-Free Guidance: $\tilde\mu^{num}(\Gamma_t, S, t) = (1+\omega)\mu_\theta^{num}(\Gamma_t, S, t) - \omega\mu_\phi(\Gamma_t, t)$. Subject-level 80:20 splits are strictly maintained across all stages to prevent data leakage.

## Key Experimental Results

### Main Results
On the UKB dataset, next-event prediction for 1944 diseases:

| Method | Backbone | AUC ↑ | F1 ↑ |
|---|---|---|---|
| Delphi | GPT2 | 0.6994 ± 0.091 | 7.09 ± 7.56 |
| Delphi | Qwen3 | 0.8931 ± 0.055 | 18.17 ± 20.67 |
| **DiffDT** | GPT2 | **0.9087 ± 0.050** | **18.60 ± 16.29** |
| **DiffDT** | Qwen3 | **0.9171 ± 0.049** | **20.92 ± 20.40** |

F1 by organ mediation group (vs DT-to-event baselines):

| Mediation Organ | Brain F1 | Heart F1 | Liver F1 | Kidney F1 |
|---|---|---|---|---|
| NeuroPath | 56.53 | 48.96 | 54.20 | 54.57 |
| BrainMass | 47.18 | 56.63 | 58.81 | 50.43 |
| **DiffDT-Brain** | **65.14** | 53.74 | 60.44 | 54.97 |
| **DiffDT-Heart** | 58.00 | **58.50** | 56.17 | 58.18 |
| **DiffDT-Liver** | 59.88 | 52.23 | **61.65** | 53.52 |
| **DiffDT-Kidney** | 53.91 | 54.72 | 57.92 | **64.32** |

**Key Observation**: Using the same pretrained BrainMass as predictor, DiffDT-Brain (generated DT as input) outperforms BrainMass (real GT as input) by ≈18 pp F1, indicating multi-pathway mediation captures additional causal signals from ICD history.

### Ablation Study

| LDM Config | RMSE ↓ | WD ↓ | r ↑ | mAcc ↑ |
|---|---|---|---|---|
| Plain VQVAE | 0.261 | 7.110 | 0.503 | 90.87 |
| SPD-VQVAE | 0.220 | 6.019 | 0.677 | 95.71 |
| **SPD-VQVAE-Dual** | **0.203** | **5.841** | **0.726** | **98.36** |

Tabular DT: heart RMSE 0.265 / WD 17.27 (112 traits), liver 0.184 / 2.48, kidney 0.146 / 0.99; organ mediation F1 correlates positively with generation quality (higher quality, better downstream classification). **Counterfactual evaluation**: Replacing an exposure ICD with `[healthy]` to generate do(healthy) DT, compared to real healthy vs diseased subjects by FID/WD/r, do(healthy) is significantly closer to GT healthy ($p=2.5e\text{-}5$ on FID), demonstrating biological plausibility of DT counterfactual interventions.

### Key Findings
- **Multi-pathway mediation is effective**: Generated DT as input > real GT as input (DiffDT-Brain outperforms BrainMass by 18 pp F1), indicating LDM injects causal signals from ICD history into generated DTs, and FM indirectly accesses richer disease context via DT.
- **SPD-VQVAE-Dual is essential**: Single-branch SPD-VQVAE already surpasses plain VQVAE, but dual low/high-frequency branches further improve by 4–7%, showing that global structure and personalized details in medical imaging require decoupled learning.
- **Backbone upgrade and DiffDT gains are orthogonal**: Replacing GPT2 with Qwen3, DiffDT still consistently adds +0.12 AUC, indicating mediation benefits are not subsumed by larger LLMs.
- **Efficiency**: Cholesky LDM is much faster per step than SPD-DDPM (Fig. 6); end-to-end, single-subject mediation takes 1.2s, five mediations only 5.6s, enabling large-scale cohort inference.
- **Off-diagonal pairs are performance amplifiers**: Fig. 4 shows AUC improvement is most pronounced for ICD pairs across chapters (semantically distant), directly addressing EHR-only model weaknesses.

## Highlights & Insights
- **Cholesky decomposition as SPD manifold diffeomorphism is the core trick**: Compared to SPD-DDPM’s affine-invariant metric requiring eigendecomposition, Theorem 3.1 enables diffusion on lower-triangular factors, strictly preserving the SPD manifold and reducing per-step complexity from $\mathcal{O}(N^3)$ to $\mathcal{O}(N^2)$. This “find a diffeomorphism to absorb manifold constraints” approach is transferable to any generative task with geometric structure (e.g., SO(3) rotations, covariance matrices, probability simplices).
- **SDoH-to-event paradigm reshapes EHR modeling**: Using ICD-coded SDoH proxies as digital surrogates allows large-scale EHRs without explicit SDoH fields (like UKB) to support social determinant analysis, expanding EHR model applicability.
- **Generated DT outperforms real GT as input**: This counterintuitive result shows that **the generative model encodes historical signals into latent biological states**; the FM receives not just a current physiological snapshot but a causally structured “augmented observation”—equivalent to distilling AR model’s temporal memory into image representations.
- **Dual SPD-VQVAE’s Fourier frequency splitting**: Medical images have strong structural symmetry; a single codebook easily collapses to “average healthy brain.” Frequency decoupling forces the generator to reproduce both global structure and personalized details, serving as a practical defense against mode collapse.

## Limitations & Future Work
- Data source is single (UKB), predominantly European white, relatively healthy middle-aged/elderly; model’s transferability to low-income countries, children, or critically ill patients is untested.
- Only four organs (brain/heart/liver/kidney) are mediated; other key organs (lung, digestive, endocrine) are not covered. Extending to more organs requires retraining corresponding SPD-VQVAE/TabDiff, with high engineering cost.
- F1 remains in the 50–65% range; ultra-long-tail ICDs (rare diseases seen only a few times per year) remain hard to predict; the paper acknowledges macro-average is dragged down by the long tail.
- Counterfactual evaluation is only at the group level (FID/WD) to show do(healthy) DT approaches real healthy distribution; no rigorous individual-level causal validation (e.g., true longitudinal two-timepoint interventions).
- SPD-VQVAE-Dual uses a fixed Fourier threshold of 25 for frequency splitting, with hyperparameters selected via ablation; future work could explore learnable frequency partitioning or graph spectral decomposition.
- Future directions include extending Cholesky LDM to other geometric structures (e.g., brain surface mesh, cellular ECM topology), or integrating RAG-like mechanisms for AR retrieval of similar cases to enhance conditioning.

## Related Work & Insights
- **vs Delphi / MOTOR (pure EHR AR)**: These only model ICD causality, lacking biological mediation; DiffDT uses generated DT as mediator, raising AUC from 0.70–0.89 to 0.91, especially reversing negative correlation for off-diagonal pairs in EHR-only models.
- **vs BrainMass / NeuroPath (pure DT-to-event)**: These only use real biomarkers for prediction; DiffDT’s generated “history-infused” DT is 18 pp more accurate than real GT, proving the value of mediation.
- **vs SPD-DDPM (li2024spd) / Riemannian Flow Matching**: These rigorously diffuse on SPD at $\mathcal{O}(N^3)$; this work diffuses on $\mathcal{L}_{++}^N$ via Cholesky, achieving geometric alignment and efficiency.
- **vs HybridVLA / vision-language AR-diffusion hybrids**: These combine AR + diffusion in Euclidean space; DiffDT handles non-Euclidean SPD manifolds + long-term sparse ICD conditioning, fundamentally different problem structure.
- **Insights**: (1) “Diffusing on manifold-constrained latent spaces, letting constraints be automatically satisfied by decoder geometry” is a general recipe, applicable to molecular conformations, pose estimation, point cloud generation; (2) “Generated DT as input outperforms real GT” suggests generative models are not just for data augmentation, but can serve as information-distilling mediators.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ SDoH-to-event paradigm + Cholesky LDM is the first strictly manifold-preserving and scalable solution in this direction
- Experimental Thoroughness: ⭐⭐⭐⭐ 33K–44K real UKB subjects + 1944 diseases + 4 organs + counterfactual evaluation + backbone scaling controls
- Writing Quality: ⭐⭐⭐⭐ Geometric motivation—algorithm—experiments—counterfactuals are tightly linked; occasional formula typos do not affect understanding
- Value: ⭐⭐⭐⭐⭐ Provides a framework for digital twin medical AI capable of long-term multi-pathway reasoning, with strong potential for clinical decision support deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Evidential Reasoning Advances Interpretable Real-World Disease Screening](evidential_reasoning_advances_interpretable_real-world_disease_screening.md)
- [\[NeurIPS 2025\] MIRA: Medical Time Series Foundation Model for Real-World Health Data](../../NeurIPS2025/medical_imaging/mira_medical_time_series_foundation_model_for_real-world_health_data.md)
- [\[ICLR 2026\] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](../../ICLR2026/medical_imaging/can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)
- [\[ICML 2026\] Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)
- [\[ICLR 2026\] Human Behavior Atlas: Benchmarking Unified Psychological and Social Behavior Understanding](../../ICLR2026/medical_imaging/human_behavior_atlas_benchmarking_unified_psychological_and_social_behavior_unde.md)

</div>

<!-- RELATED:END -->
