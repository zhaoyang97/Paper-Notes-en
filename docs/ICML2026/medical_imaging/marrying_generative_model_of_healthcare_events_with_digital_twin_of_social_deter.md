---
title: >-
  [Paper Note] Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning
description: >-
  [ICML 2026][Medical Imaging][digital twin] This paper proposes DiffDT: a conditional Latent Diffusion framework that bridges Electronic Health Records (ICD-coded event sequences) with multi-organ biomarker digital twins…
tags:
  - "ICML 2026"
  - "Medical Imaging"
  - "digital twin"
  - "ICD autoregressive"
  - "latent diffusion"
  - "SPD manifold"
  - "Cholesky decomposition"
  - "multi-organ biomarkers"
date: 2026-05-08
content_hash: 8bb14e80ffba615f
---

# Marrying Generative Model of Healthcare Events with Digital Twin of Social Determinants of Health for Disease Reasoning

**Conference**: ICML 2026  
**arXiv**: [2605.09771](https://arxiv.org/abs/2605.09771)  
**Code**: None  
**Area**: Medical Imaging / Generative Models / Digital Twin / Disease Prediction  
**Keywords**: digital twin, ICD autoregressive, latent diffusion, SPD manifold, Cholesky decomposition, multi-organ biomarkers

## TL;DR
This paper proposes DiffDT: a conditional Latent Diffusion framework that bridges Electronic Health Records (ICD-coded event sequences) with multi-organ biomarker digital twins (imaging-derived tabular features for brain/heart/liver/kidney and brain functional connectivity SPD matrices). The key innovation is an SPD-VQVAE based on Cholesky decomposition that reduces $\mathcal{O}(N^3)$ SPD manifold diffusion to a manifold-preserving and efficient latent space. This allows an AR model to perform multi-pathway disease reasoning via the intermediary path of "generating digital twins $\rightarrow$ predicting next ICD." On the UKB dataset, the AUC for predicting the next occurrence of 1,944 disease categories reached 0.91, setting a new SOTA.

## Background & Motivation

**Background**: Two mainstream routes in medical AI exist: (a) **EHR-to-event**, which uses transformer autoregression to learn ICD event sequences (e.g., MOTOR, Delphi), treating disease progression as next-token prediction but ignoring specific physiological biomarkers; (b) **DT-to-event**, which uses pre-trained imaging foundation models (e.g., BrainMass, NeuroPath) to predict diseases from single-organ in-vivo biomarkers but lacks cross-temporal causal chains. Neither can perform long-term, multi-pathway, personalized disease reasoning.

**Limitations of Prior Work**: (1) Pure EHR routes learn medical utilization patterns rather than disease mechanisms; Figure 2 shows that prediction accuracy is negatively correlated with the "semantic distance between current and historical diseases"—prediction is poor for diseases far from the primary diagnosis chapter, exposing the lack of multi-pathway probabilistic mediation. (2) Pure DT routes rely only on current biomarkers and cannot predict future physiological states based on past medical history. (3) Existing AR-diffusion hybrid architectures (e.g., ye2025hybrid, HybridVLA) are designed in Euclidean space; applying Gaussian noise directly to brain functional connectivity SPD matrices destroys the geometry (violating positive definiteness). (4) Geometric diffusion methods like SPD-DDPM using affine-invariant / log-Euclidean metrics are rigorous but computationally prohibitive due to $\mathcal{O}(N^3)$ complexity ($N=116$ under the AAL atlas).

**Key Challenge**: To combine EHR temporal causality with multi-organ physiological states, one must: (i) make the generative model responsive to ICD history, (ii) ensure generated brain networks remain on the SPD manifold, and (iii) maintain low enough complexity to train on 50K+ subjects. The first two requirements naturally conflict within Euclidean latent diffusion frameworks.

**Goal**: (1) Establish an SDoH-to-event paradigm: use ICD-coded SDoH proxies (including chapters Z and V–Y) as conditions to generate multi-organ DTs as biological mediators, then predict future ICDs. (2) Design manifold-preserving SPD latent diffusion that reduces $\mathcal{O}(N^3)$ to a computable scale. (3) Validate on next-occurrence prediction for 1,944 long-tail disease categories in UKB.

**Key Insight**: Cholesky decomposition $M = LL^\top$ provides a **unique and smooth** factorization for SPD manifolds (Theorem 3.1: $\mathcal{S}_{++}^N \to \mathcal{L}_{++}^N$ is a diffeomorphism). Performing diffusion on Cholesky factors allows the use of Euclidean machinery while guaranteeing the reconstruction remains SPD through $LL^\top$.

**Core Idea**: An SPD-VQVAE encodes brain functional connectivity matrices into a discrete latent space (where the decoder outputs a lower-triangular matrix, then computes $LL^\top$ for SPD reconstruction). A Cholesky LDM then runs on this latent space conditioned on ICD history. Tabular biomarkers use TabDiff with mixed SDE/absorbing diffusion. Finally, an AR model treats "generating DT $\rightarrow$ predicting next ICD" as a mediated reasoning path: $P(\text{Future}|\text{Biomarker})\cdot P(\text{Biomarker}|\text{History})$.

## Method

### Overall Architecture
Three core components collaborate: (1) **Adaptive Medical History Tokenizer + AR Model $\phi$**: ICD sequences are mapped to a uniform temporal grid with 1-year granularity, using `[healthy]` tokens for years without events. Token embeddings + age embeddings are fed into a causal self-attention block for next-token prediction. (2) **Multi-organ DT Generation**: Tabular biomarkers use TabDiff-style mixed diffusion (continuous via VE SDE, discrete via absorbing process). Brain functional connectivity uses SPD-VQVAE + Cholesky LDM. (3) **Prediction Model**: Organ-specific FMs (BrainMass for brain, Transformer encoder for tabular) are fine-tuned on generated DTs for next-ICD classification. **During inference**, for each multi-pathway causal node, history is encoded by the AR model $\rightarrow$ hypothetical multi-organ DTs are generated by the diffusion model $\rightarrow$ fine-tuned FMs predict the next ICD, completing the multi-pathway mediation of $\sum_{\text{organ}} P(s_{t+1}|\text{DT}_t^{\text{organ}}) P(\text{DT}_t^{\text{organ}}|S_{<t})$.

### Key Designs

1.  **SPD-VQVAE: Cholesky-based Manifold-Preserving VQVAE**:
    *   **Function**: Compresses $116\times 116$ brain functional connectivity SPD matrices into a discrete latent space while ensuring decodings always land on the SPD manifold.
    *   **Mechanism**: The encoder $\mathcal{E}$ uses an MLP to flatten and project $\mathbf{M}$ to $z_e \in \mathbb{R}^{N_q \times d}$, then quantizes to a codebook $\mathcal{Z} \in \mathbb{R}^{N_{code} \times d}$ to get $z$. The decoder $\mathcal{D}$ **does not directly predict $\hat{\mathbf{M}}$**; it predicts a lower-triangular $\hat L$ (with softplus on the diagonal to ensure positivity), then $\hat{\mathbf{M}} = \hat L \hat L^\top$. The loss $\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{SPD}}(L, \hat L) + \mathcal{L}_{\text{recon}}(\mathbf{M}, \hat{\mathbf{M}}) + \|\text{sg}[z_e]-e\|_2^2 + \beta\|z_e-\text{sg}[e]\|_2^2$ simultaneously supervises Cholesky factors, final SPD reconstruction, codebook, and commitment.
    *   **Design Motivation**: Direct VQVAE decoding of SPD matrices has no structural guarantees, and adding Gaussian noise immediately breaks positive definiteness. SPD-DDPM diffusion under affine-invariant metrics is $\mathcal{O}(N^3)$. Theorem 3.1 guarantees that the Cholesky map is a diffeomorphism between SPD and lower-triangular matrices with positive diagonals. Training in this factor space is equivalent to training on the manifold using Euclidean MSE loss, reducing complexity to $\mathcal{O}(N^2)$. This "finding a diffeomorphism to satisfy manifold constraints automatically" is the most elegant engineering trick in the paper.

2.  **Cholesky Conditional LDM + Cross-attention for ICD History**:
    *   **Function**: Runs conditional diffusion on the SPD-VQVAE latent space, conditioned on medical history embeddings $\hat{\mathbf{y}}$ from the AR model.
    *   **Mechanism**: A U-Net backbone uses Residual MLP Blocks to process latent sequences. Two conditioning paths are used: (i) timestep embeddings $\text{Embed}_\text{diff}(t)$ added to each layer; (ii) Cross-attention treating $\hat{\mathbf{y}} \in \mathbb{R}^{T \times d_\phi}$ as key/value and $\hat z$ as query, outputting $\hat z = \text{Softmax}(QK^\top/\sqrt{C_{hid}})V$. Loss follows standard noise prediction: $\mathcal{L}_{\text{LDM}} = \mathbb{E}\|\epsilon - \epsilon_\theta(z_t, t, \hat{\mathbf{y}})\|^2$. **SPD-VQVAE-Dual**: Empirical evidence showed a single codebook tends to generate "average brains" lacking personalization. Thus, two SPD-VQVAEs model low-pass and high-pass Fourier components separately (threshold 25), allowing the LDM to learn global structure and personalized details simultaneously.
    *   **Design Motivation**: Cross-attention allows generation to attend to any critical ICD record in history (e.g., how a hypertension diagnosis years ago affects today's brain network). The Dual design addresses the mode collapse tendency of generative models on highly symmetric medical data.

3.  **Adaptive ICD Tokenizer + Multi-pathway Mediated Reasoning**:
    *   **Function**: Enables the AR model $\phi$ to handle sparse/irregular ICD timestamps and provide causality-aware conditions for diffusion.
    *   **Mechanism**: A uniform yearly grid $\tau$ is constructed. For each year $\tau_t$, if a subject has ICD $c$, then $s_t = c$; otherwise, the token is `[healthy]`. Input tokens $\mathbf{y}_t = \text{Embed}_\text{ICD}(s_t) + \text{Embed}_\text{age}(\tau_t)$ are fed into a transformer. The next-token objective is $\mathcal{L}_\text{AR} = -\sum_t \log p(s_{t+1}|s_{\leq t}; \phi)$. **During inference**, for each mediation node, the $\phi$ output is fed as $\hat{\mathbf{y}}$ to the LDM to generate multi-organ DTs, which are then processed by fine-tuned FMs to calculate $P(s_{t+1}|\hat{\mathbf{M}}_t)$.
    *   **Design Motivation**: Prior AR medical models (e.g., Delphi) used time-to-event embeddings where intervals could reach decades, causing discontinuous signals for diffusion. The uniform grid with `[healthy]` fillers defines conditions densely over time, stabilizing the cross-attention in the diffusion model.

### Loss & Training
The process involves three stages: (i) AR model $\phi$ pre-training on next-token prediction using 7.28M ICD tokens from 448,651 subjects; (ii) Training SPD-VQVAE and TabDiff on paired imaging + ICD data (Brain: 44,834; Heart: 23,987; Liver: 28,722; Kidney: 32,155 samples) with losses $\mathcal{L}_{\text{VAE}}$ and $\mathcal{L}_{tab} = \lambda_{num}\mathcal{L}_{num} + \lambda_{cat}\mathcal{L}_{cat}$; (iii) Fine-tuning organ FMs on generated DTs for next-ICD classification. Tabular generation uses Classifier-Free Guidance: $\tilde\mu^{num}(\Gamma_t, S, t) = (1+\omega)\mu_\theta^{num}(\Gamma_t, S, t) - \omega\mu_\phi(\Gamma_t, t)$. A subject-level 80:20 split is strictly maintained across all stages.

## Key Experimental Results

### Main Results
UKB dataset, next-occurrence prediction for 1,944 disease categories:

| Method | Backbone | AUC ↑ | F1 ↑ |
|---|---|---|---|
| Delphi | GPT2 | 0.6994 ± 0.091 | 7.09 ± 7.56 |
| Delphi | Qwen3 | 0.8931 ± 0.055 | 18.17 ± 20.67 |
| **DiffDT** | GPT2 | **0.9087 ± 0.050** | **18.60 ± 16.29** |
| **DiffDT** | Qwen3 | **0.9171 ± 0.049** | **20.92 ± 20.40** |

F1 scores grouped by organ mediation (compared against DT-to-event baselines):

| Mediation Organ | Brain F1 | Heart F1 | Liver F1 | Kidney F1 |
|---|---|---|---|---|
| NeuroPath | 56.53 | 48.96 | 54.20 | 54.57 |
| BrainMass | 47.18 | 56.63 | 58.81 | 50.43 |
| **DiffDT-Brain** | **65.14** | 53.74 | 60.44 | 54.97 |
| **DiffDT-Heart** | 58.00 | **58.50** | 56.17 | 58.18 |
| **DiffDT-Liver** | 59.88 | 52.23 | **61.65** | 53.52 |
| **DiffDT-Kidney** | 53.91 | 54.72 | 57.92 | **64.32** |

**Key Observation**: Using the same pre-trained BrainMass as a predictor, DiffDT-Brain (generated DT as input) outperforms BrainMass (real GT as input) by $\approx$18 pp F1, suggesting that multi-pathway mediation captures additional causal signals from the ICD history.

### Ablation Study

| LDM Config | RMSE ↓ | WD ↓ | r ↑ | mAcc ↑ |
|---|---|---|---|---|
| Standard VQVAE | 0.261 | 7.110 | 0.503 | 90.87 |
| SPD-VQVAE | 0.220 | 6.019 | 0.677 | 95.71 |
| **SPD-VQVAE-Dual** | **0.203** | **5.841** | **0.726** | **98.36** |

Tabular DT: Heart RMSE 0.265 / WD 17.27 (112 traits), Liver 0.184 / 2.48, Kidney 0.146 / 0.99. Performance is positively correlated with organ mediation F1 scores. **Counterfactual Evaluation**: Replacing an exposure ICD with `[healthy]` to generate do(healthy) DTs and comparing against real healthy vs. real diseased subjects (FID/WD/r) showed do(healthy) was significantly closer to GT healthy ($p=2.5e\text{-}5$ on FID), proving biological plausibility.

### Key Findings
- **Multi-pathway mediation is effective**: Generated DT as input > Real GT as input (DiffDT-Brain is 18 pp F1 higher than BrainMass), indicating the LDM injects causal signals from history into the generated DT.
- **SPD-VQVAE-Dual is essential**: While the single-branch SPD-VQVAE outperforms standard VQVAE, the dual-branch setup adds another 4–7%, proving that global structure and personalized details need decoupled learning.
- **Backbone upgrades are orthogonal to DiffDT contributions**: When using Qwen3 instead of GPT2, DiffDT still provides a stable +0.12 AUC gain.
- **Operational Efficiency**: Cholesky LDM is significantly faster than SPD-DDPM. End-to-end inference takes 1.2s per subject/mediation, making large-scale cohort reasoning feasible.
- **Off-diagonal pairs provide high gain**: AUC gains are most significant for ICD pairs across different chapters (semantically distant), addressing the primary weakness of EHR-only models.

## Highlights & Insights
- **Cholesky Diffeomorphism as a Core Trick**: Unlike SPD-DDPM which requires eigendecomposition for affine-invariant metrics, Theorem 3.1 transforms SPD diffusion into Euclidean diffusion on lower-triangular factors. This ensures manifold consistency while reducing complexity from $\mathcal{O}(N^3)$ to $\mathcal{O}(N^2)$.
- **SDoH-to-event Paradigm**: Utilizing ICD-coded SDoH proxies allows large-scale EHR datasets without explicit SDoH fields (like UKB) to undergo social determinant analysis.
- **Generated DT > Real GT as Input**: This counter-intuitive result suggests the **generative model encodes historical signals into latent biological states**. The FM receives an "enhanced observation" containing causal temporal structure rather than just a physiological snapshot.
- **Dual SPD-VQVAE Fourier Design**: Decoupling frequency components prevents mode collapse toward the "average healthy brain" in highly symmetric medical data.

## Limitations & Future Work
- Data source is limited to UKB, which is dominated by European white individuals and relatively healthy older adults; generalization to other demographics remains unverified.
- Only four organs are covered (brain/heart/liver/kidney); extending to lungs or endocrine systems requires training new SPD-VQVAEs/TabDiffs.
- Absolute F1 values remain in the 50-65% range; predicting ultra-long-tail rare diseases remains difficult.
- Counterfactual evidence is currently at the group level (FID/WD); individual-level verification with longitudinal intervention data is still needed.
- The Fourier threshold is fixed; learnable frequency splitting or graph spectral decomposition could be explored.

## Related Work & Insights
- **vs Delphi / MOTOR (Pure EHR AR)**: These models lack biological intermediaries; DiffDT improves AUC from 0.70-0.89 to 0.91, particularly for semantically distant disease pairs.
- **vs BrainMass / NeuroPath (Pure DT-to-event)**: DiffDT's use of generated DTs with "history signals" makes it more accurate than using real GT biomarkers by 18 pp F1.
- **vs SPD-DDPM / Riemannian Flow Matching**: DiffDT achieves manifold alignment efficiently via Cholesky decomposition instead of $\mathcal{O}(N^3)$ metric-based diffusion.
- **Insight**: "Performing diffusion on a manifold-constrained latent space, where constraints are naturally satisfied by the decoder geometry" is a universal recipe applicable to molecular conformations and point cloud generation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First rigorously manifold-preserving and scalable SDoH-to-event framework.
- Experimental Thoroughness: ⭐⭐⭐⭐ 33K-44K real UKB subjects, 1,944 diseases, 4 organs, and counterfactual validation.
- Writing Quality: ⭐⭐⭐⭐ Logical flow from geometric motivation to counterfactual loops is tight.
- Value: ⭐⭐⭐⭐⭐ High potential for clinical decision support via long-term multi-pathway reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: Beyond Sensitive Attributes, ML Fairness Should Quantify Structural Injustice via Social Determinants](position_beyond_sensitive_attributes_ml_fairness_should_quantify_structural_inju.md)
- [\[ICML 2026\] Evidential Reasoning Advances Interpretable Real-World Disease Screening](evidential_reasoning_advances_interpretable_real-world_disease_screening.md)
- [\[ICML 2026\] Controllable Generative Sandbox for Causal Inference](controllable_generative_sandbox_for_causal_inference.md)
- [\[ICML 2026\] Auditing Sybil: Explaining Deep Lung Cancer Risk Prediction Through Generative Interventional Attributions](auditing_sybil_explaining_deep_lung_cancer_risk_prediction_through_generative_in.md)
- [\[ICLR 2026\] Can SAEs Reveal and Mitigate Racial Biases of LLMs in Healthcare?](../../ICLR2026/medical_imaging/can_saes_reveal_and_mitigate_racial_biases_of_llms_in_healthcare.md)

</div>

<!-- RELATED:END -->
