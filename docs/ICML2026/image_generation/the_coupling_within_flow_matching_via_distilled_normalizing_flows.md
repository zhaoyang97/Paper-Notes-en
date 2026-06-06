---
title: >-
  [Paper Note] The Coupling Within: Flow Matching via Distilled Normalizing Flows
description: >-
  [ICML 2026][Image Generation][Flow Matching] This paper proposes NFM (Normalized Flow Matching), which uses the "accurate data→noise bijection" produced by a pretrained autoregressive normalizing flow (NF) such as TarFlo…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Flow Matching"
  - "Normalizing Flow"
  - "coupling"
  - "distillation"
  - "TarFlow"
date: 2026-05-08
content_hash: 7ff578e56655eb83
---

# The Coupling Within: Flow Matching via Distilled Normalizing Flows

**Conference**: ICML 2026  
**arXiv**: [2603.09014](https://arxiv.org/abs/2603.09014)  
**Code**: https://github.com/apple/ml-nfm  
**Area**: Diffusion Models / Generative Models / Flow Matching  
**Keywords**: Flow Matching, Normalizing Flow, coupling, distillation, TarFlow

## TL;DR
This paper proposes NFM (Normalized Flow Matching), which uses the "accurate data→noise bijection" produced by a pretrained autoregressive normalizing flow (NF) such as TarFlow as the noise-data pairing for Flow Matching. This approach simultaneously advances FM's convergence speed and low-step FID, and, in turn, achieves inference speeds several orders of magnitude faster than the NF teacher.

## Background & Motivation

**Background**: Flow Matching (FM) has become a mainstream training paradigm for large-scale generative models—using $x_t=(1-t)x+t\epsilon$ to linearly interpolate between data and noise, then regressing the velocity $v_t=\epsilon-x$, and during inference, using an ODE solver to reverse from $x_1\sim\mathcal{N}(0,I)$ to $x_0$. A key design that determines performance is the "noise-data pairing (coupling)". Independent coupling is simplest but leads to slow training and high inference curvature, so OT-based coupling (e.g., SD-FM) has become the mainstream improvement direction.

**Limitations of Prior Work**: OT-based methods are essentially geometry-based, model-agnostic preprocessing; they do not truly leverage "model-learnable data representations". Meanwhile, another line—Normalizing Flow (NF), especially the recent TarFlow—can directly learn a data ↔ Gaussian bijection. In theory, once the pairing is determined, velocity regression is error-free and sampling can be done in one step, but NF's autoregressive sampling is prohibitively slow.

**Key Challenge**: FM is fast at inference but has coarse pairing; NF has perfect pairing but slow sampling. Both have structural shortcomings, and there has been little work combining the two.

**Goal**: (i) Replace FM's random/OT pairing with the (near-bijective) pairing learned by NF, providing FM students with more "aligned" training pairs; (ii) Verify that such "distilling NF into FM" can both accelerate FM convergence and, in turn, surpass the NF teacher in FID.

**Key Insight**: The authors view NF as a "learned, model-dependent optimal transport approximation". Even if NF's mapping is not OT-optimal in the likelihood sense (being limited by network capacity), as long as it can stably map each data point to a specific Gaussian representation, it suffices as a high-quality coupling.

**Core Idea**: During FM student training, replace the noise endpoint $\epsilon$ with the encoding $z_{\epsilon'}=f_{\text{NF}}(x+\eta\epsilon',c)/\sigma_f$ from the pretrained NF teacher, and set the velocity target as $v_t=z_{\epsilon'}-x$, with other FM procedures unchanged.

## Method

### Overall Architecture
NFM is a two-stage distillation: In the first stage, a TarFlow teacher $f_{\text{NF}}$ is trained to learn a reversible mapping $x\mapsto z$ via maximum likelihood. In the second stage, the teacher is frozen, and an FM student $g$ (of any architecture, not necessarily invertible) is trained. The student sees training pairs $(x, z_{\epsilon'})$ instead of $(x, \epsilon)$, where $z_{\epsilon'}=f_{\text{NF}}(x+\eta\epsilon',c)/\sigma_f$, $\eta$ is the input perturbation magnitude used during teacher training, and $\sigma_f$ is a scalar to normalize the teacher's output to unit variance. The student is trained with the standard FM loss $\mathcal{L}_{\text{FM}}=\|g((1-t)x+tz_{\epsilon'},c,t)-(z_{\epsilon'}-x)\|_2^2$, and inference is identical to standard FM.

### Key Designs

1. **Replacing Random Noise with NF Teacher's $z$**:

    - **Function**: Each data $x$ is paired with a $z_{\epsilon'}$ determined by the NF teacher and close to Gaussian, so the FM student learns a near-deterministic mapping from a specific $z$ to a specific $x$, rather than a many-to-many mapping from arbitrary noise to arbitrary data.
    - **Mechanism**: During training, sample $z_{\epsilon'}=f_{\text{NF}}(x+\eta\epsilon',c)/\sigma_f$, where $\sigma_f^2=\mathbb{E}[f_{\text{NF}}(x+\eta\epsilon',c)^2]$ ensures $z$ has variance approximately 1, so the overall distribution remains close to $\mathcal{N}(0,I)$, and the FM student can start from standard Gaussian during sampling. Notably, translating the variational explosion notation to variance-preserving notation, TarFlow's perturbation $\eta=0.05$ corresponds to FM's maximum noise level $t=\eta/(1+\eta)\approx0.0476$, much smaller than standard FM's $t=1$.
    - **Design Motivation**: The core challenge of FM is that $x_t$ has high conditional variance at large $t$, and the velocity target $v_t=\epsilon-x$ contains only "endpoint difference" variance information. Replacing $\epsilon$ with $z_{\epsilon'}$ from NF significantly reduces $\text{Var}(v_t|x_t,t)$, stabilizes gradients, straightens trajectories, and directly translates to FID improvements at lower NFE.

2. **TarFlow Teacher + Input Perturbation $\eta$**:

    - **Function**: TarFlow is chosen as the teacher because it matches diffusion models in image generation; adding a small $\eta\epsilon'$ perturbation to $x$ during teacher training ensures the mapping is smooth in the data neighborhood.
    - **Mechanism**: TarFlow is an auto-regressive flow implemented with a Transformer, generating patches autoregressively within each meta-block, thus slow to sample but highly invertible. During training, $x'=x+\eta\epsilon'$ is input to the network, and NLL is minimized. NFM retains this $\eta$ to ensure $z$ is smooth in a small neighborhood and naturally reduces FM's effective noise level to $\sim\eta/(1+\eta)$.
    - **Design Motivation**: The teacher's perturbation $\eta$ serves two purposes in NFM: it prevents $z$ from degenerating into a purely deterministic mapping (retaining some stochasticity), and implicitly controls the maximum noise level seen by the FM student. In experiments, larger $\eta$ yields optimal FID at higher NFE, while smaller $\eta$ favors performance at low-step sampling.

3. **Flexible Student Architecture + FM-Equivalent Training Objective**:

    - **Function**: The student $g$ can be a standard ViT/CNN, not necessarily invertible, and thus can be smaller than TarFlow and have adjustable inference steps.
    - **Mechanism**: After replacing $\epsilon$ with $z_{\epsilon'}$, the FM form of "regressing velocity along linear interpolation" remains unchanged, as do time weights, so it can be seamlessly integrated into any FM training pipeline (SiT-XL is used in experiments). Inference uses Euler (NFE ≤ 5) or Heun (NFE ≥ 5), with step sizes scheduled as $t^2=\{1, (1-\delta t)^2,\ldots\}$.
    - **Design Motivation**: Retaining the FM training form means NFM introduces no new hyperparameters and does not disrupt existing codebases; the student need not be invertible, unlocking arbitrary architecture choices and enabling inference speeds several orders of magnitude faster than the invertible TarFlow teacher.

### Loss & Training
The student is trained with $\mathcal{L}_{\text{FM}}=\|g((1-t)x+tz_{\epsilon'},c,t)-(z_{\epsilon'}-x)\|_2^2$, with class labels randomly dropped with probability $p=0.1$ to support classifier-free guidance; time $t$ follows $\text{lognorm}(-0.2,1)$. The teacher is trained on 512 MiB samples (about 420 epochs), and the student on 256 MiB (about 210 epochs).

## Key Experimental Results

### Main Results

| Dataset / Teacher / NFE | FM | SD-FM | NFM (256 MiB) | NFM vs FM |
|------------------------|----|-------|---------------|-----------|
| ImageNet64, SiT-XL/4, 31 | 2.57 | 2.68 | 1.78 | -0.79 |
| ImageNet64, 15 | 4.80 | 3.15 | 2.15 | -2.65 |
| ImageNet64, 7 | 13.01 | 6.41 | 3.23 | -9.78 |
| ImageNet64, Euler-5 | 21.05 | 12.18 | 3.92 | -17.13 |
| ImageNet256, SiT-XL/2, 31 | 2.30 | – | 2.29 | -0.01 |
| ImageNet256, 7 | 12.41 | – | 3.43 | -8.98 |

The student achieves FID 1.78 on ImageNet64, outperforming the TarFlow teacher with equivalent parameters (FID=1.98); on ImageNet256, NFM (FID 2.29) significantly outperforms the teacher (FID 3.96).

### Ablation Study

| Configuration / Phenomenon | Result | Notes |
|---------------------------|--------|-------|
| Heun(t²), 31 NFE | FM 2.57 → NFM 1.78 | NFM's advantage is smaller but stable at high NFE |
| Euler(t), 128 NFE curvature $\kappa$ | FM 0.0386 / SD-FM 0.0289 / NFM 0.0181 | NFM's path is significantly straighter, enabling fewer NFE |
| Heun, 5 NFE | 17.56 → 9.29 → 4.01 | NFM's relative gain is greatest at very low-step sampling |
| Large $\eta$ vs small $\eta$ (z-space structure) | Larger $\eta$ brings different images' $z$ closer | Provides FM with weaker "quasi-determinism", harming low-step FID |

### Key Findings
- Student FID surpasses the teacher: Attributed to the combination of "almost deterministic pairing + flexible student architecture + EMA"—the teacher is constrained by invertibility and limited capacity, while the student is not and can fit better.
- Unexpected $z$-space structure: For the same image, $z$ projections under different $\eta\epsilon'$ are not nearest neighbors; instead, different images under the same noise are closer. This suggests NF deeply entangles "image identity" and "noise vector" in $z$-space, but NFM still works, indicating the FM student learns endpoint correspondence rather than neighborhood structure.
- Benefit distribution: In the typical deployment range of NFE = 7, NFM reduces FID to 1/4 of FM and still halves SD-FM's FID. This is precisely the region where SD-FM's OT pairing cannot reach.

## Highlights & Insights
- Treating NF as "model-learned OT approximation" is a highly explanatory perspective: OT's core value is "pairing", not "geometric optimality". NFM demonstrates that as long as pairing is sufficiently deterministic, OT geometric optimality is unnecessary.
- The "FM distilled from NF" approach can be interpreted as both distillation (NF→FM) and hybridization (NF+FM), and the student samples much faster than the teacher—a rare case where the student is both faster and better than the teacher.
- The method does not modify FM's training formula, sampler, guidance, or time schedule, so engineering integration cost is extremely low—any existing FM code can simply replace $\epsilon$ with $z_{\epsilon'}$.

## Limitations & Future Work
- Still requires training a high-quality NF teacher, which is itself expensive; if the teacher's FID is poor, the distilled student will also suffer. The paper does not specify how poor the teacher can be before distillation fails.
- Experiments only cover ImageNet64/256 + class-conditional; text-to-image, video, and high-resolution (1024+) scenarios are untested, and whether TarFlow can be stably trained in these settings remains an open question.
- The counterintuitive $z$-space structure (same-image $z$ not being nearest neighbors) is acknowledged as not fully understood and may pose long-term risks to distillation stability.
- NFM is not in conflict with OT-based pairing methods like SD-FM; the paper suggests future combination, but no actual combined experiments are provided.

## Related Work & Insights
- **vs SD-FM / OT-CFM**: Both modify coupling, but OT-based methods approximate discrete optimal transport, while this work replaces it with a deterministic map learned by NF, yielding significantly better results on ImageNet256, especially at low NFE.
- **vs DiffFlow / DiNof**: These works jointly train NF and diffusion; NFM adopts a "teacher frozen, student distilled" approach, which is simpler and more reproducible in practice.
- **vs Consistency Models / Rectified Flow**: CM and RF aim to shorten ODE paths to reduce NFE, while NFM "improves the endpoints", fundamentally reducing path curvature (experimentally, $\kappa$ is halved).
- **vs General Distillation Methods** (e.g., Progressive Distillation): Traditional distillation has the student mimic the teacher's sampling trajectory; NFM has the student learn FM on a new target distribution, so the student need not align per NFE and generalizes better.

## Rating
- Novelty: ⭐⭐⭐⭐ Using NF as the source of FM coupling is a fresh and natural perspective, and the cross-family distillation is cleanly implemented.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive experiments across NFE, solvers, different $\eta$, and curvature analysis, with comparisons covering FM, SD-FM, and TarFlow itself.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, with honest reporting of counterintuitive $z$-space structure analysis.
- Value: ⭐⭐⭐⭐ Provides a low-cost FID improvement for existing FM models, especially effective in low-step sampling (practical deployment range).

## Related Papers

- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/image_generation/amortized_sampling_with_transferable_normalizing_flows.md)
- [\[AAAI 2026\] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment](../../AAAI2026/image_generation/flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)
- [\[ICML 2026\] Cascaded Flow Matching for Heterogeneous Tabular Data with Mixed-Type Features](cascaded_flow_matching_for_heterogeneous_tabular_data_with_mixed-type_features.md)
- [\[ICML 2025\] Normalizing Flows are Capable Generative Models](../../ICML2025/image_generation/normalizing_flows_are_capable_generative_models.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)

</div>

<!-- RELATED:END -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Amortized Sampling with Transferable Normalizing Flows](../../NeurIPS2025/image_generation/amortized_sampling_with_transferable_normalizing_flows.md)
- [\[AAAI 2026\] Flowing Backwards: Improving Normalizing Flows via Reverse Representation Alignment](../../AAAI2026/image_generation/flowing_backwards_improving_normalizing_flows_via_reverse_representation_alignme.md)
- [\[ICML 2026\] Exploring and Exploiting Stability in Latent Flow Matching](exploring_and_exploiting_stability_in_latent_flow_matching.md)
- [\[ICML 2026\] Adversarial Flow Models](adversarial_flow_models.md)
- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](../../ICLR2026/image_generation/multi-agent_coordination_via_flow_matching.md)

</div>

<!-- RELATED:END -->
