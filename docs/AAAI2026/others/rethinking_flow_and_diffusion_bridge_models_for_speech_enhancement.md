---
title: >-
  [Paper Note] Rethinking Flow and Diffusion Bridge Models for Speech Enhancement
description: >-
  [AAAI 2026][Speech Enhancement] This paper proposes a unified theoretical framework that subsumes flow matching, score-based diffusion, and Schrödinger bridge models for speech enhancement as processes that construct different Gaussian probability paths between paired data. It further reveals that each sampling step in such generative models is intrinsically equivalent to predictive speech enhancement, and leverages this insight to improve bridge model performance by adopting high-performance backbone networks, refined loss functions, and fine-tuning strategies from the predictive paradigm.
tags:
  - AAAI 2026
  - Speech Enhancement
  - Diffusion Bridge Models
  - Flow Matching
  - Schrödinger Bridge
  - Unified Framework
date: 2026-05-08
content_hash: c11e9651ca5ab23f
---

# Rethinking Flow and Diffusion Bridge Models for Speech Enhancement

**Conference**: AAAI 2026
**arXiv**: [2602.18355](https://arxiv.org/abs/2602.18355)
**Code**: [GitHub](https://github.com/Dahan-Wang/Rethinking-Flow-and-Diffusion-Bridge-Models-for-Speech-Enhancement)
**Area**: Other
**Keywords**: Speech Enhancement, Diffusion Bridge Models, Flow Matching, Schrödinger Bridge, Unified Framework

## TL;DR

This paper proposes a unified theoretical framework that subsumes flow matching, score-based diffusion, and Schrödinger bridge models for speech enhancement as processes that construct different Gaussian probability paths between paired data. It further reveals that each sampling step in such generative models is intrinsically equivalent to predictive speech enhancement, and leverages this insight to improve bridge model performance by adopting high-performance backbone networks, refined loss functions, and fine-tuning strategies from the predictive paradigm.

## Background & Motivation

Speech Enhancement (SE) aims to recover clean speech from noisy observations. Deep learning approaches can be broadly categorized into **predictive** methods (directly learning a mapping from noisy to clean signals) and **generative** methods (modeling the conditional distribution of clean speech). In recent years, flow- and diffusion-based generative methods have proliferated in SE, primarily comprising:

**Score-based diffusion models**: Design the drift term of an SDE via the OU process or Brownian bridge to establish a diffusion process between clean and noisy signals (e.g., SGMSE+, BBED).

**Schrödinger bridge (SB)**: Optimize the path measure under Dirac endpoint constraints, achieving state-of-the-art performance with data prediction training strategies (e.g., SBVE).

**Flow matching (FM)**: Construct probability paths conditioned on noisy speech and enable efficient sampling via conditional vector fields (e.g., FlowSE).

However, these approaches rest on distinct theoretical foundations (score matching, SB optimization, flow matching) and **have not been unified under a common framework in the SE domain**. Furthermore, the data prediction objective used in SB models implies an intrinsic connection to predictive methods, yet this connection **has not been thoroughly explored** in prior work.

The paper is motivated by two core questions:
- Can these seemingly disparate generative SE methods be unified into a single framework?
- What is the fundamental relationship between generative bridge models and predictive SE models, and can this relationship be exploited to improve performance?

## Method

### Overall Architecture

The central idea of the proposed unified framework is that **all flow and diffusion bridge models can be interpreted as constructing different Gaussian probability paths between paired data** (noisy $\mathbf{y}$ and clean $\mathbf{s}$). The probability path is defined as:

$$p_t(\mathbf{x}_t|\mathbf{s},\mathbf{y}) = \mathcal{N}(\mathbf{x}_t; \boldsymbol{\mu}_t(\mathbf{s},\mathbf{y}), \sigma_t^2 \mathbf{I})$$

where the mean is an interpolation between the clean and noisy signals: $\boldsymbol{\mu}_t = a_t \mathbf{s} + b_t \mathbf{y}$. Different methods differ only in the choice of $a_t$, $b_t$, and $\sigma_t$. Once the probability path is specified, the corresponding sampling ODE and forward/backward SDEs follow directly.

### Key Designs

1. **Unified Probability Path Framework**: The sampling ODE is derived via conditional flow matching:

$$\frac{\mathrm{d}\mathbf{x}_t}{\mathrm{d}t} = \frac{\sigma_t'}{\sigma_t}\mathbf{x}_t + \left(a_t' - a_t\frac{\sigma_t'}{\sigma_t}\right)\mathbf{s} + \left(b_t' - b_t\frac{\sigma_t'}{\sigma_t}\right)\mathbf{y}$$

and extended to forward/backward SDEs via the Fokker-Planck equation. The probability path parameters for different models are summarized below:

| Method | $a_t$ | $b_t$ | $\sigma_t$ |
|--------|--------|--------|-------------|
| OUVE | $e^{-\gamma t}$ | $1-e^{-\gamma t}$ | complex expression |
| BBED | $1-t$ | $t$ | $c(1-t)E_t$ |
| SB | $\alpha_t\bar{\rho}_t^2/\rho_1^2$ | $\bar{\alpha}_t\rho_t^2/\rho_1^2$ | $\alpha_t^2\bar{\rho}_t^2\rho_t^2/\rho_1^2$ |
| OT-CFM | $t$ | $1-t$ | $(1-t)\sigma_{\max}+t\sigma_{\min}$ |
| SB-CFM | $1-t$ | $t$ | $\sigma^2 t(1-t)$ |

This unification eliminates the need to derive each model separately from SDE design or KL divergence minimization, substantially simplifying theoretical analysis.

2. **Predictive Equivalence Insight**: This is the paper's most central finding. The authors prove that under the data prediction training strategy, the network output at each sampling step is essentially performing predictive speech enhancement. The final sampling result can be expressed as a weighted sum of the network outputs across all steps:

$$\mathbf{x}_{t_0} = \sum_{n=1}^{N} w_n \mathbf{s}_{t_n} + w_y \mathbf{y}$$

Numerical simulations under SB-CFM parameterization reveal that the weight of the final step $w_N$ overwhelmingly dominates (approaching 1), while contributions from intermediate steps and the noisy input $\mathbf{y}$ are negligible. This implies that:

- **Single-step sampling is nearly equivalent to a predictive model**, relying entirely on data prediction without exploiting intermediate state information.
- The performance ceiling of multi-step sampling is bounded by the capacity of the predictive model.
- Training at time steps other than $t=1$ (i.e., pure noisy input) becomes redundant, as only the single-step case at $t=1$ is meaningful.

3. **Improved Bridge Model**: Building on the above insights, the paper replaces the commonly used NCSN++ U-Net architecture with the state-of-the-art predictive SE model TF-GridNet. To enable TF-GridNet to accept the diffusion time $t$ as input, a **time embedding mechanism** is designed: a time embedding vector is first obtained via Fourier embedding, followed by a fully connected layer and SiLU activation, and then injected into the input features at the beginning of each TF-GridNet block via a dedicated linear layer. This reduces the parameter count from 65.6M to 2.2M and MACs from 66G to 38G.

### Loss & Training

The data prediction loss is improved by incorporating a combination of loss functions commonly used in predictive SE:

- **Negative SI-SNR loss**: $\mathcal{L}_{\text{SI-SNR}}(\hat{x}, x) = -\log_{10}\frac{\|x_t\|^2}{\|\hat{x}-x_t\|^2}$, emphasizing signal-level SNR.
- **Power-compressed spectral magnitude loss**: $\mathcal{L}_{\text{mag}} = \text{MSE}(|\hat{X}|^{0.3}, |X|^{0.3})$, providing better focus on spectral magnitude.
- **Power-compressed real/imaginary loss**: $\mathcal{L}_{\text{real/imag}} = \text{MSE}(\hat{X}_{r/i}/|\hat{X}|^{0.7}, X_{r/i}/|X|^{0.7})$.

The total loss is: $\mathcal{L} = \lambda_1 \mathcal{L}_{\text{SI-SNR}} + \lambda_2 \mathcal{L}_{\text{mag}} + \lambda_3 (\mathcal{L}_{\text{real}} + \mathcal{L}_{\text{imag}})$

In addition, a **CRP fine-tuning strategy** (Correcting the Reverse Process) is employed: only the model weights at the final step are updated during sampling, compensating for potential under-optimization at the last step. This is consistent with the weight analysis finding that the final step has the greatest influence on the output.

## Key Experimental Results

### Main Results

**DNS3 Test Set**:

| Model | Params (M) | MACs (G) | SI-SNR | ESTOI | PESQ | DNSMOS |
|-------|-----------|----------|--------|-------|------|--------|
| Noisy | - | - | 5.613 | 0.669 | 1.406 | 2.147 |
| SGMSE+ (OUVE) | 65.6 | 66×60 | 11.873 | 0.796 | 2.336 | 3.647 |
| SBVE | 65.6 | 66×60 | 14.959 | 0.844 | 2.592 | 3.729 |
| TF-GridNet (predictive) | 2.1 | 38 | 16.448 | 0.872 | 3.187 | 3.743 |
| **Ours (NFEs=1)** | **2.2** | **38×1** | **16.245** | **0.870** | **3.185** | **3.740** |
| **Ours (NFEs=5)** | **2.2** | **38×5** | **16.424** | **0.874** | **3.213** | **3.752** |

**VoiceBank+DEMAND Test Set**:

| Model | SI-SNR | ESTOI | PESQ | DNSMOS |
|-------|--------|-------|------|--------|
| SGMSE+ | 17.3 | 0.87 | 2.93 | 3.56 |
| SBVE | 19.4 | 0.88 | 2.91 | 3.59 |
| FlowSE | 19.0 | 0.88 | 3.12 | 3.58 |
| **Ours** | **19.6** | **0.89** | **3.30** | **3.57** |

### Ablation Study

| Configuration | SI-SNR | PESQ | Notes |
|---------------|--------|------|-------|
| NCSN++ + original loss + SBVE | 14.158 | 2.706 | Original baseline |
| TF-GridNet + improved loss + SBVE | 16.646 | 3.068 | Large gain from backbone + loss |
| TF-GridNet + improved loss + SBVE + CRP | 16.424 | 3.213 | CRP fine-tuning improves PESQ |
| TF-GridNet + improved loss + OUVE | 11.302 | 2.129 | OUVE path performs poorly |
| TF-GridNet + improved loss + OT-CFM | 14.866 | 2.834 | FM path inferior to SB |
| TF-GridNet + improved loss + SB-CFM | 16.177 | 3.102 | SB-CFM approaches SBVE |

### Key Findings

1. **Single-step sampling ≈ predictive model**: The proposed model at NFEs=1 (PESQ 3.185) nearly matches the state-of-the-art predictive TF-GridNet (3.187), directly validating the theoretical analysis.
2. **Dirac endpoint + exponential integrator sampler = optimal configuration**: The probability paths of SBVE and SB-CFM have zero variance at the sampling starting point (Dirac distribution), which works best in conjunction with exponential integrator samplers.
3. **Predictive nature limits the performance ceiling**: Five-step sampling yields only marginal improvement over single-step, remaining on par with the predictive model, confirming that the predictive essence of the generative framework constrains the potential to surpass predictive models.

## Highlights & Insights

- **Elegance of theoretical unification**: Reducing four methodologically distinct approaches to a triplet parameterization $(a_t, b_t, \sigma_t)$ of probability paths is remarkably concise.
- **Deep insight of generation-as-prediction**: The paper reveals that diffusion bridge models under data prediction loss are "generative in form but predictive in essence." This not only explains why single-step sampling works, but also identifies the root cause of limited improvement from multi-step sampling.
- **Dramatic efficiency gains**: Parameter count is reduced from 65.6M to 2.2M (~30×), and computation from 66G×60 steps to 38G×5 steps (~20×), while performance improves substantially.
- **Predictive paradigm enriching generative models**: Loss functions, backbone networks, and fine-tuning strategies are all drawn from predictive methods, forming a productive synthesis.

## Limitations & Future Work

1. **Bounded performance ceiling**: The authors explicitly acknowledge that due to the predictive nature of the generative framework, performance is unlikely to significantly exceed that of the corresponding predictive model.
2. **Generalizability to be verified**: Experiments are conducted only on denoising and dereverberation tasks; applicability to other speech tasks (e.g., speech separation, bandwidth extension) remains unclear.
3. **Limitations of CRP fine-tuning**: CRP only fine-tunes the final step, and error accumulation from earlier steps may not be fully resolved.
4. **Automated probability path design**: Path parameters are currently selected manually; automatically discovering optimal paths through learning is a direction worth exploring.

## Related Work & Insights

- The proposed unified framework can be extended to other paired-data bridge model tasks (e.g., image translation, text-to-speech synthesis).
- The "generation is essentially prediction" insight has broad implications for the application of diffusion models to other signal processing tasks such as image restoration.
- The success of TF-GridNet as a backbone suggests that selecting a high-performance, task-specific backbone for diffusion models is more effective than blindly scaling up U-Net architectures.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | ⭐⭐⭐⭐ | The unified framework and predictive equivalence analysis represent strong theoretical contributions |
| Practicality | ⭐⭐⭐⭐⭐ | Significantly reduces parameters and computation while outperforming all baselines |
| Theoretical Depth | ⭐⭐⭐⭐⭐ | Sampling output decomposition and weight analysis are highly rigorous |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Two datasets with detailed ablations, but validation across more tasks is lacking |
| Writing Quality | ⭐⭐⭐⭐ | Logically clear with complete theoretical derivations |
| Overall | ⭐⭐⭐⭐½ | Theoretically insightful and practically instructive; an important contribution to understanding diffusion models |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Beyond World Models: Rethinking Understanding in AI Models](beyond_world_models_rethinking_understanding_in_ai_models.md)
- [\[AAAI 2026\] MF-Speech: Achieving Fine-Grained and Compositional Control in Speech Generation via Factor Disentanglement](mf-speech_achieving_fine-grained_and_compositional_control_in_speech_generation_.md)
- [\[AAAI 2026\] Cash Flow Underwriting with Bank Transaction Data: Advancing MSME Financial Inclusion in Malaysia](cash_flow_underwriting_with_bank_transaction_data_advancing_msme_financial_inclu.md)
- [\[AAAI 2026\] Controllable Financial Market Generation with Diffusion Guided Meta Agent](controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)
- [\[AAAI 2026\] ASAG: Toward the Frontiers of Reliable Diffusion Sampling via Adversarial Sinkhorn Attention Guidance](toward_the_frontiers_of_reliable_diffusion_sampling_via_adversarial_sinkhorn_att.md)

</div>

<!-- RELATED:END -->
