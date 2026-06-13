---
title: >-
  [Paper Note] E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models
description: >-
  [ICML 2026][Image Generation][GRPO] Addressing the issues of intra-group variance collapse and signal disappearance in group-based RL (e.g., GRPO/DiffusionNFT) for flow model alignment…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "GRPO"
  - "Flow Matching"
  - "Text Embedding Perturbation"
  - "Reward Hacking"
  - "Intra-group Variance"
date: 2026-05-08
content_hash: f7381252c9f0c9a3
---

# E²PO: Embedding-perturbed Exploration Preference Optimization for Flow Models

**Conference**: ICML 2026  
**arXiv**: [2605.15803](https://arxiv.org/abs/2605.15803)  
**Code**: To be confirmed  
**Area**: Image Generation / Flow Matching / RL Alignment  
**Keywords**: GRPO, Flow Matching, Text Embedding Perturbation, Reward Hacking, Intra-group Variance

## TL;DR
Addressing the issues of intra-group variance collapse and signal disappearance in group-based RL (e.g., GRPO/DiffusionNFT) for flow model alignment, E²PO injects a set of learnable structured perturbations into the text embedding space to maintain discriminative variance. Combined with noise-aware scheduling and reference-anchored batch strategies, it improves GenEval on SD3.5-M from 0.917 to 0.932 while significantly enhancing diversity.

## Background & Motivation

**Background**: Adapting the RLHF-style GRPO paradigm to flow/diffusion models—by rewriting deterministic ODEs as SDEs to explore high-reward trajectories during sampling—is the current mainstream route for aligning text-to-image models with human preferences. DiffusionNFT further pulls this process back to the forward process, using contrastive objectives to directly optimize the velocity field.

**Limitations of Prior Work**: Such methods rely heavily on "intra-group relative advantage" signals, where rewards are normalized within a group before backpropagating gradients. However, during training, the policy naturally converges toward high-reward modes, causing intra-group samples to become increasingly similar. Consequently, the intra-group variance $\sigma_R$ rapidly decays toward zero. The paper monitors GenEval/PickScore and finds that the baseline std drops below the threshold early in training.

**Key Challenge**: There is an inherent conflict between the signal source (intra-group variance) and the optimization goal (making intra-group samples converge to high-score modes). As optimization succeeds, variance decreases, rendering gradients meaningless. Common countermeasures involve either expanding the latent group size $G$ or using different initial noises; the former (expanding $G$ from 4 to 48) fails to stop the decay and scales compute linearly, while the latter easily triggers reward hacking (learning high-score but visually incoherent artifacts).

**Goal**: To construct an exploration mechanism that does not rely on latent noise expansion and can continuously maintain "semantically meaningful" intra-group differences, avoiding signal disappearance while pushing the policy away from narrow local optima.

**Key Insight**: The authors observe that the text embedding space is a low-dimensional, structured semantic manifold. Fine-tuning along this manifold is more efficient at inducing meaningful semantic changes than brute-force sampling in high-dimensional noise. Prior work in prompt engineering and textual inversion has proven that small shifts in embedding space significantly alter generation trajectories.

**Core Idea**: Switch the source of "intra-group diversity" from noise priors to text embedding perturbations. For each prompt, a set of structured perturbations $\bm{\delta}_k$ is learned and injected into content tokens. This forces the policy to see a set of semantically distinct but intent-consistent conditions at each step, thereby pinning the discriminative variance at a non-zero level.

## Method

### Overall Architecture
Given a prompt $p$, it is first tokenized into embeddings $\mathbf{E} \in \mathbb{R}^{S\times d}$. An index set $\mathcal{I}$ is defined to cover only "substantive content tokens" (excluding [SOS]/[EOS]/padding, with effective length $L$). The training of E²PO consists of two phases: Phase I freezes the flow model and learns a set of perturbations $\Delta=\{\bm{\delta}_k\}_{k=1}^{K}$ independently. Phase II freezes these perturbations and feeds them along with the original prompt into the RL alignment loop (based on the DiffusionNFT contrastive objective). During sampling, a batch consists of one original condition $\mathbf{C}^{\text{orig}}$ and $K-1$ perturbed conditions $\mathbf{C}_k(t)$, switched according to a noise-aware temporal schedule. Policy updates only compute gradients on $\mathbf{C}^{\text{orig}}$, explicitly decoupling "what is used for exploration" from "what is used for updates."

### Key Designs

1.  **Embedding-Perturbed Mechanism**:
    *   **Function**: Learns $K$ semantically non-overlapping yet intent-consistent perturbation vectors for each prompt to serve as a stable source of intra-group variance.
    *   **Mechanism**: Perturbations are parameterized as $\bm{\delta}_k \in \mathbb{R}^{L\times d}$, acting only on content token positions $t=\mathcal{I}_j$ to obtain $\tilde{\mathbf{E}}_{k,t} = \mathbf{E}_t + \bm{\delta}_{k,j}$. These are passed through a frozen text encoder $f_\phi$ to get global embeddings $e_k$. Optimization uses a hybrid objective $\mathcal{L}_{\text{emb}} = \lambda_{div}\mathcal{L}_{\text{div}} + \mathcal{L}_{\text{anc}}$, where $\mathcal{L}_{\text{div}}$ minimizes the average cosine similarity between $K$ variants to push them apart, and $\mathcal{L}_{\text{anc}} = \sum_k |\cos(e_k, e_{\text{anc}}) - \mu|_\epsilon$ uses an $\epsilon$-insensitive loss to anchor each variant's similarity to the original prompt around $\mu$.
    *   **Design Motivation**: Relying solely on diversity would diverge into semantically irrelevant regions, triggering reward hacking; relying solely on anchoring would collapse to a single point, losing variance. Perturbing only content tokens constrains exploration to "different directions within the semantic neighborhood," maintaining signals without deviating from intent.

2.  **Noise-Aware Sampling Schedule**:
    *   **Function**: Determines when perturbed conditions act during the sampling trajectory to prevent semantic drift from destroying high-frequency details in later stages.
    *   **Mechanism**: The real-time condition for the $k$-th variant is defined as a time-dependent interpolation $\mathbf{C}_k(t) = \gamma(\sigma_t)\mathbf{C}^{\text{opt}}_k + (1-\gamma(\sigma_t))\mathbf{C}^{\text{orig}}$, where $\gamma(\sigma_t) = \text{clip}((\sigma_t - (1-\rho))/\rho, 0, 1)$ decreases monotonically with the normalized noise level $\sigma_t$ (decaying from 1 to 0). The hyperparameter $\rho \in (0,1]$ controls the length of the high-noise interval covered by perturbations.
    *   **Design Motivation**: Ablations show that static use of $\mathbf{C}^{\text{opt}}_k$ leads to artifacts. Diffusion models have varying sensitivity at different timesteps—coarse structures are fixed early, while details are added later—so condition strength should decay over time.

3.  **Reference-Anchored Batching Strategy**:
    *   **Function**: Explicitly retains an unperturbed "anchor" in each training batch and decouples exploration from update conditions, preventing the policy from being biased by the perturbations themselves.
    *   **Mechanism**: The set $\mathcal{C}_{\text{batch}} = \{\mathbf{C}^{\text{orig}}\} \cup \{\mathbf{C}_k(t)\}_{k=1}^{K-1}$ is used to "sample generative trajectories." However, when substituting into the DiffusionNFT contrastive loss, all gradients are forced to be calculated relative to $\mathbf{C}^{\text{orig}}$. Thus, reward evaluation covers a semantic space expanded by perturbations, while the distribution actually being pushed is that of the original prompt.
    *   **Design Motivation**: If updating on the same conditions used for sampling, the policy learns to "perform well under perturbed prompts" rather than "real prompts." The anchor + decoupling combination ensures intra-group variance serves reward estimation while pushing the model toward high-score modes of the original prompt.

### Loss & Training
Phase I: Learn $\Delta$ separately using $\mathcal{L}_{\text{emb}}$, with $\bm{\delta}_k \sim \mathcal{N}(0, \sigma_{\text{init}}^2 \mathbf{I})$ initialization. Phase II: Apply DiffusionNFT's self-normalized $\bm{x}_0$-regression loss, splitting reward $r(\bm{x}_0, \bm{c}) \in [0,1]$ into weighted regression terms for positive/negative policies $v_\theta^{\pm} = (1\mp\beta)v^{\text{old}} \pm \beta v_\theta$. Backbone is SD3.5-Medium on 8×H20, with three configurations: High-Exploration ($G=4, K=12$), Efficient ($G=2, K=4$), and Human Preference ($G=8, K=3$).

## Key Experimental Results

### Main Results

Comparison of GenEval rewards, in-domain rewards, and diversity metrics after training (SD3.5-M is zero-shot baseline; lower IDS is better, higher others are better):

| Method | $G$/$K$ | GenEval ↑ | IDS ↓ | ASC ↑ | SDI ↑ | PVS ↑ |
|------|---------|-----------|-------|-------|-------|-------|
| SD3.5-M | — | 0.263 | 0.044 | 0.143 | 0.458 | 0.392 |
| Flow-GRPO | 24/— | 0.776 | 0.064 | 0.123 | 0.422 | 0.318 |
| DiffusionNFT | 24/— | 0.922 | 0.054 | 0.118 | 0.418 | 0.259 |
| DiffusionNFT | 48/— | 0.917 | 0.051 | 0.109 | 0.463 | 0.196 |
| **E²PO (Ours)** | 4/12 | **0.932** | **0.048** | **0.127** | **0.467** | **0.322** |

Generalization to unseen rewards under PickScore ($K=3$):

| Method | $G$ | PickScore ↑ | Aesthetic ↑ | ImgRwd ↑ | HPSv2.1 ↑ | IDS ↓ |
|------|------|-------------|-------------|----------|-----------|-------|
| SD3.5-M | — | 19.93 | 5.600 | -0.50 | 0.203 | 0.044 |
| Flow-GRPO | 24 | 22.72 | 6.273 | 1.30 | 0.324 | 0.222 |
| DiffusionNFT | 24 | 23.34 | 6.514 | 1.27 | 0.324 | 0.239 |
| **E²PO (Ours)** | 8 | **23.38** | **6.538** | **1.29** | **0.325** | **0.167** |

### Ablation Study

| Configuration | Key Phenomenon | Description |
|------|---------|------|
| Full E²PO ($G=4, K=12$) | GenEval 0.932 | Noise + semantic dual-source exploration is most stable. |
| Extreme $G=1$ (Only $K$) | Significant drop across tasks | Noise prior alone is insufficient for exploration. |
| Extreme $K=1$ (NFT baseline) | Comparable to high-$G$ NFT | Loss of discriminative variance stabilizer. |
| Static $\mathbf{C}^{\text{opt}}_k$ (No schedule) | Semantic drift and artifacts | Semantic perturbations contaminate low-noise stages. |
| No $\mathbf{C}^{\text{orig}}$ Anchor | Slower convergence, worse peak performance | Perturbations bias the policy without an anchor. |

### Key Findings
- Intra-group variance monitoring (Fig. 2) shows baseline reward std drops to the log-scale "floor" within 150 steps, while E²PO maintains stable variance throughout training—the direct reason for sustained gains in GenEval/PickScore.
- Given a fixed compute budget $N=G\times K$, neither "all-in $G$" nor "all-in $K$" is optimal; a balanced allocation ($G=4, K=12$) performs best, indicating noise and semantic diversity are complementary.
- E²PO aligns with the zero-shot baseline on IDS (0.048 vs 0.044) while achieving the highest GenEval, demonstrating that it increases reward without sacrificing diversity. Qualitative results show superiority in tasks prone to reward hacking like counting and attribute binding.

## Highlights & Insights
- Shifting the source of "sustained exploration" from noise space to embedding space is a clever cognitive pivot—high-dimensional isotropic noise priors are often powerless once the model has converged, whereas a small step on the low-dimensional semantic manifold can switch to entirely different generation modes.
- Explicitly constraining perturbations using $\mathcal{L}_{\text{div}} + \mathcal{L}_{\text{anc}}$ effectively draws a "near but not identical" ring on the embedding sphere. This is simpler and more controllable than KL penalties.
- The "sampling with perturbations + updating with anchor" decoupling can be transferred to any RL scenario where reward estimation requires diversity but the policy itself should remain unimodal (e.g., applying embedding perturbations to LLM prompts during RLHF while keeping updates centered).

## Limitations & Future Work
- Perturbations are learned per-prompt, requiring a $\Delta$ optimization pass for unseen prompts during online RL, which is not user-friendly for inference-time applications; an amortized prompt $\to \Delta$ generator would be more practical.
- Evaluation is limited to SD3.5-Medium; transferability to larger models (SD3.5-Large/FLUX) or non-NFT policy gradient frameworks is not fully explored.
- The optimal $(G, K)$ balance likely depends on the backbone and reward type. The paper provides empirical values but lacks a theoretical analysis based on reward landscape properties.

## Related Work & Insights
- **vs Flow-GRPO / DiffusionNFT**: These are also group-based RL alignment methods but explore only in latent noise; E²PO argues this path has no solution for variance collapse and introduces structural diversity at the embedding layer.
- **vs Textual Inversion / DreamBooth**: These learn embeddings to replicate static concepts. E²PO treats embedding learning as a dynamic exploration mechanism within online RL, moving prompt tuning into the RL inner loop.
- **vs Initial Noise Diversification**: While diversifying initial noise, prior work noted this can lead to reward hacking with visually incoherent images. E²PO’s semantic anchoring naturally constrains exploration within the prompt's meaning.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Attributes the GRPO variance problem to signal source selection and provides a clean alternative via embedding perturbation; a refreshing incremental improvement.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Coversions GenEval/PickScore with diversity metrics and three ablation sets, though it lacks verification on larger backbones.
- **Writing Quality**: ⭐⭐⭐⭐ Clear chain of motivation $\to$ phenomenon $\to$ formula $\to$ ablation. Fig. 2's variance monitoring is very persuasive.
- **Value**: ⭐⭐⭐⭐ Provides a plug-and-play stabilizer for flow/diffusion RL alignment with low implementation cost; expected to be rapidly adopted by subsequent work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Offline Preference Optimization for Rectified Flow with Noise-Tracked Pairs](offline_preference_optimization_for_rectified_flow_with_noise-tracked_pairs.md)
- [\[ICML 2026\] Principled RL for Flow Matching Emerges from the Chunk-level Policy Optimization](principled_rl_for_flow_matching_emerges_from_the_chunk-level_policy_optimization.md)
- [\[AAAI 2026\] Rethinking Direct Preference Optimization in Diffusion Models](../../AAAI2026/image_generation/rethinking_direct_preference_optimization_in_diffusion_models.md)
- [\[CVPR 2026\] Neighbor GRPO: Contrastive ODE Policy Optimization Aligns Flow Models](../../CVPR2026/image_generation/neighbor_grpo_contrastive_ode_policy_optimization_aligns_flow_models.md)
- [\[ICML 2026\] Adversarial Flow Models](adversarial_flow_models.md)

</div>

<!-- RELATED:END -->
