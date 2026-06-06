---
title: >-
  [Paper Note] Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences
description: >-
  [ICML 2026][LLM Safety][Federated Preference Alignment] This paper proposes FedVPA-GP: under the privacy constraints of Federated Learning (FL)…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Federated Preference Alignment"
  - "Variational Inference"
  - "Gumbel-Softmax Mixture Prior"
  - "Posterior Collapse"
  - "Personalized RLHF"
date: 2026-05-08
content_hash: b06f1583384995d3
---

# Federated Variational Preference Alignment with Gumbel-Softmax Prior for Personalized User Preferences

**Conference**: ICML 2026  
**arXiv**: [2605.30873](https://arxiv.org/abs/2605.30873)  
**Code**: TBD  
**Area**: Alignment RLHF / Federated Learning / LLM Personalization  
**Keywords**: Federated Preference Alignment, Variational Inference, Gumbel-Softmax Mixture Prior, Posterior Collapse, Personalized RLHF

## TL;DR
This paper proposes FedVPA-GP: under the privacy constraints of Federated Learning (FL), it models each client's preference as a continuous latent variable $z$ using a "client mixture prior + Gumbel-Softmax learnable weights + orthogonal prototype loss." This fundamentally resolves the "posterior collapse" encountered when directly applying Variational Preference Learning (VPL) to FL, enabling a single reward model to switch dynamically between conflicting preferences like "helpful" and "harmless."

## Background & Motivation
**Background**: Current mainstream LLM alignment workflows (RLHF / DPO / IPO / KTO) assume that preference data can be centralized to train a global reward model. To circumvent privacy and compliance constraints, federated schemes like FedDPO and FedBiscuit have recently emerged, allowing RLHF to be completed locally on clients while only exchanging gradients or lightweight selectors.

**Limitations of Prior Work**: All existing federated schemes default to the assumption that human preferences can be fitted by a monolithic reward function. However, datasets like HH-RLHF show that "helpful" and "harmless" demands are often in direct conflict. Averaging heterogeneous preferences across clients into a consensus model forces a non-existent "lowest common denominator," failing to satisfy either objective effectively.

**Key Challenge**: Personalization requires building a preference representation for each user/client. However, in a federated setting, each client has extremely few samples and highly heterogeneous distributions (in extreme cases, only seeing "helpful" or "harmless" preferences). If centralized Variational Preference Learning (VPL) is applied directly, the KL regularization term overwhelms the reconstruction term, pulling the posterior distribution back to $\mathcal{N}(0, I)$. This leads to classic "posterior collapse," where the latent variable $z$ loses information and personalization fails (as shown in Fig 2(a), where $z_{\text{VPL}}$ of FedVPL clusters into a single blob).

**Goal**: Without exchanging raw preference data, ensure that local variational inference for each client is (a) stable (not broken by data sparsity) and (b) decoupled (different preference patterns are clearly separated in the latent space).

**Key Insight**: "Local sparsity" is essentially a "lack of global context," while the collective distribution in FL can act as a dynamic prior. Instead of raw data, posterior statistics $(\bar\mu_j, \bar\sigma_j^2)$ are transmitted, allowing each client to treat "others' posteriors" as its own prior. By adding a geometric constraint to keep prototypes mutually orthogonal, the encoder is forced to map different preferences into distinct subspaces.

**Core Idea**: Replace the standard Gaussian prior with a "weighted mixture of peer posteriors" and use Gumbel-Softmax to make mixture weights learnable for each client. Furthermore, use an orthogonal prototype loss to fix the "helpful mode" and "harmless mode" onto orthogonal directions in the latent space, simultaneously addressing "training instability" and "posterior collapse."

## Method

### Overall Architecture
FedVPA-GP follows the two-stage paradigm of FedBiscuit but replaces the core selector with a variational module. The base LLM (Qwen-2 0.5B / Gemma-2B) is frozen throughout, with a minimal trainable part (approx. 0.18% of parameters):

**Stage 1 (Federated Selector Training)**: Each client $i$ holds a preference pair dataset $\mathcal{D}_i=\{(s_A, s_B, y)\}$. The local process involves "LLM feature extraction $\rightarrow$ calculating response difference $\Delta h = h_{\text{chosen}} - h_{\text{rejected}}$ $\rightarrow$ feature extraction via MLP $\rightarrow$ variational encoder outputting $(\mu_i, \sigma_i^2)$ $\rightarrow$ reparameterization sampling $z_i$ $\rightarrow$ adding $f_\theta(z_i)$ as a logit residual to the base $\{A,B\}$ logits $\rightarrow$ calculating BTL preference likelihood." The local loss consists of ELBO plus orthogonal regularization, where the KL term pushes $q_i$ toward the "federated mixture prior." The server aggregates LoRA/encoder parameters and broadcasts the $(\bar\mu_j, \bar\sigma_j^2)$ from all participating clients of the previous round to the next—serving as the carrier for the mixture prior. A parallel server-side balanced $k$-means clusters client means to assign an orthogonal prototype index $y_i^*$ to each client.

**Stage 2 (Conditional RLHF)**: Once the selector is trained, it is frozen to serve as a conditional reward model $\text{logits}(s_A, s_B \mid z)$. On the server side, a policy conditioned on $z$ is trained via DPO: $z$ is injected into input embeddings through a z-to-embedding mapping, and two responses are generated on-policy. These are scored by the selector given $z$, yielding (chosen, rejected) pairs for DPO updates. This step uses prompts only and does not touch user preference labels, offloading "expensive federated generation" entirely to the server.

### Key Designs

1.  **Federated Mixture Prior + Gumbel-Softmax Learnable Weights**:
    -   **Function**: Replaces the fixed standard Gaussian prior $\mathcal{N}(0, I)$ in VPL, making each client's prior a weighted combination of other clients' posteriors to inject collective prior information and stabilize local inference under sparse data.
    -   **Mechanism**: The prior is defined as $p_{\text{mixture}}^{(i)}(z) = \sum_{j \in \mathcal{S}} w_j \mathcal{N}_j(z)$, where $\mathcal{N}_j$ is the posterior uploaded by peer $j$ in the previous round. The weights $w_j$ are not simple averages but are learned end-to-end with the KL term using Gumbel-Softmax reparameterization $w_j = \mathrm{softmax}((\log\pi_j + g_j)/\tau)$. Each client maintains its own $\{\pi_j\}$, which is not federated, preserving the local strategy of "which peers to trust." The KL term $\mathbb{D}_{KL}(q_i \,\|\, p_{\text{mixture}}^{(i)})$ is evaluated stably via log-sum-exp.
    -   **Design Motivation**: A fixed Gaussian prior in FL provides "no global guidance," causing the KL term to collapse the posterior to the origin. Conversely, blindly mixing all peers averages out conflicting preferences. Learnable Gumbel-Softmax allows clients with similar preference patterns to provide priors for each other while shielding different patterns, automatically achieving "neighbor filtering." Transmitting only $(\bar\mu_j, \bar\sigma_j^2)$ (256 bytes) is much smaller than LoRA adapters, adding negligible communication overhead.

2.  **Orthogonal Prototype Loss (Orthogonal Loss)**:
    -   **Function**: Explicitly partitions multiple mutually orthogonal preference subspaces in the latent space, forcing the encoder to place conflicting preferences in different directions to geometrically cure posterior collapse.
    -   **Mechanism**: Maintains $M$ learnable prototypes $\{\mathbf{p}_m\}_{m=1}^M$, initialized via QR decomposition to ensure strict initial orthogonality and distance from the origin. After each aggregation round, the server uses balanced $k$-means to assign a prototype label $y_i^*$ to each client (in HH-RLHF experiments, $M=2$ for helpful/harmless axes). The local loss is $\mathcal{L}_{\text{ortho}}(z) = \|z - \mathbf{p}_{y_i^*}\|_2^2 + \gamma\|\mathbf{P}\mathbf{P}^T - \mathbf{I}_M\|_F^2$, where the first term pulls $z$ of the current sample toward the assigned prototype, and the second prevents prototype collapse.
    -   **Design Motivation**: KL regularization alone cannot guarantee separation of patterns (t-SNE of FedVPL shows entangled $z$). Introducing "labeled geometric attractors" injects a "discrete preference structure" prior into the continuous latent space. This provides a clear training target for the encoder and an addressable $z$ for the Stage 2 conditional policy. $M$ can be increased to accommodate finer-grained preference spectra.

3.  **Variance Capping + Base-logit Dropout**:
    -   **Function**: Provides double security at the engineering level to block "posterior collapse" escape routes.
    -   **Mechanism**: (a) A hard truncation is applied to the encoder's log-variance output: $\log\sigma_i^2 \leftarrow \min(\log\sigma_i^2, \log\sigma_{\max}^2)$, preventing the encoder from "cheaply matching the prior" by inflating $\sigma$; (b) For small models like Qwen-2 0.5B with strong base signals for $\{A,B\}$, Bernoulli dropout ($p_{\text{logit}}=0.5$) is applied to base choice-logits. This forces $f_\theta(z)$ to assume prediction responsibility on those steps, ensuring gradients flow back to $z$. For Gemma-2B, $p_{\text{logit}}=0$ as the base signal is weaker.
    -   **Design Motivation**: The VAE field has long observed that the KL term is an easy target for shortcuts—if $q$ equals $p$, KL becomes 0 and $z$ carries no information. Capping $\sigma$ blocks the "variance inflation" shortcut, while logit dropout blocks the "base model already knows the answer" shortcut, forming a layered defense with the other designs.

### Loss & Training
The total local loss is defined in Eq. (10):

$\mathcal{L}_i(\theta, \phi) = \mathcal{L}_{\text{recon}} + \beta \cdot \mathbb{D}_{KL}(q_\phi(z\mid\mathcal{D}_i) \,\|\, p_{\text{mixture}}^{(i)}(z)) + \lambda \cdot \mathcal{L}_{\text{ortho}}(z)$

Where $\mathcal{L}_{\text{recon}}$ is the negative log-likelihood of BTL preferences. Stage 1 utilizes LoRA fine-tuning + FedAvg to aggregate LoRA/encoder parameters ($\pi_j$ excluded). Stage 2 runs DPO on the server, injecting $z$ via additive embeddings into the policy and using the frozen selector's $\text{logits}(s_A, s_B \mid z)$ as instant rewards. For HH-RLHF, $K\in\{10,50,100\}$, sampling 5 or 10 clients per round; $M=2$ for helpful/harmless.

## Key Experimental Results

### Main Results
GPT-4o Judgment Win-rate (%) on HH-RLHF, Non-IID split (50% clients see helpful, 50% see harmless).

| Base | Method | 10 Clients H/Hm | 50 Clients H/Hm | 100 Clients H/Hm |
|------|------|----------------|----------------|------------------|
| Qwen-2 0.5B | FedDPO | 48.12 / 77.34 | 43.05 / 69.22 | 41.48 / 67.15 |
| Qwen-2 0.5B | FedBiscuit | 48.85 / 75.12 | 44.21 / 71.45 | 42.33 / 69.42 |
| Qwen-2 0.5B | FedVPL (naive) | 62.24 / 84.56 | 54.18 / 78.12 | 53.05 / 77.34 |
| Qwen-2 0.5B | **FedVPA-GP** | **66.45 / 89.21** | **58.32 / 84.05** | **55.18 / 82.31** |
| Gemma-2B | FedDPO | 52.34 / 83.12 | 44.15 / 78.45 | 41.22 / 75.33 |
| Gemma-2B | FedBiscuit | 51.65 / 82.45 | 46.21 / 78.12 | 43.44 / 76.05 |
| Gemma-2B | FedVPL (naive) | 66.82 / 89.15 | 56.41 / 84.34 | 53.25 / 80.42 |
| Gemma-2B | **FedVPA-GP** | **73.21 / 96.34** | **64.48 / 95.12** | **60.15 / 92.45** |

FedVPA-GP achieves Pareto improvements (not sacrificing one goal for another) across both models and all client scales. As client numbers increase and local data becomes sparser, all baselines drop significantly, while Ours shows the smallest decline, validating the anti-sparsity effect of mixture priors.

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| FedVPL (naive) | 62.24 / 84.56 (Qwen, N=10) | Standard Gaussian prior + no orthogonality, acting as lower bound |
| FedVPL + Ortho | Between FedVPL and Full | Adding orthogonal loss alone mitigates posterior collapse |
| FedVPL + GB Prior | Between FedVPL and Full | Adding mixture prior alone stabilizes training under sparse data |
| Full FedVPA-GP | 66.45 / 89.21 (Qwen, N=10) | Synergy between both yields the best trade-off |

Generalization to unseen clients (20 clients, 10 train / 10 test): FedVPA-GP achieves 63.16 / 91.23 on Unseen, nearly identical to Seen (65.28 / 94.25). Contrastingly, FedVPL drops sharply on Unseen from 56.23 / 83.82 to 49.25 / 75.21, proving the latent space learned here is continuous and semantic, allowing adaptation to new users via inference alone.

Imbalanced client ratios (H/Hm at 70/30, 30/70, etc., Qwen N=10): FedVPA-GP consistently outperforms FedBiscuit by ~17–20 points in helpfulness and 12–17 points in harmlessness—Gumbel-Softmax weights automatically assign higher weights to "minority peers," preventing minority preference patterns from being drowned out by the majority.

### Key Findings
- **Visual Validation of Posterior Collapse**: Figure 3 t-SNE shows that while FedVPL mixes red and blue points throughout training, FedVPA-GP separates the two types of latent variables around orthogonal prototypes (∗) within a few rounds, confirming the synergy of "mixture priors providing global context + orthogonal loss providing a geometric backbone."
- **Near-Zero Cost for Personalized Deployment**: The variational module has only ~0.9M parameters (0.18% of the base). Single-round communication only adds 256 bytes ($\bar\mu, \bar\sigma^2$ at 32 dimensions each), and training latency is only 1.18× that of FedDPO—meaning any federated alignment pipeline using LoRA can replace its selector with this method directly.
- **Systemic Deficiencies of Monolithic Reward Models**: Helpfulness win-rates for FedDPO/FedBiscuit stagnate or drop as client numbers increase on Qwen-2, fundamentally due to heterogeneous preferences canceling each other out. This proves that the "average first" paradigm is wrong for conflicting preference scenarios.

## Highlights & Insights
- **Turning FL's "Weakness" into a "Prior"**: FL is usually seen as a disaster for VPL (local sparsity + heterogeneity). Ours reverses this by treating "group posteriors" as the most natural dynamic prior—requiring no raw data exchange while filling the missing global perspective for individual clients.
- **Gumbel-Softmax on "Prior Mixture Weights"**: This makes the choice of "which peers to trust" an end-to-end learnable discrete selection, equivalent to embedding a "soft clustering" layer within variational inference. This is more robust than manual similarity metrics (e.g., cosine or client clustering) and is transferable to any federated task requiring personalized neighbor selection.
- **Orthogonal Prototype Loss + Balanced $k$-means**: This explicitly declares an inductive bias that the "preference space is a superposition of finite discrete modes with continuous fine-tuning." Orthogonal prototypes ensure independence between modes, while continuous $z$ captures individual differences. This "discrete skeleton + continuous filling" design can be ported to multi-objective alignment or multi-task personalization.
- **Double Insurance Against Collapse**: Variance truncation and base-logit dropout are subtle but crucial engineering details. They serve as a reminder to identify trivial solutions that satisfy the objective function when building variational models.

## Limitations & Future Work
- For HH-RLHF, $M=2$ perfectly matches the helpful/harmless axes. The stability of balanced $k$-means label assignment when $M$ is large and preference dimensions are finer remains unverified; orthogonal prototypes may become an overly strong prior if the preference spectrum is a continuous manifold rather than discrete clusters.
- The latent variable $z$ is a 32-dimensional "black box direction." The paper lacks a unified interface for selecting $z$ during deployment—Algorithm 1 mentions three ways (mean / $\bar\mu_i$ / online $q_\phi$ inference) but lacks analysis of online latency vs. privacy trade-offs.
- Experiments are limited to $\le$2B base models. With 70B+ models, base logits are extremely strong; whether $p_{\text{logit}}$ should be increased further or if $\beta, \lambda$ require extensive re-searching is unverified.
- Privacy discussions focus on "no raw data exchange," but transmitting $(\bar\mu_j, \bar\sigma_j^2)$ might still leak preference statistics. Applying this to DP-Federated settings would require adding noise and re-analyzing its coupling with the KL term.

## Related Work & Insights
- **vs FedBiscuit (Wu et al., 2024)**: Both train "lightweight binary selectors + frozen base + two-stage RLHF," but FedBiscuit uses a monolithic reward. Ours upgrades the selector to a conditional variational model, maintaining compatibility as a drop-in replacement.
- **vs FedDPO (Ye et al., 2024)**: FedDPO federates DPO gradients. Ours federates the "posterior of preference latents," keeping the expensive policy DPO on the server side—this decoupling is superior in communication and stability.
- **vs Variational Preference Learning (Poddar et al., 2024)**: Original VPL assumes pooled data to estimate posteriors. This work addresses how to survive VPL in FL by framing posterior stability as a "prior deficiency" and offering an engineered solution.
- **vs Multi-objective alignment (Rame et al., 2023) / Weight Merging (Jang et al., 2023)**: Those methods combine weights or objectives. Ours combines in the latent dimension, which is parameter-efficient and naturally supports switching preferences of the same policy via $z$.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Deeply adapting VPL to FL with mixture priors and orthogonal prototypes is a well-posed novelty, though individual components have precedents in their respective communities.
- **Experimental Thoroughness**: ⭐⭐⭐ Main table covers 2 models × 3 scales, including unseen client generalization and imbalance ablations. However, it lacks datasets beyond HH-RLHF, tests only $M=2$, and lacks validation on larger base models.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear (narrative of "posterior collapse as the core FL+VPL problem" is smooth), pseudocode is complete, and reproducibility is high.
- **Value**: ⭐⭐⭐⭐ Simultaneously addresses "handling conflicting preferences in federated alignment" and "preventing variational inference collapse in FL." Highly relevant for personalized, privacy-preserving LLM alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)
- [\[NeurIPS 2025\] A Systematic Evaluation of Preference Aggregation in Federated RLHF for Pluralistic Alignment of LLMs](../../NeurIPS2025/llm_safety/a_systematic_evaluation_of_preference_aggregation_in_federated_rlhf_for_pluralis.md)
- [\[ICML 2026\] From Volume to Value: Preference-Aligned Memory Construction for On-Device RAG](from_volume_to_value_preference-aligned_memory_construction_for_on-device_rag.md)
- [\[ICML 2026\] Decoupled Training with Local Reinforcement Fine-Tuning in Federated Learning](decoupled_training_with_local_reinforcement_fine-tuning_in_federated_learning.md)
- [\[ICML 2026\] Position: Retire the "Positive Backdoor" Label -- Secret Alignment Requires Strict and Systematic Evaluation](position_retire_the_positive_backdoor_label_--_secret_alignment_requires_strict_.md)

</div>

<!-- RELATED:END -->
