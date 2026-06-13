---
title: >-
  [Paper Note] LOCKET: Robust Feature-Locking Technique for Language Models
description: >-
  [ACL 2026][Dialogue Systems][Feature-Locking] LOCKET is a password-free, scalable, and jailbreak-resistant feature-locking scheme designed for the "pay-to-unlock" commercial model of LLMs. It trains a LoRA adapter for ea…
tags:
  - "ACL 2026"
  - "Dialogue Systems"
  - "Feature-Locking"
  - "LoRA Adapter Merging"
  - "Latent Adversarial Training"
  - "Spectral Norm Clipping"
  - "Jailbreak Defense"
  - "Pay-per-feature"
date: 2026-05-08
content_hash: 230e5425a33f409e
---

# LOCKET: Robust Feature-Locking Technique for Language Models

**Conference**: ACL 2026  
**arXiv**: [2510.12117](https://arxiv.org/abs/2510.12117)  
**Code**: https://github.com/ssg-research/locket (Available)  
**Area**: LLM Control / Feature Locking / Model Commercialization  
**Keywords**: Feature-Locking, LoRA Adapter Merging, Latent Adversarial Training, Spectral Norm Clipping, Jailbreak Defense, Pay-per-feature

## TL;DR
LOCKET is a password-free, scalable, and jailbreak-resistant feature-locking scheme designed for the "pay-to-unlock" commercial model of LLMs. It trains a LoRA adapter for each locked feature (enhanced with Latent Adversarial Training for robust refusal). When merging multiple adapters, it applies per-layer spectral norm clipping to prevent "over-refusal" collapse. Tested across 3 models and 4 features (Math/SQL/Summarize/MMLU), it achieves a 100% refusal rate, $\leq$ 7% utility loss, and $\leq$ 5% jailbreak attack success rate, significantly outperforming password-locking baselines.

## Background & Motivation

**Background**: LLM providers like OpenAI and Anthropic currently use "tiered subscription" models (Free = Basic, Paid = Premium) for API sales. OpenAI's Sam Altman mentioned on Twitter that "Pro subscriptions are losing money," which is unsustainable. SaaS and mobile games have long shifted to more granular "pay-to-unlock" models, but LLMs lack the technical foundation to support "free base models with paid unlocking of advanced capabilities like math, coding, or summarization."

**Limitations of Prior Work**: Implementing this business model requires a Feature-Locking Technique (FLoTE) that meets four requirements: (R1) **Effective**—must refuse unauthorized features; (R2) **Utility-Preserving**—authorized features must maintain performance; (R3) **Robust**—resistant to jailbreaks, credential sharing, and brute-forcing; (R4) **Scalable**—support for multiple features and customers without combinatorial explosion. Existing password-locking schemes (Greenblatt 2024 / Tang 2024 / Su 2025 / Hofstätter 2025) fail: they suffer from severe utility loss, lack defense against adaptive jailbreaks, use vulnerable/sharable passwords, or require re-tuning the entire model for every new feature/customer.

**Key Challenge**: Traditional methods bind unlocking to a **secret credential**. If leaked, the mechanism collapses. Supporting multiple features/customers via SFT on the whole model inevitably causes catastrophic forgetting. These issues are interlinked: avoiding passwords requires "intrinsic functional locks," but intrinsic locks via SFT destroy utility. Alternative "strawman" solutions like System Prompts, Unlearning, API Routers, or Prompt Filtering all violate at least R3 or R4.

**Goal**: (a) Formally define R1-R4 for FLoTE; (b) Design a password-free, modular adapter-based FLoTE that allows "train once, reuse forever"; (c) Solve the over-refusal collapse caused by the accumulation of refusal directions when merging multiple adapters.

**Key Insight**: Convert "functional locks" from "password-triggered backdoors" into "hot-swappable LoRA adapters." One adapter is trained for each feature to be locked. Upon user login, an access control module dynamically attaches "unauthorized feature adapters" to the base model, forcing refusal for those queries. Authorized features remain functional as no adapter is attached. No password means nothing to steal or share; per-feature adapters mean adding new features is $O(N)$ instead of $O(2^N)$.

**Core Idea**: Use LoRA adapters instead of password triggers + LAT to enhance refusal robustness + spectral norm clipping to prevent refusal direction explosion during multi-adapter merging.

## Method

### Overall Architecture
LOCKET consists of two independent processes:

**Offline Training**: For each feature $f \in \mathcal{F}$, a LoRA adapter $a_f$ is trained independently. The objective is $\mathcal{L}_{\text{lock}} = \mathcal{L}_{\text{utility}} + \mathcal{L}_{\text{robust}}$. The former uses KL divergence against a frozen reference model $\pi_{\theta'}$ to preserve basic dialogue capabilities; the latter is a refusal-enhancement loss via Latent Adversarial Training (LAT). Each adapter is roughly 1.6-1.7% of the base model parameters (e.g., 120-130M for DeepSeek-7B/Llama-3-8B).

**Online Inference**: Upon login: (1) The Authorization Module updates the user's authorized feature set based on payment/credentials; (2) For every request, the Access Control Module identifies the **unauthorized** adapter set $\{a_k : k \notin \text{auth}(C)\}$, merges them using LOCKET Merging, and attaches them to the frozen base LLM. Attaching occurs once at login ($\sim 1$s), and TTFT is unaffected by the number of adapters ($\sim 3$ms).

### Key Designs

1.  **Latent Adversarial Training (LAT) for Robust Refusal**:
    *   **Function**: Ensures the adapter is robust against jailbreaks (e.g., GCG, AutoDAN-Turbo, Many-shot). Even if an attacker uses adversarial prompts to "activate" a locked feature, the model consistently refuses.
    *   **Mechanism**: For each prompt $x_i$, pairs are defined (chosen = fixed refusal string $c_i$, rejected = ground-truth answer $r_i$). It first finds the "worst-case perturbation" $\delta_i$ in latent activations to maximize evasion: $\delta_i = \arg\min_\delta \mathcal{L}_{\text{evade}}(c_i, r_i; \delta)$, where $\mathcal{L}_{\text{evade}} = -\log \pi_\theta(c_i | \alpha(x_i, \delta)) - \log(1 - \pi_\theta(r_i | \alpha(x_i, \delta)))$. This is solved via 16-step PGD within $\|\delta\|_2 \leq \epsilon$. The adapter weights are then updated using this $\delta_i$.
    *   **Design Motivation**: Standard SFT/refusal leaves a "refusal direction" in latent space that is vulnerable to adaptive attacks. LAT reinforces this direction across the entire worst-case perturbation ball, equivalent to adversarial training. This is how LOCKET reduces jailbreak ASR from 0.95 in PWD to 0.05.

2.  **LOCKET Merging: Spectral Norm Clipping**:
    *   **Function**: Prevents over-refusal disasters during merging, where refusal directions overlap and norms explode, causing the model to output "Sorry..." for all prompts (including unlocked features).
    *   **Mechanism**: Two phases—**Offline**: For each layer $\ell$ and adapter $a_i$, compute the SVD $\Delta W_\ell^i \approx \mathbf{U}^i \mathbf{S}^i (\mathbf{V}^i)^T$ and identify the spectral norm $\sigma^i = \|\Delta W_\ell^i\|_2$. Define a threshold $Clip_\ell = \tau \cdot \max_i \sigma^i$ (where $\tau \in (0, 1]$). **Online**: Merge unauthorized adapters using CAT: $\Delta W_\ell = \sum_{i \in L} \Delta W_\ell^i$. If $\|\Delta W_\ell\|_2 > Clip_\ell$, rescale: $\Delta W_\ell \leftarrow \frac{Clip_\ell}{\|\Delta W_\ell\|_2} \Delta W_\ell$.
    *   **Design Motivation**: Standard merging (CAT, TIES, DARE) collapses on LAT-trained adapters. Appendix Table 8 shows that SVD/DARE methods either lead to $ACC=0$ (over-refusal) or $ALA=0.37$ (locking failure). Clipping ensures the merged norm does not exceed that of a single strong adapter, keeping the refusal effective but controlled.

3.  **Modular Training + Dynamic Attachment**:
    *   **Function**: Reduces $O(2^N)$ complexity to $O(N)$ and removes the need to retrain for new customers or授权 changes.
    *   **Mechanism**: Each feature $f$ is trained on its own $D_f$. Adding a new feature requires only one new adapter.
    *   **Design Motivation**: The authors chose independent adapters over a single multi-feature adapter because the latter still faces $O(2^N)$ complexity. Merging provides a $2^N-1$ combination space with only $O(N)$ storage and native compatibility with PEFT frameworks.

### Loss & Training
Adapters: LoRA rank=64, alpha=64, RSLoRA scaling. LAT: 16-step PGD on layers $[8, 16, 24, 30]$. Utility dataset: UltraChat (165k). Feature datasets: MATH (7.5k), SQL Create Context (62.8k), SAMSum (819), MMLU (99.8k). Merging hyperparameter $\tau$ is grid-searched (e.g., 0.75 for DeepSeek-7B-Math). Compute: 8×A100 40GB × 6000 GPU hours.

## Key Experimental Results

### Main Results
Effectiveness and Utility across 3 models and 4 features (M/Q/S/U):

| Model | Lock → | Baseline M | Lock M (on M) | Lock M (on Q) | Lock M (on S) | Lock M (on U) | Note |
|------|------|-----------|----------|----------|----------|----------|------|
| DeepSeek-7B-Math | Math | 0.40 | **0.00** | 0.95 | 0.23 | 0.51 | 100% lock on M, others stable |
| DeepSeek-7B-Math | SQL | 0.93 | 0.45 | **0.00** | 0.24 | 0.50 | Slight M gain when locking Q |
| DeepSeek-7B-Math | MMLU | 0.53 | 0.42 | 0.93 | 0.24 | **0.00** | 100% lock on U |
| Llama-3-8B-Instruct | Math | 0.28 | **0.00** | 0.92 | 0.34 | 0.64 | 6% loss on U (M-U interference) |

Diagonal results are 0.00 (100% refusal), with non-diagonal utility losses $\leq 7\%$ (mostly within $\pm 2\%$). Compared to PWD, LOCKET preserves utility much better. **Robustness** (ASR, lower is better):

| Attack | Feature | PWD ASR | LOCKET ASR |
|------|--------|---------|-----------|
| Many-shot | Math | 0.57 | **0.00** |
| GCG | Math | 0.87 | **0.01** |
| AutoDAN-Turbo | Math | 0.95 | **0.05** |

Scalability: Across all 15 combinations of 4 features, effectiveness remained at 100% and utility loss $\leq 7\%$.

### Ablation Study
Comparison of merging methods (DeepSeek-7B-Math, 4 LAT adapters, Math unlocked):

| Merging Method | Unlock Math ACC↑ | Refusal ALA↓ (others) | Conclusion |
|---------|--------------|-------------------------------|------|
| Baseline (No LOCK) | 0.40 | 0.00 | Upper bound |
| **LOCKET Merging** | **0.45** | **0.00** | ✓ Utility kept + Perfect lock |
| Pure SVD | 0.00 | 0.00 | Over-refusal collapse |
| DARE TIES | 0.40 | 0.37 | Utility OK but locking fails |

$\tau$ Sensitivity: $\tau > 0.9$ fails to lock; $\tau < 0.5$ causes over-refusal. The sweet spot is $\tau \in [0.7, 0.85]$.

### Key Findings
*   **Spectral norm clipping is essential for merge stability**: All other LoRA merging methods collapse or fail with LAT adapters. Standard methods ignore the magnitude of the refusal direction, while clipping specifically targets it.
*   **PWD suffers from catastrophic forgetting in multi-lock settings**: Training PWD for M+Q+S drops Summarize performance from 0.27 to 0.12. LOCKET avoids this by frozen base weights.
*   **Feature interference vs. Catastrophic forgetting**: The 6% drop in Math vs. MMLU is due to semantic overlap (math problems in MMLU), not forgetting.
*   **Marginal cost for new features is low**: Expanding to 8 features only requires 4 additional adapters. Effectiveness remains high.
*   **Storage efficiency**: Adapters are only ~130M, saving $100\times$ storage compared to independent full models.

## Highlights & Insights
*   **Business-driven technical research**: The paper derives FLoTE requirements R1-R4 from the economic reality of "OpenAI Pro losing money." The narrative is highly relevant for industry deployment.
*   **Spectral norm clipping as a lightweight trick**: Using SVD thresholds for LoRA merging is elegant and requires no retraining. This can generalize to any scenario involving the merging of multiple alignment adapters.
*   **Innovative use of LAT**: Repurposing LAT from "preventing jailbreaks" to "reinforcing functional refusal" is a key innovation, reducing jailbreak ASR by an order of magnitude compared to SFT.
*   **Complete FLoTE solution**: The modular adapter + dynamic merge + spectral clipping trio solves the four major pitfalls of password models with negligible overhead (1s attach time / constant TTFT).

## Limitations & Future Work
*   **Unresolved Feature Interference**: Semantic overlap (e.g., Math in MMLU) leads to unavoidable utility drops. Data cleaning is suggested but not fully explored.
*   **Jailbreak Arms Race**: Current $\leq 5\%$ ASR is relative to existing attacks. Future white-box attacks like adapter inversion might pose new risks.
*   **Closed-box context**: LOCKET is designed for black-box API settings. It offers limited protection if the base model and adapters are both released openly.
*   **No Energy Savings**: LOCKET solves revenue issues, not inference costs. It still burns the same compute as the base model.
*   **Scalability to large models**: While 70B results are in the appendix, the scaling laws of this approach were not systematically studied.

## Related Work & Insights
*   **vs Greenblatt et al. 2024**: Move from password-triggered backdoors to credential-free adapters to prevent credential sharing.
*   **vs Tang et al. 2024**: Better scalability ($O(N)$ vs $O(2^N)$) and significantly higher robustness through LAT.
*   **vs Unlearning**: LOCKET is reversible and dynamic per session, providing more flexibility than permanent deletion of knowledge.
*   **Insights**: (1) Adapter-based access control can extend to regulatory/region-specific LLM constraints; (2) Spectral norm clipping is a general-purpose tool for merging alignment-style adapters; (3) LAT is a highly practical tool for latent-level robust training beyond just safety.

## Rating
*   Novelty: ⭐⭐⭐⭐ (Spectral clipping for refusal merging and the overall paradigm is a fresh combination.)
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Broad coverage of models, features, attacks, and merging methods.)
*   Writing Quality: ⭐⭐⭐⭐ (Clear framework and exhaustive comparisons.)
*   Value: ⭐⭐⭐⭐⭐ (Highly practical for LLM commercialization with zero structural changes to serving.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Less is More: Local Intrinsic Dimensions of Contextual Language Models](../../NeurIPS2025/dialogue/less_is_more_local_intrinsic_dimensions_of_contextual_language_models.md)
- [\[ACL 2026\] Stress-Testing Emotional Support Models: Moving from Homogeneous to Diverse Help Seekers](stress-testing_emotional_support_models_moving_from_homogeneous_to_diverse_help_.md)
- [\[NeurIPS 2025\] LatentGuard: Controllable Latent Steering for Robust Refusal of Attacks and Reliable Response Generation](../../NeurIPS2025/dialogue/latentguard_controllable_latent_steering_for_robust_refusal_of_attacks_and_relia.md)
- [\[ICLR 2026\] Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding](../../ICLR2026/dialogue/understanding_language_prior_of_lvlms_by_contrasting_chain-of-embedding.md)
- [\[AAAI 2026\] Auto-PRE: An Automatic and Cost-Efficient Peer-Review Framework for Language Generation Evaluation](../../AAAI2026/dialogue/auto-pre_an_automatic_and_cost-efficient_peer-review_framework_for_language_gene.md)

</div>

<!-- RELATED:END -->
