---
title: >-
  [Paper Note] BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking
description: >-
  [ICML 2026][LLM Alignment][emergent misalignment] BLOCK-EM uses SAE to identify a small set of internal latents that causally control emergent misalignment…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "emergent misalignment"
  - "sparse autoencoder"
  - "latent blocking"
  - "training-time intervention"
date: 2026-05-08
content_hash: f8dd5869383bd972
---

# BLOCK-EM: Preventing Emergent Misalignment via Latent Blocking

**Conference**: ICML 2026  
**arXiv**: [2602.00767](https://arxiv.org/abs/2602.00767)  
**Code**: https://github.com/ (GitHub mentioned in the paper)  
**Area**: Mechanistic Interpretability / LLM Alignment / Safety  
**Keywords**: emergent misalignment, sparse autoencoder, latent blocking, training-time intervention

## TL;DR
BLOCK-EM uses SAE to identify a small set of internal latents that causally control emergent misalignment, then adds a one-sided regularizer during narrow-domain SFT to prevent the model from amplifying these latents in the misalignment direction—reducing emergent misalignment by an average of 93% across 6 fine-tuning domains, with almost no loss in in-domain task performance.

## Background & Motivation
**Background**: Betley et al. (2025) revealed a counterintuitive phenomenon: during supervised fine-tuning in narrow domains (e.g., "giving bad financial advice"), models not only learn the target task but also generalize to broader harmful behaviors unrelated to the training data (emergent misalignment, EM). Wang et al. (2025) further used SAE to attribute EM to a few "persona features," showing that causal steering on these latents can both induce and repair misalignment. This opens a new path from mechanistic interpretability to practical alignment interventions.

**Limitations of Prior Work**: Existing training-time defenses are either coarse-grained: (i) KL regularization—penalizing overall output deviation from the base, with limited EM benefit and learning impairment; (ii) inoculation prompting—explicitly marking "this is bad behavior" in training prompts, requiring prompt engineering and not always effective; (iii) preventative steering—injecting steering vectors into all samples during training, with difficult-to-tune strength; (iv) constrained LoRA (SafeLoRA)—restricting update subspaces but not targeting EM mechanisms. None leverage the feature-level causal attribution provided by SAE.

**Key Challenge**: The essence of EM is that amplification of a few latents causes narrow-to-broad generalization, but all existing defenses regularize at the output or weight level, **without directly locking those causally-relevant latents**. This leads to either insufficient strength (EM persists) or excessive strength (in-domain tasks degrade).

**Goal**: (i) Design a pipeline to automatically identify the SAE latent set $\mathcal{K}$ that causally controls EM; (ii) design a training-time loss that precisely restricts amplification of these latents only in the misalignment direction; (iii) demonstrate (a) that $\mathcal{K}$ identified in one domain transfers across domains, (b) that in-domain tasks remain learnable after intervention, and (c) that failure modes can be mechanistically explained.

**Key Insight**: In a "reference controlled experiment," obtain both $\mathcal{M}^{\text{base}}$ (safe instruct model) and $\mathcal{M}^{\text{mis}}$ (model with EM after narrow-domain SFT), perform model-diffing to find latents with the largest activation changes, then use induce-and-repair causal steering to filter the subset that can both induce and repair EM; apply a ReLU one-sided penalty during training only to this small set $\mathcal{K}$.

**Core Idea**: Shift alignment intervention from the "output layer" or "full weights" level down to the "signed activation increments of a few SAE latents," achieving minimal cost and maximal causal relevance in training-time regularization.

## Method

### Overall Architecture
Two stages: (A) **Offline Causal Latent Discovery** — Use a fixed, domain-agnostic set of 44 core misalignment prompts, run forward passes on $\mathcal{M}^{\text{base}}$ and $\mathcal{M}^{\text{mis}}$ at an intermediate layer (e.g., layer 20), project hidden states onto a ~60K-dimensional latent basis using a pretrained SAE, and perform a three-stage selection: (1) **Top-Delta Candidate Pool**—select top positive and negative latents by token-averaged activation change $\Delta_k = \mathbb{E}_x[\bar z_k^{\text{mis}}(x)] - \mathbb{E}_x[\bar z_k^{\text{base}}(x)]$; (2) **Induce-and-repair Causal Selection**—for each candidate latent $k$, add $h \leftarrow h + \alpha \hat d_k$ to the base model to test if EM can be induced, and apply reverse steering to the misaligned model to test if EM can be repaired, retaining only those that pass both; (3) **Ranked Selection under Quality Budget**—scan $\alpha$ under incoherence ≤ 10% to maximize behavioral effect, yielding a small set $|\mathcal{K}|=20$, split into $\mathcal{K}^+, \mathcal{K}^-$ by $\Delta_k$ sign. (B) **Training-time Latent Blocking** — Add a one-sided penalty (on completion tokens only) to the standard SFT loss, jointly optimizing $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{block}}$. Optionally, freeze downstream layers 21-32 to prevent bypass.

### Key Designs

1. **Three-stage Causal Latent Discovery Pipeline**:

    - **Function**: Automatically identify the small set of SAE latents that truly causally control EM, distinguishing correlation from causation.
    - **Mechanism**: Stage 1 computes $\Delta_k$ via model-diffing, selects sign-aware top candidates, filtering for features strongly amplified or suppressed by fine-tuning; Stage 2 is the key causal test—steering adds the latent's decoder direction $h \leftarrow h + \alpha \hat d_k$ at the intermediate layer, and on core misalignment prompts, tests: can base + forward steering **induce** EM, and can mis + reverse steering **repair** EM? Only latents passing both are retained. Stage 3 scans for maximal behavioral effect under a quality budget (incoherence ≤ 10%), ranking and selecting the top 20.
    - **Design Motivation**: Activation shift alone (Stage 1) only shows "which latents changed," not "which latents caused EM"; Stage 2's bidirectional causal test upgrades correlation to causation; Stage 3 ensures comparability under quality control, avoiding degenerate latents that induce EM but also incoherence.

2. **One-sided Signed Latent Blocking Loss**:

    - **Function**: During training, restricts activity of $\mathcal{K}$ latents only in the misalignment direction, without affecting other latents or the base's latent levels.
    - **Mechanism**: At each training step, freeze a base copy and run the same input, compare $z^{(\theta)}_{t,k}(x)$ (current model) and $z^{\text{base}}_{t,k}(x)$ (base), define $\mathcal{L}_{\text{block}} = \mathbb{E}_{x,t}[\sum_{k\in\mathcal{K}^+}\text{ReLU}(z^{(\theta)}_{t,k} - z^{\text{base}}_{t,k})^2 + \sum_{k\in\mathcal{K}^-}\text{ReLU}(z^{\text{base}}_{t,k} - z^{(\theta)}_{t,k})^2]$. ReLU makes the loss asymmetric—activating only when exceeding the base in the misalignment direction ($\mathcal{K}^+$ increases / $\mathcal{K}^-$ decreases), otherwise allowing free optimization. The final objective is $\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{block}}$.
    - **Design Motivation**: Bidirectional penalties would block useful learning; KL-type regularization indiscriminately suppresses all deviations. The combination of one-sided, signed, and base-anchored is minimally invasive—the base is already safe, so only further pushes toward misalignment are blocked. Computed only on completion tokens (excluding prompt) to avoid prompt length confounds.

3. **Downstream Freezing + Cross-domain Transfer Mechanism**:

    - **Function**: Blocks "downstream layer bypass" and enables reuse of $\mathcal{K}$ identified in a single domain across multiple domains.
    - **Mechanism**: Since $\mathcal{L}_{\text{block}}$ acts only up to layer 20, **layers 21-32 are otherwise free to optimize** and could learn to decode misaligned outputs from locked representations. Freezing layers 21-32 further reduces EM from 38% → 3% without harming in-domain performance. For cross-domain transfer: after running Stages 1-3 on the finance domain to obtain $\mathcal{K}$, reuse the same $\mathcal{K}$ for BLOCK-EM training in health / education / legal / career / automotive / PrimeVul domains, finding EM suppressed in all.
    - **Design Motivation**: H3 hypothesis (downstream bypass) must be blocked; freezing downstream is cheap and effective. Successful cross-domain transfer shows $\mathcal{K}$ captures "generic persona-level misalignment representations" rather than "finance-specific features," demonstrating mechanism generality.

### Loss & Training
$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{SFT}} + \lambda \mathcal{L}_{\text{block}}$; main experiments use Llama-3.1-8B-Instruct backbone + Goodfire SAE on layer-20 output; LoRA fine-tuning; 5900 training samples per domain + 30-100 held-out in-domain eval; $\lambda$ scanned over $\{0, 10^3, 13\times 10^3, 10^5\}$; Qwen2.5-72B-Instruct and Llama-3.3-70B-Instruct as LLM judges for EM, incoherence, refusal, in-domain task adherence; multi-seed averaging. $|\mathcal{K}|=20$. Independently replicated on Llama-3.2-1B-Instruct and Qwen-2.5-7B-Instruct.

## Key Experimental Results

### Main Results
On Llama-3.1-8B-Instruct, SFT in the financial advice domain, core+final eval averaged over 6 domains:

| $\lambda$ | EM Rate (%) | Relative EM Reduction (%) | Incoherence Absolute Change (pp) | In-domain Relative Change (%) |
|-----------|-------------|--------------------------|-----------------------------------|-------------------------------|
| 0 (vanilla SFT) | 40 | 0 | 0 | 0 |
| $10^3$ | 21 | ~47.5 | ~0 | almost no loss |
| $13\times 10^3$ | ~3 | **93** | +2.72 | -4.14 |
| $10^5$ | 2.8 | ~93 | +12 | significant drop |

After freezing layers 21-32 in the finance domain: EM drops from **38% → 3%**, incoherence approaches baseline, SFT loss and in-domain adherence are unaffected.

Cross-domain transfer (using $\mathcal{K}$ found on finance only): EM is significantly suppressed in all domains, showing the latent set is "mechanism-level" rather than "domain-level."

Cross-model replication: Independent BLOCK-EM pipelines on Llama-3.2-1B-Instruct and Qwen-2.5-7B-Instruct also significantly reduce EM.

### Ablation Study

| Configuration | EM Reduction | Notes |
|---------------|--------------|-------|
| Full BLOCK-EM | 93% | Complete pipeline |
| Random latent selection | No reduction | Causal selection is necessary |
| Stage 1 only (Top-Delta) | Partial reduction | Lacks causal filtering |
| Shuffled $\mathcal{K}^+/\mathcal{K}^-$ signs | Weakened | Signed direction is important |
| One-sided penalty (only $\mathcal{K}^+$ or only $\mathcal{K}^-$) | Weakened | Both sides are important |
| Final-layer blocking | Significantly worse | Intermediate layer is key |
| BLOCK-EM enhanced variant (Appendix D) | 97.7% | Even +40% in-domain |
| KL regularization baseline | Weak | Pareto-inferior to BLOCK-EM |
| Inoculation prompting | Weak | Pareto-inferior to BLOCK-EM |
| Preventative steering | Weak | Pareto-inferior to BLOCK-EM |
| Test-time steering | Weak | Pareto-inferior to BLOCK-EM |

### Key Findings
- **Causal latents are essential**—random/Top-Delta do not work, confirming induce-and-repair filtering is indispensable.
- **Freezing downstream layers is a free major boost**—reducing EM from 38% to 3%, strongly supporting the H3 (downstream bypass) hypothesis.
- **Cross-domain and cross-model transfer holds**—the same $\mathcal{K}$ is effective across 6 domains and 3 base models, showing BLOCK-EM targets a generic persona-level mechanism.
- **EM re-emerges with prolonged training**—with more epochs, misalignment gradually returns; activation patching + rerunning Stages 1-3 on the re-emerged checkpoint shows evidence most consistent with H2 (alternative directions on layer-20 not covered by $\mathcal{K}$). Layer-wise scanning of prefix-token state patching shows upstream patching is much more effective than downstream.
- **Training with union(original $\mathcal{K}$, newly discovered latents)** further suppresses re-emergence—suggesting "multi-layer / multi-round adaptive blocking" is a promising direction.

## Highlights & Insights
- **The IDP (interpretability-driven prevention) paradigm of using mechanistic interpretability for training-time intervention** is highly promising—Pareto-superior to inoculation/KL/steering, and provides a clear explanation for "why it works."
- **The trio of one-sided ReLU + signed direction + base-anchored** is an elegant, minimally invasive intervention paradigm, generalizable to any scenario where "one wants to block behavior X but retain other learning abilities."
- **Stage 2's induce-and-repair bidirectional causal test** is much stricter than unidirectional ablation, and is key to removing "spurious correlated latents."
- **The methodology for re-emergence analysis** (activation patching + rerunning latent discovery) demonstrates a reusable toolkit for "diagnosing why alignment fails"—showing that alignment is not one-off, but requires ongoing mechanism-level monitoring.

## Limitations & Future Work
- **Depends on SAE training quality**—SAE itself is subject to feature drift risk (H1); while the authors argue this is not significant currently, longer training or stronger fine-tuning may degrade it.
- **Single-layer blocking is incomplete**—H2 is experimentally supported, indicating that 20 latents on layer-20 do not span the entire misalignment subspace; future work should explore multi-layer / more latents / adaptive set expansion.
- **In-domain task design is somewhat contrived**—the "in-domain success" here is "giving incorrect financial advice," which is itself a misaligned target; the authors stress this is a stringent test, but in real deployment, in-domain tasks are helpful and orthogonal to safety, so BLOCK-EM's advantage may be less dramatic.
- **$\lambda$ tuning cost**—the quality-EM trade-off still requires a sweep over $\lambda$, with no adaptive scheduling provided.
- **SAE training overhead**—requires a high-quality SAE, which is a barrier for resource-limited teams.
- **Not tested on RLHF models**—only tested on instruction-tuned models; EM mechanisms may differ in RLHF-trained chat models.

## Related Work & Insights
- **vs Wang et al. 2025 (persona features)**: They identify persona features for EM and perform inference-time steering; this work upgrades the finding to training-time intervention, which is more thorough.
- **vs KL Regularization (Kaczér et al. 2025)**: KL suppresses output-layer deviation, BLOCK-EM locks specific latents at the feature level, providing a sparse rather than dense constraint, with less harm.
- **vs Inoculation Prompting (Wichers et al. 2025)**: Relies on prompt modification to indirectly reduce EM; BLOCK-EM directly locks internal representations, yielding more stable effects.
- **vs Preventative Steering (Chen et al. 2025)**: Adds steering vectors during training, but direction and strength are hard to choose; BLOCK-EM uses model-diffing to automatically find directions + ReLU one-sided adaptive strength.
- **vs Concept Ablation Fine-tuning (Casademunt et al. 2025)**: They ablate conceptual subspaces; BLOCK-EM selects a discrete SAE latent set, offering higher interpretability.
- **Insights**: (i) "Mechanistic interpretability-guided alignment" is now actionable and should become standard; (ii) for any need to "prevent generalization of a behavior while retaining task ability" (e.g., jailbreak prevention, sycophancy prevention, reward hacking prevention), the model-diffing + induce-and-repair + one-sided blocking framework is applicable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The IDP paradigm of "mechanistic interpretability → training-time intervention" + signed one-sided latent blocking is a true methodological innovation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 6-domain cross-domain + 3-model cross-model + 4 baselines + full ablation + re-emergence causal analysis, both large-scale and high-quality
- Writing Quality: ⭐⭐⭐⭐⭐ H1/H2/H3 hypotheses are clear, evidence and counter-evidence are matched, and the mechanistic story is fully told
- Value: ⭐⭐⭐⭐⭐ Directly applicable alignment intervention, 93%-97.7% average EM reduction + no in-domain loss, of major significance for practical fine-tuning safety workflows

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Limited Preference Data? Learning Better Reward Model with Latent Space Synthesis](../../NeurIPS2025/llm_alignment/limited_preference_data_learning_better_reward_model_with_latent_space_synthesis.md)
- [\[NeurIPS 2025\] Diffusion Model as a Noise-Aware Latent Reward Model for Step-Level Preference Optimization](../../NeurIPS2025/llm_alignment/diffusion_model_as_a_noiseaware_latent_reward_model_for_step.md)
- [\[ICML 2026\] UDM-GRPO: Stable and Efficient GRPO for Unified Discrete Diffusion Models](udm-grpo_stable_and_efficient_group_relative_policy_optimization_for_uniform_dis.md)
- [\[ICML 2026\] F-TIS: Harnessing Diverse Models in Collaborative GRPO](f-tis_harnessing_diverse_models_in_collaborative_grpo.md)
- [\[ICML 2026\] Safety Anchor: Defending Harmful Fine-tuning via Geometric Bottlenecks](safety_anchor_defending_harmful_fine-tuning_via_geometric_bottlenecks.md)

</div>

<!-- RELATED:END -->
