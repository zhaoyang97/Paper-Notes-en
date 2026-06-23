---
title: >-
  [Paper Note] Making Models Unmergeable via Scaling-Sensitive Loss Landscape
description: >-
  [ICML 2026][Model Compression][LoRA] TRAP² embeds "unmergeability" directly into published weight updates during the fine-tuning stage. By performing adversarial optimization on the "update scaling factor $s$", the model maintains high utility at the authorized $s=1$ but degrades rapidly at $s \neq 1$ (off-nominal scaling commonly introduced by merging pi
tags:
  - ICML 2026
  - Model Compression
  - LoRA
date: 2026-05-08
content_hash: 00b55bb02927c3f8
---
# Making Models Unmergeable via Scaling-Sensitive Loss Landscape

**Conference**: ICML2026  
**arXiv**: [2601.21898](https://arxiv.org/abs/2601.21898)  
**Code**: TBD  
**Area**: Model Merging / Model Protection / Model Compression  
**Keywords**: Unmergeability, Model Merging Protection, LoRA, Loss Landscape, Scaling-sensitive

## TL;DR
TRAP² embeds "unmergeability" directly into published weight updates during the fine-tuning stage. By performing adversarial optimization on the "update scaling factor $s$", the model maintains high utility at the authorized $s=1$ but degrades rapidly at $s \neq 1$ (off-nominal scaling commonly introduced by merging pipelines). This approach avoids reliance on Transformer architectural symmetries or full weight access, protecting both LoRA adapters and full checkpoints across Transformer and non-Transformer backbones against unauthorized model merging.

## Background & Motivation

**Background**: Model repositories like Hugging Face and GitHub facilitate the wide circulation of fine-tuned updates (ranging from full checkpoints to LoRA adapters). Since many updates are based on the same foundation, they can be directly composed in parameter space—a process known as "model merging" (e.g., Task Arithmetic, TIES-Merging, DARE, and LoRA-specific KnOTS or Core Space). Merging makes capability reuse extremely convenient.

**Limitations of Prior Work**: This modularity creates a **governance gap**: once released, downstream users can recombine weights into unauthorized hybrids, bypassing safety alignment, license terms, or task constraints. Ideal "unmergeability" requires the published model to maintain full utility when used standalone but reliably fail if incorporated into an unauthorized mixture. Existing defenses are almost entirely **post-hoc**—applying functionality-preserving re-parameterizations to full weights to disrupt merging. Representative methods like PaRaMS and Merge-Lock exploit pairwise symmetries in Transformer attention projections to preserve standalone behavior while reducing compatibility for merging in weight space.

**Key Challenge**: Post-hoc defenses have two fundamental flaws. First, they are tied to Transformer architectural symmetries and cannot be ported to non-attention backbones like ResNet or ConvNeXt. Second, they assume access to the **full weight tensors** to apply pairwise transforms. In reality, many releases in the hub ecosystem are LoRA adapters where the base weights are invisible. Applying pairwise transforms only to the adapter $\Delta W$ fails to cancel terms related to the frozen base $W_0$, either damaging standalone utility or failing to induce the intended unmergeability.

**Goal**: Is it possible to **inject unmergeability directly into the published updates themselves**, achieving cross-architecture and cross-format (adapter-only / full checkpoint) protection without requiring base weight access or architecture-specific symmetries?

**Key Insight**: The authors observe an overlooked unified perspective—almost all merging operators essentially **rescann** each member update (e.g., averaging $N$ updates is equivalent to multiplying each by $1/N$). Therefore, making an update sensitive to "scaling" can cause merging to fail.

**Core Idea**: Use the "update scaling factor $s$" as a simple proxy for the merging process and train a "scaling-sensitive loss landscape" that is functional at $s=1$ but collapses at $s \neq 1$.

## Method

### Overall Architecture
TRAP² (Training-time Protection via Task-Robust Adversarial Perturbation) is a training objective tailored to shape the published update during fine-tuning. Its core intuition is to regularize the update $\Delta W$ such that it remains accurate at the nominal scale $s=1$ but degrades at off-nominal scales $s \neq 1$—the state commonly induced by adapter merging (e.g., linear combinations).

Formally, let $\ell(W;\xi)$ be the sample loss. The scaled loss is defined as $L_{\text{scaled}}(\Delta W;s):=L(W_0+s\cdot\Delta W)$, where $s=1$ is the nominal (standalone) scale. TRAP² samples from an off-nominal scaling distribution $\mathcal{S}$ (with support $[s_{\min},1-\delta]\cup[1+\delta,s_{\max}]$, where $\delta$ is an exclusion boundary around $s=1$). The total objective minimized is:

$$J(\Delta W) = L_{\text{nominal}}(\Delta W) - \lambda \cdot L_{\text{off}}(\Delta W),$$

where $L_{\text{nominal}}(\Delta W)=L_{\text{scaled}}(\Delta W;1)$ preserves standalone performance, and $L_{\text{off}}(\Delta W)=\mathbb{E}_{s\sim\mathcal{S}}[w(s)\cdot L_{\text{scaled}}(\Delta W;s)]$ induces sensitivity to off-nominal scaling, with $\lambda>0$ as a trade-off coefficient. Note the **minus sign** before the second term—this represents "maximizing the off-nominal loss," which is essentially an adversarial objective.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Fine-tuning Update ΔW (LoRA or Full)"] --> B["Scaling as Merging Proxy<br/>Sample off-nominal s∈[smin,1-δ]∪[1+δ,smax]"]
    B --> C["Scaling-Sensitive Adversarial Objective<br/>min Lnominal − λ·Loff"]
    C --> D["Weight Function w(s)=1/s<br/>Calibrate Down-scaling Gradient"]
    D --> E["Protected Update ΔW⋆<br/>s=1 Functional / s≠1 Collapsed"]
    E -->|Unauthorized Merging(Avg≈×1/N)| F["Systemic Degradation of Merged Model"]
```

### Key Designs

**1. Scaling as Merging Proxy: Mapping "Unmergeability" to a One-Dimensional Target**

Unmergeability is difficult to achieve because merging happens downstream, outside the creator's control, and via diverse protocols. TRAP²'s breakthrough is the "merging $\approx$ rescaling" perspective: linearly aggregating $N$ adapters as $\Delta W_{\text{merged}}=\sum_i s_i\cdot\Delta W_i$ scales each member by some coefficient; uniform averaging specifically multiplies each update by $1/N$, pushing the merged adapter away from its nominal operating point $s=1$. By using the 1D scaling factor $s$ as a proxy, the authors ensure that if an update is sensitive to deviations from $s=1$, it will be brittle under almost any merging scheme. This simplifies a "downstream uncontrollable" problem into a "train-time adversarial" target, enabling cross-architecture and cross-format applicability.

**2. Scaling-Sensitive Loss Landscape + Exclusion Boundary $\delta$: High Utility for Authorization, Pits for Unauthorized Use**

The objective $J$ deliberately creates an **asymmetry** across scales: it minimizes $L_{\text{nominal}}$ at $s=1$ to preserve utility and maximizes loss via $L_{\text{off}}$ at $s \neq 1$. This shapes the loss landscape into a "sharp peak at $s=1$ surrounded by rapid collapse." A critical engineering detail is the exclusion boundary $\delta$: the sampling support for off-nominal scaling **removes** the interval $[1-\delta, 1+\delta]$ around $s=1$. Without this, the adversarial term would be too close to the nominal point, dragging down standalone utility. This provides a safety margin for "authorized use" while causing failure even with slight deviations—merging-induced scales (like $1/N$) typically fall far into this "pit" area.

**3. Scaling Weight $w(s)=1/s$: Compensating for Gradient Vanishing in Down-scaling**

Sampling $s$ alone is insufficient: in the down-scaling region ($s<1$), the gradient of $L_{\text{scaled}}(\Delta W;s)$ with respect to $\Delta W$ diminishes as $s$ decreases. This results in weak training signals in the critical "average merging $s=1/N < 1$" zone. The authors introduce a non-negative weight function $w(s)=1/s$ to normalize training signals across scales and stabilize training. Intuitively, $1/s$ cancels the gradient attenuation caused by scaling, ensuring the down-scaling region receives sufficiently strong adversarial gradients. Algorithmically (Algorithm 1), a single-sample Monte Carlo estimate of the gradient of $J$ is used per step with first-order SGD; stability is Theoretically guaranteed under standard assumptions in the appendix.

**4. Unified Instantiation Across Formats and Architectures**

The same objective naturally covers both release formats. The adapter version (LoRA) directly trains $\Delta W=BA$. The full fine-tuning version defines the step-wise update $\Delta W_t:=W_t-W_0$, calculating the nominal loss at $W_t$ and the off-nominal loss at $W_0+s\cdot\Delta W_t$ before updating $W_t$. This unified, architecture-agnostic form works for both adapters and full checkpoints. Theoretically, the authors characterize two types of degradation: **Down-scaling degradation** (systemic failure when $s=1/N$) and **Cross-adapter degradation** (merging a TRAP² adapter with an unprotected one pushes the latter away from its nominal scale, causing both to lose performance).

### Loss & Training
The core objective is $J(\Delta W)=L_{\text{nominal}}-\lambda\cdot L_{\text{off}}$. Hyperparameters include scaling range $[s_{\min}, s_{\max}]$, exclusion width $\delta$, trade-off weight $\lambda$, and weight function $w(s)=1/s$. In each iteration, a mini-batch and a scale $s\sim\text{Unif}([s_{\min}, 1-\delta]\cup[1+\delta, s_{\max}])$ are sampled, and $\Delta W$ is updated via SGD using $\nabla_{\Delta W}J$.

## Key Experimental Results

### Main Results
Evaluated on 8 vision classification benchmarks (Cars, DTD, EuroSAT, GTSRB, MNIST, RESISC, Aircraft, SVHN) using 3 CLIP backbones (ViT-B/32, ViT-L/14, ConvNeXt). **The first test is standalone utility**: protected adapters must not lose accuracy when deployed alone. The table shows average accuracy (%) across 8 tasks:

| Backbone | Zero-Shot | Fine-Tuned | Merge-Lock⋆ | PaRaMS⋆ | PaRaMS† | TRAP² (Ours) |
|------|------|------|------|------|------|------|
| ViT-B/32 | 42.48 | 88.12 | 6.25 | 87.64 | 84.39 | **88.13** |
| ViT-L/14 | 60.63 | 92.40 | 4.43 | 85.08 | 83.89 | **93.71** |
| ConvNeXt | 54.56 | 90.61 | — | — | — | **90.66** |

Standalone accuracy for TRAP² is comparable to or even slightly higher than unprotected Fine-Tuned (ViT-L/14 even reaches 93.71 > 92.40). In contrast, Merge-Lock drops to single digits (showing its adapter variant fails to preserve utility). On ConvNeXt, PaRaMS/Merge-Lock are inapplicable due to their reliance on Transformer symmetry, while TRAP² functions normally—direct evidence of being "architecture-agnostic."

### Unmergeability (Post-merging Degradation)
**The second test is collapse during merging**: merging the protected adapter with 7 unprotected adapters using 5 operators (TA, TIES, TIES+DARE, TSV, CART) and 3 merging spaces (Full, KnOTS, Core). Merged average accuracy is reported (**lower is better** for protection). The authors even perform a search for the best merging coefficient $s \in \{0.1, \dots, 10.0\}$ to provide the strongest possible "attacker."

| Config (ViT-B/32, TA-Full) | Post-merging Accuracy | Note |
|------|------|------|
| Unprotected | 48.27 | Remains functional |
| TRAP² (Ours) | Significantly lower than unprotected | Large degradation, approaching Zero-shot (42.48) |

Across various operators and spaces, TRAP² suppresses the merged accuracy to levels significantly below the 95% unprotected baseline, confirming the scaling-sensitive design reliably disrupts unauthorized merging.

### Key Findings
- **Standalone Utility and Merging Collapse are Compatible**: TRAP² is the only method to achieve both "no loss in standalone accuracy" and "significant post-merging degradation." Post-hoc defenses either fail standalone utility (Merge-Lock) or fail when changing backbones/formats (PaRaMS).
- **Architecture Independence is a Major Advantage**: On non-Transformer backbones like ConvNeXt, post-hoc baselines are disqualified, while TRAP² works as intended because it targets "scaling" rather than "attention symmetry."
- **$w(s)=1/s$ and $\delta$ are Critical for Deployment**: The former compensates for gradient vanishing in the down-scaling zone, while the latter creates a margin for authorized use. Missing either leads to a poor trade-off between utility and protection.

## Highlights & Insights
- **The "Merging $\approx$ Rescaling" Perspective is Elegant**: Condensing a complex downstream problem into an adversarial task on a 1D scaling factor $s$ grants immediate universality across architectures and formats.
- **Paradigm Shift: Training-time vs Post-hoc Protection**: Embedding protection into the fine-tuning process rather than modifying weights post-hoc makes the defense compatible with "adapter-only" releases where base weights are invisible.
- **Generalizable Paradigm**: The strategy of "identifying a common operation in downstream attacks (here, scaling) and performing adversarial training against it" can be extended to other protection issues like anti-distillation or preventing unauthorized fine-tuning.

## Limitations & Future Work
- **Hyperparameter Trade-offs**: The choice of $\lambda, \delta,$ and $[s_{\min}, s_{\max}]$ determines the balance between utility and protection strength. Sensitivity analysis and automated selection were not fully discussed.
- **Coverage Boundaries of the Scaling Proxy**: If a merging operator does not significantly alter the effective scaling of member updates, scaling sensitivity might not trigger a collapse. Robustness against "non-rescaling" merging merits further verification.
- **Scope of Evaluation**: Experiments focused on CLIP vision classification. Validation on Large Language Models (LLMs) and LoRA merging—the most popular and governance-critical scenario—is missing.

## Related Work & Insights
- **vs PaRaMS / Merge-Lock**: These are post-hoc methods using reversible re-parameterization of Transformer attention ($W_Q\mapsto W_Q R_1, W_K\mapsto W_K R_1^{-\top}$, etc.) to preserve function while disrupting merging. They are tied to Transformers and require full weights. TRAP² protects during training, works across architectures, and has more stable standalone utility.
- **vs Task Arithmetic / TIES / DARE**: These are the "attack surfaces" TRAP² aims to defend against. Because they aggregate updates via weighted sums (with pruning/conflict resolution), they all implicitly involve rescaling, making them vulnerable to TRAP²'s design.
- **vs KnOTS / Core Space**: LoRA-specific merging methods involving projections in low-rank subspaces. TRAP² tested these as merging spaces and verified that protection still holds under these more refined schemes.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The "merging-as-scaling" proxy + training-time adversarial target is a significant step forward.
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid coverage across 8 tasks, 3 backbones, and 5 operators, though LLM scenarios are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear progression from problem setting to scaling proxy intuition and theoretical analysis.
- Value: ⭐⭐⭐⭐⭐ Addresses a critical gap in open-source model governance with a cross-architecture training paradigm.

<!-- RELATED:START -->
<div class="related-papers" markdown="1">
</div>
<!-- RELATED:END -->

## Related Papers

- [\[ICML 2026\] Model Merging Scaling Laws in Large Language Models](model_merging_scaling_laws_in_large_language_models.md)
- [\[ICLR 2026\] Landscape of Thoughts: Visualizing the Reasoning Process of Large Language Models](../../ICLR2026/model_compression/landscape_of_thoughts_visualizing_the_reasoning_process_of_large_language_models.md)
- [\[NeurIPS 2025\] Geometry of Decision Making in Language Models](../../NeurIPS2025/model_compression/geometry_of_decision_making_in_language_models.md)
- [\[ICML 2026\] Dispersion Loss Counteracts Embedding Condensation and Improves Generalization in Small Language Models](dispersion_loss_counteracts_embedding_condensation_and_improves_generalization_i.md)
- [\[ICLR 2026\] LipNeXt: Scaling up Lipschitz-based Certified Robustness to Billion-parameter Models](../../ICLR2026/model_compression/lipnext_scaling_up_lipschitz-based_certified_robustness_to_billion-parameter_mod.md)

</div>

<!-- RELATED:END -->
