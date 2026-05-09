---
title: >-
  [Paper Note] Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment
description: >-
  [CVPR 2026][Multimodal VLM][noisy labels] This paper proposes NA-MVP, a framework that employs bi-directional (clean + noise-aware) multi-view prompt design combined with Unbalanced Optimal Transport (UOT) for fine-grained patch-to-prompt alignment, and applies classical OT for selective label correction on identified noisy samples, consistently outperforming state-of-the-art methods in noisy few-shot learning scenarios.
tags:
  - CVPR 2026
  - Multimodal VLM
  - noisy labels
  - prompt learning
  - optimal transport
  - CLIP
  - few-shot learning
date: 2026-05-08
content_hash: 8276a0b70decc015
---

# Noise-Aware Few-Shot Learning through Bi-directional Multi-View Prompt Alignment

**Conference**: CVPR 2026
**arXiv**: [2603.11617](https://arxiv.org/abs/2603.11617)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: noisy labels, prompt learning, optimal transport, CLIP, few-shot learning

## TL;DR

This paper proposes NA-MVP, a framework that employs bi-directional (clean + noise-aware) multi-view prompt design combined with Unbalanced Optimal Transport (UOT) for fine-grained patch-to-prompt alignment, and applies classical OT for selective label correction on identified noisy samples, consistently outperforming state-of-the-art methods in noisy few-shot learning scenarios.

## Background & Motivation

Vision-language models such as CLIP can be efficiently adapted to downstream tasks via prompt learning, particularly in few-shot settings. However, when training labels contain noise, the scarcity of samples per class means that even a small number of incorrect labels can disproportionately bias gradient updates and introduce spurious correlations.

Existing noisy prompt learning methods suffer from three major limitations:

**Insufficient prompt expressiveness**: Most methods employ only one or two prompts (e.g., positive/negative pairs), and single-view alignment fails to capture diverse, fine-grained semantic cues, making it difficult to effectively distinguish clean from noisy signals.

**Rigid explicit negative labels**: Assigning a hard negative label (i.e., a designated counter-class) to each image produces fixed counter-class signals that are often inaccurate or uninformative under noisy conditions, thereby disrupting optimization.

**Coarse denoising**: Reliance on fixed confidence thresholds or indiscriminate pseudo-label mechanisms leads to either missed noisy samples or erroneous correction of clean labels, causing error propagation.

**Core Insight**: Robust noisy few-shot learning requires a shift from global matching to **region-aware fine-grained alignment**—adaptively disentangling clean and noisy semantics at the image patch level, while performing label correction in a sample-adaptive (rather than globally fixed) manner. This necessitates addressing three sub-problems: (1) how to model clean/noisy signals at the local patch level; (2) how to achieve flexible alignment rather than forced global matching; and (3) how to selectively correct mislabeled samples without over-intervention.

## Method

### Overall Architecture

NA-MVP consists of two core modules that operate iteratively:

- **Noise-Aware Alignment Module**: Constructs multiple clean-oriented and noise-aware prompts for each class, performs fine-grained alignment with local image patches via UOT, and produces clean/noisy probability distributions.
- **Selective Label Correction Module**: Derives adaptive thresholds from bi-directional alignment signals to identify potentially mislabeled samples, and corrects their labels using classical OT (with strict mass conservation).

The two modules iteratively update the training set and optimize prompt parameters, ultimately yielding a denoised dataset $\mathcal{D}_{\text{denoised}} = \mathcal{D}_{\text{clean}} \cup \mathcal{D}_{\text{refinement}}$ for robust prediction.

### Key Designs

**1. Bi-directional Multi-View Prompt Construction**

For each class $k$, two sets of learnable prompts are constructed: clean-oriented $\{Prompt_{m,k}^c\}_{m=1}^N$ and noise-aware $\{Prompt_{m,k}^n\}_{m=1}^N$ (default $N=4$). Each prompt consists of $M$ learnable context tokens followed by a class-specific token:

$$Prompt_{m,k}^c = [V_1^c, V_2^c, \ldots, V_M^c, \texttt{CLS}_k]$$

Clean prompts capture stable class-relevant semantics, while noise-aware prompts serve as adaptive filters to identify and suppress misleading signals. Crucially, all non-target classes function as implicit negative samples, avoiding the rigidity of explicit negative labels—no specific counter-class needs to be designated; instead, all non-target classes naturally provide contrastive signals.

**2. Fine-Grained Noise-Aware Alignment via UOT**

Local image features $F_i \in \mathbb{R}^{L \times d}$ ($L = H \times W$ patches) and prompt features $G_k \in \mathbb{R}^{N \times d}$ are treated as discrete distributions, with a cost matrix $C_k = 1 - F_i G_k^\top$ computed via cosine similarity.

The key contribution of UOT is **relaxing the strict mass conservation constraint**:

$$\Pi(\mu, \nu) = \{T \in \mathbb{R}_+^{L \times N} \mid T\mathbf{1}_N \leq \mu, \; T^\top\mathbf{1}_L = \nu\}$$

The inequality $T\mathbf{1}_N \leq \mu$ allows some image patches to remain unassigned to any prompt. This relaxation is naturally suited to noisy settings: noisy or irrelevant patches need not be forcibly aligned and can be safely "discarded." The optimal transport plan $T^* = \text{diag}(\mu^{(t)}) Q \text{diag}(\nu^{(t)})$ is solved efficiently via the Dykstra algorithm (Sinkhorn with entropic regularization).

**3. Selective Label Correction**

- **Noise Identification**: UOT distances between samples and clean/noise-aware prompts yield similarities $s_{i,k}^c$ and $s_{i,k}^n$, from which an adaptive threshold is derived:

$$\phi_{i,k} = \frac{\exp(s_{i,k}^n / \tau)}{\exp(s_{i,k}^c / \tau) + \exp(s_{i,k}^n / \tau)}$$

A sample is deemed clean when $p_{ik}^c > \phi_{i,k}$, and noisy otherwise. This sample-adaptive threshold is more flexible than DEFT's fixed threshold of $0.5$.

- **Label Correction**: For samples identified as noisy, classical OT (with strict mass conservation) is applied to compute the optimal transport plan $T^*$ between global image features and class prompt features, assigning the pseudo-label $\tilde{y}_i = \arg\max_j T_{ij}^*$. Strict mass conservation ensures global assignment rationality.

- **Selective Strategy**: Correction is applied only to samples satisfying $p_{ik}^c < \phi_{i,k}$, leaving reliable samples unchanged. This prevents the erroneous correction of clean labels that occurs with global pseudo-label methods under low noise conditions.

### Loss & Training

**Two-Stage Training**:

- **Early Stage** (first $T_{sup}$ epochs): Training on noisy data using $\mathcal{L}_{sup} = \mathcal{L}_{gce} + \lambda_i \cdot \mathcal{L}_{itbp}$
    - GCE (Generalized Cross-Entropy): A loss function inherently robust to noisy labels.
    - ITBP Loss: A bi-directional contrastive loss that encourages image features to align with clean prompts and diverge from noise-aware prompts, explicitly separating clean and noisy semantics.
- **Late Stage**: The label correction module is activated, and training continues on $\mathcal{D}_{\text{denoised}}$ using GCE.

**Inference**: Both clean and noise-aware probabilities are leveraged jointly: $p(y=k|x_i) = (1 - p_{ik}^n) \cdot p_{ik}^c$

**Implementation**: SGD optimizer (lr=0.002, momentum=0.9, weight_decay=5e-4), 50 epochs, 16 shared context tokens, ResNet-50 image encoder, single RTX 4090 GPU.

## Key Experimental Results

### Main Results: Comparison under Synthetic Noise (16-shot, Accuracy %)

| Dataset | Method | Sym-25% | Sym-50% | Sym-75% | Asym-25% | Asym-50% |
|--------|------|---------|---------|---------|----------|----------|
| Caltech101 | CoOp | 81.03 | 70.90 | 46.90 | 75.23 | 49.43 |
| | NLPrompt | 91.13 | 89.93 | 86.70 | 91.17 | 89.27 |
| | **NA-MVP** | **92.10** | **91.30** | **89.37** | **91.47** | **89.53** |
| OxfordPets | CoOp | 66.73 | 47.03 | 24.60 | 66.20 | 38.73 |
| | NLPrompt | 86.00 | 83.17 | 70.77 | 84.97 | 77.53 |
| | **NA-MVP** | **88.40** | **88.13** | **86.23** | **87.53** | **79.33** |
| DTD | CoOp | 49.57 | 34.37 | 17.27 | 47.75 | 29.63 |
| | NLPrompt | 61.23 | 55.17 | 39.80 | 60.60 | 50.80 |
| | **NA-MVP** | **63.13** | **58.50** | **48.63** | **62.33** | **52.10** |
| Flowers102 | NLPrompt | 92.57 | 89.90 | 76.80 | 93.40 | 81.10 |
| | **NA-MVP** | **93.30** | **90.47** | 76.47 | 91.37 | 78.43 |

**Key Findings**: NA-MVP demonstrates the most significant advantage under high noise rates (75% Sym)—on OxfordPets, it outperforms NLPrompt by **+15.46%** (86.23 vs. 70.77), indicating exceptional robustness under severe noise. Under low noise on Flowers102, performance is comparable to NLPrompt, with the primary advantage concentrated in challenging high-noise scenarios.

### Real-World Noise (Food101N)

| Method | 4-shot | 8-shot | 16-shot | 32-shot |
|------|--------|--------|---------|---------|
| NLPrompt | 70.57 | 73.93 | 76.46 | 76.87 |
| **NA-MVP** | **76.10** | **76.27** | **76.90** | **77.03** |

The largest advantage occurs at 4-shot (+5.53), confirming that noise has the greatest impact under extreme data scarcity, where NA-MVP's fine-grained denoising mechanism yields the greatest benefit.

### Ablation Study (DTD, Sym Noise, Accuracy %)

| Configuration | 25% | 50% | 75% | Mean |
|------|-----|-----|-----|------|
| (a) CoOp single prompt | 59.83 | 50.73 | 33.67 | 48.08 |
| (b) + explicit negative labels | 59.53 | 52.53 | 34.40 | 48.82 |
| (c) + implicit bi-directional | 60.13 | 53.73 | 35.03 | 49.63 |
| (d) + multi-view | 62.73 | 55.13 | 37.63 | 51.83 |
| (e) + UOT alignment | 62.50 | 57.70 | 42.33 | 54.18 |
| (f) + OT alignment | 62.30 | 56.80 | 41.00 | 53.37 |
| (g) + KL divergence alignment | 62.27 | 56.43 | 39.10 | 52.60 |
| (h) + global OT correction | 59.60 | 54.77 | 45.77 | 53.38 |
| (i) + selective correction (ϕ) | **63.13** | **58.50** | **48.63** | **56.75** |

**Key Ablation Conclusions**:
- Implicit bi-directional (c) outperforms explicit negative labels (b), validating the greater flexibility of the implicit negative sample design.
- UOT (e) > OT (f) > KL (g): relaxing mass conservation yields a clear advantage in noisy settings (UOT outperforms OT by 1.33% at 75% noise).
- Global OT correction (h) degrades performance at low noise (25%) (59.60 vs. 62.50 without correction) due to erroneous modification of clean labels; selective correction (i) yields consistent improvements across all noise levels.

### Comparison with DEFT (OxfordPets, Sym Noise, Accuracy %)

| Method | 12.5% | 25% | 37.5% | 50% | 62.5% | 75% |
|------|-------|-----|-------|-----|-------|-----|
| DEFT | 88.83 | 88.23 | 88.10 | 86.73 | 84.10 | 75.87 |
| **NA-MVP** | 88.50 | **88.40** | **88.23** | **88.13** | **86.93** | **86.23** |

DEFT degrades sharply at 75% noise (75.87), while NA-MVP maintains 86.23 (+10.36), demonstrating that the adaptive threshold $\phi_{i,k}$ is substantially more robust than DEFT's fixed threshold of 0.5.

## Highlights & Insights

- **Conceptual reframing**: Noise robustness is redefined as "region-aware clean-noisy semantic decomposition," transcending the global matching paradigm of prior work.
- **Elegant adaptation of UOT**: Relaxing mass conservation naturally suits noisy settings—noisy patches need not be forcibly aligned and can be safely discarded.
- **Complementary roles of two OT variants**: UOT enables local fine-grained alignment (tolerating noise), while classical OT enables global label correction (strict mass conservation ensures assignment rationality).
- The inference formula $(1-p^n) \cdot p^c$ is concise and elegant, jointly leveraging information from both views.
- Analysis of prompt count $N=4$ reveals a diminishing returns pattern in multi-view benefit (redundancy begins at $N=8$).

## Limitations & Future Work

- Validation is limited to classification tasks; extension to more complex visual tasks such as detection and segmentation remains unexplored.
- Only ResNet-50 is used as the backbone; whether the advantage persists with stronger backbones (e.g., ViT-L/14) is unverified.
- The Sinkhorn iterations in UOT introduce additional computational overhead, which is not discussed in detail in the paper.
- The noise assumption is limited to standard symmetric/asymmetric patterns; real-world noise may be more complex (e.g., instance-dependent noise).
- The optimal number of multi-view prompts $N$ may vary by dataset, and no automatic selection mechanism is proposed.

## Related Work & Insights

- **vs. CoOp**: The baseline prompt learning method does not account for noise, and performance degrades sharply with increasing noise rate.
- **vs. NLPrompt**: Employs OT-Filter for noise identification and global OT for relabeling, representing a coarse-grained global approach; NA-MVP achieves finer-grained processing via patch-level UOT and adaptive thresholds.
- **vs. PLOT**: PLOT applies OT for multi-prompt alignment on clean data; NA-MVP extends this to a noise-aware bi-directional design.
- **vs. CLIPN**: CLIPN uses positive/negative prompt pairs for OOD detection; NA-MVP transfers the bi-directional idea to noisy label learning with the addition of multi-view prompts.
- **Broader Implications**: The "relaxed matching" principle of UOT is transferable to scenarios such as noisy annotations in object detection and imprecise labels in medical imaging; the finding that implicit negatives outperform explicit negative labels has general applicability in contrastive learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Introducing UOT into prompt-based noisy label learning is novel, and the bi-directional multi-view design offers distinctive contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five synthetic noise datasets and one real-world noise dataset, with ablations covering components, alignment strategies, prompt counts, and thresholding schemes.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, the three key limitations are well summarized, and the method is described systematically.
- Value: ⭐⭐⭐⭐ Provides an effective solution for the practically important setting of noisy few-shot learning, with substantial improvements under high noise rates.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] DeAR: Fine-Grained VLM Adaptation by Decomposing Attention Head Roles](dear_fine-grained_vlm_adaptation_by_decomposing_attention_head_roles.md)
- [\[CVPR 2026\] FlowComposer: Composable Flows for Compositional Zero-Shot Learning](flowcomposer_composable_flows_for_compositional_zeroshot_learning.md)
- [\[NeurIPS 2025\] VaMP: Variational Multi-Modal Prompt Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/vamp_variational_multi-modal_prompt_learning_for_vision-language_models.md)
- [\[CVPR 2026\] Towards Calibrating Prompt Tuning of Vision-Language Models](towards_calibrating_prompt_tuning_of_vision-language_models.md)
- [\[CVPR 2026\] Evolving Prompt Adaptation for Vision-Language Models](evolving_prompt_adaptation_for_visionlanguage_mode.md)

</div>

<!-- RELATED:END -->
