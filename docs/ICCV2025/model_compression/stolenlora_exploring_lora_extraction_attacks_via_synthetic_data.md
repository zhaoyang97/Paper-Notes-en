---
title: >-
  [Paper Note] StolenLoRA: Exploring LoRA Extraction Attacks via Synthetic Data
description: >-
  [ICCV 2025][Model Compression][LoRA extraction] StolenLoRA is the first work to formulate model extraction attacks targeting LoRA-adapted models. It leverages LLM-driven Stable Diffusion to synthesize high-quality training data, eliminating the need to search real datasets, and designs a Disagreement-based Semi-supervised Learning (DSL) strategy that maximizes information gain through selective querying. With only 10k queries, StolenLoRA achieves an attack success rate (ASR) of up to 96.60%, exposing critical security vulnerabilities in LoRA-adapted models.
tags:
  - ICCV 2025
  - Model Compression
  - LoRA extraction
  - model extraction attack
  - PEFT
  - synthetic data
  - Stable Diffusion
  - disagreement-based semi-supervised learning
  - LLM-driven prompting
date: 2026-05-08
content_hash: 107fca8b18a21fee
---

# StolenLoRA: Exploring LoRA Extraction Attacks via Synthetic Data

**Conference**: ICCV 2025
**arXiv**: [2509.23594](https://arxiv.org/abs/2509.23594)
**Code**: To be confirmed
**Area**: Model Security / Model Extraction Attack / LoRA / Parameter-Efficient Fine-Tuning
**Keywords**: LoRA extraction, model extraction attack, PEFT, synthetic data, Stable Diffusion, disagreement-based semi-supervised learning, LLM-driven prompting

## TL;DR
StolenLoRA is the first work to formulate model extraction attacks targeting LoRA-adapted models. It leverages LLM-driven Stable Diffusion to synthesize high-quality training data, eliminating the need to search real datasets, and designs a Disagreement-based Semi-supervised Learning (DSL) strategy that maximizes information gain through selective querying. With only 10k queries, StolenLoRA achieves an attack success rate (ASR) of up to 96.60%, exposing critical security vulnerabilities in LoRA-adapted models.

## Background & Motivation

LoRA (Low-Rank Adaptation) has become the dominant method for efficiently fine-tuning large-scale pretrained models, but its lightweight and compact nature introduces new security risks:

**Vulnerability of LoRA Parameters**: LoRA fine-tunes only a small number of low-rank matrices, with the model's core knowledge residing in a publicly available pretrained base. This means an attacker only needs to replicate the LoRA adaptation to obtain full model functionality—a significantly lower barrier than extracting an entire model.

**Limitations of Existing Extraction Methods**:
- **Sample selection methods** (KnockoffNets/ActiveThief): Require searching large datasets for in-domain samples, incurring enormous computational overhead when the victim LoRA model has slow inference, and struggling to find domain-specific samples.
- **GAN-based methods** (DFME): GANs struggle to generate high-dimensional data (e.g., 224×224 images), require millions of queries, and are prone to mode collapse.

**Availability of Pretrained Models**: A large number of pretrained ViT models are publicly released on platforms such as Hugging Face, allowing attackers to easily obtain the same or similar base model as the victim, further lowering the attack barrier.

**Research Gap**: Traditional model extraction research focuses on replicating the functionality of an entire model. The LoRA setting, where only compact adaptation parameters need to be extracted, is a new direction that has not been adequately explored.

## Method

### Problem Formulation: LoRA Extraction

Given a victim model $F = F_\text{base} + \Delta F$ (where $F_\text{base}$ is public and $\Delta F$ represents LoRA adjustments), the attacker's goal is to train a surrogate model $F' = F'_\text{base} + \Delta F'$ whose functionality approximates $F$ as closely as possible.

**Two Attack Scenarios**:
- **In-Backbone (IB)**: The attacker uses the same pretrained model $F_\text{base}$ as the victim.
- **Cross-Backbone (XB)**: The attacker uses a different pretrained model $G_\text{base}$ (e.g., the victim uses a supervised ImageNet-21k pretrained ViT, while the attacker uses a MAE self-supervised pretrained ViT).

### Stage 1: LLM-Driven Data Synthesis

The attacker typically has access to deployment context (functional description, target class names), which serves as the starting point for synthesizing training data:

1. **LLM-Driven Prompt Generation**:
   - Given a set of target class names $C = \{c_1, \ldots, c_n\}$, GPT-4o mini is used to generate diverse image descriptions.
   - Prompt template $T = [\text{Subject, Background, Angle/Pose, Lighting, Style}]$.
   - For each class, $m$ distinct prompt variants are generated: $p_{i,j} = \text{LLM}(c_i, T, \omega_j)$.

2. **Image Synthesis**:
   - SDXL-Turbo (requiring only 4 sampling steps) is used to generate one image per prompt.
   - Full synthetic dataset: $X = \bigcup_i \{\text{SD}(p_{i,j})\}$.
   - Synthesized images carry pseudo-labels derived from the class information in the generation prompts.

### Stage 2: Efficient Querying and Training

**Option A: Random Learning (StolenLoRA-Rand)**

The synthetic dataset is used to query the victim model API directly, and the surrogate model's LoRA parameters are trained with cross-entropy loss. This serves as the baseline.

**Option B: Disagreement-based Semi-supervised Learning (StolenLoRA-DSL)**

DSL iteratively improves attack efficiency through selective querying and label refinement:

1. **Initialization**: Generate an initial synthetic dataset $X_0$, and train an initial surrogate model $F'_0$ using pseudo-labels from generation prompts.
2. **Iterative Process (each round $t$)**:
   - Generate $\beta \cdot b_t$ new candidate samples $X_\text{cand}^t$.
   - **Disagreement Filtering**: Apply the current surrogate $F'_t$ to predict class $\hat{c}$ and confidence $\hat{p}$ for each candidate:
     - If $\hat{c} = \text{pseudo-label } c(x)$ and $\hat{p} \geq \tau$ (threshold 0.95): sample is high-confidence, added to $X_\text{conf}^t$, trained with pseudo-labels directly without querying the victim.
     - Otherwise: sample is uncertain, added to $X_\text{uncer}^t$.
   - **Selective Querying**: Select the $b_t$ samples with lowest confidence from $X_\text{uncer}^t$ to query the victim model for true labels.
   - **Training**: Merge $X_\text{conf}^t$ (pseudo-labels) and $X_\text{query}^t$ (true labels) to update the surrogate model.
3. **Label Refining**:
   - Soft labels are updated via EMA: $q^{(i+1)} = \mu \cdot q^{(i)} + (1-\mu) \cdot p^{(i+1)}$.
   - The surrogate is trained with cross-entropy on soft labels, mitigating pseudo-label noise and distribution shift.
4. Repeat until the query budget is exhausted.

### Key Design Principles

- **Maximize Information Gain**: Only uncertain samples are queried, avoiding wasting the query budget on easy samples the surrogate already handles correctly.
- **Synthetic Data + Pseudo-Labels Reduce Query Dependency**: Most high-confidence samples are trained directly with pseudo-labels, so the actual number of victim queries is far smaller than the total synthetic data volume.
- **Dual Role of EMA Label Refinement**: Simultaneously suppresses pseudo-label noise and bridges the distributional gap between synthetic and real data.

## Key Experimental Results

### Main Results (10k Query Budget)

#### In-Backbone Scenario (IB)

| Method | CUBS200 Acc/ASR | Caltech256 Acc/ASR | Indoor67 Acc/ASR | Food101 Acc/ASR | Flowers102 Acc/ASR |
|------|---:|---:|---:|---:|---:|
| KnockoffNets | 70.95/80.54 | 85.69/90.71 | 79.18/92.59 | 82.53/90.62 | 76.24/77.28 |
| ActiveThief | 72.33/82.11 | 86.92/92.01 | 77.46/90.57 | 80.34/88.22 | 77.51/78.56 |
| DFME | 0.56/0.64 | 0.48/0.51 | 2.71/3.17 | 1.62/1.78 | 2.81/2.85 |
| E³ | 71.94/81.67 | 82.36/87.18 | 81.43/95.22 | 79.65/87.46 | 80.67/81.77 |
| **StolenLoRA-Rand** | **75.35/85.54** | 87.62/92.75 | 82.38/96.33 | 79.00/86.75 | **93.74/95.01** |
| **StolenLoRA-DSL** | 73.23/83.13 | **89.30/94.53** | **82.61/96.60** | **80.57/88.47** | 87.46/88.65 |

#### Cross-Backbone Scenario (XB)

| Method | CUBS200 Acc/ASR | Caltech256 Acc/ASR | Indoor67 Acc/ASR |
|------|---:|---:|---:|
| KnockoffNets | 6.77/7.69 | 41.34/43.76 | 41.79/48.87 |
| ActiveThief | 15.42/17.50 | 42.89/45.40 | 36.04/42.14 |
| E³ | 17.55/19.92 | 48.17/50.99 | 55.85/65.31 |
| **StolenLoRA-Rand** | **45.70/51.88** | 51.75/54.78 | 59.18/69.20 |
| **StolenLoRA-DSL** | **50.14/56.92** | **65.01/68.82** | **65.07/76.09** |

- Peak ASR in the IB scenario reaches 96.60% (Indoor67) with only 10k queries.
- StolenLoRA-DSL substantially outperforms all baselines in the XB scenario; on CUBS200, its ASR is 7.4× that of KnockoffNets.
- The GAN-based method DFME completely fails in the LoRA extraction setting (ASR < 3.17%), confirming that GAN-based approaches are unsuitable for this scenario.

### Hard-Label Scenario

When only one-hot predictions are available (no probability distributions), StolenLoRA remains competitive:
- IB scenario shows a modest performance drop (e.g., CUBS200: 75.35% → 67.00%).
- XB scenario shows slight improvements on some datasets (Caltech256: 51.75% → 56.69%), where hard labels appear to act as regularization.

### Ablation Study

| Component | IB CUBS200 | XB CUBS200 |
|------|---:|---:|
| Full StolenLoRA-DSL | 73.23 | 50.14 |
| − Template | 72.33 | 42.21 |
| − LLM | Significant drop | Larger drop |
| − DSL (→ Random) | Drop | Noticeable drop |
| GPT-4o → Llama-3.1-8B | Slight drop | Larger drop |

- LLM-driven prompting is the core component; removing it causes the most severe degradation in the XB scenario.
- Confidence threshold $\tau = 0.95$ is optimal; setting it too high (0.99) excludes informative samples.

## Highlights & Insights

- **A New Security Direction**: StolenLoRA is the first work to systematically define and study LoRA extraction attacks, revealing a new security surface introduced by PEFT methods. The compactness of LoRA, which is an advantage in efficiency, becomes a vulnerability from a security perspective.
- **LLM + Diffusion Model Attack Data Pipeline**: LLM-generated diverse textual descriptions feed SDXL-Turbo image synthesis, yielding pseudo-labeled data without any dependence on real datasets, at extremely low attack cost.
- **Elegant Query Optimization via DSL**: Rather than querying uniformly, DSL focuses the query budget on samples where the surrogate model is uncertain, achieving maximum utilization of the budget. An ASR of 96.60% is attained with just 10k queries.
- **Feasibility of Cross-Backbone Attacks**: Even when the attacker does not know the exact pretrained model used by the victim, effective extraction remains achievable (XB ASR up to 76.09%), making the attack practically threatening in real-world deployments.
- **Dual Role of EMA Label Refinement**: A single lightweight mechanism simultaneously suppresses pseudo-label noise and bridges the distributional gap between synthetic and real data.

## Limitations & Future Work

- Experiments are conducted only on ViT-Base; larger ViT variants and non-ViT architectures (e.g., CNNs, hybrid architectures) are not evaluated.
- Only image classification LoRA extraction is considered; the difficulty and feasibility of LoRA extraction for more complex tasks (detection, segmentation, NLP) remain unknown.
- LoRA rank is fixed at $r=4$; the effect of varying rank on extraction difficulty is not systematically analyzed.
- Defense strategies (diversified LoRA deployment) are only preliminarily explored, and more comprehensive defenses are lacking.
- Synthetic data quality depends on LLM and diffusion model capabilities; generating sufficiently accurate in-domain data for highly specialized domains (e.g., medical imaging) may be challenging.
- Practical API-level protections such as rate limiting and query detection are not considered.
- The attacker must know target class names to generate synthetic data, which limits applicability in fully black-box settings.

## Related Work & Insights

- **vs. KnockoffNets**: KnockoffNets searches large-scale datasets such as CC3M to select query samples, requiring traversal of 3M images. StolenLoRA's synthesis strategy is more efficient and eliminates dependence on real datasets.
- **vs. DFME**: DFME uses GANs to generate query data and fails completely in the LoRA extraction setting (ASR < 3.17%), as GANs struggle to produce high-quality 224×224 in-domain images. StolenLoRA addresses this limitation by adopting Stable Diffusion.
- **vs. E³**: E³ selects samples from real datasets based on semantic similarity; StolenLoRA outperforms E³ across all settings without relying on any real dataset.
- **A Warning for PEFT Security Research**: The widespread deployment of LoRA (with large numbers of LoRA adapters shared on Hugging Face) calls for accompanying intellectual property protection mechanisms. Future work may explore LoRA watermarking, differentially private training, and output perturbation as defenses.
- **Synthetic Data as a New Attack Vector**: Synthetic data can be weaponized for attack scenarios beyond its conventional roles in data augmentation and training, and the security community should be alert to this dual-use nature.

## Rating
- Novelty: ⭐⭐⭐⭐ First to define the LoRA extraction problem; the LLM+SD synthesis pipeline and DSL strategy are original contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets × 2 scenarios × 4 baselines + ablation + hard-label + hyperparameter analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear; IB/XB scenario distinction is well-motivated; algorithmic pseudocode is complete and rigorous.
- Value: ⭐⭐⭐⭐ Reveals a novel security threat in the PEFT era with direct implications for security practices in the LoRA ecosystem.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](../../NeurIPS2025/model_compression/toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [\[ICCV 2025\] OuroMamba: A Data-Free Quantization Framework for Vision Mamba](ouromamba_a_data-free_quantization_framework_for_vision_mamba.md)
- [\[NeurIPS 2025\] FirstAidQA: A Synthetic Dataset for First Aid and Emergency Response in Low-Connectivity Settings](../../NeurIPS2025/model_compression/firstaidqa_a_synthetic_dataset_for_first_aid_and_emergency_response_in_low-conne.md)
- [\[ICCV 2025\] Partial Forward Blocking: A Novel Data Pruning Paradigm for Lossless Training Acceleration](partial_forward_blocking_a_novel_data_pruning_paradigm_for_lossless_training_acc.md)
- [\[AAAI 2026\] LexChronos: An Agentic Framework for Structured Event Timeline Extraction in Indian Jurisprudence](../../AAAI2026/model_compression/lexchronos_an_agentic_framework_for_structured_event_timeline_extraction_in_indi.md)

<!-- RELATED:END -->
