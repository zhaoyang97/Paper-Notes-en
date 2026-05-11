---
title: >-
  [Paper Note] A Framework for Double-Blind Federated Adaptation of Foundation Models
description: >-
  [ICCV 2025][AI Safety][double-blind federated learning] BlindFed proposes a double-blind federated foundation model adaptation framework combining FHE-friendly architectural redesign (polynomial approximation of nonlinea…
tags:
  - "ICCV 2025"
  - "AI Safety"
  - "double-blind federated learning"
  - "fully homomorphic encryption"
  - "split learning"
  - "foundation model adaptation"
  - "privacy preservation"
date: 2026-05-08
content_hash: 96e33a68dd527505
---

# A Framework for Double-Blind Federated Adaptation of Foundation Models

**Conference**: ICCV 2025
**arXiv**: [2502.01289](https://arxiv.org/abs/2502.01289)
**Code**: [https://github.com/tnurbek/blindfed](https://github.com/tnurbek/blindfed)
**Authors**: Nurbek Tastan (MBZUAI), Karthik Nandakumar (MBZUAI / MSU)
**Area**: Model Compression / Federated Learning / Privacy Preservation
**Keywords**: double-blind federated learning, fully homomorphic encryption, split learning, foundation model adaptation, privacy preservation

## TL;DR
BlindFed proposes a double-blind federated foundation model adaptation framework combining FHE-friendly architectural redesign (polynomial approximation of nonlinear operations), a two-stage split learning protocol (offline knowledge distillation + online encrypted inference), and privacy enhancements (sample permutation + random block sampling), achieving adaptation accuracy close to LoRA under the constraint that the data owner cannot observe the model and the model owner cannot observe the data.

## Background & Motivation
Foundation models (e.g., ViT, CLIP) exhibit strong zero-shot performance but still require adaptation for specific downstream tasks (e.g., medical imaging, satellite remote sensing). In practice, two core contradictions arise:

1. **Data privacy**: Data is distributed across multiple institutions (e.g., hospitals), and regulations prohibit data sharing.
2. **Model ownership**: Learning service providers (LSPs) are unwilling to release their expensively trained foundation models.

Conventional federated learning addresses only data privacy (clients do not share raw data) but still requires distributing the model to clients for local training, thereby violating model privacy. Existing private inference methods focus primarily on the inference stage and lack double-blind support for the training/adaptation stage. A collaborative adaptation framework that simultaneously protects both data and model privacy is therefore needed.

## Core Problem
How can cross-institutional federated adaptation of foundation models be achieved under the **double-blind constraint** — where the data owner cannot access the model and the model owner cannot access the data?

This is a practically important yet extremely challenging problem: (1) encrypted inference requires operations over ciphertexts, but FHE supports only polynomial arithmetic, making nonlinear operations such as Softmax, GELU, and LayerNorm in Transformers infeasible in the encrypted domain; (2) the adaptation method must avoid backpropagating through the FM, otherwise clients would need to store the model; (3) intermediate representations may leak information that enables model extraction attacks.

## Method

### Overall Architecture
BlindFed operates in three stages:

**Inputs**: The server holds a pretrained FM and an auxiliary public dataset; $K$ clients each hold private data for the downstream task.
**Output**: A parallel adapter and classification head adapted to the downstream task.

1. **FHE-friendly architectural redesign** (server-side, one-time): Replace nonlinear operations in the FM with polynomial approximations.
2. **Offline knowledge distillation** (server-side, one-time): Distill knowledge from the original FM into the approximated FM using the auxiliary dataset.
3. **Online encrypted inference + local learning + secure aggregation** (interactive, multi-round): Clients encrypt their data and send it to the server → the server performs inference over ciphertexts → clients decrypt intermediate representations → local updates to the parallel adapter → MPC-based secure aggregation.

### Key Designs

1. **FHE-Friendly Nonlinear Approximations**

   - **Softmax → ASoftmax**: The exponential function is approximated via a Taylor series $e^x \approx \sum_{i=0}^{d} \frac{x^i}{i!}$ (with $d=6$), followed by normalization by the sum of approximated exponentials.
   - **GELU → Quad**: A quadratic approximation $\text{Quad}(x) = 0.125x^2 + 0.25x + 0.5$.
   - **Division (in LayerNorm/Softmax) → Goldschmidt algorithm**: $\frac{1}{x} = \prod_{i=0}^{d} (1 + (1-x)^{2^i})$ (with $d=7$).

   These approximations ensure that all operations can be performed using only additions and multiplications in the FHE domain.

2. **Parallel Adapter (LoSA) Design**

   - Key design choice: LoRA cannot be used since it requires backpropagation through the FM; a **parallel adapter** is required instead.
   - The adapter runs in parallel with the FM: $\mathbf{h}_\ell = g_\ell(\mathbf{b}_\ell + \mathbf{h}_{\ell-1}) + \mathbf{h}_{\ell-1}$
   - Adapter function: $g_\ell(\mathbf{z}) = \alpha \mathbf{W}_\ell^u \text{GELU}(\mathbf{W}_\ell^d \mathbf{z})$
   - Advantage: Only intermediate representations from the FM (obtained via encrypted inference) are required; no backpropagation gradients through the FM are needed.

3. **Block-wise Encrypted Inference Protocol**

   Since the multiplicative depth of the entire FM exceeds what current FHE schemes support, **block-wise processing** is adopted: after each Transformer block, the client decrypts and re-encrypts the intermediate representation. This incurs communication overhead but avoids frequent bootstrapping operations.

4. **Privacy Enhancement Mechanisms**

   - **Sample-level permutation**: After each block, the server shuffles the order of samples in the batch using a random permutation matrix $\Pi_\ell$. Only the relative permutation $\Pi_{\ell-1}^{-1} \cdot \Pi_\ell$ is revealed to the client, making the absolute permutation unrecoverable — there are $n!$ possible solutions (approximately $2 \times 10^{13}$ when $n=16$).
   - **Stochastic Block Sampling (SBS)**: Only a subset of block outputs is returned per forward pass; the rest are set to zero. The sampling rule prevents consecutive blocks from being selected simultaneously, since features from adjacent blocks are highly similar and could be exploited for attacks. The expected sampling rate is approximately $L/3$.

### Loss & Training

**Stage 1: Offline Distillation (30 epochs)**
- First 15 epochs — Transformer layer distillation:
  - Attention matrix distillation: $\mathcal{L}_a = \frac{1}{h} \sum_{i=1}^{h} \| \mathbf{A}_i^{\mathcal{S}} - \mathbf{A}_i^{\mathcal{T}} \|^2$
  - Hidden state distillation: $\mathcal{L}_h = \| \mathbf{H}^{\mathcal{S}} - \mathbf{H}^{\mathcal{T}} \|^2$
  - Total loss: $\mathcal{L} = \mathcal{L}_a + \mathcal{L}_h$
- Last 15 epochs — prediction layer distillation:
  - $\mathcal{L}_p = \mathcal{L}_{CE}(\mathbf{z}^{\mathcal{S}}/\tau, \mathbf{z}^{\mathcal{T}}/\tau)$, with temperature $\tau=5$

**Stage 2: Federated Adaptation**
- Cross-entropy loss, SGD optimizer (lr=0.001), 50 communication rounds, learning rate decayed by 0.1 at rounds 25 and 40.

## Key Experimental Results

Foundation model: ViT-Base (pretrained on ImageNet-1K), backbone input resolution 384×384.

| Dataset | Method | Double-Blind | Centralized | FL (α=100) | FL (α=1) | FL (α=0.01) |
|--------|------|------|--------|-----------|---------|------------|
| CIFAR-10 | Full FT | ✗ | Best | Best | Best | Best |
| CIFAR-10 | LoRA | ✗ | 2nd best | 2nd best | 2nd best | 2nd best |
| CIFAR-10 | Linear Probe | ✓ | Lower | Lower | Lower | Very poor |
| CIFAR-10 | **BlindFed** | **✓** | ≈ LoRA | ≈ LoRA | ≈ LoRA | ≈ LoRA |
| CIFAR-10 | BlindFed+SBS | ✓ | ≈ BlindFed | ≈ BlindFed | ≈ BlindFed | ≈ BlindFed |

**Key Findings**:
- Under the double-blind constraint, BlindFed achieves accuracy close to LoRA (which is not double-blind) and substantially outperforms Linear Probing (which is also double-blind).
- SBS has minimal impact on accuracy and in some cases yields slight improvements due to its regularization effect.
- Sample permutation does not affect accuracy since the adapter operates sample-wise.
- Experiments on Fed-ISIC2019 demonstrate that the auxiliary dataset can be out-of-distribution (Tiny-ImageNet) and knowledge transfer remains effective.
- Scalability: BlindFed remains stable across 10/20/50 clients; full fine-tuning becomes infeasible at 50 clients due to GPU memory constraints.

**Computational Overhead**:

| Metric | Value |
|------|------|
| Encryption time (per sample) | ~1062 ms |
| Decryption time (per sample) | ~168.7 ms |
| Ciphertext size (per sample) | 17.33 MB (plaintext: 6.21 MB, ~2.8× expansion) |
| Encrypted inference time per block | ~136 s/sample |
| Server-side memory | >22 GB |
| Client-side memory | <1 GB |

### Ablation Study
- **SBS sampling rate**: Expecting $L/3$ sampled blocks with no consecutive selection; negligible impact on accuracy.
- **Softmax approximation degree**: $d=6$ achieves a good trade-off between accuracy and computational cost.
- **Inverse function approximation degree**: $d=7$ closely approximates the true inverse.
- **Adapter choice**: The parallel adapter (LoSA) is the only PEFT method (besides Linear Probing) that does not require backpropagation through the FM.
- **Parameter count**: BlindFed's trainable parameters (~0.25M) are comparable to LoRA and far fewer than full fine-tuning (~86M), with significantly lower GPU memory usage.

## Highlights & Insights
- **First double-blind federated FM adaptation framework**: The problem formulation itself — simultaneously protecting data privacy and model ownership — is a significant contribution.
- **Insightful architectural choice**: The selection of a parallel adapter over LoRA is key; among PEFT methods, only parallel adapters and Linear Probing do not require backpropagation through the FM.
- **Mathematical guarantee for sample permutation**: Proposition 1 proves that an adversary who observes only the relative permutation cannot recover the absolute permutation, requiring $n!$ brute-force attempts.
- **Dual value of SBS**: SBS not only defends against model extraction attacks but may also improve generalization as a regularizer.
- **Auxiliary dataset can be OOD**: The auxiliary data used in the distillation stage need not be in-distribution with respect to the downstream task.

## Limitations & Future Work
- **Prohibitive computational cost**: Encrypted inference through a single block for a single sample takes ~136 seconds; across 12 blocks, multiple samples, and 50 communication rounds, practical deployment is nearly infeasible.
- **High communication overhead**: Ciphertext expansion of 2.8× combined with block-wise transmission results in total communication volume of $N_k \times L \times C$.
- **Overly strong semi-honest assumption**: In practice, malicious participants may deviate from the protocol.
- **Accuracy loss from FHE approximation**: Polynomial approximations inevitably introduce errors that accumulate across deep layers.
- **Evaluation limited to image classification**: More complex tasks (detection, segmentation) and larger models are not explored.
- **GPU-accelerated FHE**: The paper notes that GPU acceleration could yield over 10× speedup, but this is not used in the experiments.

## Related Work & Insights

| Compared Method | Double-Blind | Requires FM Backprop | Key Difference |
|----------|------|----------------|----------|
| FedAvg Full FT | ✗ | ✓ | Requires distributing the full model to clients |
| LoRA / FedIT | ✗ | ✓ | LoRA has few parameters but still requires FM backpropagation |
| Linear Probing | ✓ | ✗ | Does not update the backbone; limited adaptation capacity |
| MPCFormer | N/A | N/A | Focuses only on private inference, not training/adaptation |
| SAL-ViT | N/A | N/A | Optimizes PI efficiency but does not address federated adaptation |
| **BlindFed** | **✓** | **✗** | Achieves double-blind adaptation via parallel adapter + FHE inference |

A key distinction from MPCFormer: MPCFormer also uses knowledge distillation and MPC for private inference, but BlindFed extends this paradigm to federated training/adaptation. The comparison with LoRA highlights an important insight: under the double-blind constraint, methods that require FM backpropagation are inapplicable.

The following directions are suggested for future investigation:
- **FHE-friendly model architecture search**: Can NAS automatically discover FHE-optimal architectures rather than relying on manual approximation?
- **Parallel adapters in other privacy-constrained settings**: The "no backpropagation through FM" property of LoSA is valuable in other resource-constrained scenarios.
- **Intersection with model compression**: Smaller models lead to fewer blocks, lower communication/computation costs, and more practical double-blind FL.
- **SBS as a regularizer**: Randomly dropping partial block outputs resembles Dropout and may be generalizable.

## Rating
- Novelty: ⭐⭐⭐⭐ — The problem formulation and framework for double-blind FL adaptation are novel, though individual components (FHE approximation, LoSA, knowledge distillation) are not new.
- Experimental Thoroughness: ⭐⭐⭐ — Experiments use a relatively small ViT-Base and cover only image classification; actual FHE experiments are limited in scope.
- Writing Quality: ⭐⭐⭐⭐ — The problem is clearly defined, the framework is well-structured, and the mathematics is rigorous.
- Value: ⭐⭐⭐ — The concept is compelling, but the extreme computational cost places practical deployment far out of reach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICCV 2025\] Find a Scapegoat: Poisoning Membership Inference Attack and Defense to Federated Learning](find_a_scapegoat_poisoning_membership_inference_attack_and_defense_to_federated_.md)
- [\[ICCV 2025\] FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](fedvla_federated_vision-language-action_learning_with_dual_gating_mixture-of-exp.md)
- [\[ICCV 2025\] Ask and Remember: A Questions-Only Replay Strategy for Continual Visual Question Answering](ask_and_remember_a_questions-only_replay_strategy_for_continual_visual_question_.md)
- [\[ICCV 2025\] FRET: Feature Redundancy Elimination for Test Time Adaptation](fret_feature_redundancy_elimination_for_test_time_adaptation.md)

</div>

<!-- RELATED:END -->
