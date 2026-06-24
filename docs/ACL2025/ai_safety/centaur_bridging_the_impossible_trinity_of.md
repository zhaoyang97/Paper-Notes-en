---
title: >-
  [Paper Note] CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference
description: >-
  [ACL 2025][AI Safety][privacy-preserving inference] This paper proposes the Centaur framework, which integrates random permutation matrices and Secure Multi-Party Computation (SMPC) to break the "impossible trinity" in Privacy-Preserving Transformer Inference (PPTI)—simultaneously achieving strong privacy protection, 5-30x speedup, and plaintext-level inference accuracy.
tags:
  - "ACL 2025"
  - "AI Safety"
  - "privacy-preserving inference"
  - "SMPC"
  - "permutation matrix"
  - "Transformer"
  - "impossible trinity"
date: 2026-05-08
content_hash: 140a62f256b56e2b
---

# CENTAUR: Bridging the Impossible Trinity of Privacy, Efficiency, and Performance in Privacy-Preserving Transformer Inference

**Conference**: ACL 2025  
**arXiv**: [2412.10652](https://arxiv.org/abs/2412.10652)  
**Code**: None  
**Area**: AI Security  
**Keywords**: privacy-preserving inference, SMPC, permutation matrix, Transformer, impossible trinity

## TL;DR
This paper proposes the Centaur framework, which integrates random permutation matrices and Secure Multi-Party Computation (SMPC) to break the "impossible trinity" in Privacy-Preserving Transformer Inference (PPTI)—simultaneously achieving strong privacy protection, 5-30x speedup, and plaintext-level inference accuracy.

## Background & Motivation

**Background**: Transformer models are widely deployed in the cloud (e.g., chatbots, code assistants), exposing model parameters and inference data to privacy risks. For landscape perspective, Samsung once banned employees from using external LLM services due to internal code leaks.

**Limitations of Prior Work**: PPTI faces an "impossible trinity"—(1) SMPC methods offer strong privacy but suffer from massive communication overhead (e.g., BERT inference takes 881 seconds to transmit 66GB); replacing non-linear operations can improve efficiency but compromises accuracy. (2) Permutation matrix methods achieve efficiency and accuracy close to plaintext but expose model parameters, allowing intermediate resulting states to be restored to original inference inputs via data reconstruction attacks.

**Key Challenge**: SMPC preserves privacy but is slow, while permutation preserves efficiency but leaks information—the two are complementary, but combining them effectively remains a challenge.

**Goal**: To design a PPTI framework that simultaneously satisfies privacy, efficiency, and performance.

**Key Insight**: Adopt the most suitable protection strategies for the linear and non-linear layers of Transformers respectively—the linear layers use permutation to convert ciphertext-ciphertext multiplication to plaintext-ciphertext multiplication (eliminating communication), while the non-linear layers utilize SMPC to perform precise computation in the permuted state.

**Core Idea**: Combining communication-free acceleration using permutation matrices for linear layers, exact computation in the permuted state via SMPC for non-linear layers, and a three-party protocol to guarantee privacy, thereby breaking the impossible trinity.

## Method

### Overall Architecture
Three participating parties: $\mathcal{P}_0$ (model developer holding model parameters), $\mathcal{P}_1$ (cloud platform executing computations), and $\mathcal{P}_2$ (client holding inference data). In the initialization phase, $\mathcal{P}_0$ generates the permutation matrices $\Pi = \{\pi, \pi_1, \pi_2\}$ and sends the permuted model parameters to $\mathcal{P}_1$. In the inference phase, $\mathcal{P}_2$ secretly shares the inference data with $\mathcal{P}_0$ and $\mathcal{P}_1$, and both parties jointly execute PPTI.

### Key Designs

1. **Linear Layer Acceleration**:

    - Function: Converts matrix multiplication between ciphertexts $\Pi_{\text{MatMul}}$ into plaintext-ciphertext multiplication $\Pi_{\text{ScalMul}}$ (requiring no communication).
    - Mechanism: Model parameters $W$ are permuted into $W\pi$ (plaintext) using the permutation matrix $\pi$, while the inference data remains secretly shared as $[[X_E\pi]]$. Since $X_E\pi \cdot (W\pi)^\top = X_E W^\top$ (the orthogonality of permutation ensures computational correctness), $\Pi_{\text{ScalMul}}$ requires no communication.
    - Design Motivation: Linear layers account for the vast majority of the communication overhead in PPTI; eliminating this communication yields substantial acceleration.

2. **Non-linear Layer Processing**:

    - Function: Converts the secretly shared state $[[X\pi]]$ to the permuted state $X\pi$, performing element-wise non-linear operations (Softmax, GeLU, LayerNorm) on plaintext.
    - Mechanism: $\mathcal{P}_0$ sends the share $[X\pi]_0$ to $\mathcal{P}_1$, which reconstructs $X\pi$ and computes $f_e(X\pi) = f_e(X)\pi$ in the permuted state (as element-wise operations are commutative with permutation), and then re-shares the result. This requires only 2 rounds of communication.
    - Design Motivation: Evaluating precise non-linear operations in SMPC is extremely expensive; computing in plaintext under the permuted state is both accurate and efficient, while the permutation protects privacy.

3. **Privacy-Permutation-Secret Sharing Protocol (PPP)**:

    - Function: Solves the cancellation of permutations in the attention mechanism—where the output $O_1$ of $Q \cdot K^\top$ is unpermuted and needs to be re-permuted.
    - Mechanism: Converts $[[X]]$ to $[[X\pi]]$ via the privacy-preserving matrix multiplication $\Pi_{\text{MatMul}}$. This is only applied in the two matrix multiplications within the attention mechanism, with all other parts utilizing the communication-free $\Pi_{\text{ScalMul}}$.

### Loss & Training
Centaur requires no training, fine-tuning, or model modification, directly replacing the inference workflow.

## Key Experimental Results

### Main Results

| Method | Privacy | Speed (relative to CrypTen) | Accuracy |
|------|:---:|:---:|:---:|
| CrypTen (Pure SMPC) | ✓ Strong | 1x | ✓ Plaintext-level |
| MPCFORMER (Replaced Non-linear) | ✓ Strong | ~3x | ✗ Degraded |
| PUMA (Optimized SMPC) | ✓ Strong | ~2x | ✓ Plaintext-level |
| Yuan23 (Pure Permutation) | ✗ Leaked | ~30x | ✓ Plaintext-level |
| **Centaur** | **✓ Strong** | **5-30x** | **✓ Plaintext-level** |

### Ablation Study (Defense Against Data Reconstruction Attacks)

| Attack Method | Without Centaur (Reconstruction Rate) | With Centaur (Reconstruction Rate) | Description |
|------|:---:|:---:|------|
| SIP Attack | 84-88% | **3.9-5.4%** | Almost completely defended |
| EIA Attack | 59-91% | **2.6-6.4%** | Almost completely defended |
| Random Guess | 3.9-5.2% | 3.9-5.2% | Centaur is on par with random guessing |

### Key Findings
- **Truly Breaking the Impossible Trinity**: Centaur is the first PPTI framework to simultaneously achieve strong privacy, high efficiency, and plaintext-level accuracy.
- **Communication-Free Linear Layers**: The communication overhead of all linear layers is reduced to zero through permutation, with communication only required during the twice matrix multiplications in attention and the transitions of non-linear operations.
- **Robustness to Data Reconstruction Attacks**: Under two mainstream attack schemes, Centaur reduces the reconstruction rate from 60-91% down to 3-6%, which is close to random guessing.
- **Security of Permutation Matrices**: When $d=1280$, the brute-force recovery probability is approximately $1/2^{11372}$.

## Highlights & Insights
- **The Design Philosophy of "Complementary Strategies"**: The linear and non-linear layers each leverage the most suitable protection methods instead of relying on a single approach to forcibly cover all operations. This tailored mindset can be extended to other privacy-preserving computation scenarios.
- **Plaintext-Level Accuracy as a Key Competitive Edge**: Without replacing non-linear operations or modifying models, it can be directly applied to pre-trained models without retraining or fine-tuning.
- **Security Justification of Permuting-State Non-linear Computation**: Although reconstructing secret shares into plaintext permuted states presents a theoretical leakage risk, experiments demonstrate that the permutation provides sufficient protection (dropping attack success rates to random guessing levels).

## Limitations & Future Work
- The three-party protocol relies on a non-collusion assumption (where $\mathcal{P}_0$ and $\mathcal{P}_1$ do not collude), which might hold limitations in practice.
- Two SMPC matrix multiplications are still required within the attention mechanism, serving as the efficiency bottleneck.
- Tested only on BERT and GPT-2; scalability to larger models such as LLaMA-70B remains unverified.
- The efficiency and security of generative inference (multi-step autoregression) tasks are not discussed.

## Related Work & Insights
- **vs MPCFORMER/PUMA (SMPC Methods)**: They purely use SMPC, resulting in massive communication overhead. Centaur eliminates communication in linear layers via permutation, achieving a 5-30x speedup.
- **vs Yuan et al. 2023 (Permutation Methods)**: Pure permutation exposes embedding layer parameters and some intermediate results. Centaur secures these sensitive parts using SMPC.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Formulates the first fusion of permutation and SMPC to break the impossible trinity with elegant design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across privacy, efficiency, and accuracy, plus attack experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with a compelling discussion of the impossible trinity.
- Value: ⭐⭐⭐⭐⭐ Highly practical, offering significant advancement for secure AI deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Crafting Privacy-Preserving Adversarial Examples: A Defense Against Membership Inference](crafting_privacy-preserving_adversarial_examples_a_defense_against_membership_inf.md)
- [\[ICCV 2025\] FedMeNF: Privacy-Preserving Federated Meta-Learning for Neural Fields](../../ICCV2025/ai_safety/fedmenf_privacy-preserving_federated_meta-learning_for_neural_fields.md)
- [\[ICLR 2026\] Privacy Beyond Pixels: Latent Anonymization for Privacy-Preserving Video Understanding](../../ICLR2026/ai_safety/privacy_beyond_pixels_latent_anonymization_for_privacy-preserving_video_understa.md)
- [\[CVPR 2026\] Bridging Privacy and Provenance: Traceable Virtual Identity Generation](../../CVPR2026/ai_safety/bridging_privacy_and_provenance_traceable_virtual_identity_generation.md)
- [\[ACL 2025\] PrivaCI-Bench: Evaluating Privacy with Contextual Integrity and Legal Compliance](privacibench_evaluating_privacy_with_contextual_integrity.md)

</div>

<!-- RELATED:END -->
