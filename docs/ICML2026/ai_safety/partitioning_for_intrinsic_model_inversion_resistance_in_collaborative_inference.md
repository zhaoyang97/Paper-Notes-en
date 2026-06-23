---
title: >-
  [Paper Note] Partitioning for Intrinsic Model Inversion Resistance in Collaborative Inference
description: >-
  [ICML 2026][AI Safety][Paper Note] This paper moves beyond the traditional defense paradigm of "adding noise or masking shallow intermediate representations." From an information-theoretic perspective, it proves that in edge-cloud collaborative inference, the model should be partitioned at the layer where the representation undergoes a "feature-to-decis
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: 2478e8ff3186da69
---
# Partitioning for Intrinsic Model Inversion Resistance in Collaborative Inference

**Conference**: ICML 2026  
**arXiv**: [2506.15412](https://arxiv.org/abs/2506.15412)  
**Code**: https://github.com/GoldenPartitionZone/GoldenPartitionZone  
**Area**: AI Safety / Collaborative Inference / Model Inversion Defense  
**Keywords**: Model Inversion Attack, Collaborative Inference, Partitioning Strategy, Information Entropy, Label Smoothing

## TL;DR
This paper moves beyond the traditional defense paradigm of "adding noise or masking shallow intermediate representations." From an information-theoretic perspective, it proves that in edge-cloud collaborative inference, the model should be partitioned at the layer where the representation undergoes a "feature-to-decision" phase transition (named the Golden Partition Zone, GPZ by the authors). The intra-class mean square radius $R_c^2$ is identified as the key variable for locating the GPZ and can be actively contracted during training via label smoothing dynamics.

## Background & Motivation

**Background**: Collaborative Inference (CI) partitions a deep network into an edge segment $f_{\text{edge}}$ and a cloud segment $f_{\text{cloud}}$, where the edge uploads intermediate representations $z = f_{\text{edge}}(x)$ to the cloud. This deployment is widely used in UAVs, IoT, and private cloud inference. However, Model Inversion Attacks (MIA) can train a generator $g \approx f_{\text{edge}}^{-1}$ to reconstruct the original input $x$ from $z$, leading to sample-level privacy leakage.

**Limitations of Prior Work**: Existing MIA defenses almost exclusively focus on perturbing shallow representations $z$ (adding noise, masking, bottleneck layers, homomorphic encryption, etc.). The cost is either a sacrifice in downstream accuracy or the introduction of additional computational overhead, essentially struggling with the "privacy-utility" tradeoff.

**Key Challenge**: The authors argue that the question should not be how to patch the representation. The fundamental question is—at which layer should the network be partitioned so that $z$ naturally and irreversibly loses input information before transmission? This shifts defense from a "post-hoc patch" to the "partition location" itself.

**Goal**: (1) Theoretically characterize the relationship between "partition location" and MIA difficulty; (2) Provide a computable and monitorable layer-wise metric to allow users to actively locate the optimal partition point; (3) Actively shape this metric during the training phase to enter the inversion-resistant zone earlier.

**Key Insight**: Traditional intuition suggests "deeper is safer." However, on ViT, patch tokens retain detailed information for each sample even at the final layer, allowing successful inversion. On architectures with residuals like IR-152/ResNet-50, increasing depth causes $I(X; Z)$ to decay more slowly due to skip connections. These counterexamples forced the authors to shift focus from "depth" to the "intrinsic mutation of representation morphology."

**Core Idea**: The transition from "feature-level" to "decision-level" representation is used as a necessary condition for intrinsic defense. The intra-class mean square radius $R_c^2 = \frac{1}{N_c} \sum_{i:y_i=c} \|z_i - \mu_c\|^2$ is employed as the sole computable proxy variable to locate this transition interval (the GPZ), while $R_c^2$ is actively contracted during training via techniques such as label smoothing.

## Method

### Overall Architecture

The paper follows a chain of "Theory → Metric → Training Dynamics → Experimental Validation." First, a lower bound for $H(X \mid Z)$ is derived, revealing that this bound is dominated by the global variance $\sigma_{\text{feat}}^2$ at feature layers, whereas at decision layers, it is dominated by the intra-class radius $R_c^2$ (which is typically much smaller), causing the lower bound to jump. Second, $R_c^2$ is refined into a usable layer-wise probe. Finally, label distributions are used to reverse-regulate the training dynamics of $R_c^2$ (termed the Neural Vortex), making $R_c^2$ at decision layers smaller and the inversion resistance stronger.

### Key Designs

**1. GPZ Localization Criterion: An $R_c^2$ Probe Derived from the $H(X\mid Z)$ Lower Bound**

To transform the engineering problem of "where to cut" into an observable and automated process, a scalar reflecting whether the representation has "decisionized" must be calculated for each layer. The authors treat $z$ as a continuous variable and use the maximum entropy principle with the determinant-trace inequality to obtain upper bounds for differential entropy at two levels: the feature level $h(Z_{\text{feat}}) \le \frac{d}{2}\ln(2\pi e \sigma_{\text{feat}}^2)$ depends mainly on dimension $d$ and global variance $\sigma_{\text{feat}}^2$; the decision level, after conditioning on the class, becomes $h(Z_{\text{dec}} \mid Y=c) \le \frac{D}{2}\ln(2\pi e R_c^2/D)$, where the decisive variance term shifts from "global variance" to the "intra-class mean square radius $R_c^2$." Substituting these into the mutual information identity $H(X\mid Z) \ge H(X) - h(Z) - \kappa_\Delta$ shows that when the representation enters the decision zone and $h(Z)$ drops significantly, the inversion resistance lower bound jumps. In practice, one simply scans $R_c^2$ across candidate layers to find the layer where an abrupt drop occurs, completely bypassing high-variance and costly mutual information estimators like MINE.

**2. Neural Vortex: Training Dynamics for Actively Contracting $R_c^2$ via Label Smoothing**

Locating the GPZ is insufficient; the authors aim to further "pull down" $R_c^2$ at the decision layers during late-stage training to further raise the inversion resistance lower bound. Writing the backpropagation step as $\Delta R_c^2 = -\frac{2\gamma}{N_c} \sum_{i\in c} (z_i - \mu_c)^\top \tilde g_i$ and substituting $\tilde g_i = J_i^\top (p_i - y_i)$, it can be decomposed into a "correct class pulling term" $(p_{ic}-1)T_{\text{corr},i}$ and an "incorrect class interference term" $\sum_{k\ne c} p_{ik} T_{k,i}$. Under one-hot supervision, the pulling term approaches zero as $p_{ic} \to 1$, halting the decline of $R_c^2$. With Label Smoothing (LS), the correct class coefficient becomes $(p_{ic}-1+\alpha)$. Once $p_{ic} > 1-\alpha$, the sign of this coefficient flips, and geometrically $T_{\text{corr},i}$ also reverses, ultimately maintaining $\Delta R_c^2 < 0$ and continuously tightening the intra-class point cloud. The authors name this counter-intuitive coupling of "output-side entropy increase and intermediate-side entropy decrease" the Neural Vortex. Unlike simple IB regularization or post-hoc observation of neural collapse, it actively regulates training dynamics with almost no loss in downstream accuracy (LS+ even slightly improves it in experiments).

**3. Bidirectional Stress Testing for Decision-Layer Resistance: Enhanced Entropy + Enhanced Inversion**

Since defenses often fail against stronger attack models, it is necessary to confirm that the GPZ is not bolstered by a weak attacker. The authors apply "buffs" to both ends for stress testing: on the representation side, they enrich the transmitted $z$ with FFT residuals/concatenation, global normalization, and small NN modules with dropout; on the attack side, they progressively insert multi-head attention, Attention-as-Conv, SE, LSK, and MSCA between deconvolutional blocks, following a "shallow weak attention → deep strong decoupling" pairing principle, while also testing inverse IR-152 residual blocks. These enhancements serve to verify that the GPZ still suppresses inversion quality—solidifying the conclusion that "decision-layer inversion resistance" holds a significant gap even against enhanced attacks.

### Loss & Training

Target models were trained on 7 datasets (CIFAR-10, FaceScrub, KMNIST, etc.) at $64\times 64$ resolution, using three label distributions: one-hot, LS+ ($\alpha=0.3$), and LS- ($\alpha=-0.05$, reverse smoothing as a control). The inversion model adopts the deconvolutional backbones of Yang et al. (2019) and Zhang et al. (2023). Evaluations used MSE / PSNR / SSIM / LPIPS (default AlexNet weights), with MSE $<0.02$ as the empirical threshold for "high-fidelity reconstruction."

## Key Experimental Results

### Main Results: Impact of Representation Layer on MIA Difficulty (IR-152, CIFAR-10)

| Partition Point | Representation Type | MSE (Test) | PSNR (Test) | Reconstruction Readable? |
| :--- | :--- | :--- | :--- | :--- |
| Block 40 | Feature Level | 0.018 | 22.17 | Yes |
| Block 48 | Residual accumulation, still feature | $<0.02$ | $\approx 22$ | Yes |
| Block 50 | Decision Level (GPZ) | 0.057 | 17.22 | No |
| Block 30→39 (VGG19) | Feature→Decision Mutation | 0.066 → 0.137 | — | Significant degradation at mutation |

In IR-152, the spatial resolution is compressed to $4\times 4$ at Block 49, where the representation suddenly becomes decision-oriented, and MSE jumps from $<0.02$ to $0.057$. This is the source of the claimed "4× higher MSE at GPZ vs shallow cuts." The paper also notes that because ViT always retains 256 patch tokens, no representation transition occurs, making it impossible to form a GPZ regardless of where it is partitioned.

### Ablation Study: Stability of GPZ under Representation/Attack Enhancements (IR-152, Block 50 vs Block 40)

| Configuration | Block 50 (GPZ) MSE | Block 40 (Feature) MSE | GPZ Relative Disadvantage Shrink? |
| :--- | :--- | :--- | :--- |
| Baseline Attack | 0.057 | 0.018 | — |
| Representation: Normalize+Dropout-Concat | 0.052 | 0.014 | No (~3.7× gap maintained) |
| Attack: Attention-as-Conv+SE+LSK+MSCA | 0.051 | — | No |
| Combined Enhancement + Inversion-IR152 | 0.049–0.052 | 0.012 | No (~4× gap maintained) |

### Key Findings

- True intrinsic defense comes not from "cutting deep," but from "cutting after the morphological transition of the representation." Residual connections and ViT patch tokens delay or eliminate this transition, making them nearly ineffective against MIA.
- Under both attack and representation enhancements, decision-level representations maintain an average 66% inversion resistance advantage over feature-level ones, indicating GPZ is not a fragile artifact of weak attacks.
- Data distribution determines GPZ position: On FaceScrub, the GPZ is earlier and narrower; on KMNIST, many zero pixels cause feature extraction to persist deeper, shifting GPZ to around Block 43. This aligns with the joint effect of $H(X)$ and $R_c^2$ in the lower bound.
- When an inversion model trained on MNIST is used for KMNIST, reconstructions lean toward "0"; if trained on EMNIST, they lean toward "D." This suggests that after the GPZ, what is reconstructed is no longer private content but priors from auxiliary data, further confirming the stripping of private information.
- VGG19 offers the best deployment efficiency: only 2.5% of parameters need to remain on the edge to reach the GPZ, whereas IR-152 requires 78%+; upgrading VGG from depth-26 to depth-30 halves the transmission payload with almost no increase in latency.

## Highlights & Insights

- Using "where to partition" as a defense dimension is more upstream and fundamental than "how to perturb," avoiding the constant push-and-pull on the privacy-utility curve. It follows a "change the question, not the answer" logic.
- The seamless connection between information-theoretic bounds and a computable engineering metric ($R_c^2$) makes the theory both explanatory and actionable. The "theory provides the pointer, pointer scans the layers" combination is highly practical.
- The Neural Vortex section provides a significant "Aha!" moment: increasing output-side entropy (label smoothing for a flatter softmax) paradoxically leads to intermediate-layer entropy reduction (tighter intra-class point clouds). This surface contradiction is naturally derived from the $(p_{ic}-1+\alpha)$ sign flip, providing a level of training dynamics analysis rare in MIA literature.
- Transferability: The $R_c^2$ probe could be applied to any scenario where making intermediate representations harder to invert is desirable, such as gradient leakage defense in Federated Learning or state hiding in MPC.

## Limitations & Future Work

- Verification was primarily on vision models; whether text and sequence models have a GPZ depends on whether patch/token information is retained at deeper levels. The "no GPZ" result for ViT suggests that language Transformers may be harder to partition into clear transition zones.
- For extremely simple data (MNIST/KMNIST), the GPZ shifts significantly later and narrows, suggesting that low-entropy edge cases may still require additional perturbation.
- The $R_c^2$ probe requires available labels and clear classes. In self-supervised or multi-task scenarios, defining a "class" is not obvious and requires extension to prototype or clustering perspectives.
- Systematic comparison with the synergy of active defenses (e.g., noisy IB, HE) has not been done. A composite defense of GPZ partition + lightweight perturbation might outperform either alone.

## Related Work & Insights

- **vs Information Bottleneck methods (Wang et al., 2021; Duan et al., 2023)**: IB explicitly penalizes $I(X;Z)$, but requires mutual information estimation and often drops accuracy; this work achieves comparable or stronger intrinsic defense without changing the loss or estimating MI.
- **vs Neural Collapse (Papyan et al., 2020)**: Neural Collapse is a post-hoc observation of geometric finality; Neural Vortex provides a controllable dynamic description during training linked to privacy bounds.
- **vs Shallow Perturbation defenses (Wang et al., 2022; Ding et al., 2024)**: Shallow noise/masking treats the symptoms, while moving the partition point to the decision side treats the cause; the two are complementary.


## Rating
- Novelty: To be evaluated
- Experimental Thoroughness: To be evaluated
- Writing Quality: To be evaluated
- Value: To be evaluated

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)
- [\[ICML 2026\] PipeSD: An Efficient Cloud-Edge Collaborative Pipeline Inference Framework with Speculative Decoding](pipesd_an_efficient_cloud-edge_collaborative_pipeline_inference_framework_with_s.md)
- [\[ICLR 2026\] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [\[NeurIPS 2025\] Model Inversion with Layer-Specific Modeling and Alignment for Data-Free Continual Learning](../../NeurIPS2025/ai_safety/model_inversion_with_layer-specific_modeling_and_alignment_for_data-free_continu.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)

</div>

<!-- RELATED:END -->
