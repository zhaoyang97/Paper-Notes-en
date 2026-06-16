---
title: >-
  [Paper Note] Partitioning for Intrinsic Model Inversion Resistance in Collaborative Inference
description: >-
  [ICML 2026][AI Safety][Paper Note] This paper moves beyond the traditional defense paradigm of "adding noise/masking to shallow intermediate representations." From an information-theoretic perspective, it proves that in edge-cloud collaborative inference, the model should be partitioned at the layer where the representation undergoes a "feature $\to$ de
tags:
  - ICML 2026
  - AI Safety
date: 2026-05-08
content_hash: acf8d4c1a632dc4a
---
# Partitioning for Intrinsic Model Inversion Resistance in Collaborative Inference

**Conference**: ICML 2026  
**arXiv**: [2506.15412](https://arxiv.org/abs/2506.15412)  
**Code**: https://github.com/GoldenPartitionZone/GoldenPartitionZone  
**Area**: AI Safety / Collaborative Inference / Model Inversion Defense  
**Keywords**: Model Inversion Attack, Collaborative Inference, Partitioning Strategy, Information Entropy, Label Smoothing

## TL;DR
This paper moves beyond the traditional defense paradigm of "adding noise/masking to shallow intermediate representations." From an information-theoretic perspective, it proves that in edge-cloud collaborative inference, the model should be partitioned at the layer where the representation undergoes a "feature $\to$ decision" mutation (named the Golden Partition Zone, GPZ). The intra-class mean square radius $R_c^2$ is identified as the key variable for locating the GPZ and can be actively contracted through label smoothing training dynamics.

## Background & Motivation

**Background**: Collaborative Inference (CI) partitions a deep network into two segments: an edge part $f_{\text{edge}}$ and a cloud part $f_{\text{cloud}}$. The edge device uploads the intermediate representation $z = f_{\text{edge}}(x)$ to the cloud. This deployment model is widely used in drones, IoT, and private cloud inference. However, Model Inversion Attacks (MIA) can train a generator $g \approx f_{\text{edge}}^{-1}$ to reconstruct the original input $x$ from $z$, leading to sample-level privacy leaks.

**Limitations of Prior Work**: Existing MIA defenses almost exclusively focus on perturbing shallow $z$ (noise, masking, bottleneck layers, homomorphic encryption, etc.). The cost is either a sacrifice in downstream accuracy or the introduction of additional computational overhead, essentially struggling with the "privacy-utility" tradeoff.

**Key Challenge**: The authors argue that the question should not be about how to perturb. The fundamental question is: at which layer should the network be partitioned so that $z$ naturally and irreversibly loses input information before transmission? This shifts defense from "post-hoc patching" to the "partition position" itself.

**Goal**: (1) Theoretically characterize the relationship between the partition point and MIA difficulty; (2) Provide a computable and monitorable layer-wise metric to allow users to actively locate the optimal partition point; (3) Actively shape this metric during the training phase to enter the inversion-resistant zone earlier.

**Key Insight**: Conventional intuition suggests "deeper is safer." However, in ViT, even at the last layer, patch tokens retain fine-grained information for each sample, allowing successful inversion. In IR-152/ResNet-50, increasing depth can actually cause $I(X; Z)$ to decay more slowly due to skip connections. These counter-examples forced the authors to shift focus from "depth" to the "essential mutation of representation morphology."

**Core Idea**: Utilize the representation transition from "feature-level $\to$ decision-level" as a necessary condition for intrinsic defense. Use the intra-class mean square radius $R_c^2 = \frac{1}{N_c} \sum_{i:y_i=c} \|z_i - \mu_c\|^2$ as the sole computable proxy variable to locate this transition zone (GPZ), while actively contracting $R_c^2$ during training via methods like label smoothing.

## Method

### Overall Architecture

The paper follows a logical chain of "Theory $\to$ Metric $\to$ Training Dynamics $\to$ Experimental Validation." It first derives the lower bound of $H(X \mid Z)$, revealing that this bound is dominated by the global variance $\sigma_{\text{feat}}^2$ at the feature level, but shifts to the intra-class radius $R_c^2$ at the decision level (where it is typically much smaller), causing the lower bound to jump upward. This $R_c^2$ is then refined into a usable layer-wise probe. Finally, the training dynamics of $R_c^2$ are actively regulated using label distributions (termed the Neural Vortex), making the decision-layer $R_c^2$ smaller and more inversion-resistant.

### Key Designs

**1. GPZ Localization Criterion: $R_c^2$ Probe from the $H(X\mid Z)$ Lower Bound**

To turn the engineering question of "where to partition" into an observable and automated process, a scalar reflecting whether a representation has been "decisionized" must be calculated for each layer. Treating $z$ as a continuous variable, the authors use the maximum entropy principle and determinant-trace inequalities to obtain differential entropy upper bounds for two stages: the feature level $h(Z_{\text{feat}}) \le \frac{d}{2}\ln(2\pi e \sigma_{\text{feat}}^2)$ depends mainly on dimension $d$ and global variance $\sigma_{\text{feat}}^2$; the decision level (after class conditioning) becomes $h(Z_{\text{dec}} \mid Y=c) \le \frac{D}{2}\ln(2\pi e R_c^2/D)$, where the decisive variance term shifts from "global variance" to "intra-class mean square radius $R_c^2$." Substituting these into the mutual information identity $H(X\mid Z) \ge H(X) - h(Z) - \kappa_\Delta$ shows that when the representation enters the decision zone and $h(Z)$ drops significantly, the inversion resistance lower bound jumps. In practice, one simply scans $R_c^2$ across candidate layers and identifies the layer where an abrupt drop occurs, bypassing high-variance mutual information estimators like MINE.

**2. Neural Vortex: Training Dynamics for Active $R_c^2$ Contraction via Label Smoothing**

Locating the GPZ is not enough; the authors aim to further "pull down" $R_c^2$ at the decision layer during late training to further raise the inversion resistance bound. Writing the backpropagation step as $\Delta R_c^2 = -\frac{2\gamma}{N_c} \sum_{i\in c} (z_i - \mu_c)^\top \tilde g_i$ and substituting $\tilde g_i = J_i^\top (p_i - y_i)$, it can be decomposed into a "correct class pull term" $(p_{ic}-1)T_{\text{corr},i}$ and an "incorrect class interference term" $\sum_{k\ne c} p_{ik} T_{k,i}$. Under one-hot supervision, as $p_{ic} \to 1$, the pull term approaches zero, and $R_c^2$ stops decreasing. With Label Smoothing (LS), the correct class coefficient becomes $(p_{ic}-1+\alpha)$. Once $p_{ic} > 1-\alpha$, the sign flips, and geometrically $T_{\text{corr},i}$ also reverses, ultimately maintaining $\Delta R_c^2 < 0$, which continues to tighten the intra-class point cloud. The authors name this counter-intuitive coupling of "entropy increase at the output and entropy decrease at the intermediate stage" the Neural Vortex. Unlike simple IB regularization or post-hoc neural collapse observation, this is active regulation at the training dynamics level with almost no loss in downstream accuracy (LS+ actually showed slight gains in experiments).

**3. Bidirectional Stress Testing for Decision-Layer Inversion Resistance**

Defenses often fail against stronger attack models, so it is necessary to confirm that the GPZ is not a "paper tiger" supported by weak attacks. Stress tests were performed by adding "buffs" to both ends: the representation end used FFT residuals/concatenation, global normalization, and small NN modules with dropout to enrich the transmitted $z$; the attack end progressively inserted multi-head attention, Attention-as-Conv, SE, LSK, and MSCA between deconvolutional blocks, following a "shallow weak attention $\to$ deep strong decoupling" strategy, along with inverse IR-152 residual blocks. These enhancements serve to verify that the GPZ still suppresses inversion quality, establishing that decision-layer inversion resistance remains significant even against augmented attacks.

### Loss & Training

The target models were trained on 7 $64\times 64$ datasets, including CIFAR-10, FaceScrub, and KMNIST, using three label distributions: one-hot, LS+ ($\alpha=0.3$), and LS- ($\alpha=-0.05$, reverse smoothing for control). The inversion model follows deconvolutional architectures from Yang et al. (2019) and Zhang et al. (2023). Evaluation metrics include MSE / PSNR / SSIM / LPIPS (AlexNet weights), with MSE $<0.02$ used as the empirical threshold for "high-fidelity reconstruction."

## Key Experimental Results

### Main Results: Impact of Representation Layer on MIA Difficulty (IR-152, CIFAR-10)

| Partition Point | Representation Type | MSE (Test) | PSNR (Test) | Legible Reconstruction? |
|-----------------|---------------------|------------|-------------|-------------------------|
| Block 40        | Feature-level       | 0.018      | 22.17       | Yes                     |
| Block 48        | Residual accumulated| $<0.02$    | $\approx 22$| Yes                     |
| Block 50        | Decision-level (GPZ)| 0.057      | 17.22       | No                      |
| Block 30→39 (VGG19) | Feature $\to$ Decision | 0.066 → 0.137 | — | Significant degradation |

In IR-152, at Block 49, the spatial resolution is compressed to $4\times 4$, and the representation suddenly becomes decisionized, with MSE jumping from $<0.02$ to $0.057$. This is the source of the claim that "GPZ yields an average 4× higher MSE than shallow partitioning." The paper also notes that because ViT retains 256 patch tokens, no representation transition occurs, making it impossible to form a GPZ regardless of where it is partitioned.

### Ablation Study: Stability of GPZ under Representation/Attack Augmentation (IR-152, Block 50 vs Block 40)

| Configuration | Block 50 (GPZ) MSE | Block 40 (Feature) MSE | GPZ Gap Narrowed? |
|---------------|--------------------|------------------------|-------------------|
| Baseline Attack | 0.057 | 0.018 | — |
| Rep: Normalize+Dropout-Concat | 0.052 | 0.014 | No (Gap ~3.7×) |
| Attack: Attention-as-Conv+SE+LSK+MSCA | 0.051 | — | No |
| Attack + Rep Combo + Inversion-IR152 | 0.049–0.052 | 0.012 | No (Gap ~4×) |

### Key Findings

- True intrinsic defense is not "partitioning deep," but "partitioning after the representation transition." Residual connections and ViT patch tokens delay or eliminate this transition, making them nearly useless against MIA.
- Decision-level representations maintain an average 66% inversion resistance advantage over feature-level representations under dual attack/representation buffs, proving GPZ is not a fragile illusion.
- Data distribution determines the GPZ location: On FaceScrub, the GPZ is earlier and narrower; on KMNIST, due to many zero pixels, feature extraction persists deeper, shifting the GPZ to around Block 43. This aligns with the joint effect of $H(X)$ and $R_c^2$ in the lower bound.
- MIA models trained on MNIST to invert KMNIST tend toward "0," while those trained on EMNIST tend toward "D." This indicates that reconstructions after the GPZ are based on auxiliary data priors rather than private content, confirming private information has been stripped.
- VGG19 offers the best cost-to-performance: only 2.5% of parameters are needed on the edge to reach the GPZ, whereas IR-152 requires 78%+. Moving VGG from depth-26 to depth-30 introduces almost no latency but halves the transmission payload.

## Highlights & Insights

- Using "where to partition" as a defense dimension is more upstream and fundamental than "how to perturb," avoiding the constant grind on the privacy-utility curve. It feels like "changing the question rather than the answer."
- It bridges the gap between information-theoretic bounds and an engineering-computable metric ($R_c^2$), making the theory both explanatory and actionable. The "theory provides the pointer, and the pointer scans the layers" approach is highly practical.
- The Neural Vortex section is an "Aha!" moment: output-side entropy increase (label smoothing) leads to intermediate-layer entropy decrease (tighter intra-class clouds). This apparent contradiction is naturally derived from the $(p_{ic}-1+\alpha)$ sign flip. Such detailed training dynamics analysis is rare in MIA literature.
- Portability: The $R_c^2$ probe can be applied to any scenario where intermediate representations should be hard to invert, such as gradient leak defense in Federated Learning or hiding intermediate states in MPC. The strategy is to find the intrinsic transition layer via $R_c^2$ and then add selective perturbations.

## Limitations & Future Work

- Validated primarily on vision models; whether GPZ exists in text/sequence models depends on whether patch/tokens retain sample-level information in deep layers. The "no GPZ" result for ViT suggests Transformers for language might be harder to partition into clear transition zones.
- For extremely simple data (MNIST/KMNIST), the GPZ shifts significantly deeper and narrows, implying that low-entropy edge cases still require additional perturbation reinforcement.
- The $R_c^2$ probe requires available labels and clear categories. In self-supervised or multi-task scenarios, defining a "class" is non-obvious and needs extension to prototype or cluster perspectives.
- Systematic comparison with active defenses (like noisy IB or HE) is missing; a compound defense of GPZ partitioning + lightweight perturbation might be superior and is a worthwhile next step.

## Related Work & Insights

- **vs. Information Bottleneck methods (Wang et al., 2021; Duan et al., 2023)**: IB explicitly penalizes $I(X;Z)$ but requires mutual information estimation and often drops accuracy; this work fulfills intrinsic defense without altering the loss or estimating MI, relying on "partition point selection + LS."
- **vs. Neural Collapse (Papyan et al., 2020)**: Neural Collapse is a post-hoc observation of the geometric endstate; Neural Vortex provides a controllable dynamical description during training and links it to privacy bounds.
- **vs. Shallow Perturbation Defenses (Wang et al., 2022; Ding et al., 2024)**: Shallow noise/masking treats the symptom; moving the partition point to the decision side treats the cause. These methods are actually stackable.
- **vs. Antoniadis series (online learning enhancement)**: Both "shift the perspective," but this work focuses on "information flow morphology," which is closer to representation learning than algorithmic design.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] One Model to Translate Them All: Universal Any-to-Any Translation for Heterogeneous Collaborative Perception](one_model_to_translate_them_all_universal_any-to-any_translation_for_heterogeneo.md)
- [\[ICLR 2026\] Co-LoRA: Collaborative Model Personalization on Heterogeneous Multi-Modal Clients](../../ICLR2026/ai_safety/co-lora_collaborative_model_personalization_on_heterogeneous_multi-modal_clients.md)
- [\[NeurIPS 2025\] Model Inversion with Layer-Specific Modeling and Alignment for Data-Free Continual Learning](../../NeurIPS2025/ai_safety/model_inversion_with_layer-specific_modeling_and_alignment_for_data-free_continu.md)
- [\[AAAI 2026\] Privacy Auditing of Multi-Domain Graph Pre-Trained Model under Membership Inference Attack](../../AAAI2026/ai_safety/privacy_auditing_of_multi-domain_graph_pre-trained_model_under_membership_infere.md)
- [\[ICML 2026\] How Does Bayesian Sampling Help Membership Inference Attacks?](how_does_bayesian_sampling_help_membership_inference_attacks.md)

</div>

<!-- RELATED:END -->
