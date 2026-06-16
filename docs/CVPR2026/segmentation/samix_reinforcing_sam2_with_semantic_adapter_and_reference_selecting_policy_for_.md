---
title: >-
  [Paper Note] SAMIX: Reinforcing SAM2 with Semantic Adapter and Reference Selecting Policy for Mix-Supervised Segmentation
description: >-
  [CVPR 2026][Segmentation][SAM2] SAMIX transforms the video "instance tracking" memory mechanism of SAM2 into cross-image "semantic tracking." By employing a lightweight semantic adapter and a reference selecting policy network trained via reinforcement learning, it selects a set of semantically similar reference images for each weakly-labeled or unla
tags:
  - CVPR 2026
  - Segmentation
  - SAM2
  - Reinforcement Learning
date: 2026-05-08
content_hash: e61632df8e6756a8
---
# SAMIX: Reinforcing SAM2 with Semantic Adapter and Reference Selecting Policy for Mix-Supervised Segmentation

**Conference**: CVPR 2026  
**paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_SAMIX_Reinforcing_SAM2_with_Semantic_Adapter_and_Reference_Selecting_Policy_CVPR_2026_paper.html)  
**Code**: https://github.com/Huster-Hq/SAMIX  
**Area**: Semantic Segmentation  
**Keywords**: Mix-supervised segmentation, SAM2, Semantic Adapter, Reinforcement Learning, Pseudo-label generation

## TL;DR
SAMIX transforms the video "instance tracking" memory mechanism of SAM2 into cross-image "semantic tracking." By employing a lightweight semantic adapter and a reference selecting policy network trained via reinforcement learning, it selects a set of semantically similar reference images for each weakly-labeled or unlabeled image as dense contextual prompts. This generates high-quality pseudo-labels to unify mixed-supervised training (mask/box/scribble/point/class/unlabeled), achieving SOTA performance on VOC, Cityscapes, Camouflaged Object Detection (COD), and polyp segmentation datasets.

## Background & Motivation
**Background**: Pixel-level annotation is expensive, prompting extensive research into semi-supervised and weakly-supervised segmentation (using box, scribble, point, or class labels). More advanced mixed-supervised segmentation aims to integrate various heterogeneous annotations into a unified training framework. A recent category of strong methods utilizes SAM/SAM2 directly as pseudo-label generators, feeding weak annotations as sparse spatial prompts (points/boxes) to SAM to generate pseudo-masks.

**Limitations of Prior Work**: This approach faces two major issues. First, in scenarios with blurry boundaries (e.g., medical lesions, camouflaged objects), sparse spatial prompts alone cannot accurately delineate targets. Second, images without any spatial annotation (class labels or unlabeled) are excluded from training because they cannot provide prompts, leading to low data utilization. Furthermore, both optimization-based and SAM-based methods **learn each sample in isolation**, neglecting the potential for collaborative learning across heterogeneous data.

**Key Challenge**: Sparse spatial prompts carry insufficient information, yet prompt-based models like SAM are dependent on them. To enable SAM2 to utilize data without prompts, a new form of prompting must be introduced—**dense contextual prompts**, which use a set of semantically related reference images with reliable masks to guide segmentation, rather than relying solely on the points or boxes of the current image.

**Key Insight**: The authors noted that the SAM2 memory mechanism inherently supports in-context learning, originally used to store historical information across video frames for instance tracking. They hypothesized that "inter-frame instance tracking" could be reinterpreted as "inter-image semantic tracking" by storing image-mask reference pairs of the same semantics in memory. However, a barrier exists: vanilla SAM2 features are trained to be "instance-distinct," meaning features of different instances within the same class are inconsistent, causing cross-image semantic matching to fail.

**Core Idea**: A lightweight semantic adapter is used to shift SAM2 features from "instance-distinct" to "semantically-consistent" (forming SA-SAM2). Then, a reference selecting network (SPNet), trained via reinforcement learning, actively selects a combination of references for each query image that maximizes pseudo-label quality. This allows even weakly-labeled or unlabeled data to produce reliable pseudo-labels.

## Method

### Overall Architecture
SAMIX consists of three collaborative components: **SA-SAM2** (the semantic-adapted SAM2 acting as a pseudo-label generator), **SPNet** (the reference selection policy network), and **Seg-model** (a Mean Teacher segmentation model). For each query image (with any or no annotation), SPNet autoregressively selects a set of semantically consistent reference images from a data pool containing all labeled and pseudo-labeled samples. These references are encoded by the SAM2 memory encoder into reference embeddings $\mathcal{S}_{ref}$, serving as **dense contextual prompts**. The query image features $\mathbf{F}$ (adjusted by the semantic adapter) absorb this context through memory attention $\mathtt{MemAtt}(\mathbf{F}, \mathcal{S}_{ref})$, and optionally fuse with sparse spatial prompts $\mathbf{P}$ derived from weak labels. Finally, the mask decoder outputs the segmentation mask and confidence:

$$\mathcal{M}, s = \mathtt{MaskDec}\big(\mathtt{MemAtt}(\mathbf{F}, \mathcal{S}_{ref}), \mathbf{P}\big)$$

The high-quality pseudo-labels then supervise the Seg-model. Since SPNet has no ground truth (GT) for training, the authors model "reference selection" as a reinforcement learning problem driven by verifiable rewards. In the entire framework, SAM2 is frozen, and only the semantic adapter and SPNet are trained.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Query Image + Weak/No Label"] --> S["SPNet Autoregressive Ref Selection<br/>HGP Filtering + Cosine Probability"]
    P["Data Pool<br/>(image-mask pairs)"] --> S
    S -->|Ref Set as Dense Contextual Prompt| C["SA-SAM2: Semantic Adapter<br/>Instance-Distinct → Semantically-Consistent"]
    A -->|Weak labels to Sparse Spatial Prompts| C
    C --> M["Pseudo-Label M*"]
    M --> T["Mean Teacher Seg-model<br/>Mixed-Supervised Collaborative Training"]
    M -->|Verifiable Rewards rα/rβ/rγ| R["GRPO Reinforcement Learning<br/>Update SPNet"]
    R -.-.-> S
```

### Key Designs

**1. Semantic Adapter: Converting SAM2 from Instance Tracking to Semantic Tracking (SA-SAM2)**

The vanilla SAM2 image encoder (Hiera, a 4-stage hierarchical ViT) is trained to distinguish instances in videos, resulting in "instance-distinct" features. Consequently, query features $\mathbf{F}$ and reference embeddings $\mathcal{S}_{ref}$ may not reside in the same semantic space, causing $\mathtt{MemAtt}(\mathbf{F}, \mathcal{S}_{ref})$ to distort. The authors insert an AdaptFormer adapter (down-projection—ReLU—up-projection) near the MLP of each ViT block in a residual manner to fine-tune features toward semantic alignment:

$$\tilde{x} = \mathtt{ReLU}(x \cdot \mathbf{W}_{down}) \cdot \mathbf{W}_{up}$$

where $\mathbf{W}_{down}\in\mathbb{R}^{d\times\hat{d}}$, $\mathbf{W}_{up}\in\mathbb{R}^{\hat{d}\times d}$, and $\hat{d}\ll d$ (set to 384) ensures parameter efficiency (~10M trainable parameters). This step is the foundation of the framework; without semantic consistency, using reference images as dense prompts is ineffective. Ablation studies show that adding SPNet without the adapter yields minimal gains. Notably, the ability to activate semantic consistency with a lightweight adapter suggests this capability is latent within SAM2.

**2. SPNet: Autoregressively Selecting Reference Sets for Each Query**

Meaningful reference selection is critical for pseudo-label quality. The authors first narrow down candidates using the **Hierarchical Guidance Principle (HGP)**: only samples with supervision intensity finer than the query image are retained (e.g., if the query is a point label, candidates are limited to mask/box/scribble labels). The intuition is that finer annotations provide more reliable masks as "teachers," allowing strong supervision to guide weak supervision across heterogeneous data. Each sample in the pool stores a representation pair $\{E, e\}$: spatial memory embedding $E$ and compact semantic embedding $e=\mathtt{AvgPool}(E)+\mathtt{MaskPool}(E,M)$. SPNet is a two-stage Transformer: the encoder models global relations among candidate embeddings, and the decoder, initialized with the query embedding $f=\mathtt{AvgPool}(\mathbf{F})$, **autoregressively** predicts $T$ selection tokens. Each step considers previously selected references to ensure complementarity. At step $t$, selection probability for candidate $k$ is normalized via cosine similarity:

$$p_{t,k} = \frac{\cos(\mathbf{d}_t, \mathbf{e}_k)}{\sum_{k=1}^{K}\cos(\mathbf{d}_t, \mathbf{e}_k)}$$

Compared to fixed-rule selection (RGB/DINO/feature similarity), autoregressive modeling captures inter-sample complementarity, yielding more diverse and informative reference combinations.

**3. Verifiable Reward + GRPO: Training SPNet with Reinforcement Learning**

SPNet lacks GT labels for supervision, so reference selection is modeled as policy learning and trained via Group Relative Policy Optimization (GRPO). For each query, $N$ diverse reference sets $\{\mathcal{S}_{ref,i}\}$ are sampled from SPNet using stochastic beam search. Each set is fed into SA-SAM2 to produce a mask $M_i$ and confidence $s_i$, which are scored using three **verifiable rewards**: quality reward $r_{\alpha,i}=s_i$ (confidence), consistency reward $r_{\beta,i}=\mathrm{Dice}(M_i, M^*_{seg})$ (Dice similarity with Seg-model output), and supervision reward $r_{\gamma,i}=-L_{typ}(M_i, Y_{typ})$ (supervision loss against available weak labels, ignored for unlabeled images). Rewards are normalized within the group and summed to calculate the advantage $A_i = A_{\alpha,i}+A_{\beta,i}+A_{\gamma,i}$ for the GRPO clipping objective. Finally, the set with the highest advantage generates the final pseudo-label $\mathcal{M}^*$, which is momentum-updated into the data pool. This "sampling-evaluating-selecting" loop allows SPNet to explore reference combinations that maximize pseudo-label quality actively.

**4. Collaborative Training with Mean Teacher**

The pseudo-labels from SA-SAM2 are used to train a standard segmentation model. The authors utilize a Mean Teacher paradigm to coordinate Seg-model with SA-SAM2/SPNet: Seg-model predicts $M^*_{seg}$ for augmented views, while its EMA copy produces $M_{seg}$ for original images. Seg-model is constrained by three losses: a supervision loss $L_{typ1}$ tailored to the annotation type (skipped for class/unlabeled samples), a consistency regularization $L_{reg}$ between $M^*_{seg}$ and $M_{seg}$, and a pseudo-label loss $L_{pseudo}$ provided by SA-SAM2 (skipped for mask samples). Simultaneously, the semantic adapter is optimized via $L_{pseudo}$ and supervision losses. The output of Seg-model flows back as the reference for SPNet’s consistency reward $r_\beta$, creating a closed loop: better pseudo-labels $\to$ better Seg-model $\to$ more accurate rewards $\to$ better reference selection. This plug-and-play design allows SAMIX to work with various segmentation models (e.g., Mask2Former, Polyp-PVT, PFNet).

### Loss & Training
SAM2 is fully frozen; only the semantic adapter (AdaptFormer, ~10M parameters) and SPNet are trained. Training uses AdamW with a learning rate of $1\times10^{-4}$. Reference set length $T=4$, GRPO sampling size $N=4$, and adapter projection dimension $\hat{d}=384$. SPNet uses 3 Transformer layers for both encoder and decoder. Training is conducted on 4×NVIDIA 4090.

## Key Experimental Results

### Main Results
On four datasets (VOC 2012, Cityscapes, COD: CAMO/COD10K, Polyp: IPS), SAMIX achieves SOTA across standardized data settings (utilizing mask/box/scribble/point/class/unlabeled). With only 5% full annotations, it approaches the fully supervised upper bound.

| Dataset / Metric | Full Supervision | MixSegNet | SAM-COD | WISH | **SAMIX** |
|--------|------|------|------|------|------|
| VOC val mIoU ↑ | 77.3 | 68.0 | 71.3 | 71.8 | **75.9** |
| Cityscapes mIoU ↑ | 79.4 | 72.0 | 74.5 | 75.0 | **78.3** |
| COD10K $S_m$ ↑ | 80.0 | 73.1 | 76.4 | 75.0 | **79.1** |
| COD10K $M$ ↓ | 4.0 | 6.8 | 5.0 | 5.4 | **4.1** |
| IPS mDice ↑ | 83.3 | 72.5 | 75.9 | 76.6 | **80.1** |

(Under identical settings compared to the second-best WISH, SAMIX gains +4.1 mIoU on VOC and +3.3 mIoU on Cityscapes; on class/unlabeled samples, it leads MixSegNet by 8.4%, i.e., 71.1 vs 62.7.)

### Ablation Study
The two main components are indispensable and highly complementary (Metrics for COD10K use $S_m$):

| Adapter | SPNet | VOC mIoU | Cityscapes mIoU | COD10K $S_m$ |
|------|------|------|------|------|
| ✘ | ✘ | 66.5 | 69.3 | 71.8 |
| ✔ | ✘ | 72.5 | 75.7 | 75.3 |
| ✘ | ✔ | 68.7 | 71.0 | 73.9 |
| ✔ | ✔ (SAMIX) | **75.9** | **78.3** | **79.1** |

Comparison of reference selection strategies (SPNet vs. Heuristics) and increments from HGP and Momentum Update (MU):

| Selecting Policy | VOC mIoU | Cityscapes mIoU | COD10K $S_m$ |
|------|------|------|------|
| Random | 70.4 | 73.6 | 72.9 |
| Similarity (DINO) | 72.0 | 75.2 | 75.9 |
| Similarity sim(F,E) | 72.7 | 75.8 | 77.0 |
| SPNet w/o HGP | 74.4 | 77.0 | 78.2 |
| SPNet w/o MU | 75.3 | 77.9 | 78.4 |
| **SPNet (Ours)** | **75.9** | **78.3** | **79.1** |

Stacking the three rewards ($r_\alpha \to r_\alpha r_\beta \to$ all) increased VOC mIoU from 73.5 to 75.0 and finally 75.9, demonstrating synergistic benefits.

### Key Findings
- **Semantic Adapter is the Foundation**: Adding SPNet alone (68.7) barely competes with adding the adapter alone (72.5). If features remain "instance-distinct," even optimal reference selection fails to match them. The leap to 75.9 requires both.
- **Learned Selection > Any Heuristic**: Even the strongest heuristic (feature similarity sim(F,E) at 72.7) lags significantly behind SPNet (75.9), confirming the limitations of independent sample selection and ignoring inter-sample relationships.
- **Adapter Selection**: AdaptFormer achieves the best results (75.9 on VOC) with minimal parameters (10M), outperforming Naive Adapter, LoRA, and Med-SA. This suggests SAM2’s semantic consistency only requires lightweight activation.
- **Data Utilization**: The most significant improvements occur on class-only and unlabeled data where SAM-based methods previously struggled, validating that "dense contextual prompts" solve the "data without prompts" bottleneck.

## Highlights & Insights
- **Ingenious Mechanism Reuse**: Reinterpreting SAM2's video memory mechanism for cross-image semantic tracking using a lightweight adapter is an elegant example of "new interpretation of existing modules," requiring almost no structural overhead.
- **Dense Contextual Prompts**: Moving beyond the point/box prompt mindset to use semantically similar reference images as prompts unlocks the potential of unlabeled data, a concept transferable to other SAM-based weakly-supervised tasks.
- **RL for "Unsupervised Retrieval"**: Since reference selection lacks supervision signals, modeling it as a policy problem with verifiable rewards (Confidence + Dice + Loss) and GRPO provides a template for any sub-problem involving data/sample selection.
- **Closed-loop Self-Improvement**: Pseudo-label quality, Seg-model performance, reward accuracy, and reference selection quality form a positive feedback loop.

## Limitations & Future Work
- The authors acknowledge that after training on specific datasets, SA-SAM2 and SPNet exhibit **limited cross-domain generalization**, which requires larger-scale training data to mitigate.
- Complexity: The training overhead is non-trivial, as each query image requires $N$ sets of GRPO samples and corresponding SA-SAM2 inference passes to calculate rewards. Training time/memory costs were not detailed.
- Scaling the data pool to store $\{E, e\}$ for all samples may lead to high retrieval and storage costs in large-scale scenarios, which was not fully discussed.
- HGP assumes "finer labels = more reliable masks," but if fine labels are noisy, hierarchical guidance might propagate errors. Robustness analysis on this aspect is missing.

## Related Work & Insights
- **vs. SAM-COD / WISH (SAM-based Mixed Supervision)**: These methods rely on sparse spatial prompts and process samples in isolation. SAMIX introduces dense contextual prompts and cross-sample collaboration, filling the gap for promptless data and blurry boundaries.
- **vs. MixSegNet / MixPolyp (Optimization-based Mixed Supervision)**: These utilize customized losses for joint optimization. SAMIX follows a "unified pseudo-label generator" route, converting heterogeneous labels into mask supervision signals, offering greater flexibility.
- **vs. Pure Similarity Retrieval (DINOv2 / RGB / sim(F,E))**: These are heuristic-based and select samples independently. SPNet uses autoregressive modeling and RL to optimize based on inter-selection relationships and pseudo-label quality, representing an upgrade from "rule-based retrieval" to "learnable selection."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Innovative reuse of SAM2 video memory and the use of RL with verifiable rewards for reference selection.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Solid evidence across 4 datasets, multiple supervision configurations, and extensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic chain (motivation-conflict-method), though some notation variants require cross-referencing.
- Value: ⭐⭐⭐⭐⭐ Significantly improves utilization of heterogeneous labels; plug-and-play design offers high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Correspondence as Video: Test-Time Adaption on SAM2 for Reference Segmentation in the Wild](../../ICCV2025/segmentation/correspondence_as_video_test-time_adaption_on_sam2_for_reference_segmentation_in.md)
- [\[CVPR 2026\] Reinforcing Video Object Segmentation to Think before it Segments](reinforcing_video_object_segmentation_to_think_before_it_segments.md)
- [\[CVPR 2026\] Frequency-Aware Affinity for Weakly Supervised Semantic Segmentation](frequency-aware_affinity_for_weakly_supervised_semantic_segmentation.md)
- [\[CVPR 2026\] Leveraging Class Distributions in CLIP for Weakly Supervised Semantic Segmentation](leveraging_class_distributions_in_clip_for_weakly_supervised_semantic_segmentati.md)
- [\[CVPR 2026\] From Softmax to Dirichlet: Evidential Learning for Semi-supervised Semantic Segmentation](from_softmax_to_dirichlet_evidential_learning_for_semi-supervised_semantic_segme.md)

</div>

<!-- RELATED:END -->
