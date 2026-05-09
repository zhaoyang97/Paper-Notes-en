---
title: >-
  [Paper Note] Vision Transformers Need More Than Registers
description: >-
  [CVPR 2026][Self-Supervised Learning][Vision Transformer] This paper argues that dense feature artifacts in ViTs trained under label supervision, text supervision, and self-supervision share a common root cause: rather than a simple high-norm token problem, models learn to exploit background patches as global semantic shortcuts, driven by coarse-grained supervision combined with global attention. The authors accordingly propose LaSt-ViT, which replaces standard CLS aggregation with frequency-domain stability-guided selective aggregation, yielding consistent improvements in localization, segmentation, and open-vocabulary tasks across 12 benchmarks.
tags:
  - CVPR 2026
  - Self-Supervised Learning
  - Vision Transformer
  - Lazy Aggregation
  - Register Token
  - DINO
  - Dense Feature Alignment
date: 2026-05-08
content_hash: 4f7899f38fe6dca2
---

# Vision Transformers Need More Than Registers

**Conference**: CVPR 2026
**arXiv**: [2602.22394](https://arxiv.org/abs/2602.22394)
**Code**: [https://github.com/ChengShiest/LAST-ViT](https://github.com/ChengShiest/LAST-ViT)
**Area**: Self-Supervised Learning
**Keywords**: Vision Transformer, Lazy Aggregation, Register Token, DINO, Dense Feature Alignment

## TL;DR
This paper argues that dense feature artifacts in ViTs trained under label supervision, text supervision, and self-supervision share a common root cause: rather than a simple high-norm token problem, models learn to exploit background patches as global semantic shortcuts, driven by coarse-grained supervision combined with global attention. The authors accordingly propose LaSt-ViT, which replaces standard CLS aggregation with frequency-domain stability-guided selective aggregation, yielding consistent improvements in localization, segmentation, and open-vocabulary tasks across 12 benchmarks.

## Background & Motivation
ViTs have evolved well beyond classification backbones and now serve as general-purpose feature extractors for a broad range of downstream vision systems. However, when these features are applied to tasks requiring spatially aligned dense representations—such as localization, segmentation, and open-vocabulary detection—models frequently attend to incorrect regions.

Prior work has addressed this issue from different angles.
- Under label supervision, certain studies have noted that ViT dense features are insensitive to foreground content.
- Under text supervision, patch-text alignment in CLIP-style ViTs is often poor, degrading zero-shot dense prediction.
- Under self-supervision, the DINO family exhibits high-norm / artifact tokens that disrupt object discovery.

The authors argue that despite their surface differences, these phenomena likely reflect a single underlying mechanism manifesting differently across training paradigms. Existing remedies largely apply patch-level fixes within a single setting—for instance, adding register tokens to DINO to offload some anomalous global information—without explaining why these artifact tokens arise in the first place, nor why analogous phenomena persist under text or label supervision.

The paper's core motivation comprises three layers.
- First, the authors seek a unified, comparable definition of artifacts across supervised, text-supervised, and self-supervised settings, rather than allowing each community to operate with its own terminology.
- Second, they aim to determine whether register tokens address the root cause or merely relocate the symptom.
- Third, they seek a unified approach that suppresses artifacts during pretraining without depending on any particular training objective.

The authors' key observation is as follows: for a ViT receiving only image-level supervision, the CLS token must capture the overall image semantics but is never explicitly required to align with foreground semantics at the patch level. Under this setup, the path of least resistance is not to diligently extract global semantics from foreground patches, but rather to leverage global self-attention to diffuse sparse foreground information across many background patches and then aggregate global semantics from these "background shortcuts." The authors term this behavior **lazy aggregation**.

This explanation is compelling because it simultaneously accounts for two observations.
- Background patches tend to receive high patch scores, because the model treats them as depositories of global semantics.
- Register tokens mitigate but do not eliminate the problem, because they merely provide new storage locations for global information without altering the model's tendency toward shortcut aggregation.

In short, the problem this paper addresses is not "how to relocate anomalous tokens" but rather "how to prevent ViTs from learning to use background content as global semantic shortcuts in the first place."

## Method
The proposed method consists of two components: a unified diagnostic tool that confirms the existence of lazy aggregation, followed by a new CLS aggregation mechanism designed to ground the global representation in patches that are genuinely stable and foreground-relevant.

Rather than beginning by modifying the loss function, the authors first redefine the measurement framework. This step is essential, because without a unified metric it is impossible to analyze supervised, text-supervised, and self-supervised systems jointly.

### Overall Architecture
The overall pipeline can be summarized in four steps.

1. A standard ViT encoder produces all patch token representations $\mathbf{x}_{patch} \in \mathbb{R}^{N \times D}$.
2. A Patch Score is defined as the cosine similarity between each patch and the global CLS representation, measuring the degree to which the model treats that patch as a key carrier of whole-image semantics.
3. Observing that high-scoring patches frequently fall in the background rather than the foreground, the authors further introduce the Point-in-Box (PiB) metric, which directly records whether the highest-scoring patch lies within the annotated bounding box.
4. During training, CLS no longer aggregates from all patches indiscriminately. Instead, a per-channel stability score is computed for each patch, and only the Top-K most stable tokens per channel are used for aggregation, yielding a new CLS representation.

From an input–output perspective, LaSt-ViT does not rewrite the ViT backbone or introduce an additional complex supervision branch. Its contribution is more precisely characterized as "redefining which patches CLS draws from and according to what principle," making it transferable to supervised, text-supervised, and self-supervised pretraining paradigms.

### Key Designs

1. **Patch Score and Point-in-Box: Unified Diagnostic for ViT Artifacts**
   - *Function*: Uses CLS–patch cosine similarity as a unified probe to analyze whether the global semantics of a ViT fall on foreground or background regions.
   - *Mechanism*: The patch score is defined as $\mathcal{S}_p = \frac{\mathbf{x}_{patch} \cdot Q_{CLS}}{\lVert \mathbf{x}_{patch} \rVert_2 \lVert Q_{CLS} \rVert_2}$. A higher score indicates that the patch is closer to the global semantic representation.
   - *Design Motivation*: In a healthy dense feature space, the regions most representative of overall image semantics should typically coincide with the main object. If high-scoring patches consistently reside in the background, the model is taking shortcuts.
   - Further quantification is provided by PiB, which records whether the highest patch score falls within the foreground bounding box. This metric is sufficiently direct and comparable across architectures and pretraining paradigms.

2. **Lazy Aggregation Hypothesis: The Root Cause Is Background Shortcuts, Not Norm**
   - *Function*: Validates the artifact formation mechanism through training dynamics, masking experiments, and structural interventions.
   - *Mechanism*: The authors find that removing high-scoring patches does not harm classification accuracy—and occasionally produces marginal improvements—whereas removing low-scoring patches causes substantial accuracy drops. This indicates that high-scoring patches do not correspond to critical semantic regions.
   - *Design Motivation*: If the highest-scoring patches were truly key foreground locations, removing them should degrade performance. The opposite result implies these positions function more as redundant but highly correlated background shortcuts.
   - Further validation shows that ViT's PiB is already low very early in training and remains nearly constant throughout, even as classification accuracy continues to rise. This demonstrates that artifacts are a stable strategy adopted early in optimization, not a late-stage side effect.

3. **Frequency-Domain Stability Scoring: Identifying Foreground-Candidate Tokens via Channel Stability**
   - *Function*: Assigns a stability score to each channel of each patch token, estimating which features remain stable after low-pass filtering.
   - *Mechanism*: The intuition is that foreground regions exhibit more semantically consistent deep features, with smoother variation along the channel dimension, whereas background regions are semantically diverse and exhibit larger changes after low-pass filtering. A 1-D FFT is applied per patch along the channel dimension, multiplied by a Gaussian low-pass weight $\mathbf{g}$, and inverse-transformed to obtain a low-frequency version $\hat{\mathbf{x}}_{patch}$.
   - Formally, the stability score is $\mathbf{S}_{i,j} = \frac{\hat{\mathbf{x}}_{patch}[i,j]}{|\hat{\mathbf{x}}_{patch}[i,j] - \mathbf{x}_{patch}[i,j]| + \varepsilon}$.
   - *Design Motivation*: If a particular channel of a patch changes little after low-pass filtering, that channel's information is more "stable" and more likely to belong to continuously shared object semantics rather than scattered high-frequency background cues.

4. **Channel-wise Top-K Aggregation: Restricting CLS to the Most Reliable Local Evidence**
   - *Function*: Replaces uniform averaging over all patches and single attention pooling with per-channel selection of the K most stable patches, whose mean forms the corresponding channel of the new CLS representation.
   - *Mechanism*: For the $j$-th channel, the set of indices with the highest stability scores $\mathcal{I}_K(j)$ is identified, and the aggregated value is $\mathcal{Q}_{CLS}[j] = \frac{1}{K}\sum_{i \in \mathcal{I}_K(j)} \mathbf{x}_{patch}[i,j]$.
   - *Design Motivation*: A given patch need not be reliable across all channels; therefore, the authors perform channel-wise rather than token-wise selection. This preserves representational granularity while preventing global averaging from injecting substantial background noise into CLS.
   - *Distinction from registers*: Register tokens provide additional storage slots; LaSt-ViT directly constrains the information sources of CLS. The former offers "a new container for the shortcut," while the latter "reduces the opportunity to take the shortcut."

5. **Vote Count Visualization: Interpreting Where CLS Actually Attends**
   - *Function*: Counts how many channel-wise Top-K sets include each patch, yielding a vote count per token.
   - *Mechanism*: $v_i = \sum_{j=1}^{D} \mathbf{1}\{i \in \mathcal{I}_K(j)\}$. A higher vote count indicates that the patch is considered reliable evidence in more semantic channels.
   - *Design Motivation*: The authors aim to demonstrate that LaSt-ViT is not a black-box trick but genuinely shifts CLS attention back toward foreground regions. Visualization results show that high-vote patches are strongly aligned with foreground areas and adapt naturally to the amount of foreground evidence present.

### Loss & Training
This paper introduces no new supervision objective; the original training paradigm is preserved, with modifications confined to the CLS aggregation mechanism.

- Under fully supervised training, standard image classification supervision is retained.
- Under text-supervised training, CLIP-style image–text contrastive learning is retained.
- Under self-supervised training, DINO-style self-supervised training is retained.

LaSt-ViT therefore functions more as a universal aggregation module than as a task-specific loss. This design is practically advantageous because it concentrates the method's benefit on foreground semantic alignment without requiring additional annotations or complex multi-task training.

The authors also conduct two key validation experiments to further support their motivation.
- Increasing patch size to reduce the proportion of background tokens raises PiB from 0.44 to 0.52, but lowers classification top-1 accuracy from 62% to 55%. This confirms that coarse-grained supervision promotes background shortcuts, while also showing that simply coarsening patches is not a viable solution.
- Replacing global attention with window attention consistently raises PiB but consistently lowers top-1 accuracy. For example, applying window attention in all layers of ViT-Small raises PiB from 50.1 to 59.8, while top-1 drops from 72.3 to 63.9. This indicates that global dependencies both benefit recognition and facilitate background absorption of foreground semantics.

## Key Experimental Results
The experiments span three training paradigms and multiple dense downstream tasks, rather than focusing on a single benchmark. The authors' primary goal is to demonstrate that suppressing artifacts leads to synchronous improvements across multiple tasks, not merely to report another state-of-the-art number.

### Main Results
The most direct evidence that LaSt-ViT targets the root cause is provided by the Point-in-Box metric.

| Training Paradigm / Model | Baseline PiB | LaSt-ViT PiB | Gain |
|---|---:|---:|---:|
| Supervised ViT | 42.7 | 55.1 | +12.4 |
| DINO-v1 | 44.5 | 69.7 | +25.2 |
| CLIP ViT | 39.8 | 50.1 | +10.3 |
| ResNet reference | 68.4 / 71.1 / 53.9 | — | — |

This result is significant for three reasons.
- First, artifacts are not specific to any single training paradigm.
- Second, LaSt-ViT delivers substantial corrections to foreground alignment, not marginal gains.
- Third, the largest improvement is observed on DINO-v1, indicating that self-supervised ViTs are particularly susceptible to lazy aggregation in object-centric dense features.

The object discovery results, most directly relevant to the self-supervised setting, are as follows.

| Method | FPS | VOC07 CorLoc | VOC12 CorLoc | COCO CorLoc |
|---|---:|---:|---:|---:|
| DINO-seg | 29.4 | 45.8 | 46.2 | 42.1 |
| LOST | 29.4 | 61.9 | 64.0 | 50.7 |
| DINO + LaSt-ViT | 55.9 | 64.4 | 67.6 | 51.6 |

This table demonstrates that the gains from LaSt-ViT translate into concrete improvements in unsupervised object discovery, not merely more interpretable patch scores. Notably, LaSt-ViT outperforms both DINO-seg and LOST while operating at higher speed, without requiring additional heavy feature decomposition steps.

Improvements under fully supervised and text-supervised settings are also reported.

| Task | Baseline | LaSt-ViT | Gain |
|---|---:|---:|---:|
| VOC12 coarse segmentation, ViT-B/16 supervised | 22.3 mIoU | 32.8 | +10.5 |
| VOC12 coarse segmentation, ViT-S/16 supervised | 29.5 mIoU | 41.9 | +12.4 |
| VOC12 coarse segmentation, ViT-S/16 DINO | 47.7 mIoU | 55.1 | +7.4 |
| CLIP ViT-B/16 on VOC20 segmentation | 49.0 mIoU | 75.0 | +26.0 |
| F-ViT ViT-B/16 on OV-COCO novel AP50 | 117.5 | 133.3 | +15.8 |

Although these results span different tasks and metrics, they converge on a single conclusion: once CLS is no longer overly dependent on background shortcuts, ViT dense representations become substantially more useful across the board.

### Ablation Study
The ablations address two questions: what value of K is optimal, and whether LaSt-ViT's improvements reduce to a particular pooling bias.

K ablation under label-supervised training:

| Configuration | IN1K Top-1 | VOC07 CorLoc | VOC12 CorLoc | Notes |
|---|---:|---:|---:|---|
| Attention-Pool | 59.1 | 14.1 | 28.7 | Original aggregation |
| Mean-Pool | 64.3 | 15.3 | 29.6 | Simple average |
| LaSt-ViT, K=1 | 64.6 | 30.4 | 35.6 | Extremely selective |
| LaSt-ViT, K=7 | 64.8 | 32.1 | 37.6 | Optimal trade-off |
| LaSt-ViT, K=49 (Full) | 64.9 | 15.8 | 30.3 | Approaches full aggregation |

The most informative observation is that localization gains degrade noticeably as K grows large. The method's effectiveness is therefore not attributable to a change in pooling formula but rather to selectivity itself: at large K, background tokens are reintroduced, and the gains disappear.

K ablation under text-supervised training:

| Configuration | IN1K | VOC mIoU | COCO mIoU | Notes |
|---|---:|---:|---:|---|
| Attention-Pool | 55.8 | 10.7 | 3.3 | Original CLIP aggregation |
| Max-Pool | 53.1 | 71.9 | 12.2 | Naturally suppresses background, but classification degrades |
| LaSt-ViT, K=1 | 53.5 | 72.7 | 13.5 | Overly aggressive |
| LaSt-ViT, K=49 | 55.8 | 75.8 | 18.5 | Optimal |
| LaSt-ViT, K=98 | 56.2 | 75.9 | 18.0 | Near-optimal |
| LaSt-ViT, K=196 (Full) | 55.3 | 13.5 | 4.8 | Near-complete failure |

These results further demonstrate that LaSt-ViT is not simply performing sparser pooling. By filtering out unstable background tokens, it retains locally grounded evidence more valuable for dense prediction. The near-complete failure at K=196 confirms that full aggregation is itself the source of the problem.

### Key Findings
- Removing high-scoring patches causes negligible harm to classification, confirming that high-scoring patches are not critical foreground regions but rather background shortcuts exploited by CLS.
- Artifacts appear early in training, with PiB remaining nearly flat throughout—indicating that lazy aggregation is an early-acquired optimization habit, not a late-stage overfitting artifact.
- Register tokens eliminate high-norm phenomena, yet PiB remains low, confirming that high-norm is a symptom of lazy aggregation rather than its cause.
- Restricting global attention or reducing background tokens improves PiB but sacrifices classification accuracy, showing that the problem cannot be resolved by simply weakening model capacity; the aggregation mechanism must be reformed.
- Top-K selection degrades significantly as K approaches the full token set, confirming that selectivity is the operative factor.

## Highlights & Insights
- The paper's greatest strength lies not in proposing a complex new module but in unifying previously scattered artifact phenomena across multiple communities under the single explanatory framework of lazy aggregation—a framework simple enough to simultaneously account for anomalies in supervised, CLIP, and DINO settings.
- The authors reframe the question "does register work?" as "is the ViT still taking background shortcuts?" This perspective is more fundamental than focusing solely on high-norm tokens, as it directly concerns the mechanism by which representations are formed.
- The frequency-domain stability approach is an elegant design choice. Rather than explicitly supervising foreground detection, it exploits the statistical regularity that foreground semantics tend to be more consistent and background semantics more diverse, providing a natural selection criterion for CLS aggregation.
- Channel-wise Top-K selection is also worth noting. Many token selection methods score tokens holistically; this work highlights that different channels may correspond to different semantic subspaces, making per-channel selection a finer-grained alternative.
- Methodologically, this paper exemplifies the "diagnose first, then model, then validate uniformly" paradigm, offering a relatively complete chain of reasoning and a useful structural reference for analytical paper writing.

## Limitations & Future Work
- The theoretical explanation remains primarily empirical. Although the authors support lazy aggregation with extensive observations, no rigorous optimization-level derivation is provided to explain why CLS preferentially converges to background shortcuts early in training.
- The frequency-domain stability assumption presupposes that "foreground semantics are smoother and background semantics more diverse"—a generalization that typically holds for natural images but may not hold in texture-dominated or complex multi-instance scenes.
- The method centers on CLS aggregation and is thus most directly applicable to ViT variants that rely on an explicit global token. Whether it transfers to architectures without a dedicated CLS token or with heavy token mixing remains to be verified.
- Although the experiments are broad in scope, the core self-supervised results are primarily demonstrated on DINO-v1 object discovery. Coverage of stronger self-supervised backbones such as DINOv2, MAE, and iBOT would strengthen the case.
- The paper argues that registers are insufficient, but does not systematically investigate whether "registers + LaSt-ViT" are complementary. A joint configuration might yield a stronger practical system.

## Related Work & Insights
- **vs. Register**: The register approach provides additional global storage slots to reduce anomalous high-norm tokens in the patch map; this paper argues that this merely relocates the symptom without altering the background-shortcut formation mechanism. LaSt-ViT modifies the sources from which CLS aggregates, and is thus more directly remedial.
- **vs. CLIP dense alignment methods**: Many existing methods apply additional patch-text alignment or modify the final few attention layers at the downstream stage. This paper moves the intervention upstream to the pretraining representation formation stage, yielding a more fundamental and unified approach.
- **vs. token pruning / token selection**: Pruning addresses redundancy and efficiency, whereas this paper addresses whether the semantic sources are healthy. The two perspectives are complementary: LaSt-ViT could correct semantic grounding first, enabling more reliable subsequent token pruning.
- **Implications for self-supervised learning**: The object-centric performance of self-supervised ViTs is not determined solely by the teacher–student loss; how CLS aggregates from patches is equally important. Future DINO-style methods could treat aggregation bias as an independent design dimension.
- **Personal takeaway**: If a backbone's global representation is formed via unhealthy shortcuts, then failures on dense downstream tasks are merely symptoms. When analyzing foundation models, one should first scrutinize where global semantics originate, rather than focusing exclusively on downstream head design.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Proposes lazy aggregation as a unified root-cause explanation, connecting the register phenomenon and cross-paradigm artifacts into a coherent narrative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers 12 benchmarks across three supervision paradigms; the self-supervised main line could benefit from additional modern backbones.
- **Writing Quality**: ⭐⭐⭐⭐☆ Diagnostic logic is clear and experimental organization is convincing; some methodological intuitions remain stronger than formal derivations.
- **Value**: ⭐⭐⭐⭐⭐ Beyond a technique for improving dense features, this work provides an analytical framework for understanding how global semantics are formed in ViTs.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Zero-Ablation Overstates Register Content Dependence in DINO Vision Transformers](zero_ablation_overstates_register_content_dependence_in_dino_vision_transformers.md)
- [\[CVPR 2026\] Group-DINOmics: Incorporating People Dynamics into DINO for Self-supervised Group Activity Feature Learning](group_dinomics_incorporating_people_dynamics_into_dino_for_self_supervised_group_activity_feature_learning.md)
- [\[CVPR 2026\] DiverseDiT: Towards Diverse Representation Learning in Diffusion Transformers](diversedit_towards_diverse_representation_learning_in_diffusion_transformers.md)
- [\[CVPR 2026\] Robustness of Vision Foundation Models to Common Perturbations](robustness_of_vision_foundation_models_to_common_perturbations.md)
- [\[CVPR 2026\] Chain-of-Models Pre-Training: Rethinking Training Acceleration of Vision Foundation Models](com_pt_chain_of_models_pretraining.md)

<!-- RELATED:END -->
