---
title: >-
  [Paper Note] PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts
description: >-
  [AAAI 2026][LLM/NLP][Zero-Shot Anomaly Detection] PromptMoE shifts prompt learning from a monolithic paradigm to a compositional one. Through a visually-guided Mixture of Experts (MoE) mechanism, it dynamically assembles instance-adaptive normal/abnormal state prompts from a learnable semantic primitive bank, achieving state-of-the-art zero-shot anomaly detection (ZSAD) performance across 15 industrial and medical datasets.
tags:
  - AAAI 2026
  - LLM/NLP
  - Zero-Shot Anomaly Detection
  - CLIP
  - Mixture of Experts
  - Compositional Prompt Learning
  - Visually-Guided Routing
date: 2026-05-08
content_hash: 327bd406bb0ea4b1
---

# PromptMoE: Generalizable Zero-Shot Anomaly Detection via Visually-Guided Prompt Mixing of Experts

**Conference**: AAAI 2026
**arXiv**: [2511.18116](https://arxiv.org/abs/2511.18116)
**Code**: Unavailable
**Area**: Anomaly Detection / Vision-Language Models
**Keywords**: Zero-Shot Anomaly Detection, CLIP, Mixture of Experts, Compositional Prompt Learning, Visually-Guided Routing

## TL;DR

PromptMoE shifts prompt learning from a monolithic paradigm to a compositional one. Through a visually-guided Mixture of Experts (MoE) mechanism, it dynamically assembles instance-adaptive normal/abnormal state prompts from a learnable semantic primitive bank, achieving state-of-the-art zero-shot anomaly detection (ZSAD) performance across 15 industrial and medical datasets.

## Background & Motivation

Zero-Shot Anomaly Detection (ZSAD) aims to detect and localize anomalous regions in object categories unseen during training, with critical applications in industrial quality inspection and medical diagnosis. Methods based on vision-language models (VLMs) such as CLIP have shown promise, yet existing prompt engineering strategies face significant limitations:

**Single-prompt representation bottleneck**: Whether hand-crafted fixed prompts (WinCLIP) or learned single normal/abnormal prompt pairs (AnomalyCLIP), a single fixed prompt vector struggles to capture the diverse normal and abnormal patterns present in unseen categories.

**Static multi-prompt overfitting**: Naively increasing the number of learnable static prompts substantially raises the risk of overfitting to auxiliary data—the model tends to memorize specific pattern–prompt combinations in the training set rather than learning generalizable abstract concepts.

**Limitations of single dynamic mapping**: CoCoOp-style methods (AdaCLIP, VCP-CLIP) employ a single mapping network to dynamically generate prompts from visual instances, making it difficult to produce specialized prompts for specific fine-grained anomaly patterns.

**Generalization failure of deep prompts**: AnomalyCLIP has demonstrated that when static learnable prompts are inserted into intermediate layers of the text encoder, only shallow-layer insertion is effective; deeper layers instead degrade zero-shot detection on unseen categories.

Core insight: **Robust ZSAD requires compositional rather than monolithic prompt learning**—"learning how to compose" is more effective than "learning a complete prompt."

## Method

### Overall Architecture

PromptMoE is built upon a frozen CLIP (ViT-L/14@336px), with the **Visually-Guided Mixture of Prompts (VGMoP)** module as its core innovation. As illustrated in Figure 3:

1. The visual encoder extracts multi-layer patch features $\mathbf{F}_x^{(l)}$ and global features $\mathbf{F}_x^{cls}$ from the input image.
2. VGMoP takes visual features as input and dynamically generates instance-specific normal and abnormal text prompts $\mathbf{T}_n^{(l)}$ and $\mathbf{T}_a^{(l)}$.
3. Layer-wise similarities between aggregated text embeddings and patch features are accumulated to produce the anomaly map $\mathbf{M}$.

### Key Designs

#### 1. Mixed Text Prompt Structure

Two mixed text prompts are constructed for each visual instance:

$$\mathbf{T}_n = [\mathbf{S}_{\text{agg}}^n][\texttt{cls}][\mathbf{Q}_{\text{ctx}}]$$

$$\mathbf{T}_a = [\mathbf{S}_{\text{agg}}^n][\mathbf{S}_{\text{agg}}^a][\texttt{cls}][\mathbf{Q}_{\text{ctx}}]$$

where:
- $\mathbf{S}_{\text{agg}}^n \in \mathbb{R}^{M_n \times D}$: aggregated normal-state prompt ($M_n=5$)
- $\mathbf{S}_{\text{agg}}^a \in \mathbb{R}^{M_a \times D}$: aggregated abnormal-state suffix ($M_a=6$)
- $\mathbf{Q}_{\text{ctx}} \in \mathbb{R}^{M_q \times D}$: shared learnable context tokens ($M_q=8$)
- $[\texttt{cls}]$: embedding of the category name or the generic placeholder "object"

The abnormal prompt appends an anomaly suffix $\mathbf{S}_{\text{agg}}^a$ to the normal prompt, a design motivated by PromptAD.

#### 2. Visually-Guided MoE State Aggregation (Core of VGMoP)

For each layer $l \in \mathcal{I}$, the following procedure is executed independently (Figure 4):

**Cross-attention visual distillation**: Learnable state queries $\mathbf{q}^{(l)}$ actively distill state-relevant visual signals from patch features via cross-attention:

$$\mathbf{O}^{(l)} = \text{Softmax}\left(\frac{Q^{(l)}{K^{(l)}}^\top}{\sqrt{D}}\right)V^{(l)}$$

Average pooling over $\mathbf{O}^{(l)}$ yields the routing representation $\mathbf{r}^{(l)} = \text{mean}(\mathbf{O}^{(l)})$.

**Sparse routing**: $\mathbf{r}^{(l)}$ is fed into a layer-specific image-gated sparse router $G^{(l)}$ (a two-layer MLP: Linear–ReLU–Linear), producing routing logits over the expert prompt pool $\mathcal{E}^{(l)} = \{\mathbf{s}_j^{(l)} \in \mathbb{R}^{M \times D}\}_{j=1}^E$. The top-$k$ experts are selected and weighted aggregated:

$$\mathbf{S}_{\text{agg}}^{(l)} = \sum_{i=1}^{k} \mathbf{w}_i^{(l)} \mathbf{s}_{\text{top},i}^{(l)}, \quad \mathbf{w}^{(l)} = \text{Softmax}(\mathbf{z}_{\text{top}}^{(l)})$$

Normal and abnormal states use **independent expert pools** $\mathcal{E}_n$ and $\mathcal{E}_a$ to avoid negative transfer.

#### 3. Auxiliary Losses

**Load balancing loss** $\mathcal{L}_{\text{balance}}$: Encourages uniform expert utilization across the batch, preventing routing from collapsing into a fixed combination:

$$\mathcal{L}_{\text{balance}} = \alpha \sum_{l \in \mathcal{I}} \left(E \sum_{j=1}^{E} \left(\frac{1}{B}\sum_{i=1}^{B} \mathbf{p}_{i,j}^{(l)}\right)^2\right)$$

**Expert decoupling loss** $\mathcal{L}_{\text{decouple}}$: Promotes representational diversity within the expert pool via orthogonality constraints:

$$\mathcal{L}_{\text{decouple}} = \beta \sum_{l \in \mathcal{I}} \|\hat{\mathbf{S}}^{(l)}(\hat{\mathbf{S}}^{(l)})^T - \mathbf{I}_E\|_F^2$$

The two losses work synergistically: $\mathcal{L}_{\text{decouple}}$ ensures expert diversity, which is a prerequisite for effective load balancing (Figure 7).

### Loss & Training

The total loss comprises classification, segmentation, and auxiliary components:

$$\mathcal{L}_{\text{total}} = \underbrace{\text{BCE}(s, c)}_{\text{classification}} + \underbrace{\text{Dice}(\mathbf{M}, \mathbf{m}) + \text{Focal}(\mathbf{M}, \mathbf{m})}_{\text{segmentation}} + \underbrace{\mathcal{L}_{\text{balance}} + \mathcal{L}_{\text{decouple}}}_{\text{auxiliary}}$$

The anomaly score combines peak and global similarity: $s = \frac{1}{2}(\max(\mathbf{M}) + \text{Softmax}(\mathbf{F}_x^{cls} \mathbf{F}_T^{(\max(\mathcal{I}))\top}/\tau'))$

Training configuration:
- CLIP is fully frozen; only the VGMoP module is trained.
- Images are resized to 518×518; features are extracted from layers $\{6, 12, 18, 24\}$.
- 15 epochs, Adam optimizer, lr=0.001, with 3-epoch warmup.
- $E=8$ experts, top-$k=4$, $\alpha=0.01$, $\beta=0.005$.

## Key Experimental Results

### Main Results

Comprehensive evaluation on 15 datasets (7 industrial + 8 medical). Training is performed on MVTec AD; zero-shot inference is applied to the remaining 14 datasets.

| Dataset (Domain) | Metric | Ours | Prev. SOTA | Gain |
|-----------------|--------|------|------------|------|
| MVTec AD (Industrial) | I-AUROC | 93.8 | 92.0 (AdaCLIP) | +1.8 |
| VisA (Industrial) | I-AUROC | 85.0 | 84.5 (FAPrompt) | +0.5 |
| BTAD (Industrial) | I-AUROC | 93.4 | 92.0 (FAPrompt) | +1.4 |
| SDD (Industrial) | P-AUROC | 98.1 | 98.3 (FAPrompt) | −0.2 |
| HeadCT (Medical) | I-AUROC | 98.2 | 94.8 (FAPrompt) | +3.4 |
| HeadCT (Medical) | AP | 98.2 | 93.5 (FAPrompt) | +4.7 |
| Industrial Avg. | I-(AUC, AP) | (92.4, 93.4) | (91.7, 92.5) | +0.7/+0.9 |
| Industrial Avg. | P-(AUC, PRO) | (96.2, 89.2) | (96.2, 88.0) | 0/+1.2 |
| Medical I-level Avg. | (AUC, AP) | (97.4, 97.5) | (96.0, 95.5) | +1.4/+2.0 |

State-of-the-art results are achieved in both industrial and medical domains, with particularly strong generalization on unseen-domain medical datasets.

### Ablation Study

| Configuration | MVTec I-AUC | MVTec PRO | VisA I-AUC | VisA PRO |
|--------------|-------------|-----------|------------|----------|
| Static Prompt (baseline) | 91.7 | 82.0 | 82.4 | 88.0 |
| +Static Ensemble | 92.2 | 82.9 | 83.3 | 88.3 |
| +VGMoP (single layer, no aux. loss) | 93.1 | 83.3 | 84.1 | 89.0 |
| **PromptMoE (full)** | **93.8** | **83.2** | **85.0** | **89.2** |

**Additional ablation findings**:
- $\alpha=0$ (removing load balancing) → MVTec I-AUC drops to 92.1, confirming the indispensability of load balancing.
- Shared expert pool → I-AUC drops to 91.4, validating the necessity of the normal/abnormal separation design.
- Shared both (pool + cross-attention) → drops to 90.9, exhibiting the most severe negative transfer.
- Multi-layer features ($\{6,12,18,24\}$) outperform using only the last layer.

### Key Findings

1. **Dynamic composition >> static ensemble**: VGMoP yields substantial gains over Static Ensemble (MVTec +0.9 I-AUC), demonstrating that the core advantage stems from visually-guided dynamic composition rather than simply increasing the number of prompts.

2. **Normal/abnormal expert routing patterns are distinctly different** (Figure 6):
    - Normal state: routing consistently converges to a small set of core experts, indicating that the model learns generalizable semantic primitives of "normality."
    - Abnormal state: routing is highly dynamic and sparse, with different datasets activating different expert subsets, reflecting flexible anomaly composition capability.

3. **Synergistic effect of the two auxiliary losses** (Figure 7): Removing $\mathcal{L}_{\text{decouple}}$ causes expert representations to degenerate into redundancy, which in turn degrades $\mathcal{L}_{\text{balance}}$ and ultimately harms detection performance—expert diversity is a prerequisite for effective load balancing.

4. **Cross-domain generalization from industrial training to medical inference**: Trained solely on MVTec AD, the model achieves 98.2% I-AUROC on HeadCT, surpassing the second-best method by 3.4%, validating the success of compositional prompt learning in concept-level generalization.

## Highlights & Insights

1. **Paradigm shift from monolithic to compositional**: The paper redefines MoE from "selecting MLP layers to process tokens" to "selecting semantic primitives to construct prompts," representing an innovative application of MoE to prompt engineering.
2. **Query-driven visual distillation**: Rather than compressing visual features via average pooling (which discards critical local information), learnable queries actively distill state-relevant signals through cross-attention.
3. **Interpretability of normal/abnormal routing patterns**: The stable routing for normal states versus dynamic routing for abnormal states provides intuitive justification and interpretability.
4. **Implementation simplicity**: With CLIP fully frozen and only the lightweight VGMoP module trained, the model is trainable on an RTX 3090.

## Limitations & Future Work

1. Although the code is claimed to be available, no GitHub link is provided in the original paper, which may pose obstacles to reproduction.
2. Training is conducted solely on MVTec AD (with VisA used when evaluating on MVTec AD itself); the impact of auxiliary training set selection on performance is not sufficiently discussed.
3. The hyperparameter choices of $E=8, k=4$ for the MoE are relatively conservative. Table 6 shows that $E=16$ improves PRO but does not necessarily improve I-AUC, suggesting that the optimal configuration may vary by scenario.
4. PromptMoE does not surpass FAPrompt on SDD and certain medical pixel-level metrics, suggesting that compositional prompts may be less effective than dense dynamic prompts for certain specific anomaly patterns.
5. Only ViT-L/14@336px is evaluated; the effects of larger backbones (e.g., ViT-G) or more recent VLMs (e.g., SigLIP) are not explored.

## Related Work & Insights

- **AnomalyCLIP**: A pioneering work applying prompt learning to ZSAD, but limited by the generalization failures of static prompts.
- **AdaCLIP / VCP-CLIP**: Forerunners of dynamic prompt generation, but constrained by the specialization limitations of single mapping networks.
- **Switch Transformer / MoE**: The sparse activation mechanism of MoE is naturally suited to alleviating overfitting; this paper ingeniously transfers it to the prompt space.
- Insight: In prompt learning, *compositionality* is more important than *quantity* or *dynamism*—learning to compose a small number of high-quality primitives is more effective than learning many fixed prompts or relying on a single dynamic mapping.

## Rating

| Dimension | Score (1–5) | Notes |
|-----------|-------------|-------|
| Novelty | 5 | Paradigm innovation of MoE as a prompt composer; deep insight into normal/abnormal routing patterns |
| Technical Depth | 4 | VGMoP design is elegant; synergistic analysis of auxiliary losses is thorough |
| Experimental Thoroughness | 5 | 15 datasets, multi-dimensional ablations, and comprehensive expert activation analysis |
| Value | 4 | Frozen CLIP + lightweight module makes deployment friendly; impact of training set selection warrants further investigation |
| Writing Quality | 4 | Paradigm comparison figure (Figure 1) is intuitive; overall narrative is clear |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] CoPS: Conditional Prompt Synthesis for Zero-Shot Anomaly Detection](../../CVPR2026/llm_nlp/cops_conditional_prompt_synthesis_for_zero-shot_anomaly_detection.md)
- [\[AAAI 2026\] Soft Filtering: Guiding Zero-Shot Composed Image Retrieval with Prescriptive and Proscriptive Prompts](soft_filtering_guiding_zero-shot_composed_image_retrieval_with_prescriptive_and_.md)
- [\[AAAI 2026\] From Classification to Ranking: Enhancing LLM Reasoning for MBTI Personality Detection](from_classification_to_ranking_enhancing_llm_reasoning_capabilities_for_mbti_per.md)
- [\[ACL 2026\] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot NER](../../ACL2026/llm_nlp/diziner_disagreement-guided_instruction_refinement_via_pilot_annotation_simulati.md)
- [\[CVPR 2026\] Boosting Quantitive and Spatial Awareness for Zero-Shot Object Counting](../../CVPR2026/llm_nlp/boosting_quantitive_and_spatial_awareness_for_zero-shot_object_counting.md)

</div>

<!-- RELATED:END -->
