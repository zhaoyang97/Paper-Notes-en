---
title: >-
  [Paper Note] LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations
description: >-
  [ICML 2026][Multimodal VLM][Incomplete Multimodal Learning] The authors reformulate multimodal action quality assessment with "missing modalities during training" as a "LLM-based conditional sequence-to-score reasoning"…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Incomplete Multimodal Learning"
  - "LLM Reasoning"
  - "Action Quality Assessment"
  - "Mask-Aware Fusion"
  - "Token-level Regularization"
date: 2026-05-08
content_hash: 85f3990925f87e24
---

# LIMSSR: LLM-Driven Sequence-to-Score Reasoning under Training-Time Incomplete Multimodal Observations

**Conference**: ICML 2026  
**arXiv**: [2605.00434](https://arxiv.org/abs/2605.00434)  
**Code**: https://github.com/XuHuangbiao/LIMSSR  
**Area**: Multimodal VLM / Incomplete Multimodal Learning / Action Quality Assessment  
**Keywords**: Incomplete Multimodal Learning, LLM Reasoning, Action Quality Assessment, Mask-Aware Fusion, Token-level Regularization

## TL;DR
The authors reformulate multimodal action quality assessment with "missing modalities during training" as a "LLM-based conditional sequence-to-score reasoning" problem. By using prompts and special tokens, the LLM is guided to complete missing semantics without full data supervision. Combined with mask-aware dual-path fusion to suppress hallucination, the method outperforms SOTA models that rely on complete training data across three AQA datasets.

## Background & Motivation

**Background**: In real-world scenarios, multimodal data often lack certain modalities—sensor failures, privacy filtering, and collection costs can result in missing video/audio/flow data. Academic research on Incomplete Multimodal Learning (IML) mainly follows two lines: (a) reconstruction-based (ActionMAE, IMDer, GAIN, DMVG) directly reconstruct missing modality features; (b) distillation/prior-based (CorrKD, MoMKE, MCMoE) use complete modalities as teachers for distillation or priors.

**Limitations of Prior Work**: Both approaches implicitly assume a "god's-eye view"—complete modalities must be available during training as reconstruction targets or distillation teachers. However, in real data collection, missingness is inherent (e.g., some subjects never recorded audio). When training data itself is incomplete, there is no GT for reconstruction or teacher for distillation, causing the IML framework to collapse.

**Key Challenge**: When modalities are missing during training, how can the missing semantics be "imagined out of thin air"? Traditional reconstruction-distillation routes require "complete-incomplete" pairs, but such pairs do not exist; simply zero-filling leads the model to treat "missingness" as noise, hurting the main task. A mechanism is needed to "infer" missing semantics without paired supervision.

**Goal**: (i) Formalize the more realistic setting of "incomplete observations during training"; (ii) Propose a framework that infers missing semantics without relying on complete training data; (iii) Validate on long-video Action Quality Assessment (AQA), a task highly dependent on multimodality.

**Key Insight**: The authors observe that LLMs are not only sequence models but also possess vast world knowledge and reasoning ability—given observable modalities and a description of missing structure, LLMs should be able to "fill in the blanks" and infer semantic representations for missing parts, without pixel-level reconstruction.

**Core Idea**: Reformulate incomplete multimodal learning as "conditional sequence reasoning"—use prompts to describe the task and missing status, missing tokens as placeholders, and fusion tokens for aggregation, enabling the LLM to infer latent semantics under missing modalities. Mask-aware gating calibrates the uncertainty of reasoning.

## Method

### Overall Architecture
Given a sample $(\mathbf{X} \odot \boldsymbol{m}, \boldsymbol{m}, y)$ ($\boldsymbol{m}\in\{0,1\}^M$ is the missing mask), LIMSSR proceeds in three steps: (1) Context Construction $\Phi_{in}$ concatenates instruction prompt, visible modality features $\tilde{\mathbf{X}}^m$, missing token placeholder sequences, and fusion tokens into a unified embedding $\mathbf{Z}_{in}$; (2) LLM Reasoning $\mathbf{H}_{out} = \mathrm{LLM}(\mathbf{Z}_{in})$ simultaneously infers missing semantics and performs multimodal fusion; (3) Mask-Aware Dual-Path Aggregation $\Psi_{agg}$ fuses high-level semantic and low-level cross-modal paths with mask-weighted fusion to output the action quality score $\hat{y}$. Modality features are extracted using frozen VST/AST/I3D for video/audio/flow, projected into the LLM input space via two conv layers.

### Key Designs

1. **Prompt-Guided Context-Aware Modality Imputation (PCMI)**:

    - **Function**: Elevates missing modalities from "zero vectors" to "latent variables to be inferred," allowing the LLM to treat missing positions as fillable tokens for reasoning.
    - **Mechanism**: Each modality $m$ is wrapped with boundary tokens `<m_start>, <m_end>`. For visible modalities, $\tilde{\mathbf{X}}^m$ is placed inside; for missing modalities, $T$ repeated learnable `<missing_m>` embeddings are used. A task prompt explicitly describes visible and missing modalities: "Given the available {avail} features... The {miss} modality is missing. Based on the available modalities, please infer and reconstruct the useful latent representations for the missing {miss} modalities at the designated positions." After LLM output, hidden states at missing token positions $\mathbf{H}_{miss}^m = \mathrm{LLM}(\mathbf{Z}_{in})|_{\text{positions of }\mathbf{E}_{miss}^m}$ are extracted as inferred missing representations.
    - **Design Motivation**: Traditional zero-filling causes missing signals to be "buried" in attention; MissRAG/TAMML use RAG or text bridging, requiring extra retrieval or pre-alignment. PCMI encodes missing structure directly into the sequence, making LLM's next-token reasoning naturally suitable—"guessing the next token" and "inferring missing latent" are mathematically equivalent.

2. **LLM-Driven Multidimensional Representation Fusion (LMRF)**:

    - **Function**: Distills cross-modal information into $K$ fusion slots without disrupting the LLM output space, yielding a compact task-relevant representation.
    - **Mechanism**: Appends $K$ special tokens `<emb_dim_1>, ..., <emb_dim_K>` at the end of the prompt as "information slots," and explicitly instructs the LLM to "integrate and enhance all multimodal features for action quality assessment. Output the fused multi-dimensional feature representations at the designated feature dimension positions." The LLM's final layer outputs at these positions $\mathbf{H}_{fusion} = \{\boldsymbol{h}_1, \dots, \boldsymbol{h}_K\}$ are assumed to encode different evaluation dimensions (e.g., difficulty, execution, artistry). Learnable role weights $\boldsymbol{w}_{role}$ compute $\boldsymbol{z}_{main} = \sum_k \mathrm{Softmax}(\boldsymbol{w}_{role})_k \cdot \boldsymbol{h}_k$ as the main fusion vector.
    - **Design Motivation**: Mean-pooling LLM outputs destroys long-sequence generation ability; inspired by BERT's `[CLS]` but generalized to multiple dimensions, the LLM learns to "pack different aspects into different slots," which is more structured than pooling and more interpretable than attention heads.

3. **Mask-Aware Dual-Path Aggregation (MDA)**:

    - **Function**: Uses the LLM reasoning path for high-level semantics and the cross-modal attention path for low-level features, dynamically calibrating the reliability of both paths based on the missing mask to avoid hallucination under severe missingness.
    - **Mechanism**: Path 1 (Uncertainty-Calibrated Reasoning)—compute gating $\boldsymbol{g} = \sigma(\mathrm{MLP}_{gate}([\boldsymbol{z}_{main}, \boldsymbol{m}]))$ and residual $\boldsymbol{\delta} = \mathrm{MLP}_{res}([\boldsymbol{z}_{main}, \boldsymbol{m}])$, yielding refined representation $\tilde{\boldsymbol{z}}_{main} = \boldsymbol{z}_{main} + \boldsymbol{g}\odot \boldsymbol{\delta}$. Path 2 (Cross-Modal Pattern Recovery)—temporal pooling of LLM hidden states at each modality yields $\boldsymbol{h}_v, \boldsymbol{h}_a, \boldsymbol{h}_f$, stacked and self-attended to get $\mathbf{Z}_{attn}$; weighted by $\alpha_{m_j} = \boldsymbol{m}_j \cdot 1 + (1-\boldsymbol{m}_j)\cdot \gamma_{m_j}$ according to availability ($\gamma_m = \sigma(\lambda_m)$ is a learnable modality-level confidence), finally $\boldsymbol{z}_{aux} = \sum_m \alpha_m (\boldsymbol{z}_{attn}^m \odot \mathcal{G}(\mathbf{H}_{stack})^m)$. The two paths are fused for the final score.
    - **Design Motivation**: Relying solely on LLM reasoning risks hallucination under severe missingness; relying only on statistical aggregation lacks high-level semantics. Mask-aware adaptive fusion of both paths gives the model a "meta-cognitive" ability to assess its own confidence, which is crucial for extreme missing cases.

### Loss & Training

In addition to the main regression loss, the authors introduce: (1) Consistency Learning to enforce agreement between the two paths, encouraging mutual verification; (2) Token-Level Metric Regularization to ensure different fusion tokens learn distinct feature dimensions (avoiding collapse), specifically maximizing off-diagonal distances in the token similarity matrix with a regularization term; (3) Optional LoRA fine-tuning for the LLM backbone to avoid full-parameter training.

## Key Experimental Results

### Main Results (FS1000, 7-class, Spearman ↑ / MSE ↓, T-Miss indicates missing modalities during training)

| Method | T-Miss | {v,f} | {v,a} | {v} | {a} | Average | {v,f,a} |
|------|--------|-------|-------|------|------|---------|---------|
| ActionMAE | ✗ | 0.775/24.66 | 0.766/64.13 | 0.761/50.64 | 0.458/41.66 | 0.651/38.18 | 0.809/17.96 |
| GCNet | ✗ | 0.730/25.56 | 0.740/23.86 | 0.696/26.67 | 0.442/39.40 | 0.610/28.62 | 0.764/21.82 |
| MoMKE | ✗ | 0.798/18.86 | 0.805/23.88 | 0.785/37.96 | 0.499/27.53 | 0.668/26.08 | 0.819/16.85 |
| MCMoE | ✗ | 0.845/12.66 | 0.882/11.85 | 0.845/13.64 | 0.615/16.72 | 0.782/15.37 | 0.881/11.53 |
| **LIMSSR** | **✓** | **0.854/12.51** | **0.891/10.54** | **0.853/12.50** | **0.687/15.51** | **0.789/14.08** | **0.891/10.44** |

| Δ vs SOTA | {v,f} | {v,a} | {v} | {a} | Average | {v,f,a} |
|-----------|-------|-------|------|------|---------|---------|
| ΔSpearman | ↑1.1% | ↑1.0% | ↑0.9% | ↑11.7% | ↑0.9% | ↑1.1% |
| ΔMSE | ↓1.2% | ↓11.1% | ↓8.4% | ↓7.2% | ↓8.4% | ↓9.5% |

Note: LIMSSR is the only model in the table trained under T-Miss ✓, yet it outperforms all T-Miss ✗ (i.e., trained with complete data) methods in almost all missing combinations. This is the strongest "qualitative difference" evidence in the paper.

### Ablation Study

| Configuration | Average Spearman | Notes |
|------|------------------|------|
| Full LIMSSR | 0.789 | Complete framework |
| w/o PCMI (zero-filling missing modalities) | Significant drop | LLM cannot infer missing semantics |
| w/o LMRF (mean pooling instead of fusion tokens) | Drop | Multidimensional info collapse |
| w/o MDA Path 1 (cross-modal aggregation only) | Drop | Lacks high-level semantic calibration |
| w/o MDA Path 2 (LLM reasoning only) | Drop | Hallucination under severe missingness |
| w/o Consistency Loss | Drop | Lacks mutual verification |
| w/o Token-Level Regularization | Drop | Fusion tokens become redundant |

### Key Findings
- **Training with missing modalities can outperform training with complete data**: The most counterintuitive result—under the extreme case of audio-only, LIMSSR achieves 11.7% higher Spearman and 7.2% lower MSE than SOTA, indicating that LLM world knowledge provides a qualitative advantage in inferring missing semantics.
- **Path 1 + Path 2 are complementary**: Either path alone leads to performance drop; mask-aware adaptive fusion in MDA is key to resisting hallucination.
- **Sweet spot for number of fusion tokens $K$**: $K=3$ best matches AQA's three dimensions (difficulty/execution/artistry); more leads to overfitting.
- **Audio modality is hardest to impute**: All methods perform worst in {a}-only setting, as audio is least correlated with action quality, but LIMSSR still far exceeds baselines, showing LLM's relative gain is greatest for low-information modalities.

## Highlights & Insights
- **Task reformulation is the main contribution**: Recasting IML from "reconstruction/distillation" to "conditional sequence reasoning" turns a supervision-limited problem into an LLM-friendly next-token problem; this "reformulate as LM task" idea is transferable to many incomplete multimodal scenarios.
- **Special token design is elegant**: Missing token placeholders, boundary tokens for segmentation, and fusion tokens for aggregation turn the LLM into a programmable "semantic calculator," enabling custom functions without modifying the LLM architecture.
- **Mask-aware dual-path adaptation**: Encoding "how confident am I" into the network, with learnable modality-level confidence $\gamma_m$, reflects an engineering approach to reasoning uncertainty.
- **Training with missing data outperforms training with complete data**: This finding offers the IML community a new perspective—LLM priors may be more valuable than paired data, suggesting a need to rethink the "paired data paradigm."

## Limitations & Future Work
- Validation is mainly on AQA; effectiveness in other IML scenarios (emotion recognition, medical diagnosis) needs further study.
- LLM reasoning introduces significant computational cost, making it impractical for real-time applications (e.g., live scoring).
- Lacks systematic experiments on LLM scale (7B/13B/70B); LLM world knowledge is only effective when task-relevant, possibly less so for low-resource languages or rare action types.
- "Hallucination" is not quantitatively defined or measured, only indirectly mitigated via MDA.
- No experiments on modalities beyond text-vision-audio (e.g., physiological signals, depth maps).

## Related Work & Insights
- **vs ActionMAE / IMDer / DMVG (reconstruction)**: These methods rely on complete training pairs; this work breaks that constraint.
- **vs MoMKE / MCMoE / CorrKD (distillation/prior)**: They still require complete modalities as teachers; LIMSSR replaces this with LLM priors, essentially "using general world knowledge instead of domain paired supervision."
- **vs MissRAG / TAMML (LLM-based IML)**: MissRAG requires a pre-built modality prototype pool; TAMML textualizes all modalities, losing fine-grained information; LIMSSR lets the LLM reason directly in the original embedding space, making it more general and dependency-free.
- **vs Hedgehog / LoLCATs (LLM for other tasks)**: Similar in spirit—leveraging LLM non-linguistic abilities for domain problems; LIMSSR focuses on "missing information inference" rather than "long sequence modeling."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Proposes a new "incomplete observation during training" setting and reshapes the IML paradigm with LLM sequence reasoning; both problem and method are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three public AQA benchmarks, multiple missing combinations, and comparison with 10+ baselines; but only validated on AQA, lacking cross-task generalization evidence.
- Writing Quality: ⭐⭐⭐⭐ Clear narrative, Figure 1's comparison of three paradigms is intuitive; formulas are numerous but well-explained.
- Value: ⭐⭐⭐⭐ Offers the IML community a new paradigm and a convincing non-linguistic use case for LLM-as-tool applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reasoning-Driven Multimodal LLM for Domain Generalization](../../ICLR2026/multimodal_vlm/reasoning-driven_multimodal_llm_for_domain_generalization.md)
- [\[AAAI 2026\] MCMoE: Completing Missing Modalities with Mixture of Experts for Incomplete Multimodal Action Quality Assessment](../../AAAI2026/multimodal_vlm/mcmoe_completing_missing_modalities_with_mixture_of_experts_for_incomplete_multi.md)
- [\[ICML 2026\] Learn to Think: Improving Multimodal Reasoning through Vision-Aware Self-Improvement Training](learn_to_think_improving_multimodal_reasoning_through_vision-aware_self-improvem.md)
- [\[ICML 2026\] Instruction Lens Score: Your Instruction Contributes a Powerful Object Hallucination Detector for Multimodal Large Language Models](instruction_lens_score_your_instruction_contributes_a_powerful_object_hallucinat.md)
- [\[CVPR 2026\] MODIX: Training-Free Multimodal Information-Driven Positional Index Scaling for VLMs](../../CVPR2026/multimodal_vlm/modix_positional_index_scaling.md)

</div>

<!-- RELATED:END -->
