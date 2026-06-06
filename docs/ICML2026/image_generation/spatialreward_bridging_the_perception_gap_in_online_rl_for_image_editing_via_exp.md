---
title: >-
  [Paper Note] SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning
description: >-
  [ICML 2026][Image Generation][Reward Model] The authors identify an "attention collapse" problem in MLLM-based editing reward models—when scoring, models fail to compare the original and edited images…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Reward Model"
  - "Image Editing"
  - "Online RL"
  - "Think-with-Boxes"
  - "Spatial Reasoning"
date: 2026-05-08
content_hash: 768ade57b3bc778f
---

# SpatialReward: Bridging the Perception Gap in Online RL for Image Editing via Explicit Spatial Reasoning

**Conference**: ICML 2026  
**arXiv**: [2602.07458](https://arxiv.org/abs/2602.07458)  
**Code**: Project Page https://lorangan-ddup.github.io/SpatialReward/ (Available)  
**Area**: Image Generation / Image Editing / RLHF / Reward Model / Multimodal Evaluation  
**Keywords**: Reward Model, Image Editing, Online RL, Think-with-Boxes, Spatial Reasoning

## TL;DR
The authors identify an "attention collapse" problem in MLLM-based editing reward models—when scoring, models fail to compare the original and edited images, instead collapsing onto sink tokens for blind judgment. They propose SpatialReward: an 8B model that first predicts bounding boxes for edited regions and then performs interleaved cross-image reasoning using these box tokens as anchors. Coupled with a 260K-sample spatial-aware dataset and two-stage GRPO training, it achieves SOTA on three reward benchmarks and boosts the GEdit-Bench score of OmniGen2 by +0.90 (double the gain from GPT-4.1).

## Background & Motivation

**Background**: Instructed image editing (e.g., InstructPix2Pix, MagicBrush, OmniGen, Qwen-Edit, FLUX) has expanded from "style transfer" to complex multi-region editing. Recently, Flow-GRPO and Dance-GRPO introduced online RL to diffusion models, treating editing as an interactive trial-and-error process aligned with human preferences, significantly outperforming SFT. However, the effectiveness of online RL is strictly bottlenecked by the reward model—reward signals must be reliable, efficient, interpretable, and capable of fine-grained judgment across different image regions.

**Limitations of Prior Work**: Existing reward designs fall into three categories, none of which are suitable for online RL in editing tasks: (i) pairwise rewards (e.g., MMRB2) excel at zero-shot relative ranking, but online RL requires absolute scalars; converting rankings to scalars introduces ambiguity, and $O(N^2)$ inference costs are prohibitive; (ii) pointwise discriminative models (e.g., EditReward) add a linear regression head on VLM embeddings, lacking an explicit reasoning chain and suffering from high labeling costs and poor scalability; (iii) pointwise generative models ("MLLM-as-a-judge," e.g., EditScore, GPT-5) can output a chain-of-thought, but editing requires strict "cross-image regional comparison"—current MLLMs lack explicit spatial anchors, and even top closed-source models like GPT-5 fall into "attention collapse," where attention distributions collapse onto a few head/tail sink tokens, almost ignoring the source image and degrading to single-image evaluation, thereby missing subtle differences.

**Key Challenge**: The training dynamics of online RL require the reward model to perform fine-grained region-level discrimination across images while outputting absolute scalars. However, current MLLM evaluators lack spatial anchors to guide cross-image comparison, meaning neither prompt engineering nor parameter distillation can cure "attention collapse," leading to systematic deviations from human preferences.

**Goal**: (i) Quantitatively diagnose the "attention collapse" perception gap; (ii) design an architecture that forces MLLMs to perform cross-image regional comparison; (iii) construct a large-scale spatial-aware dataset to support this capability; (iv) align preferences on hard samples using GRPO; (v) verify substantial improvements in downstream online RL for editing models.

**Key Insight**: The authors found that humans follow a "locate first, then compare" two-step process when judging edits, which MLLMs lack. If the model is explicitly required to predict bounding boxes for the edited areas before reasoning, and these box tokens are injected into the reasoning chain as "look here" hard pointers, attention can be redirected from sink tokens back to the relevant pixel areas.

**Core Idea**: Utilize "Think-with-Boxes" to treat spatial anchors (bounding boxes) as interleaved tokens that the language model can directly cite, forcing it to "look back" for every region-level judgment to produce fine-grained scores based on pixel evidence. This capability is solidified into a stable reward signal through a spatial-prior data pipeline and two-stage SFT $\rightarrow$ GRPO training.

## Method

### Overall Architecture
SpatialReward models reward as a conditional generation task where the model maps input $X$ to structured output $Y=(B, \mathcal T, s)$: $B$ is a sequence of bounding boxes, $\mathcal T$ is a textual rationale, and $s$ is a scalar score. Following the VIEScore protocol, the evaluation is decoupled into Semantic Consistency (SC, including instruction following $s_{if}$ and source consistency $s_{con}$) and Perceptual Quality (PQ, including naturalness $s_{nat}$ and artifacts $s_{art}$). The final reward is a hierarchical aggregation $R_{final}=(S_{SC})^{\alpha}(S_{PQ})^{1-\alpha}$ with $\alpha=0.8$. The SC flow follows the "Think-with-Boxes" path (locate then compare), while the PQ flow performs reference-free evaluation on the edited image alone, separating these two distinct types of judgment.

### Key Designs

1.  **Think-with-Boxes Architecture (Mechanism for Forced Cross-Image Comparison)**:
    - **Function**: Explicitly writes "where to look" into the model's reasoning process via interleaved box tokens, structurally eliminating attention collapse in MLLM evaluation.
    - **Mechanism**: The SC flow consists of three steps: (a) Localization: the model predicts bounding boxes $B$ for all edited objects, outputting formats like `<|bbox_0|>(x1,y1,x2,y2)`; (b) Anchored Verification: every time a box token like `<|bbox_id|>` appears in the rationale $\mathcal T$, the model is forced to "look back" at the corresponding pixel region, with a `<|global|>` token added to trigger global context scanning; (c) the model outputs SC scores $s_{sc}=[s_{if}, s_{con}]$. The PQ flow only receives the edited image with $B=\emptyset$, outputting a plain text rationale and $s_{pq}=[s_{nat}, s_{art}]$. The backbone is Qwen-2.5-VL-8B-Instruct.
    - **Design Motivation**: MLLMs collapse to sink tokens in cross-image tasks because they are never forced to ground to specific pixels; by embedding box tokens in text, the model must "re-observe" the region for each citation, restoring a healthy distribution of cross-image attention. Attention visualizations in Fig.1c show SpatialReward's attention successfully re-aligning with relevant regions in the source image.

2.  **Spatial-Prior-Guided Data Pipeline (260K SpatialReward-260k Dataset)**:
    - **Function**: Constructs a large-scale dataset aligning boxes, rationales, and scores to enable the SFT stage to learn the "grounding-before-reasoning" paradigm.
    - **Mechanism**: A three-step pipeline: Step I uses Qwen-2.5-VL-72B to pre-generate bounding boxes $B$ for all samples as spatial priors; Step II routes experts based on category—human editing uses Gemini-1.5-Pro with crop prompts, object editing uses GPT-4o with visual bounding box overlays to force focus, and PQ is independently evaluated by GPT-4o; Step III feeds the generated $\mathcal T_{raw}$ and $B$ back into Qwen-2.5-VL-72B for alignment (interleaved format) and hallucination checks (discarding if $\mathcal T$ and $B$ are visually inconsistent). The final 260K set includes: 100K cleaned EditScore data (with $B$ injected), 100K re-generated EditReward data (discarding original coarse scores), and 60K custom multi-region editing data.
    - **Design Motivation**: Human labeling cannot provide aligned "box+reasoning+score" data at scale. Using expert routing + visual box overlays utilizes each teacher model's strengths, while consistency checks eliminate noise, ensuring a clean training distribution.

3.  **Two-Stage Training: SFT + GRPO Online Consistency RL**:
    - **Function**: Teaches structured generation $(B, \mathcal T, s)$ via SFT, then aligns with human consistency on hard samples via online RL to eliminate scoring hallucinations.
    - **Mechanism**: Stage 1 performs SFT on Qwen-2.5-VL-8B-Instruct with 260K data, targeting $\mathcal L_{SFT}=-\sum_t \log P_\theta(y_t|y_{<t}, X)$. Stage 2 identifies 7K low-score hard samples and uses Gemini-1.5-Flash as an Online Supervisor to provide 0–1 consistency rewards for model rollouts. The target is GRPO: $\mathcal J_{GRPO}=\mathbb E[\frac{1}{G}\sum_i \frac{\pi_\theta(o_i|q)}{\pi_{\theta_{old}}(o_i|q)}\hat A_i] - \beta D_{KL}(\pi_\theta\|\pi_{ref})$ where $\hat A_i=(r_i-\mathrm{mean}\{r_j\})/\mathrm{std}\{r_j\}$. Reward aggregation uses a weighted geometric mean $R=(S_{SC})^\alpha (S_{PQ})^{1-\alpha}$.
    - **Design Motivation**: SFT only fits the "average" of the teacher distribution and may hallucinate on long-tail hard samples. GRPO optimizes relative group differences based on consistency, correcting SFT biases. The weighted geometric mean provides denser gradient signals than a minimum (bucket principle) and penalizes weaknesses better than an arithmetic mean.

### Loss & Training
Standard cross-entropy for SFT; group size $G$ and KL coefficient $\beta$ for GRPO follow DeepSeek-V3 defaults. Reward aggregation parameters: $\alpha=0.80$, $w_{SC}=\{0.6, 0.4\}$ (instruction following : source consistency), $w_{PQ}=\{0.5, 0.5\}$, determined via grid search on a 2K validation set.

## Key Experimental Results

### Main Results
Overall accuracy on three reward benchmarks (Bold denotes highest; SpatialReward uses Qwen-2.5-VL-8B):

| Model | EditReward-Bench (Ovrl) | MMRB2 (Ovrl) | MER-Bench (Ovrl) |
| :--- | :---: | :---: | :---: |
| GPT-4o | 0.705 | 0.535 | 0.358 |
| GPT-4o-vision | 0.755 | 0.619 | 0.423 |
| Gemini-1.5-Pro | 0.722 | 0.534 | 0.462 |
| Gemini-1.5-Flash | 0.769 | 0.621 | **0.508** |
| EditScore-8B (baseline) | 0.690 | 0.570 | 0.350 |
| EditReward (Discrim.) | 0.792 | 0.657 | 0.448 |
| **SpatialReward (Ours, 8B)** | **0.803** | **0.661** | 0.483 |

Downstream Online RL (OmniGen2 + Flow-GRPO, $\Delta$ gain in GEdit-Bench-EN Overall score):

| Reward signal | GEdit Ovrl | $\Delta$ |
| :--- | :---: | :---: |
| Baseline OmniGen2 | 6.42 | — |
| w/ GPT-4o | 6.73 | +0.45 |
| w/ EditScore | 6.89 | +0.61 |
| w/ EditReward | 7.19 | +0.77 |
| **w/ SpatialReward** | **7.32** | **+0.90** |

### Ablation Study

| Configuration | EditReward-Bench Acc | Note |
| :--- | :---: | :--- |
| SFT baseline (No spatial anchors) | 0.743 | Starting point |
| SFT w/ Box Only (Predict box, no cite) | 0.761 | Adding box helps |
| SFT w/ Think-with-Box | 0.778 | Explicit cite adds more |
| **+ Online GRPO** | **0.803** | RL stage is critical |
| Weighted Geometric Mean (Ours) | **0.803** | Balances dense gradients/weaknesses |

**Attention Diagnosis** (on 776 EditReward-Bench samples):
| Method | Entropy Diff $|\Delta H|$ ↓ | Source Entropy $H_{src}$ | Concentration ↓ |
| :--- | :---: | :---: | :---: |
| Baseline | 3.48 ± 0.57 | 2.88 ± 0.71 | 0.84 ± 0.05 |
| **Ours** | **1.16 ± 1.10** | **5.71 ± 0.81** | **0.37 ± 0.14** |

### Key Findings
- Box prediction alone yields a 1.8% gain; adding "cited Reasoning" adds another 1.7%, proving spatial anchors and active citation are independent, effective components.
- On MER-Bench 4-Pair difficulty (fine-grained sub-dimensions), SpatialReward (21.5%) outperforms Gemini-1.5-Flash (19.5%), showing that spatial anchors are particularly effective for hard distinctions.
- In RL, SpatialReward's +0.90 gain is 1.5x larger than EditScore (+0.61) and 2x larger than GPT-4o (+0.45), while being 1.5x faster than EditReward via vLLM.
- Attention diagnosis quantitatively confirms the "attention collapse" hypothesis: baseline concentration 0.84 $\to$ 0.37, source entropy 2.88 $\to$ 5.71, restoring a healthy distribution.

## Highlights & Insights
- Diagnosing "MLLM-as-a-judge failing cross-image comparison" as "attention collapse" and curing it with a box-cite mechanism opens a new design space for reward models.
- The essence of Think-with-Boxes is "passing spatial hard pointers between generated tokens," a concept applicable to any task involving cross-region judgment (UI verification, multi-doc RAG).
- Using weighted geometric mean for reward aggregation is an underrated engineering detail: it provides denser gradients for RL than min-aggregation and penalizes flaws better than arithmetic means.

## Limitations & Future Work
- The reward is a single scalar; future work could explore region-level credit assignment, feeding local rewards back to the generator.
- Bounding boxes are coarse; fine-grained tasks (e.g., hair-level editing) might require segmentation masks.
- 8B models still show a gap compared to giant models on extremely complex multi-constraint reasoning in MER-Bench.

## Related Work & Insights
- **vs EditScore**: Both use 8B backbones, but without spatial anchors, EditScore suffers from attention collapse. Ours performs +11.3% better under the same parameters.
- **vs EditReward**: EditReward lacks source consistency modeling, which can lead to "content drift" during RL; Oura explicitly models SC to prevent this.
- **vs VLM Spatial Reasoning**: While prior work shows coordinates improve object-attribute binding, this is the first to use "explicitly citing spatial coordinates during reasoning" as a core mechanism for reward modeling.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Diagnosing attention collapse and introducing spatial hard pointers.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple benchmarks, downstream RL, and attention analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Extremely clear diagrams and narrative.
- **Value**: ⭐⭐⭐⭐⭐ A SOTA reward model and an open RL pipeline component.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](../../ICLR2026/image_generation/editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)
- [\[ICML 2026\] A Systematic Investigation of RL-Jailbreaking in LLMs](a_systematic_investigation_of_rl-jailbreaking_in_llms.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[ICML 2026\] Semantic Granularity Navigation in Image Editing](semantic_granularity_navigation_in_image_editing.md)
- [\[CVPR 2026\] Probing and Bridging Geometry–Interaction Cues for Affordance Reasoning in Vision Foundation Models](../../CVPR2026/image_generation/probing_and_bridging_geometry-interaction_cues_for_affordance_reasoning_in_visio.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] EditScore: Unlocking Online RL for Image Editing via High-Fidelity Reward Modeling](../../ICLR2026/image_generation/editscore_unlocking_online_rl_for_image_editing_via_high-fidelity_reward_modelin.md)
- [\[ICML 2026\] A Systematic Investigation of RL-Jailbreaking in LLMs](a_systematic_investigation_of_rl-jailbreaking_in_llms.md)
- [\[ICLR 2026\] Bridging Generalization Gap of Heterogeneous Federated Clients Using Generative Models](../../ICLR2026/image_generation/bridging_generalization_gap_of_heterogeneous_federated_clients_using_generative_.md)
- [\[ICML 2026\] Semantic Granularity Navigation in Image Editing](semantic_granularity_navigation_in_image_editing.md)
- [\[ICML 2026\] Stage-wise Distortion-Perception Traversal in Zero-shot Inverse Problems with Diffusion Models](stage-wise_distortion-perception_traversal_in_zero-shot_inverse_problems_with_di.md)

</div>

<!-- RELATED:END -->
