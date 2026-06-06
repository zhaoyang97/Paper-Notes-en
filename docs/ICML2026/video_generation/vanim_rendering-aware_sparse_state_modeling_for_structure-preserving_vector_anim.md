---
title: >-
  [Paper Note] VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation
description: >-
  [ICML 2026][Video Generation][SVG Animation] VAnim models open-domain text-to-SVG animation as "sparse state updates on a persistent DOM tree" + "Identification-First motion planning" + "GRPO rendering-aware reinforcemen…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "SVG Animation"
  - "Sparse State Updates"
  - "Identification-First CoT"
  - "GRPO"
  - "Rendering-Aware RL"
date: 2026-05-08
content_hash: 2a6001fa216dd8cb
---

# VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation

**Conference**: ICML 2026  
**arXiv**: [2605.01517](https://arxiv.org/abs/2605.01517)  
**Code**: None (Project page only)  
**Area**: Generative Models / Vector Animation / Multimodal LLMs  
**Keywords**: SVG Animation, Sparse State Updates, Identification-First CoT, GRPO, Rendering-Aware RL

## TL;DR
VAnim models open-domain text-to-SVG animation as "sparse state updates on a persistent DOM tree" + "Identification-First motion planning" + "GRPO rendering-aware reinforcement learning." This approach compresses sequence length by $9.86\times$ while maintaining topological consistency, significantly outperforming GPT-5.2, Gemini 3 Pro, and LiveSketch.

## Background & Motivation
**Background**: SVG is the de facto standard in UI/Web/Icon design due to its scalability, editability, and small file size. Vector animation (e.g., loading indicators, micro-interactions) requires adding a temporal dimension to SVGs. Current approaches follow two paths: optimization-based differentiable rendering (LiveSketch series), which uses SDS to iterate thousands of steps in pixel space to approximate text-to-video priors, and general LLM-based methods (GPT-5.2, Gemini 3 Pro, Keyframer), which directly generate CSS/SMIL transformation code.

**Limitations of Prior Work**: Differentiable rendering methods (i) suffer from slow inference (minutes), preventing interactivity, and (ii) treat vectors as independent strokes, lacking structural awareness, which leads to the collapse of closed shapes or occlusions, limiting them to sparse sketches. LLM-based methods suffer from "affine bias": CSS/SMIL codes are mathematically limited to translation, rotation, and scaling, failing to express path-level non-rigid deformations (e.g., a waving flag or a deforming water drop). Furthermore, rewriting the entire SVG frame-by-frame triggers (a) context explosion (86k tokens for 24 frames) and (b) identity drift (random modifications to static elements).

**Key Challenge**: The fundamental tension between expressiveness (non-rigid geometric deformation requires modifying the `d` attribute of paths) and stability (modifying paths easily destroys DOM topology and identity consistency). Any paradigm that "autoregressively generates the entire animated SVG" cannot escape both issues simultaneously.

**Goal**: (i) Compress animation sequences to a length manageable by LLM contexts; (ii) impose hard constraints ensuring elements not involved in the animation remain byte-for-byte unchanged; (iii) provide path-level non-rigid deformation capabilities; and (iv) incorporate non-differentiable SVG rendering into the training loop.

**Key Insight**: The authors observe that 85%+ of SVG syntax is redundant between adjacent frames. Only a few attributes like `d`, `transform`, and `opacity` actually change. Thus, an animation can be rewritten as an "initial SVG + a stream of ID-anchored attribute differences." This reduces the generation target from a "full tree token sequence" to "sparse diffs," naturally resolving context explosion and identity drift.

**Core Idea**: Redefine animation from "sequence generation" to "sparse state updates (SSU) on a persistent DOM tree." Combined with "Identification-First" CoT and rendering-aware GRPO, this allows the LLM to learn geometric deformations while preserving structure.

## Method
VAnim reconstructs data, representation, inference, and training to support SSU.

### Overall Architecture
Input: Initial static SVG $S_0$, its rendered image $I_0$, and a natural language instruction $P$. Output: A sequence of sparse state updates $\mathcal{D}=\{\Delta_t\mid t=1,\dots,T\}$, where each $\Delta_t$ is a set of "(id, attribute, new value)" triplets, listing only attributes that changed relative to the previous frame.

The model is based on Qwen3-VL-8B-Thinking. The vision encoder projects $I_0$ into tokens, which are interleaved with $S_0$ and $P$ in the same sequence, enabling the model to align visual objects with DOM IDs across modalities. Generation is explicitly divided into two stages, corresponding to the probability decomposition $p_\theta(o\mid x)=p_\theta(C\mid x)\cdot p_\theta(\mathcal{D}\mid C,x)$, where $C$ is the Structure-Bound CoT and $o=(C,\mathcal{D})$. Training involves two phases: Phase I performs structured SFT on SVGAnim-SFT (123k), and Phase II performs rendering-aware GRPO on SVGAnim-RL (a high-complexity 10k subset).

On the data side, the authors collected Lottie files from Flaticon. A Node.js rendering script generated ID-anchored SVG DOM sequences. After coordinate normalization, absolute-to-relative coordinate conversion, and cleaning, the SVGAnim-134k dataset was produced. Doubao-Seed-1.6 was used for dual-stream annotation: user-centric prompt $P$ + Structure-Bound CoT $C$ (consisting of "Entity Identification: blue circle → ID 05" and "Visual Dynamic Planning: ID 05 scale up/down"). Strict ID consistency filtering was applied to ensure all IDs referenced in the CoT exist.

### Key Designs

1.  **Sparse State Update (SSU) Representation**:
    *   **Function**: Compresses animation sequences from "full SVG per-frame rewriting" to "initial SVG + attribute diff stream," avoiding context explosion and ensuring structural consistency by construction.
    *   **Mechanism**: $\Delta_t=\{(id, attr, v_t)\mid v_t\ne v_{t-1}, (id, attr, v_t)\in A(S_t)\}$. The entire animation is represented as $(S_0, \Delta_1, \dots, \Delta_T)$. During serialization, `<|time=t|>` and `<|ID=id|>` control tokens anchor changes to persistent DOM nodes. A 24-frame animation is compressed from 86k tokens to 9.2k tokens (a $9.86\times$ ratio). Notably, diffs account for 61% of the tokens, meaning the model capacity is spent on "learning dynamics" rather than "copying statics."
    *   **Design Motivation**: 85% syntax redundancy + LLM rewriting leads to random noise and identity drift. SSU keeps all unlisted attributes unchanged by construction, eradicating identity drift at the architectural level while linearizing generation length to "animation complexity" rather than "SVG size."

2.  **Identification-First Motion Planning (CoT)**:
    *   **Function**: Forces the model to ground visual entities to specific DOM IDs before writing temporal logic, preventing structural confusion caused by mixing "what to do" with "which node to modify."
    *   **Mechanism**: Inference is split into a Director stage (Input $I_0, S_0, P$; Output structured CoT $C$ with fixed formats: Entity Identification mapping objects to IDs, and Visual Dynamic Planning describing temporal behavior) and an Animator stage (Output $\mathcal{D}$ based on $C$). All training samples are grounded via ID consistency filtering.
    *   **Design Motivation**: Ablations show that removing CoT drops Semantic Alignment from 0.281 to 0.255, with errors like rotating a whole cabinet instead of just the door becoming common. Explicit grounding is a prerequisite for structural integrity, consistent with ReAct/CoT logic in LLM agents.

3.  **Rendering-Aware Reinforcement Learning (GRPO + Hybrid Reward)**:
    *   **Function**: Uses rendered video quality signals to fine-tune the LLM, encouraging path-level non-rigid deformations instead of hiding behind affine transforms due to the "minimal modification bias" of SFT.
    *   **Mechanism**: For each input, $G=8$ candidates $\{o_1, \dots, o_G\}$ are sampled. Each is rendered into a $500\times 500$ video via a Playwright headless browser and fed into the PE-Core video encoder. The reward is $\mathcal{R}=\lambda_{\text{align}}\mathcal{R}_{\text{align}}+\lambda_{\text{fmt}}\mathcal{R}_{\text{fmt}}$, where $\mathcal{R}_{\text{align}}=\mathrm{CosineSim}(E_{\text{text}}(P), E_{\text{video}}(V_{\text{pred}}))$ measures semantic alignment, and $\mathcal{R}_{\text{fmt}}\in\{-1, +1\}$ strictly evaluates renderability, length matching, and ID validity. The objective is $\mathcal{L}_{\text{GRPO}}=\mathbb{E}\bigl[\tfrac{1}{G}\sum_i\min(\tfrac{\pi_\theta(o_i\mid x)}{\pi_{\theta_{\text{old}}}(o_i\mid x)}\hat A_i, \text{clip}(\cdot)\hat A_i)-\beta D_{\text{KL}}\bigr]$, with $\beta=0.01$ and temperature 0.9.
    *   **Design Motivation**: SFT only supervises code correctness, ignoring visual aesthetics. This leads to conservative strategies (translation or slight scaling). Dense gradient signals from PE-Core guide the model to directly manipulate Bézier control points in the `d` attribute, activating non-rigid deformations. The format reward acts as a hard constraint to prevent the policy from wandering into "cool but broken" regions.

### Loss & Training
Stage I: $\mathcal{L}_{\text{SFT}}(\theta)=-\mathbb{E}_{(I_0,S_0,P)\sim D_{\text{SFT}}}[\log p_\theta(C,\mathcal{D}\mid I_0,S_0,P)]$, maximum sequence 25k tokens, full-parameter fine-tuning. Stage II: GRPO as defined, $G=8$, $\beta=0.01$, $\lambda_{\text{align}}=\lambda_{\text{fmt}}=1.0$, performed on $8\times$ H100 GPUs.

## Key Experimental Results

### Main Results
Evaluated on SVGAnim-Test (1k held-out) using PE-Core-G14-448 for Semantic Alignment and calculating the Success Rate (renderability):

| Method | Semantic Alignment ↑ | Success Rate ↑ |
| :--- | :--- | :--- |
| LiveSketch | 0.158 | 100.0% |
| GPT-5.2 | 0.234 | 88.5% |
| Gemini 3 Pro | 0.243 | 86.2% |
| VAnim (SFT-only) | 0.268 | 95.2% |
| **VAnim (GRPO)** | **0.281** | **100.0%** |

VAnim-GRPO achieves the highest semantic alignment and a 100% success rate. While LiveSketch is 100% renderable, its low semantic score (0.158) reflects frequent topological collapses. GPT/Gemini models suffer from unclosed tags and ID hallucinations in long sequences, dropping success rates to 86–88%.

### Ablation Study

| Configuration | Semantic Alignment ↑ | Success Rate ↑ | Description |
| :--- | :--- | :--- | :--- |
| Full VAnim | 0.281 | 100.0% | Full method |
| w/o Rendering-Aware RL | 0.268 (-0.013) | 95.2% (-4.8%) | SFT baseline; "lazy motion" (e.g., door barely opens) |
| w/o Structure-Bound CoT | 0.255 (-0.026) | 98.6% (-1.4%) | Correct logic applied to wrong objects |
| w/o SSU (Appendix) | — | 62.3% | Naive frame generation; success rate collapses |
| w/o input image (Appendix) | — | — | Significant drop in SSIM/temporal smoothness |

### Key Findings
- All three components are indispensable: CoT solves "modifying the correct node," SSU solves "not breaking the structure," and RL solves "daring to perform large deformations." CoT provides the largest single-point contribution (0.026 semantic drop), showing that explicit grounding is the foundation for non-rigid animation.
- The 62.3% success rate of naive generation validates the identity drift hypothesis: without SSU constraints, LLMs randomly modify static attributes, leading to rendering failures.
- GRPO group size $G$ sweeps indicate a trade-off between semantic alignment and identity preservation; larger $G$ encourages broader exploration but may increase identity drift.
- Visual input $I_0$ is crucial for SSIM and temporal smoothness; pure code + prompt input is insufficient for mapping visual objects to DOM IDs.

## Highlights & Insights
- Reformulating "sequence generation" as "sparse updates on persistent state" is a deep insight: it effectively adds a "topological invariance" hard constraint to the generative model at the architectural level rather than the loss level. This logic applies to HTML/UI editing, 3D scene graphs, CAD modifications, and robot trajectory editing.
- Identification-First CoT bridges the ReAct paradigm with "Visual Entity → DOM ID" mapping, using ID consistency filtering to embed execution into the data pipeline rather than treating CoT as a mere prompting trick.
- Using video encoders like PE-Core for RL rewards is an elegant way to incorporate non-differentiable SVG rendering into the gradient chain. The combination of sparse format rewards and dense semantic rewards serves as a template for other code-to-render tasks (HTML/CSS, shaders, SQL-to-charts).

## Limitations & Future Work
- Data is sourced entirely from Flaticon Lottie files, which are well-structured with IDs/grouping. Generalization to messy, tool-exported SVGs (no IDs, chaotic nesting) remains an open question.
- Rendering-aware RL relies on headless browsers and video encoders, making the per-step cost significantly higher than standard RLHF and hard to replicate in resource-constrained settings.
- Current VAnim focuses on visual animation, lacking support for JavaScript-triggered interactions or multi-scene narratives, which limits end-to-end deployment in real UI/Web workflows.
- Evaluation depends heavily on PE-Core, creating a risk of metric circularity relative to training rewards. Independent metrics (InternVideo2, SSIM, flow) mitigate this, but large-scale human evaluation is still needed.
- SSU assumes a "topologically isomorphic" DOM across adjacent frames, which is not natural for animations involving the appearance/disappearance of elements or DOM restructuring.

## Related Work & Insights
- **vs LiveSketch (Gal et al. 2024)**: LiveSketch optimizes strokes in pixel space via SDS, leading to structural collapse in closed shapes; VAnim edits the SVG DOM directly, preserving topology by design and finishing in one pass rather than hundreds of iterations.
- **vs Keyframer / GPT-5.2 / Gemini 3 Pro**: General LLMs stay within the comfort zone of affine transforms; VAnim's RL signal pushes the policy to manipulate Bézier control points, achieving true path-level deformation.
- **vs DeepSVG / SVGformer**: These focus on static vector composition; VAnim is the first to port the LLM paradigm to open-domain vector animation while handling the temporal dimension via SSU without context explosion.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The SSU + Identification-First CoT + Rendering-Aware GRPO trio represents the first systematic solution for open-domain vector animation, redefining the generation paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes three strong baselines (optimization, closed-source LLMs, SFT-only), core ablations, and extensive appendix experiments (SSU, input image, reward, group size), though cross-domain generalization evaluation is missing.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation clearly articulates affine bias, context explosion, and identity drift. Figures 1/2/4 explain data, representation, and inference modules thoroughly.
- **Value**: ⭐⭐⭐⭐ Open-sourced data, paradigm, and framework hold high value for design tools and UI/Web automation, with the only drawback being limited support for interactive logic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] OLAF-World: Orienting Latent Actions for Video World Modeling](olaf-world_orienting_latent_actions_for_video_world_modeling.md)
- [\[CVPR 2026\] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](../../CVPR2026/video_generation/identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)
- [\[ACL 2026\] OSCBench: Benchmarking Object State Change in Text-to-Video Generation](../../ACL2026/video_generation/oscbench_benchmarking_object_state_change_in_text-to-video_generation.md)
- [\[ICML 2026\] VEDA: Scalable Video Diffusion via Distilled Sparse Attention](veda_scalable_video_diffusion_via_distilled_sparse_attention.md)
- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)

</div>

<!-- RELATED:END -->
