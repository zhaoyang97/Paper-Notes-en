---
title: >-
  [Paper Note] VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation
description: >-
  [ICML 2026][Video Generation][SVG animation] VAnim models open-domain text-to-SVG animation as "sparse state updates on a persistent DOM tree" + "Identification-First motion planning" + "GRPO rendering-aware reinforcemen…
tags:
  - "ICML 2026"
  - "Video Generation"
  - "SVG animation"
  - "sparse state update"
  - "Identification-First CoT"
  - "GRPO"
  - "rendering-aware RL"
date: 2026-05-08
content_hash: a900bdfc7a076d8e
---

# VAnim: Rendering-Aware Sparse State Modeling for Structure-Preserving Vector Animation

**Conference**: ICML 2026  
**arXiv**: [2605.01517](https://arxiv.org/abs/2605.01517)  
**Code**: None (project homepage only)  
**Area**: Generative Models / Vector Animation / Multimodal LLM  
**Keywords**: SVG animation, sparse state update, Identification-First CoT, GRPO, rendering-aware RL

## TL;DR
VAnim models open-domain text-to-SVG animation as "sparse state updates on a persistent DOM tree" + "Identification-First motion planning" + "GRPO rendering-aware reinforcement learning," achieving a $9.86\times$ sequence length compression while preserving topology, and significantly surpassing GPT-5.2, Gemini 3 Pro, and LiveSketch.

## Background & Motivation
**Background**: SVG is the de facto standard in UI/Web/icon design due to its scalability, editability, and small file size; vector animation (e.g., loading indicators, micro-interactions) requires adding a temporal dimension to SVG. Current approaches fall into two camps: optimization-based differentiable rendering methods (LiveSketch series) use SDS to iteratively approach text-video priors in pixel space over thousands of steps; general LLM-based methods (GPT-5.2, Gemini 3 Pro, Keyframer) directly generate CSS/SMIL transformation code.

**Limitations of Prior Work**: Differentiable rendering methods (i) are slow at inference (minutes per sample), making interaction impossible; (ii) treat vectors as independent strokes, lacking structural awareness, leading to frequent breakdowns in closed shapes/occlusion, and can only handle sparse sketches. LLM-based methods suffer from affine bias: CSS/SMIL can only mathematically express translation, rotation, and scaling, but cannot perform path-level non-rigid deformations (e.g., flag waving, water droplet morphing). Moreover, rewriting the entire SVG for each frame triggers (a) context explosion (24 frames already require 86k tokens) and (b) identity drift (static elements are randomly modified, causing identity loss).

**Key Challenge**: There is a fundamental tension between expressiveness (non-rigid geometric deformation requires modifying the path's `d` attribute) and stability (modifying the path most easily breaks DOM topology/identity consistency). Any "autoregressive generation of the entire animation SVG" paradigm cannot simultaneously overcome both.

**Goal**: (i) Compress the animation sequence to a length manageable by LLM context; (ii) enforce a hard constraint during generation that "elements not involved in the animation must remain byte-for-byte unchanged"; (iii) enable path-level non-rigid deformation; (iv) incorporate non-differentiable SVG rendering into the training loop.

**Key Insight**: The authors observe that over 85% of SVG syntax between adjacent frames is redundant; only a few attributes like `d`, `transform`, and `opacity` actually change. Thus, animation can be rewritten as "initial SVG + a sequence of attribute diffs anchored by ID." This reduces the generation target from "entire tree token sequence" to "sparse diff," naturally resolving context explosion and identity drift.

**Core Idea**: Redefine animation from "sequence generation" to "sparse state updates (SSU) on a persistent DOM tree," combined with "Identification-First" CoT and rendering-aware GRPO, enabling the LLM to perform geometric deformations while preserving structure.

## Method
VAnim reconstructs data, representation, inference, and training to align with SSU.

### Overall Architecture
Input: initial static SVG $S_0$, its rendered image $I_0$, and natural language instruction $P$. Output: sparse state update sequence $\mathcal{D}=\{\Delta_t\mid t=1,\dots,T\}$, where each $\Delta_t$ is a set of (id, attribute, new value) tuples, listing only attributes that change from the previous frame.

The model is based on Qwen3-VL-8B-Thinking. The visual encoder projects $I_0$ into tokens, which are interleaved with $S_0$ and $P$ in the same sequence, allowing the model to align visual objects and DOM IDs across modalities. Generation is explicitly divided into two stages, corresponding to the probability decomposition $p_\theta(o\mid x)=p_\theta(C\mid x)\cdot p_\theta(\mathcal{D}\mid C,x)$, where $C$ is the Structure-Bound CoT and $o=(C,\mathcal{D})$. Training is two-stage: Stage I performs structured SFT on SVGAnim-SFT (123k); Stage II applies rendering-aware GRPO on SVGAnim-RL (10k high-complexity subset).

On the data side, the authors crawl Lottie files from Flaticon, use Node.js rendering scripts to generate SVG DOM sequences anchored by ID, perform coordinate normalization, absolute-to-relative conversion, and cleaning to obtain SVGAnim-134k. Doubao-Seed-1.6 is used for dual-stream annotation: user-centric prompt $P$ + Structure-Bound CoT $C$ (including "Entity Identification: blue circle → ID 05" and "Visual Dynamic Planning: ID 05 scale up/down" in two stages), with strict ID consistency filtering to ensure all IDs referenced in CoT exist.

### Key Designs

1. **Sparse State Update (SSU) Representation**:

    - **Function**: Compresses the animation sequence from "full SVG rewritten per frame" to "initial SVG + attribute diff stream," avoiding context explosion and structurally guaranteeing topology consistency.
    - **Mechanism**: Defines $\Delta_t=\{(id, attr, v_t)\mid v_t\ne v_{t-1}, (id, attr, v_t)\in A(S_t)\}$, representing the entire animation as $(S_0,\Delta_1,\dots,\Delta_T)$. Serialization uses `<|time=t|>` and `<|ID=id|>` control markers to anchor changes to persistent DOM nodes. A 24-frame animation is compressed from 86k tokens to 9.2k tokens, a $9.86\times$ reduction; the diff part accounts for 61%, indicating most model capacity is spent "learning dynamics" rather than "copying statics."
    - **Design Motivation**: 85% syntax repetition between adjacent frames + LLM rewriting easily introduces random perturbations → identity drift. SSU ensures all unspecified attributes "by construction" remain unchanged, eliminating identity drift at the architectural level; it also linearizes generation length to "animation complexity" rather than "SVG size."

2. **Identification-First Motion Planning (CoT)**:

    - **Function**: Before generating diffs, the model is forced to ground visual entities in the instruction to specific DOM IDs, then write temporal logic, avoiding mixing "what to do" and "which node to modify," which can cause structural errors.
    - **Mechanism**: Inference is split into a Director stage (inputs $I_0,S_0,P$, outputs structured CoT $C$ with two fixed-format sections: Entity Identification maps visual objects to IDs, Visual Dynamic Planning describes time-based behaviors for each ID) and an Animator stage (generates $\mathcal{D}$ based on $C$). Each CoT in the training data is filtered for ID consistency, ensuring 100% of training samples are structurally grounded.
    - **Design Motivation**: Ablation shows that removing CoT drops Semantic Alignment from 0.281 to 0.255, and mis-modification (e.g., rotating the entire cabinet instead of the door) becomes common, indicating explicit grounding is essential for structural integrity; this is logically consistent with ReAct/Chain-of-Thought as decision mediators in LLM agents.

3. **Rendering-Aware Reinforcement Learning (GRPO + Hybrid Reward)**:

    - **Function**: Uses rendered video quality signals to close the training loop for the LLM, encouraging the model to perform path-level non-rigid deformations instead of hiding in affine transformations due to SFT's "minimal modification bias."
    - **Mechanism**: For each input, $G=8$ candidate outputs $\{o_1,\dots,o_G\}$ are sampled, each rendered as a $500\times 500$ video via Playwright headless browser, and fed to the PE-Core video perception encoder. The reward is $\mathcal{R}=\lambda_{\text{align}}\mathcal{R}_{\text{align}}+\lambda_{\text{fmt}}\mathcal{R}_{\text{fmt}}$, where $\mathcal{R}_{\text{align}}=\mathrm{CosineSim}(E_{\text{text}}(P),E_{\text{video}}(V_{\text{pred}}))$ measures semantic alignment, and $\mathcal{R}_{\text{fmt}}\in\{-1,+1\}$ strictly checks renderability, length match, and ID validity. The GRPO objective is $\mathcal{L}_{\text{GRPO}}=\mathbb{E}\bigl[\tfrac{1}{G}\sum_i\min(\tfrac{\pi_\theta(o_i\mid x)}{\pi_{\theta_{\text{old}}}(o_i\mid x)}\hat A_i,\text{clip}(\cdot)\hat A_i)-\beta D_{\text{KL}}\bigr]$, with $\beta=0.01$, temperature $0.9$.
    - **Design Motivation**: SFT only supervises code correctness and cannot "see" the actual rendered result, so learned strategies are conservative, sticking to translation or slight scaling. The PE-Core visual encoder provides dense gradient signals, guiding the model to directly manipulate Bézier control points in the `d` attribute, activating non-rigid deformations that are "possible but not preferred"; the format reward acts as a hard constraint, preventing the RL policy from drifting into "cool but broken" regions.

### Loss & Training
Stage I: $\mathcal{L}_{\text{SFT}}(\theta)=-\mathbb{E}_{(I_0,S_0,P)\sim D_{\text{SFT}}}[\log p_\theta(C,\mathcal{D}\mid I_0,S_0,P)]$, with a maximum sequence length of 25k tokens, full-parameter fine-tuning. Stage II: GRPO as above, $G=8$, $\beta=0.01$, $\lambda_{\text{align}}=\lambda_{\text{fmt}}=1.0$, hardware $8\times$ H100.

## Key Experimental Results

### Main Results
On the custom SVGAnim-Test (1k held-out), semantic alignment is measured by PE-Core-G14-448, and renderability (Success Rate) is reported:

| Method | Semantic Alignment ↑ | Success Rate ↑ |
|--------|---------------------|----------------|
| LiveSketch | 0.158 | 100.0% |
| GPT-5.2 | 0.234 | 88.5% |
| Gemini 3 Pro | 0.243 | 86.2% |
| VAnim (SFT-only) | 0.268 | 95.2% |
| **VAnim (GRPO)** | **0.281** | **100.0%** |

VAnim-GRPO achieves both the highest semantic alignment and 100% executability; LiveSketch, while 100% renderable, lags far behind in semantic score (0.158), indicating frequent topology breakdowns; GPT-5.2/Gemini exhibit unclosed tags and ID hallucinations in long sequences, dropping executability to 86–88%.

### Ablation Study

| Configuration | Semantic Alignment ↑ | Success Rate ↑ | Notes |
|---------------|---------------------|----------------|-------|
| Full VAnim | 0.281 | 100.0% | Complete method |
| w/o Rendering-Aware RL | 0.268 (-0.013) | 95.2% (-4.8%) | Degrades to SFT, "motion is lazy," door only opens a crack |
| w/o Structure-Bound CoT | 0.255 (-0.026) | 98.6% (-1.4%) | Wrong object modified, rotates cabinet instead of door |
| w/o SSU (Appendix) | — | 62.3% | Naive per-frame generation, executability collapses |
| w/o input image (Appendix) | — | SSIM/temporal smoothness drops significantly | Visual anchor missing, identity preservation fails |

### Key Findings
- All three core components are indispensable: CoT ensures "correct node modification," SSU ensures "structure is not broken," RL ensures "willingness for large deformations." CoT contributes the most (semantic drops by 0.026), indicating explicit grounding is fundamental for non-rigid animation.
- Naive per-frame generation's Success Rate drops to 62.3%, directly validating the identity drift hypothesis: without SSU's hard constraint, the LLM randomly modifies static attributes, causing rendering failures.
- GRPO group size $G$ sweep shows a trade-off between "semantic alignment" and "identity preservation"; larger $G$ explores more but exacerbates identity drift.
- Visual input ablation shows $I_0$ is crucial for SSIM and temporal smoothness; code + prompt alone is insufficient for mapping visual objects to DOM IDs.

## Highlights & Insights
- Reformulating "sequence generation" as "sparse updates on persistent state" is the deepest insight of this work: it is equivalent to adding a "topology invariance" hard constraint to the generative model, and by controlling the token set, this is enforced at the architectural rather than loss level—cleaner than any regularization. The same idea can be transferred to HTML/UI editing, 3D scene graphs, CAD modification, robot trajectory editing, or any task involving "local temporal changes on persistent structures."
- Identification-First CoT concretely bridges the ReAct paradigm to "visual entity → DOM ID," and "ID consistency filtering" writes CoT executability into the data pipeline, avoiding CoT being just a prompting trick as in many papers.
- Using PE-Core-like video perception encoders for RL reward elegantly incorporates non-differentiable SVG rendering into the gradient chain: the combination of sparse +1/-1 format reward and dense semantic reward both prevents breakdowns and provides rich signals, serving as a template for other code-to-render tasks (HTML/CSS, shader, SQL→chart).

## Limitations & Future Work
- All data comes from Flaticon's Lottie-style works, which are structurally regular and come with IDs/groups; generalization to messy hand-crafted SVGs or tool-exported SVGs (no IDs, chaotic nesting, overuse of `g` wrappers) is questionable, and the authors explicitly note this as an open question.
- Rendering-Aware RL relies on real-time rendering via headless browsers and video encoder scoring, making each step much more expensive than standard RLHF, and difficult to reproduce in resource-constrained settings.
- Current VAnim only generates visual animation, lacking support for JavaScript-triggered interactive behaviors and multi-scene narratives, limiting end-to-end deployment in real UI/Web workflows.
- Evaluation mainly relies on PE-Core metrics, which share the same source as the training reward, posing a risk of metric circularity; the appendix supplements with independent metrics like InternVideo2, SSIM, and flow, but evaluation remains mostly automatic, lacking large-scale human assessment.
- SSU representation assumes "adjacent frame DOM topology is isomorphic," making support for animations involving "element appearance/disappearance" or "DOM restructuring" unnatural; the authors do not fully discuss this.

## Related Work & Insights
- **vs LiveSketch (Gal et al. 2024)**: LiveSketch optimizes strokes in pixel space using SDS, lacking structural awareness, leading to frequent breakdowns in closed shapes/occlusion; VAnim directly sparsely edits the SVG DOM, structurally preserving topology by construction, and completes inference in one pass rather than hundreds of iterations.
- **vs Keyframer / GPT-5.2 / Gemini 3 Pro**: General LLMs are confined to affine transformations in CSS/SMIL, approximating non-rigid deformations as translations; VAnim's RL signal pushes the policy to Bézier control points in the `d` attribute, enabling true path-level deformation.
- **vs DeepSVG / SVGformer and other static vector generation**: These focus on spatial composition, not time; VAnim is the first to transfer the LLM paradigm to open-domain vector animation, incorporating the temporal dimension via SSU without exploding context.
- **Insights**: "Sparse state update" can be a general paradigm for any generative task requiring temporal changes on persistent structures; "Structure-Bound CoT + consistency filtering" can be a standard for multimodal grounding data production; "Rendering-aware RL" provides a feasible path for incorporating non-differentiable renderers into code-to-X training.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The SSU + Identification-First CoT + rendering-aware GRPO trio is the first systematic solution for open-domain vector animation, redefining "animation generation = sparse updates on persistent structure."
- Experimental Thoroughness: ⭐⭐⭐⭐ Three strong baselines (optimization-based, closed-source LLM, in-house SFT-only) + two core ablations + four extended experiments in the appendix (SSU/input image/reward/group size) + user study, but lacks cross-domain (hand-crafted SVG) generalization evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ The Motivation section clearly articulates the three pain points of affine bias, context explosion, and identity drift; Figures 1/2/4 thoroughly explain data, representation, and inference; formulas are concise and well-illustrated, making it far more readable than similar multimodal long papers.
- Value: ⭐⭐⭐⭐ Open-source data + paradigm + framework, with direct application value for design toolchains, UI/Web automation, and educational animation; the only shortcoming is limited support for hand-crafted SVG and interactive logic.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lightning Unified Video Editing via In-Context Sparse Attention](lightning_unified_video_editing_via_in-context_sparse_attention.md)
- [\[CVPR 2026\] Identity-Preserving Image-to-Video Generation via Reward-Guided Optimization](../../CVPR2026/video_generation/identity-preserving_image-to-video_generation_via_reward-guided_optimization.md)
- [\[ACL 2026\] OSCBench: Benchmarking Object State Change in Text-to-Video Generation](../../ACL2026/video_generation/oscbench_benchmarking_object_state_change_in_text-to-video_generation.md)
- [\[ICLR 2026\] MoSA: Motion-Coherent Human Video Generation via Structure-Appearance Decoupling](../../ICLR2026/video_generation/mosa_motion-coherent_human_video_generation_via_structure-appearance_decoupling.md)
- [\[ICCV 2025\] ReCamMaster: Camera-Controlled Generative Rendering from A Single Video](../../ICCV2025/video_generation/recammaster_camera-controlled_generative_rendering_from_a_single_video.md)

</div>

<!-- RELATED:END -->
