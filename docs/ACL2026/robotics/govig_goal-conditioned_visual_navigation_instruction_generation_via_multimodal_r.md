---
title: >-
  [Paper Note] GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning
description: >-
  [ACL 2026][Robotics][Navigation instruction generation] GoViG proposes a new task for generating navigation instructions using **only first-person initial and goal observations**. The task is decomposed into two steps: "…
tags:
  - "ACL 2026"
  - "Robotics"
  - "Navigation instruction generation"
  - "egocentric"
  - "world model"
  - "multimodal reasoning"
  - "Anole-7B"
date: 2026-05-08
content_hash: 2a0d6499fbb7ec5c
---

# GoViG: Goal-Conditioned Visual Navigation Instruction Generation via Multimodal Reasoning

**Conference**: ACL 2026  
**arXiv**: [2508.09547](https://arxiv.org/abs/2508.09547)  
**Code**: https://github.com/F1y1113/GoViG (Available)  
**Area**: Robotics / Vision-Language Navigation (VLN)  
**Keywords**: Navigation instruction generation, egocentric, world model, multimodal reasoning, Anole-7B

## TL;DR
GoViG proposes a new task for generating navigation instructions using **only first-person initial and goal observations**. The task is decomposed into two steps: "imagine intermediate frames, then write instructions." By training Anole-7B under a dual objective of token-level MSE and label-smoothing CE, and employing one-pass or interleaved multimodal reasoning strategies, the method improves BLEU-4 from a baseline of 0.08 to 0.32, maintaining 0.27 on cross-domain real-world videos.

## Background & Motivation

**Background**: Mainstream research in VLN focuses on "instruction-following navigation," while the inverse "instruction generation from visuals" is primarily used for data augmentation. Representative methods (Speaker-Follower, LANA, C-Instructor, BEV-Instructor, NavRAG, MapInstructor) almost exclusively rely on **privileged inputs**—panoramas, action history, orientation, GPS, 3D bboxes, BEV maps, or scene graphs.

**Limitations of Prior Work**: (1) These privileged signals are unavailable in real-world deployments (blind assistance, household robots, disaster relief in unknown environments). (2) Compressing visuals into landmarks or text summaries for LLMs causes the loss of critical **spatial and semantic details**. (3) General MLLMs (GPT-4o, Gemini, Claude) lack a "mental rehearsal" mechanism—humans visualize intermediate scenes when planning a route, whereas models often jump directly from two-end observations to natural language, resulting in temporal discontinuities and directional errors.

**Key Challenge**: To achieve generalization, privileged inputs must be discarded in favor of egocentric RGB. However, providing only the start and end views is extremely sparse in information. Directly generating long instructions leads to "hallucinations." It is necessary to **explicitly generate intermediate states** as visual anchors for the instructions.

**Goal**: (1) Formulate the GoViG task where input consists only of $\mathcal{O}=\{o_1,\dots,o_n\}$ and $o_g$, and output is the natural language instruction $I$. (2) Design a unified autoregressive MLLM capable of simultaneous "frame prediction + instruction generation." (3) Construct a hybrid benchmark of real and synthetic data to verify cross-domain generalization.

**Key Insight**: Borrowing from the idea of "world models"—since instructions are essentially linguistic descriptions of "future observation sequences," the model should "imagine first, then speak" like a human. The task is decomposed into Navigation Visualization (predicting the next frame) and Instruction Generation with Visual Cues (writing instructions based on real + predicted frames).

**Core Idea**: Utilize Anole-7B (a unified image-text autoregressive model based on Chameleon) to **jointly learn visual token prediction and text token prediction using the same Transformer**. Two reasoning strategies—one-pass and interleaved—are used to choose between "visualizing everything then speaking" and "visualizing one step, then speaking one step."

## Method

### Overall Architecture
1. **Data**: Constructed R2R-Goal (74,737 synthetic trajectories + 1080 real videos). Each trajectory retains 6 initial egocentric frames + 1 goal frame + natural language instructions.
2. **Training**: Each trajectory is split into two sample types: (a) Navigation Visualization: given $k$ historical frames + goal frame, predict the next VQ image token sequence; (b) Instruction Generation: given start frames + $m-1$ intermediate frames + goal frame, predict text instruction tokens. These are interleaved within batches, sharing the same Anole-7B + LoRA (rank=16, qkv-projection only).
3. **Inference**: Use SSIM > $\tau=0.7$ as the "target reached" condition. The model is called iteratively to generate predicted frames, which are then fed back to generate instructions. Two scheduling strategies: **one-pass** predicts all frames before writing the instruction; **interleaved** updates the instruction immediately after each frame is predicted.

### Key Designs

1. **VQ-token Level Token Discrepancy Loss (Alternative to CE for Visual Supervision)**:
    - **Function**: Trains the model to predict next-frame VQ image tokens while giving partial credit to **visually similar codebook items**, preventing standard CE from penalizing "nearly correct visuals" as completely wrong.
    - **Mechanism**: For a ground-truth token embedding $\text{emb}_i$ at position $i$, calculate the MSE vector $\text{MSE}(\text{emb}_i, \mathcal{C}) \in \mathbb{R}^{1\times N}$ against the entire codebook $\mathcal{C}=\{\text{emb}_1,\dots,\text{emb}_N\}$. Take the dot product with the model’s predicted distribution $P(t_i) \in \mathbb{R}^{1\times N}$: $\mathcal{L}_{\text{vis}} = \sum_{i=1}^n \text{MSE}(\text{emb}_i, \mathcal{C}) \cdot P(t_i)$.
    - **Design Motivation**: Standard CE treats visual tokens as discrete categories, ignoring semantic continuity. Predicting a "brown door" as a "dark brown door" should be penalized less than predicting it as a "red chair." Table 3 shows that replacing $\mathcal{L}_{\text{vis}}$ with label-smoothing CE drops SSIM from 0.69 to 0.52 and PSNR from 20.02 to 15.35.

2. **One-Pass vs Interleaved Multimodal Reasoning**:
    - **Function**: Two scheduling strategies for the "imagine-express" process, catering to structured scenes vs. uncertain environments.
    - **Mechanism**: **One-pass** iteratively predicts frames until $\text{SSIM}(\hat{o}_{k+t}, o_g) > \tau$, then uniformly samples representative frames to generate the complete instruction $I = F_\Theta(\{o_1, \hat{o}_{i_1},\dots,\hat{o}_{i_{m-1}}, o_g\})$. **Interleaved** calls the model to update the instruction $I_t$ after each new predicted frame $\hat{o}_{k+t}$, using the previous instruction as context until resolution, simulating the "plan-as-you-go" cognitive process.
    - **Design Motivation**: One-pass is efficient (1.2× faster), suitable for known indoor scenes. Interleaved is more accurate (BLEU-4 0.32 vs 0.29, user rating 4.85 vs 4.52), suitable for unknown/real-world scenes. Both use the same trained model.

3. **Egocentric-only Minimalist Input Interface**:
    - **Function**: Ensures the method accepts only $\{o_1,\dots,o_n, o_g\}$ RGB frames, eliminating priors like panoramas, orientation, or GPS.
    - **Mechanism**: Visual inputs are discretized into 784 tokens/frame (256×256) via the Chameleon VQ tokenizer; text is processed via BPE, both fed into a 4096-token causal Transformer. Experiments found that a context size of 2 with 784 tokens/frame is the optimal trade-off.
    - **Design Motivation**: Only egocentric cameras are typically available in real deployment. Avoiding external pre-processing (landmarks, scene graphs) lowers engineering barriers and ensures stability during cross-domain transfer.

### Loss & Training
- **Joint Objective**: $\mathcal{L} = \mathcal{L}_{\text{vis}}$ (visualization samples) + $\mathcal{L}_{\text{ins}}$ (instruction samples, label-smoothing CE).
- Input labels are set to $-100$ so loss is only calculated on predicted targets.
- AdamW lr=$2\times 10^{-4}$, 20 epochs, 4× A100 80GB, global batch size 8.
- Tokenizer frozen; only LoRA adapters (rank=16, qkv-projection) in the Transformer are updated.

## Key Experimental Results

### Main Results

Instruction generation quality on R2R-Goal (BLEU-4 / CIDEr):

| Method | Val Seen BL-4 | Val Seen CI | Val Unseen BL-4 | Val Unseen CI | Test BL-4 | Test CI |
|---|---|---|---|---|---|---|
| Speaker-Follower | 0.10 | 0.08 | 0.09 | 0.06 | 0.09 | 0.06 |
| LANA | 0.05 | 0.05 | 0.05 | 0.06 | 0.05 | 0.03 |
| C-Instructor (Prev. SOTA) | 0.21 | 0.19 | 0.22 | 0.19 | 0.22 | 0.18 |
| GPT-4o + CoT | 0.08 | 0.17 | 0.09 | 0.16 | 0.08 | 0.17 |
| Gemini 3.0 | 0.09 | 0.13 | 0.09 | 0.14 | 0.08 | 0.12 |
| Claude 4 Opus | 0.10 | 0.15 | 0.09 | 0.13 | 0.09 | 0.14 |
| Anole-7B + CoT | 0.10 | 0.14 | 0.09 | 0.13 | 0.09 | 0.10 |
| **Anole-7B + One-pass (Ours)** | 0.34 | 0.20 | 0.29 | 0.18 | 0.29 | 0.19 |
| **Anole-7B + Interleaved (Ours)** | **0.36** | **0.22** | **0.32** | **0.20** | **0.33** | 0.18 |

Navigation Visualization Quality (Val Unseen):

| Method | SSIM ↑ | PSNR ↑ | LPIPS ↓ | DreamSim ↓ |
|---|---|---|---|---|
| GPT-4o + DALL·E | 0.29 | 9.57 | 0.72 | 0.61 |
| Anole-7B (Direct) | 0.50 | 14.98 | 0.39 | 0.27 |
| **Ours** | **0.69** | **20.02** | **0.27** | **0.13** |

Practical Usability (Success Rate of navigation agents following generated instructions):

| Instruction Generator | ETPNav SR | ETPNav SPL | BEVBert SR | BEVBert SPL |
|---|---|---|---|---|
| Human Annotation | 0.36 | 0.28 | 0.34 | 0.27 |
| GPT-4o + CoT | 0.25 | 0.17 | 0.24 | 0.17 |
| C-Instructor | 0.29 | 0.19 | 0.27 | 0.18 |
| Anole-7B + One-pass | 0.31 | 0.20 | 0.29 | 0.21 |
| **Anole-7B + Interleaved** | **0.34** | **0.25** | **0.33** | 0.25 |

### Ablation Study

| Config | SSIM | PSNR | LPIPS | DreamSim | Description |
|---|---|---|---|---|---|
| w/o $\mathcal{L}_{\text{vis}}$ (using label-smoothing CE) | 0.52 | 15.35 | 0.36 | 0.23 | Visual tokens as discrete classes |
| **w/ $\mathcal{L}_{\text{vis}}$ (Token Discrepancy Loss)** | **0.69** | **20.02** | **0.27** | **0.13** | +17 SSIM points |

Context-size / token-length trade-off: Factors improve as context moves from 1 to 2. However, expanding to context 4-5 requires compressing each frame from 784 to 400 tokens, which leads to a performance drop. This indicates that **single-frame information density is more critical than the number of frames**.

### Key Findings
- **Interleaved universally outperforms one-pass**: +3 points in BLEU-4, higher user rating (4.85 vs 4.52), and +3 points in follower success rate.
- **Token Discrepancy Loss is the core of visual quality**: Simply replacing CE with the similarity-weighted loss yields a 17-point SSIM gain.
- **Cross-domain stability**: BLEU-4 (0.27) is 3x higher than GPT-4o (0.09) on real videos without specific fine-tuning, highlighting explicit visualization as the key to generalization.
- **Nearing human performance**: Instructions generated by the interleaved strategy allow agents to achieve 0.34 SR, close to the 0.36 SR of human annotations.

## Highlights & Insights
- **"World-Modelized" Instruction Generation**: Redefining instruction generation as predicting future observations and then summarizing them grounded instructions in a visual representation, which can be extended to tasks like video captioning or surgical reports.
- **Simplicity of Token-Similarity Loss**: Achieving significant SSIM gains without extra network structures or second-stage training makes this a highly portable trick for any LM using VQ tokenizers.
- **Decoupled Scheduling**: One-pass and interleaved strategies derive from the same model, suggesting that MLLM scheduling strategies can be decoupled from core model capabilities.

## Limitations & Future Work
- **Lack of real-time environmental feedback**: Incorrect intermediate frame predictions can lead to cumulative errors in subsequent instructions.
- **Context length constraints**: In long corridors or multi-floor scenes, the 4096-token limit restricts the number of usable frames.
- **Real-world training data scarcity**: 1080 real videos were only used for testing; scaling requires larger-scale egocentric datasets.
- **Visual realism**: While useful as semantic anchors, the generated images remain blurry/distorted compared to standard video generation models.

## Related Work & Insights
- **vs C-Instructor**: C-Instructor uses LLMs with landmark vocabularies (BLEU-4 0.22); GoViG reaches 0.32 without landmarks, proving that "explicit visualization" is more effective than "explicit landmarks."
- **vs GPT-4o + DALL·E**: DALL·E's visual predictions (SSIM 0.29) are much lower because it lacks navigation context. GoViG’s end-to-end autoregressive prediction maintains spatial coherence.
- **vs World Models (Dreamer)**: While traditional world models predict in latent space for control, GoViG predicts in RGB token space for language generation—representing a "linguistic variant" of world models.

## Rating
- Novelty: ⭐⭐⭐⭐ New task definition (egocentric-only) + unified MLLM "imagine-then-speak" framework.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Coverage of 9+ baselines, 4 metrics, user studies, and cross-domain zero-shot tests.
- Writing Quality: ⭐⭐⭐⭐ Clear task definition and pipeline diagrams, though some table layouts are dense.
- Value: ⭐⭐⭐⭐ Strong first "no-privileged-input" baseline for blind assistance and household robotics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] GROKE: Vision-Free Navigation Instruction Evaluation via Graph Reasoning on OpenStreetMap](groke_vision-free_navigation_instruction_evaluation_via_graph_reasoning_on_opens.md)
- [\[ACL 2026\] Cultivating Forensic Reasoning for Generalizable Multimodal Manipulation Detection](cultivating_forensic_reasoning_for_generalizable_multimodal_manipulation_detecti.md)
- [\[CVPR 2026\] Semantic Audio-Visual Navigation in Continuous Environments](../../CVPR2026/robotics/semantic_audio-visual_navigation_in_continuous_environments.md)
- [\[NeurIPS 2025\] Benchmarking Egocentric Multimodal Goal Inference for Assistive Wearable Agents](../../NeurIPS2025/robotics/benchmarking_egocentric_multimodal_goal_inference_for_assist.md)
- [\[NeurIPS 2025\] EfficientNav: Towards On-Device Object-Goal Navigation with Navigation Map Caching and Retrieval](../../NeurIPS2025/robotics/efficientnav_towards_on-device_object-goal_navigation_with_navigation_map_cachin.md)

</div>

<!-- RELATED:END -->
