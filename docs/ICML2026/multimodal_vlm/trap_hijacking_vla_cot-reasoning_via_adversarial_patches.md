---
title: >-
  [Paper Note] TRAP: Hijacking CoT Reasoning of VLA with Adversarial Patches for Targeted Behavior Attacks
description: >-
  [ICML 2026][Multimodal VLM][VLA] TRAP is the first targeted behavior hijacking attack against reasoning VLAs. By utilizing tablecloth-sized physical adversarial patches…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "VLA"
  - "Chain-of-Thought"
  - "Adversarial Patch"
  - "Targeted Behavior Hijacking"
  - "Physical Attack"
date: 2026-05-08
content_hash: 9e3f626dab61b401
---

# TRAP: Hijacking CoT Reasoning of VLA with Adversarial Patches for Targeted Behavior Attacks

**Conference**: ICML 2026  
**arXiv**: [2603.23117](https://arxiv.org/abs/2603.23117)  
**Code**: TRAP-website (Project Page)  
**Area**: VLA Security / Adversarial Attack / Embodied AI  
**Keywords**: VLA, Chain-of-Thought, Adversarial Patch, Targeted Behavior Hijacking, Physical Attack

## TL;DR
TRAP is the first targeted behavior hijacking attack against reasoning VLAs. By utilizing tablecloth-sized physical adversarial patches, it hijacks the CoT reasoning (bounding boxes/trajectories/sub-tasks) of the VLA, causing the robot to perform actions like "handing a knife to a person" while the user instruction remains "pick up the apple." It achieves an average ASR of 52.54% across MolmoAct, GraspVLA, and InstructVLA paradigms. Real-world printed patches achieve an 86.7% interference success rate and a 33.3% full control rate in occlusion-free deployment on GraspVLA.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models enable robots to perform manipulation tasks in open-world settings (OpenVLA, $\pi_0$, GraspVLA) through end-to-end training. Recently, the integration of Chain-of-Thought (CoT) reasoning has produced "reasoning VLAs" that generate intermediate reasoning—such as sub-task decomposition, target bounding boxes, or predicted trajectories—before generating actions. CoT not only improves generalization but is also believed to enhance interpretability and security.

**Limitations of Prior Work**: (1) Existing VLA adversarial attacks are primarily untargeted, aiming to disrupt perception (UPA-RFAS) or action generation (RoboticAttack) to cause task failure without precise control. (2) These attacks target vanilla VLAs and do not investigate the new attack surfaces introduced by CoT. (3) CoT causes the model to explicitly output "task intent" (e.g., "I will first grab the apple, then hand it to the person"), providing a precise entry point for attackers.

**Key Challenge**: While CoT is promoted as a means to "enhance VLA security and interpretability," preliminary experiments (Table 1) show that when instructions and CoT conflict, GraspVLA almost entirely follows the CoT (TSR=94.2% vs 0%). In other models, CoT carries at least comparable influence to the instruction. Thus, CoT acts as a new attack surface rather than a safety net; attackers can control the final action by hijacking the intermediate CoT without modifying the user instruction.

**Goal**: To demonstrate that CoT reasoning can be hijacked by adversarial patches (e.g., a printable tablecloth) without modifying instructions, forcing the VLA to execute attacker-specified target behaviors. The objective is to verify the universality of this attack across three representative reasoning VLA paradigms (discrete-token integrated, continuous-regression integrated, and hierarchical).

**Key Insight**: A preliminary experiment quantifies the "causal role of CoT in action generation" through instruction masking and cross-sample shuffling, confirming that CoT is a strong causal signal. Based on this, a patch attack is designed using joint optimization of "CoT hijacking loss + action loss + stealthiness loss."

**Core Idea**: The adversarial objective is shifted from "causing direct action errors" to "forcing CoT to output attacker-defined content." The CoT hijacking loss uses cross-entropy against target CoT token sequences $R^*$. The action loss provides a fallback using CE or MSE depending on whether the VLA is discrete or continuous. Content loss, TV loss, and DIP optimization are added to ensure physical printability and visual stealth.

## Method

### Overall Architecture

Threat Model: White-box (known VLA architecture/parameters/gradients). The attacker can place an adversarial patch (e.g., a tablecloth or wall sticker) in the scene, while user instructions remain benign. The attack requires the patch to remain effective throughout the entire multi-step rollout.

Attack Pipeline: (1) Offline collection of a clean trajectory $\mathcal{D} = \{(O, R, a)\}$. (2) Optimization of patch $\delta$ to satisfy $\min_{\delta} \mathbb{E}_{\tau \sim \mathcal{D}}[\mathcal{L}_{\mathrm{cot}} + \lambda_1 \mathcal{L}_{\mathrm{action}} + \lambda_2 \mathcal{L}_{\mathrm{content}} + \lambda_3 \mathcal{L}_{\mathrm{tv}}]$. (3) Updating $\delta$ via PGD: $\delta_{t+1} = \mathrm{Proj}(\delta_t + \eta \nabla L)$. (4) Physical deployment involving homography transformation, color calibration MLP, and EoT data augmentation.

### Key Designs

1.  **CoT Hijacking Loss as the Primary Signal**:
    - **Function**: Forces the VLA's intermediate CoT to output an attacker-specified $R^*$ (e.g., "Target is knife, bbox at (x,y)"), thereby hijacking the entire reasoning-action pipeline.
    - **Mechanism**: Most mainstream reasoning VLAs use VLMs for next-token prediction to generate CoT (whether it be sub-task text, bbox coordinates, or trajectory points). Thus, CoT loss is unified as CE: $\mathcal{L}_{\mathrm{cot}} = -\sum_{t=1}^T \log P_\theta(r_t^* | r_{<t}^*, \tilde{O}, I)$, where adversarial observation $\tilde{O} = (1-M) \odot O + M \odot \delta$. Preliminary experiments show CoT almost entirely dominates actions in GraspVLA (TSR=94.2% after shuffling). Consequently, a CoT-only attack ($\mathrm{TRAP}_{\mathrm{CoT\text{-}only}}$) achieves 69.04% ASR on GraspVLA.
    - **Design Motivation**: Compared to direct action attacks, CoT loss leverages the advantage of CoT as a linguistic sequence with clear discrete supervision, resulting in more stable gradients and precise targets. Since CoT is a mid-level abstraction, attacking it maintains consistency across multiple action time steps.

2.  **Dual-mode Action Loss (Discrete/Continuous)**:
    - **Function**: Covers two types of VLA action heads (autoregressive tokens vs. diffusion/flow regression) to ensure CoT hijacking reliably translates to the action layer.
    - **Mechanism**: For models like MolmoAct using discrete-token actions (quantized into bins), $\mathcal{L}_{\mathrm{action}}^{\mathrm{disc}} = -\log P_\theta(a^* | R^*, \tilde{O}, I)$ is used. For continuous regression models like GraspVLA/InstructVLA, MSE on trajectory waypoints is used: $\mathcal{L}_{\mathrm{action}}^{\mathrm{cont}} = \|f_{\mathrm{traj}}(a) - f_{\mathrm{traj}}(a^*)\|_2^2$. On InstructVLA, the CoT-only attack reaches only 4.03% ASR (due to mode collapse), but increases to 33.71% with action loss.
    - **Design Motivation**: The strength of CoT-to-action coupling varies; it is strong in GraspVLA but weak in InstructVLA (hierarchical). A unified action loss ensures the attack succeeds regardless of coupling strength.

3.  **DIP + Color Calibration + EoT for Physical Stealth**:
    - **Function**: Transforms the adversarial patch from "visual noise" into printable "tablecloths or decorations" that remain effective under real camera/lighting conditions.
    - **Mechanism**: (a) Content loss $\mathcal{L}_{\mathrm{content}} = \frac{1}{C_l H_l W_l} \|\phi_l(\delta) - \phi_l(I_{\mathrm{ref}})\|_2^2$ aligns patch features with a reference image at intermediate CNN layers. (b) TV loss suppresses high-frequency artifacts. (c) Deep Image Prior (DIP) optimizes CNN parameters $\theta$ such that $\delta = f_\theta(z)$, utilizing implicit regularization for visual coherence. (d) Homography simulates projection. (e) MLP learns color distortion. (f) Expectation over Transformation (EoT) improves robustness.
    - **Design Motivation**: Pure PGD patches are high-frequency noise easily detected by humans. The content + TV + DIP combination makes the patch look like a "patterned tablecloth," ensuring stealth in the physical world.

### Loss & Training

PGD projected gradients: $\delta_{t+1} = \mathrm{Proj}_{\|\cdot\|_\infty \le \epsilon}(\delta_t + \eta \nabla_\delta L)$ with a pixel update step of $8/255$ and a batch size of 4. Regularization is annealed: higher weight for content+TV early on to ensure stealth, followed by decay to allow the attack objective to dominate. Optimization was conducted on a single H800 GPU; simulator evaluation used an RTX 4090 with 25 layouts for training and 10 unseen layouts for testing per task (175 rollouts per task).

## Key Experimental Results

### Main Results: Attack Performance Across Three VLAs

| Method | MolmoAct ASR / Score | InstructVLA ASR / Score | GraspVLA ASR / Score | Average ASR / Score |
| :--- | :--- | :--- | :--- | :--- |
| Random Noise | 0.97 / -0.377 | 3.39 / -0.328 | 0.32 / -0.306 | 1.56 / -0.337 |
| Action Attack (TMA-like) | 9.68 / 0.128 | 6.77 / -0.274 | 0.00 / -0.295 | 5.48 / -0.147 |
| $\mathrm{TRAP}_{\mathrm{CoT\text{-}only}}$ | 49.52 / 0.342 | 4.03 / -0.033 | 69.04 / 0.390 | 40.86 / 0.233 |
| **TRAP** | 48.06 / **0.390** | **33.71** / **0.172** | **75.84** / **0.425** | **52.54** / **0.329** |
| TRAP (unseen layout) | 48.00 / 0.183 | 31.60 / 0.131 | 75.20 / 0.402 | 51.60 / 0.239 |

TRAP significantly outperforms Action Attack (Average ASR 52.54% vs 5.48%). CoT-only performance is comparable to TRAP on GraspVLA (69 vs 75) but fails on InstructVLA (4 vs 33), validating the necessity of action loss in hierarchical VLAs. Performance on unseen layouts remains high (51.60 vs 52.54), indicating the patch learns layout-invariant features.

### Robustness to Instruction Variants

| Instruction Variant | MolmoAct ASR | InstructVLA ASR |
| :--- | :--- | :--- |
| Original | 72.0 | 67.4 |
| Paraphrasing | 70.6 | 25.1 |
| Extra-Context | 60.0 | 44.8 |

MolmoAct remains robust to variants (trajectory-based CoT is less sensitive to linguistic shifts), while InstructVLA is more fragile due to text-based decomposition, with ASR dropping significantly after paraphrasing.

### Key Findings

- **CoT is a strong causal signal in reasoning VLAs**: Cross-sample shuffling experiments show GraspVLA follows the CoT even when it contradicts the instructions.
- **Patches learn "concept-visual feature" mappings**: Attention visualization (Figure 4) shows patches shift VLA attention from the benign target (e.g., orange) to the adversarial target (e.g., coke can) at a concept level.
- **High generalization across layouts**: The minor drop in ASR on unseen layouts suggests the patch captures model-level vulnerabilities rather than layout-specific shortcuts.
- **Stealth and effectiveness are compatible**: DIP optimization allows the patch to resemble a standard patterned tablecloth with only a marginal decrease in attack success.

## Highlights & Insights

- **First targeted attack on reasoning VLAs**: Moves beyond simple performance degradation to precise hijacking of robotic behavior.
- **CoT as a dual-edged sword**: CoT improves generalization but exposes an explicit reasoning chain that attackers can exploit.
- **Cross-paradigm applicability**: Effective across discrete, continuous, and hierarchical architectures.
- **Physical-world readiness**: Demonstrated effectiveness with printed patches in physical environments.
- **Lightweight defense solutions**: Provides millisecond-level detection patches (bbox checks, trajectory consistency) that perform near GPT-5 baselines.

## Limitations & Future Work

- **White-box dependency**: Core experiments require model gradients; black-box transferability remains a challenge.
- **Task scope**: Limited to pick-and-place tasks; long-horizon task attacks are unexplored.
- **Scene-specific patches**: Patches currently require optimization for specific tasks/scenes rather than being universal.
- **Evolving defenses**: The game between detectors and adversarial optimization (e.g., "detector-aware" patches) is an area for future research.

## Related Work & Insights

- **vs RoboticAttack / TMA**: Those focus on untargeted action degradation; TRAP achieves targeted hijacking via CoT.
- **Comparison to LLM Jailbreaking**: Unlike prompt-based jailbreaking, TRAP utilizes physical environment perturbations to manipulate the reasoning-action pipeline.
- **Inspiration**: As embodied agents increasingly rely on explicit reasoning chains (CoT, ToT), these chains become primary attack surfaces. Future AI safety research must prioritize audits of reasoning processes.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐⭐ 
- Value: ⭐⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] MIP against Agent: Malicious Image Patches Hijacking Multimodal OS Agents](../../NeurIPS2025/multimodal_vlm/mip_against_agent_malicious_image_patches_hijacking_multimod.md)
- [\[ICML 2026\] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds](any3d-vla_enhancing_vla_robustness_via_diverse_point_clouds.md)
- [\[AAAI 2026\] Phantom Menace: Exploring and Enhancing the Robustness of VLA Models Against Physical Sensor Attacks](../../AAAI2026/multimodal_vlm/phantom_menace_exploring_and_enhancing_the_robustness_of_vla_models_against_phys.md)
- [\[ICML 2026\] On the Adversarial Robustness of Large Vision-Language Models under Visual Token Compression](on_the_adversarial_robustness_of_large_vision-language_models_under_visual_token.md)
- [\[ICML 2026\] VLANeXt: A Recipe for Building Robust VLA Models](vlanext_recipes_for_building_strong_vla_models.md)

</div>

<!-- RELATED:END -->
