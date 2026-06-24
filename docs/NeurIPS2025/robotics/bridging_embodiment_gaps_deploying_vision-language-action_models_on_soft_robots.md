---
title: >-
  [Paper Note] Bridging Embodiment Gaps: Deploying Vision-Language-Action Models on Soft Robots
description: >-
  [NeurIPS 2025][Robotics][Vision-Language-Action] This paper deploys Vision-Language-Action (VLA) models on a soft continuum robotic arm (Embuddy) for the first time. It reveals that out-of-the-box pre-trained policies for rigid robots fail completely due to kinematic and dynamic discrepancies. However, targeted fine-tuning on a small demonstration dataset of soft robots successfully bridges the rigid-to-soft embodiment gap, enabling the soft platform to achieve task success r…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "Vision-Language-Action"
  - "Soft Continuum Robotic Arm"
  - "Bridging Embodiment Gaps"
  - "OpenVLA-OFT"
  - "π₀"
  - "Safe Human-Robot Interaction"
date: 2026-05-08
content_hash: e759e3cc7daa40a1
---

# Bridging Embodiment Gaps: Deploying Vision-Language-Action Models on Soft Robots

**Conference**: NeurIPS 2025  
**arXiv**: [2510.17369](https://arxiv.org/abs/2510.17369)  
**Code**: [https://huggingface.co/HCSuMoss](https://huggingface.co/HCSuMoss) (open-sourcing the first manipulation demonstration dataset for soft robots)  
**Area**: Robotics / Soft Robotics / VLA / Cross-Embodiment Transfer / Safe Human-Robot Interaction  
**Keywords**: Vision-Language-Action, Soft Continuum Robotic Arm, Bridging Embodiment Gaps, OpenVLA-OFT, π₀, Safe Human-Robot Interaction

## TL;DR

This paper deploys Vision-Language-Action (VLA) models on a soft continuum robotic arm (Embuddy) for the first time. It reveals that out-of-the-box pre-trained policies for rigid robots fail completely due to kinematic and dynamic discrepancies. However, targeted fine-tuning on a small demonstration dataset of soft robots successfully bridges the rigid-to-soft embodiment gap, enabling the soft platform to achieve task success rates comparable to a rigid UR5 arm in grasping and human-robot interaction tasks.

## Background & Motivation

**Background**: Vision-Language-Action (VLA) models are currently among the most promising general-purpose control frameworks in robotics, unifying visual perception, natural language understanding, and action generation within a single multimodal policy. From CLIPort and SayCan to RT-2 and OpenVLA, VLA models have achieved remarkable progress in task generalization and setting zero-shot capability. However, almost all existing VLA models and deployment efforts focus on **rigid serial-link robotic arms** (such as UR5, Franka, etc.). These robots possess predictable kinematics, which simplifies the control problem, but they exhibit fundamental limitations in safety and adaptability—rigid structures can cause injury during close contact with humans and cannot flexibly adapt to complex unstructured environments.

**Limitations of Prior Work**: A core blind spot in current VLA research is that all mainstream datasets and benchmarks (e.g., Open X-Embodiment, LIBERO, ALOHA) rely on rigid robotic platforms. This raises two critical issues. First, the applicability of VLAs is restricted to scenarios that do not require compliant interaction, lacking solutions for human-centric environments that require **intrinsic safety** (such as assistive feeding and close-range collaboration). Second, the non-linear, underactuated dynamics of soft robots are fundamentally different from those of rigid robots. Whether policies trained on rigid arms can transfer to soft platforms remains a completely unknown, open question. Existing cross-embodiment transfer works are mainly conducted between different rigid platforms that share similar inverse kinematics structures and appearances, making transfer relatively easy. In contrast, soft continuum robotic arms feature entirely different morphologies, kinematic constraints, and dynamic behaviors—a setup that has never been covered in VLA benchmarks.

**Key Challenge**: The semantic understanding and task generalization capabilities of VLA models must be integrated with safe and reliable physical interaction in the real world. Yet, the most natural pathway to achieve the latter—using soft robots—has been completely overlooked due to the lack of corresponding datasets, deployment pipelines, and systematic evaluations. Although rigid robots have controllable kinematics, they fundamentally cannot offer the intrinsic safety needed to share environments with humans. Conversely, while soft robots are naturally safe, their complex non-linear dynamics make learning-based control methods difficult to apply directly.

**Goal**: This work focuses on three core questions: (1) Can VLA policies directly transfer to soft robots without fine-tuning? (2) If not, what kind of fine-tuning pipeline can bridge this embodiment gap? (3) How do different VLA model architectures (OpenVLA-OFT vs. π₀) perform in comparison on soft platforms?

**Key Insight**: The authors' key observation is that soft continuum robotic arms (such as Embuddy) possess two critical advantages that make them naturally suited for deployment in human environments: their underactuated structure means that regardless of motor positions, the soft segments can always be deformed by external forces, which, combined with a lightweight design (only 5kg), restricts inertial forces. If the high-level reasoning capabilities of VLAs can be combined with the physical safety of soft robots, a previously non-existent optimal point between safety and intelligence can be achieved.

**Core Idea**: By constructing the first soft robot demonstration dataset and a structured fine-tuning pipeline, this paper demonstrates that VLA models can be successfully deployed on soft continuum robotic arms after being fine-tuned on a small amount of data, achieving performance comparable to rigid robots in manipulation tasks while retaining the intrinsic safety characteristics of the soft platform.

It is worth emphasizing that the gap between soft and rigid robots is not as simple as "different control parameters." There is an analytical mapping between the joint space and Cartesian space of rigid robots (inverse kinematics has closed-form or highly efficient numerical solutions), whereas the deformation of soft continuum robots is continuously distributed with infinite degrees of freedom, heavily coupled with material non-linearities and gravity. This means that conceptually, the "embodiment" of a rigid robot is mainly reflected in its joint count and link geometry, while the "embodiment" of a soft robot is deeply embedded in its material mechanics and structural dynamics. Therefore, the ability of VLAs to bridge this gap through fine-tuning purely at the vision and action level suggests that the high-level semantic and temporal patterns learned by VLAs do indeed possess a considerable degree of physical invariance.

## Method

### Overall Architecture

The methodology in this paper adopts a structured pipeline design, covering a complete closed loop from task design, data collection, preprocessing, and model adaptation to evaluation. The overall pipeline is as follows: First, three representative manipulation tasks are defined to span the capabilities of the soft robot. Then, a data collection environment is established to record multimodal demonstration data via teleoperation. Next, raw data are converted into standardized formats (RLDS for OpenVLA-OFT and LeRobot for π₀). Based on this, the two VLA models are fine-tuned under comparable conditions. Finally, inference evaluation is performed on the designed tasks to evaluate the policies' success rates and qualitative behaviors.

Inputs include third-person camera images, wrist camera images, proprioceptive states (end-effector poses represented as an 8-dimensional vector $s = [x, y, z, r, p, y, \text{pad}, g]$), and natural language task instructions. Outputs are delta action vectors (7-dimensional: $a = [\Delta x, \Delta y, \Delta z, \Delta r, \Delta p, \Delta y, g]$), which directly control the robot's motion in Cartesian space.

### Key Designs

1. **Soft Continuum Robot Platform Embuddy**:

    - **Function**: Serving as the core physical platform of the experiments, Embuddy is a custom-designed continuum robotic arm used to validate the feasibility of VLAs on compliant platforms.
    - **Mechanism**: Embuddy consists of three modular segments, each containing a standard rotary joint and a soft continuum segment. The continuum segment employs a **tendon-driven** mechanism to bend within a plane (constrained by an incompressible backbone) and is constructed using 3D-printed thermoplastic polyurethane (TPU). The total height of the robot is approximately 1m, comparable in scale to standard serial-link robotic arms, but its workspace is limited by the bending angles of each segment: the maximum bending of the first segment is 80°, while the second and third segments bend up to 50° each.
    - **Design Motivation**: The **intrinsic safety** of Embuddy is derived from two key properties. First, the underactuated structure ensures that no matter where the motors are positioned, the flexible segments can always be deformed by external forces—providing a passive safety mechanism that rigid robots cannot offer. Second, the entire robot weighs only 5kg, which significantly confines inertial forces and prevents severe injury even in the event of a collision. These two properties make Embuddy naturally suitable for close-range interactive tasks in human-shared environments. In the experiments, Embuddy utilizes the same gripper and camera configurations as the UR5 to ensure a fair cross-platform comparison.

2. **Teleoperation and Inverse Kinematics Based on the Piecewise Constant Curvature Model**:

    - **Function**: Map the Cartesian space commands of human operators into tendon length control signals of the soft robot, achieving fluent teleoperation data collection.
    - **Mechanism**: A **Piecewise Constant Curvature (PCC) model** is adopted as the inverse kinematics scheme. The PCC model approximates each continuum segment as a constant-curvature arc, thereby relating tendon lengths to the modeled shape to determine the end-effector pose. Teleoperation data are collected at a frequency of 5Hz using a 3dconnexion space mouse as the joystick controller. Each demonstration episode records third-person images, wrist images, proprioceptive states, and language instructions. After collection, the data are cropped and downsampled to 256×256 resolution, and the wrist view is flipped to make it more intuitive.
    - **Design Motivation**: The PCC model is the most commonly used simplified model for soft robot control, providing the computational efficiency required for real-time control while ensuring sufficient accuracy. Compared to the Finite Element Method (FEM), PCC sacrifices some accuracy but gains real-time capability, which is crucial for teleoperation data collection. Angle differences are processed using the periodic handling formula $\Delta = ((\Delta + \pi) \bmod 2\pi) - \pi$ to avoid erroneous large jumps at the $[-\pi, \pi]$ boundaries.

3. **Differentiated Fine-Tuning Strategies for Two VLA Architectures**:

    - **Function**: Design fine-tuning schemes adapted to the architectural characteristics of OpenVLA-OFT and π₀ respectively, balancing computational efficiency and accuracy.
    - **Mechanism**:
        - **OpenVLA-OFT Fine-tuning**: Due to the massive number of parameters in its Llama 2 7B backbone, **LoRA (rank=32) low-rank adaptation** is applied instead of full fine-tuning. The inputs contain proprioceptive states and dual-perspective images (third-person + wrist), and a continuous action head is trained using an L1 regression objective. Key hyperparameters are set to an action chunk size of 8, a batch size of 8, a learning rate of $5 \times 10^{-4}$, and a maximum of 200k training steps (Task 1/2) or 150k steps (Task 3), with the learning rate decaying by a factor of 10 after 120k steps. For Task 2, which requires language differentiation, the FiLM module is enabled to enhance language understanding, and the training steps are increased to 240k. Training for 150k steps on an A100 takes about 56 hours.
        - **π₀ Fine-tuning**: Since its backbone PaliGemma has only 3B parameters, **full-parameter fine-tuning** is performed directly. The training uses a batch size of 32, a learning rate of $2.5 \times 10^{-5}$, a 1000-step warmup, and a cosine decay strategy, with a maximum of 30k training steps. Training on an H100 takes about 11 hours.
    - **Design Motivation**: The choice between the two fine-tuning strategies stems directly from the difference in model sizes—full-parameter fine-tuning of the 7B OpenVLA-OFT is computationally prohibitive and prone to overfitting on small demonstration datasets, whereas LoRA efficiently adapts the model under low-rank constraints while retaining most of its pre-trained knowledge. Conversely, the smaller scale of the 3B π₀ makes full-parameter fine-tuning computationally feasible and capable of leveraging limited data effectively. Both models use an action chunk size of 8 to ensure a fair comparison. Data augmentation (random cropping, brightness/contrast/saturation/hue adjustments) is applied only to the input images of OpenVLA-OFT.

### Task Design and Evaluation Protocol

This work designs three representative manipulation tasks covering varying difficulty levels from simple pick-and-place to close-range human-robot interaction:

- **Task 1**: "Put the orange in the plate"—A simple pick-and-place task, where four common foodstuffs (orange, milk, yogurt, baguette) are randomly placed in the workspace, and the plate's position is roughly fixed. The dataset contains 50 demonstration episodes.
- **Task 2**: "Put the X in the plate" (where X can be orange or milk)—A selective pick-and-place task, requiring the model to correctly identify the target object based on language instructions. The dataset contains 100 demonstration episodes (50 per class).
- **Task 3**: "Feed the person with marshmallow"—A close-range human-robot interaction task, requiring the robot to pick up a marshmallow from a plate and feed it to a person in the scene. The dataset contains only 20 demonstration episodes, making it the most challenging task.

Evaluation employs the standard success rate over 10 trials as the quantitative metric, with objects randomly placed in the workspace in each trial.

### Inference and Deployment Architecture

During inference, a **remote-local communication architecture** is adopted, which is a common deployment pattern in large-model-driven robot control. A local PC connects to the robot and collects observations in real time (third-person images, wrist images, proprioceptive states, language instructions), packaging and sending the data to a remote GPU server for model forward inference. The server returns an action chunk (an action sequence of size 8). The local PC executes each action in sequence before collecting observations again, continuing this loop until the task is complete or the maximum number of steps is reached. This asynchronous "collect $\rightarrow$ transmit $\rightarrow$ infer $\rightarrow$ transmit $\rightarrow$ execute" loop introduces inevitable network latency, but since the action chunk mechanism predicts 8-step actions at once, it effectively amortizes the impact of single-step latency.

Despite the network latency (using a remote HPC cluster in the soft robot experiments resulted in latency significantly higher than the Azure virtual machine used in the UR5 experiments), the soft robot still maintained a control frequency of at least 25 Hz (OpenVLA-OFT on H100: 25.1 Hz, π₀ on H100: 38.0 Hz, OpenVLA-OFT on A100 on UR5: 32.3 Hz). π₀'s inference speed is significantly faster than OpenVLA-OFT, which is primarily due to its smaller backbone network (3B vs. 7B) and JAX's highly efficient parallel execution. Notably, even under the slowest configuration (25.1 Hz), the control frequency remains much higher than the data collection frequency (5 Hz), theoretically providing sufficient bandwidth for real-time closed-loop visual control.

### Loss & Training

- **OpenVLA-OFT**: Employs **L1 regression loss** to train the continuous action head, directly regressing the delta action vector of the end-effector pose. Compared to digitized token predictions (as in the original OpenVLA), continuous output prevents discretization artifacts and increases action precision. Training is considered complete when the training loss stabilizes around 0.01.
- **π₀**: Employs the **Conditional Flow Matching** objective. During training, a noisy action sequence is generated, and the model learns to predict the "denoising" flow that maps the noise back to the ground-truth action. Compared to diffusion models, this flow-based method offers advantages in inference speed and directly outputs continuous actions. It uses a cosine learning rate decay strategy, decaying from $2.5 \times 10^{-5}$ to $2.5 \times 10^{-6}$.

## Key Experimental Results

### Main Results

The core experiments of the paper validate two key hypotheses: (1) out-of-the-box VLA policies cannot work directly on soft robots; (2) fine-tuning can effectively bridge the embodiment gap.

**Zero-shot Transfer Experiments**: All unfine-tuned VLA models (including π₀, which is renowned for cross-embodiment generalization) fail completely on Embuddy, with a success rate of 0%. The root cause of this failure lies in the mismatch of dynamics mapping between soft and rigid robots—when the model generates motion instructions suitable for a rigid robot arm, Embuddy gets "stuck" during execution because of the bending angle limits of each segment (80° for the first segment, 50° each for the second and third), failing to continue following the predicted trajectory. Specifically, the joints of a rigid arm can rotate freely within their full range, allowing the generated motion sequence to contain large-amplitude, fast displacements. In contrast, the bending of a soft continuum segment is a continuous deformation process driven by tendon tension, and its response speed and motion range are strictly limited by material elasticity and structural constraints. This intrinsic dynamic mismatch makes out-of-the-box policies completely unusable, clearly verifying the existence of a significant domain gap between rigid and soft robots.

This result also prompts an interesting reflection: while the cross-embodiment pre-training of π₀ spans 7 different rigid platforms and 68 manipulation tasks, this "breadth" does not compensate for the lack of "depth"—all training platforms share the fundamental assumption of rigid kinematics, causing its generalization capability to drop to zero when encountering a new embodiment that violates this assumption. This suggests that true cross-embodiment generalization may require training data to explicitly include samples of different physical morphologies, rather than simply increasing the number of similar platforms.

| Task | Platform | Model | Success Rate | Explanation |
|------|------|------|--------|------|
| Task 1: Simple Pick-and-Place | UR5 (Rigid) | OpenVLA-OFT | 90% | Rigid baseline |
| Task 1: Simple Pick-and-Place | Embuddy (Soft) | OpenVLA-OFT | 90% | Comparable to rigid arm |
| Task 1: Simple Pick-and-Place | Embuddy (Soft) | π₀ | 80% | Slightly lower than OpenVLA-OFT |
| Task 2: Selective Pick-and-Place | UR5 (Rigid) | OpenVLA-OFT | 70% | Requires language differentiation |
| Task 2: Selective Pick-and-Place | Embuddy (Soft) | OpenVLA-OFT | 70% | Comparable to rigid arm |
| Task 3: Feeding Interaction | UR5 (Rigid) | OpenVLA-OFT | 80% | Close-range human-robot interaction |
| Task 3: Feeding Interaction | Embuddy (Soft) | OpenVLA-OFT | 80% | Comparable to rigid arm |
| Task 3: Feeding Interaction | Embuddy (Soft) | π₀ | 70% | Slightly lower than OpenVLA-OFT |

### Inference Efficiency Comparison

| Platform | Model | GPU Device | Control Frequency (Hz) |
|------|------|---------|--------------|
| UR5 | OpenVLA-OFT | A100 (Azure VM) | 32.3 |
| Embuddy | OpenVLA-OFT | H100 (Remote cluster) | 25.1 |
| Embuddy | π₀ | H100 (Remote cluster) | 38.0 |

### Key Findings

- **Fine-Tuning Fully Bridges the Embodiment Gap**: On Task 1 and Task 2, fine-tuned OpenVLA-OFT achieved the **exact same** success rates on the soft robot as on the UR5 (90% and 70%, respectively). This is a surprising result—despite the fact that the kinematics and dynamics of soft robots are completely different from those of rigid robots, the fine-tuning strategy fits perfectly. In Task 3 (the feeding task), both platforms also reached the identical success rate of 80%. This consistent conclusion across all three tasks manifests that the effect of fine-tuning in bridging the embodiment gap is not accidental, but rather a reliable, reproducible phenomenon.

- **OpenVLA-OFT Outperforms π₀ on the Soft Platform**: Interestingly, although π₀ exhibits stronger generalization capabilities across rigid embodiments (which is its design objective), after transferring to the soft platform with completely different dynamics, the properly fine-tuned OpenVLA-OFT is found to perform better (Task 1: 90% vs 80%, Task 3: 80% vs 70%). This counterintuitive result may have several explanations. First, the LoRA fine-tuning strategy of OpenVLA-OFT retains more pre-trained knowledge as a strong prior, adjusting only the necessary dynamics mapping through low-rank updates; whereas full-parameter fine-tuning of π₀ may carry a risk of overfitting given small datasets. Second, the FiLM module of OpenVLA-OFT provides a deeper level of language-vision fusion, allowing the policy to condition action generation more precisely based on instructions. Third, the continuous action head using L1 regression in OpenVLA-OFT might converge more stably on small datasets compared to the flow matching of π₀.

- **Good Semantic Adherence to Language Instructions**: In Task 2, the FiLM module effectively guides the model to focus on the target object specified in the instructions rather than making arbitrary choices. In Task 3, when an orange is placed in the plate to replace the marshmallow, the model is able to correctly refuse to execute the task instead of erroneously manipulating the visible object, indicating that the policy is driven by semantics rather than visual saliency.

- **Physical Robustness of Soft Robots Validated**: During the inference process of Task 3 (approximately 2–3 minutes per run), manually pushing Embuddy out of its original pose (about 5 seconds each time, 2 times in total) did not lead to performance degradation; the robot was able to autonomously recover its pose and continue completing the task under the VLA control loop. This demonstrates a natural synergy between the compliant nature of soft robots and the closed-loop feedback mechanism of VLA control.

- **Robustness to Human Presence**: During inference, the free movement of humans in the scene had no impact on model performance. The model successfully focuses its attention on the workspace, functioning normally as long as the workspace is not physically blocked or disrupted.

- **Deterministic Constraints of the Workspace**: When target objects are placed outside the workspace present in the training set (even by an offset of only 10cm), the model fails in all trials. This reveals a heavy reliance of VLA policies on the training workspace distribution, representing a major deployment limitation.

## Highlights & Insights

- **First End-to-End Pipeline from VLA to Soft Robots**: This is the first work to systematically deploy a VLA model on a soft continuum robotic arm, not only demonstrating its feasibility but also providing a complete open-source pipeline from data collection to deployment. This precedent unlocks new directions for integrating advanced language-guided robotic control with intrinsically safe physical platforms.

- **Discovery that "Minimal Data Suffices to Bridge Massive Domain Gaps"**: Utilizing only 50–100 teleoperated episodes (with only a few dozen frames per episode), the VLA is elevated from complete failure to a 70–90% success rate. This indicates that the pre-trained knowledge of VLA models (visual understanding, semantic alignment, and temporal action patterns) has incredible transferability—what fine-tuning primarily needs to learn is merely the relatively low-dimensional adaptation of the new platform's "kinematics mapping." This observation holds vital practical value for novel robotic platforms where data collection at scale is exceptionally difficult: if 50–100 episodes suffice, almost any new platform can acquire a usable VLA policy with just a few hours of teleoperation effort, lowering the barrier to deploying state-of-the-art AI on non-standard robotic platforms.

- **Comparison Insights of LoRA vs. Full-Parameter Fine-Tuning across Scale**: OpenVLA-OFT (7B) uses LoRA, whilst π₀ (3B) uses full-parameter fine-tuning, and the LoRA approach ultimately performs better on the soft platform. This suggests that in scenarios requiring adaptation to new domains with extremely limited data, parameter-efficient fine-tuning methods may have an advantage over full-parameter fine-tuning by retaining more pre-trained knowledge, despite the latter having more degrees of freedom for adaptation. Full-parameter fine-tuning might be prone to overfitting the details of the training distribution given only 20–50 episodes, whereas LoRA's low-rank constraint acts as an implicit regularization, preserving the generalization capabilities accumulated during pre-training. This discovery offers a direct methodological reference for other data-scarce robotics domains (e.g., underwater robots, space robots, etc.).

- **Natural Complementarity between Soft Compliance and Closed-Loop VLA**: When Embuddy is pushed off its trajectory by external forces, the VLA policy can autonomously recover via visual feedback. This robustness does not stem from explicit force control or disturbance rejection algorithms, but rather from the natural combination of the passive compliance of the soft structure and closed-loop visual control—the soft body deforms rather than breaks, and the VLA automatically corrects the deviation upon seeing it. During the inference of Task 3 (2–3 minutes per run), the experimenter manually perturbed Embuddy twice (about 5 seconds each time), and the robot autonomously recovered and successfully completed the task, with zero observable performance drop. This insight implies that the "weaknesses" of soft robots (insufficient rigidity, low positional accuracy) actually become "strengths" under closed-loop VLA control—compliant deformation provides a safety buffer while visual closed loops compensate for positional uncertainty. This synergistic mechanism can be extended to any fault-tolerant scenario requiring safe human-robot interaction.

- **Crucial Role of the FiLM Module in Language Conditioning**: In Task 2, enabling FiLM is key to successfully distinguishing between instructions for different objects. By injecting language embeddings into visual features at each Transformer layer, FiLM achieves a deeper level of feature fusion than simple token concatenation. This design philosophy can be readily transferred to any VLA task requiring strong language-visual association.

## Limitations & Future Work

- **Limited Task Complexity and Diversity**: The paper evaluates only three relatively simple manipulation tasks (pick-and-place and feeding), lacking complex multi-step reasoning tasks, dexterous manipulation, or scenarios requiring precise force control. On more complex tasks, the kinematic constraints of soft robots may present more severe bottlenecks.

- **Lack of Workspace Generalization Capability**: Objects shifting outside the training workspace by even 10cm leads to complete failure, which severely restricts the flexibility of actual deployment. Future work is needed to explore how to expand the spatial generalization range of the policy through data augmentation or domain randomization.

- **Extremely Small Dataset Scale**: The pocket-sized Task 3 has only 20 episodes. While this demonstrates data efficiency, it also means the statistical reliability of the results is limited (evaluated over 10 trials). Larger-scale data collection and evaluation across more trials would bolster the persuasiveness of the arguments.

- **Lack of Quantitative Safety Evaluation**: Although the paper qualitatively demonstrates the safe interaction benefits of soft robots (being pushable, low inertial forces), it lacks a systematic evaluation of quantitative safety metrics such as contact force measurements and collision energy.

- **Accuracy Limitations of the PCC Model**: The Piecewise Constant Curvature model is a simplified approximation, which may introduce significant errors under large deformations or complex loading conditions. How such modeling errors affect the long-term execution stability of VLA policies remains undiscussed.

- **Latency Issues of Remote Inference**: The current architecture relies on a remote GPU server for inference, and the network latency decreases the control frequency from 38 Hz to 25 Hz. For more dynamic interactive tasks, edge deployment (e.g., using Jetson devices) will be necessary.

- **Only Two VLA Models Compared**: The absence of comparison with other policy classes (such as the diffusion-based policy RDT-1B or reinforcement learning methods) on the soft platform prevents a comprehensive assessment of which policy architecture is best suited for compliant embodiments.

- **Path for Sim-to-Real Unexplored**: This work relies entirely on real-world teleoperated data. However, establishing high-fidelity soft robot simulation environments and leveraging large-scale sim data for pre-training could be an effective path to reduce data collection costs. The current lack of simulation platforms for soft continuum robots is an ecological challenge.

- **Representativeness Issue of a Single Soft Platform**: Experiments were conducted solely on the Embuddy soft robot. The morphological diversity of soft robots far exceeds that of rigid robots (owing to pneumatic, electroactive polymer, shape memory alloy actuator drives, etc.), making it still unknown whether current conclusions generalize to other types of soft platforms.

## Related Work & Insights

- **vs. OpenVLA (Original)**: The original OpenVLA uses discretized token predictions for actions, whereas OpenVLA-OFT switches to continuous action outputs, parallel decoding, and action chunking. Building upon this, this study further validates the fine-tuning effect of the OFT version on novel embodiments. The continuous output and FiLM module of OpenVLA-OFT demonstrate key advantages on the soft platform.

- **vs. π₀**: π₀ utilizes a flow-matching action generation scheme and cross-embodiment pre-training, showing strong generalization capabilities on various rigid platforms. However, this study finds that when the cross-embodiment gap expands from "different rigid arms" to "rigid-to-soft", the full-parameter fine-tuning strategy of π₀ actually underperforms compared to the LoRA scheme of OpenVLA-OFT, which provides important insights into understanding the boundaries of cross-embodiment transfer.

- **vs. RT-2**: RT-2 also adapts vision-language foundation models to robot control, but is evaluated solely on rigid RT hardware. This paper extends this paradigm to soft platforms, demonstrating that the general-purpose potential of the VLA paradigm is not limited to traditional robotic morphologies.

- **vs. Traditional Soft Robot Control**: Traditional methods rely on analytical models like PCC/affine curvature paired with sensor feedback closed loops, achieving precise control but lacking semantic understanding and task generalization. This work can be viewed as grafting "task-level intelligence (VLA)" onto "physical-level safety (soft robot)," and the two paradigms can be further merged in the future—for instance, using traditional control as the low-level actuator of VLA policies.

- **vs. Open X-Embodiment**: The OXE dataset covers 7+ rigid robotic platforms and is currently the largest cross-embodiment dataset, but it completely lacks soft robot data. The soft robot dataset open-sourced in this work serves as an important addition to the OXE ecosystem and can be integrated into unified pre-training pipelines in the future.

- **vs. LIBERO / ALOHA Benchmarks**: These benchmarks evaluate VLA policies in simulation and the real world, respectively, yet are restricted to rigid platforms. The results of this paper demonstrate that embodiment type is a heavily neglected dimension when conducting VLA generalization evaluations—policies performing excellently on LIBERO might fail completely on soft platforms. This calls for the community to construct more comprehensive evaluation benchmarks incorporating diverse physical morphologies.

- **Potential Comparison with the RDT-1B Diffusion Policy**: Although not directly compared in the paper, RDT-1B also employs diffusion models to generate continuous actions and performs outstandingly on bimanual manipulation tasks. Considering the advantage of diffusion policies in modeling multimodal distributions, it may be better suited for handling redundancy in soft robot action spaces (where the same end-effector pose corresponds to multiple possible deformation paths). This represents a valuable direction for future exploration.

## Rating

- Novelty: ⭐⭐⭐⭐ Deploys VLA on a soft continuum robotic arm for the first time. The problem formulation is novel and carries clear practical utility, though the methodology itself (fine-tuning VLAs) lacks groundbreaking innovation and leans more toward engineering integration.
- Experimental Thoroughness: ⭐⭐⭐ Limited to an evaluation scale of three tasks, two models, and 10 trials. The task complexity is somewhat low, the statistical confidence is limited, and it lacks quantitative evaluation metrics for safety.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-formulated motivations. The appendix comprehensively covers experimental details and hyperparameter settings, though the depth of quantitative analysis and ablation studies in the main text could be further enhanced.
- Value: ⭐⭐⭐⭐ As the first work to systematically study VLA + soft robotics, it carries significant pioneering value and practical reference significance, and its open-sourced dataset lays a solid foundation for follow-up research. The proposed fine-tuning pipeline possesses high reusability for direct extension to VLA deployments on other non-standard robotic platforms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] X-VLA: Soft-Prompted Transformer as Scalable Cross-Embodiment Vision-Language-Action Model](../../ICLR2026/robotics/x-vla_soft-prompted_transformer_as_scalable_cross-embodiment_vision-language-act.md)
- [\[NeurIPS 2025\] SAFE: Multitask Failure Detection for Vision-Language-Action Models](safe_multitask_failure_detection_for_vision-language-action_models.md)
- [\[ECCV 2024\] QUAR-VLA: Vision-Language-Action Model for Quadruped Robots](../../ECCV2024/robotics/quar-vla_vision-language-action_model_for_quadruped_robots.md)
- [\[NeurIPS 2025\] ThinkAct: Vision-Language-Action Reasoning via Reinforced Visual Latent Planning](thinkact_vision-language-action_reasoning_via_reinforced_visual_latent_planning.md)
- [\[NeurIPS 2025\] VLA-Cache: Efficient Vision-Language-Action Manipulation via Adaptive Token Caching](vla-cache_efficient_vision-language-action_manipulation_via_adaptive_token_cachi.md)

</div>

<!-- RELATED:END -->
