---
title: >-
  [Paper Note] NeuralOS: Towards Simulating Operating Systems via Neural Generative Models
description: >-
  [ICLR 2026][Image Generation][operating system simulation] This paper proposes NeuralOS, a dual-component architecture combining an RNN-based state tracker and a diffusion renderer…
tags:
  - "ICLR 2026"
  - "Image Generation"
  - "operating system simulation"
  - "world model"
  - "diffusion rendering"
  - "GUI generation"
  - "interactive systems"
date: 2026-05-08
content_hash: 6b6a7d709bbe304f
---

# NeuralOS: Towards Simulating Operating Systems via Neural Generative Models

**Conference**: ICLR 2026  
**arXiv**: [2507.08800](https://arxiv.org/abs/2507.08800)  
**Code**: [neural-os.com](https://neural-os.com)  
**Area**: Image Generation / Interactive World Models  
**Keywords**: operating system simulation, world model, diffusion rendering, GUI generation, interactive systems

## TL;DR

This paper proposes NeuralOS, a dual-component architecture combining an RNN-based state tracker and a diffusion renderer, which directly predicts graphical interface frame sequences from user input events (mouse movement/click/keyboard), achieving for the first time the simulation of an operating system via neural generative models.

## Background & Motivation

**Background**: Generative models have evolved from text and image generation to video generation and interactive virtual environment simulation (e.g., game world models such as GameGen and Oasis). These advances suggest that computational interfaces may shift from hand-crafted programming toward fully generative paradigms.

**Limitations of Prior Work**: Existing interactive world models primarily target video games and rely on short context windows, since game states can typically be inferred from recent frames. OS interfaces, however, are fundamentally different: (1) state transitions involve long delays (e.g., launching Firefox may take ~30 frames); (2) the user action space is enormous (mouse positions form a large pixel-level discrete space); (3) long-term state memory is required (hidden windows, prior interactions, etc.).

**Key Challenge**: OS interfaces must respond immediately to unpredictable user inputs that frequently cause abrupt visual changes (e.g., launching a new application), in sharp contrast to the smooth and predictable transitions found in video generation. A model must simultaneously maintain precise state tracking and high-quality visual rendering.

**Goal**: Can neural generative models simulate the graphical interface of an operating system end-to-end? This entails precise cursor modeling, long-term state tracking, and complex interactions such as application launching and closing.

**Key Insight**: Drawing inspiration from the functional separation between the OS kernel (state management) and desktop rendering (GUI output), the paper designs a dual-module architecture of RNN (state tracking) + diffusion renderer (frame generation), coupled with a multi-stage training strategy.

**Core Idea**: A hierarchical RNN tracks system state, a diffusion model renders interface frames, and multi-stage training enables the neural network to learn to simulate an operating system.

## Method

### Overall Architecture

NeuralOS formulates OS interface simulation as an autoregressive generation problem: $P(x_{1:T}|a_{1:T};\theta) = \prod_t P(x_t|x_{<t}, a_{\leq t};\theta)$. The architecture comprises two components: (1) a hierarchical RNN maintaining internal state, and (2) a UNet diffusion renderer that generates the next frame conditioned on state and user input. Training follows a four-stage pipeline.

### Key Designs

**Design 1: Hierarchical RNN State Tracking**
- **Function**: Maintains the internal state of the OS (open applications, hidden windows, historical interactions, etc.)
- **Mechanism**: A two-layer LSTM architecture. The lower LSTM encodes user inputs (mouse coordinates, clicks, keyboard) and integrates visual information from the previous frame via multi-head attention; the upper LSTM processes the attention-augmented representations and feeds its output back to the lower layer. Each layer has a 4096-dimensional hidden state.
- **Design Motivation**: (1) RNN incurs constant per-step computational cost, making it suitable for real-time simulation over long sequences; (2) unlike Transformers constrained to short windows, RNNs can retain arbitrarily distant history (e.g., hidden windows); (3) the two-layer design separates input encoding from state management.

**Design 2: Gaussian Encoding of Cursor Position**
- **Function**: Encodes precise cursor position as a spatial Gaussian map.
- **Mechanism**: A 2D Gaussian map centered at the cursor coordinates is constructed in latent space: $M_t(i,j) = \exp(-\frac{(i-a_t^x/s)^2 + (j-a_t^y/s)^2}{2})$, which is concatenated with the RNN output before being fed into the renderer.
- **Design Motivation**: Direct one-hot encoding loses precision due to limited latent-space resolution. Without Gaussian encoding, cursor position errors reach 130/95.8 pixels; with it, errors drop to just 1.6/1.4 pixels (<0.5% of frame size).

**Design 3: Four-Stage Training Pipeline**
- **Stage 1 – RNN Pre-training**: Pre-trains the RNN with MSE loss to predict latent frames, addressing the vanishing gradient problem where the renderer ignores RNN outputs.
- **Stage 2 – Joint Training**: Jointly optimizes the pre-trained RNN and the diffusion renderer.
- **Stage 3 – Scheduled Sampling**: With probability $p$, replaces ground-truth frames with model-generated frames as input, mitigating exposure bias and error accumulation at inference time.
- **Stage 4 – Context Length Extension**: Extends training sequence length to capture long-term dependencies.
- **Design Motivation**: Direct end-to-end training causes the renderer to ignore RNN outputs due to weak gradient flow. Staged training ensures each component is effectively utilized.

**Design 4: Curriculum Training Strategy**
- **Function**: Trains first on "challenging frame transitions" (frame pairs with pixel differences exceeding a threshold), then extends to the full dataset.
- **Design Motivation**: The majority of OS frame transitions involve only minor cursor movements, providing limited learning signal. Curriculum training prioritizes learning meaningful state changes.

### Loss & Training

- Stage 1: MSE loss (first $C$ channels of RNN output vs. target latent frame)
- Stages 2–4: Standard diffusion loss (DDPM)
- Inference: DDIM with 2-step sampling, 18 fps on H100
- Model parameters: RNN 2.2B + UNet 263M
- Training compute: ~23,000 GPU hours (H200 + H100)

## Key Experimental Results

### Main Results

**Cursor Position Accuracy**

| Method | Δx (pixels) | Δy (pixels) |
|--------|-------------|-------------|
| NeuralOS (with Gaussian map) | 1.6 | 1.4 |
| NeuralOS (without Gaussian map) | 130.0 | 95.8 |
| Random baseline | 175.4 | 126.9 |

**State Transition Accuracy**: 37.7% (over 73 cluster classes, far exceeding the majority-vote baseline of 1.4%)

**Human Discrimination Study**:

| Clip Length | Human Success Rate in Identifying Real OS |
|-------------|------------------------------------------|
| 10s | 58.3% |
| 20s | 55.0% |

Human performance is only marginally above chance for short clips.

### Ablation Study

| Component | Effect |
|-----------|--------|
| Without Gaussian cursor encoding | Δx increases from 1.6 → 130.0 px |
| Without scheduled sampling (Stage 3) | RMSE grows continuously; severe degradation on long sequences |
| Random data only | Spurious correlations emerge (e.g., moving cursor to close button triggers window closing) |
| Agent data only | Insufficient interaction diversity |

### Key Findings

1. **Precise cursor modeling is critical**: Gaussian spatial encoding reduces cursor error from 130 px to 1.6 px.
2. **RNN pre-training is necessary**: Without it, the diffusion renderer completely ignores RNN outputs.
3. **Scheduled sampling effectively mitigates error accumulation**: Generation quality on long sequences improves significantly.
4. **Synthetic data can teach new applications**: Doom was never installed, yet the model can learn to simulate it through synthetic demonstrations.
5. **Data diversity requires balance**: Random and agent data sources are complementary; either alone is insufficient.

## Highlights & Insights

- **Ambitious vision**: This is the first work to propose and partially realize the simulation of an operating system via neural generative models, representing a paradigm shift from content generation to system simulation.
- **The Doom experiment is remarkably imaginative**: By using synthetic training data, the model simulates an application that was never installed, suggesting that generative interfaces could eventually operate entirely independently of real software.
- **Engineering depth**: From data collection (64 parallel Docker containers) to training strategy (4-stage curriculum) to inference optimization (DDIM 2-step / 18 fps), the work demonstrates substantial system-level engineering capability.
- **Insight on cursor modeling**: Gaussian spatial encoding is both elegant and effective, offering a general solution for precise positional control in interactive generative models.
- **Implications for agent training**: NeuralOS can serve as a safe simulated environment for training and evaluating computer-use agents without requiring real system commands.

## Limitations & Future Work

1. **Limited resolution**: Only 512×384, far below real OS display resolutions.
2. **Narrow application scope**: Only four applications are included — Home, Trash, Terminal, and Firefox.
3. **Difficulty modeling keyboard input**: Computational constraints hinder precise modeling of fine-grained keyboard inputs.
4. **Extremely high training cost**: 23,000 GPU hours, with data processing and training spanning approximately four months.
5. **Low state transition accuracy**: 37.7% far exceeds baselines but remains far from practical utility.
6. **Cannot generalize to complex real-world applications**: Scenarios such as multi-window applications and system settings are not addressed.

## Related Work & Insights

- **Distinction from game world models (GameGen, Oasis)**: OS interfaces require longer-term state memory and a substantially larger action space.
- **Connection to interactive world models (World Labs, Genie)**: Both lines of work explore replacing hand-crafted environments with generative models.
- **Implications for computer-use agents** (e.g., Claude computer use): NeuralOS can provide a safe training environment.
- **Implications for future UI design**: Generative interfaces could be personalized and adapted in real time according to user needs.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First to propose and implement OS simulation via neural generative models; a pioneering contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-faceted evaluation (cursor accuracy / state transitions / human study / ablations), though the environment is heavily simplified.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Compelling narrative, clear problem formulation, and thorough description of the training strategy.
- **Value**: ⭐⭐⭐⭐ — An exciting vision with initial proof of feasibility, though substantial gaps remain before practical deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Amortising Inference and Meta-Learning Priors in Neural Networks (BNNP)](amortising_inference_and_meta-learning_priors_in_neural_networks.md)
- [\[ICLR 2026\] COSMO-INR: Complex Sinusoidal Modulation for Implicit Neural Representations](cosmo-inr_complex_sinusoidal_modulation_for_implicit_neural_representations.md)
- [\[ICLR 2026\] Beyond Confidence: The Rhythms of Reasoning in Generative Models](beyond_confidence_the_rhythms_of_reasoning_in_generative_models.md)
- [\[NeurIPS 2025\] AccuQuant: Simulating Multiple Denoising Steps for Quantizing Diffusion Models](../../NeurIPS2025/image_generation/accuquant_simulating_multiple_denoising_steps_for_quantizing.md)
- [\[ICLR 2026\] DoFlow: Flow-based Generative Models for Interventional and Counterfactual Forecasting](doflow_flow-based_generative_models_for_interventional_and_counterfactual_foreca.md)

</div>

<!-- RELATED:END -->
