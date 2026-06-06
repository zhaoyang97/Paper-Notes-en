---
title: >-
  [Paper Note] NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating
description: >-
  [ICML 2026][Audio & Speech][Auditory salience] A real-time salience detector inspired by cortical oscillations is implemented as a 2D oscillatory wave field (OWM)…
tags:
  - "ICML 2026"
  - "Audio & Speech"
  - "Auditory salience"
  - "oscillatory working memory"
  - "training-free gating"
  - "ALM long audio understanding"
date: 2026-05-08
content_hash: edd90df6c628378c
---

# NAACA: Training-Free NeuroAuditory Attentive Cognitive Architecture with Oscillatory Working Memory for Salience-Driven Attention Gating

**Conference**: ICML 2026  
**arXiv**: [2605.13651](https://arxiv.org/abs/2605.13651)  
**Code**: https://github.com/zjyuan1208/NAACA-Oscillatory-Working-Memory (available)  
**Area**: Audio Language Models / Neuro-Inspired Architectures / Attention Allocation  
**Keywords**: Auditory salience, oscillatory working memory, training-free gating, ALM long audio understanding

## TL;DR
A real-time salience detector inspired by cortical oscillations is implemented as a 2D oscillatory wave field (OWM), serving as a "training-free attention gate" for Audio Language Models (ALMs) on long audio. Only truly salient windows are fed into the ALM, boosting AP on XD-Violence from 53.5% to 70.6% while reducing ALM invocations by about 40%.

## Background & Motivation

**Background**: Audio Language Models (e.g., AudioQwen) can already perform open-vocabulary semantic understanding on short audio, serving as a key module for integrating speech and environmental sounds into multimodal reasoning. In long-duration audio scenarios such as street surveillance or bioacoustics, the industry typically either slides windows into the ALM or feeds entire segments into a transformer, letting it select salient parts.

**Limitations of Prior Work**: Long-stream inference suffers from "attention dilution"—background sounds consume most of the token budget, causing rare but critical events (gunshots, cries for help, sudden cheers) to be missed. In the provided demo, slicing 60 s into four 15 s windows and feeding them sequentially causes the final bagpipe onset to be missed; only by moving the last 15 s to the front does the model "see" it. Exhaustive short-window inference covers all salient points but dramatically increases ALM invocation costs, making it impractical for industrial deployment.

**Key Challenge**: There is a trade-off between perceptual recall and computational budget. One can either burn GPU resources for continuous ALM calls or reduce calls at the risk of missing rare events. Traditional statistical drift detectors (e.g., Rabanser series) or representation-based methods require long-term historical samples and significant overhead, making online, unsupervised, deployable solutions difficult.

**Goal**: To design a lightweight gating module that requires no training, no historical labels, and can determine online "when to wake up the ALM."

**Key Insight**: Drawing inspiration from cognitive neuroscience—the brain uses attention gating to filter stable backgrounds and amplify salient stimuli; cortical working memory is maintained by attractor states, with oscillatory dynamics decoupling encoding and maintenance (β for maintenance, γ for encoding). This suggests salience can be detected via "state transitions" without training a dedicated classifier.

**Core Idea**: Use the 527-class probabilities output by a PANN encoder as amplitudes for different-frequency sine drivers, injecting them into a $64 \times 64$ 2D damped oscillatory wave field (OWM). Global energy surges relative to an adaptive threshold signal "salient events," thus reframing ALM attention gating as a biophysically interpretable oscillatory energy detection problem.

## Method

### Overall Architecture
NAACA's pipeline is a staged streaming process: raw audio is sliced into 4 s sliding windows $\mathbf{x}_t$ → a pretrained PANN encoder computes a 527-dimensional class probability vector $\mathbf{p}_t = \mathrm{Enc}(\mathbf{x}_t)$ → each probability dimension is treated as the amplitude of a sine carrier, mapped to a local oscillatory drive in 2D space $S_i(x,t) = a_i(t) \sin(\omega_i t) \mathbf{1}_{\Omega_i}(x)$ → injected into the OWM 2D wave field for free evolution → monitor total system energy $E(t)$ → if $E(t)$ exceeds adaptive threshold $T_{\text{adapt}}$, the corresponding window is sent to the ALM for semantic decoding. Except for the pretrained PANN and ALM, OWM has no learnable parameters.

### Key Designs

1. **Frequency-Orthogonal Oscillatory Drive Mapping**:

    - **Function**: Encodes the 527-dimensional PANN probability vector into 527 frequency-orthogonal sine drive signals, injected into $64 \times 64 = 4096$ grid points (each class assigned 7-8 cells, row-major deterministic allocation, non-learned).
    - **Mechanism**: The $i$-th class carrier frequency $f_i = f_{\min} + i (f_{\max} - f_{\min}) / (C-1)$ is linearly distributed over $[51, 1200]$ Hz; instantaneous amplitude is the class probability $a_i(t)$; each frequency is nonzero only in its spatial patch $\Omega_i$. Thus, input distribution shifts across classes simultaneously alter oscillatory phase relations at multiple spatial locations, triggering global energy transients.
    - **Design Motivation**: Using frequency as identity encoding and space as memory slots enables both frequency-domain (distinct $\omega_i$ have different damping responses) and spatial separability (distinct $\Omega_i$). Compared to a learned classifier head, this deterministic mapping only requires recomputing frequencies/grid points when switching encoders, making transfer extremely low-cost.

2. **2D Damped Wave Field as Working Memory**:

    - **Function**: OWM is a 2D velocity-pressure field; pressure $p(x,y,t)$ stores current auditory state, velocity $\mathbf{v}$ controls lateral propagation between adjacent grid points; attractor-like dynamics maintain "stable background" and amplify "transitions" into energy signals.
    - **Mechanism**: First-order system $\partial_t p + k^p p = -c^2(x,y) \nabla \cdot \mathbf{v} + S$ and $\partial_t \mathbf{v} + k^v \mathbf{v} = -\nabla p$, discretized with time step $\Delta t = 0.01$. The wave speed field is designed as striped $c(x,y) = c(y)$ (alternating blue shades), producing Bragg-matched periodic slow-propagating coherent modes, coupling "maintenance" low-frequency and "encoding" high-frequency phases. Theorem 2.4 in the paper proves this striped structure is optimal for salience sensitivity.
    - **Design Motivation**: In steady state, the field naturally forms "sound class → spatial resonance location" attractors, akin to cortical topological organization; when input distribution is stable, energy amplitude stabilizes, but class switches trigger global energy reorganization. This shifts "what changed" detection to a biophysical quantity, eliminating the need for a trained detector.

3. **Adaptive Energy Threshold + Persistence Filtering**:

    - **Function**: Online comparison of instantaneous $E(t)$ drift with threshold $T_{\text{adapt}} = \mu + 2\sigma(1 + \alpha \cdot \text{trend})$ triggers gating; persistence filtering suppresses single-frame false alarms.
    - **Mechanism**: $\mu, \sigma$ are estimated over a sliding window of length $W=20$ for the mean and standard deviation of energy-derived drift; the trend factor weights the "trend" of drift—if energy has been rising, the threshold is raised accordingly. Final decision = threshold crossing + multi-frame persistence.
    - **Design Motivation**: Static thresholds are unreliable due to varying background noise levels across cities and times; adaptive statistical thresholds allow stable operation on both XD-Violence and USoW datasets (median gating rates 0.597 vs 0.650).

### Loss & Training
NAACA is entirely training-free—PANN and AudioQwen are frozen pretrained models, OWM has no trainable parameters, and all "hyperparameters" (frequency range 51-1200 Hz, damping $k^p=k^v=10$, grid $64\times64$, sliding window $W=20$, threshold multiplier 2) are set directly based on sensitivity analysis from Theorems 2.1/2.4. There is no gradient descent, no labels, only one-time geometric/physical parameter configuration.

## Key Experimental Results

### Main Results
On the XD-Violence audio-only track (500 test samples), compared with supervised audio models, supervised video models, and zero-shot video models:

| Method | Modality | Training | AP (%) |
|--------|----------|----------|--------|
| AudioQwen (exhaustive) | Audio | No | 53.50 |
| Random 4 s segment | Audio | No | 60.44 |
| HL-Net (supervised) | Audio | Yes | 60.50 |
| AVadCLIP (supervised) | Audio | Yes | 52.51 |
| Holmes-VAU (supervised) | Video | Yes | 87.68 |
| TRACE (with cross-attn adaptation) | Video | Partial | 83.67 |
| **NAACA (Ours)** | Audio | No | **70.60** |

NAACA, without any training, outperforms all supervised audio baselines and exceeds Random 4 s by 10.16 percentage points (demonstrating OWM segment selection is effective, not just due to shorter input). There remains a gap with supervised video methods, but this reflects the inherent audio-only upper bound.

### Ablation Study

| Configuration | XD-Violence AP | Time Sent Ratio | Notes |
|---------------|---------------|----------------|-------|
| AudioQwen exhaustive | 53.50 | 1.00 | Full sliding window baseline |
| Random 4 s (same count) | 60.44 | $\approx$ 0.6 | Isolates "short input" contribution |
| NAACA full | 70.60 | 0.597 | OWM segment selection |
| NAACA on USoW | (qualitative) | 0.650 | Cross-dataset consistency |

"Short input" alone contributes $+6.94$ AP, OWM salience selection adds another $+10.16$ AP; OWM-detected drift points overlap with ground-truth event frames at 61.1%, indicating effective selection of key moments.

### Key Findings
- OWM segment selection reduces ALM calls by about 40% (57 → 34 per 60 s segment) while increasing AP by 17.1 points, directly advancing the Pareto frontier.
- FFT spectral analysis of the $p$-field shows that steady-state background periods are dominated by β-band (15-30 Hz) oscillations (maintenance), while post-drift examples shift to γ-band (30-50 Hz) (encoding), matching the division of labor in cortical working memory and providing model interpretability evidence.
- In qualitative USoW cases, OWM distinguishes three drift types: entirely new events (car engine, bagpipe), subclass switches (hi-hat in/out), and robustness to brief pauses (baby cries not split into multiple events)—indicating it captures "distributional change" rather than "loudness change."

## Highlights & Insights
- Replacing a "trained detector" with a physics-grade wave field is an elegant, counterintuitive move: the authors use the Bragg stripe optimality theorem to reduce wave speed field parameterization to a single stripe period, making OWM nearly hyperparameter-free; transferring to a new encoder only requires recomputing frequency allocation.
- The cognitive science proposition "salience ≠ loudness, salience = context change" is translated into "system energy transient relative to adaptive threshold," offering a unified abstraction for LLM attention gating across modalities: as long as input streams can be encoded into OWM-like "quasi-attractor dynamical systems," energy surges can serve as salience signals.
- Performance gains come from "processing less" rather than "processing smarter," which is especially deployment-friendly for streaming—demonstrating that long-context does not always require expanding the context window, but can be addressed by "gating first, then feeding."

## Limitations & Future Work
- The performance ceiling is set by PANN + AudioQwen; PANN is trained on AudioSet labels and fails in specialized domains (medical bird calls, mechanical faults), requiring stronger pretrained encoders.
- Hard gating loses boundary context, which may hinder long-range causal reasoning; the authors suggest future work with KV-cache modulation for "soft gating," but this requires white-box ALM access.
- Current evaluation focuses on anomaly detection AP and temporal precision, lacking SpeechIQ-style downstream QA or instruction-following tasks, so it is unclear if OWM-selected windows suffice for true multi-turn reasoning.
- Experiments only cover XD-Violence (movie audio) and USoW (urban sounds), both short-to-medium length (60 s); stability on hour-long streams remains untested.

## Related Work & Insights
- **vs Rabanser et al. statistical drift detectors**: These require long-term historical samples to estimate reference distributions; NAACA uses only 20-frame sliding statistics for thresholding, making it more suitable for open deployment, though with weaker theoretical guarantees (no formal false-alarm rate).
- **vs AVadCLIP / HL-Net supervised methods**: These rely on domain annotations for fine-tuning and require re-annotation for new scenarios; NAACA is training-free with zero transfer cost, but its ceiling is determined by the pretrained ALM.
- **vs MA-LMM and other KV-cache long video methods**: These break the transformer context bottleneck via latent compression, while NAACA applies physical gating at the input layer; the two approaches are complementary.
- Insight: Extending the idea of "salience = physical system transient" to video/text streams is an open question—e.g., can LLM hidden state energy serve as a token-level salience signal for RAG retrievers/agents to trigger events?

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Directly using cortical wave simulation as a detector is highly original; few have explored this mechanism.
- Experimental Thoroughness: ⭐⭐⭐⭐ XD-Violence + USoW dual datasets + quantitative/qualitative + spectral analysis; solid, though lacking SpeechIQ-style downstream tasks.
- Writing Quality: ⭐⭐⭐⭐⭐ Four theorems formalize intuitions; the narrative (cognitive motivation → physical modeling → salience detection) is very clear.
- Value: ⭐⭐⭐⭐ Provides a lightweight gating module for "long audio LLM deployment" that is immediately usable and highly practical for industrial pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](../../ACL2026/audio_speech/temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ICML 2026\] Polyphonia: Zero-Shot Timbre Transfer in Polyphonic Music with Acoustic-Informed Attention Calibration](polyphonia_zero-shot_timbre_transfer_in_polyphonic_music_with_acoustic-informed_.md)
- [\[ICLR 2026\] Dynamic Parameter Memory: Temporary LoRA-Enhanced LLM for Long-Sequence Emotion Recognition in Conversation](../../ICLR2026/audio_speech/dynamic_parameter_memory_temporary_lora-enhanced_llm_for_long-sequence_emotion_r.md)
- [\[ACL 2026\] Towards Fine-Grained and Multi-Granular Contrastive Language-Speech Pre-training](../../ACL2026/audio_speech/towards_fine-grained_and_multi-granular_contrastive_language-speech_pre-training.md)
- [\[ACL 2026\] Detecting Hallucinations in SpeechLLMs at Inference Time Using Attention Maps](../../ACL2026/audio_speech/detecting_hallucinations_in_speechllms_at_inference_time_using_attention_maps.md)

</div>

<!-- RELATED:END -->
