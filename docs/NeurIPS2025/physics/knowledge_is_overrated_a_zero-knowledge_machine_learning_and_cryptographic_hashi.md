---
title: >-
  [Paper Note] Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC
description: >-
  [NeurIPS 2025][Physics & Scientific Computing][zero-knowledge proof] This paper proposes PHAZE, a framework that combines cryptographic hashing (Rabin fingerprinting) and zero-knowledge machine learning (zkML) to enable…
tags:
  - "NeurIPS 2025"
  - "Physics & Scientific Computing"
  - "zero-knowledge proof"
  - "LHC trigger"
  - "early-exit"
  - "cryptographic hashing"
  - "verifiable inference"
date: 2026-05-08
content_hash: 3244914d6c5f13de
---

# Knowledge is Overrated: A Zero-Knowledge ML and Cryptographic Hashing-Based Framework for Verifiable, Low Latency Inference at the LHC

**Conference**: NeurIPS 2025
**arXiv**: [2511.12592](https://arxiv.org/abs/2511.12592)  
**Code**: [PHAZE](https://zenodo.org/records/17370252) (open source)  
**Area**: Physics
**Keywords**: zero-knowledge proof, LHC trigger, early-exit, cryptographic hashing, verifiable inference

## TL;DR
This paper proposes PHAZE, a framework that combines cryptographic hashing (Rabin fingerprinting) and zero-knowledge machine learning (zkML) to enable verifiable early-exit inference at LHC trigger latency, achieving a theoretical online latency of ~152–253 ns while providing built-in anomaly detection capability.

## Background & Motivation

**Background**: The Large Hadron Collider (LHC) produces collision events at 40 MHz, making low-latency event selection (trigger) algorithms central to its operation. Existing approaches such as AXOL1TL and CICADA implement O(ns)-latency anomaly detection on FPGAs via frameworks like hls4ml.

**Limitations of Prior Work**: Modern high-accuracy ML models (e.g., large DNNs and foundation models) cannot satisfy the 40 MHz online latency constraint. Current acceleration schemes rely on case-by-case accuracy–speed trade-offs and lack a general framework. The verifiability and reproducibility of trigger decisions, which are critical for downstream physics analyses, are not systematically guaranteed.

**Key Challenge**: High accuracy demands large models, while online inference allows only nanosecond-level latency — these two requirements appear fundamentally incompatible.

**Goal**: Design a framework that enables trigger decisions produced by an arbitrarily large baseline model to be executed at nanosecond-level latency online, while guaranteeing cryptographic verifiability of each decision.

**Key Insight**: Decouple inference into an offline build phase (unconstrained in latency) and an online lookup phase (extremely low latency), using cryptographic hashing to map early-layer activations to a precomputed decision table.

**Core Idea**: Rabin fingerprinting compresses early-layer model activations into fixed-length hashes, which are used to query a prebuilt Verifiable Decision Map (VDM) for O(ns) inference; zkML proofs guarantee the integrity of each decision.

## Method

### Overall Architecture
The framework adopts a two-phase design. In the **Build Phase** (offline, compute-intensive), full-model inference results are precomputed into a hash-to-decision mapping table and zkML correctness proofs are generated. In the **Online Phase** (online, ultra-low latency), only early-layer activation extraction, hashing, and table lookup are performed, bypassing full model inference.

### Key Designs

1. **Verifiable Decision Map (VDM) Construction**:

    - **Function**: Stores the inference results of the full model $\mathbb{M}_{\text{full}}$ as a hash lookup table.
    - **Mechanism**: For each training event $I_j$: (1) run the full model to obtain decision $D_j$; (2) extract early-layer activations $A_j \in \mathbb{R}^k$; (3) quantize to a finite field $A_j^* \in \mathbb{F}_p^k$; (4) construct polynomial interpolation $P_{A_j^*}(x)$; (5) evaluate the Rabin fingerprint $h_j = P_{A_j^*}(r) \mod g(x)$ at a random challenge point $r$; (6) store the mapping VDM: $h_j \to D_j$.
    - **Design Motivation**: Compresses high-dimensional activation vectors into 64-bit hashes; the Schwartz–Zippel lemma guarantees negligible collision probability ($\leq 100/2^{64}$).

2. **zkML Proof Generation**:

    - **Function**: Generates cryptographic correctness proofs for each VDM entry.
    - **Mechanism**: Based on zk-STARKs, each proof attests to the compound statement: "$\mathbb{M}_{\text{full}}$ produces decision $D_j$ on input $I_j$, and the early-layer activations hash to $h_j$." Proof generation complexity is $O(T(n) \cdot \text{polylog}(T(n)))$ and verification requires only $O(\text{polylog}(T(n)))$.
    - **Design Motivation**: Ensures tamper-resistance of the VDM (e.g., against bit-flip attacks) and supports auditing and reproducibility for downstream analyses.

3. **Online Early-Exit Inference**:

    - **Function**: Nanosecond-level online trigger decisions.
    - **Mechanism**: New event $I_{\text{new}}$ → FPGA extracts early-layer activations $A_{\text{new}}^*$ (~100–200 ns) → Barycentric Lagrange interpolation with Estrin's method computes the hash (~50 ns) → VDM lookup (~2.5 ns).
    - **Design Motivation**: All heavy computation is shifted offline; the online path requires only activation extraction, hashing, and table lookup — all three steps can be efficiently implemented on FPGA.

4. **Map-Miss Anomaly Detection**:

    - **Function**: Flags potential anomalies when a VDM lookup yields no match.
    - **Mechanism**: If the hash of a new event has no entry in the VDM, it is potentially a new-physics signal or unknown detector effect, and is cached and forwarded to a dedicated anomaly detection algorithm.
    - **Design Motivation**: Provides low-level anomaly detection capability at no additional computational cost.

### Loss & Training
The baseline model training is decoupled from the framework and is compatible with any ML model. The authors recommend applying representation learning techniques such as contrastive loss to improve class separation in the latent space of early-layer activations, thereby reducing quantization error.

## Key Experimental Results

### Main Results — Latency Estimates

| Phase | Task | Latency/Event | Complexity |
|-------|------|--------------|------------|
| Build | Full model inference | O(ms–s) | $O(\|\mathbb{M}_{\text{full}}\|)$ |
| Build | zkML proof generation | O(min) | $O(\|\mathbb{M}\| \cdot \text{polylog})$ |
| Online | Early-layer activation (FPGA) | ~100–200 ns | $O(\|\mathbb{M}_{\text{early}}\|)$ |
| Online | OTF hashing (FPGA) | ~50 ns | $O(\sqrt{d})$ |
| Online | VDM lookup (FPGA) | ~2.5 ns | $O(1)$ |
| **Online Total** | | **~152.5–252.5 ns** | |

### Build Phase Benchmarks (~7M-parameter DNN, MNIST)

| Metric | Rabin Fingerprint | Shamir Secret Sharing |
|--------|------------------|-----------------------|
| Hash throughput | Higher | Lower (requires Share+Reconstruct) |
| Memory consumption | Lower | Higher |
| ezkl proof generation | ~10² s/event | N/A |
| ezkl proof verification | Sub-second | N/A |

### Key Findings
- Theoretical online latency reaches 152.5–252.5 ns, satisfying the O(ns) constraint of the LHC Level-1 trigger.
- The 64-bit hash collision probability is $\leq 100/2^{64} \approx 5.4 \times 10^{-18}$, computationally negligible.
- FPGA memory is the primary bottleneck: AMD UltraScale+ FPGAs can store ~6.3M VDM entries, which is insufficient to cover a representative dataset.
- zkML proof generation is the dominant bottleneck in the build phase (~100 s/event without optimization), requiring more efficient zkML toolchains.

## Highlights & Insights
- **Computation Shift Paradigm**: Inference computation is moved from online to offline, with the online path reduced to a table lookup. This is a general acceleration principle transferable to any low-latency inference scenario (e.g., autonomous driving, real-time trading).
- **Anomaly Detection for Free**: VDM misses naturally constitute anomaly signals, which is valuable for new-physics discovery and detector monitoring. This "by-product anomaly detection" design is particularly elegant.
- **Built-in Verifiability**: zkML proofs make every trigger decision fully auditable and traceable, which is essential for the reproducibility of scientific experiments.

## Limitations & Future Work
- **VDM Storage Bottleneck**: A single FPGA accommodates only 6.3M entries; real-world deployment requires distributed lookup schemes, which the authors acknowledge as an open problem.
- **Strong Assumptions**: The predictive sufficiency of early-layer activations (Assumption 3) is not rigorously validated; the injectivity assumption in polynomial interpolation requires careful examination.
- **High Build Phase Cost**: Running the full model and generating zkML proofs for every training event incurs substantial build-phase cost at scale.
- **MNIST-Only Benchmarks**: A 7M-parameter DNN evaluated on MNIST is far from representative of the complexity of real LHC models.
- **Dynamic VDM Not Implemented**: The paper proposes but does not experimentally validate a dynamic update mechanism for the VDM.

## Related Work & Insights
- **vs. AXOL1TL/CICADA**: These approaches involve end-to-end optimization of small models, whereas PHAZE allows arbitrarily large models to achieve low latency via table lookup — a fundamentally different philosophy.
- **vs. hls4ml**: hls4ml maps models directly to FPGA logic and is constrained by model size; PHAZE decouples the model-size constraint to the offline build phase.
- **vs. Standard Early-Exit**: Conventional early-exit methods require multiple exit branches and complex decision logic; PHAZE reduces the decision step to a hash table lookup.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to introduce zkML and cryptographic hashing into particle physics trigger design; conceptually highly original.
- Experimental Thoroughness: ⭐⭐ Only build-phase feasibility benchmarks on MNIST; real physics data and measured online latency are absent.
- Writing Quality: ⭐⭐⭐⭐ Technical exposition is clear, with balanced coverage of cryptographic and physics background.
- Value: ⭐⭐⭐⭐ The framework is forward-looking and points toward a plausible direction for next-generation LHC triggers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] The Primacy of Magnitude in Low-Rank Adaptation](the_primacy_of_magnitude_in_low-rank_adaptation.md)
- [\[CVPR 2026\] QKD: Quantum-Gated Task-interaction Knowledge Distillation for Class-Incremental Learning](../../CVPR2026/physics/qkd_quantum_gated_incremental_learning.md)
- [\[AAAI 2026\] Scientific Knowledge-Guided Machine Learning for Vessel Power Prediction: A Comparative Study](../../AAAI2026/physics/scientific_knowledge-guided_machine_learning_for_vessel_power_prediction_a_compa.md)
- [\[NeurIPS 2025\] Exoplanet Formation Inference Using Conditional Invertible Neural Networks](exoplanet_formation_inference_using_conditional_invertible_neural_networks.md)
- [\[NeurIPS 2025\] Simulation-Based Inference for Neutrino Interaction Model Parameter Tuning](simulation-based_inference_for_neutrino_interaction_model_parameter_tuning.md)

</div>

<!-- RELATED:END -->
