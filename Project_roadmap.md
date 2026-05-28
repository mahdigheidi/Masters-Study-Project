Plasticity Project Roadmap

Project Goal

This project aims to reproduce, analyze, and extend the findings from:

* “Understanding Plasticity in Neural Networks” (Lyle et al., 2023)

The main objective is to investigate:

* Plasticity loss in neural networks
* Optimization instability in deep reinforcement learning
* Loss landscape evolution
* Gradient interference
* Methods to preserve neural network plasticity

The project is structured into several implementation and research phases.

⸻

Phase 1 — Core Infrastructure

Goal

Build the foundational RL and experiment framework.

Tasks

* Implement replay buffer
* Implement DQN
* Implement target networks
* Implement epsilon-greedy exploration
* Implement training loop abstraction
* Implement experiment configuration system
* Implement logging system

Architectures

* MLP
* CNN

Environments

* True-label classification MDP
* Random-label classification MDP
* Sparse-reward classification MDP

Milestone

* Successfully train DQN agents on toy classification environments.

⸻

Phase 2 — Plasticity Probe Framework

Goal

Implement the core plasticity measurement framework from the paper.

Tasks

* Save periodic checkpoints during training
* Clone networks from checkpoints
* Generate random probe targets
* Train copied networks on probe tasks
* Measure adaptation speed and final loss

Metrics

* Probe loss
* Plasticity degradation
* Adaptation speed

Milestone

* Reproduce probe learning behavior from early vs late checkpoints.

⸻

Phase 3 — Section 4 Reproduction

Goal

Reproduce the mechanistic experiments from Section 4.

Section 4.1 — Optimizer Instability

Tasks

* Train MLP on randomized MNIST labels
* Periodically reshuffle labels
* Compare default Adam vs tuned Adam
* Measure dead neurons
* Measure accuracy collapse

Target Figure

* Figure 1

Milestone

* Reproduce optimizer instability and neuron death.

⸻

Section 4.2 — Brownian Motion vs Gradient Descent

Tasks

* Implement gradient-descent trajectory
* Implement Brownian-motion trajectory
* Match update norms between methods
* Compute Hessian spectra
* Compute gradient covariance matrices

Metrics

* Hessian sharpness
* Eigenvalue spectrum
* Gradient interference

Target Figure

* Figure 2

Milestone

* Reproduce differences in curvature and gradient structure.

⸻

Phase 4 — Section 5 Reproduction

Goal

Investigate explanations for plasticity loss.

Section 5.1 — Plasticity Measurement

Tasks

* Automate checkpoint evaluation
* Run multiple probe tasks
* Log probe statistics

Milestone

* Stable and repeatable plasticity evaluation pipeline.

⸻

Section 5.2 — Falsification Experiments

Tasks

Measure correlations between plasticity and:

* Weight norm
* Weight rank
* Feature rank
* Dead neurons

Target Figure

* Figure 3

Milestone

* Reproduce correlation reversal experiments.

⸻

Section 5.3 — Probe Learning Curves

Tasks

* Measure optimization trajectories on probe tasks
* Compare early vs late checkpoints
* Visualize optimization slowdown

Target Figure

* Figure 4

Milestone

* Reproduce optimization slowdown behavior.

⸻

Phase 5 — Interventions and Solutions

Goal

Implement and benchmark methods that preserve plasticity.

Interventions

* Layer normalization
* Weight decay
* Spectral normalization
* Reset last layer
* Shrink-and-perturb
* Two-hot categorical encoding

Architectures

* MLP
* CNN
* ResNet18
* Vision Transformer

Metrics

* Plasticity loss
* Probe adaptation speed
* Hessian sharpness
* Gradient covariance

Target Figures

* Figure 6
* Figure 8
* Figure 9
* Figure 10
* Figure 11
* Figure 14
* Figure 17
* Figure 18

Milestone

* Benchmark all interventions and compare effectiveness.

⸻

Phase 6 — Atari Experiments

Goal

Scale experiments to larger RL benchmarks.

Tasks

* Implement Double DQN
* Add Atari wrappers
* Integrate layer normalization into Atari agents
* Train on Arcade Learning Environment games
* Track gradient covariance evolution
* Track Hessian evolution

Target Figure

* Figure 7

Milestone

* Reproduce Atari performance improvements using LayerNorm.

⸻

Phase 7 — Advanced Analysis and Extensions

Goal

Extend beyond reproduction and investigate new research directions.

Potential Research Directions

Optimizer-Plasticity Dynamics

* AdamW
* SAM
* Lion
* Shampoo
* Sophia

Representation Analysis

* Feature collapse
* Singular value evolution
* NTK analysis
* Representation drift

Continual Reinforcement Learning

* Task switching
* Lifelong RL
* Continual adaptation

Transformer Plasticity

* ViT plasticity dynamics
* Attention-layer stability
* Scaling behavior

Milestone

* Develop novel experiments or thesis contributions.

⸻

Recommended Development Order

1. DQN + Replay Buffer
2. Classification MDPs
3. Probe-task framework
4. Logging + Visualization
5. Section 4 experiments
6. Hessian analysis
7. Gradient covariance analysis
8. Intervention framework
9. Atari experiments
10. Research extensions

⸻

Important Engineering Principles

Reproducibility

Always:

* Save seeds
* Save configs
* Save checkpoints
* Save optimizer states

Modular Design

Keep separate:

* Training
* Analysis
* Visualization
* Experiment management

Experiment Tracking

Recommended tools:

* Weights & Biases
* TensorBoard

Configuration Management

Recommended:

* Hydra
* YAML configs

⸻

Most Difficult Components

Hessian Estimation

* Lanczos approximation
* Eigenvalue density estimation

Gradient Covariance Analysis

* Memory intensive
* Expensive for large models

Probe Framework

* Easy to implement incorrectly
* Critical for valid results

Atari Reproduction

* High compute cost
* Sensitive hyperparameters

⸻

Immediate Next Step

Implement:

1. Replay Buffer
2. DQN
3. Classification MDP
4. Probe-task evaluation

These form the foundation of the entire project.