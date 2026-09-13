# Sequence Compressor


> Note: This is a smaller part of a larger project. See [my le-wm fork](https://github.com/kompalan/le-wm) for information on how this all fits together.

## Brief Explanation

The recent [*LeWorldModel*](https://arxiv.org/abs/2603.19312) paper trains JEPA-based dynamics models without 
heuristics like stop-gradients or EMA targets, but it flags two important 
[limitations](https://arxiv.org/html/2603.19312v3#S6): **planning horizons stay 
short**, and training data must **densely cover the task's state space**. This repository aims to investigate whether both may 
share one root cause: a lack of "good" interaction data.

### **Short horizons.** 
Each predicted latent is a slightly degraded copy of the 
ground truth. Feed that back in as input, and errors compound. As later 
predictions drift further from reality, the model never fully internalizes 
the dynamics of its environment. Even in simple settings like PushT, performance collapses as the 
goal is pushed further into the future.

### **Poor coverage (or, what makes interaction data "good"?)** 
The fix isn't just *more* data, but *redundant* data — the 
same information expressed in multiple ways, so the model generalizes the 
dynamics instead of memorizing trajectories.

**The idea:** 
Synthesize new training examples from existing interaction data. 
Given 1k episodes of length $M$, don't just train on individual transitions — 
also compress subsequences of each episode and train on those as *compound* 
transitions. This teaches the model not only how one action changes an 
observation, but how a *chain* of actions composes into a single transformation.

This gives two payoffs:
- **More data:** each $M$-step episode expands into its $M^2$ possible subsequences.
- **Hierarchical planning:** the model learns latent "actions" spanning arbitrary 
  numbers of timesteps — a vocabulary for long-horizon planning.

Here's a little video I made with Claude to better explain what I'm trying to test:
<p align="center"><video src="https://github.com/user-attachments/assets/7b6d8ab6-ae74-472e-9cbd-94f7e21e5192""></video></p>

## Related Work

- [Accelerating Reinforcement Learning with Learned Skill Priors](https://arxiv.org/abs/2010.11944): Extremely close to what I'm trying here! The authors try to extract "skills" from prior interaction data and train a network to propose relevant skills based on observations. The authors show that doing this results in much better performance on robotic manipulation in the D4RL kitchen environment.

## Architecture

<img width="666" height="269" alt="tikz-export" src="https://github.com/user-attachments/assets/c6199a68-1eb9-439a-a026-06c888f06b38" />

<img width="666" height="269" alt="Screenshot 2026-09-07 at 2 13 42 PM" src="https://github.com/user-attachments/assets/cfd71aec-6607-4670-a19e-b1182891f1a3" />


*Note: not my figure. Credit to Lucas Maes et. al
## Training

The training script is located in `src/sequence_compression/` under `train.py`. To kick off training, simply run:
```
uv run train
```

## Results

<img width="1920" height="1080" alt="sequence_compression_loss_chart" src="https://github.com/user-attachments/assets/7515a0aa-00c3-4910-90eb-ea6459e22599" />
