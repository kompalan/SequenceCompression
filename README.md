# Sequence Compressor


> Note: This is a smaller part of a larger project. See [my le-wm fork](https://github.com/kompalan/le-wm) for information on how this all fits together.

## Brief Explanation

The recent LeWorldModel paper introduced a new way to train JEPA-based dynamics models without the need for heuristics such as stop-gradients and EMA's. However, two of the claimed [limitations](https://arxiv.org/html/2603.19312v3#S6) of the LeWorldModel paper is that the planning horizon remains small, and that data must sufficiently cover the state space of the task. But what if these problems stemmed from the same lack of good interaction data? If the planning horizon remains small, then this means that each predicted latent is a subtly degraded version of the ground truth. And as degraded predictions are fed back in, further predictions stray further from the ground truth. This means that the model hasn't fully internalized how dynamics work. Even in relatively simple scenarios such as PushT, performance degrades sharply as goal latent is pushed further back.

In the case of sufficiently covering the state space of the environment, the model needs more than a sufficient set of data: it needs lots of redundancy, with the same underlying information represented a number of different ways for the model to internalize the environment's dynamics.

What if we could make informative data from existing interaction data? For example, say you have a set of interaction data over 1k episodes, each lasting M timesteps. What if you not only gave those interactions to the model, but also took a subset of the data, compressed it, and fed that as well. In essence, teach the model to recognize not only how each individual action transforms the observation space, but also how a set of actions chain into a transformation? This would not only take a dataset and enlarge it by a factor of `2^M` (you could build the superset of each M-timestep episode), but also give the model a language to plan hierarchically. It would be able to output latent actions that span an arbitrary number of timesteps, and better understand how actions affect observations due to the explosion of training data.

Here's a little video I made with Claude to better explain what I'm trying to test:
<p align="center"><video src="https://github.com/user-attachments/assets/9cef7c32-f904-422c-9067-ed9a576f1b52">
</video></c>

## Related Work

- [Accelerating Reinforcement Learning with Learned Skill Priors](https://arxiv.org/abs/2010.11944): Extremely close to what I'm trying here! The authors try to extract "skills" from prior interaction data and train a network to propose relevant skills based on observations. The authors show that doing this results in much better performance on robotic manipulation in the D4RL kitchen environment.

## Architecture

<img width="1920" height="1080" alt="tikz-export-3" src="https://github.com/user-attachments/assets/c4753f09-9555-4605-a56d-6e511994c569" />
