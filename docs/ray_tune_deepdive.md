# Ray Tune Deep Dive

Let's now dive deeper into Ray Tune and see how we can leverage it to optimize
our models.
First and foremost, we need to understand what a **trial** is.

A **trial** are basically experiments that are run by the `Tuner`.
For each choice of hyperparameters, the `Tuner` will run a trial.
In order to so we need to define a couple settings:

1. We need to define what the `trainable` is. This refers to the objective function
   that is passed to Ray Tune

2. The `trainable` needs to be aware of the **search space**, basically which
   hyperparameters we want to tune and the sampling space of each hyperparameter.

3. A trial is also characterized by _how to tune_ a given experiment. This is called
   the **search algorithm**, which dictates which trials to run next

4. We then need to finalize _when to stop_ via the **scheduler** .
   It decides if/when to stop a given trial, or prioritize certain trials over others

> [!note] How does the `Tuner` know which resources to allocate to trials?
> By default each trial will run in a separate process and consume 1 CPU core
