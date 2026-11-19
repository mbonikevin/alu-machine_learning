# Policy Gradients

REINFORCE (Monte-Carlo policy gradient) on `CartPole-v1`.

- `policy_gradient.py`: `policy(matrix, weight)` (softmax policy) and `policy_gradient(state, weight)` (returns action, gradient)
- `train.py`: `train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False)` returns the score of every episode; renders every 1000 episodes when `show_result` is True
