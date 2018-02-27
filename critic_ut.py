import numpy as np
from critic import Critic

# Critic (Value) Model
critic_local = Critic(state_size=1, action_size=1)

states = np.array([[0]])
actions = np.array([[12.5]])

Q_targets = critic_local.model.predict_on_batch([states, actions])
print('Q_targets=',Q_targets) #[[2.1525557]]

loss = critic_local.model.evaluate(x=[states, actions], y=Q_targets)
print('loss 1 =',loss) # 0.0

critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
Q_targets = critic_local.model.predict_on_batch([states, actions])
print('Q_targets=',Q_targets) #[[0.001]]

loss = critic_local.model.evaluate(x=[states, actions], y=Q_targets)
print('loss 2 =',loss) # 0.0

critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)
Q_targets = critic_local.model.predict_on_batch([states, actions])
print('Q_targets=',Q_targets) #[[0.00167005]]

loss = critic_local.model.evaluate(x=[states, actions], y=Q_targets)
print('loss 3 =',loss) # 0.0
