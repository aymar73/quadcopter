#using python3
import numpy as np
from kerasNet import KerasNet

net = KerasNet(state_size=1)

state = np.array([[1]])

value = net.model.predict_on_batch(state)
print('predict=',value) #[[1.0]]


# training the network with a target differed from the predicted value
target = np.array([[2]])
loss = net.model.evaluate(x=state, y=target)
print('loss1 =',loss) # 1.0
net.model.train_on_batch(x=state, y=target)

value = net.model.predict_on_batch(state)
print('predict=',value) #[[1.0020001]]


# training the network with a target equal to the predicted value
target = value
loss = net.model.evaluate(x=state, y=target)
print('loss2 =',loss) # 0.0
net.model.train_on_batch(x=state, y=target)

value = net.model.predict_on_batch(state)
print('predict=',value) #[[1.0033401]]

# Facts:
# the third prediction [[1.0033401]] is different from the second
# prediction [[1.0020001]].

# My thought:
# The facts are contradictory to the theory of learning which expects
# the network to learn from error/loss value. In this case there is
# no error (loss2 = 0), so there should be no learning, no changes in weights.
# Therefore the network does not change.

# Question
# Why is the last prediction not the same as the second one?



