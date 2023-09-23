# initial-d

This is a project for the course "Agents and Multi-Agent Systems", at FEUP. Our goal is to try to explore how to influence the driving style of a self-driving car without ruining its performance.

We focus on limiting the set of actions the car can perform and on changing the reward policy. For this, we use a DQN, therefore reducing the action space to a discrete set of actions, and increase rewards for actions that we want to encourage.

Specifically, we want to encourage the car to simultaneously accelerate and turn, creating a drift-like movement - hence the project's name, based on the popular manga and anime [Initial-D](https://en.wikipedia.org/wiki/Initial_D).

## Details

The project is built using the CarRacing environemnt from Gymnasium, and the DQN implementation with Tensorflow.

## Results

We were able to successfully influence the driving style of the car, but its performance took a hit - on the best cases, it only managed to complete half of the track without losing the correct path.

We believe this is due to the fact that the car isn't able to have fine-grained control over its actions due to the reduced action space, easily getting off-track and being unable to recover.

Regardless, we were able to get some interesting results, and we believe that with more time and effort, we could get better results.
