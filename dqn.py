import random
import numpy as np
from collections import deque
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# ACTION SPACE
# [steer, gas, brake]
# steer: -1.0 ~ 1.0 (left ~ right)
# gas: 0.0 ~ 1.0
# brake: 0.0 ~ 1.0
default_action_space = [
    (0.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), # full gas, left, right, brake and do nothing
    (-0.5, 0.0, 0.0), (0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), # half gas, left, right, brake
    (-0.5, 0.5, 0.0), (0.5, 0.5, 0.0), (-0.5, 0.0, 0.5), (0.5, 0.0, 0.5), # half gas and left, right and brake
    (-1.0, 0.0, 0.5), (1.0, 0.0, 0.5), # half brake and left, right
    (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), # half gas, left, right and brake
    ]

normal_space = [
        (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structure
        (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
        (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
        (-1, 0,   0), (0, 0,   0), (1, 0,   0)
    ]

# other action spaces can be defined here to emulate different driving styles

class CarRacingAgent:
    def __init__(
            self,
            action_space=normal_space,
            frame_stack_num=3,
            memory_size=5000,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.1,
            epsilon_decay=0.9999,
            learning_rate=0.001,
    ):
        self.action_space = action_space
        self.frame_stack_num = frame_stack_num
        self.memory_size = memory_size
        self.memory = self.build_memory()
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def build_memory(self):
        return deque(maxlen=self.memory_size)

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(filters=6, kernel_size=(4, 4), strides=(2,2), activation='relu', input_shape=(96, 96, self.frame_stack_num)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(filters=12, kernel_size=(4, 4), strides=(2,2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(300, activation='relu'))
        model.add(Dense(len(self.action_space), activation=None))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate, epsilon=1e-7))
        return model

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, self.action_space.index(action), reward, next_state, done))
    
    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.choice(self.action_space)
        act_values = self.model.predict(state)
        return self.action_space[np.argmax(act_values[0])]

    def replay(self, batch_size):
        rng = np.random.default_rng()
        
        """
        minibatch = random.sample(self.memory, batch_size)
        train_state = []
        train_target = []
        for state, action_index, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(state, axis=0))[0]
            if done:
                target[action_index] = reward
            else:
                t = self.target_model.predict(np.expand_dims(next_state, axis=0))[0]
                target[action_index] = reward + self.gamma * np.amax(t)
            train_state.append(state)
            train_target.append(target)
        self.model.fit(np.array(train_state), np.array(train_target), epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        """
        minibatch = rng.choice(np.array(self.memory), batch_size, replace=False)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        train_states = np.array([state for state in states])
        train_targets = self.model.predict(train_states, verbose=0)

        next_states = np.array([state for state in next_states])
        next_state_values = np.amax(self.target_model.predict(next_states, verbose=0), axis=1)

        dones = np.array(dones)  # Convert dones to NumPy array

        train_targets[np.arange(batch_size), actions] = rewards + self.gamma * next_state_values * (1 - dones)

        self.model.fit(train_states, train_targets, epochs=1, verbose=0)
        #print(f'epsilon: {self.epsilon}')

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        


    
    def load(self, name):
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        self.model.save_weights(name)

if __name__ == '__main__':
    agent = CarRacingAgent()
    print(agent.model.summary())