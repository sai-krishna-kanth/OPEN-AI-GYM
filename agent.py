import numpy as np
import tensorflow as tf
import tensorflow.keras as keras

class Memory:

    def __init__(self, shape):

        self._max_size = shape[0]
        self._memory = np.zeros(shape)
        self._memory_counter = 0

    def add(self, memory):

        index = self._memory_counter % self._max_size
        self._memory[index,:] = memory
        self._memory_counter += 1

    def sample(self, amount):
        
        if self._memory_counter > self._max_size:
            sample_index = np.random.choice(self._max_size, size=amount)
        else:
            sample_index = np.random.choice(self._memory_counter, size=amount)
        return self._memory[sample_index, :]

    @property
    def length(self):
        return len(self._memory)

class DDQN:

    def __init__(self, n_features=None, n_actions=None, state_dim=None,
                 batch_size=32, memory_size=5000, lr=0.01, lr_decay=0.01,
                 gamma=1, eps_max=1, eps_min=0.01, eps_decay=0.0005, tau=0.08,
                 load_path=''):
        
        if state_dim is not None: # atari
            x,y,c = self.state_dim = state_dim
            self.n_features = x*y*c
        else:
            self.n_features = n_features
            self.state_dim = None
        self.n_actions = n_actions

        self.memory = Memory([memory_size, self.n_features*2+3]) #s, a, r, s_, d
        self.batch_size = batch_size

        if load_path:
            self.eval_model = keras.models.load_model(load_path)
            self.target_model = self._build_network('target')
            self.target_model.set_weights(self.eval_model.get_weights())
        else:
            self.eval_model = self._build_network('eval')
            self.eval_model.compile(loss="mse",
                                    optimizer=keras.optimizers.Adam(lr=lr,
                                                                    decay=lr_decay))
            self.target_model = self._build_network('target')

        self.gamma = gamma

        self.eps = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.steps = 0

        self.tau = tau
    
    def _build_network(self, name):
        
        if self.state_dim is not None: # cnn
            inputs = keras.Input(shape=(self.state_dim), name='state')
            conv = keras.layers.Conv2D(32, 5, activation=tf.nn.relu,
                                       name='conv_1')(inputs)
            conv = keras.layers.MaxPooling2D(name='max_pool_1')(conv)
            conv = keras.layers.Conv2D(32, 5, activation=tf.nn.relu,
                                       name='conv_2')(conv)
            conv = keras.layers.MaxPooling2D(name='max_pool_2')(conv)
            conv = keras.layers.Conv2D(64, 4, activation=tf.nn.relu,
                                       name='conv_3')(conv)
            conv = keras.layers.MaxPooling2D(name='max_pool_3')(conv)
            conv = keras.layers.Conv2D(64, 3, activation=tf.nn.relu,
                                       name='conv_4')(conv)
            x = keras.layers.Flatten(name='flat')(conv)
            x = keras.layers.Dense(512, activation=tf.nn.relu, name='dense')(x)
        else: # dff
            inputs = keras.Input(shape=(self.n_features,), name='state')
            x = keras.layers.Dense(32, activation=tf.nn.relu,
                                   name='dense_1')(inputs)
            x = keras.layers.Dense(32, activation=tf.nn.relu,
                                   name='dense_2')(x)

        outputs = keras.layers.Dense(self.n_actions, activation='linear',
                                     name='action_values')(x)

        return keras.Model(inputs=inputs, outputs=outputs, name=name)
        
    def _update_epsilon(self):

        self.steps += 1
        self.eps = (self.eps_min + (self.eps_max - self.eps_min) *
                    np.exp(-self.eps_decay * self.steps))
        
    def _update_target_weights(self):
        for t, e in zip(self.target_model.trainable_variables,
                        self.eval_model.trainable_variables):
            t.assign(t * (1 - self.tau) + e * self.tau)
        
    def act(self, state):
    
        if np.random.rand() <= self.eps:
            return np.random.randint(self.n_actions)
        if self.state_dim is not None:
            state = state.reshape(np.hstack(([1],self.state_dim)))
        else:
            state = state.reshape(1,-1)
        action_values = self.eval_model(state.astype('float32'))
        return np.argmax(action_values)
  
    def train(self, state, action, reward, state_next, done):
    
        # memorize transition
        if self.state_dim is not None:
            self.memory.add(np.hstack((state.flatten(), [action, reward],
                                       state_next.flatten(), [done])))
        else:
            self.memory.add(np.hstack((state, [action, reward], state_next,
                                       [done])))

        # check if enough memory for fitting
        if self.memory.length < self.batch_size * 2:
            return 0

        # split batch
        batch = self.memory.sample(self.batch_size)
        states = batch[:, :self.n_features]
        actions = batch[:, self.n_features].astype(int)
        rewards = batch[:, self.n_features+1]
        states_ = batch[:, self.n_features+2:-1] # next states
        done = batch[:, -1]

        # reshape for atari if applicable
        if self.state_dim is not None:
            states = states.reshape(np.hstack(([self.batch_size],
                                              self.state_dim)))
            states_ = states_.reshape(np.hstack(([self.batch_size],
                                              self.state_dim)))

        # predict Q(s,a) from eval net
        eval_q_sa = self.eval_model(states)

        # predict Q(s',a') from eval net
        eval_q_s_a_ = self.eval_model(states_)

        # build target
        targets = eval_q_sa.numpy()
        updates = rewards

        # only apply td update to idxs in which episode has not terminated
        valid_idxs = (1-done).astype(bool)

        batch_idxs = np.arange(self.batch_size)
        q_max_act = np.argmax(eval_q_s_a_.numpy(), axis=1)
        target_q = self.target_model(states_)
        updates[valid_idxs] += self.gamma * target_q.numpy()[batch_idxs[valid_idxs],
                                                             q_max_act[valid_idxs]]
        targets[batch_idxs, actions] = updates

        # fit model
        loss = self.eval_model.train_on_batch(states,targets)

        # update target weights
        self._update_target_weights()

        # update epsilon
        self._update_epsilon()

        return loss

    def save_model(self, env_name):
        
        self.eval_model.save("models/"+env_name+"_model.h5", overwrite=True)