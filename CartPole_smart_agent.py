## This Agent was trained using hill climbing with Adaptive Noise Scaling which comes under POLICY BASED METHODS.
## And purpose of this script is to show a smart agent!
import pickle
import gym
import numpy as np

with open('hill_climbing_weight.pickle','rb') as f:
    weight = pickle.load(f)


class Policy():
    def __init__(self, s_size=4, a_size=2):
        """
        Here I'm intializing the self.w with trained weights
        The basic purpose of this function is to randomly initalize the weights for the network and then we would
        optimize these weights with noise scaling(adaptive).
        
        Shape: [state_dimension,action_dimension] and softmax activation function at output layer-- when action space is discrete
               [state_dimension, 1 node] and no activation function when action space is not discrete.
        """
        #self.w = 1e-4*np.random.rand(s_size, a_size) ##weights for simple linear policy: state_space x action_space
        self.w = weight
    
    def forward(self, state):
        """
        Here we multipy(vectorized) our state with weights and get corresponding output 
        """
        x = np.dot(state,self.w)
        ## below is the implementation of softmax function!!
        return np.exp(x)/sum(np.exp(x))
    
    def act(self,state):
        """
        This function decides whether we want our policy to be stochastic or determinstic policy.
        """
        probs = self.forward(state)
        #action = np.random.choice(2,p=probs)
        action = np.argmax(probs)
        # option 1: stochastic policy
        # option 2: stochastic policy
        return action
    
    
policy = Policy()

env = gym.make('CartPole-v0')

for i in range(3):
    state = env.reset()
    while True:
        env.render()
        action = policy.act(state)
        state,reward,done,_=env.step(action)
        if done:
            break